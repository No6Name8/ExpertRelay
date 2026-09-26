"""Selective tensor fetcher for HuggingFace safetensors checkpoints.

Downloads only the specific named tensors we need from a (possibly huge)
sharded safetensors checkpoint on the HF Hub, using HTTP Range requests
against each shard's byte offsets -- never the full shard file. Large
tensors can be fetched a block of rows at a time (`fetch_rows`), so even
the 311M-parameter embedding never has to be resident in RAM at once.

Safetensors file layout (per shard):
    [8 bytes]  little-endian u64 header length N
    [N bytes]  UTF-8 JSON header: {tensor_name: {dtype, shape, data_offsets:[s,e]}, "__metadata__": {...}}
    [rest]     raw tensor bytes, back-to-back, row-major, offsets relative to end of header

Pin the revision: pass a commit hash (see `resolve_revision`), not "main".
"main" can move between two requests of the same conversion, silently
mixing tensors from two different checkpoints.
"""

from __future__ import annotations

import json
import math
import os
import struct
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

HF_BASE = "https://huggingface.co"

# Bytes per element, and how to decode, for each safetensors dtype we accept.
# bfloat16 has no native numpy type; we read it as raw uint16 and upcast by
# hand (bf16 is literally the top 16 bits of fp32).
_ITEMSIZE = {"BF16": 2, "F16": 2, "F32": 4, "I32": 4, "I64": 8}
_NUMPY_DTYPES = {"F32": np.float32, "F16": np.float16, "I64": np.int64, "I32": np.int32}


def _auth_headers() -> dict[str, str]:
    """Send HF_TOKEN if the environment has one. Anonymous requests work but
    are rate-limited by the Hub. Never logged or stored anywhere."""
    token = os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _http_get(url: str, timeout: int = 60) -> bytes:
    with urllib.request.urlopen(
        urllib.request.Request(url, headers=_auth_headers()), timeout=timeout
    ) as resp:
        return resp.read()


def _http_get_range(url: str, start: int, end_inclusive: int, timeout: int = 120, retries: int = 8) -> bytes:
    """GET bytes [start, end_inclusive], retrying with exponential backoff.

    The response length is checked. A server that ignores the Range header
    and returns 200 with the whole file, or a connection that drops mid-body,
    must not be silently treated as the requested bytes.
    """
    expected = end_inclusive - start + 1
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end_inclusive}", **_auth_headers()})
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            if len(data) != expected:
                raise OSError(f"got {len(data)} bytes, expected {expected}")
            return data
        except Exception as e:  # network hiccup, rate limit, short body -- back off and retry
            last_err = e
            time.sleep(min(60.0, 2.0**attempt))
    raise RuntimeError(f"range GET failed after {retries} tries: {url} [{start}-{end_inclusive}]: {last_err}")


def _resolve_url(repo_id: str, filename: str, revision: str = "main") -> str:
    return f"{HF_BASE}/{repo_id}/resolve/{revision}/{filename}"


def resolve_revision(repo_id: str, revision: str = "main") -> str:
    """The commit hash a branch/tag/hash currently points to on the Hub."""
    url = f"{HF_BASE}/api/models/{repo_id}/revision/{urllib.parse.quote(revision, safe='')}"
    return json.loads(_http_get(url))["sha"]


def fetch_file(repo_id: str, filename: str, revision: str) -> bytes:
    """One small repo file (config, index, tokenizer) at a revision."""
    return _http_get(_resolve_url(repo_id, filename, revision))


def fetch_config(repo_id: str, revision: str = "main") -> dict:
    """The model's config.json. Architecture dimensions always come from here,
    never from constants in this repo."""
    return json.loads(fetch_file(repo_id, "config.json", revision))


def fetch_index(repo_id: str, revision: str = "main") -> dict:
    """Download the (small) *.safetensors.index.json weight map."""
    return json.loads(fetch_file(repo_id, "model.safetensors.index.json", revision))


@dataclass
class ShardHeader:
    header_len: int
    entries: dict[str, dict]  # name -> {"dtype", "shape", "data_offsets": [s, e]}


_shard_header_cache: dict[str, ShardHeader] = {}


def _get_shard_header(shard_url: str) -> ShardHeader:
    """Fetch + parse one shard's safetensors header (cached: a shard with
    many wanted tensors should only pay this cost once)."""
    if shard_url in _shard_header_cache:
        return _shard_header_cache[shard_url]
    length_bytes = _http_get_range(shard_url, 0, 7)
    header_len = struct.unpack("<Q", length_bytes)[0]
    header_bytes = _http_get_range(shard_url, 8, 8 + header_len - 1)
    header = json.loads(header_bytes)
    header.pop("__metadata__", None)
    sh = ShardHeader(header_len=header_len, entries=header)
    _shard_header_cache[shard_url] = sh
    return sh


@dataclass(frozen=True)
class TensorLocation:
    """Where one tensor's bytes live: shard URL plus absolute byte range."""

    name: str
    url: str
    dtype: str
    shape: tuple[int, ...]
    data_start: int  # absolute offset of the tensor's first byte in the shard file
    nbytes: int

    @property
    def rows(self) -> int:
        return self.shape[0] if self.shape else 1

    @property
    def row_bytes(self) -> int:
        """Bytes per leading-dimension row (the whole tensor for a scalar)."""
        return _ITEMSIZE[self.dtype] * math.prod(self.shape[1:])


def locate_tensors(
    repo_id: str, tensor_names: list[str], revision: str, index: dict
) -> dict[str, TensorLocation]:
    """Resolve each tensor name to its shard URL and byte range (fetches each
    needed shard header once)."""
    weight_map = index["weight_map"]
    missing = [n for n in tensor_names if n not in weight_map]
    if missing:
        raise KeyError(
            f"tensor(s) not found in checkpoint index: {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    out: dict[str, TensorLocation] = {}
    for name in tensor_names:
        url = _resolve_url(repo_id, weight_map[name], revision)
        header = _get_shard_header(url)
        entry = header.entries[name]
        if entry["dtype"] not in _ITEMSIZE:
            raise ValueError(f"unsupported dtype {entry['dtype']} for tensor {name}")
        s, e = entry["data_offsets"]
        out[name] = TensorLocation(
            name=name,
            url=url,
            dtype=entry["dtype"],
            shape=tuple(entry["shape"]),
            data_start=8 + header.header_len + s,
            nbytes=e - s,
        )
    return out


def row_byte_range(loc: TensorLocation, row_start: int, row_end: int) -> tuple[int, int]:
    """Absolute [first, last] byte (inclusive, as HTTP Range wants) of rows
    [row_start, row_end). Row-major storage makes any run of rows contiguous."""
    if not 0 <= row_start < row_end <= loc.rows:
        raise ValueError(
            f"row range [{row_start}, {row_end}) out of bounds for {loc.name} with {loc.rows} rows"
        )
    first = loc.data_start + row_start * loc.row_bytes
    return first, loc.data_start + row_end * loc.row_bytes - 1


def _bf16_bytes_to_fp32(raw: bytes, shape) -> np.ndarray:
    """bf16 is literally the top 16 bits of fp32 -- upcast is a left-shift, no rounding needed."""
    u16 = np.frombuffer(raw, dtype="<u2")
    u32 = u16.astype(np.uint32) << 16
    f32 = u32.view(np.float32)
    return f32.reshape(shape).copy()


def decode(raw: bytes, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    """Raw safetensors bytes -> ndarray. BF16 is upcast to fp32 (exactly)."""
    if dtype == "BF16":
        return _bf16_bytes_to_fp32(raw, shape)
    return np.frombuffer(raw, dtype=_NUMPY_DTYPES[dtype]).reshape(shape).copy()


def fetch_rows(loc: TensorLocation, row_start: int = 0, row_end: int | None = None) -> np.ndarray:
    """Fetch rows [row_start, row_end) of one tensor (default: all of it)."""
    row_end = loc.rows if row_end is None else row_end
    first, last = row_byte_range(loc, row_start, row_end)
    shape = (row_end - row_start, *loc.shape[1:]) if loc.shape else ()
    return decode(_http_get_range(loc.url, first, last), loc.dtype, shape)


def fetch_tensors(
    repo_id: str,
    tensor_names: list[str],
    revision: str = "main",
    index: dict | None = None,
    progress_cb: Callable[[int, int, str, int], None] | None = None,
) -> dict[str, np.ndarray]:
    """Fetch exactly `tensor_names`, each whole, as {name: ndarray}. Holds
    them all in RAM -- for small slices only; the full-model store build
    streams via locate_tensors + fetch_rows instead."""
    if index is None:
        index = fetch_index(repo_id, revision)
    locations = locate_tensors(repo_id, tensor_names, revision, index)
    out: dict[str, np.ndarray] = {}
    for done, name in enumerate(tensor_names, start=1):
        out[name] = fetch_rows(locations[name])
        if progress_cb:
            progress_cb(done, len(tensor_names), name, out[name].nbytes)
    return out
