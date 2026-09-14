"""Selective tensor fetcher for HuggingFace safetensors checkpoints.

Downloads only the specific named tensors we need from a (possibly huge)
sharded safetensors checkpoint on the HF Hub, using HTTP Range requests
against each shard's byte offsets -- never the full shard file.

This is what makes converting a "reduced" slice of a 28GB checkpoint
(a few layers, a handful of experts) practical on an 8GB machine: we never
pull more than the bytes of the tensors we actually asked for.

Safetensors file layout (per shard):
    [8 bytes]  little-endian u64 header length N
    [N bytes]  UTF-8 JSON header: {tensor_name: {dtype, shape, data_offsets:[s,e]}, "__metadata__": {...}}
    [rest]     raw tensor bytes, back-to-back, offsets relative to end of header
"""
from __future__ import annotations

import json
import struct
import time
import urllib.request
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

HF_BASE = "https://huggingface.co"

# HF dtype string -> numpy dtype for the ones we expect in this checkpoint.
# bfloat16 has no native numpy type; we read it as raw uint16 and upcast by
# hand (bf16 -> fp32 is just: left-shift the 16 bits into the top half of a
# uint32, then view as float32 -- bf16 IS the top 16 bits of fp32).
_SAFE_DTYPES = {
    "F32": np.float32,
    "F16": np.float16,
    "I64": np.int64,
    "I32": np.int32,
}


def _http_get_range(url: str, start: int, end_inclusive: int, timeout: int = 60, retries: int = 4) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end_inclusive}"})
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as e:  # network hiccup / rate limit -- back off and retry
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"range GET failed after {retries} tries: {url} [{start}-{end_inclusive}]: {last_err}")


def _resolve_url(repo_id: str, filename: str, revision: str = "main") -> str:
    return f"{HF_BASE}/{repo_id}/resolve/{revision}/{filename}"


def fetch_index(repo_id: str, revision: str = "main") -> dict:
    """Download the (small) *.safetensors.index.json weight map."""
    url = _resolve_url(repo_id, "model.safetensors.index.json", revision)
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


@dataclass
class ShardHeader:
    header_len: int
    entries: Dict[str, dict]  # name -> {"dtype", "shape", "data_offsets": [s, e]}


_shard_header_cache: Dict[str, ShardHeader] = {}


def _get_shard_header(shard_url: str) -> ShardHeader:
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


def _bf16_bytes_to_fp32(raw: bytes, shape) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype="<u2")
    u32 = u16.astype(np.uint32) << 16
    f32 = u32.view(np.float32)
    return f32.reshape(shape).copy()


def fetch_tensors(
    repo_id: str,
    tensor_names: List[str],
    revision: str = "main",
    index: dict | None = None,
    progress_cb=None,
) -> Dict[str, np.ndarray]:
    """Fetch exactly `tensor_names` from a sharded HF safetensors checkpoint.

    Returns {name: np.float32 ndarray}. Groups requests by shard file and
    reuses each shard's parsed header, so a shard with 40 wanted tensors
    only pays the header-parse cost once.
    """
    if index is None:
        index = fetch_index(repo_id, revision)
    weight_map = index["weight_map"]

    missing = [n for n in tensor_names if n not in weight_map]
    if missing:
        raise KeyError(f"tensor(s) not found in checkpoint index: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    by_shard: Dict[str, List[str]] = {}
    for name in tensor_names:
        by_shard.setdefault(weight_map[name], []).append(name)

    out: Dict[str, np.ndarray] = {}
    done = 0
    total = len(tensor_names)
    for shard_file, names in by_shard.items():
        shard_url = _resolve_url(repo_id, shard_file, revision)
        header = _get_shard_header(shard_url)
        data_start = 8 + header.header_len
        for name in names:
            entry = header.entries[name]
            dtype_str = entry["dtype"]
            shape = entry["shape"]
            s, e = entry["data_offsets"]
            raw = _http_get_range(shard_url, data_start + s, data_start + e - 1)
            if dtype_str == "BF16":
                arr = _bf16_bytes_to_fp32(raw, shape)
            elif dtype_str in _SAFE_DTYPES:
                arr = np.frombuffer(raw, dtype=_SAFE_DTYPES[dtype_str]).reshape(shape).copy()
            else:
                raise ValueError(f"unsupported dtype {dtype_str} for tensor {name}")
            out[name] = arr
            done += 1
            if progress_cb:
                progress_cb(done, total, name, arr.nbytes)
    return out
