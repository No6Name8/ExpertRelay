"""Read tensors from a local, sharded safetensors checkpoint, without torch.

For building stores from the ORIGINAL bf16 weights already on disk
(store.download_checkpoint), e.g. the int4 expert store. The format is the
one store.fetch_hf_tensors reads over HTTP: 8-byte little-endian header
length, a JSON header (dtype, shape, byte offsets per tensor), then the raw
data; bf16 is upcast to float32 exactly by the same decode().

Shards are memory-mapped read-only, at most MAX_OPEN_SHARDS at a time
(least recently used closed first), so reading a whole ~29 GB checkpoint
tensor by tensor never maps more than two ~4 GB files at once.
"""

from __future__ import annotations

import json
import struct
from collections import OrderedDict
from pathlib import Path

import numpy as np

from expertrelay.store.fetch_hf_tensors import decode

INDEX_FILE = "model.safetensors.index.json"


class LocalCheckpoint:
    MAX_OPEN_SHARDS = 2

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.weight_map: dict[str, str] = json.loads((self.directory / INDEX_FILE).read_text())["weight_map"]
        self._open: OrderedDict[str, tuple[np.memmap, int, dict]] = OrderedDict()

    def _shard(self, shard: str) -> tuple[np.memmap, int, dict]:
        if shard in self._open:
            self._open.move_to_end(shard)
            return self._open[shard]
        while len(self._open) >= self.MAX_OPEN_SHARDS:
            self._open.popitem(last=False)
        path = self.directory / shard
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        header.pop("__metadata__", None)
        entry = (np.memmap(path, dtype=np.uint8, mode="r"), 8 + header_len, header)
        self._open[shard] = entry
        return entry

    def get(self, name: str) -> np.ndarray:
        """The tensor as a fresh array: bf16 upcast to float32 exactly, other dtypes
        (GPTQ's int32 / float16) as stored."""
        mm, data_start, header = self._shard(self.weight_map[name])
        e = header[name]
        start, end = e["data_offsets"]
        return decode(bytes(mm[data_start + start : data_start + end]), e["dtype"], tuple(e["shape"]))
