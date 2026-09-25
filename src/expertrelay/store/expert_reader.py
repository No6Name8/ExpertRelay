"""Read experts from a built store: one unbuffered read per expert.

This is the load path the Manager's cache will use when an expert has to
come off the SSD. Each record already contains the int8 weights and their
scales, padded and 4096-aligned (see layout.py). So loading an expert is
exactly one `unbuffered_io.read_at` call of `record_size` bytes, which goes
straight from the SSD to a page-aligned buffer with no OS file cache.

Windows only for now, like unbuffered_io.
"""

from __future__ import annotations

import hashlib
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from expertrelay.store import unbuffered_io
from expertrelay.store.layout import (
    ExpertIndexEntry,
    ExpertRecordLayout,
    parse_expert_record,
    read_expert_index,
)
from expertrelay.store.quantize import dequantize_rowwise_int8

EXPERTS_BIN = "experts.bin"
EXPERTS_INDEX = "experts_index.json"


@dataclass
class ExpertWeights:
    layer: int
    expert: int
    q: dict[str, np.ndarray]  # int8, [rows, cols]
    scales: dict[str, np.ndarray]  # float32, [rows]

    def dequantize(self, matrix: str) -> np.ndarray:
        return dequantize_rowwise_int8(self.q[matrix], self.scales[matrix])


class ExpertStoreReader:
    """Use as a context manager: holds one open unbuffered handle and one
    record-sized aligned buffer, reused for every read."""

    def __init__(self, store_dir: Path):
        self.store_dir = Path(store_dir)
        self.layout: ExpertRecordLayout
        self.layout, entries = read_expert_index(self.store_dir / EXPERTS_INDEX)
        self._entries = {(e.layer, e.expert): e for e in entries}
        self._stack: ExitStack | None = None
        self._handle: int | None = None
        self._buf = None
        self._addr = 0

    def __enter__(self) -> ExpertStoreReader:
        self._stack = ExitStack()
        self._handle = self._stack.enter_context(
            unbuffered_io.unbuffered_handle(self.store_dir / EXPERTS_BIN, write=False)
        )
        self._buf, self._addr = self._stack.enter_context(
            unbuffered_io.aligned_buffer(self.layout.record_size)
        )
        return self

    def __exit__(self, *exc) -> None:
        self._stack.close()
        self._stack = self._handle = self._buf = None

    def entry(self, layer: int, expert: int) -> ExpertIndexEntry:
        return self._entries[(layer, expert)]

    def entries(self) -> list[ExpertIndexEntry]:
        return list(self._entries.values())

    def read_raw(self, layer: int, expert: int) -> memoryview:
        """The whole record, via exactly one unbuffered read. The view is only
        valid until the next read (the buffer is reused)."""
        if self._handle is None:
            raise RuntimeError("ExpertStoreReader must be used as a context manager")
        e = self.entry(layer, expert)
        unbuffered_io.read_at(self._handle, self._addr, e.offset, e.size)
        return memoryview(self._buf)[: e.size]

    def read(self, layer: int, expert: int) -> ExpertWeights:
        raw = self.read_raw(layer, expert)
        parsed = parse_expert_record(self.layout, raw)
        return ExpertWeights(
            layer=layer,
            expert=expert,
            q={m: q.copy() for m, (q, _s) in parsed.items()},
            scales={m: s.copy() for m, (_q, s) in parsed.items()},
        )

    def verify(self, layer: int, expert: int) -> bool:
        """Does the record on disk hash to what the index recorded at build time?"""
        return hashlib.sha256(self.read_raw(layer, expert)).hexdigest() == self.entry(layer, expert).sha256
