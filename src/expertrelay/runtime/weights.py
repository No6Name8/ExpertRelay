"""Where the model's weights come from at runtime.

Resident weights (attention, shared expert, router, norms, lm_head) are
loaded once. Routed experts come from an ExpertSource, which is exactly
what the three Phase 2 baselines swap out; the forward code is the same
for all of them:

  UnbufferedExpertSource  read the one expert the router picked, with one
                          unbuffered read (store.expert_reader), use it,
                          drop it. "Ours": no cache, no OS page cache either.
  MmapExpertSource        memory-map experts.bin and let the OS page experts
                          in (and out) as the forward pass touches them.
  RamExpertSource         read all of experts.bin into RAM first: the
                          "normal load" baseline. 12.5 GB on an 8 GB machine,
                          so it is expected to fail; the benchmark records how.

The embedding table is the one resident tensor NOT read into RAM by
default: each token needs a single 2 KB row of a 311 MB table, so it's
memory-mapped and only touched rows get paged in. On this machine
(~1-2 GB actually free with the dev tools open) that 311 MB matters. It's
a deliberate deviation from "resident = fully in RAM", recorded in
docs/limitations.md.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from expertrelay.store.expert_reader import EXPERTS_BIN, ExpertStoreReader
from expertrelay.store.layout import (
    KIND_INT8,
    ExpertRecordLayout,
    parse_expert_record,
    read_expert_index,
    read_resident_index,
)

RESIDENT_BIN = "resident.bin"
RESIDENT_INDEX = "resident_index.json"
EMBEDDING = "model.embed_tokens.weight"


@dataclass(frozen=True)
class Int8Matrix:
    """Plain data: int8 weights and their per-row scales. Knows nothing about
    compute. Any backend can consume it (runtime.backends), which is what
    keeps the store and expert sources backend-independent."""

    q: np.ndarray  # int8 [out, in]
    scales: np.ndarray  # float32 [out]

    def dequantized_rows(self, rows: np.ndarray) -> np.ndarray:
        """Just the requested rows, as f32. The embedding lookup."""
        return self.q[rows].astype(np.float32) * self.scales[rows][:, None]

    @property
    def nbytes(self) -> int:
        return self.q.nbytes + self.scales.nbytes


class ResidentWeights:
    """Everything used on every token, by checkpoint tensor name."""

    def __init__(self, tensors: dict[str, Int8Matrix | np.ndarray], ram_bytes: int):
        self._tensors = tensors
        self.ram_bytes = ram_bytes  # bytes actually read into process memory (mmapped views excluded)

    def __getitem__(self, name: str) -> Int8Matrix | np.ndarray:
        return self._tensors[name]

    def __contains__(self, name: str) -> bool:
        return name in self._tensors

    @classmethod
    def load(cls, store_dir: Path, *, mmap_everything: bool = False) -> ResidentWeights:
        """Read resident.bin per its index. `mmap_everything` is the OS-paging
        baseline; otherwise everything except the embedding is read into RAM."""
        path = Path(store_dir) / RESIDENT_BIN
        entries = read_resident_index(Path(store_dir) / RESIDENT_INDEX)
        mm = np.memmap(path, dtype=np.uint8, mode="r")
        tensors: dict[str, Int8Matrix | np.ndarray] = {}
        ram_bytes = 0
        with open(path, "rb") as f:
            for e in entries:
                in_ram = not mmap_everything and e.name != EMBEDDING

                def take(
                    dtype: str, count: int, offset: int, shape: tuple[int, ...], in_ram=in_ram
                ) -> np.ndarray:
                    if in_ram:
                        f.seek(offset)
                        return np.fromfile(f, dtype=dtype, count=count).reshape(shape)
                    return np.ndarray(shape, dtype=dtype, buffer=mm, offset=offset)

                shape = tuple(e.shape)
                if e.kind == KIND_INT8:
                    rows = shape[0]
                    t = Int8Matrix(
                        q=take("int8", e.weight_bytes, e.offset, shape),
                        scales=take("<f4", rows, e.scales_offset, (rows,)),
                    )
                    ram_bytes += t.nbytes if in_ram else 0
                else:
                    t = take("<f4", e.weight_bytes // 4, e.offset, shape)
                    ram_bytes += t.nbytes if in_ram else 0
                tensors[e.name] = t
        return cls(tensors, ram_bytes)


ExpertWeights = dict[str, Int8Matrix]  # "gate_proj" / "up_proj" / "down_proj"


@dataclass
class SourceStats:
    loads: int = 0
    bytes_read: int = 0
    read_seconds: float = 0.0


class ExpertSource(ABC):
    """Hands the forward pass one routed expert's weights."""

    layout: ExpertRecordLayout
    stats: SourceStats

    @abstractmethod
    def load(self, layer: int, expert: int) -> ExpertWeights:
        """The expert's weights, as views. Valid only until the next load():
        sources may reuse one buffer for every load, so nothing is copied.
        The forward pass relies on this. It uses each expert and drops it
        before loading the next, which is what "no cache" means."""

    def close(self) -> None:  # noqa: B027 - optional hook, most sources hold nothing
        pass

    def __enter__(self) -> ExpertSource:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _to_matrices(parsed: dict[str, tuple[np.ndarray, np.ndarray]]) -> ExpertWeights:
    return {m: Int8Matrix(q, s) for m, (q, s) in parsed.items()}


class UnbufferedExpertSource(ExpertSource):
    """One unbuffered read per expert use; nothing kept afterwards."""

    def __init__(self, store_dir: Path):
        self._reader = ExpertStoreReader(store_dir).__enter__()
        self.layout = self._reader.layout
        self.stats = SourceStats()

    def load(self, layer: int, expert: int) -> ExpertWeights:
        t = time.perf_counter()
        raw = self._reader.read_raw(layer, expert)
        self.stats.read_seconds += time.perf_counter() - t
        self.stats.loads += 1
        self.stats.bytes_read += len(raw)
        return _to_matrices(parse_expert_record(self.layout, raw))

    def close(self) -> None:
        self._reader.__exit__(None, None, None)


class _IndexedFileSource(ExpertSource):
    """Shared by the mmap and in-RAM baselines: records addressed by offset
    into one buffer that holds all of experts.bin."""

    def __init__(self, store_dir: Path, buffer: np.ndarray):
        self.layout, entries = read_expert_index(Path(store_dir) / "experts_index.json")
        self._offsets = {(e.layer, e.expert): e.offset for e in entries}
        self._buffer = buffer
        self.stats = SourceStats()

    def load(self, layer: int, expert: int) -> ExpertWeights:
        off = self._offsets[(layer, expert)]
        record = self._buffer[off : off + self.layout.record_size]
        self.stats.loads += 1
        self.stats.bytes_read += self.layout.record_size
        # With mmap the disk read happens lazily, as page faults inside the
        # matmul, so it can't be timed separately here.
        return _to_matrices(parse_expert_record(self.layout, record))


class MmapExpertSource(_IndexedFileSource):
    def __init__(self, store_dir: Path):
        super().__init__(store_dir, np.memmap(Path(store_dir) / EXPERTS_BIN, dtype=np.uint8, mode="r"))


class RamExpertSource(_IndexedFileSource):
    def __init__(self, store_dir: Path):
        super().__init__(store_dir, np.fromfile(Path(store_dir) / EXPERTS_BIN, dtype=np.uint8))
