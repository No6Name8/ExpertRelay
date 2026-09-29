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
  CachedExpertSource      (cache.expert_cache) an LRU cache of experts in RAM
                          under a fixed budget, with background prefetching.

The embedding table is the one resident tensor NOT read into RAM by
default: each token needs a single 2 KB row of a 311 MB table, so it's
memory-mapped and only touched rows get paged in. On this machine
(~1-2 GB actually free with the dev tools open) that 311 MB matters. It's
a deliberate deviation from "resident = fully in RAM", recorded in
docs/limitations.md.
"""

from __future__ import annotations

import threading
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


class ReadLog:
    """Every expert read, for finding out why reads take as long as they do.

    Per read: when it started (seconds since the log was created), how long
    it took, how long the drive had been idle before it (no read of ours in
    flight; 0 when another read was already running), how many reads were
    already in flight, who issued it ("demand": the forward pass waited for
    it; "prefetch": a background thread) and the (layer, expert). "Idle"
    only knows this process's reads. Thread-safe: I/O threads log too."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._t0 = time.perf_counter()
        self._in_flight = 0
        self._idle_since = self._t0
        self.entries: list[tuple[float, float, float, int, str, int, int]] = []

    def start(self) -> tuple[float, float, int]:
        with self._lock:
            now = time.perf_counter()
            gap = 0.0 if self._in_flight else now - self._idle_since
            before = self._in_flight
            self._in_flight += 1
            return now, gap, before

    def end(self, started: tuple[float, float, int], kind: str, layer: int, expert: int) -> None:
        with self._lock:
            now = time.perf_counter()
            self._in_flight -= 1
            if self._in_flight == 0:
                self._idle_since = now
            t, gap, before = started
            self.entries.append((t - self._t0, now - t, gap, before, kind, layer, expert))


@dataclass
class SourceStats:
    """Cumulative; the forward pass reports per-call deltas.

    `read_seconds` is time the FORWARD PASS spent blocked on expert bytes:
    its own reads, plus waiting for a background read already in flight.
    `bytes_read` is what those blocking reads moved; background reads are
    counted separately. Sources without a cache or prefetcher leave the
    cache counters at 0 and count every load as a demand read."""

    loads: int = 0  # experts handed to the forward pass
    bytes_read: int = 0
    read_seconds: float = 0.0
    demand_reads: int = 0  # read by the forward pass itself, which waited for it
    cache_hits: int = 0  # already in RAM, not from a prefetch still unused
    prefetch_hits: int = 0  # in RAM thanks to a prefetch that had finished
    prefetch_waits: int = 0  # a prefetch was already reading it: waited for that read
    prefetches_issued: int = 0
    prefetch_reads: int = 0  # background reads completed
    prefetch_bytes: int = 0
    prefetches_wasted: int = 0  # read in the background, evicted before any use
    prefetches_cancelled: int = 0  # queued, never started, no longer useful
    prefetches_skipped_full: int = 0  # no evictable slot at the time
    prefetches_skipped_budget: int = 0  # over the adaptive prefetcher's read budget


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

    @property
    def supports_prefetch(self) -> bool:
        return False

    def prefetch(  # noqa: B027
        self,
        layer: int,
        experts: list[int],
        keep: list[tuple[int, int]],
        max_new_reads: int | None = None,
    ) -> None:
        """Start loading `experts` of `layer` in the background, in order,
        without evicting the (layer, expert) keys in `keep`, and starting at
        most `max_new_reads` reads (experts already cached cost none). A
        no-op for sources without a cache."""

    def end_layer(self, layer: int) -> None:  # noqa: B027
        """The forward pass is done with `layer`'s experts for this call."""

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
        self.read_log: ReadLog | None = None

    def load(self, layer: int, expert: int) -> ExpertWeights:
        t = time.perf_counter()
        logged = self.read_log.start() if self.read_log else None
        raw = self._reader.read_raw(layer, expert)
        if logged:
            self.read_log.end(logged, "demand", layer, expert)
        self.stats.read_seconds += time.perf_counter() - t
        self.stats.loads += 1
        self.stats.demand_reads += 1
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
