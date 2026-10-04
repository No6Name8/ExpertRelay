"""The runtime's expert cache: routed experts kept in RAM under a fixed
budget, LRU eviction, optional pinned experts, and background prefetching.

Memory. The budget buys a whole number of slots, each one expert record
(store layout record_size, 8.67 MB for Qwen1.5-MoE-A2.7B), each its own
page-aligned block. Nothing else grows: reads go straight from the SSD
into a slot (unbuffered, store.expert_reader.read_into), and the forward
pass computes on views into the slot. So RAM for experts is exactly
slots x record_size, whatever happens. The budget can change between
tokens (resize(): the Manager's live memory budget); a slot is its own
allocation so that shrinking really returns its memory.

Slot states:
  FREE     holds nothing
  QUEUED   a prefetch is waiting for an I/O thread
  LOADING  a read is in progress (an I/O thread's or the forward pass's own)
  READY    holds the expert's bytes

A slot the forward pass is using (refs > 0), a pinned slot, or a slot being
read is never evicted. Eviction takes the least recently used READY slot.
The forward pass touches an expert when it uses it; a prefetch touches
experts it predicts that are already cached, so they aren't evicted just
before use.

"Wait for that load instead of starting a second one": when the forward
pass needs an expert whose prefetch is LOADING, it waits for that read.
When the prefetch is only QUEUED (no I/O thread has started it), the
forward pass takes it over and reads it itself right away; the I/O thread
then skips it. Either way the expert is read once.

One ExpertStoreReader (one file handle) per thread, since a handle has one
file position. Reads release the GIL (ctypes), and so does most of numpy's
compute, so background reads really do overlap compute.

Windows only, like the unbuffered reader (docs/limitations.md).
"""

from __future__ import annotations

import queue
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from expertrelay.runtime.weights import ExpertSource, ExpertWeights, ReadLog, SourceStats, _to_matrices
from expertrelay.store.expert_reader import ExpertStoreReader
from expertrelay.store.layout import parse_expert_record
from expertrelay.store.unbuffered_io import ALIGNMENT

FREE, QUEUED, LOADING, READY = range(4)
# Starting estimate of one background read, until reads have been timed:
# bench/read_diagnosis.py measured ~4.2 ms back to back for a store record.
INITIAL_READ_SECONDS = 0.0045
READ_EMA = 0.2  # weight of the newest read in the running estimate
Key = tuple[int, int]  # (layer, expert)


class CacheTooSmall(RuntimeError):
    pass


@dataclass
class _Slot:
    index: int
    address: int
    buf: np.ndarray  # uint8 view of this slot's bytes
    key: Key | None = None
    state: int = FREE
    refs: int = 0
    pinned: bool = False
    prefetched: bool = False  # filled by a prefetch and not used since
    done: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None


def _aligned_pool(nbytes: int) -> np.ndarray:
    """One uint8 block whose first byte is ALIGNMENT-aligned, as unbuffered
    reads require. Plain numpy memory, so it's freed like any array."""
    raw = np.empty(nbytes + ALIGNMENT, dtype=np.uint8)
    pad = (-raw.ctypes.data) % ALIGNMENT
    return raw[pad : pad + nbytes]


def slots_for(capacity_bytes: int, record_size: int) -> int:
    return capacity_bytes // record_size


class CachedExpertSource(ExpertSource):
    def __init__(
        self,
        store_dir: Path,
        *,
        capacity_bytes: int,
        pinned: list[Key] = (),
        io_threads: int = 1,
        min_free_slots: int = 1,
    ):
        """`min_free_slots`: slots that must remain after pinning, e.g. one
        layer's experts + one prefetch batch. Checked here so a budget too
        small for the settings fails at startup, not mid-generation."""
        self._demand_reader = ExpertStoreReader(store_dir).__enter__()
        self.layout = self._demand_reader.layout
        record = self.layout.record_size
        n = slots_for(capacity_bytes, record)
        if n < len(pinned) + min_free_slots:
            self._demand_reader.__exit__(None, None, None)
            raise CacheTooSmall(
                f"{capacity_bytes / 1e9:.2f} GB holds {n} experts; need {len(pinned)} pinned + "
                f"{min_free_slots} free"
            )
        self.capacity_bytes = n * record
        self.min_free_slots = min_free_slots
        self.stats = SourceStats()
        self.read_log: ReadLog | None = None
        self.io_threads = io_threads
        self._read_seconds_ema = INITIAL_READ_SECONDS
        self._slots = [self._new_slot(i) for i in range(n)]
        self._map: dict[Key, _Slot] = {}
        self._lru: OrderedDict[int, None] = OrderedDict()  # unpinned slots holding a key, oldest first
        self._free = list(range(n - 1, -1, -1))
        self._lock = threading.Lock()
        self._current: _Slot | None = None
        self._queue: queue.Queue = queue.Queue()
        self._closed = False
        self._threads = [
            threading.Thread(target=self._io_worker, name=f"expert-io-{i}", daemon=True)
            for i in range(io_threads)
        ]
        for t in self._threads:
            t.start()
        for layer, expert in pinned:
            self._pin(layer, expert)

    @property
    def num_slots(self) -> int:
        return len(self._slots)

    @property
    def min_slots(self) -> int:
        """The smallest size resize() accepts: pinned experts + min_free_slots."""
        with self._lock:
            return sum(s.pinned for s in self._slots) + self.min_free_slots

    @property
    def supports_prefetch(self) -> bool:
        return True

    @property
    def read_seconds_estimate(self) -> float:
        """Running estimate of one background read, from the reads so far."""
        return self._read_seconds_ema

    # ------------------------------------------------------------------ internal

    def _new_slot(self, index: int) -> _Slot:
        buf = _aligned_pool(self.layout.record_size)
        return _Slot(index, buf.ctypes.data, buf)

    def _touch(self, slot: _Slot) -> None:
        if not slot.pinned:
            self._lru[slot.index] = None
            self._lru.move_to_end(slot.index)

    def _drop(self, slot: _Slot) -> None:
        """Forget the slot's key (lock held)."""
        if slot.key is not None and self._map.get(slot.key) is slot:
            del self._map[slot.key]
        self._lru.pop(slot.index, None)
        slot.key, slot.state, slot.prefetched, slot.error = None, FREE, False, None

    def _victim(self, *, take_queued: bool) -> _Slot | None:
        """A slot to reuse (lock held): free, else the least recently used
        READY slot nobody is using, else (demand only) a QUEUED prefetch."""
        if self._free:
            return self._slots[self._free.pop()]
        for i in self._lru:
            s = self._slots[i]
            if s.state == READY and s.refs == 0:
                if s.prefetched:
                    self.stats.prefetches_wasted += 1
                self._drop(s)
                return s
        if take_queued:
            for i in self._lru:
                s = self._slots[i]
                if s.state == QUEUED:
                    self.stats.prefetches_cancelled += 1
                    self._drop(s)
                    return s
        return None

    def _pin(self, layer: int, expert: int) -> None:
        with self._lock:
            slot = self._slots[self._free.pop()]
            slot.key, slot.state, slot.pinned = (layer, expert), LOADING, True
            self._map[slot.key] = slot
        self._demand_reader.read_into(layer, expert, slot.address)
        with self._lock:
            slot.state = READY
            slot.done.set()

    def _release_current(self) -> None:
        if self._current is not None:
            with self._lock:
                self._current.refs -= 1
            self._current = None

    def _io_worker(self) -> None:
        with ExpertStoreReader(self._demand_reader.store_dir) as reader:
            while True:
                item = self._queue.get()
                if item is None:
                    return
                slot, key = item
                with self._lock:
                    if slot.state != QUEUED or slot.key != key:
                        continue  # cancelled, or the forward pass took it over
                    slot.state = LOADING
                logged = self.read_log.start() if self.read_log else None
                t = time.perf_counter()
                try:
                    n = reader.read_into(key[0], key[1], slot.address)
                    err = None
                except BaseException as e:  # handed to whoever waits for this slot
                    n, err = 0, e
                took = time.perf_counter() - t
                if logged:
                    self.read_log.end(logged, "prefetch", key[0], key[1])
                with self._lock:
                    slot.error = err
                    slot.state = READY
                    if err is None:
                        self.stats.prefetch_reads += 1
                        self.stats.prefetch_bytes += n
                        self._read_seconds_ema += READ_EMA * (took - self._read_seconds_ema)
                    slot.done.set()

    # -------------------------------------------------------------- ExpertSource

    def load(self, layer: int, expert: int) -> ExpertWeights:
        """The expert's weights as views into its slot, valid until the next
        load() or end_layer(): the slot can't be evicted before then."""
        self._release_current()
        key = (layer, expert)
        t0 = time.perf_counter()
        read_here = wait = False
        with self._lock:
            slot = self._map.get(key)
            if slot is not None and slot.state == READY and slot.error is None:
                if slot.prefetched:
                    self.stats.prefetch_hits += 1
                    slot.prefetched = False
                else:
                    self.stats.cache_hits += 1
            elif slot is not None and slot.state == QUEUED:
                slot.state, slot.prefetched = LOADING, False
                self.stats.prefetches_cancelled += 1
                read_here = True
            elif slot is not None and slot.state == LOADING:
                wait = True
            else:
                if slot is not None:  # a failed background read: read it here instead
                    self._drop(slot)
                    self._free.append(slot.index)
                slot = self._victim(take_queued=True)
                if slot is None:
                    raise CacheTooSmall("no evictable slot for a demand read")
                slot.key, slot.state, slot.prefetched = key, LOADING, False
                slot.done = threading.Event()
                self._map[key] = slot
                read_here = True
            slot.refs += 1
            self._touch(slot)
        if wait:
            slot.done.wait()
            with self._lock:
                if slot.error is not None:
                    raise RuntimeError(f"background read of expert {key} failed") from slot.error
                slot.prefetched = False
                self.stats.prefetch_waits += 1
        if read_here:
            logged = self.read_log.start() if self.read_log else None
            n = self._demand_reader.read_into(layer, expert, slot.address)
            if logged:
                self.read_log.end(logged, "demand", layer, expert)
            with self._lock:
                slot.state = READY
                slot.done.set()
                self.stats.demand_reads += 1
                self.stats.bytes_read += n
        if wait or read_here:
            self.stats.read_seconds += time.perf_counter() - t0
        self.stats.loads += 1
        self._current = slot
        return _to_matrices(parse_expert_record(self.layout, slot.buf))

    def prefetch(
        self, layer: int, experts: list[int], keep: list[Key], max_new_reads: int | None = None
    ) -> None:
        new_reads = 0
        with self._lock:
            for k in keep:
                s = self._map.get(k)
                if s is not None:
                    self._touch(s)
            for expert in experts:
                key = (layer, int(expert))
                slot = self._map.get(key)
                if slot is not None:
                    self._touch(slot)
                    continue
                if max_new_reads is not None and new_reads >= max_new_reads:
                    self.stats.prefetches_skipped_budget += 1
                    continue
                slot = self._victim(take_queued=False)
                if slot is None:
                    self.stats.prefetches_skipped_full += 1
                    continue
                slot.key, slot.state, slot.prefetched = key, QUEUED, True
                slot.done = threading.Event()
                self._map[key] = slot
                self._touch(slot)
                self.stats.prefetches_issued += 1
                new_reads += 1
                self._queue.put((slot, key))

    def end_layer(self, layer: int) -> None:
        """Release the expert in use, and cancel prefetches for `layer` that
        never started: this call is done with that layer."""
        self._release_current()
        with self._lock:
            for s in self._slots:
                if s.state == QUEUED and s.key is not None and s.key[0] == layer:
                    self.stats.prefetches_cancelled += 1
                    self._drop(s)
                    self._free.append(s.index)

    def resize(self, capacity_bytes: int) -> int:
        """Change the budget to `capacity_bytes` (whole slots); returns the new
        slot count. Call between forward calls: queued prefetches are
        cancelled and reads in flight are waited for first. Shrinking drops
        free slots, then the least recently used experts; pinned experts
        stay. Refused (CacheTooSmall, size unchanged) if the pinned experts
        plus min_free_slots wouldn't fit."""
        record = self.layout.record_size
        n = slots_for(capacity_bytes, record)
        self._release_current()
        with self._lock:
            pinned = sum(s.pinned for s in self._slots)
            if n < pinned + self.min_free_slots:
                raise CacheTooSmall(
                    f"{capacity_bytes / 1e9:.2f} GB holds {n} experts; need {pinned} pinned + "
                    f"{self.min_free_slots} free"
                )
            for s in self._slots:
                if s.state == QUEUED:
                    self.stats.prefetches_cancelled += 1
                    self._drop(s)
                    self._free.append(s.index)
            loading = [s for s in self._slots if s.state == LOADING]
        for s in loading:
            s.done.wait()
        with self._lock:
            if n < len(self._slots):
                ready = [self._slots[i] for i in self._lru]  # least recently used first
                drop = [s for s in self._slots if s.state == FREE and not s.pinned]
                drop += [s for s in ready if s.refs == 0 and s.state == READY]
                for s in drop[: len(self._slots) - n]:
                    if s.prefetched:
                        self.stats.prefetches_wasted += 1
                    self._drop(s)
                    s.index = -1
                kept = [s for s in self._slots if s.index != -1]
            else:
                kept = self._slots + [self._new_slot(-1) for _ in range(n - len(self._slots))]
            order = [self._slots[i] for i in self._lru]  # keep the LRU order across re-indexing
            for i, s in enumerate(kept):
                s.index = i
            self._slots = kept
            self._lru = OrderedDict((s.index, None) for s in order if s.index != -1)
            self._free = [s.index for s in reversed(kept) if s.state == FREE]
            self.capacity_bytes = len(kept) * record
            return len(kept)

    def cached_keys(self) -> set[Key]:
        with self._lock:
            return {k for k, s in self._map.items() if s.state == READY}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._release_current()
        for _ in self._threads:
            self._queue.put(None)
        for t in self._threads:
            t.join()
        self._demand_reader.__exit__(None, None, None)
