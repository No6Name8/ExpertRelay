"""The runtime expert cache (cache.expert_cache) on a tiny real store.

Background reads are made deterministic where it matters: the I/O thread's
reads are held on an Event, so a test decides when a prefetch is "still
loading" or "still queued" instead of racing the thread.
"""

from __future__ import annotations

import sys
import threading

import numpy as np
import pytest
from store_helpers import build_test_store, random_checkpoint

from expertrelay.cache.expert_cache import CachedExpertSource, CacheTooSmall
from expertrelay.store.expert_reader import ExpertStoreReader

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="unbuffered expert reads are Windows-only")

CONFIG = {
    "vocab_size": 64,
    "hidden_size": 32,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 16,
    "shared_expert_intermediate_size": 24,
    "norm_topk_prob": False,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
}


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    d = tmp_path_factory.mktemp("store")
    build_test_store(d, CONFIG, random_checkpoint(CONFIG, seed=2), mp)
    mp.undo()
    return d


@pytest.fixture(scope="module")
def record(store) -> int:
    with ExpertStoreReader(store) as r:
        return r.layout.record_size


def cache(store, record, slots: float, **kw) -> CachedExpertSource:
    kw.setdefault("min_free_slots", 1)
    return CachedExpertSource(store, capacity_bytes=int(slots * record), **kw)


class GatedReads:
    """Patch ExpertStoreReader.read_into so reads on I/O threads block until
    released, and record every read (thread kind, key)."""

    def __init__(self, monkeypatch):
        self.release = threading.Event()
        self.started = threading.Event()
        self.reads: list[tuple[str, tuple[int, int]]] = []
        original = ExpertStoreReader.read_into
        gate = self

        def read_into(reader, layer, expert, address):
            io = threading.current_thread().name.startswith("expert-io")
            gate.reads.append(("io" if io else "forward", (layer, expert)))
            if io:
                gate.started.set()
                gate.release.wait(10)
            return original(reader, layer, expert, address)

        monkeypatch.setattr(ExpertStoreReader, "read_into", read_into)

    def count(self, key) -> int:
        return sum(1 for _, k in self.reads if k == key)


def weights_equal(a, b) -> bool:
    return all(np.array_equal(a[m].q, b[m].q) and np.array_equal(a[m].scales, b[m].scales) for m in a)


def test_budget_buys_whole_slots_and_nothing_more(store, record):
    c = cache(store, record, 5.9)
    assert c.num_slots == 5
    assert c.capacity_bytes == 5 * record
    for layer in range(3):
        for e in range(8):
            c.load(layer, e)
    c.end_layer(2)
    assert len(c.cached_keys()) == 5
    c.close()


def test_cached_bytes_are_the_stored_expert(store, record):
    c = cache(store, record, 4)
    with ExpertStoreReader(store) as r:
        for layer, e in [(0, 1), (2, 7), (1, 3), (0, 1)]:
            w = c.load(layer, e)
            ref = r.read(layer, e)
            for m in w:
                np.testing.assert_array_equal(w[m].q, ref.q[m])
                np.testing.assert_array_equal(w[m].scales, ref.scales[m])
    c.close()


def test_lru_eviction(store, record):
    c = cache(store, record, 3)
    for key in [(0, 0), (0, 1), (0, 2), (0, 0), (0, 3)]:  # (0, 1) is least recently used when (0, 3) arrives
        c.load(*key)
    c.end_layer(0)
    assert c.cached_keys() == {(0, 0), (0, 2), (0, 3)}
    assert (c.stats.demand_reads, c.stats.cache_hits, c.stats.loads) == (4, 1, 5)
    c.close()


def test_expert_in_use_is_not_evicted(store, record):
    c = cache(store, record, 2)
    w = c.load(0, 0)
    before = {m: w[m].q.copy() for m in w}
    c.prefetch(1, [0, 1, 2], keep=[])  # only one slot is free: the in-use one must survive
    c.close()
    assert all(np.array_equal(w[m].q, before[m]) for m in w)
    assert c.stats.prefetches_skipped_full >= 1


def test_pinned_experts_stay(store, record):
    pinned = [(0, e) for e in range(8)]
    c = cache(store, record, 11, pinned=pinned)
    reads_after_pinning = c.stats.demand_reads
    for layer in (1, 2):
        for e in range(8):
            c.load(layer, e)
    for e in range(8):
        c.load(0, e)
    c.end_layer(0)
    assert set(pinned) <= c.cached_keys()
    assert c.stats.demand_reads - reads_after_pinning == 16  # layers 1-2 only; layer 0 all hits
    assert c.stats.cache_hits == 8
    c.close()


def test_too_small_budget_is_rejected(store, record):
    with pytest.raises(CacheTooSmall):
        cache(store, record, 9, pinned=[(0, e) for e in range(8)], min_free_slots=2)


def test_prefetch_then_use_is_a_prefetch_hit(store, record):
    c = cache(store, record, 6)
    c.prefetch(1, [3, 5], keep=[])
    for _ in range(2000):
        if c.stats.prefetch_reads == 2:
            break
        threading.Event().wait(0.005)
    c.load(1, 5)
    c.load(1, 3)
    c.end_layer(1)
    assert (c.stats.prefetch_hits, c.stats.demand_reads, c.stats.prefetch_reads) == (2, 0, 2)
    c.close()


def test_waits_for_an_in_flight_read_instead_of_reading_again(store, record, monkeypatch):
    gate = GatedReads(monkeypatch)
    c = cache(store, record, 6)
    c.prefetch(2, [4], keep=[])
    assert gate.started.wait(10)  # the I/O thread is now reading (2, 4)
    threading.Timer(0.1, gate.release.set).start()
    w = c.load(2, 4)  # must wait for that read, not start another
    c.end_layer(2)
    c.close()
    assert gate.count((2, 4)) == 1
    assert (c.stats.prefetch_waits, c.stats.demand_reads) == (1, 0)
    with ExpertStoreReader(store) as r:
        ref = r.read(2, 4)
    assert all(np.array_equal(w[m].q, ref.q[m]) for m in w)


def test_queued_prefetch_is_taken_over_not_read_twice(store, record, monkeypatch):
    gate = GatedReads(monkeypatch)
    c = cache(store, record, 6)
    c.prefetch(1, [0, 6], keep=[])  # (1, 0) starts and blocks; (1, 6) stays queued
    assert gate.started.wait(10)
    c.load(1, 6)  # read right here by the forward pass
    gate.release.set()
    c.end_layer(1)
    c.close()  # joins the I/O thread: it has seen (1, 6)'s queue entry by now
    assert gate.count((1, 6)) == 1
    assert gate.reads.count(("forward", (1, 6))) == 1
    assert c.stats.demand_reads == 1
    assert c.stats.prefetches_cancelled == 1


def test_end_layer_cancels_prefetches_that_never_started(store, record, monkeypatch):
    gate = GatedReads(monkeypatch)
    c = cache(store, record, 6)
    c.prefetch(1, [0, 1, 2], keep=[])
    assert gate.started.wait(10)
    c.end_layer(1)  # (1, 1) and (1, 2) are still queued
    gate.release.set()
    c.close()
    assert [k for _, k in gate.reads] == [(1, 0)]
    assert c.stats.prefetches_cancelled == 2


def test_unused_prefetch_evicted_counts_as_wasted(store, record):
    c = cache(store, record, 3)
    c.prefetch(2, [7], keep=[])
    for _ in range(2000):
        if c.stats.prefetch_reads == 1:
            break
        threading.Event().wait(0.005)
    for e in range(3):
        c.load(0, e)
    c.end_layer(0)
    assert c.stats.prefetches_wasted == 1
    assert (2, 7) not in c.cached_keys()
    c.close()


def test_prefetch_keeps_the_listed_experts(store, record):
    c = cache(store, record, 3)
    for e in range(3):
        c.load(0, e)
    c.end_layer(0)
    c.prefetch(1, [4], keep=[(0, 0)])  # (0, 0) was least recently used, but is about to be needed
    c.close()
    assert (0, 0) in c.cached_keys()
    assert (0, 1) not in c.cached_keys()


def test_prefetch_read_budget_counts_only_new_reads(store, record):
    c = cache(store, record, 8)
    c.load(1, 0)
    c.end_layer(1)
    c.prefetch(1, [0, 2, 3, 4], keep=[], max_new_reads=2)  # (1, 0) is cached: costs nothing
    c.close()
    assert c.stats.prefetches_issued == 2
    assert c.stats.prefetches_skipped_budget == 1
    assert {(1, 0), (1, 2), (1, 3)} <= c.cached_keys()
    assert (1, 4) not in c.cached_keys()


def test_read_log_records_reads_with_their_idle_gap(store, record):
    from expertrelay.runtime.weights import ReadLog

    c = cache(store, record, 6)
    c.read_log = ReadLog()
    c.load(0, 1)
    threading.Event().wait(0.05)  # the drive sits idle for ~50 ms
    c.load(0, 2)
    c.prefetch(2, [5], keep=[])
    c.end_layer(0)
    c.close()
    entries = c.read_log.entries
    kinds = [e[4] for e in entries]
    assert kinds.count("demand") == 2 and kinds.count("prefetch") == 1
    second = [e for e in entries if e[4] == "demand"][1]
    assert second[2] >= 0.04  # the idle gap before it
    assert all(e[1] > 0 for e in entries)  # every read took some time


def test_resize_shrinks_keeping_the_most_recent_experts(store, record):
    c = cache(store, record, 6)
    for e in range(5):
        c.load(0, e)
    c.end_layer(0)
    assert c.resize(3 * record) == 3 and c.capacity_bytes == 3 * record
    assert c.cached_keys() == {(0, 2), (0, 3), (0, 4)}  # the least recently used went first
    hits = c.stats.cache_hits
    w = c.load(0, 4)
    assert c.stats.cache_hits == hits + 1
    with ExpertStoreReader(store) as r:
        ref = r.read(0, 4)
        for m in w:
            np.testing.assert_array_equal(w[m].q, ref.q[m])
    del w
    c.end_layer(0)
    c.close()


def test_resize_grows_and_new_slots_are_used(store, record):
    c = cache(store, record, 2)
    assert c.resize(5 * record) == 5
    for e in range(5):
        c.load(1, e)
    c.end_layer(1)
    assert len(c.cached_keys()) == 5 and c.stats.demand_reads == 5
    c.close()


def test_resize_keeps_pinned_and_refuses_too_small(store, record):
    c = cache(store, record, 6, pinned=[(0, 0), (0, 1)], min_free_slots=2)
    for e in range(2, 6):
        c.load(1, e)
    c.end_layer(1)
    with pytest.raises(CacheTooSmall):
        c.resize(3 * record)
    assert c.num_slots == 6  # unchanged after a refusal
    assert c.resize(4 * record) == 4
    assert {(0, 0), (0, 1)} <= c.cached_keys()
    c.close()


def test_resize_cancels_queued_and_waits_for_in_flight_reads(store, record, monkeypatch):
    gate = GatedReads(monkeypatch)
    c = cache(store, record, 6, io_threads=1)
    c.prefetch(2, [0, 1, 2], keep=[])
    assert gate.started.wait(5)
    resized = threading.Thread(target=c.resize, args=(2 * record,))
    resized.start()
    gate.release.set()
    resized.join(10)
    assert not resized.is_alive() and c.num_slots == 2
    assert c.stats.prefetches_cancelled >= 1
    c.close()
