"""Startup probe, read side: how long one expert read takes on the actual
store, with 1 and with 2 reads in flight.

The machine profile's disk benchmark (manager.profile.measure_disk_read)
reads a scratch file; this reads the store itself, with the runtime's own
reader (unbuffered, one aligned read per expert), so it measures exactly
the reads the cache and prefetcher will issue. The compute side of the
probe (time per layer) needs the model and so lives in the runtime
(runtime.auto); the Manager only receives its numbers.

Experts are sampled without repetition (seeded), and the 1-in-flight and
2-in-flight samples don't share experts, so neither hits a drive cache
warmed by the other. bench/read_diagnosis.py measured ~4.2 ms per record
back to back and +25% throughput with two reads in flight on the dev
machine; this probe repeats a small version of that at every --auto start.
"""

from __future__ import annotations

import random
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from expertrelay.store.expert_reader import ExpertStoreReader
from expertrelay.store.unbuffered_io import aligned_buffer

DEFAULT_READS = 48  # per depth; ~0.4 s of reads in all at ~4 ms each


@dataclass(frozen=True)
class ReadProbe:
    reads_per_depth: int
    seconds_per_read_1: float  # median latency, one read at a time
    seconds_per_read_2: float  # wall time / reads, two reads always in flight
    record_bytes: int

    @property
    def two_in_flight_gain(self) -> float:
        """Throughput with 2 reads in flight relative to 1 (1.25 = 25% more)."""
        return self.seconds_per_read_1 / self.seconds_per_read_2


def _sample(store_dir: Path, n: int, seed: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    with ExpertStoreReader(store_dir) as r:
        keys = [(e.layer, e.expert) for e in r.entries()]
    picked = random.Random(seed).sample(keys, min(len(keys), 2 * n))
    return picked[: len(picked) // 2], picked[len(picked) // 2 :]


def probe_reads(store_dir: Path, reads: int = DEFAULT_READS, seed: int = 0) -> ReadProbe:
    one, two = _sample(store_dir, reads, seed)
    with ExpertStoreReader(store_dir) as reader:
        record = reader.layout.record_size
        with aligned_buffer(record) as (_buf, address):
            latencies = []
            for layer, expert in one:
                t = time.perf_counter()
                reader.read_into(layer, expert, address)
                latencies.append(time.perf_counter() - t)

    # each thread opens its reader (which parses the store's index) BEFORE the
    # clock starts; the barrier releases both threads and the clock together
    start = threading.Barrier(3)

    def worker(keys: list[tuple[int, int]]) -> None:
        with ExpertStoreReader(store_dir) as r, aligned_buffer(record) as (_b, addr):
            start.wait()
            for layer, expert in keys:
                r.read_into(layer, expert, addr)

    threads = [threading.Thread(target=worker, args=(two[i::2],)) for i in range(2)]
    for th in threads:
        th.start()
    start.wait()
    t = time.perf_counter()
    for th in threads:
        th.join()
    wall = time.perf_counter() - t
    return ReadProbe(
        reads_per_depth=len(one),
        seconds_per_read_1=statistics.median(latencies),
        seconds_per_read_2=wall / max(len(two), 1),
        record_bytes=record,
    )
