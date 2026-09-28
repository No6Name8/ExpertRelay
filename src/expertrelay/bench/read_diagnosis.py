"""Why does an expert read take ~4.9 ms in the runtime but 2.08 ms in the
machine profile's disk benchmark?

    python -m expertrelay.bench.read_diagnosis --files experts.bin experts.bin.new

Both time the same call (one unbuffered ReadFile of one record into a
page-aligned buffer), so this varies the CONDITIONS, one at a time, on real
store records instead of the profile's freshly written test file:

  back_to_back  one read at a time, no pause (queue depth 1): the profile's pattern
  gapped        bursts of 4 reads with a 40 ms pause after each burst: the
                runtime's pattern (a layer's 4 experts, then its compute)
  qd2, qd4      2 or 4 threads, each with its own handle and buffer, reading
                at once: more reads in flight

Every test reads the same seeded random sample of records, so the mix of
records that are in one piece on disk vs split across extents
(store.file_extents) is the same everywhere, and latencies are also
reported per piece count. Each named file (in the store directory) is
tested in every round, and rounds alternate the file order, so drift
(heat, background activity) shows up as round-to-round differences instead
of as a difference between files. Device read counters confirm every test
really read from the SSD.

Writes benchmarks/results/read_diagnosis.json. Run with nothing else using
the disk.
"""

from __future__ import annotations

import argparse
import statistics
import threading
import time
from pathlib import Path

import numpy as np

from expertrelay import REPO_ROOT
from expertrelay.benchmarking import append_benchmark_record, base_record, peak_process_rss_mb
from expertrelay.manager.profile import collect_machine_profile, device_read_bytes
from expertrelay.memory_budget import enforce_ram_budget
from expertrelay.paths import BENCHMARK_RESULTS_DIR
from expertrelay.store import unbuffered_io
from expertrelay.store.expert_reader import EXPERTS_INDEX
from expertrelay.store.file_extents import file_extents, pieces
from expertrelay.store.layout import ExpertIndexEntry, read_expert_index
from expertrelay.store.rewrite_experts import extent_summary

GAP_EVERY = 4
GAP_SECONDS = 0.040
QUEUE_DEPTHS = [2, 4]
MAX_RAM_GB = 0.2


def _stats(lat: list[float]) -> dict:
    ms = sorted(x * 1000 for x in lat)
    return {
        "reads": len(ms),
        "median_ms": statistics.median(ms),
        "mean_ms": statistics.fmean(ms),
        "p90_ms": ms[int(0.9 * (len(ms) - 1))],
    }


def _read_list(path: Path, entries: list[ExpertIndexEntry], out: list[float], gap: bool) -> None:
    size = entries[0].size
    with (
        unbuffered_io.unbuffered_handle(path, write=False) as h,
        unbuffered_io.aligned_buffer(size) as (_buf, addr),
    ):
        for i, e in enumerate(entries):
            t = time.perf_counter()
            unbuffered_io.read_at(h, addr, e.offset, e.size)
            out.append(time.perf_counter() - t)
            if gap and (i + 1) % GAP_EVERY == 0:
                time.sleep(GAP_SECONDS)


def run_test(path: Path, entries: list[ExpertIndexEntry], test: str, piece_counts: list[int]) -> dict:
    depth = int(test[2:]) if test.startswith("qd") else 1
    lanes = [entries[i::depth] for i in range(depth)]
    lat: list[list[float]] = [[] for _ in lanes]
    dev0, t0 = device_read_bytes(), time.perf_counter()
    if depth == 1:
        _read_list(path, entries, lat[0], gap=test == "gapped")
    else:
        threads = [
            threading.Thread(target=_read_list, args=(path, lane, out, False))
            for lane, out in zip(lanes, lat, strict=True)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    wall = time.perf_counter() - t0
    device = device_read_bytes() - dev0
    nbytes = sum(e.size for e in entries)
    # latencies back in sample order, to pair them with piece counts
    flat = [0.0] * len(entries)
    for i, out in enumerate(lat):
        flat[i::depth] = out
    reading = wall if test != "gapped" else sum(flat)  # gapped: throughput while reading, pauses excluded
    by_pieces = {}
    for label, keep in (("1", lambda p: p == 1), ("2+", lambda p: p > 1)):
        sel = [x for x, p in zip(flat, piece_counts, strict=True) if keep(p)]
        if sel:
            by_pieces[label] = _stats(sel)
    return {
        "test": test,
        "queue_depth": depth,
        **_stats(flat),
        "mb_per_s": nbytes / 1e6 / reading,
        "wall_seconds": wall,
        "device_bytes_read": device,
        "cache_bypass_verified": device >= nbytes,
        "by_pieces": by_pieces,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store-dir", type=Path, default=REPO_ROOT / "models" / "qwen1.5-moe-a2.7b-int8")
    ap.add_argument("--files", nargs="+", default=["experts.bin"], help="files in the store directory")
    ap.add_argument("--reads", type=int, default=240, help="records per test")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="", help="free text stored with the record, e.g. when it ran")
    args = ap.parse_args()

    layout, entries = read_expert_index(args.store_dir / EXPERTS_INDEX)
    enforce_ram_budget(max(QUEUE_DEPTHS) * layout.record_size, MAX_RAM_GB, "read diagnosis buffers")
    tests = ["back_to_back", "gapped", *[f"qd{d}" for d in QUEUE_DEPTHS]]
    files = {name: args.store_dir / name for name in args.files}
    extents = {name: file_extents(p) for name, p in files.items()}
    record = base_record(
        label="read diagnosis: expert records under different read conditions",
        seed=args.seed,
        model={"store": args.store_dir.name},
        config={
            "files": args.files,
            "reads_per_test": args.reads,
            "rounds": args.rounds,
            "tests": tests,
            "gap": {"every_reads": GAP_EVERY, "seconds": GAP_SECONDS},
            "record_bytes": layout.record_size,
            "label": args.label,
        },
        machine=collect_machine_profile(measure_disk=False),
    )
    record["files"] = {name: extent_summary(p, entries) for name, p in files.items()}
    results = []
    for rnd in range(args.rounds):
        rng = np.random.default_rng(args.seed + rnd)
        sample = [entries[i] for i in rng.choice(len(entries), size=args.reads, replace=False)]
        order = list(files) if rnd % 2 == 0 else list(files)[::-1]
        for name in order:
            counts = [pieces(extents[name], e.offset, e.size) for e in sample]
            for test in tests:
                r = {"round": rnd, "file": name, **run_test(files[name], sample, test, counts)}
                results.append(r)
                print(
                    f"round {rnd} {name:22s} {test:12s} median {r['median_ms']:.2f} ms  "
                    f"p90 {r['p90_ms']:.2f} ms  {r['mb_per_s']:.0f} MB/s  bypass {r['cache_bypass_verified']}",
                    flush=True,
                )
    record["results"] = results
    record["peak_rss_mb"] = peak_process_rss_mb()
    out = BENCHMARK_RESULTS_DIR / "read_diagnosis.json"
    append_benchmark_record(out, record)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
