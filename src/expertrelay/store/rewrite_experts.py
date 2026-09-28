"""Rewrite a store's experts.bin as one fresh, contiguous file.

    python -m expertrelay.store.rewrite_experts --store-dir models/qwen1.5-moe-a2.7b-int8
    python -m expertrelay.store.rewrite_experts --store-dir ... --swap

Why: the build wrote experts.bin record by record over many hours, so it
ended up in hundreds of extents (store.file_extents), and some records
span several. This copies it front to back into a new file whose full size
is allocated up front (unbuffered_io.set_size_and_rewind), with unbuffered
reads and write-through writes, so neither file passes through the OS file
cache.

Without --swap: the copy is written to experts.bin.new, then EVERY record
is read back from it and checked against the sha256 in experts_index.json.
With --swap: the existing experts.bin.new is verified again, record by
record, and only if all match is it put in place: experts.bin is renamed
to experts.bin.fragmented (kept, not deleted) and the copy becomes
experts.bin. The index is unchanged: offsets and contents are identical.
Windows only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from expertrelay.memory_budget import enforce_ram_budget
from expertrelay.store import unbuffered_io
from expertrelay.store.expert_reader import EXPERTS_BIN, EXPERTS_INDEX
from expertrelay.store.file_extents import file_extents, pieces
from expertrelay.store.layout import ExpertIndexEntry, read_expert_index

CHUNK_BYTES = 64 * 2**20
NEW_SUFFIX = ".new"
OLD_SUFFIX = ".fragmented"
MAX_RAM_GB = 0.2  # one copy buffer and one record buffer


def copy_unbuffered(src: Path, dst: Path, chunk: int = CHUNK_BYTES) -> float:
    """Seconds taken. `src`'s size must be a multiple of the I/O alignment,
    which every store file is."""
    size = src.stat().st_size
    t = time.perf_counter()
    with (
        unbuffered_io.unbuffered_handle(src, write=False) as rh,
        unbuffered_io.unbuffered_handle(dst, write=True) as wh,
        unbuffered_io.aligned_buffer(chunk) as (_buf, addr),
    ):
        unbuffered_io.set_size_and_rewind(wh, size)
        done = 0
        while done < size:
            n = min(chunk, size - done)
            unbuffered_io.read_at(rh, addr, done, n)
            unbuffered_io.write_sequential(wh, addr, n)
            done += n
    return time.perf_counter() - t


def bad_records(path: Path, entries: list[ExpertIndexEntry], record_size: int) -> list[tuple[int, int]]:
    """(layer, expert) of every record in `path` whose sha256 isn't the index's."""
    bad = []
    with (
        unbuffered_io.unbuffered_handle(path, write=False) as h,
        unbuffered_io.aligned_buffer(record_size) as (buf, addr),
    ):
        for e in entries:
            unbuffered_io.read_at(h, addr, e.offset, e.size)
            if hashlib.sha256(memoryview(buf)[: e.size]).hexdigest() != e.sha256:
                bad.append((e.layer, e.expert))
    return bad


def extent_summary(path: Path, entries: list[ExpertIndexEntry]) -> dict:
    ex = file_extents(path)
    split = [pieces(ex, e.offset, e.size) for e in entries]
    return {
        "extents": len(ex),
        "records_in_one_piece": sum(1 for p in split if p == 1),
        "records_split": sum(1 for p in split if p > 1),
        "max_pieces_per_record": max(split),
        "mean_pieces_per_record": sum(split) / len(split),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store-dir", type=Path, required=True)
    ap.add_argument("--swap", action="store_true", help="put the verified copy in place of experts.bin")
    args = ap.parse_args()

    enforce_ram_budget(CHUNK_BYTES + 16 * 2**20, MAX_RAM_GB, "experts.bin rewrite")
    store = args.store_dir
    src, dst = store / EXPERTS_BIN, store / (EXPERTS_BIN + NEW_SUFFIX)
    layout, entries = read_expert_index(store / EXPERTS_INDEX)
    report: dict = {"source": src.name}
    if not args.swap:
        report["before"] = extent_summary(src, entries)
        print(json.dumps(report["before"]), flush=True)
        report["copy_seconds"] = copy_unbuffered(src, dst)
    elif not dst.exists():
        raise SystemExit(f"no {dst.name} to swap in: run without --swap first")
    report["copy"] = extent_summary(dst, entries)
    # verified here in both modes: --swap never trusts an earlier check
    t = time.perf_counter()
    bad = bad_records(dst, entries, layout.record_size)
    report["verify_seconds"] = time.perf_counter() - t
    report["records_verified"] = len(entries)
    report["bad_records"] = bad
    if bad:
        print(json.dumps(report, indent=1))
        raise SystemExit(f"{len(bad)} record(s) of the copy don't match the index; not swapping")
    if args.swap:
        src.rename(store / (EXPERTS_BIN + OLD_SUFFIX))
        dst.rename(src)
        report["swapped"] = True
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
