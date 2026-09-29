"""Build an int4 expert store from the ORIGINAL bf16 checkpoint.

    python -m expertrelay.store.build_int4_store --group-size 128
    python -m expertrelay.store.build_int4_store --group-size 64

Routed experts only are re-quantized (store.int4: symmetric round-to-nearest,
float16 scales per group of `group-size` input columns), each straight
from the bf16 weights in the local checkpoint (store.download_checkpoint),
never from the int8 store. Everything resident (attention, shared expert,
router, norms, embedding, lm_head) is the int8 store's, unchanged: its
resident.bin, resident_index.json and tokenizer files are hard-linked
(copied if the volume can't link), so the two stores differ ONLY in the
routed experts.

Records keep the int8 store's rules: one 4096-aligned record per expert at
slot (layer * experts + expert), loadable with one read, sha256 per record
in experts_index.json, with the quantization error per matrix. After
writing, every record is read back and its sha256 checked.

Writes models/<int8 store name with int8 -> int4g{G}>/ and appends a build
record (sizes, error statistics next to the int8 store's) to
benchmarks/results/int4_store_build.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import time
from pathlib import Path

from expertrelay import REPO_ROOT
from expertrelay.benchmarking import append_benchmark_record, base_record, peak_process_rss_mb
from expertrelay.manager.profile import collect_machine_profile
from expertrelay.memory_budget import enforce_ram_budget
from expertrelay.paths import BENCHMARK_RESULTS_DIR, MODELS_ROOT
from expertrelay.store.download_checkpoint import checkpoint_dir
from expertrelay.store.expert_reader import EXPERTS_BIN, EXPERTS_INDEX
from expertrelay.store.int4 import (
    Int4RecordLayout,
    pack_int4,
    quant_error,
    quantize_groupwise_int4,
    serialize_int4_record,
)
from expertrelay.store.layout import (
    ExpertIndexEntry,
    expert_offset,
    expert_tensor_name,
    read_expert_index,
    sha256_file_region,
    write_expert_index,
)
from expertrelay.store.safetensors_local import LocalCheckpoint

SHARED_FILES = ("resident.bin", "resident_index.json", "tokenizer.json", "tokenizer_config.json")
MAX_RAM_GB = 0.5  # one expert's three f32 matrices plus its record, with room


def _display(path: Path) -> str:
    """Repo-relative when inside the repo (records stay portable), else absolute."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def link_or_copy(src: Path, dst: Path) -> str:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def build(int8_store: Path, bf16_dir: Path, out_dir: Path, group: int) -> dict:
    manifest = json.loads((int8_store / "store.json").read_text())
    config = manifest["source"]["config"]
    layers, experts = config["num_hidden_layers"], config["num_experts"]
    layout = Int4RecordLayout.from_config(config, group)
    ckpt = LocalCheckpoint(bf16_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    entries: list[ExpertIndexEntry] = []
    with open(out_dir / EXPERTS_BIN, "wb") as f:
        f.truncate(layers * experts * layout.record_size)
        for layer in range(layers):
            for e in range(experts):
                quantized, errors = {}, {}
                sq_err = sq_norm = 0.0
                for name, _shape in layout.matrices:
                    w = ckpt.get(expert_tensor_name(layer, e, name))
                    q, scales = quantize_groupwise_int4(w, group)
                    packed = pack_int4(q)
                    quantized[name] = (packed, scales)
                    err = quant_error(w, packed, scales, group)
                    sq_err, sq_norm = sq_err + err.pop("sq_error"), sq_norm + err.pop("sq_norm")
                    errors[name] = err
                errors["expert"] = {"rel_fro_error": (sq_err / sq_norm) ** 0.5 if sq_norm else 0.0}
                record = serialize_int4_record(layout, quantized)
                offset = expert_offset(layer, e, experts, layout.record_size)
                f.seek(offset)
                f.write(record)
                entries.append(
                    ExpertIndexEntry(
                        layer=layer,
                        expert=e,
                        offset=offset,
                        size=layout.record_size,
                        shapes={n: list(s) for n, s in layout.matrices},
                        sha256=hashlib.sha256(record).hexdigest(),
                        quant_error=errors,
                    )
                )
            print(f"layer {layer + 1}/{layers} ({time.perf_counter() - t0:.0f} s)", flush=True)
        f.flush()
        os.fsync(f.fileno())
    write_expert_index(out_dir / EXPERTS_INDEX, layout, entries)
    build_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    bad = [
        (x.layer, x.expert)
        for x in entries
        if sha256_file_region(out_dir / EXPERTS_BIN, x.offset, x.size) != x.sha256
    ]
    verify_s = time.perf_counter() - t0

    linked = {n: link_or_copy(int8_store / n, out_dir / n) for n in SHARED_FILES if (int8_store / n).exists()}
    store = {
        **{k: v for k, v in manifest.items() if k not in ("expert_layout", "runs")},
        "expert_layout": layout.to_dict(),
        "experts_quantization": {
            "method": "int4 symmetric round-to-nearest, float16 scale per group of input columns (store.int4)",
            "group_size": group,
            "from": f"bf16 checkpoint {_display(bf16_dir)}",
        },
        "resident_from": {"store": int8_store.name, "files": linked},
    }
    (out_dir / "store.json").write_text(json.dumps(store, indent=1))
    return {
        "entries": entries,
        "layout": layout,
        "bad": bad,
        "build_s": build_s,
        "verify_s": verify_s,
        "linked": linked,
    }


def error_stats(entries: list[ExpertIndexEntry]) -> dict:
    rel = [x.quant_error["expert"]["rel_fro_error"] for x in entries]
    worst = max(entries, key=lambda x: x.quant_error["expert"]["rel_fro_error"])
    max_over_scale = max(
        v["max_abs_error_over_scale"] for x in entries for k, v in x.quant_error.items() if k != "expert"
    )
    return {
        "experts": len(rel),
        "rel_fro_error_mean": statistics.fmean(rel),
        "rel_fro_error_median": statistics.median(rel),
        "rel_fro_error_max": max(rel),
        "worst_expert": [worst.layer, worst.expert],
        "max_abs_error_over_scale": max_over_scale,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group-size", type=int, required=True)
    ap.add_argument("--int8-store", type=Path, default=MODELS_ROOT / "qwen1.5-moe-a2.7b-int8")
    ap.add_argument("--bf16-dir", type=Path, default=None, help="default: the int8 store's source checkpoint")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    enforce_ram_budget(200 * 2**20, MAX_RAM_GB, "int4 store build (one expert at a time)")
    int8_store = args.int8_store.resolve()
    source = json.loads((int8_store / "store.json").read_text())["source"]
    bf16_dir = (args.bf16_dir or checkpoint_dir(source["repo_id"], source["revision"])).resolve()
    out_dir = (
        args.out_dir or int8_store.with_name(int8_store.name.replace("int8", f"int4g{args.group_size}"))
    ).resolve()
    record = base_record(
        label=f"int4 expert store build (group size {args.group_size})",
        seed=0,
        model={"source_repo": source["repo_id"], "revision": source["revision"], "store": out_dir.name},
        config={
            "group_size": args.group_size,
            "bf16_checkpoint": bf16_dir.relative_to(REPO_ROOT).as_posix(),
            "resident_from": int8_store.name,
        },
        machine=collect_machine_profile(measure_disk=False),
    )
    out = build(int8_store, bf16_dir, out_dir, args.group_size)
    int8_layout, int8_entries = read_expert_index(int8_store / EXPERTS_INDEX)
    layout = out["layout"]
    record["summary"] = {
        "store": out_dir.name,
        "record_size": layout.record_size,
        "weights_bytes": layout.weights_bytes,
        "scales_bytes": layout.scales_bytes,
        "experts_bin_bytes": len(out["entries"]) * layout.record_size,
        "records_verified": len(out["entries"]),
        "bad_records": out["bad"],
        "build_seconds": out["build_s"],
        "verify_seconds": out["verify_s"],
        "shared_files": out["linked"],
        "quant_error": error_stats(out["entries"]),
        "int8_for_comparison": {
            "record_size": int8_layout.record_size,
            "experts_bin_bytes": len(int8_entries) * int8_layout.record_size,
            "quant_error": error_stats(int8_entries),
        },
    }
    record["peak_rss_mb"] = peak_process_rss_mb()
    append_benchmark_record(BENCHMARK_RESULTS_DIR / "int4_store_build.json", record)
    print(json.dumps(record["summary"], indent=1))
    if out["bad"]:
        raise SystemExit(f"{len(out['bad'])} record(s) failed sha256 verification")


if __name__ == "__main__":
    main()
