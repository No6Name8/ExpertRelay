"""Build an int4 expert store next to an int8 store, differing only in the
routed experts.

    python -m expertrelay.store.build_int4_store --group-size 128
    python -m expertrelay.store.build_int4_store --int8-store models/qwen1.5-moe-a2.7b-chat-int8 --group-size 128 --quantizer rtn_gptq_grid
    python -m expertrelay.store.build_int4_store --int8-store models/qwen1.5-moe-a2.7b-chat-int8 --group-size 128 --quantizer gptq

Where the routed experts come from (store.int4 describes each quantizer):
  rtn_absmax7 / rtn_gptq_grid  re-quantized from the ORIGINAL bf16
      checkpoint (store.download_checkpoint), never from the int8 store.
  gptq  converted losslessly from a GPTQ release (store.gptq) at a pinned
      revision; the bf16 checkpoint is read too, only to report each
      expert's weight error against it.
Everything resident (attention, shared expert, router, norms, embedding,
lm_head) is the int8 store's, unchanged: its resident.bin,
resident_index.json and tokenizer files are hard-linked (copied if the
volume can't link), so the two stores differ ONLY in the routed experts.

Records keep the int8 store's rules: one 4096-aligned record per expert at
slot (layer * experts + expert), loadable with one read, sha256 per record
in experts_index.json, with the error against bf16 per matrix. After
writing, every record is read back and its sha256 checked.

Resumable: each finished layer's records are fsynced, then journaled
(build_journal.jsonl in the output directory, first line = the build's
settings). Rerunning the same command re-verifies every journaled record's
sha256 on disk, keeps the ones that match and writes the rest. A journal
from different settings stops the build instead of mixing two stores.

Writes models/<int8 store name with int8 -> int4g{G}[-rtn|-gptq]>/ and
appends a build record (sizes, error statistics next to the int8 store's)
to benchmarks/results/int4_store_build.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from expertrelay import REPO_ROOT
from expertrelay.benchmarking import append_benchmark_record, base_record, peak_process_rss_mb
from expertrelay.manager.profile import collect_machine_profile
from expertrelay.memory_budget import enforce_ram_budget
from expertrelay.paths import BENCHMARK_RESULTS_DIR, MODELS_ROOT
from expertrelay.store.download_checkpoint import checkpoint_dir
from expertrelay.store.expert_reader import EXPERTS_BIN, EXPERTS_INDEX
from expertrelay.store.gptq import GPTQ_TENSORS, gptq_to_int4
from expertrelay.store.int4 import (
    QUANTIZER_GPTQ,
    QUANTIZER_RTN_ABSMAX7,
    QUANTIZER_RTN_GPTQ_GRID,
    QUANTIZERS,
    Int4RecordLayout,
    pack_int4,
    quant_error,
    quantize_groupwise_int4,
    quantize_groupwise_int4_gptq_grid,
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
JOURNAL = "build_journal.jsonl"
MAX_RAM_GB = 0.5  # one expert's three f32 matrices plus its record, with room
GPTQ_REPO = "Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4"
GPTQ_REVISION = "81b132adfae58e03b96ae1ed1f0d578d0cc4d09a"
STORE_SUFFIX = {QUANTIZER_RTN_ABSMAX7: "", QUANTIZER_RTN_GPTQ_GRID: "-rtn", QUANTIZER_GPTQ: "-gptq"}
METHOD = {
    QUANTIZER_RTN_ABSMAX7: "int4 symmetric round-to-nearest, values -7..7, scale absmax/7 (store.int4)",
    QUANTIZER_RTN_GPTQ_GRID: "int4 round-to-nearest on GPTQ's symmetric grid, values -8..7, no error "
    "compensation (store.int4)",
    QUANTIZER_GPTQ: "GPTQ codes and scales from the release, re-packed without changing a value (store.gptq)",
}
# the architecture keys a GPTQ release must share with the int8 store's model
ARCH_KEYS = ("num_hidden_layers", "num_experts", "hidden_size", "moe_intermediate_size", "vocab_size")


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


def check_gptq_config(gptq_dir: Path, config: dict, group: int) -> dict:
    """The release's quantization_config, after checking it is the kind store.gptq converts exactly."""
    gptq_config = json.loads((gptq_dir / "config.json").read_text())
    q = gptq_config["quantization_config"]
    expected = {"quant_method": "gptq", "bits": 4, "group_size": group, "sym": True, "desc_act": False}
    wrong = {k: q.get(k) for k, v in expected.items() if q.get(k) != v}
    wrong.update({k: gptq_config.get(k) for k in ARCH_KEYS if gptq_config.get(k) != config[k]})
    if q.get("checkpoint_format", "gptq") != "gptq":
        wrong["checkpoint_format"] = q["checkpoint_format"]
    if wrong:
        raise SystemExit(f"GPTQ release doesn't match what store.gptq converts: {wrong}")
    return q


class ExpertQuantizer:
    """Produces one expert's quantized matrices and their error against bf16."""

    def __init__(self, quantizer: str, group: int, bf16_dir: Path, gptq_dir: Path | None):
        self.quantizer, self.group = quantizer, group
        self.bf16 = LocalCheckpoint(bf16_dir)
        self.gptq = LocalCheckpoint(gptq_dir) if gptq_dir else None

    def matrix(self, layer: int, expert: int, name: str) -> tuple[np.ndarray, np.ndarray]:
        if self.quantizer == QUANTIZER_GPTQ:
            prefix = f"model.layers.{layer}.mlp.experts.{expert}.{name}."
            t = {k: self.gptq.get(prefix + k) for k in GPTQ_TENSORS}
            return gptq_to_int4(t["qweight"], t["qzeros"], t["scales"], t["g_idx"], t["bias"], self.group)
        w = self.bf16.get(expert_tensor_name(layer, expert, name))
        quantize = (
            quantize_groupwise_int4_gptq_grid
            if self.quantizer == QUANTIZER_RTN_GPTQ_GRID
            else quantize_groupwise_int4
        )
        q, scales = quantize(w, self.group)
        return pack_int4(q), scales

    def expert(self, layout: Int4RecordLayout, layer: int, expert: int) -> tuple[bytearray, dict]:
        quantized, errors = {}, {}
        sq_err = sq_norm = 0.0
        for name, _shape in layout.matrices:
            packed, scales = self.matrix(layer, expert, name)
            quantized[name] = (packed, scales)
            err = quant_error(
                self.bf16.get(expert_tensor_name(layer, expert, name)), packed, scales, self.group
            )
            sq_err, sq_norm = sq_err + err.pop("sq_error"), sq_norm + err.pop("sq_norm")
            errors[name] = err
        errors["expert"] = {"rel_fro_error": (sq_err / sq_norm) ** 0.5 if sq_norm else 0.0}
        return serialize_int4_record(layout, quantized), errors


def resume_entries(
    out_dir: Path, settings: dict, total_bytes: int
) -> dict[tuple[int, int], ExpertIndexEntry]:
    """Journaled records whose bytes on disk still match their sha256."""
    journal, data = out_dir / JOURNAL, out_dir / EXPERTS_BIN
    if not journal.exists():
        return {}
    lines = journal.read_text().splitlines()
    if not lines or json.loads(lines[0]) != settings:
        raise SystemExit(f"{journal} is from a different build; move it and {EXPERTS_BIN} away to start over")
    if not data.exists() or data.stat().st_size != total_bytes:
        return {}
    kept = {}
    for line in lines[1:]:
        try:
            e = ExpertIndexEntry(**json.loads(line))
        except (json.JSONDecodeError, TypeError):  # a line torn by a crash
            continue
        if sha256_file_region(data, e.offset, e.size) == e.sha256:
            kept[(e.layer, e.expert)] = e
    return kept


def build(
    int8_store: Path,
    bf16_dir: Path,
    out_dir: Path,
    group: int,
    quantizer: str = QUANTIZER_RTN_ABSMAX7,
    gptq_dir: Path | None = None,
) -> dict:
    manifest = json.loads((int8_store / "store.json").read_text())
    config = manifest["source"]["config"]
    layers, experts = config["num_hidden_layers"], config["num_experts"]
    layout = Int4RecordLayout.from_config(config, group, quantizer)
    gptq_config = check_gptq_config(gptq_dir, config, group) if quantizer == QUANTIZER_GPTQ else None
    source = ExpertQuantizer(quantizer, group, bf16_dir, gptq_dir if quantizer == QUANTIZER_GPTQ else None)
    settings = {
        "layout": layout.to_dict(),
        "bf16": _display(bf16_dir),
        "gptq": _display(gptq_dir) if quantizer == QUANTIZER_GPTQ else None,
    }
    total_bytes = layers * experts * layout.record_size
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    done = resume_entries(out_dir, settings, total_bytes)
    resume_s = time.perf_counter() - t0
    resumed = len(done)
    if not done:
        with open(out_dir / EXPERTS_BIN, "wb") as f:
            f.truncate(total_bytes)
        (out_dir / JOURNAL).write_text(json.dumps(settings) + "\n")
    print(f"{len(done)} of {layers * experts} records already written and verified", flush=True)

    t0 = time.perf_counter()
    with open(out_dir / EXPERTS_BIN, "r+b") as f, open(out_dir / JOURNAL, "a") as journal:
        for layer in range(layers):
            new = []
            for e in range(experts):
                if (layer, e) in done:
                    continue
                record, errors = source.expert(layout, layer, e)
                offset = expert_offset(layer, e, experts, layout.record_size)
                f.seek(offset)
                f.write(record)
                new.append(
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
            if new:
                f.flush()
                os.fsync(f.fileno())
                journal.writelines(json.dumps(asdict(x)) + "\n" for x in new)
                journal.flush()
                os.fsync(journal.fileno())
                done.update({(x.layer, x.expert): x for x in new})
                print(f"layer {layer + 1}/{layers} ({time.perf_counter() - t0:.0f} s)", flush=True)
    entries = sorted(done.values(), key=lambda x: x.offset)
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
            "method": METHOD[quantizer],
            "quantizer": quantizer,
            "group_size": group,
            "from": f"GPTQ release {_display(gptq_dir)}"
            if quantizer == QUANTIZER_GPTQ
            else f"bf16 checkpoint {_display(bf16_dir)}",
            "error_measured_against": f"bf16 checkpoint {_display(bf16_dir)}",
            **({"gptq_quantization_config": gptq_config} if gptq_config else {}),
        },
        "resident_from": {"store": int8_store.name, "files": linked},
    }
    (out_dir / "store.json").write_text(json.dumps(store, indent=1))
    return {
        "entries": entries,
        "layout": layout,
        "bad": bad,
        "resumed_records": resumed,
        "build_s": build_s,
        "verify_s": verify_s,
        "resume_check_s": resume_s,
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
    ap.add_argument("--quantizer", choices=QUANTIZERS, default=QUANTIZER_RTN_ABSMAX7)
    ap.add_argument("--int8-store", type=Path, default=MODELS_ROOT / "qwen1.5-moe-a2.7b-int8")
    ap.add_argument("--bf16-dir", type=Path, default=None, help="default: the int8 store's source checkpoint")
    ap.add_argument("--gptq-dir", type=Path, default=checkpoint_dir(GPTQ_REPO, GPTQ_REVISION))
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    enforce_ram_budget(200 * 2**20, MAX_RAM_GB, "int4 store build (one expert at a time)")
    int8_store = args.int8_store.resolve()
    source = json.loads((int8_store / "store.json").read_text())["source"]
    bf16_dir = (args.bf16_dir or checkpoint_dir(source["repo_id"], source["revision"])).resolve()
    gptq_dir = args.gptq_dir.resolve() if args.quantizer == QUANTIZER_GPTQ else None
    name = int8_store.name.replace("int8", f"int4g{args.group_size}") + STORE_SUFFIX[args.quantizer]
    out_dir = (args.out_dir or int8_store.with_name(name)).resolve()
    record = base_record(
        label=f"int4 expert store build ({args.quantizer}, group size {args.group_size})",
        seed=0,
        model={"source_repo": source["repo_id"], "revision": source["revision"], "store": out_dir.name},
        config={
            "group_size": args.group_size,
            "quantizer": args.quantizer,
            "bf16_checkpoint": _display(bf16_dir),
            "gptq_checkpoint": _display(gptq_dir) if gptq_dir else None,
            "resident_from": int8_store.name,
        },
        machine=collect_machine_profile(measure_disk=False),
    )
    out = build(int8_store, bf16_dir, out_dir, args.group_size, args.quantizer, gptq_dir)
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
        "resume_check_seconds": out["resume_check_s"],
        "records_resumed": out["resumed_records"],
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
