"""Build the int8 expert store for the full Qwen1.5-MoE-A2.7B.

    python -m expertrelay.store.build_store            # build, or resume a partial build
    python -m expertrelay.store.build_store --verify   # re-check every sha256 of a finished store

What it does:
  1. Pins the Hugging Face revision to a commit hash, recorded in store.json.
     An existing store stays pinned to its own hash. Asking for a different
     revision is an error, never a silent mix of two checkpoints.
  2. Streams tensors with range requests (store.fetch_hf_tensors). Experts
     are fetched one matrix at a time. Big resident tensors (embedding,
     lm_head) are fetched in row blocks of at most MAX_BLOCK_ELEMENTS. The
     full model is never in RAM, and the RAM budget is enforced before the
     build starts (memory_budget).
  3. Quantizes to int8 (store.quantize), writes every expert as one aligned
     record and every resident tensor as one aligned region (store.layout).
  4. Verifies as it goes: each expert record is parsed back from its
     serialized bytes, dequantized, and compared with the original bf16
     values. Error stats are recorded per matrix and per expert.
  5. Resumes: every finished item is appended to build_journal.jsonl only
     after its bytes are fsync'ed. On rerun, journaled items are re-hashed
     from disk. Anything missing or mismatched is rebuilt; everything else
     is skipped.
  6. Finishes by writing the two index files, re-reading every expert with
     the real one-read unbuffered loader and checking its sha256, and
     saving a summary to store.json and
     benchmarks/results/expert_store_build_<host>.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import statistics
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path

from expertrelay import REPO_ROOT
from expertrelay.benchmarking import append_benchmark_record, base_record, peak_process_rss_mb
from expertrelay.manager.profile import collect_machine_profile
from expertrelay.memory_budget import enforce_ram_budget
from expertrelay.paths import BENCHMARK_RESULTS_DIR, DEFAULT_EXPERT_STORE_DIR
from expertrelay.store.expert_reader import EXPERTS_BIN, EXPERTS_INDEX, ExpertStoreReader
from expertrelay.store.fetch_hf_tensors import (
    TensorLocation,
    fetch_config,
    fetch_index,
    fetch_rows,
    locate_tensors,
    resolve_revision,
)
from expertrelay.store.layout import (
    EXPERT_MATRICES,
    KIND_INT8,
    ExpertIndexEntry,
    ExpertRecordLayout,
    ResidentEntry,
    expert_offset,
    expert_tensor_name,
    parse_expert_record,
    parse_expert_tensor_name,
    plan_resident_layout,
    read_resident_index,
    serialize_expert_record,
    sha256_file_region,
    write_expert_index,
    write_resident_index,
)
from expertrelay.store.quantize import ErrorAccumulator, quantize_rowwise_int8

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

DEFAULT_REPO = "Qwen/Qwen1.5-MoE-A2.7B"
STORE_FORMAT_VERSION = 1
MANIFEST = "store.json"
RESIDENT_BIN = "resident.bin"
RESIDENT_INDEX = "resident_index.json"
JOURNAL = "build_journal.jsonl"
EXPERT_STORE_DOC = REPO_ROOT / "docs" / "expert-store.md"

# Largest block of a resident tensor fetched and quantized at once: 4M
# elements = 8 MB of bf16 on the wire, 16 MB as fp32. Expert matrices
# (2.9M elements) always fit in one block.
MAX_BLOCK_ELEMENTS = 4 * 2**20

# RAM estimate used to enforce --max-ram-gb before starting.
# Per worker: one expert's three matrices as fp32 (35 MB), their int8 copies
# and the packed record (17 MB), and the float64 temporaries of the error
# check on one matrix (~70 MB). About 125 MB; rounded up to 200 MB for
# allocator slack. Baseline covers the interpreter, numpy and the parsed
# shard headers.
PER_WORKER_BYTES = 200 * 2**20
BASELINE_BYTES = 300 * 2**20

QUANTIZATION_METHOD = {
    "scheme": "weight-only int8, symmetric, per-output-channel (per-row) absmax",
    "formula": "scale_r = max|W[r,:]| / 127; Q = clip(round_half_even(W / scale_r), -127, 127)",
    "scales_dtype": "float32",
    "applies_to": "routed experts and every 2-D resident matrix except the router and shared_expert_gate",
    "kept_fp32": "norms, biases, mlp.gate (router), mlp.shared_expert_gate",
    "references": [
        "Krishnamoorthi 2018, arXiv:1806.08342",
        "Dettmers et al. 2022 (LLM.int8()), arXiv:2208.07339",
    ],
}


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


@dataclass
class Plan:
    repo_id: str
    revision: str
    config: dict
    num_layers: int
    num_experts: int
    layout: ExpertRecordLayout
    resident: list[ResidentEntry]
    locations: dict[str, TensorLocation]

    @property
    def experts_bin_bytes(self) -> int:
        return self.num_layers * self.num_experts * self.layout.record_size

    @property
    def resident_bin_bytes(self) -> int:
        last = self.resident[-1]
        return last.offset + last.region_size


def _resident_order_key(name: str) -> tuple:
    """embeddings, then layers in numeric (not string) order, then final norm and lm_head."""
    if name.startswith("model.embed_tokens"):
        return (0, 0, name)
    if name.startswith("model.layers."):
        return (1, int(name.split(".")[2]), name)
    return (2, 0, name)


def make_plan(repo_id: str, revision: str) -> Plan:
    config = fetch_config(repo_id, revision)
    index = fetch_index(repo_id, revision)
    names = list(index["weight_map"])
    locations = locate_tensors(repo_id, names, revision, index)

    num_layers, num_experts = config["num_hidden_layers"], config["num_experts"]
    layout = ExpertRecordLayout.from_config(config)
    expert_names = [n for n in names if parse_expert_tensor_name(n)]
    if len(expert_names) != num_layers * num_experts * len(EXPERT_MATRICES):
        raise ValueError(
            f"expected {num_layers * num_experts * 3} expert tensors, checkpoint has {len(expert_names)}"
        )
    for n in expert_names:
        _layer, _expert, matrix = parse_expert_tensor_name(n)
        if locations[n].shape != layout.shapes[matrix]:
            raise ValueError(f"{n}: shape {locations[n].shape} != layout {layout.shapes[matrix]}")

    resident_names = sorted((n for n in names if not parse_expert_tensor_name(n)), key=_resident_order_key)
    resident = plan_resident_layout([(n, locations[n].shape) for n in resident_names])
    return Plan(repo_id, revision, config, num_layers, num_experts, layout, resident, locations)


# ---------------------------------------------------------------------------
# Journal (resumability)
# ---------------------------------------------------------------------------


def append_journal(path: Path, record: dict) -> None:
    """Append one completed item. Callers write and fsync the data FIRST, so a
    journal line always refers to bytes that are already on disk."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_journal(path: Path) -> list[dict]:
    """All journal records. A malformed LAST line is a torn write from a crash
    and is dropped; that item just gets rebuilt. A malformed line anywhere
    else means real corruption and raises."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    records = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                print(f"journal: dropping torn last line ({len(line)} chars)")
                continue
            raise ValueError(f"{path}: corrupt journal line {i + 1}") from None
    return records


def verified_journal(store_dir: Path, records: list[dict]) -> tuple[dict, dict, int]:
    """Journal records whose bytes on disk still hash correctly. Returns
    ({(layer, expert): rec}, {name: rec}, number dropped). Later records win,
    so a rebuilt item supersedes its earlier entry."""
    experts, resident, dropped = {}, {}, 0
    for rec in records:
        if rec["kind"] == "expert":
            experts[(rec["layer"], rec["expert"])] = rec
        else:
            resident[rec["name"]] = rec
    for key, rec in list(experts.items()):
        if sha256_file_region(store_dir / EXPERTS_BIN, rec["offset"], rec["size"]) != rec["sha256"]:
            print(f"resume: expert {key} fails its sha256 on disk, will rebuild")
            del experts[key]
            dropped += 1
    for name, rec in list(resident.items()):
        if sha256_file_region(store_dir / RESIDENT_BIN, rec["offset"], rec["region_size"]) != rec["sha256"]:
            print(f"resume: resident {name} fails its sha256 on disk, will rebuild")
            del resident[name]
            dropped += 1
    return experts, resident, dropped


# ---------------------------------------------------------------------------
# Work items
# ---------------------------------------------------------------------------


def _write_at(path: Path, offset: int, chunks: list[tuple[int, bytes]]) -> None:
    """Write (relative offset, bytes) pieces at `offset` and fsync before returning."""
    with open(path, "r+b") as f:
        for rel, data in chunks:
            f.seek(offset + rel)
            f.write(data)
        f.flush()
        os.fsync(f.fileno())


def build_expert(plan: Plan, store_dir: Path, layer: int, expert: int) -> dict:
    layout = plan.layout
    originals, quantized = {}, {}
    for m in EXPERT_MATRICES:
        originals[m] = fetch_rows(plan.locations[expert_tensor_name(layer, expert, m)])
        quantized[m] = quantize_rowwise_int8(originals[m])
    record = serialize_expert_record(layout, quantized)
    del quantized

    # Error is measured on what was actually serialized: parse the record
    # back, dequantize, compare with the original bf16 (exactly upcast to fp32).
    parsed = parse_expert_record(layout, record)
    total, per_matrix = ErrorAccumulator(), {}
    for m in EXPERT_MATRICES:
        acc = ErrorAccumulator()
        acc.add(originals[m], *parsed[m])
        per_matrix[m] = acc.summary()
        total.merge(acc)
    del originals, parsed

    offset = expert_offset(layer, expert, plan.num_experts, layout.record_size)
    _write_at(store_dir / EXPERTS_BIN, offset, [(0, bytes(record))])
    return {
        "kind": "expert",
        "layer": layer,
        "expert": expert,
        "offset": offset,
        "size": layout.record_size,
        "shapes": {m: list(s) for m, s in layout.matrices},
        "sha256": hashlib.sha256(record).hexdigest(),
        "quant_error": {"expert": total.summary(), **per_matrix},
        "source_bytes": sum(
            plan.locations[expert_tensor_name(layer, expert, m)].nbytes for m in EXPERT_MATRICES
        ),
    }


def build_resident(plan: Plan, store_dir: Path, entry: ResidentEntry) -> dict:
    loc = plan.locations[entry.name]
    path = store_dir / RESIDENT_BIN
    quant_error = None
    if entry.kind == KIND_INT8:
        rows, cols = entry.shape
        rows_per_block = max(1, MAX_BLOCK_ELEMENTS // cols)
        acc = ErrorAccumulator()
        for r0 in range(0, rows, rows_per_block):
            r1 = min(rows, r0 + rows_per_block)
            w = fetch_rows(loc, r0, r1)
            q, scales = quantize_rowwise_int8(w)
            acc.add(w, q, scales)
            _write_at(
                path,
                0,
                [
                    (entry.offset + r0 * cols, q.tobytes()),
                    (entry.scales_offset + r0 * 4, scales.astype("<f4").tobytes()),
                ],
            )
        quant_error = acc.summary()
    else:
        data = fetch_rows(loc).astype("<f4").tobytes()
        _write_at(path, entry.offset, [(0, data)])

    return {
        "kind": "resident",
        **asdict(entry),
        "sha256": sha256_file_region(path, entry.offset, entry.region_size),
        "quant_error": quant_error,
        "source_bytes": loc.nbytes,
    }


# ---------------------------------------------------------------------------
# Manifest, finalize, verify
# ---------------------------------------------------------------------------


def _write_json_atomic(path: Path, doc: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    os.replace(tmp, path)


def _preallocate(path: Path, size: int) -> None:
    """Create/extend a data file to its planned size. Unwritten bytes read as
    zeros, which is exactly what record padding must contain."""
    path.touch(exist_ok=True)
    if path.stat().st_size != size:
        with open(path, "r+b") as f:
            f.truncate(size)


def expert_error_stats(expert_records: list[dict]) -> dict:
    rel = [r["quant_error"]["expert"]["rel_fro_error"] for r in expert_records]
    per_matrix = {
        m: {
            "mean_rel_fro_error": statistics.fmean(
                r["quant_error"][m]["rel_fro_error"] for r in expert_records
            ),
            "max_rel_fro_error": max(r["quant_error"][m]["rel_fro_error"] for r in expert_records),
        }
        for m in EXPERT_MATRICES
    }
    worst = max(expert_records, key=lambda r: r["quant_error"]["expert"]["rel_fro_error"])
    return {
        "experts": len(rel),
        "mean_rel_fro_error": statistics.fmean(rel),
        "median_rel_fro_error": statistics.median(rel),
        "max_rel_fro_error": max(rel),
        "worst_expert": {"layer": worst["layer"], "expert": worst["expert"]},
        "max_abs_error_over_scale": max(
            r["quant_error"]["expert"]["max_abs_error_over_scale"] for r in expert_records
        ),
        "per_matrix": per_matrix,
    }


def verify_store(store_dir: Path) -> dict:
    """Re-read every expert with the real loader (one unbuffered read each)
    and every resident region, and check each sha256 against its index."""
    with ExpertStoreReader(store_dir) as reader:
        entries = reader.entries()
        bad_experts = [(e.layer, e.expert) for e in entries if not reader.verify(e.layer, e.expert)]
    resident = read_resident_index(store_dir / RESIDENT_INDEX)
    bad_resident = [
        e.name
        for e in resident
        if sha256_file_region(store_dir / RESIDENT_BIN, e.offset, e.region_size) != e.sha256
    ]
    return {
        "experts_checked": len(entries),
        "experts_bad": bad_experts,
        "resident_checked": len(resident),
        "resident_bad": bad_resident,
        "ok": not bad_experts and not bad_resident,
    }


def finalize(plan: Plan, store_dir: Path, manifest: dict, experts: dict, resident: dict) -> dict:
    expert_records = [experts[k] for k in sorted(experts)]
    write_expert_index(
        store_dir / EXPERTS_INDEX,
        plan.layout,
        [
            ExpertIndexEntry(
                layer=r["layer"],
                expert=r["expert"],
                offset=r["offset"],
                size=r["size"],
                shapes=r["shapes"],
                sha256=r["sha256"],
                quant_error=r["quant_error"],
            )
            for r in expert_records
        ],
    )
    fields = ResidentEntry.__dataclass_fields__
    write_resident_index(
        store_dir / RESIDENT_INDEX,
        [ResidentEntry(**{k: resident[e.name][k] for k in fields}) for e in plan.resident],
    )

    print("verifying every sha256 (experts via one unbuffered read each) ...")
    verification = verify_store(store_dir)
    if not verification["ok"]:
        raise RuntimeError(f"store verification failed: {verification}")

    resident_int8 = [resident[e.name] for e in plan.resident if e.kind == KIND_INT8]
    summary = {
        "sizes": {
            "experts_bin_bytes": plan.experts_bin_bytes,
            "resident_bin_bytes": plan.resident_bin_bytes,
            "total_bytes": plan.experts_bin_bytes + plan.resident_bin_bytes,
            "expert_record_bytes": plan.layout.record_size,
            "expert_weights_bytes": plan.layout.weights_bytes,
            "expert_scales_bytes": plan.layout.scales_bytes,
            "source_checkpoint_bytes": sum(loc.nbytes for loc in plan.locations.values()),
        },
        "expert_quant_error": expert_error_stats(expert_records),
        "resident_int8_quant_error": {
            "tensors": len(resident_int8),
            "max_rel_fro_error": max(r["quant_error"]["rel_fro_error"] for r in resident_int8),
            "mean_rel_fro_error": statistics.fmean(r["quant_error"]["rel_fro_error"] for r in resident_int8),
            "max_abs_error_over_scale": max(
                r["quant_error"]["max_abs_error_over_scale"] for r in resident_int8
            ),
        },
        "verification": verification,
        "peak_rss_mb_max_over_runs": max(
            run["peak_rss_mb"] for run in manifest["runs"] if "peak_rss_mb" in run
        ),
    }
    manifest["complete"] = True
    manifest["summary"] = summary
    _write_json_atomic(store_dir / MANIFEST, manifest)
    return summary


# ---------------------------------------------------------------------------
# Main build loop
# ---------------------------------------------------------------------------


def _fmt_eta(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    return f"{h}h{rem // 60:02d}m"


def build(store_dir: Path, repo_id: str, revision: str | None, workers: int, max_ram_gb: float) -> int:
    enforce_ram_budget(
        BASELINE_BYTES + workers * PER_WORKER_BYTES, max_ram_gb, f"building with {workers} workers"
    )
    store_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = store_dir / MANIFEST

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        pinned = manifest["source"]["revision"]
        if revision is not None and resolve_revision(repo_id, revision) != pinned:
            raise SystemExit(f"store is pinned to {pinned}; refusing to mix in revision {revision!r}")
        print(f"resuming store pinned to {repo_id}@{pinned}")
    else:
        pinned = resolve_revision(repo_id, revision or "main")
        print(f"new store: pinned {repo_id}@{revision or 'main'} -> {pinned}")
        manifest = None

    plan = make_plan(repo_id, pinned)
    if manifest is not None and manifest["expert_layout"] != plan.layout.to_dict():
        raise SystemExit("store was built with a different record layout than this code produces; rebuild it")
    if manifest is None:
        manifest = {
            "format_version": STORE_FORMAT_VERSION,
            "source": {"repo_id": repo_id, "revision": pinned, "config": plan.config},
            "quantization": QUANTIZATION_METHOD,
            "expert_layout": plan.layout.to_dict(),
            "num_layers": plan.num_layers,
            "num_experts": plan.num_experts,
            "files": {
                "experts": EXPERTS_BIN,
                "experts_index": EXPERTS_INDEX,
                "resident": RESIDENT_BIN,
                "resident_index": RESIDENT_INDEX,
                "journal": JOURNAL,
            },
            "complete": False,
            "summary": None,
            "runs": [],
        }
    run = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": socket.gethostname()}
    manifest["runs"].append(run)
    _write_json_atomic(manifest_path, manifest)

    _preallocate(store_dir / EXPERTS_BIN, plan.experts_bin_bytes)
    _preallocate(store_dir / RESIDENT_BIN, plan.resident_bin_bytes)

    done_experts, done_resident, dropped = verified_journal(store_dir, read_journal(store_dir / JOURNAL))
    todo: list[tuple[str, object]] = [("resident", e) for e in plan.resident if e.name not in done_resident]
    todo += [
        ("expert", (layer, expert))
        for layer in range(plan.num_layers)
        for expert in range(plan.num_experts)
        if (layer, expert) not in done_experts
    ]
    total_items = len(plan.resident) + plan.num_layers * plan.num_experts
    todo_bytes = sum(
        plan.locations[item.name].nbytes
        if kind == "resident"
        else sum(plan.locations[expert_tensor_name(*item, m)].nbytes for m in EXPERT_MATRICES)
        for kind, item in todo
    )
    print(
        f"{total_items - len(todo)}/{total_items} items already done and verified ({dropped} dropped); "
        f"{len(todo)} to go, {todo_bytes / 1e9:.2f} GB to download"
    )

    downloaded, completed, failures = 0, 0, []
    started = time.perf_counter()

    def submit(ex: ThreadPoolExecutor, kind: str, item) -> Future:
        if kind == "resident":
            return ex.submit(build_resident, plan, store_dir, item)
        return ex.submit(build_expert, plan, store_dir, *item)

    # Bounded in-flight window: submitting all ~1,800 tasks at once would be
    # harmless for RAM (tasks allocate only when running) but would make
    # Ctrl-C wait for the whole queue.
    with ThreadPoolExecutor(max_workers=workers) as ex:
        queue = list(todo)
        in_flight: dict[Future, tuple[str, object]] = {}
        while queue or in_flight:
            while queue and len(in_flight) < workers * 2:
                kind, item = queue.pop(0)
                in_flight[submit(ex, kind, item)] = (kind, item)
            finished, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for fut in finished:
                kind, item = in_flight.pop(fut)
                label = item.name if kind == "resident" else f"expert L{item[0]} E{item[1]}"
                try:
                    rec = fut.result()
                except Exception as e:  # keep going; a rerun resumes the failed item
                    failures.append(label)
                    print(f"FAILED {label}: {e!r}")
                    continue
                append_journal(store_dir / JOURNAL, rec)
                downloaded += rec["source_bytes"]
                completed += 1
                elapsed = time.perf_counter() - started
                rate = downloaded / elapsed if elapsed else 0.0
                eta = (todo_bytes - downloaded) / rate if rate else float("inf")
                print(
                    f"[{total_items - len(todo) + completed}/{total_items}] {label} | "
                    f"{downloaded / 1e9:.2f}/{todo_bytes / 1e9:.2f} GB | {rate / 1e6:.2f} MB/s | "
                    f"ETA {_fmt_eta(eta) if rate else '?'} | peak RSS {peak_process_rss_mb():.0f} MB"
                )

    run.update(
        finished=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        items_built=completed,
        bytes_downloaded=downloaded,
        failures=failures,
        peak_rss_mb=peak_process_rss_mb(),
    )
    _write_json_atomic(manifest_path, manifest)
    if failures:
        print(f"{len(failures)} item(s) failed; rerun the same command to resume.")
        return 1

    done_experts, done_resident, _ = verified_journal(store_dir, read_journal(store_dir / JOURNAL))
    summary = finalize(plan, store_dir, manifest, done_experts, done_resident)
    _save_build_record(plan, workers, summary)
    print(json.dumps(summary, indent=1))
    return 0


def _save_build_record(plan: Plan, workers: int, summary: dict) -> None:
    record = base_record(
        label="expert store build",
        seed=0,
        model={"source_repo": plan.repo_id, "revision": plan.revision},
        config={
            "workers": workers,
            "max_block_elements": MAX_BLOCK_ELEMENTS,
            "quantization": QUANTIZATION_METHOD,
        },
        machine=collect_machine_profile(measure_disk=False),
        summary=summary,
    )
    append_benchmark_record(BENCHMARK_RESULTS_DIR / f"expert_store_build_{socket.gethostname()}.json", record)


def _gb(n: int) -> str:
    return f"{n / 1e9:.2f} GB ({n:,} bytes)"


def format_store_report(manifest: dict) -> str:
    """docs/expert-store.md, generated from a finished store's store.json.
    Never edited by hand."""
    s = manifest["summary"]
    if not manifest.get("complete") or s is None:
        raise ValueError("store is not complete; nothing to report")
    sizes, err, res, ver = (
        s["sizes"],
        s["expert_quant_error"],
        s["resident_int8_quant_error"],
        s["verification"],
    )
    lay = manifest["expert_layout"]
    src = manifest["source"]
    lines = [
        "# Expert store: Qwen1.5-MoE-A2.7B, int8",
        "",
        "Generated by `python -m expertrelay.store.build_store --report` from the store's `store.json`.",
        "Do not edit by hand. Machine-readable copy: `benchmarks/results/expert_store_build_<host>.json`.",
        "",
        f"Source: `{src['repo_id']}` pinned at commit `{src['revision']}`, "
        f"{manifest['num_layers']} layers x {manifest['num_experts']} routed experts.",
        "",
        "## Sizes",
        "",
        "| | |",
        "|---|---|",
        f"| Total store | {_gb(sizes['total_bytes'])} |",
        f"| `experts.bin` (all {err['experts']} routed experts) | {_gb(sizes['experts_bin_bytes'])} |",
        f"| `resident.bin` (used on every token) | {_gb(sizes['resident_bin_bytes'])} |",
        f"| One expert record (weights + scales + pad) | {sizes['expert_record_bytes']:,} bytes |",
        f"| of which int8 weights / f32 scales | {sizes['expert_weights_bytes']:,} / {sizes['expert_scales_bytes']:,} bytes |",
        f"| Source checkpoint (bf16) | {_gb(sizes['source_checkpoint_bytes'])} |",
        f"| Peak RAM during conversion (max over runs) | {s['peak_rss_mb_max_over_runs']:.0f} MB |",
        "",
        "## Expert record layout",
        "",
        "One record per expert, all the same size, back to back from offset 0, so every record starts on a",
        f"{lay['alignment']}-byte boundary. One unbuffered read of `record_size` bytes loads a whole expert:",
        "",
        "```",
        "[gate_proj int8][up_proj int8][down_proj int8][gate scales f32][up scales f32][down scales f32][zero pad]",
        "```",
        "",
        "| Matrix | Shape |",
        "|---|---|",
        *[f"| {m['name']} | {m['shape'][0]} x {m['shape'][1]} |" for m in lay["matrices"]],
        "",
        "## Quantization",
        "",
        f"{manifest['quantization']['scheme']}: `{manifest['quantization']['formula']}`.",
        f"Kept fp32: {manifest['quantization']['kept_fp32']}. References: "
        + "; ".join(manifest["quantization"]["references"])
        + ".",
        "",
        "Error is measured on the serialized bytes: each record is parsed back, dequantized, and compared",
        "with the original bf16 values. `rel_fro_error` = ||W - W_hat||_F / ||W||_F. "
        "`max_abs_error_over_scale` is the largest",
        "per-element error in units of the row's step size (rounding bounds it at 0.5).",
        "",
        "| Routed experts (all three matrices per expert) | |",
        "|---|---|",
        f"| Mean relative error | {err['mean_rel_fro_error']:.4%} |",
        f"| Median relative error | {err['median_rel_fro_error']:.4%} |",
        f"| Max relative error | {err['max_rel_fro_error']:.4%} "
        f"(layer {err['worst_expert']['layer']}, expert {err['worst_expert']['expert']}) |",
        f"| Max abs error / step | {err['max_abs_error_over_scale']:.4f} |",
        "",
        "| Per matrix | Mean rel. error | Max rel. error |",
        "|---|---|---|",
        *[
            f"| {m} | {v['mean_rel_fro_error']:.4%} | {v['max_rel_fro_error']:.4%} |"
            for m, v in err["per_matrix"].items()
        ],
        "",
        f"Resident int8 tensors ({res['tensors']}): mean relative error {res['mean_rel_fro_error']:.4%}, "
        f"max {res['max_rel_fro_error']:.4%}, max abs error / step {res['max_abs_error_over_scale']:.4f}.",
        "",
        "This is reconstruction error, not model quality. What int8 does to outputs vs. bf16 is a separate",
        "measurement that hasn't been done yet; see docs/limitations.md.",
        "",
        "## Verification",
        "",
        f"Every sha256 re-checked after the build: {ver['experts_checked']} experts, each via one unbuffered",
        f"read with the real loader, plus {ver['resident_checked']} resident tensors. Mismatches: "
        f"{len(ver['experts_bad']) + len(ver['resident_bad'])}.",
        "",
        "## Build runs",
        "",
        "| Started (UTC) | Finished | Items built | Downloaded | Peak RSS | Failures |",
        "|---|---|---|---|---|---|",
        *[
            f"| {r['started']} | {r.get('finished', 'interrupted')} | {r.get('items_built', '-')} | "
            f"{r['bytes_downloaded'] / 1e9:.2f} GB | {r['peak_rss_mb']:.0f} MB | {len(r.get('failures', []))} |"
            if "finished" in r
            else f"| {r['started']} | interrupted (hard kill) | - | - | - | - |"
            for r in manifest["runs"]
        ],
    ]
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store-dir", type=Path, default=DEFAULT_EXPERT_STORE_DIR)
    ap.add_argument("--repo-id", default=DEFAULT_REPO)
    ap.add_argument(
        "--revision",
        default=None,
        help="branch/tag/commit; new stores default to main, existing stores stay pinned",
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-ram-gb", type=float, default=1.5)
    ap.add_argument("--verify", action="store_true", help="only re-check every sha256 of a finished store")
    ap.add_argument(
        "--report", action="store_true", help="only (re)generate docs/expert-store.md from store.json"
    )
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.report:
        manifest = json.loads((args.store_dir / MANIFEST).read_text())
        EXPERT_STORE_DOC.write_text(format_store_report(manifest))
        print(f"wrote {EXPERT_STORE_DOC}")
        return
    if args.verify:
        result = verify_store(args.store_dir)
        print(json.dumps(result, indent=1))
        raise SystemExit(0 if result["ok"] else 1)
    raise SystemExit(build(args.store_dir, args.repo_id, args.revision, args.workers, args.max_ram_gb))


if __name__ == "__main__":
    main()
