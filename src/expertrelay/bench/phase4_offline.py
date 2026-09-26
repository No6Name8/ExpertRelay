"""Phase 4 offline check: does pinning every layer-0 expert pay off?

    python -m expertrelay.bench.phase4_offline                      # base store
    python -m expertrelay.bench.phase4_offline --report             # redo the doc section from saved JSON

Simulation only, from finished traces, replaying the TEST prompts (even
numbered in every category; bench/phase35_prediction.py's split). The
policy is the one the runtime implements (cache.expert_cache,
predictor.prefetch_policy), with its settings read from the cache config:

  * LRU cache, prefetching the top-k Fate-guessed experts for layers 1-23
    one layer ahead, skipping guesses whose calibrated probability is below
    prefetch_min_probability (calibrator from that config's file, fitted on
    the TUNING prompts);
  * no guess for layer 0: the runtime has none, since layer 0 has no
    previous layer;
  * with or without all layer-0 experts pinned, at the SAME total cache
    size, so pinning costs 60 slots the other layers lose.

Projected tokens/s are a PROJECTION (cache.predictive.overlapped_decode_seconds),
not a measurement: Phase 2's compute time spread evenly over the layers,
reads one at a time, a prefetch hidden behind the previous layer's compute.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from expertrelay import REPO_ROOT
from expertrelay.bench.phase3_analysis import doc_path_for, load_traces, phase2_speeds, upsert_doc_section
from expertrelay.bench.phase3_traces import round_robin, trace_dir_for
from expertrelay.bench.phase35_prediction import (
    belady_result,
    build_events,
    raw_ssd_read_s,
    split_halves,
    summarize,
    to_seq,
)
from expertrelay.benchmarking import append_benchmark_record, base_record, peak_process_rss_mb
from expertrelay.cache.predictive import Predict, simulate
from expertrelay.manager.profile import collect_machine_profile
from expertrelay.memory_budget import enforce_ram_budget
from expertrelay.paths import BENCHMARK_RESULTS_DIR
from expertrelay.predictor.offline import softmax
from expertrelay.predictor.prefetch_policy import load_calibrator
from expertrelay.runtime.generate import RuntimeConfig

DEFAULT_CACHE_CONFIG = REPO_ROOT / "configs" / "runtime_cache.json"
CAPACITIES = [144, 192, 288, 384, 480, 576]
MAX_RAM_GB = 1.0
RAM_PER_TRACE_BYTE = 7  # as bench/phase35_prediction.py: same arrays


def runtime_probs(seqs, calibrator) -> list[np.ndarray]:
    """[T_decode, L, E] per sequence: calibrated P(picked) of the Fate guess.
    Ranked by the raw probability, as the runtime ranks (the 1e-7 term only
    breaks the isotonic map's ties in that order). Layer 0 is unused."""
    out = []
    for s in seqs:
        raw = softmax(s.fate)
        out.append(calibrator.predict(raw) + np.float32(1e-7) * raw)
    return out


def analyze(seqs, rt: RuntimeConfig, record_bytes: int) -> dict:
    _, test = split_halves(seqs)
    num_layers, num_experts = seqs[0].used.shape[1], seqs[0].used.shape[2]
    calibrator, provenance = load_calibrator(rt.prefetch_calibration)
    events = [
        e
        for e in build_events(test, runtime_probs(test, calibrator))
        if not (isinstance(e, Predict) and e.layer == 0)
    ]
    speeds = phase2_speeds(record_bytes)
    raw_s, raw_src = raw_ssd_read_s()
    read_times = {"phase2_runtime": speeds["read_s_per_expert"], "raw_ssd": raw_s}
    per_layer = speeds["compute_s_per_token"] / num_layers
    pinned_keys = list(range(num_experts))  # layer 0: keys 0..E-1

    rows = {}
    for cap in CAPACITIES:
        row = {"belady": summarize(belady_result(events, num_experts, cap), per_layer, read_times)}
        for k in (0, rt.prefetch_k):
            for pin in (False, True):
                res = simulate(
                    events,
                    num_layers=num_layers,
                    num_experts=num_experts,
                    capacity=cap,
                    policy="lru",
                    prefetch_k=k,
                    pinned=pinned_keys if pin else [],
                    min_prob=rt.prefetch_min_probability,
                )
                v = summarize(res, per_layer, read_times)
                d0 = res.decode & (res.layer == 0)
                v["layer0_decode_hit_rate"] = float(res.hits[d0].sum() / res.accesses[d0].sum())
                row[f"lru+k{k}" + ("+pin0" if pin else "")] = v
        rows[str(cap)] = row
        print(f"  capacity {cap}: done", flush=True)
    return {
        "test_prompts": [s.prompt_id for s in test],
        "decode_tokens": int(sum(len(s.experts) - s.prompt_len for s in test)),
        "policy": {
            "cache": "lru",
            "prefetch_k": rt.prefetch_k,
            "prefetch_min_probability": rt.prefetch_min_probability,
            "layer0_guess": None,
            "calibration": provenance,
        },
        "capacities": CAPACITIES,
        "cache_ram_gb": {str(c): c * record_bytes / 1e9 for c in CAPACITIES},
        "pinned_ram_gb": num_experts * record_bytes / 1e9,
        "rows": rows,
        "projection": {
            "label": "PROJECTION from simulated reads + Phase 2 measured compute; not a measurement",
            "compute_s_per_layer": per_layer,
            "read_s_per_expert": read_times,
            "read_sources": {"phase2_runtime": speeds["source"], "raw_ssd": raw_src},
        },
    }


def _pct(x: float) -> str:
    return f"{x:.1%}"


def format_markdown(record: dict) -> str:
    a = record["analysis"]
    pol, ram, proj = a["policy"], a["cache_ram_gb"], a["projection"]
    caps = [str(c) for c in a["capacities"]]
    k = pol["prefetch_k"]
    configs = [
        ("lru+k0", "LRU, no prefetch"),
        ("lru+k0+pin0", "LRU, no prefetch, layer 0 pinned"),
        (f"lru+k{k}", f"LRU + prefetch top-{k}"),
        (f"lru+k{k}+pin0", f"LRU + prefetch top-{k}, layer 0 pinned"),
        ("belady", "Belady (optimal demand cache, needs the future)"),
    ]
    lines = [
        "## 8. Phase 4 offline check: pinning layer 0",
        "",
        f"Generated by `python -m expertrelay.bench.phase4_offline` at {record['timestamp']} "
        f"(commit {(record['git_commit'] or 'unknown')[:7]}) from "
        f"`benchmarks/results/phase4_offline_{record['model']['store']}.json`. Do not edit by hand.",
        "",
        f"The runtime's policy, simulated on the {len(a['test_prompts'])} test prompts "
        f"({a['decode_tokens']} generated tokens): LRU; prefetch the top-{k} Fate-guessed experts for "
        f"layers 1-23 one layer ahead, skipping guesses below a calibrated probability of "
        f"{pol['prefetch_min_probability']}; no guess for layer 0. Pinning all 60 layer-0 experts takes "
        f"{a['pinned_ram_gb']:.2f} GB OF the cache size shown, not on top of it.",
        "",
        "Decode hit rate (layer 0's own hit rate in brackets):",
        "",
        "| policy | " + " | ".join(f"{c} ({ram[c]:.2f} GB)" for c in caps) + " |",
        "|---|" + "---|" * len(caps),
    ]
    for key, label in configs:
        cells = []
        for c in caps:
            v = a["rows"][c][key]
            l0 = v.get("layer0_decode_hit_rate")
            cells.append(_pct(v["decode_hit_rate"]) + ("" if l0 is None else f" ({_pct(l0)})"))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Reads per generated token (demand + prefetch):",
        "",
        "| policy | " + " | ".join(caps) + " |",
        "|---|" + "---|" * len(caps),
    ]
    for key, label in configs:
        cells = [
            f"{a['rows'][c][key]['demand_reads_per_token']:.1f} + {a['rows'][c][key]['prefetch_reads_per_token']:.1f}"
            for c in caps
        ]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"**Projected decode tok/s: a projection, not a measurement.** {proj['label']}; same model as "
        "section 7 (compute spread evenly over layers, one read at a time, a prefetch hidden behind the "
        "previous layer's whole compute, which is optimistic). Read times: "
        + ", ".join(
            f"{name.replace('_', ' ')} {t * 1000:.2f} ms ({proj['read_sources'][name]})"
            for name, t in proj["read_s_per_expert"].items()
        )
        + ".",
        "",
        "| policy | read time | " + " | ".join(caps) + " |",
        "|---|---|" + "---|" * len(caps),
    ]
    for key, label in configs:
        for rt in proj["read_s_per_expert"]:
            cells = [f"{a['rows'][c][key]['projected_tok_s'][rt]:.2f}" for c in caps]
            lines.append(f"| {label} | {rt.replace('_', ' ')} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CACHE_CONFIG, help="runtime cache config")
    ap.add_argument("--report", action="store_true", help="only regenerate the doc section")
    args = ap.parse_args()

    rt = RuntimeConfig.load(args.config)
    store_dir = rt.store_dir.resolve()
    out = BENCHMARK_RESULTS_DIR / f"phase4_offline_{store_dir.name}.json"
    if args.report:
        upsert_doc_section(
            doc_path_for(store_dir.name), "phase4_offline", format_markdown(json.loads(out.read_text())[-1])
        )
        return
    if rt.prefetch_calibration is None:
        raise SystemExit(f"{args.config} names no prefetch_calibration file")

    trace_dir = trace_dir_for(store_dir)
    prompts_file = REPO_ROOT / "benchmarks" / "prompts" / "phase3.json"
    prompts = round_robin(json.loads(prompts_file.read_text(encoding="utf-8"))["prompts"])
    done = [p["id"] for p in prompts if (trace_dir / f"{p['id']}.done.json").exists()]
    enforce_ram_budget(
        sum((trace_dir / f"{pid}.ert").stat().st_size for pid in done) * RAM_PER_TRACE_BYTE,
        MAX_RAM_GB,
        "phase 4 offline check (traces + derived arrays)",
    )
    seqs = [to_seq(t) for t in load_traces(trace_dir, done)]
    manifest = json.loads((store_dir / "store.json").read_text())
    record = base_record(
        label="phase4 offline: pinning layer-0 experts (simulation)",
        seed=0,
        model={
            "store": store_dir.name,
            "repo_id": manifest["source"]["repo_id"],
            "revision": manifest["source"]["revision"],
        },
        config={
            "runtime_config": Path(args.config).resolve().relative_to(REPO_ROOT).as_posix(),
            "trace_dir": trace_dir.relative_to(REPO_ROOT).as_posix(),
            "capacities": CAPACITIES,
            "max_ram_gb": MAX_RAM_GB,
        },
        machine=collect_machine_profile(measure_disk=False),
    )
    record["analysis"] = analyze(seqs, rt, manifest["expert_layout"]["record_size"])
    record["peak_rss_mb"] = peak_process_rss_mb()
    append_benchmark_record(out, record)
    upsert_doc_section(doc_path_for(store_dir.name), "phase4_offline", format_markdown(record))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
