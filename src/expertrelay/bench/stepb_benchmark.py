"""Step B benchmark: faster compute (fused int8 kernel), pipelined prefill,
adaptive prefetching, and a log of every expert read.

    python -m expertrelay.bench.stepb_benchmark
    python -m expertrelay.bench.stepb_benchmark --report     # redo the doc from saved JSON

Phase 2 prompts and settings (base store, 32 new tokens, greedy), 1.25 GB
cache and 2 I/O threads where there is a cache. Every run logs every
expert read (runtime.weights.ReadLog). Setups:

  C_blocked        no cache, the original (blocked) int8 kernel
  cache_blocked    cache only, blocked kernel
  before           the Phase 4+5 best: prefetch top-8 + layer 0 pinned,
                   blocked kernel, fixed k, single calibration map, no
                   pipelined prefill: the code paths as they were
  C_fused          no cache, fused kernel
  cache_fused      cache only, fused kernel, pipelined prefill
  after_fixed      prefetch top-8 + layer 0 pinned, fused, pipelined
                   prefill, rank-aware calibration, fixed k, threshold 0.05
  after_adapt_005  as after_fixed, adaptive read budget, threshold 0.05
  after_adapt_01   as after_fixed, adaptive read budget, threshold 0.1

Each twice, the second round in reverse order. Tokens are compared with
the same kernel's no-cache run (the correctness rule: cache and prefetch
change no token), and blocked runs also with Phase 2's recorded setup C.
The fused kernel sums in a different order, so its tokens are compared
with the original model by bench/reference_check.py instead.

Writes benchmarks/results/stepb_benchmark.json and docs/stepb-benchmark.md.
"""

from __future__ import annotations

import argparse
import json
import statistics

from expertrelay import REPO_ROOT
from expertrelay.bench.phase2_baselines import run_one
from expertrelay.bench.phase4_benchmark import CACHE_CONFIG, cache_metrics, fit_cache, phase2_c_tokens
from expertrelay.benchmarking import append_benchmark_record, base_record
from expertrelay.manager.profile import collect_machine_profile
from expertrelay.paths import BENCHMARK_RESULTS_DIR
from expertrelay.runtime.generate import DEFAULT_RUNTIME_CONFIG, RuntimeConfig

CACHE_GB = 1.25
TIMEOUT_S = 3600
OUT = BENCHMARK_RESULTS_DIR / "stepb_benchmark.json"
DOC = REPO_ROOT / "docs" / "stepb-benchmark.md"
GAP_BINS_MS = [(0.0, 0.0, "another read in flight"), (0.0, 1.0, "0-1 ms"), (1.0, 5.0, "1-5 ms"),
               (5.0, 20.0, "5-20 ms"), (20.0, 50.0, "20-50 ms"), (50.0, 1e9, "> 50 ms")]  # fmt: skip


def setups() -> list[dict]:
    io = ["--io-threads", "2", "--log-reads"]
    blocked = ["--int8-kernel", "blocked"]
    fused = ["--int8-kernel", "fused"]
    old = [*blocked, "--no-prefill-pipelining", "--no-prefetch-adaptive", "--calibration-kind", "single"]
    new = [*fused, "--prefill-pipelining", "--calibration-kind", "rank"]
    pf8pin = ["--prefetch", "--prefetch-k", "8", "--pin-layer0", "--prefetch-min-probability"]

    def s(sid, name, source, args, kernel):
        return {"id": sid, "name": name, "source": source, "budget": True, "cache": source == "cached",
                "kernel": kernel, "args": args}  # fmt: skip

    return [
        s("C_blocked", "no cache, blocked kernel", "unbuffered", [*blocked, "--log-reads"], "blocked"),
        s(
            "cache_blocked",
            "cache only, blocked kernel",
            "cached",
            [*old, "--no-prefetch", "--no-pin-layer0", *io],
            "blocked",
        ),  # fmt: skip
        s(
            "before",
            "BEFORE: Phase 4+5 best (prefetch top-8 + pin layer 0), old code paths",
            "cached",
            [*old, *pf8pin, "0.05", *io],
            "blocked",
        ),  # fmt: skip
        s("C_fused", "no cache, fused kernel", "unbuffered", [*fused, "--log-reads"], "fused"),
        s(
            "cache_fused",
            "cache only, fused kernel, pipelined prefill",
            "cached",
            [*new, "--no-prefetch", "--no-pin-layer0", "--no-prefetch-adaptive", *io],
            "fused",
        ),  # fmt: skip
        s(
            "after_fixed",
            "AFTER: fused, pipelined prefill, rank-aware map, fixed top-8 + pin, 0.05",
            "cached",
            [*new, "--no-prefetch-adaptive", *pf8pin, "0.05", *io],
            "fused",
        ),  # fmt: skip
        s(
            "after_adapt_005",
            "AFTER + adaptive read budget, threshold 0.05",
            "cached",
            [*new, "--prefetch-adaptive", *pf8pin, "0.05", *io],
            "fused",
        ),  # fmt: skip
        s(
            "after_adapt_01",
            "AFTER + adaptive read budget, threshold 0.1",
            "cached",
            [*new, "--prefetch-adaptive", *pf8pin, "0.1", *io],
            "fused",
        ),  # fmt: skip
    ]


def read_log_summary(run: dict) -> dict:
    """Read latency by the idle gap before the read, decode steps only
    would need step boundaries, so this covers the whole run (prefill
    reads included), split by who issued the read."""
    log = run.get("read_log") or []
    out = {}
    for kind in ("demand", "prefetch"):
        rows = [e for e in log if e[4] == kind]
        bins = []
        for lo, hi, label in GAP_BINS_MS:
            if hi == 0.0:
                sel = [e for e in rows if e[3] > 0]  # another read already in flight
            else:
                sel = [e for e in rows if e[3] == 0 and lo < e[2] * 1000 <= hi]
            if sel:
                lat = sorted(e[1] * 1000 for e in sel)
                bins.append(
                    {
                        "gap": label,
                        "reads": len(sel),
                        "median_ms": statistics.median(lat),
                        "mean_ms": statistics.fmean(lat),
                    }
                )
        out[kind] = {
            "reads": len(rows),
            "mean_ms": statistics.fmean(e[1] * 1000 for e in rows) if rows else None,
            "by_idle_gap": bins,
        }
    return out


def tokens(result: dict) -> list[list[int]]:
    return [p["generated_ids"] for p in result["run"]["prompts"]]


def agreement(results: list[dict], plan: list[dict]) -> dict:
    ok = [r for r in results if r["status"] == "ok"]
    kernel = {s["id"]: s["kernel"] for s in plan}
    ref = {k: next((r for r in ok if r["id"] == f"C_{k}"), None) for k in ("blocked", "fused")}
    phase2 = phase2_c_tokens()
    out = {}
    for r in ok:
        c = ref[kernel[r["id"]]]
        out[f"{r['id']}#{r['round']}"] = {
            "same_as_same_kernel_no_cache": tokens(r) == tokens(c) if c else None,
            "same_as_phase2_C": tokens(r) == phase2 if phase2 else None,
        }
    fused_c, blocked_c = ref["fused"], ref["blocked"]
    if fused_c and blocked_c:
        same = [a == b for a, b in zip(tokens(fused_c), tokens(blocked_c), strict=True)]
        out["fused_vs_blocked_no_cache"] = {"prompts_identical": sum(same), "prompts": len(same)}
    return out


def _mean(runs: list[dict], key: str, sub: str) -> float:
    return statistics.fmean(r[sub][key] for r in runs)


def format_markdown(record: dict) -> str:
    res = [r for r in record["results"] if r["status"] == "ok"]
    plan = record["config"]["setups"]
    free = record["machine"]["memory"]["available_bytes"] / 1e9
    lines = [
        "# Step B benchmark: fused kernel, pipelined prefill, adaptive prefetching",
        "",
        f"Generated by `python -m expertrelay.bench.stepb_benchmark` at {record['timestamp']} "
        f"(commit {record['git_commit'][:7]}{', dirty tree' if record['git_dirty'] else ''}) from "
        "`benchmarks/results/stepb_benchmark.json`. Do not edit by hand.",
        "",
        f"Base store, Phase 2 settings ({len(record['config']['prompts'])} prompts x "
        f"{record['config']['max_new_tokens']} tokens, greedy). Cache {CACHE_GB} GB, 2 I/O threads. Each setup "
        f"ran twice (second round reversed); means of the two runs. RAM free at the start: {free:.2f} GB.",
        "",
        "| setup | decode tok/s (runs) | first token, s | compute, s/token | blocked on reads, s/token | "
        "demand reads / token | prefetch reads / token | wasted / token | cache hit rate | peak RAM MB |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in plan:
        runs = [r for r in res if r["id"] == s["id"]]
        if not runs:
            lines.append(f"| {s['name']} | failed |" + " |" * 8)
            continue
        each = ", ".join("{:.3f}".format(r["metrics"]["decode_tokens_per_s"]) for r in runs)
        lines.append(
            f"| {s['name']} | **{_mean(runs, 'decode_tokens_per_s', 'metrics'):.3f}** ({each}) | "
            f"{_mean(runs, 'time_to_first_token_mean_s', 'metrics'):.1f} | "
            f"{_mean(runs, 'decode_compute_s_per_token', 'metrics'):.3f} | "
            f"{_mean(runs, 'decode_expert_read_s_per_token', 'metrics'):.3f} | "
            f"{_mean(runs, 'decode_demand_reads_per_token', 'cache'):.1f} | "
            f"{_mean(runs, 'decode_prefetch_reads_per_token', 'cache'):.1f} | "
            f"{_mean(runs, 'decode_prefetches_wasted_per_token', 'cache'):.1f} | "
            f"{_mean(runs, 'decode_hit_rate', 'cache'):.1%} | "
            f"{statistics.fmean(r['peak_rss_mb_polled'] for r in runs):.0f} |"
        )
    lines += ["", "## Correctness", "", "| run | same tokens as the same kernel without cache | same as Phase 2's C |",
              "|---|---|---|"]  # fmt: skip
    for k, v in record["agreement"].items():
        if "same_as_same_kernel_no_cache" in v:
            lines.append(f"| {k} | {v['same_as_same_kernel_no_cache']} | {v['same_as_phase2_C']} |")
    fb = record["agreement"].get("fused_vs_blocked_no_cache")
    if fb:
        lines += [
            "",
            f"Fused vs blocked kernel, no cache: identical tokens on {fb['prompts_identical']} of {fb['prompts']} "
            "prompts (the fused kernel sums in another order; its accuracy is measured against the original "
            "model in `docs/reference-check.md`).",
        ]
    lines += [
        "",
        "## Read latency by the idle gap before the read",
        "",
        "Every read of each run, grouped by how long this process had no read in flight before it. Median "
        "latency, ms (reads). Round 1 runs.",
        "",
    ]
    for s in plan:
        run = next((r for r in res if r["id"] == s["id"] and r["round"] == 0), None)
        if not run:
            continue
        rl = run["read_log_summary"]
        for kind in ("demand", "prefetch"):
            if rl[kind]["reads"]:
                cells = ", ".join(
                    f"{b['gap']}: {b['median_ms']:.2f} ({b['reads']})" for b in rl[kind]["by_idle_gap"]
                )
                lines.append(
                    f"- **{s['id']}**, {kind} ({rl[kind]['reads']} reads, mean {rl[kind]['mean_ms']:.2f} ms): {cells}"
                )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        DOC.write_text(format_markdown(json.loads(OUT.read_text(encoding="utf-8"))[-1]), encoding="utf-8")
        print(f"wrote {DOC}")
        return

    rt = RuntimeConfig.load(DEFAULT_RUNTIME_CONFIG)
    rt_cache = RuntimeConfig.load(CACHE_CONFIG)
    prompts = json.loads(rt.prompts_file.read_text(encoding="utf-8"))["prompts"]
    plan = setups()
    record = base_record(
        label="step B benchmark: fused kernel, pipelined prefill, adaptive prefetch, read log",
        seed=0,
        model={"store": rt.store_dir.relative_to(REPO_ROOT).as_posix()},
        config={
            "prompts": prompts,
            "max_new_tokens": rt.max_new_tokens,
            "max_seq": rt.max_seq,
            "setups": plan,
            "cache_gb": CACHE_GB,
            "rounds": args.rounds,
        },
        machine=collect_machine_profile(measure_disk=False),
    )
    results = []
    for rnd in range(args.rounds):
        for s in plan if rnd % 2 == 0 else plan[::-1]:
            s = dict(s)
            fit = None
            if s["cache"]:
                fit = fit_cache(rt_cache, CACHE_GB)
                s["args"] = [*s["args"], "--cache-gb", f"{fit['used_gb']:.6f}"]
            r = run_one(s, CACHE_CONFIG if s["cache"] else DEFAULT_RUNTIME_CONFIG, TIMEOUT_S)
            r["round"], r["cache_fit"] = rnd, fit
            if r["status"] == "ok":
                r["cache"] = cache_metrics(r["run"])
                r["read_log_summary"] = read_log_summary(r["run"])
            results.append(r)
    record["results"] = results
    record["agreement"] = agreement(results, plan)
    append_benchmark_record(OUT, record)
    DOC.write_text(format_markdown(record), encoding="utf-8")
    print(f"saved {OUT} and {DOC}")


if __name__ == "__main__":
    main()
