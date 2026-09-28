"""Phase 4+5 benchmark: the expert cache and prefetcher vs. the no-cache
baseline, on the Phase 2 prompt set with Phase 2's settings (base store,
max_seq 128, 32 new tokens, greedy).

    python -m expertrelay.bench.phase4_benchmark --io-threads 1
    python -m expertrelay.bench.phase4_benchmark --report     # redo the doc from saved JSON

Setups (each its own subprocess, bench.phase2_baselines.run_one):
  C       no cache: one unbuffered read per expert use (Phase 2 setup C, rerun now)
  cache   LRU cache, 1.25 GB, no prefetch
  pf4     cache + prefetch top-4
  pf8     cache + prefetch top-8
  pf8pin  cache + prefetch top-8 + all layer-0 experts pinned (0.52 GB of the cache)
Prefetch settings not named here (confidence threshold, calibration) come
from configs/runtime_cache.json.

Every setup runs twice; the second round runs in reverse order, so drift
over the session doesn't favour whichever setup ran first. Before each
cached run the RAM estimate (runtime.generate.estimate_ram_bytes) is
checked against the RAM free at that moment; if it doesn't fit with a
margin, the cache is made smaller and the record says so.

Correctness: every run's tokens are compared with today's setup C and with
the tokens Phase 2's setup C recorded (benchmarks/results/phase2_baselines.json).

Results: benchmarks/results/phase4_benchmark.json, and docs/phase4-benchmark.md
generated from it, next to the projections for the same settings.
"""

from __future__ import annotations

import argparse
import json
import statistics

from expertrelay import REPO_ROOT
from expertrelay.bench.phase2_baselines import run_one
from expertrelay.bench.phase3_traces import PHASE2_RESULTS
from expertrelay.benchmarking import append_benchmark_record, base_record
from expertrelay.cache.expert_cache import slots_for
from expertrelay.manager.profile import collect_machine_profile, measure_memory
from expertrelay.paths import BENCHMARK_RESULTS_DIR
from expertrelay.runtime.generate import (
    DEFAULT_RUNTIME_CONFIG,
    RuntimeConfig,
    estimate_ram_bytes,
    model_config,
)

CACHE_CONFIG = REPO_ROOT / "configs" / "runtime_cache.json"
CACHE_GB = 1.25
FREE_RAM_MARGIN_BYTES = 400 * 2**20
TIMEOUT_S = 3600
DOC = REPO_ROOT / "docs" / "phase4-benchmark.md"
OUT = BENCHMARK_RESULTS_DIR / "phase4_benchmark.json"


def setups(io_threads: int) -> list[dict]:
    io = ["--io-threads", str(io_threads)]
    return [
        {
            "id": "C",
            "name": "no cache (Phase 2 setup C)",
            "source": "unbuffered",
            "budget": True,
            "cache": False,
        },
        {
            "id": "cache",
            "name": "cache, no prefetch",
            "source": "cached",
            "budget": True,
            "cache": True,
            "args": ["--no-prefetch", "--no-pin-layer0", *io],
        },
        {
            "id": "pf4",
            "name": "cache + prefetch top-4",
            "source": "cached",
            "budget": True,
            "cache": True,
            "args": ["--prefetch", "--prefetch-k", "4", "--no-pin-layer0", *io],
        },
        {
            "id": "pf8",
            "name": "cache + prefetch top-8",
            "source": "cached",
            "budget": True,
            "cache": True,
            "args": ["--prefetch", "--prefetch-k", "8", "--no-pin-layer0", *io],
        },
        {
            "id": "pf8pin",
            "name": "cache + prefetch top-8 + layer 0 pinned",
            "source": "cached",
            "budget": True,
            "cache": True,
            "args": ["--prefetch", "--prefetch-k", "8", "--pin-layer0", *io],
        },
    ]


def fit_cache(rt: RuntimeConfig, wanted_gb: float) -> dict:
    """The cache size to use now: `wanted_gb`, or less if the whole process
    wouldn't fit in the RAM free right now (with a margin)."""
    cfg = model_config(rt.store_dir)
    record = json.loads((rt.store_dir / "store.json").read_text())["expert_layout"]["record_size"]
    free = measure_memory().available_bytes
    base = estimate_ram_bytes(rt.store_dir, cfg, rt.max_seq, "cached", "numpy", 0.0)
    room = free - FREE_RAM_MARGIN_BYTES - base
    wanted_slots = slots_for(int(wanted_gb * 1e9), record)
    slots = min(wanted_slots, max(room, 0) // record)
    used_gb = slots * record / 1e9
    return {
        "requested_gb": wanted_gb,
        "used_gb": used_gb,
        "slots": int(slots),
        "ram_free_bytes": free,
        "estimate_without_cache_bytes": base,
        "lowered": slots < wanted_slots,
    }


def cache_metrics(run: dict) -> dict:
    decode = [s for p in run["prompts"] for s in p["steps"][1:]]
    n = len(decode)
    loads = sum(s["expert_loads"] for s in decode)

    def total(key: str) -> int:
        return sum(s[key] for s in decode)

    layers = len(decode[0]["attention_seconds"])

    def per_layer(key: str) -> list[float]:
        return [statistics.fmean(s[key][layer] for s in decode) for layer in range(layers)]

    return {
        "decode_hit_rate": (total("cache_hits") + total("prefetch_hits")) / loads,
        "decode_prefetch_hit_rate": total("prefetch_hits") / loads,
        "decode_prefetch_wait_rate": total("prefetch_waits") / loads,
        "decode_demand_reads_per_token": total("demand_reads") / n,
        "decode_prefetches_issued_per_token": total("prefetches_issued") / n,
        "decode_prefetch_reads_per_token": total("prefetch_reads") / n,
        "decode_prefetches_wasted_per_token": total("prefetches_wasted") / n,
        "decode_prefetches_cancelled_per_token": total("prefetches_cancelled") / n,
        "decode_prefetches_skipped_low_confidence_per_token": total("prefetches_skipped_low_confidence") / n,
        "per_layer_ms": {
            "attention": [x * 1000 for x in per_layer("attention_seconds")],
            "moe_compute": [x * 1000 for x in per_layer("moe_compute_seconds")],
            "read_wait": [x * 1000 for x in per_layer("read_wait_seconds")],
        },
    }


def tokens(result: dict) -> list[list[int]]:
    return [p["generated_ids"] for p in result["run"]["prompts"]]


def phase2_c_tokens() -> list[list[int]] | None:
    record = json.loads(PHASE2_RESULTS.read_text(encoding="utf-8"))[-1]
    c = next((r for r in record["results"] if r["id"] == "C" and r.get("run")), None)
    return [p["generated_ids"] for p in c["run"]["prompts"]] if c else None


def agreement(results: list[dict]) -> dict:
    ok = [r for r in results if r["status"] == "ok"]
    today_c = next((r for r in ok if r["id"] == "C"), None)
    ref2 = phase2_c_tokens()
    return {
        f"{r['id']}#{r['round']}": {
            "same_as_todays_C": tokens(r) == tokens(today_c) if today_c else None,
            "same_as_phase2_C": tokens(r) == ref2 if ref2 else None,
        }
        for r in ok
    }


def projections() -> dict:
    """What the trace simulations projected for these settings, if saved.
    Different prompts (Phase 3 test set, not Phase 2's four), so a sanity
    check, not a like-for-like prediction."""
    out = {}
    p4 = BENCHMARK_RESULTS_DIR / "phase4_offline_qwen1.5-moe-a2.7b-int8.json"
    if p4.exists():
        a = json.loads(p4.read_text())[-1]["analysis"]
        row = a["rows"].get("144")
        if row:
            for sid, key in (("cache", "lru+k0"), ("pf8", "lru+k8"), ("pf8pin", "lru+k8+pin0")):
                if key in row:
                    out[sid] = {
                        "decode_hit_rate": row[key]["decode_hit_rate"],
                        "projected_tok_s": row[key]["projected_tok_s"],
                        "source": f"{p4.name} (144 experts, runtime policy)",
                    }
    return out


def _mean(results: list[dict], sid: str, key: str, sub: str = "metrics") -> float | None:
    vals = [r[sub][key] for r in results if r["id"] == sid and r.get(sub) and key in r[sub]]
    return statistics.fmean(vals) if vals else None


def format_markdown(record: dict) -> str:
    res = record["results"]
    ok = [r for r in res if r["status"] == "ok"]
    proj = record["projections"]
    ids = [s["id"] for s in record["config"]["setups"]]
    names = {s["id"]: s["name"] for s in record["config"]["setups"]}
    free = record["machine"]["memory"]["available_bytes"] / 1e9
    lines = [
        "# Phase 4+5 benchmark: expert cache and prefetcher",
        "",
        f"Generated by `python -m expertrelay.bench.phase4_benchmark` at {record['timestamp']} "
        f"(commit {record['git_commit'][:7]}{', dirty tree' if record['git_dirty'] else ''}) from "
        "`benchmarks/results/phase4_benchmark.json`. Do not edit by hand.",
        "",
        f"Base store, Phase 2 settings: {len(record['config']['prompts'])} prompts, "
        f"{record['config']['max_new_tokens']} new tokens each, greedy, max_seq {record['config']['max_seq']}. "
        f"Each setup ran twice (second round in reverse order); numbers are the mean of the two runs. "
        f"I/O threads: {record['config']['io_threads']}. RAM free at the start: {free:.2f} GB. "
        f"Expert file: {record['config']['experts_file']}.",
        "",
        "| setup | decode tok/s (run 1, run 2) | first token, s | peak RAM MB (working set / committed) | "
        "cache hit rate | of which prefetch | waited for a prefetch | demand reads / token | "
        "prefetch reads / token | wasted / token | blocked on reads, s/token | compute, s/token |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for sid in ids:
        runs = [r for r in ok if r["id"] == sid]
        if not runs:
            status = "; ".join(r["status"] for r in res if r["id"] == sid)
            lines.append(f"| {names[sid]} | {status} |" + " |" * 10)
            continue
        tps = ", ".join(f"{r['metrics']['decode_tokens_per_s']:.3f}" for r in runs)
        c = [r for r in runs if r.get("cache")]  # every finished run, setup C included

        def cm(key: str, fmt: str, runs_c=c) -> str:
            v = [r["cache"][key] for r in runs_c]
            return fmt.format(statistics.fmean(v)) if v else "-"

        lines.append(
            f"| {names[sid]} | **{_mean(ok, sid, 'decode_tokens_per_s'):.3f}** ({tps}) | "
            f"{_mean(ok, sid, 'time_to_first_token_mean_s'):.1f} | "
            f"{statistics.fmean(r['peak_rss_mb_polled'] for r in runs):.0f} / "
            f"{statistics.fmean(r['peak_private_mb_polled'] for r in runs):.0f} | "
            f"{cm('decode_hit_rate', '{:.1%}')} | {cm('decode_prefetch_hit_rate', '{:.1%}')} | "
            f"{cm('decode_prefetch_wait_rate', '{:.1%}')} | "
            f"{cm('decode_demand_reads_per_token', '{:.1f}')} | "
            f"{cm('decode_prefetch_reads_per_token', '{:.1f}')} | {cm('decode_prefetches_wasted_per_token', '{:.1f}')} | "
            f"{_mean(ok, sid, 'decode_expert_read_s_per_token'):.3f} | {_mean(ok, sid, 'decode_compute_s_per_token'):.3f} |"
        )
    lines += ["", "Cache sizes actually used (lowered only if RAM free at that moment was short):", ""]
    for r in res:
        if r.get("cache_fit"):
            f = r["cache_fit"]
            lines.append(
                f"- {r['id']} run {r['round'] + 1}: {f['used_gb']:.2f} GB ({f['slots']} experts)"
                + (f", LOWERED from {f['requested_gb']} GB" if f["lowered"] else "")
                + f"; RAM free before: {f['ram_free_bytes'] / 1e9:.2f} GB"
            )
    lines += [
        "",
        "## Correctness on the real model",
        "",
        "Generated tokens compared with today's setup C and with Phase 2's recorded setup C "
        "(same prompts, same store):",
        "",
        "| run | same as today's C | same as Phase 2's C |",
        "|---|---|---|",
        *[
            f"| {k} | {v['same_as_todays_C']} | {v['same_as_phase2_C']} |"
            for k, v in record["agreement"].items()
        ],
        "",
        "## Measured vs. projected",
        "",
        "Projections from the trace simulation (`docs/phase3-analysis.md`, section 8) used different prompts "
        "(the Phase 3 test set) and an assumed 4.91 ms per read with reads hidden behind the whole previous "
        "layer's compute. A sanity check, not a like-for-like prediction.",
        "",
        "| setup | projected hit rate | measured hit rate | projected tok/s (4.91 ms reads) | measured tok/s |",
        "|---|---|---|---|---|",
    ]
    for sid, p in proj.items():
        runs = [r for r in ok if r["id"] == sid and r.get("cache")]
        hr = statistics.fmean(r["cache"]["decode_hit_rate"] for r in runs) if runs else None
        tps = _mean(ok, sid, "decode_tokens_per_s")
        lines.append(
            f"| {names[sid]} | {p['decode_hit_rate']:.1%} | {'-' if hr is None else f'{hr:.1%}'} | "
            f"{p['projected_tok_s']['phase2_runtime']:.2f} | {'-' if tps is None else f'{tps:.2f}'} |"
        )
    c_runs = [r for r in ok if r["id"] == "C" and r.get("cache")]
    if c_runs:
        pl = c_runs[0]["cache"]["per_layer_ms"]
        att = [
            statistics.fmean(r["cache"]["per_layer_ms"]["attention"][i] for r in c_runs)
            for i in range(len(pl["attention"]))
        ]
        moe = [
            statistics.fmean(r["cache"]["per_layer_ms"]["moe_compute"][i] for r in c_runs)
            for i in range(len(pl["attention"]))
        ]
        wait = [
            statistics.fmean(r["cache"]["per_layer_ms"]["read_wait"][i] for r in c_runs)
            for i in range(len(pl["attention"]))
        ]
        window = [moe[i] + att[i + 1] for i in range(len(att) - 1)]
        lines += [
            "",
            "## Per-layer time, setup C (no cache): the real prefetch window",
            "",
            "Mean per generated token, ms. The prefetch for layer L+1 is issued right after layer L's router, "
            "so it can hide behind layer L's MoE compute plus layer L+1's attention: that sum is the window. "
            "(MoE compute includes the router, the routed experts and the shared expert, minus time blocked on "
            "reads.)",
            "",
            f"Averages over layers: attention {statistics.fmean(att):.1f} ms, MoE compute {statistics.fmean(moe):.1f} ms, "
            f"blocked on reads {statistics.fmean(wait):.1f} ms; window {statistics.fmean(window):.1f} ms "
            f"(min {min(window):.1f}, max {max(window):.1f}).",
            "",
            "| layer | attention | MoE compute | blocked on reads | window for the next layer's prefetch |",
            "|---|---|---|---|---|",
            *[
                f"| {i} | {att[i]:.1f} | {moe[i]:.1f} | {wait[i]:.1f} | {window[i]:.1f} |"
                if i < len(window)
                else f"| {i} | {att[i]:.1f} | {moe[i]:.1f} | {wait[i]:.1f} | - |"
                for i in range(len(att))
            ],
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--io-threads", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        DOC.write_text(format_markdown(json.loads(OUT.read_text(encoding="utf-8"))[-1]), encoding="utf-8")
        print(f"wrote {DOC}")
        return

    rt = RuntimeConfig.load(DEFAULT_RUNTIME_CONFIG)
    rt_cache = RuntimeConfig.load(CACHE_CONFIG)
    for field in ("max_seq", "max_new_tokens", "prompts_file", "store_dir"):
        if getattr(rt, field) != getattr(rt_cache, field):
            raise SystemExit(f"{CACHE_CONFIG.name} and the Phase 2 config differ in {field}: not comparable")
    prompts = json.loads(rt.prompts_file.read_text(encoding="utf-8"))["prompts"]
    record_bytes = json.loads((rt.store_dir / "store.json").read_text())["expert_layout"]["record_size"]
    plan = setups(args.io_threads)
    record = base_record(
        label="phase4+5 benchmark: expert cache and prefetcher",
        seed=0,
        model={
            "store": rt.store_dir.relative_to(REPO_ROOT).as_posix(),
            "source": json.loads((rt.store_dir / "store.json").read_text())["source"]["revision"],
        },
        config={
            "max_seq": rt.max_seq,
            "max_new_tokens": rt.max_new_tokens,
            "prompts": prompts,
            "setups": plan,
            "cache_gb": CACHE_GB,
            "io_threads": args.io_threads,
            "rounds": args.rounds,
            "record_bytes": record_bytes,
            "prefetch_min_probability": rt_cache.prefetch_min_probability,
            "prefetch_calibration": rt_cache.prefetch_calibration.relative_to(REPO_ROOT).as_posix()
            if rt_cache.prefetch_calibration
            else None,
            "experts_file": "experts.bin (see docs/limitations.md for which copy)",
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
            config = CACHE_CONFIG if s["cache"] else DEFAULT_RUNTIME_CONFIG
            r = run_one(s, config, TIMEOUT_S)
            r["round"] = rnd
            r["cache_fit"] = fit
            if r["status"] == "ok":
                r["cache"] = cache_metrics(r["run"])
            results.append(r)
    record["results"] = results
    record["agreement"] = agreement(results)
    record["projections"] = projections()
    append_benchmark_record(OUT, record)
    DOC.write_text(format_markdown(record), encoding="utf-8")
    print(f"saved {OUT} and {DOC}")


if __name__ == "__main__":
    main()
