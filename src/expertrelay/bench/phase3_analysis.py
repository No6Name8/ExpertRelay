"""Phase 3, steps 3-4: analyze expert-usage traces and simulate caches.

    python -m expertrelay.bench.phase3_analysis                       # base store
    python -m expertrelay.bench.phase3_analysis --store-dir models/qwen1.5-moe-a2.7b-chat-int8
    python -m expertrelay.bench.phase3_analysis --report              # redo charts + doc from saved JSON

Reads only finished traces (models/traces/<store>/<id>.ert with a matching
.done.json). All numbers go to benchmarks/results/phase3_analysis_<store>.json;
charts (docs/phase3/<store>/*.png) and the doc are then generated from that
file alone.

What counts as an "access". The runtime loads each unique expert of a layer
once per forward call, in ascending expert-id order (runtime.qwen_moe). So
a prefill call contributes, per layer, the set of experts any prompt token
picked; a decode call contributes exactly top_k experts per layer. The
cache simulator replays that stream: prompts back to back in the run's
order, cache warm across prompts, keys = (layer, expert).

Projected tokens/s are a PROJECTION, not a measurement: Phase 2's measured
per-token compute time, plus the simulated misses per decode token times
Phase 2's measured read time per expert. It assumes nothing else changes
(no read/compute overlap, same machine load).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from expertrelay import REPO_ROOT
from expertrelay.bench.phase3_traces import DEFAULT_CONFIG, PHASE2_RESULTS, round_robin, trace_dir_for
from expertrelay.benchmarking import append_benchmark_record, base_record, peak_process_rss_mb
from expertrelay.cache.simulator import (
    INFINITE,
    lru_hit_counts,
    lru_stack_distances,
    next_use_indices,
    simulate_belady,
    simulate_lfu,
    simulate_pinned,
)
from expertrelay.manager.profile import collect_machine_profile
from expertrelay.paths import BENCHMARK_RESULTS_DIR
from expertrelay.runtime.expert_trace import NO_EXPERT, PHASE_DECODE, PHASE_PREFILL, read_trace
from expertrelay.runtime.generate import RuntimeConfig

CAPACITIES = [0, 24, 48, 96, 144, 192, 288, 384, 480, 576, 720, 960, 1200, 1440]
PINNED_FRACTION = 0.5
CONFIDENCE_BINS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0001]


@dataclass
class Trace:
    header: dict
    rec: np.ndarray

    @property
    def category(self) -> str:
        return self.header["category"]


def load_traces(trace_dir: Path, prompt_order: list[str]) -> list[Trace]:
    """Finished traces only, in the run's prompt order."""
    out = []
    for pid in prompt_order:
        if (trace_dir / f"{pid}.done.json").exists():
            header, rec = read_trace(trace_dir / f"{pid}.ert")
            out.append(Trace(header, rec))
    return out


# ---------------------------------------------------------------------------
# Access stream (what the runtime actually loads)
# ---------------------------------------------------------------------------


def access_stream(trace: Trace) -> tuple[np.ndarray, np.ndarray]:
    """(keys, phases) in load order: per forward call, per layer, the unique
    experts in ascending id order."""
    rec, h = trace.rec, trace.header
    num_layers, num_experts = h["num_layers"], h["num_experts"]
    keys, phases = [], []
    prefill = rec[rec["phase"] == PHASE_PREFILL]
    calls = [prefill] if len(prefill) else []
    decode = rec[rec["phase"] == PHASE_DECODE]
    calls += [decode[decode["position"] == p] for p in np.unique(decode["position"])]
    for call in calls:
        phase = int(call["phase"][0])
        for layer in range(num_layers):
            experts = np.unique(call["experts"][call["layer"] == layer]).astype(np.int64)  # u8 on disk
            keys.append(layer * num_experts + experts)
            phases.append(np.full(len(experts), phase, dtype=np.uint8))
    return np.concatenate(keys).astype(np.int64), np.concatenate(phases)


def concat_streams(traces: list[Trace]) -> tuple[np.ndarray, np.ndarray]:
    parts = [access_stream(t) for t in traces]
    return np.concatenate([k for k, _ in parts]), np.concatenate([p for _, p in parts])


# ---------------------------------------------------------------------------
# Usage statistics
# ---------------------------------------------------------------------------


def popularity(traces: list[Trace], phase: int | None) -> dict:
    """Share of expert selections that go to each layer's top 10% of experts."""
    h = traces[0].header
    num_layers, num_experts = h["num_layers"], h["num_experts"]
    counts = np.zeros((num_layers, num_experts), dtype=np.int64)
    for t in traces:
        r = t.rec if phase is None else t.rec[t.rec["phase"] == phase]
        np.add.at(counts, (np.repeat(r["layer"], h["top_k"]), r["experts"].reshape(-1)), 1)
    top_n = int(np.ceil(0.1 * num_experts))
    per_layer = [float(np.sort(c)[::-1][:top_n].sum() / c.sum()) if c.sum() else float("nan") for c in counts]
    return {
        "top10pct_experts": top_n,
        "top10pct_share_per_layer": per_layer,
        "top10pct_share_mean": float(np.nanmean(per_layer)),
        "uniform_share": top_n / num_experts,
        "experts_never_used_per_layer": [int((c == 0).sum()) for c in counts],
    }


def reuse_distances(traces: list[Trace]) -> np.ndarray:
    """For every decode-step selection: tokens since the same expert was last
    used at the same layer in the same sequence (prefill uses count as uses).
    First use in the sequence: INFINITE."""
    out = []
    for t in traces:
        h = t.header
        last = np.full((h["num_layers"], h["num_experts"]), -1, dtype=np.int64)
        rec = t.rec
        order = np.lexsort((rec["layer"], rec["position"]))
        for r in rec[order]:
            layer, pos = int(r["layer"]), int(r["position"])
            for e in r["experts"]:
                prev = last[layer, e]
                if r["phase"] == PHASE_DECODE:
                    out.append(INFINITE if prev < 0 else pos - prev)
                last[layer, e] = pos
    return np.asarray(out, dtype=np.int64)


def distance_summary(d: np.ndarray) -> dict:
    finite = d[d != INFINITE]
    return {
        "selections": int(len(d)),
        "first_use_fraction": float(np.mean(d == INFINITE)),
        "within_1": float(np.mean(d <= 1)),
        "within_2": float(np.mean(d <= 2)),
        "within_8": float(np.mean(d <= 8)),
        "within_32": float(np.mean(d <= 32)),
        "median_finite": float(np.median(finite)) if len(finite) else None,
        "cdf_x": list(range(1, 65)),
        "cdf_y": [float(np.mean(d <= x)) for x in range(1, 65)],
    }


def _overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """|a_i ∩ b_i| per row, for [n, k] expert-id arrays (NO_EXPERT in b never matches)."""
    return (a[:, :, None] == b[:, None, :]).any(axis=2).sum(axis=1)


def temporal_locality(rec: np.ndarray, top_k: int) -> dict:
    r = rec[rec["prev_token_experts"][:, 0] != NO_EXPERT]
    ov = _overlap(r["experts"], r["prev_token_experts"])
    return {
        "records": int(len(r)),
        "mean_shared_fraction": float(ov.mean() / top_k) if len(r) else None,
        "at_least_one_shared": float(np.mean(ov >= 1)) if len(r) else None,
        "all_shared": float(np.mean(ov == top_k)) if len(r) else None,
    }


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def fate_accuracy(rec: np.ndarray, top_k: int) -> dict:
    """How much of the true top-k the Fate-style guess catches, and whether its
    confidence means anything."""
    r = rec[rec["fate_experts"][:, 0] != NO_EXPERT]
    if not len(r):
        return {"records": 0}
    hits = _overlap(r["fate_experts"], r["experts"])
    probs = _softmax(r["fate_logits"].astype(np.float32))
    pred_p = np.take_along_axis(probs, r["fate_experts"].astype(np.int64), axis=1)  # [n, k]
    chosen = (r["fate_experts"][:, :, None] == r["experts"][:, None, :]).any(axis=2)  # [n, k]
    bins = np.digitize(pred_p.reshape(-1), CONFIDENCE_BINS) - 1
    per_bin = [
        {
            "lo": CONFIDENCE_BINS[b],
            "hi": min(CONFIDENCE_BINS[b + 1], 1.0),
            "count": int((bins == b).sum()),
            "mean_predicted_prob": float(pred_p.reshape(-1)[bins == b].mean()) if (bins == b).any() else None,
            "fraction_chosen": float(chosen.reshape(-1)[bins == b].mean()) if (bins == b).any() else None,
        }
        for b in range(len(CONFIDENCE_BINS) - 1)
    ]
    mass = pred_p.sum(axis=1)
    mass_q = np.quantile(mass, [0.25, 0.5, 0.75])
    mass_bins = np.digitize(mass, mass_q)
    return {
        "records": int(len(r)),
        "recall_at_k": float(hits.mean() / top_k),
        "hits_distribution": [float(np.mean(hits == i)) for i in range(top_k + 1)],
        "per_expert_confidence_bins": per_bin,
        # None for an empty quartile (e.g. many identical confidences): NaN isn't valid JSON
        "recall_by_confidence_quartile": [
            float(hits[mass_bins == q].mean() / top_k) if (mass_bins == q).any() else None for q in range(4)
        ],
        "confidence_quartile_edges": [float(x) for x in mass_q],
    }


def by_layer(traces: list[Trace], fn, phase: int | None = None) -> list:
    h = traces[0].header
    rec = np.concatenate([t.rec for t in traces])
    if phase is not None:
        rec = rec[rec["phase"] == phase]
    return [fn(rec[rec["layer"] == layer], h["top_k"]) for layer in range(h["num_layers"])]


# ---------------------------------------------------------------------------
# Cache simulation
# ---------------------------------------------------------------------------


def hot_keys_from(traces: list[Trace]) -> list[int]:
    """(layer, expert) keys, most-loaded first, over these traces' access streams.
    Empty with no traces: the pinned policy then pins nothing (plain LRU)."""
    if not traces:
        return []
    keys, _ = concat_streams(traces)
    counts = np.bincount(keys)
    order = np.argsort(-counts, kind="stable")
    return [int(k) for k in order if counts[k] > 0]


def simulate_hits(traces: list[Trace], hot_set: list[int]) -> dict:
    """Hit COUNTS at every capacity for every policy, over one stream: these
    traces back to back, cache warm across them. `hot_set` (for the pinned
    policy) must come from other traces, never from these."""
    h = traces[0].header
    num_keys = h["num_layers"] * h["num_experts"]
    parts = [access_stream(t) for t in traces]
    stream = np.concatenate([k for k, _ in parts])
    decode = np.concatenate([ph for _, ph in parts]) == PHASE_DECODE

    def counts(marks_per_cap: list[np.ndarray]) -> dict:
        return {
            "hits": [int(m.sum()) for m in marks_per_cap],
            "decode_hits": [int(m[decode].sum()) for m in marks_per_cap],
        }

    dist = lru_stack_distances(stream, num_keys)
    nu = next_use_indices(stream)
    return {
        "accesses": len(stream),
        "decode_accesses": int(decode.sum()),
        "decode_tokens": _decode_tokens(traces),
        "policies": {
            "lru": {
                "hits": lru_hit_counts(dist, CAPACITIES).tolist(),
                "decode_hits": lru_hit_counts(dist[decode], CAPACITIES).tolist(),
            },
            "lfu": counts([simulate_lfu(stream, c) for c in CAPACITIES]),
            "pinned_lru": counts([simulate_pinned(stream, c, hot_set, PINNED_FRACTION) for c in CAPACITIES]),
            "belady": counts([simulate_belady(stream, c, nu) for c in CAPACITIES]),
        },
    }


def hit_rate_curves(runs: list[dict]) -> dict:
    """Pool one or more simulate_hits runs (e.g. the two held-out halves) into
    hit rates. Every policy sees exactly the same streams."""
    total = sum(r["accesses"] for r in runs)
    total_decode = sum(r["decode_accesses"] for r in runs)
    policies = {}
    for name in runs[0]["policies"]:
        hits = np.sum([r["policies"][name]["hits"] for r in runs], axis=0)
        dec = np.sum([r["policies"][name]["decode_hits"] for r in runs], axis=0)
        policies[name] = {
            "hit_rate": (hits / total).tolist(),
            "decode_hit_rate": (dec / total_decode).tolist(),
        }
    return {
        "capacities": CAPACITIES,
        "accesses": total,
        "decode_accesses": total_decode,
        "decode_accesses_per_token": total_decode / sum(r["decode_tokens"] for r in runs),
        "streams": len(runs),
        "policies": policies,
    }


def _decode_tokens(traces: list[Trace]) -> int:
    return sum(int(((t.rec["layer"] == 0) & (t.rec["phase"] == PHASE_DECODE)).sum()) for t in traces)


def projection(curves: dict, speeds: dict, record_bytes: int) -> dict:
    """PROJECTED decode tok/s from simulated hit rates and Phase 2's measured
    compute and per-expert read times. Not a measurement."""
    per_token = curves["decode_accesses_per_token"]
    out = {}
    for name, pol in curves["policies"].items():
        out[name] = [
            1.0 / (speeds["compute_s_per_token"] + per_token * (1 - hr) * speeds["read_s_per_expert"])
            for hr in pol["decode_hit_rate"]
        ]
    return {
        "label": "PROJECTION from simulated hit rates + Phase 2 measured times; not a measurement",
        "inputs": speeds,
        "cache_ram_gb": [c * record_bytes / 1e9 for c in CAPACITIES],
        "decode_tok_s": out,
    }


def phase2_speeds(record_bytes: int) -> dict:
    record = json.loads(PHASE2_RESULTS.read_text(encoding="utf-8"))[-1]
    m = next(r for r in record["results"] if r["id"] == "C")["metrics"]
    loads = m["decode_expert_mb_per_token"] * 1e6 / record_bytes
    return {
        "compute_s_per_token": m["decode_compute_s_per_token"],
        "read_s_per_expert": m["decode_expert_read_s_per_token"] / loads,
        "measured_tok_s_no_cache": m["decode_tokens_per_s"],
        "source": f"{PHASE2_RESULTS.relative_to(REPO_ROOT).as_posix()} @ {record['git_commit'][:7]}",
    }


# ---------------------------------------------------------------------------
# Whole analysis -> JSON
# ---------------------------------------------------------------------------


def analyze(traces: list[Trace], record_bytes: int) -> dict:
    top_k = traces[0].header["top_k"]
    categories = sorted({t.category for t in traces})
    by_cat = {c: [t for t in traces if t.category == c] for c in categories}
    all_rec = np.concatenate([t.rec for t in traces])

    def split(fn):
        return {
            "all": fn(all_rec, top_k),
            "prefill": fn(all_rec[all_rec["phase"] == PHASE_PREFILL], top_k),
            "decode": fn(all_rec[all_rec["phase"] == PHASE_DECODE], top_k),
            "by_category": {c: fn(np.concatenate([t.rec for t in ts]), top_k) for c, ts in by_cat.items()},
        }

    # Pinned hot sets never come from the data being replayed. "all" is two
    # held-out halves (even/odd prompts in run order), each replayed warm with
    # the other half's hot set; every policy sees those same two streams.
    halves = [traces[0::2], traces[1::2]]
    runs = [simulate_hits(h, hot_keys_from(halves[1 - i])) for i, h in enumerate(halves) if h]
    curves_all = hit_rate_curves(runs)
    curves_by_cat = {
        c: hit_rate_curves([simulate_hits(ts, hot_keys_from([t for t in traces if t.category != c]))])
        for c, ts in by_cat.items()
    }
    speeds = phase2_speeds(record_bytes)

    return {
        "prompts": len(traces),
        "prompts_by_category": {c: len(ts) for c, ts in by_cat.items()},
        "tokens": int((all_rec["layer"] == 0).sum()),
        "decode_tokens": _decode_tokens(traces),
        "popularity": {
            "all": popularity(traces, None),
            "prefill": popularity(traces, PHASE_PREFILL),
            "decode": popularity(traces, PHASE_DECODE),
            "by_category": {c: popularity(ts, None) for c, ts in by_cat.items()},
        },
        "reuse_distance": {
            "all": distance_summary(reuse_distances(traces)),
            "by_category": {c: distance_summary(reuse_distances(ts)) for c, ts in by_cat.items()},
        },
        "temporal_locality": {
            **split(temporal_locality),
            "by_layer_decode": by_layer(traces, temporal_locality, PHASE_DECODE),
        },
        "fate": {**split(fate_accuracy), "by_layer_decode": by_layer(traces, fate_accuracy, PHASE_DECODE)},
        "cache": {
            "pinned_fraction": PINNED_FRACTION,
            "all": curves_all,
            "by_category": curves_by_cat,
            "projection": projection(curves_all, speeds, record_bytes),
            "projection_by_category": {
                c: projection(v, speeds, record_bytes) for c, v in curves_by_cat.items()
            },
        },
    }


# ---------------------------------------------------------------------------
# Charts and doc, generated from the saved JSON record only
# ---------------------------------------------------------------------------

POLICY_LABELS = {
    "lru": "LRU",
    "lfu": "LFU",
    "pinned_lru": "LRU + pinned hot set",
    "belady": "Belady (optimal, needs the future)",
}
BASE_STORE = "qwen1.5-moe-a2.7b-int8"
# Sections other scripts append to the doc (e.g. bench/phase35_prediction.py) start at
# this line; regenerating the Phase 3 part keeps everything from it on.
APPENDED_SECTIONS_MARKER = "<!-- appended sections: kept when this doc is regenerated -->"


def doc_path_for(store: str) -> Path:
    name = "phase3-analysis.md" if store == BASE_STORE else f"phase3-analysis-{store}.md"
    return REPO_ROOT / "docs" / name


def render(record: dict, json_path: Path) -> Path:
    # Imported here, not at the top: computing the analysis (and its tests) doesn't
    # need a plotting library, only rendering does.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a, store = record["analysis"], record["model"]["store"]
    chart_dir = REPO_ROOT / "docs" / "phase3" / store
    chart_dir.mkdir(parents=True, exist_ok=True)
    cats = sorted(a["prompts_by_category"])
    caps = a["cache"]["all"]["capacities"]
    charts: dict[str, str] = {}

    def save(fig, name: str) -> None:
        fig.tight_layout()
        fig.savefig(chart_dir / f"{name}.png", dpi=110)
        plt.close(fig)
        charts[name] = f"phase3/{store}/{name}.png"

    fig, ax = plt.subplots(figsize=(8, 4))
    for c in cats:
        ax.plot(a["popularity"]["by_category"][c]["top10pct_share_per_layer"], label=c)
    ax.axhline(
        a["popularity"]["all"]["uniform_share"], color="grey", ls="--", label="uniform (no preference)"
    )
    ax.set(xlabel="layer", ylabel="share of selections", title="Traffic to each layer's top 10% of experts")
    ax.legend(fontsize=8, ncol=2)
    save(fig, "popularity")

    fig, ax = plt.subplots(figsize=(8, 4))
    x = a["reuse_distance"]["all"]["cdf_x"]
    ax.plot(x, a["reuse_distance"]["all"]["cdf_y"], color="black", lw=2, label="all")
    for c in cats:
        ax.plot(x, a["reuse_distance"]["by_category"][c]["cdf_y"], label=c, alpha=0.8)
    ax.set(
        xlabel="tokens since this expert was last used at this layer",
        ylabel="fraction of decode selections",
        title="Reuse distance (CDF)",
        xscale="log",
    )
    ax.legend(fontsize=8, ncol=2)
    save(fig, "reuse_distance")

    fig, ax = plt.subplots(figsize=(8, 4))
    loc = [x["mean_shared_fraction"] for x in a["temporal_locality"]["by_layer_decode"]]
    fate = [x.get("recall_at_k", np.nan) for x in a["fate"]["by_layer_decode"]]
    ax.plot(range(len(loc)), loc, marker="o", label="previous token's experts (same layer)")
    ax.plot(range(len(fate)), fate, marker="s", label="Fate-style cross-layer guess")
    ax.set(
        xlabel="layer",
        ylabel="fraction of true top-4 caught",
        ylim=(0, 1),
        title="How much of the true top-4 each predictor catches (decode)",
    )
    ax.legend(fontsize=8)
    save(fig, "predictors_by_layer")

    fig, ax = plt.subplots(figsize=(5, 5))
    bins = [b for b in a["fate"]["all"]["per_expert_confidence_bins"] if b["count"]]
    ax.plot([0, 1], [0, 1], color="grey", ls="--", label="perfectly calibrated")
    ax.plot(
        [b["mean_predicted_prob"] for b in bins],
        [b["fraction_chosen"] for b in bins],
        marker="o",
        label="Fate-style guess",
    )
    ax.set(
        xlabel="predicted router probability of a guessed expert",
        ylabel="fraction actually chosen (top-4)",
        title="Is the guess's confidence trustworthy?",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend(fontsize=8)
    save(fig, "fate_confidence")

    fig, ax = plt.subplots(figsize=(8, 4))
    for name, pol in a["cache"]["all"]["policies"].items():
        ax.plot(caps, pol["decode_hit_rate"], marker=".", label=POLICY_LABELS[name])
    ax.set(
        xlabel="cache size (experts; 1440 = all)",
        ylabel="decode hit rate",
        ylim=(0, 1),
        title="Expert-cache hit rate vs cache size, all prompts",
    )
    ax.legend(fontsize=8)
    save(fig, "hit_rate_policies")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, pol in zip(axes, ("lru", "belady"), strict=True):
        for c in cats:
            ax.plot(caps, a["cache"]["by_category"][c]["policies"][pol]["decode_hit_rate"], label=c)
        ax.set(xlabel="cache size (experts)", title=POLICY_LABELS[pol], ylim=(0, 1))
    axes[0].set_ylabel("decode hit rate")
    axes[1].legend(fontsize=8)
    fig.suptitle("Hit rate per category")
    save(fig, "hit_rate_by_category")

    proj = a["cache"]["projection"]
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, tps in proj["decode_tok_s"].items():
        ax.plot(caps, tps, marker=".", label=POLICY_LABELS[name])
    ax.axhline(
        proj["inputs"]["measured_tok_s_no_cache"], color="grey", ls="--", label="measured today (no cache)"
    )
    ax.set(
        xlabel="cache size (experts)",
        ylabel="decode tokens/s",
        title="PROJECTED decode speed (simulated hits x measured times, not a measurement)",
    )
    ax.legend(fontsize=8)
    save(fig, "projected_tok_s")

    path = doc_path_for(store)
    text = format_markdown(record, json_path, charts)
    if path.exists():
        old = path.read_text(encoding="utf-8")
        if APPENDED_SECTIONS_MARKER in old:
            text += "\n" + APPENDED_SECTIONS_MARKER + old.split(APPENDED_SECTIONS_MARKER, 1)[1]
    path.write_text(text, encoding="utf-8")
    return path


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def format_markdown(record: dict, json_path: Path, charts: dict) -> str:
    a = record["analysis"]
    cats = sorted(a["prompts_by_category"])
    caps = a["cache"]["all"]["capacities"]
    idx = [caps.index(c) for c in (0, 48, 96, 192, 384, 720, 1440)]
    pol = a["cache"]["all"]["policies"]
    proj = a["cache"]["projection"]
    fate, loc, pop, reuse = a["fate"], a["temporal_locality"], a["popularity"], a["reuse_distance"]
    policy_header = "| cache (experts) | RAM | " + " | ".join(POLICY_LABELS[p] for p in pol) + " |"
    policy_rule = "|---|---|" + "---|" * len(pol)
    lines = [
        f"# Phase 3: how {record['model']['repo_id']} uses its experts",
        "",
        f"Generated by `python -m expertrelay.bench.phase3_analysis` at {record['timestamp']} "
        f"(commit {record['git_commit'][:7]}{', dirty tree' if record['git_dirty'] else ''}) from "
        f"`{json_path.relative_to(REPO_ROOT).as_posix()}`, computed from the traces in "
        f"`{record['config']['trace_dir']}`. Do not edit by hand.",
        "",
        f"Data: {a['prompts']} prompts "
        f"({', '.join(f'{c}: {n}' for c, n in a['prompts_by_category'].items())}), "
        f"{a['tokens']} tokens through the model, {a['decode_tokens']} of them generated (greedy). "
        f"Store revision `{record['model']['revision'][:7]}`, prompt format `{record['config']['prompt_format']}`.",
        "",
        "## 1. Popularity",
        "",
        f"Each layer has 60 experts. Its top 10% ({pop['all']['top10pct_experts']}) would get "
        f"{pop['all']['uniform_share']:.0%} of selections if routing were uniform. Measured share, "
        "averaged over layers:",
        "",
        "| | all | prefill | decode |",
        "|---|---|---|---|",
        f"| top 10% share | {_pct(pop['all']['top10pct_share_mean'])} | "
        f"{_pct(pop['prefill']['top10pct_share_mean'])} | {_pct(pop['decode']['top10pct_share_mean'])} |",
        "",
        "| category | " + " | ".join(cats) + " |",
        "|---|" + "---|" * len(cats),
        "| top 10% share | "
        + " | ".join(_pct(pop["by_category"][c]["top10pct_share_mean"]) for c in cats)
        + " |",
        "",
        f"![popularity]({charts['popularity']})",
        "",
        "## 2. Reuse distance",
        "",
        "Tokens until the same expert is used again at the same layer (decode selections):",
        "",
        "| | within 1 token | within 2 | within 8 | within 32 | first use in the sequence | median (reused) |",
        "|---|---|---|---|---|---|---|",
        *[
            f"| {name} | {_pct(r['within_1'])} | {_pct(r['within_2'])} | {_pct(r['within_8'])} | "
            f"{_pct(r['within_32'])} | {_pct(r['first_use_fraction'])} | {r['median_finite']} |"
            for name, r in [("all", reuse["all"]), *[(c, reuse["by_category"][c]) for c in cats]]
        ],
        "",
        f"![reuse distance]({charts['reuse_distance']})",
        "",
        "## 3. Temporal locality (same layer, previous token)",
        "",
        "| | shares at least one expert | fraction of the 4 shared | all 4 the same |",
        "|---|---|---|---|",
        *[
            f"| {name} | {_pct(x['at_least_one_shared'])} | {_pct(x['mean_shared_fraction'])} | "
            f"{_pct(x['all_shared'])} |"
            for name, x in [
                ("prefill", loc["prefill"]),
                ("decode", loc["decode"]),
                *[(c, loc["by_category"][c]) for c in cats],
            ]
        ],
        "",
        "## 4. Predictability: the Fate-style cross-layer guess",
        "",
        "Layer L+1's router applied to layer L's gate input (Fang et al., arXiv:2502.12224). "
        "Recall@4 = the fraction of the true top-4 that the guessed top-4 catches. The last column is the "
        "trivial alternative, guessing the previous token's experts at the same layer.",
        "",
        "| | recall@4 | 0 of 4 | 1 | 2 | 3 | 4 of 4 | previous-token guess |",
        "|---|---|---|---|---|---|---|---|",
        *[
            f"| {name} | {_pct(f['recall_at_k'])} | "
            + " | ".join(_pct(v) for v in f["hits_distribution"])
            + f" | {_pct(lc['mean_shared_fraction'])} |"
            for name, f, lc in [
                ("all", fate["all"], loc["all"]),
                ("prefill", fate["prefill"], loc["prefill"]),
                ("decode", fate["decode"], loc["decode"]),
                *[(c, fate["by_category"][c], loc["by_category"][c]) for c in cats],
            ]
        ],
        "",
        f"![predictors by layer]({charts['predictors_by_layer']})",
        "",
        "**Is its confidence trustworthy?** For each guessed expert: its predicted router probability, and how "
        "often it was actually among the chosen 4.",
        "",
        "| predicted probability | guesses | mean predicted | actually chosen |",
        "|---|---|---|---|",
        *[
            f"| {b['lo']:.2f}-{b['hi']:.2f} | {b['count']} | {_pct(b['mean_predicted_prob'])} | "
            f"{_pct(b['fraction_chosen'])} |"
            for b in fate["all"]["per_expert_confidence_bins"]
            if b["count"]
        ],
        "",
        "Recall@4 by the guess's total confidence (sum of its 4 predicted probabilities), lowest to highest "
        "quartile: " + ", ".join(_pct(x) for x in fate["all"]["recall_by_confidence_quartile"]) + ".",
        "",
        f"![confidence]({charts['fate_confidence']})",
        "",
        "## 5. Cache simulation",
        "",
        f"Replays what the runtime actually loads ({a['cache']['all']['accesses']:,} expert loads, "
        f"{a['cache']['all']['decode_accesses_per_token']:.0f} per generated token), with the cache warm across "
        f"prompts. Pinned policy: {a['cache']['pinned_fraction']:.0%} of the cache holds a fixed hot set chosen "
        "from OTHER prompts (the other half of the run for 'all'; the other categories, per category). "
        "Decode hit rate:",
        "",
        policy_header,
        policy_rule,
        *[
            f"| {caps[i]} | {proj['cache_ram_gb'][i]:.2f} GB | "
            + " | ".join(_pct(pol[p]["decode_hit_rate"][i]) for p in pol)
            + " |"
            for i in idx
        ],
        "",
        f"![hit rate]({charts['hit_rate_policies']})",
        "",
        f"![hit rate by category]({charts['hit_rate_by_category']})",
        "",
        "## 6. Projected decode speed: a projection, not a measurement",
        "",
        f"{proj['label']}. Inputs measured in Phase 2: {proj['inputs']['compute_s_per_token']:.2f} s compute "
        f"per token, {proj['inputs']['read_s_per_expert'] * 1000:.1f} ms per expert read, "
        f"{proj['inputs']['measured_tok_s_no_cache']:.2f} tok/s measured with no cache "
        f"({proj['inputs']['source']}).",
        "",
        policy_header,
        policy_rule,
        *[
            f"| {caps[i]} | {proj['cache_ram_gb'][i]:.2f} GB | "
            + " | ".join(f"{proj['decode_tok_s'][p][i]:.2f}" for p in pol)
            + " |"
            for i in idx
        ],
        "",
        "Even a perfect cache can't beat compute: at a 100% hit rate the projection is "
        f"{1 / proj['inputs']['compute_s_per_token']:.2f} tok/s, because compute is the bottleneck (Phase 2).",
        "",
        f"![projected tok/s]({charts['projected_tok_s']})",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--store-dir", type=Path, default=None)
    ap.add_argument(
        "--report", action="store_true", help="only regenerate charts and doc from the saved JSON"
    )
    args = ap.parse_args()

    rt = RuntimeConfig.load(args.config)
    store_dir = (args.store_dir or rt.store_dir).resolve()
    out = BENCHMARK_RESULTS_DIR / f"phase3_analysis_{store_dir.name}.json"
    if args.report:
        print(f"wrote {render(json.loads(out.read_text(encoding='utf-8'))[-1], out)}")
        return

    manifest = json.loads((store_dir / "store.json").read_text())
    trace_dir = trace_dir_for(store_dir)
    prompts = round_robin(json.loads(rt.prompts_file.read_text(encoding="utf-8"))["prompts"])
    traces = load_traces(trace_dir, [p["id"] for p in prompts])
    if not traces:
        raise SystemExit(f"no finished traces in {trace_dir}")
    print(f"analyzing {len(traces)}/{len(prompts)} finished prompts from {trace_dir}", flush=True)
    record = base_record(
        label="phase3 expert-usage analysis",
        seed=0,
        model={
            "store": store_dir.name,
            "repo_id": manifest["source"]["repo_id"],
            "revision": manifest["source"]["revision"],
        },
        config={
            "capacities": CAPACITIES,
            "pinned_fraction": PINNED_FRACTION,
            "trace_dir": trace_dir.relative_to(REPO_ROOT).as_posix(),
            "prompt_format": traces[0].header["prompt_format"],
            "prompts_total": len(prompts),
        },
        machine=collect_machine_profile(measure_disk=False),
    )
    record["analysis"] = analyze(traces, manifest["expert_layout"]["record_size"])
    record["peak_rss_mb"] = peak_process_rss_mb()
    append_benchmark_record(out, record)
    print(f"saved {out}; wrote {render(record, out)}")


if __name__ == "__main__":
    main()
