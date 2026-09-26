"""Phase 3.5: better expert prediction and caching, from recorded traces only.

    python -m expertrelay.bench.phase35_prediction            # base store
    python -m expertrelay.bench.phase35_prediction --report   # redo charts + doc section from saved JSON

Simulation only: no timing, no runtime change. Reads the finished traces
of one store (models/traces/<store>/), writes
benchmarks/results/phase35_prediction_<store>.json, charts under
docs/phase3/<store>/, and a section appended to the store's Phase 3 doc.

Held-out evaluation. Prompts are split WITHIN each category by their number:
odd-numbered prompts (en_01, en_03, ...) tune, even-numbered ones test.
Every weight, grid choice, calibration curve, reuse table and "best
combination" is picked on the tuning half; every reported number is from
the test half. (A split by position in the run order would not do this:
the run cycles through the 6 categories, so every other prompt means
whole categories on each side.)

What is predicted, and when:
  * layers 1..23: one layer ahead, from the Fate-style cross-layer gate
    logits the trace recorded (Z. Fang et al., "Fate: Fast Edge Inference
    of Mixture-of-Experts Models via Cross-Layer Gate", arXiv:2502.12224),
    optionally plus how recently each expert was picked at that layer.
  * layer 0 has no previous layer: a popularity prior plus recency. Its
    prediction for the next token is made during the current token's last
    layer.
  * decode tokens only. In prefill, a layer's picks for all prompt tokens
    come out of one call, so "recent tokens" aren't known ahead of it.
Two layers ahead can't use Fate's gate from these traces (that needs the
hidden state, which isn't recorded), so it's measured with transition
tables instead; see docs/limitations.md.

Projected tokens/s are a PROJECTION, not a measurement: Phase 2's measured
compute time, spread evenly over the 24 layers, plus simulated reads.
"""

from __future__ import annotations

import argparse
import json
import socket
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from expertrelay import REPO_ROOT
from expertrelay.bench.phase3_analysis import (
    doc_path_for,
    load_traces,
    phase2_speeds,
    upsert_doc_section,
)
from expertrelay.bench.phase3_traces import DEFAULT_CONFIG, round_robin, trace_dir_for
from expertrelay.benchmarking import append_benchmark_record, base_record, peak_process_rss_mb
from expertrelay.cache.predictive import Access, Predict, SimResult, overlapped_decode_seconds, simulate
from expertrelay.cache.simulator import simulate_belady
from expertrelay.manager.profile import collect_machine_profile, output_path_for
from expertrelay.memory_budget import enforce_ram_budget
from expertrelay.paths import BENCHMARK_RESULTS_DIR
from expertrelay.predictor.offline import (
    IsotonicCalibrator,
    ReuseModel,
    TransitionModel,
    expected_calibration_error,
    recency_scores,
    reliability_table,
    softmax,
    topk_hits,
    used_matrix,
)
from expertrelay.runtime.expert_trace import PHASE_DECODE, PHASE_PREFILL
from expertrelay.runtime.generate import RuntimeConfig

RECALL_KS = [4, 6, 8, 10, 12]
RECENCY_WINDOW = 8
BETA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
DECAY_GRID = [1.0, 0.8, 0.6]
CAPACITIES = [96, 144, 192, 288, 384, 480, 576]
PREFETCH_KS = [0, 4, 6, 8]
POLICIES = ["lru", "predictive"]
# The traces and everything derived from them stay in RAM. Several arrays
# are [tokens, 24, 60] float32 (~35 MB per half) and a few are held at once;
# this is the ceiling, checked before loading, with the Chat trace run and
# a download sharing the machine.
MAX_RAM_GB = 1.0
# Estimated peak working memory per byte of trace file. Base store: 612 MB
# measured peak for 92 MB of traces; each run's measured peak is in its JSON.
RAM_PER_TRACE_BYTE = 7


@dataclass
class Seq:
    """One prompt's routing, dense: [positions, layers, ...]."""

    prompt_id: str
    category: str
    prompt_len: int
    experts: np.ndarray  # [T, L, K] int64
    used: np.ndarray  # [T, L, E] bool
    fate: np.ndarray  # [T_decode, L, E] float32; layer 0 all zero

    @property
    def decode_slice(self) -> slice:
        return slice(self.prompt_len, len(self.experts))


def to_seq(trace) -> Seq:
    h, rec = trace.header, trace.rec
    num_layers, num_experts, k = h["num_layers"], h["num_experts"], h["top_k"]
    t = len(rec) // num_layers
    pos = rec["position"].reshape(t, num_layers)
    lay = rec["layer"].reshape(t, num_layers)
    if not ((pos == np.arange(t)[:, None]).all() and (lay == np.arange(num_layers)[None, :]).all()):
        raise ValueError(f"{h['prompt_id']}: records are not ordered by position, then layer")
    phase = rec["phase"].reshape(t, num_layers)[:, 0]
    prompt_len = int((phase == PHASE_PREFILL).sum())
    if not (phase[prompt_len:] == PHASE_DECODE).all():
        raise ValueError(f"{h['prompt_id']}: prefill positions must come first")
    experts = rec["experts"].reshape(t, num_layers, k).astype(np.int64)
    used = np.stack([used_matrix(experts[:, layer], num_experts) for layer in range(num_layers)], axis=1)
    fate = rec["fate_logits"].reshape(t, num_layers, num_experts)[prompt_len:].astype(np.float32)
    return Seq(h["prompt_id"], h["category"], prompt_len, experts, used, fate)


def split_halves(seqs: list[Seq]) -> tuple[list[Seq], list[Seq]]:
    """(tune, test): odd-numbered prompts of every category tune."""
    tune = [s for s in seqs if int(s.prompt_id.rsplit("_", 1)[1]) % 2 == 1]
    test = [s for s in seqs if int(s.prompt_id.rsplit("_", 1)[1]) % 2 == 0]
    return tune, test


# ---------------------------------------------------------------------------
# Features, stacked over a half's decode tokens: [N, L, E]
# ---------------------------------------------------------------------------


def recency(seqs: list[Seq], decay: float) -> np.ndarray:
    out = []
    for s in seqs:
        r = np.stack(
            [recency_scores(s.used[:, layer], RECENCY_WINDOW, decay) for layer in range(s.used.shape[1])],
            axis=1,
        )
        out.append(r[s.decode_slice])
    return np.concatenate(out)


def stack(seqs: list[Seq], name: str) -> np.ndarray:
    if name == "fate":
        return np.concatenate([s.fate for s in seqs])
    return np.concatenate([getattr(s, name)[s.decode_slice] for s in seqs])


def layer0_prior(seqs: list[Seq]) -> np.ndarray:
    """log popularity of each expert at layer 0 over decode tokens, smoothed."""
    counts = stack(seqs, "used")[:, 0].sum(axis=0).astype(np.float64) + 1.0
    return np.log(counts / counts.sum()).astype(np.float32)


def recall_curve(scores: np.ndarray, truth: np.ndarray) -> dict:
    top = truth.shape[1]
    return {str(k): float(topk_hits(scores, truth, k).mean() / top) for k in RECALL_KS}


def flat(x: np.ndarray, layers: slice) -> np.ndarray:
    """[N, L, ...] -> [N * len(layers), ...]"""
    x = x[:, layers]
    return x.reshape(-1, *x.shape[2:])


# ---------------------------------------------------------------------------
# 1. Prediction
# ---------------------------------------------------------------------------


def tune_combined(tune: list[Seq]) -> dict:
    """Pick decay and weight for fate + beta * recency (layers 1..), and for
    prior + beta * recency (layer 0), by recall@4 on the tuning half."""
    truth = stack(tune, "experts")
    fate = stack(tune, "fate")
    prior = layer0_prior(tune)
    best = {"recall_at_4": -1.0}
    best0 = {"recall_at_4": -1.0}
    grid = []
    for decay in DECAY_GRID:
        r = recency(tune, decay)
        for beta in BETA_GRID:
            s = flat(fate + beta * r, slice(1, None))
            rec = float(topk_hits(s, flat(truth, slice(1, None)), 4).mean() / 4)
            grid.append({"decay": decay, "beta": beta, "recall_at_4": rec})
            if rec > best["recall_at_4"]:
                best = {"decay": decay, "beta": beta, "recall_at_4": rec}
            s0 = prior[None, :] + beta * r[:, 0]
            rec0 = float(topk_hits(s0, truth[:, 0], 4).mean() / 4)
            if rec0 > best0["recall_at_4"]:
                best0 = {"decay": decay, "beta": beta, "recall_at_4": rec0}
        del r
    return {"layers_1_plus": best, "layer_0": best0, "grid": grid, "layer0_prior": prior.tolist()}


def combined_scores(seqs: list[Seq], tuned: dict) -> np.ndarray:
    """[N, L, E] scores of the tuned combined predictor (layer 0: prior + recency)."""
    b, b0 = tuned["layers_1_plus"], tuned["layer_0"]
    s = stack(seqs, "fate") + b["beta"] * recency(seqs, b["decay"])
    s[:, 0] = (
        np.asarray(tuned["layer0_prior"], dtype=np.float32)[None, :]
        + b0["beta"] * recency(seqs, b0["decay"])[:, 0]
    )
    return s


def prediction_study(test: list[Seq], tuned: dict) -> dict:
    truth = stack(test, "experts")
    fate = stack(test, "fate")
    combined = combined_scores(test, tuned)
    prior = np.asarray(tuned["layer0_prior"], dtype=np.float32)
    recent = recency(test, tuned["layers_1_plus"]["decay"])
    later = slice(1, None)
    # recency alone ties often (integer-valued with decay 1); ties go to the lower
    # expert id (predictor.offline.top_k), which is deterministic but arbitrary
    return {
        "layers_1_plus": {
            "fate": recall_curve(flat(fate, later), flat(truth, later)),
            "fate_plus_recency": recall_curve(flat(combined, later), flat(truth, later)),
            "recency_only": recall_curve(flat(recent, later), flat(truth, later)),
        },
        "layer_0": {
            "prior_only": recall_curve(
                np.broadcast_to(prior, truth[:, 0].shape[:1] + prior.shape), truth[:, 0]
            ),
            "prior_plus_recency": recall_curve(combined[:, 0], truth[:, 0]),
        },
        "tokens": int(len(truth)),
    }


def two_ahead_study(tune: list[Seq], test: list[Seq], tuned: dict) -> dict:
    """Targets are layers 2..: every method below is scored on the same records."""
    num_layers, num_experts = tune[0].used.shape[1], tune[0].used.shape[2]
    used_tune = stack(tune, "used")
    one = [TransitionModel(num_experts) for _ in range(num_layers)]  # one[l]: l-1 -> l
    two = [TransitionModel(num_experts) for _ in range(num_layers)]  # two[l]: l-2 -> l
    for layer in range(1, num_layers):
        one[layer].add(used_tune[:, layer - 1], used_tune[:, layer])
    for layer in range(2, num_layers):
        two[layer].add(used_tune[:, layer - 2], used_tune[:, layer])

    def scores(seqs: list[Seq], method: str) -> np.ndarray:
        used = stack(seqs, "used").astype(np.float32)
        fate = stack(seqs, "fate")
        out = np.zeros(used.shape, dtype=np.float32)
        for layer in range(2, num_layers):
            if method == "transition_1_ahead":
                out[:, layer] = one[layer].score(used[:, layer - 1])
            elif method == "transition_2_ahead":
                out[:, layer] = two[layer].score(used[:, layer - 2])
            else:  # fate_chain_2_ahead: Fate's guess for layer-1 (made at layer-2), then one step of transitions
                out[:, layer] = one[layer].score(softmax(fate[:, layer - 1]))
        return out

    methods = ["transition_1_ahead", "transition_2_ahead", "fate_chain_2_ahead"]
    targets = slice(2, None)
    truth_tune = flat(stack(tune, "experts"), targets)
    best_beta = {}
    for m in methods:
        base = scores(tune, m)
        best = (-1.0, 0.0, 1.0)
        for decay in DECAY_GRID:
            r = recency(tune, decay)
            for beta in BETA_GRID:
                rec = float(topk_hits(flat(base + beta * r, targets), truth_tune, 4).mean() / 4)
                if rec > best[0]:
                    best = (rec, beta, decay)
        best_beta[m] = {"beta": best[1], "decay": best[2], "tune_recall_at_4": best[0]}

    truth = flat(stack(test, "experts"), targets)
    out = {
        "targets": "layers 2..23, decode tokens",
        "fate_plus_recency_1_ahead": recall_curve(flat(combined_scores(test, tuned), targets), truth),
    }
    for m in methods:
        b = best_beta[m]
        out[m] = recall_curve(flat(scores(test, m) + b["beta"] * recency(test, b["decay"]), targets), truth)
    out["tuned"] = best_beta
    return out


def calibration_study(tune: list[Seq], test: list[Seq], tuned: dict) -> tuple[dict, dict]:
    """Calibrators (for the cache) and the before/after numbers on the test half.

    Label = the expert was among the K picked. "Before" is the Fate router
    softmax read as that probability, which it isn't (it's a distribution
    over one pick, not K), hence the need to calibrate."""
    later = slice(1, None)

    def pairs(seqs: list[Seq]):
        used = stack(seqs, "used")
        fate_p = softmax(stack(seqs, "fate"))
        comb_p = softmax(combined_scores(seqs, tuned))
        return used, fate_p, comb_p

    used_t, fate_t, comb_t = pairs(tune)
    cal_fate = IsotonicCalibrator().fit(flat(fate_t, later), flat(used_t, later))
    cal_comb = IsotonicCalibrator().fit(flat(comb_t, later), flat(used_t, later))
    cal_0 = IsotonicCalibrator().fit(comb_t[:, 0], used_t[:, 0])
    del used_t, fate_t, comb_t

    used, fate_p, comb_p = pairs(test)
    y = flat(used, later).reshape(-1)
    variants = {
        "fate_raw_softmax": flat(fate_p, later).reshape(-1),
        "fate_isotonic": cal_fate.predict(flat(fate_p, later)).reshape(-1),
        "fate_plus_recency_isotonic": cal_comb.predict(flat(comb_p, later)).reshape(-1),
    }
    out = {
        name: {
            "ece": expected_calibration_error(p, y),
            "brier": float(np.mean((p - y) ** 2)),
            "reliability": reliability_table(p, y),
        }
        for name, p in variants.items()
    }
    p0 = cal_0.predict(comb_p[:, 0]).reshape(-1)
    y0 = used[:, 0].reshape(-1)
    out["layer_0_prior_plus_recency_isotonic"] = {
        "ece": expected_calibration_error(p0, y0),
        "brier": float(np.mean((p0 - y0) ** 2)),
    }
    out["pairs"] = int(len(y))
    return out, {"layers_1_plus": cal_comb, "layer_0": cal_0}


# ---------------------------------------------------------------------------
# 2. Cache simulation
# ---------------------------------------------------------------------------


def predicted_probs(seqs: list[Seq], tuned: dict, cals: dict) -> list[np.ndarray]:
    """Per sequence, [T_decode, L, E]: calibrated P(picked) for each layer visit.
    The isotonic map is a step function, so many experts share a value; a
    1e-7 * softmax(score) term keeps the prefetch ranking the predictor's own
    order without visibly changing any probability."""
    out = []
    for s in seqs:
        sc = combined_scores([s], tuned)
        sm = softmax(sc)
        p = np.empty_like(sm)
        p[:, 1:] = cals["layers_1_plus"].predict(sm[:, 1:])
        p[:, 0] = cals["layer_0"].predict(sm[:, 0])
        out.append(p + np.float32(1e-7) * sm)
    return out


def build_events(seqs: list[Seq], probs: list[np.ndarray]) -> list[Access | Predict]:
    """Prompts back to back (cache warm across them), global positions."""
    events: list[Access | Predict] = []
    offset = 0
    for s, p in zip(seqs, probs, strict=True):
        num_layers = s.experts.shape[1]
        pl = s.prompt_len
        prefill_pos = offset + np.arange(pl)
        for layer in range(num_layers):
            ex = s.experts[:pl, layer]
            events.append(Access(layer, np.unique(ex), False, prefill_pos, ex))
        n_decode = len(s.experts) - pl
        if n_decode:
            events.append(Predict(0, p[0, 0]))
        for i in range(n_decode):
            pos = np.array([offset + pl + i])
            for layer in range(num_layers):
                ex = s.experts[pl + i, layer]
                events.append(Access(layer, np.unique(ex), True, pos, ex[None, :]))
                if layer + 1 < num_layers:
                    events.append(Predict(layer + 1, p[i, layer + 1]))
                elif i + 1 < n_decode:
                    events.append(Predict(0, p[i + 1, 0]))
        offset += len(s.experts)
    return events


def belady_result(events: list[Access | Predict], num_experts: int, capacity: int) -> SimResult:
    acc = [e for e in events if isinstance(e, Access)]
    stream = np.concatenate([e.layer * num_experts + e.experts for e in acc]).astype(np.int64)
    hits = simulate_belady(stream, capacity)
    sizes = np.array([len(e.experts) for e in acc])
    per_event = np.add.reduceat(hits.astype(np.int64), np.concatenate([[0], np.cumsum(sizes)[:-1]]))
    return SimResult(
        layer=np.array([e.layer for e in acc]),
        decode=np.array([e.decode for e in acc]),
        accesses=sizes,
        hits=per_event,
        prefetch_reads=np.zeros(len(acc), dtype=np.int64),
    )


def summarize(res: SimResult, compute_s_per_layer: float, read_times: dict) -> dict:
    tokens = int(np.sum(res.decode & (res.layer == 0)))
    out = {
        "decode_hit_rate": res.decode_hit_rate(),
        "demand_reads_per_token": float((res.accesses - res.hits)[res.decode].sum() / tokens),
        "prefetch_reads_per_token": float(res.prefetch_reads[res.decode].sum() / tokens),
        "prefetches_used": res.prefetches_used,
        "prefetches_wasted": res.prefetches_wasted,
        "projected_tok_s": {},
    }
    for name, t in read_times.items():
        seconds, n = overlapped_decode_seconds(res, compute_s_per_layer, t)
        out["projected_tok_s"][name] = n / seconds
    return out


def cache_study(
    halves: dict[str, tuple[list[Seq], list[np.ndarray]]],
    reuse: ReuseModel,
    num_layers: int,
    num_experts: int,
    compute_s_per_layer: float,
    read_times: dict,
) -> dict:
    out: dict = {}
    for half, (seqs, probs) in halves.items():
        events = build_events(seqs, probs)
        rows = {}
        for cap in CAPACITIES:
            row = {
                "belady": summarize(belady_result(events, num_experts, cap), compute_s_per_layer, read_times)
            }
            for policy in POLICIES:
                for k in PREFETCH_KS:
                    res = simulate(
                        events,
                        num_layers=num_layers,
                        num_experts=num_experts,
                        capacity=cap,
                        policy=policy,
                        prefetch_k=k,
                        reuse=reuse,
                    )
                    row[f"{policy}+k{k}"] = summarize(res, compute_s_per_layer, read_times)
            lru, bel = row["lru+k0"]["decode_hit_rate"], row["belady"]["decode_hit_rate"]
            for v in row.values():
                v["gap_closed"] = (v["decode_hit_rate"] - lru) / (bel - lru) if bel > lru else None
            rows[str(cap)] = row
            print(f"  {half} capacity {cap}: done", flush=True)
        out[half] = rows
        del events
    return out


def pick_best(cache: dict, read_times: dict) -> dict:
    """Per capacity and read time: the config with the best PROJECTED tok/s on
    the TUNING half (Belady excluded: it needs the future), reported on the
    test half."""
    best: dict = {}
    for name in read_times:
        best[name] = {}
        for cap, row in cache["tune"].items():
            cands = {c: v for c, v in row.items() if c != "belady"}
            choice = max(cands, key=lambda c: cands[c]["projected_tok_s"][name])
            best[name][cap] = {"config": choice, "test": cache["test"][cap][choice]}
    return best


# ---------------------------------------------------------------------------
# Whole study -> JSON
# ---------------------------------------------------------------------------


def raw_ssd_read_s() -> tuple[float, str]:
    """Median time for one uncached random 8.65 MB read (one expert's weights)
    from the saved machine profile of THIS machine."""
    path = output_path_for(socket.gethostname())
    record = json.loads(path.read_text(encoding="utf-8"))
    record = record[-1] if isinstance(record, list) else record
    r = next(
        x
        for x in record["machine"]["disk_read"]["results"]
        if x["pattern"] == "random" and x["chunk_bytes"] < 10_000_000
    )
    seconds = statistics.median(r["seconds_per_run"]) / r["reads_per_run"]
    return seconds, f"{path.relative_to(REPO_ROOT).as_posix()} @ {record['git_commit'][:7]}"


def analyze(seqs: list[Seq], record_bytes: int) -> dict:
    tune, test = split_halves(seqs)
    num_layers, num_experts = seqs[0].used.shape[1], seqs[0].used.shape[2]
    print(f"tune {len(tune)} prompts, test {len(test)} prompts", flush=True)

    tuned = tune_combined(tune)
    print(f"tuned: {tuned['layers_1_plus']} / layer 0 {tuned['layer_0']}", flush=True)
    prediction = prediction_study(test, tuned)
    two_ahead = two_ahead_study(tune, test, tuned)
    calibration, cals = calibration_study(tune, test, tuned)
    print("prediction, two-ahead and calibration done", flush=True)

    reuse = ReuseModel()
    for s in tune:
        for layer in range(num_layers):
            reuse.add(s.used[:, layer], s.prompt_len)
    reuse.fit()

    speeds = phase2_speeds(record_bytes)
    raw_s, raw_src = raw_ssd_read_s()
    read_times = {"phase2_runtime": speeds["read_s_per_expert"], "raw_ssd": raw_s}
    compute_s_per_layer = speeds["compute_s_per_token"] / num_layers
    halves = {
        "tune": (tune, predicted_probs(tune, tuned, cals)),
        "test": (test, predicted_probs(test, tuned, cals)),
    }
    cache = cache_study(halves, reuse, num_layers, num_experts, compute_s_per_layer, read_times)

    return {
        "split": {
            "rule": "within each category: odd-numbered prompts tune, even-numbered prompts test",
            "tune_prompts": [s.prompt_id for s in tune],
            "test_prompts": [s.prompt_id for s in test],
        },
        "tuned": {k: v for k, v in tuned.items() if k != "layer0_prior"},
        "prediction": prediction,
        "two_ahead": two_ahead,
        "calibration": calibration,
        "reuse_table": reuse.table.tolist(),
        "cache": {
            "capacities": CAPACITIES,
            "cache_ram_gb": {str(c): c * record_bytes / 1e9 for c in CAPACITIES},
            "prefetch_ks": PREFETCH_KS,
            "policies": POLICIES,
            **cache,
        },
        "projection": {
            "label": "PROJECTION from simulated reads + Phase 2 measured compute; not a measurement",
            "compute_s_per_token": speeds["compute_s_per_token"],
            "compute_s_per_layer": compute_s_per_layer,
            "read_s_per_expert": read_times,
            "read_sources": {"phase2_runtime": speeds["source"], "raw_ssd": raw_src},
            "measured_tok_s_no_cache": speeds["measured_tok_s_no_cache"],
            "model": "per decode layer: compute window + demand misses x read time + prefetch spill "
            "beyond the previous layer's window; one read at a time",
        },
        "best": pick_best(cache, read_times),
    }


# ---------------------------------------------------------------------------
# Charts and doc section, from the saved JSON only
# ---------------------------------------------------------------------------

CONFIG_LABELS = {
    "lru+k0": "LRU",
    "predictive+k0": "prediction-aware eviction",
    "belady": "Belady (optimal demand cache, needs the future)",
}


def config_label(c: str) -> str:
    if c in CONFIG_LABELS:
        return CONFIG_LABELS[c]
    policy, k = c.split("+k")
    base = "LRU" if policy == "lru" else "prediction-aware"
    return f"{base} + prefetch top-{k}"


def render(record: dict) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a, store = record["analysis"], record["model"]["store"]
    chart_dir = REPO_ROOT / "docs" / "phase3" / store
    chart_dir.mkdir(parents=True, exist_ok=True)
    charts = {}
    caps = a["cache"]["capacities"]
    test = a["cache"]["test"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for c in ["lru+k0", "predictive+k0", "lru+k8", "predictive+k4", "predictive+k8", "belady"]:
        ax.plot(
            caps, [test[str(cap)][c]["decode_hit_rate"] for cap in caps], marker=".", label=config_label(c)
        )
    ax.set(xlabel="cache size (experts)", ylabel="decode hit rate (test prompts)", title="Hit rate by policy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(chart_dir / "phase35_hit_rate.png", dpi=110)
    plt.close(fig)
    charts["hit_rate"] = f"phase3/{store}/phase35_hit_rate.png"

    fig, ax = plt.subplots(figsize=(5, 5))
    for name, label in [
        ("fate_raw_softmax", "Fate router softmax (raw)"),
        ("fate_plus_recency_isotonic", "Fate + recency, isotonic"),
    ]:
        rel = a["calibration"][name]["reliability"]
        ax.plot([b["mean_predicted"] for b in rel], [b["observed"] for b in rel], marker="o", label=label)
    ax.plot([0, 1], [0, 1], color="grey", ls="--", label="perfect")
    ax.set(xlabel="predicted P(picked)", ylabel="observed rate (test prompts)", title="Calibration")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(chart_dir / "phase35_calibration.png", dpi=110)
    plt.close(fig)
    charts["calibration"] = f"phase3/{store}/phase35_calibration.png"

    path = doc_path_for(store)
    upsert_doc_section(path, "phase35", format_markdown(record, charts))
    return path


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def format_markdown(record: dict, charts: dict) -> str:
    a = record["analysis"]
    pred, two, cal, cache = a["prediction"], a["two_ahead"], a["calibration"], a["cache"]
    proj, best = a["projection"], a["best"]
    ks = [str(k) for k in RECALL_KS]
    t1, t0 = a["tuned"]["layers_1_plus"], a["tuned"]["layer_0"]
    ram = cache["cache_ram_gb"]
    lines = [
        "## 7. Phase 3.5: better prediction and caching (simulation only)",
        "",
        f"Generated by `python -m expertrelay.bench.phase35_prediction` at {record['timestamp']} "
        f"(commit {(record['git_commit'] or 'unknown')[:7]}) from "
        f"`benchmarks/results/phase35_prediction_{record['model']['store']}.json`. Do not edit by hand.",
        "",
        f"**Held out.** {a['split']['rule']}: {len(a['split']['tune_prompts'])} tuning prompts, "
        f'{len(a["split"]["test_prompts"])} test prompts. Every weight, calibration curve and "best" '
        "choice comes from the tuning prompts; every number below is from the test prompts "
        f"({pred['tokens']} generated tokens).",
        "",
        "### Recall when guessing more experts",
        "",
        "Share of the router's true 4 experts caught by the top-k guess, generated tokens, layers 1-23 "
        "(one layer ahead):",
        "",
        "| predictor | " + " | ".join(f"k={k}" for k in ks) + " |",
        "|---|" + "---|" * len(ks),
    ]
    names = {
        "fate": "Fate-style (cross-layer gate)",
        "fate_plus_recency": f"Fate + recency (weight {t1['beta']}, decay {t1['decay']})",
        "recency_only": "recency only (last 8 tokens)",
    }
    for key, label in names.items():
        lines.append(f"| {label} | " + " | ".join(_pct(pred["layers_1_plus"][key][k]) for k in ks) + " |")
    lines += [
        "",
        "Layer 0 has no layer before it, so its guess (made during the previous token's last layer) is "
        f"popularity plus recency (weight {t0['beta']}, decay {t0['decay']}):",
        "",
        "| layer 0 | " + " | ".join(f"k={k}" for k in ks) + " |",
        "|---|" + "---|" * len(ks),
        "| popularity only | " + " | ".join(_pct(pred["layer_0"]["prior_only"][k]) for k in ks) + " |",
        "| popularity + recency | "
        + " | ".join(_pct(pred["layer_0"]["prior_plus_recency"][k]) for k in ks)
        + " |",
        "",
        "### Two layers ahead",
        "",
        "Fate's gate can't be applied two layers ahead from these traces: that needs the hidden state, "
        "which the traces don't record. So the drop is measured with expert-to-expert transition tables "
        "(learned on the tuning prompts), each plus tuned recency. Same targets for every row (layers 2-23):",
        "",
        "| predictor | " + " | ".join(f"k={k}" for k in ks) + " |",
        "|---|" + "---|" * len(ks),
    ]
    two_names = {
        "fate_plus_recency_1_ahead": "1 ahead: Fate + recency",
        "transition_1_ahead": "1 ahead: transitions from the previous layer's picks",
        "transition_2_ahead": "2 ahead: transitions from the picks two layers back",
        "fate_chain_2_ahead": "2 ahead: Fate's guess for the next layer, then one transition step",
    }
    for key, label in two_names.items():
        lines.append(f"| {label} | " + " | ".join(_pct(two[key][k]) for k in ks) + " |")
    lines += [
        "",
        "### Calibrated confidence",
        "",
        "P(expert is among the 4 picked), over all 60 experts of every layer-1-23 visit "
        f"({cal['pairs']:,} test pairs). ECE: 20 equal-width bins. Lower is better for both columns.",
        "",
        "| probability | ECE | Brier |",
        "|---|---|---|",
    ]
    cal_names = {
        "fate_raw_softmax": "Fate router softmax, as is",
        "fate_isotonic": "Fate, isotonic-calibrated",
        "fate_plus_recency_isotonic": "Fate + recency, isotonic-calibrated",
    }
    for key, label in cal_names.items():
        lines.append(f"| {label} | {cal[key]['ece']:.4f} | {cal[key]['brier']:.4f} |")
    l0 = cal["layer_0_prior_plus_recency_isotonic"]
    lines += [
        f"| layer 0: popularity + recency, isotonic | {l0['ece']:.4f} | {l0['brier']:.4f} |",
        "",
        f"![calibration]({charts['calibration']})",
        "",
        "### Cache policies",
        "",
        "Test prompts replayed back to back, cache warm across them, decode hit rate. "
        '"Prediction-aware" evicts the expert least likely to be picked at its layer\'s next visit '
        "(the calibrated prediction for the next layer; a reuse table, from the tuning prompts, for the "
        'others). "Prefetch top-k" loads the k most likely experts for the next layer one layer early. '
        "Gap closed = share of the LRU-to-Belady gap; above 100% is possible with prefetching, "
        "because Belady is only optimal among caches that load on demand.",
        "",
    ]
    configs = ["lru+k0"] + [f"{p}+k{k}" for p in POLICIES for k in PREFETCH_KS if f"{p}+k{k}" != "lru+k0"]
    configs.append("belady")
    lines.append(
        "| policy | " + " | ".join(f"{c} ({ram[str(c)]:.2f} GB)" for c in cache["capacities"]) + " |"
    )
    lines.append("|---|" + "---|" * len(cache["capacities"]))
    for c in configs:
        cells = []
        for cap in cache["capacities"]:
            v = cache["test"][str(cap)][c]
            gap = "" if c in ("lru+k0", "belady") else f" ({_pct(v['gap_closed'])})"
            cells.append(f"{_pct(v['decode_hit_rate'])}{gap}")
        lines.append(f"| {config_label(c)} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Reads per generated token (demand + prefetch), test prompts:",
        "",
        "| policy | " + " | ".join(f"{c}" for c in cache["capacities"]) + " |",
        "|---|" + "---|" * len(cache["capacities"]),
    ]
    for c in configs:
        cells = []
        for cap in cache["capacities"]:
            v = cache["test"][str(cap)][c]
            cells.append(f"{v['demand_reads_per_token']:.1f} + {v['prefetch_reads_per_token']:.1f}")
        lines.append(f"| {config_label(c)} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"![hit rate]({charts['hit_rate']})",
        "",
        "### Projected decode speed: a projection, not a measurement",
        "",
        f"{proj['label']}. Compute: {proj['compute_s_per_token']:.3f} s per token (Phase 2), spread evenly "
        f"over the layers ({proj['compute_s_per_layer'] * 1000:.1f} ms each). One read at a time; a "
        "prefetch for the next layer is hidden behind the current layer's compute, and any part that "
        "doesn't fit delays the next layer, as do demand misses. The whole layer's compute is assumed "
        "available to hide the prefetch; in the real forward pass the prediction is ready only after "
        "that layer's attention, so this is optimistic about the window. Two read times:",
        "",
        f"- **Phase 2 runtime**: {proj['read_s_per_expert']['phase2_runtime'] * 1000:.2f} ms per expert, "
        f"what the runtime achieved end to end ({proj['read_sources']['phase2_runtime']}).",
        f"- **raw SSD**: {proj['read_s_per_expert']['raw_ssd'] * 1000:.2f} ms per expert, the uncached "
        f"random-read benchmark ({proj['read_sources']['raw_ssd']}); the runtime doesn't reach this today.",
        "",
        f"Measured today, no cache: {proj['measured_tok_s_no_cache']:.2f} tok/s. Compute-only ceiling: "
        f"{1 / proj['compute_s_per_token']:.2f} tok/s.",
        "",
        "| policy | read time | " + " | ".join(f"{c}" for c in cache["capacities"]) + " |",
        "|---|---|" + "---|" * len(cache["capacities"]),
    ]
    for c in configs:
        for rt in proj["read_s_per_expert"]:
            cells = [
                f"{cache['test'][str(cap)][c]['projected_tok_s'][rt]:.2f}" for cap in cache["capacities"]
            ]
            lines.append(f"| {config_label(c)} | {rt.replace('_', ' ')} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Belady here has no prefetch, so its reads aren't overlapped.",
        "",
        "**Best combination per cache size** (chosen by projected tok/s on the TUNING prompts, "
        "shown on the test prompts):",
        "",
        "| read time | cache | chosen | hit rate | reads/token (demand + prefetch) | projected tok/s |",
        "|---|---|---|---|---|---|",
    ]
    for rt, per_cap in best.items():
        for cap, b in per_cap.items():
            v = b["test"]
            lines.append(
                f"| {rt.replace('_', ' ')} | {cap} ({ram[cap]:.2f} GB) | {config_label(b['config'])} | "
                f"{_pct(v['decode_hit_rate'])} | {v['demand_reads_per_token']:.1f} + "
                f"{v['prefetch_reads_per_token']:.1f} | {v['projected_tok_s'][rt]:.2f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--store-dir", type=Path, default=None)
    ap.add_argument("--report", action="store_true", help="only regenerate charts and doc section")
    args = ap.parse_args()

    rt = RuntimeConfig.load(args.config)
    store_dir = (args.store_dir or rt.store_dir).resolve()
    out = BENCHMARK_RESULTS_DIR / f"phase35_prediction_{store_dir.name}.json"
    if args.report:
        print(f"wrote {render(json.loads(out.read_text(encoding='utf-8'))[-1])}")
        return

    manifest = json.loads((store_dir / "store.json").read_text())
    trace_dir = trace_dir_for(store_dir)
    prompts = round_robin(json.loads(rt.prompts_file.read_text(encoding="utf-8"))["prompts"])
    trace_bytes = sum(
        (trace_dir / f"{p['id']}.ert").stat().st_size
        for p in prompts
        if (trace_dir / f"{p['id']}.done.json").exists()
    )
    enforce_ram_budget(
        trace_bytes * RAM_PER_TRACE_BYTE, MAX_RAM_GB, "phase 3.5 analysis (traces + derived arrays)"
    )
    seqs = [to_seq(t) for t in load_traces(trace_dir, [p["id"] for p in prompts])]
    if not seqs:
        raise SystemExit(f"no finished traces in {trace_dir}")
    print(f"{len(seqs)} finished prompts from {trace_dir}", flush=True)
    record = base_record(
        label="phase3.5 prediction and caching study (simulation)",
        seed=0,
        model={
            "store": store_dir.name,
            "repo_id": manifest["source"]["repo_id"],
            "revision": manifest["source"]["revision"],
        },
        config={
            "recall_ks": RECALL_KS,
            "recency_window": RECENCY_WINDOW,
            "beta_grid": BETA_GRID,
            "decay_grid": DECAY_GRID,
            "capacities": CAPACITIES,
            "prefetch_ks": PREFETCH_KS,
            "policies": POLICIES,
            "trace_dir": trace_dir.relative_to(REPO_ROOT).as_posix(),
            "max_ram_gb": MAX_RAM_GB,
        },
        machine=collect_machine_profile(measure_disk=False),
    )
    record["analysis"] = analyze(seqs, manifest["expert_layout"]["record_size"])
    record["peak_rss_mb"] = peak_process_rss_mb()
    append_benchmark_record(out, record)
    print(f"saved {out}; wrote {render(record)}")


if __name__ == "__main__":
    main()
