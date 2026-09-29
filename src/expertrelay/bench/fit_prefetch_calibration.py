"""Fit the prefetcher's confidence calibration from recorded traces.

    python -m expertrelay.bench.fit_prefetch_calibration            # base store

Offline, from finished traces only. Fits two maps from the raw Fate
router softmax probability of one expert to P(it is among the experts
picked): the Phase 3.5 single isotonic map, and a rank-aware one (one map
per rank of the guess, predictor.offline.RankedIsotonicCalibrator), on the
TUNING prompts (odd-numbered in every category, as in
bench/phase35_prediction.py), and reports on the TEST prompts what a
confidence threshold would cost: for the top-k guess, how many prefetches
each threshold drops and how many of those would have been used.

Writes benchmarks/results/prefetch_calibration_<store>.json. The runtime
reads the calibrator from the last record of that file (configs point to
it, predictor.prefetch_policy.load_calibrator).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from expertrelay import REPO_ROOT
from expertrelay.bench.phase3_analysis import load_traces
from expertrelay.bench.phase3_traces import DEFAULT_CONFIG, round_robin, trace_dir_for
from expertrelay.bench.phase35_prediction import flat, split_halves, stack, to_seq
from expertrelay.benchmarking import append_benchmark_record, base_record, peak_process_rss_mb
from expertrelay.manager.profile import collect_machine_profile
from expertrelay.memory_budget import enforce_ram_budget
from expertrelay.paths import BENCHMARK_RESULTS_DIR
from expertrelay.predictor.offline import (
    IsotonicCalibrator,
    RankedIsotonicCalibrator,
    expected_calibration_error,
    softmax,
    top_k,
)
from expertrelay.runtime.generate import RuntimeConfig

PREFETCH_K = 8
MAX_RANK = 12
THRESHOLDS = [0.01, 0.02, 0.05, 0.1, 0.2]
MAX_RAM_GB = 1.0
RAM_PER_TRACE_BYTE = 7  # as bench/phase35_prediction.py: same arrays


def _threshold_rows(cal_top: np.ndarray, used_top: np.ndarray, total_used: int) -> list[dict]:
    rows = []
    for t in THRESHOLDS:
        drop = cal_top < t
        rows.append(
            {
                "min_probability": t,
                "prefetches_dropped": float(drop.mean()),
                "dropped_that_would_be_used": float(used_top[drop].mean()) if drop.any() else None,
                "recall_kept": float((used_top & ~drop).sum() / total_used),
            }
        )
    return rows


def fit(tune, test) -> dict:
    """Both maps fitted on the tuning prompts; everything reported is on the
    test prompts. The runtime uses the rank-aware one (load_calibrator)."""
    later = slice(1, None)  # layer 0 has no previous layer, so no Fate guess
    probs_t = flat(softmax(stack(tune, "fate")), later)
    used_t = flat(stack(tune, "used"), later)
    cal = IsotonicCalibrator().fit(probs_t, used_t)
    ranked = RankedIsotonicCalibrator(max_rank=MAX_RANK).fit(probs_t, used_t)
    del probs_t, used_t

    probs = flat(softmax(stack(test, "fate")), later)
    used = flat(stack(test, "used"), later)
    guess = top_k(probs, MAX_RANK)  # ranked: top_k sorts by score, highest first
    guess_p = np.take_along_axis(probs, guess, axis=1)
    guess_used = np.take_along_axis(used, guess, axis=1)
    single = cal.predict(guess_p)
    by_rank_cal = ranked.predict(guess_p)  # guess_p is in rank order, so ranks are right
    by_rank = [
        {
            "rank": r + 1,
            "observed_picked": float(guess_used[:, r].mean()),
            "mean_calibrated": float(single[:, r].mean()),
            "mean_rank_calibrated": float(by_rank_cal[:, r].mean()),
        }
        for r in range(guess.shape[1])
    ]
    top = slice(0, PREFETCH_K)
    total = int(used.sum())
    return {
        "calibrator": cal.to_dict(),
        "rank_calibrator": ranked.to_dict(),
        "test_rank_summary": by_rank,
        "test_ece_top12": {
            "single_map": expected_calibration_error(single, guess_used),
            "rank_aware": expected_calibration_error(by_rank_cal, guess_used),
        },
        "test_mean_abs_rank_error": {
            "single_map": float(np.mean([abs(x["mean_calibrated"] - x["observed_picked"]) for x in by_rank])),
            "rank_aware": float(
                np.mean([abs(x["mean_rank_calibrated"] - x["observed_picked"]) for x in by_rank])
            ),
        },
        "test_thresholds_for_top_k": {
            "k": PREFETCH_K,
            "rows": _threshold_rows(single[:, top], guess_used[:, top], total),
            "rows_rank_aware": _threshold_rows(by_rank_cal[:, top], guess_used[:, top], total),
        },
        "test_recall_top_k_no_threshold": float(guess_used[:, top].sum() / total),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--store-dir", type=Path, default=None)
    args = ap.parse_args()

    rt = RuntimeConfig.load(args.config)
    store_dir = (args.store_dir or rt.store_dir).resolve()
    trace_dir = trace_dir_for(store_dir)
    prompts = round_robin(json.loads(rt.prompts_file.read_text(encoding="utf-8"))["prompts"])
    done = [p["id"] for p in prompts if (trace_dir / f"{p['id']}.done.json").exists()]
    enforce_ram_budget(
        sum((trace_dir / f"{pid}.ert").stat().st_size for pid in done) * RAM_PER_TRACE_BYTE,
        MAX_RAM_GB,
        "prefetch calibration fit (traces + derived arrays)",
    )
    seqs = [to_seq(t) for t in load_traces(trace_dir, done)]
    if not seqs:
        raise SystemExit(f"no finished traces in {trace_dir}")
    tune, test = split_halves(seqs)
    manifest = json.loads((store_dir / "store.json").read_text())
    record = base_record(
        label="prefetch confidence calibration (isotonic, Fate router softmax -> P(picked))",
        seed=0,
        model={
            "store": store_dir.name,
            "repo_id": manifest["source"]["repo_id"],
            "revision": manifest["source"]["revision"],
        },
        config={
            "trace_dir": trace_dir.relative_to(REPO_ROOT).as_posix(),
            "split": "within each category: odd-numbered prompts fit, even-numbered prompts test",
            "tune_prompts": len(tune),
            "test_prompts": len(test),
            "layers": "1..23 (layer 0 has no Fate guess), generated tokens",
            "prefetch_k": PREFETCH_K,
            "max_rank": MAX_RANK,
            "thresholds": THRESHOLDS,
        },
        machine=collect_machine_profile(measure_disk=False),
    )
    record.update(fit(tune, test))
    record["peak_rss_mb"] = peak_process_rss_mb()
    out = BENCHMARK_RESULTS_DIR / f"prefetch_calibration_{store_dir.name}.json"
    append_benchmark_record(out, record)
    print(
        json.dumps(
            {k: record[k] for k in ("test_ece_top12", "test_mean_abs_rank_error", "test_rank_summary")},
            indent=1,
        )
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main()
