"""Fit the prefetcher's confidence calibration from recorded traces.

    python -m expertrelay.bench.fit_prefetch_calibration            # base store

Offline, from finished traces only. Fits the Phase 3.5 isotonic map (raw
Fate router softmax probability of one expert -> P(it is among the experts
picked)) on the TUNING prompts (odd-numbered in every category, as in
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
from expertrelay.predictor.offline import IsotonicCalibrator, softmax, top_k
from expertrelay.runtime.generate import RuntimeConfig

PREFETCH_K = 8
THRESHOLDS = [0.01, 0.02, 0.05, 0.1, 0.2]
MAX_RAM_GB = 1.0
RAM_PER_TRACE_BYTE = 7  # as bench/phase35_prediction.py: same arrays


def fit(tune, test) -> dict:
    later = slice(1, None)  # layer 0 has no previous layer, so no Fate guess
    probs_t = flat(softmax(stack(tune, "fate")), later)
    used_t = flat(stack(tune, "used"), later)
    cal = IsotonicCalibrator().fit(probs_t, used_t)
    del probs_t, used_t

    probs = flat(softmax(stack(test, "fate")), later)
    used = flat(stack(test, "used"), later)
    guess = top_k(probs, 12)  # ranked: top_k sorts by score, highest first
    guess_p = np.take_along_axis(probs, guess, axis=1)
    guess_cal = cal.predict(guess_p)
    guess_used = np.take_along_axis(used, guess, axis=1)
    by_rank = [
        {
            "rank": r + 1,
            "mean_calibrated": float(guess_cal[:, r].mean()),
            "observed_picked": float(guess_used[:, r].mean()),
        }
        for r in range(guess.shape[1])
    ]
    top = slice(0, PREFETCH_K)
    thresholds = []
    for t in THRESHOLDS:
        drop = guess_cal[:, top] < t
        thresholds.append(
            {
                "min_probability": t,
                "prefetches_dropped": float(drop.mean()),
                "dropped_that_would_be_used": float(guess_used[:, top][drop].mean()) if drop.any() else None,
                "recall_kept": float((guess_used[:, top] & ~drop).sum() / used.sum()),
            }
        )
    return {
        "calibrator": cal.to_dict(),
        "test_rank_summary": by_rank,
        "test_thresholds_for_top_k": {"k": PREFETCH_K, "rows": thresholds},
        "test_recall_top_k_no_threshold": float(guess_used[:, top].sum() / used.sum()),
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
            {k: record[k] for k in ("test_thresholds_for_top_k", "test_recall_top_k_no_threshold")}, indent=1
        )
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main()
