"""Scoring and pass rule of the int8/int4 quality study, and the speed
benchmark's bookkeeping, on hand-made inputs."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from expertrelay.bench.int4_benchmark import agreement, disk_per_token  # noqa: E402
from expertrelay.bench.quantization_quality import aggregate, score, verdict  # noqa: E402


def test_score_counts_agreement_and_generated_positions():
    ref = np.log(np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6], [0.5, 0.4, 0.1]]))
    ours = np.log(np.array([[0.6, 0.3, 0.1], [0.5, 0.4, 0.1], [0.2, 0.2, 0.6], [0.5, 0.4, 0.1]]))
    s = score(ref, ours, prompt_len=2)
    assert (s["positions"], s["agree"]) == (4, 3)
    # positions 1 and 2 predict the generated tokens; they disagree at position 1
    assert (s["generated_positions"], s["generated_agree"]) == (2, 1)
    assert s["kl_max"] > 0 and s["kl_sum"] >= s["kl_max"]


def test_aggregate_weights_by_positions():
    rows = [
        {
            "positions": 10,
            "agree": 10,
            "generated_positions": 5,
            "generated_agree": 5,
            "kl_sum": 0.1,
            "kl_max": 0.05,
        },
        {
            "positions": 30,
            "agree": 27,
            "generated_positions": 15,
            "generated_agree": 12,
            "kl_sum": 0.3,
            "kl_max": 0.2,
        },
    ]
    a = aggregate(rows)
    assert a["top1_agreement"] == pytest.approx(37 / 40)
    assert a["top1_agreement_generated"] == pytest.approx(17 / 20)
    assert a["kl_mean"] == pytest.approx(0.01) and a["kl_max"] == 0.2


def _prec(overall: float, cats: dict[str, float]) -> dict:
    return {
        "overall": {"top1_agreement": overall},
        "by_category": {c: {"top1_agreement": v} for c, v in cats.items()},
    }


def test_pass_rule_needs_overall_and_every_category():
    v = verdict(
        {
            "good": _prec(
                0.97, {"en": 0.98, "ar_gulf": 0.93, "phase2_ar": 0.5}
            ),  # phase2 prompts aren't categories
            "low_overall": _prec(0.955, {"en": 0.97, "ar_gulf": 0.94}),
            "collapsed": _prec(0.97, {"en": 0.99, "ar_gulf": 0.85}),
        }
    )
    assert v["good"]["pass"] and v["good"]["worst_category"] == "ar_gulf"
    assert not v["low_overall"]["pass"]
    assert not v["collapsed"]["pass"] and v["collapsed"]["worst_category_top1"] == 0.85


def test_speed_bookkeeping():
    def run(ids):
        step = {"device_bytes": 2e6, "expert_bytes": 1e6, "prefetch_bytes": 3e6}
        return {"prompts": [{"generated_ids": ids, "steps": [step, step, step]}]}

    assert disk_per_token(run([1]))["expert_mb_per_token"] == pytest.approx(4.0)
    results = [
        {"id": "int4_nocache", "precision": "int4", "round": 0, "status": "ok", "run": run([1, 2])},
        {"id": "int4_cache", "precision": "int4", "round": 0, "status": "ok", "run": run([1, 2])},
        {"id": "int4_adaptive", "precision": "int4", "round": 0, "status": "ok", "run": run([1, 3])},
    ]
    a = agreement(results)
    assert a["int4_cache#0"] is True and a["int4_adaptive#0"] is False
