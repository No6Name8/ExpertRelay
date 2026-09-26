"""Choosing experts to prefetch from the Fate-style router logits."""

from __future__ import annotations

import json

import numpy as np
import pytest

from expertrelay.predictor.offline import IsotonicCalibrator
from expertrelay.predictor.prefetch_policy import PrefetchPolicy, load_calibrator


def calibrator() -> IsotonicCalibrator:
    """P(picked) = 0 below probability 0.12, 0.5 up to 0.25, 0.9 above."""
    cal = IsotonicCalibrator(num_bins=3)
    cal.edges = np.array([0.12, 0.25])
    cal.values = np.array([0.0, 0.5, 0.9])
    return cal


def test_top_k_by_router_probability():
    logits = np.log(np.array([0.05, 0.4, 0.1, 0.3, 0.15]))
    choice = PrefetchPolicy(top_k=3).choose(logits)
    assert choice.experts == [1, 3, 4]
    assert choice.skipped_low_confidence == 0


def test_low_confidence_guesses_are_skipped():
    logits = np.log(np.array([0.05, 0.4, 0.1, 0.3, 0.15]))
    choice = PrefetchPolicy(top_k=4, min_probability=0.4, calibrator=calibrator()).choose(logits)
    # calibrated: 0.4 -> 0.9, 0.3 -> 0.9, 0.15 -> 0.5, 0.1 -> 0.0
    assert choice.experts == [1, 3, 4]
    assert choice.skipped_low_confidence == 1
    choice = PrefetchPolicy(top_k=4, min_probability=0.6, calibrator=calibrator()).choose(logits)
    assert choice.experts == [1, 3]
    assert choice.skipped_low_confidence == 2


def test_several_tokens_use_each_experts_best_probability():
    logits = np.log(np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.7]]))
    assert PrefetchPolicy(top_k=2).choose(logits).experts == [0, 3]


def test_calibrator_round_trip_and_validation():
    cal = calibrator()
    back = IsotonicCalibrator.from_dict(cal.to_dict())
    x = np.linspace(0, 1, 11)
    np.testing.assert_array_equal(back.predict(x), cal.predict(x))
    with pytest.raises(ValueError):
        IsotonicCalibrator.from_dict({"num_bins": 2, "edges": [0.5], "values": [0.9, 0.1]})


def test_load_calibrator_uses_the_last_record(tmp_path):
    old = {"calibrator": {"num_bins": 1, "edges": [], "values": [0.1]}, "model": {"store": "a"}}
    new = {"calibrator": calibrator().to_dict(), "model": {"store": "b"}, "git_commit": "abc"}
    path = tmp_path / "prefetch_calibration_b.json"
    path.write_text(json.dumps([old, new]))
    cal, provenance = load_calibrator(path)
    assert cal.predict(np.array([0.5]))[0] == pytest.approx(0.9)
    assert provenance["store"] == "b" and provenance["git_commit"] == "abc"
