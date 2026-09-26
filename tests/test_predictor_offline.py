"""Offline predictor tools on inputs with known answers."""

from __future__ import annotations

import numpy as np
import pytest

from expertrelay.predictor.offline import (
    NEVER,
    IsotonicCalibrator,
    ReuseModel,
    TransitionModel,
    expected_calibration_error,
    gap_bucket,
    pool_adjacent_violators,
    recency_scores,
    topk_hits,
    used_matrix,
)


def test_used_matrix():
    u = used_matrix(np.array([[0, 2], [1, 2]]), 4)
    assert u.tolist() == [[True, False, True, False], [False, True, True, False]]


def test_recency_scores_count_only_earlier_positions():
    used = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=bool)
    r = recency_scores(used, window=2, decay=0.5)
    # t=0: nothing before; t=1: used[0]; t=2: used[1] + 0.5*used[0]; t=3: used[2] + 0.5*used[1]
    np.testing.assert_allclose(r, [[0, 0], [1, 0], [0.5, 1], [1, 1.5]])


def test_topk_hits():
    scores = np.array([[0.9, 0.1, 0.8, 0.0], [0.0, 0.1, 0.2, 0.3]])
    truth = np.array([[0, 1], [0, 3]])
    assert topk_hits(scores, truth, 2).tolist() == [1, 1]
    assert topk_hits(scores, truth, 4).tolist() == [2, 2]


def test_pool_adjacent_violators():
    out = pool_adjacent_violators(np.array([1.0, 3.0, 2.0, 4.0]), np.ones(4))
    np.testing.assert_allclose(out, [1.0, 2.5, 2.5, 4.0])
    weighted = pool_adjacent_violators(np.array([3.0, 1.0]), np.array([1.0, 3.0]))
    np.testing.assert_allclose(weighted, [1.5, 1.5])


def test_isotonic_calibrator_recovers_rates_and_is_monotone():
    # score s in 0..9, each 100 times; exactly s*10 of those 100 are positive
    scores = np.repeat(np.arange(10.0), 100)
    labels = np.concatenate([np.arange(100) < s * 10 for s in range(10)]).astype(float)
    cal = IsotonicCalibrator(num_bins=10).fit(scores, labels)
    np.testing.assert_allclose(cal.predict(np.arange(10.0)), np.arange(10) / 10, atol=1e-6)
    grid = cal.predict(np.linspace(-5, 15, 200))
    assert (np.diff(grid) >= 0).all()


def test_isotonic_calibrator_needs_fit():
    with pytest.raises(RuntimeError):
        IsotonicCalibrator().predict(np.zeros(3))


def test_expected_calibration_error():
    assert expected_calibration_error(np.full(4, 0.5), np.array([1, 0, 1, 0])) == pytest.approx(0.0)
    assert expected_calibration_error(np.ones(4), np.zeros(4)) == pytest.approx(1.0)


def test_transition_model_learns_a_deterministic_mapping():
    src = used_matrix(np.array([[0], [1], [2], [0], [1], [2]]), 3)
    dst = used_matrix(np.array([[2], [0], [1], [2], [0], [1]]), 3)
    tm = TransitionModel(3, alpha=0.1)
    tm.add(src, dst)
    assert tm.conditional().argmax(axis=1).tolist() == [2, 0, 1]
    np.testing.assert_allclose(tm.conditional().sum(axis=1), 1.0)
    assert tm.score(src[:3].astype(float)).argmax(axis=1).tolist() == [2, 0, 1]


def test_gap_bucket():
    assert gap_bucket(np.array([1, 16, 17, 32, 33, 64, 65, 10_000, -1])).tolist() == [
        0,
        15,
        16,
        16,
        17,
        17,
        18,
        18,
        NEVER,
    ]


def test_reuse_model_rates():
    # expert 0 every position, expert 1 every other position, expert 2 never
    t = 40
    used = np.zeros((t, 3), dtype=bool)
    used[:, 0] = True
    used[::2, 1] = True
    rm = ReuseModel(min_count=1)
    rm.add(used, first_target=8)  # targets only once the 8-position history is full
    rm.fit()
    # gap 1 after full recent history: expert 0 is always used again, expert 1 never (it skips a position)
    assert rm.lookup(np.array([1]), np.array([8]))[0] == pytest.approx(1.0)
    assert rm.lookup(np.array([1]), np.array([4]))[0] == pytest.approx(0.0)
    assert rm.lookup(np.array([2]), np.array([4]))[0] == pytest.approx(1.0)
    assert rm.lookup(np.array([-1]), np.array([0]))[0] == pytest.approx(0.0)  # expert 2


def test_reuse_model_thin_cells_fall_back_to_gap_rate():
    used = np.zeros((10, 2), dtype=bool)
    used[:, 0] = True
    rm = ReuseModel(min_count=10**6)
    rm.add(used, first_target=0)
    rm.fit()
    # every cell is "thin", so any count at gap 1 gets the gap-1 rate (expert 0 always reused)
    assert rm.lookup(np.array([1, 1]), np.array([0, 8])).tolist() == pytest.approx([1.0, 1.0])
