"""int8 quantize/dequantize round trip (expertrelay.store.quantize)."""

from __future__ import annotations

import numpy as np
import pytest

from expertrelay.store.quantize import ErrorAccumulator, dequantize_rowwise_int8, quantize_rowwise_int8


def _weights(rows=64, cols=128, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.02, size=(rows, cols)).astype(np.float32)
    w[3] *= 50  # one large-magnitude row: must not hurt the other rows' resolution
    return w


def test_round_trip_error_is_within_half_a_step_per_element():
    w = _weights()
    q, scales = quantize_rowwise_int8(w)
    w_hat = dequantize_rowwise_int8(q, scales)
    assert q.dtype == np.int8 and scales.dtype == np.float32 and scales.shape == (64,)
    # rounding to the nearest grid point: never more than half a step off (float slack for the division)
    assert np.all(np.abs(w - w_hat) <= scales[:, None] * (0.5 + 1e-6))


def test_symmetric_range_and_absmax_maps_to_127():
    q, scales = quantize_rowwise_int8(_weights())
    assert q.min() >= -127 and q.max() <= 127  # -128 is never used
    assert np.all(np.abs(q).max(axis=1) == 127)  # each row's largest-magnitude weight hits the edge


def test_scales_are_per_row():
    w = _weights()
    _, scales = quantize_rowwise_int8(w)
    np.testing.assert_allclose(scales, np.abs(w).max(axis=1) / 127, rtol=1e-6)


def test_zero_row_round_trips_exactly():
    w = _weights()
    w[5] = 0.0
    q, scales = quantize_rowwise_int8(w)
    assert scales[5] == 0 and not q[5].any()
    assert not dequantize_rowwise_int8(q, scales)[5].any()


def test_quantization_is_deterministic():
    w = _weights()
    q1, s1 = quantize_rowwise_int8(w)
    q2, s2 = quantize_rowwise_int8(w.copy())
    np.testing.assert_array_equal(q1, q2)
    np.testing.assert_array_equal(s1, s2)


def test_rejects_non_2d():
    with pytest.raises(ValueError):
        quantize_rowwise_int8(np.zeros(8, dtype=np.float32))


def test_error_accumulator_matches_direct_computation_and_blocks_merge():
    w = _weights()
    q, s = quantize_rowwise_int8(w)
    whole = ErrorAccumulator()
    whole.add(w, q, s)
    expected = np.linalg.norm(w - dequantize_rowwise_int8(q, s)) / np.linalg.norm(w)
    assert whole.rel_fro_error == pytest.approx(expected, rel=1e-6)
    assert 0 < whole.max_abs_error_over_scale <= 0.5 + 1e-6

    # the same tensor quantized in two row blocks (how big resident tensors are built)
    blocks = ErrorAccumulator()
    for lo, hi in ((0, 20), (20, 64)):
        blocks.add(w[lo:hi], *quantize_rowwise_int8(w[lo:hi]))
    assert blocks.rel_fro_error == pytest.approx(whole.rel_fro_error, rel=1e-9)
