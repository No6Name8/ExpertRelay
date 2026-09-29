"""The int8 kernel against float64 math on the same int8 weights."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from expertrelay.runtime.int8_linear import BLOCK_ROWS_DECODE, BLOCK_ROWS_PREFILL, block_rows_for, int8_linear
from expertrelay.store.quantize import quantize_rowwise_int8


def _case(n: int, out_dim: int, in_dim: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    q, s = quantize_rowwise_int8(rng.normal(0, 0.05, (out_dim, in_dim)).astype(np.float32))
    x = rng.normal(0, 1, (n, in_dim)).astype(np.float32)
    exact = x.astype(np.float64) @ (q.astype(np.float64) * s.astype(np.float64)[:, None]).T
    return x, q, s, exact


@pytest.mark.parametrize("n", [1, 3, 40])
@pytest.mark.parametrize("block_rows", [None, 7, 128, 1000])
def test_matches_float64_for_any_block_size(n, block_rows):
    # 300 output rows: not a multiple of any block size, so the tail block is exercised
    x, q, s, exact = _case(n, 300, 96)
    y = int8_linear(x, q, s, block_rows=block_rows)
    assert y.dtype == np.float32 and y.shape == (n, 300)
    np.testing.assert_allclose(y, exact, rtol=1e-5, atol=1e-5)


def test_bias_and_out_buffer():
    x, q, s, exact = _case(2, 50, 16)
    bias = np.arange(50, dtype=np.float32)
    out = np.empty((2, 50), dtype=np.float32)
    y = int8_linear(x, q, s, bias=bias, out=out)
    assert y is out
    np.testing.assert_allclose(y, exact + bias, rtol=1e-5, atol=1e-5)


def test_block_size_choice():
    assert block_rows_for(1) == BLOCK_ROWS_DECODE
    assert block_rows_for(64) == BLOCK_ROWS_PREFILL


def test_results_do_not_depend_on_scratch_reuse():
    # a big call grows the shared scratch buffer; a later small call must not see stale data
    x1, q1, s1, _ = _case(1, 600, 256, seed=1)
    int8_linear(x1, q1, s1)
    x2, q2, s2, exact2 = _case(1, 10, 8, seed=2)
    np.testing.assert_allclose(int8_linear(x2, q2, s2), exact2, rtol=1e-5, atol=1e-5)


needs_numba = pytest.mark.skipif(importlib.util.find_spec("numba") is None, reason="numba not installed")


@needs_numba
@pytest.mark.parametrize("n", [1, 2, 4])
@pytest.mark.parametrize(("out_dim", "in_dim"), [(300, 96), (7, 1000), (1, 1)])
def test_fused_kernel_matches_float64(n, out_dim, in_dim):
    x, q, s, exact = _case(n, out_dim, in_dim, seed=3)
    y = int8_linear(x, q, s, kernel="fused")
    assert y.dtype == np.float32 and y.shape == (n, out_dim)
    np.testing.assert_allclose(y, exact, rtol=1e-5, atol=1e-5)


@needs_numba
def test_fused_kernel_bias_and_non_contiguous_input():
    x, q, s, exact = _case(4, 50, 64, seed=4)
    bias = np.arange(50, dtype=np.float32)
    wide = np.zeros((4, 128), dtype=np.float32)
    wide[:, ::2] = x  # a strided view, like h[token_idx] can be
    y = int8_linear(wide[:, ::2], q, s, bias, kernel="fused")
    np.testing.assert_allclose(y, exact + bias, rtol=1e-5, atol=1e-5)


def test_fused_kernel_only_for_decode_sized_inputs():
    from expertrelay.runtime.int8_linear import DECODE_MAX_ROWS, use_fused

    assert use_fused(1, "fused") and use_fused(DECODE_MAX_ROWS, "fused")
    assert not use_fused(DECODE_MAX_ROWS + 1, "fused")
    assert not use_fused(1, "blocked")


def test_unknown_kernel_is_rejected():
    from expertrelay.runtime.int8_linear import set_kernel

    with pytest.raises(ValueError):
        set_kernel("avx9000")
