"""y = x @ W_hat^T for int8 weights with one f32 scale per output row.

The kernel ("numpy_blocked" in bench/int8_kernels.py) was chosen by
measurement over dequantize-then-matmul and MindSpore f32/f16 matmul. The
numbers are in docs/limitations.md, raw data in
benchmarks/results/int8_kernels.json.

Two things make it fast on a CPU without int8 matmul instructions exposed
to Python:
  - Per-row scales factor out of the dot product:
    y[:, r] = s_r * (x @ Q[r]^T). So the weights are only converted to f32,
    never multiplied, and the scale is applied to the much smaller output.
  - Q is converted a block of rows at a time into one reused, cache-sized
    f32 buffer and handed to BLAS from there. The full f32 matrix is never
    materialized (lm_head's would be 1.24 GB), and the buffer is allocated
    once, not per call. Fresh multi-MB allocations cost page faults on
    every call on Windows.

A second kernel, "fused", is for decode (a few input rows), where the
blocked kernel's cost is mostly the int8 -> f32 conversion, not the
multiply. It multiplies the int8 weights directly by the activations, one
output row per iteration, rows spread across all cores (numba prange), and
applies the row scale once at the end. The weights are read once, as int8,
and never copied. It sums in a different order than BLAS (vectorized
partial sums), so its outputs differ in the last bits; the reference
check (bench/reference_check.py) measures it against the original model.
Prefill (many rows) stays on the blocked kernel, where BLAS is compute-bound
and faster. numba is imported only when the fused kernel is selected.
"""

from __future__ import annotations

import numpy as np

# Measured (benchmarks/results/int8_kernels.json): for decode (1 input row)
# 128 rows was the fastest block on all five real shapes. 128 x 2048 f32 is
# 1 MB, which fits this CPU's 1.25 MB per-core L2. For multi-row prefill no
# size won consistently (the measurements are noisy on a busy 8 GB machine),
# so prefill keeps 512.
BLOCK_ROWS_DECODE = 128
BLOCK_ROWS_PREFILL = 512
DECODE_MAX_ROWS = 4


KERNELS = ("blocked", "fused")
_kernel = "blocked"
_fused = None  # the compiled numba function, once the fused kernel is selected


def block_rows_for(n: int) -> int:
    return BLOCK_ROWS_DECODE if n <= DECODE_MAX_ROWS else BLOCK_ROWS_PREFILL


def _compile_fused():
    import numba

    @numba.njit(parallel=True, fastmath=True, cache=True)
    def fused(x, q, scales, out):  # pragma: no cover - compiled; tested through int8_linear
        n = x.shape[0]
        rows, cols = q.shape
        for r in numba.prange(rows):
            for i in range(n):
                acc = np.float32(0.0)
                for j in range(cols):
                    acc += x[i, j] * np.float32(q[r, j])
                out[i, r] = acc * scales[r]

    return fused


def _fused_kernel():
    global _fused
    if _fused is None:
        _fused = _compile_fused()
    return _fused


def set_kernel(name: str) -> None:
    """Process-wide choice for decode-sized inputs ("blocked" or "fused")."""
    global _kernel
    if name not in KERNELS:
        raise ValueError(f"unknown int8 kernel {name!r}; available: {', '.join(KERNELS)}")
    if name == "fused":
        _fused_kernel()  # compile now, so a missing numba fails here, not mid-run
    _kernel = name


def current_kernel() -> str:
    return _kernel


def use_fused(n: int, kernel: str | None = None) -> bool:
    return (kernel or _kernel) == "fused" and n <= DECODE_MAX_ROWS


_scratch = np.empty(0, dtype=np.float32)


def _scratch_block(rows: int, cols: int) -> np.ndarray:
    global _scratch
    if _scratch.size < rows * cols:
        _scratch = np.empty(rows * cols, dtype=np.float32)
    return _scratch[: rows * cols].reshape(rows, cols)


def int8_linear(
    x: np.ndarray,
    q: np.ndarray,
    scales: np.ndarray,
    bias: np.ndarray | None = None,
    out: np.ndarray | None = None,
    block_rows: int | None = None,
    kernel: str | None = None,
) -> np.ndarray:
    """x: [n, in] f32, q: [out, in] int8, scales: [out] f32 -> [n, out] f32.

    Not thread-safe (shared scratch buffer); the forward pass is single-threaded,
    and BLAS (blocked) or numba (fused) parallelizes inside the call.
    """
    n = x.shape[0]
    out_dim, in_dim = q.shape
    if out is None:
        out = np.empty((n, out_dim), dtype=np.float32)
    if use_fused(n, kernel):
        _fused_kernel()(np.ascontiguousarray(x, dtype=np.float32), q, scales, out)
        if bias is not None:
            out += bias
        return out
    block = block_rows or block_rows_for(n)
    for r0 in range(0, out_dim, block):
        r1 = min(out_dim, r0 + block)
        w = _scratch_block(r1 - r0, in_dim)
        np.copyto(w, q[r0:r1], casting="unsafe")
        np.matmul(x, w.T, out=out[:, r0:r1])
    out *= scales
    if bias is not None:
        out += bias
    return out
