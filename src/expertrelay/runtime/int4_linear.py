"""y = x @ W_hat^T for packed int4 weights with float16 group scales
(store.int4: symmetric, two values per byte, value + 8, even column in
the low 4 bits).

Two paths, like runtime.int8_linear, chosen by the same process-wide
setting (int8_linear.set_kernel / use_fused):
  fused    decode-sized inputs (<= 4 rows): numba, rows spread over all
           cores; each byte is unpacked into its two values in registers
           and multiplied directly, one partial sum per group, scaled by
           the group's scale. The weights are read once, as 4-bit.
           Prefill (more rows) with the fused setting: numba unpacks and
           scales the whole matrix in parallel into one reused float32
           buffer (an expert matrix is at most ~11.5 MB as f32), then one
           BLAS multiply.
  blocked  without numba: the same unpack in numpy, a block of rows at a
           time, then BLAS.
The group scales are converted to float32 once per call (they are 1/64
or 1/128 of the weights).
"""

from __future__ import annotations

import numpy as np

from expertrelay.runtime import int8_linear as _int8

_fused = None
_scratch = np.empty(0, dtype=np.float32)


def _compile_fused():
    import numba

    @numba.njit(parallel=True, fastmath=True, cache=True)
    def fused(x, packed, scales, group, out):  # pragma: no cover - compiled; tested through int4_linear
        n = x.shape[0]
        rows = packed.shape[0]
        half_group = group // 2
        groups = scales.shape[1]
        for r in numba.prange(rows):
            for i in range(n):
                total = np.float32(0.0)
                for g in range(groups):
                    acc = np.float32(0.0)
                    base = g * half_group
                    for b in range(half_group):
                        byte = packed[r, base + b]
                        j = 2 * (base + b)
                        acc += x[i, j] * np.float32((byte & 15) - 8) + x[i, j + 1] * np.float32(
                            (byte >> 4) - 8
                        )
                    total += acc * scales[r, g]
                out[i, r] = total

    @numba.njit(parallel=True, cache=True)
    def dequantize(packed, scales, group, w):  # pragma: no cover - compiled; tested through int4_linear
        rows, half = packed.shape
        for r in numba.prange(rows):
            for b in range(half):
                byte = packed[r, b]
                s = scales[r, (2 * b) // group]
                w[r, 2 * b] = np.float32((byte & 15) - 8) * s
                w[r, 2 * b + 1] = np.float32((byte >> 4) - 8) * s

    return fused, dequantize


def _fused_kernel():
    global _fused
    if _fused is None:
        _fused = _compile_fused()
    return _fused


def _block(rows: int, cols: int) -> np.ndarray:
    global _scratch
    if _scratch.size < rows * cols:
        _scratch = np.empty(rows * cols, dtype=np.float32)
    return _scratch[: rows * cols].reshape(rows, cols)


def int4_linear(
    x: np.ndarray,
    packed: np.ndarray,
    scales: np.ndarray,
    group: int,
    bias: np.ndarray | None = None,
    kernel: str | None = None,
) -> np.ndarray:
    """x [n, in] f32, packed [out, in/2] uint8, scales [out, in/group] f16 -> [n, out] f32."""
    n = x.shape[0]
    out_dim, half = packed.shape
    in_dim = half * 2
    s32 = scales.astype(np.float32)
    out = np.empty((n, out_dim), dtype=np.float32)
    if _int8.use_fused(n, kernel):
        _fused_kernel()[0](np.ascontiguousarray(x, dtype=np.float32), packed, s32, group, out)
    elif (kernel or _int8.current_kernel()) == "fused":
        # prefill with numba available: unpack the whole matrix in parallel
        # (an expert is at most ~11.5 MB as f32), then one BLAS multiply
        w = _block(out_dim, in_dim)
        _fused_kernel()[1](packed, s32, group, w)
        np.matmul(x, w.T, out=out)
    else:
        block = _int8.block_rows_for(n)
        for r0 in range(0, out_dim, block):
            r1 = min(out_dim, r0 + block)
            w = _block(r1 - r0, in_dim)
            p = packed[r0:r1]
            w[:, 0::2] = (p & 0x0F).astype(np.float32) - 8.0
            w[:, 1::2] = (p >> 4).astype(np.float32) - 8.0
            w.reshape(r1 - r0, in_dim // group, group)[...] *= s32[r0:r1, :, None]
            np.matmul(x, w.T, out=out[:, r0:r1])
    if bias is not None:
        out += bias
    return out
