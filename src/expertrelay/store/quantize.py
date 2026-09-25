"""Weight-only int8 quantization: symmetric, one scale per output channel.

Method: per-output-channel symmetric absmax quantization.
    scale_r = max_j |W[r, j]| / 127
    Q[r, j] = clip(round(W[r, j] / scale_r), -127, 127)
    W_hat[r, j] = Q[r, j] * scale_r
Each output channel (row of an [out, in] weight matrix) gets its own scale,
so one large-magnitude row can't wipe out the resolution of every other row.
  - Per-channel symmetric weight quantization: R. Krishnamoorthi, "Quantizing
    deep convolutional networks for efficient inference: A whitepaper",
    arXiv:1806.08342 (2018).
  - Row-wise ("vector-wise") absmax int8 for transformer weights:
    T. Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for
    Transformers at Scale", NeurIPS 2022, arXiv:2208.07339.
    We use only its weight-side absmax scaling, NOT its mixed-precision
    outlier decomposition, and we don't quantize activations.

The range is [-127, 127], not [-128, 127], so the grid is symmetric around
zero and negation is exact. Rounding is round-half-to-even (np.rint). The
bound |W - W_hat| <= scale / 2 holds elementwise, and tests check it.

An all-zero row has scale 0 and dequantizes back to exact zeros.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INT8_MAX = 127


def quantize_rowwise_int8(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[rows, cols] float -> (int8 [rows, cols], float32 scales [rows])."""
    w = np.asarray(w, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError(f"expected a 2-D weight matrix, got shape {w.shape}")
    scales = (np.abs(w).max(axis=1) / INT8_MAX).astype(np.float32)
    divisor = np.where(scales > 0, scales, np.float32(1.0))[:, None]
    q = np.clip(np.rint(w / divisor), -INT8_MAX, INT8_MAX).astype(np.int8)
    return q, scales


def dequantize_rowwise_int8(q: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return q.astype(np.float32) * scales.astype(np.float32)[:, None]


@dataclass
class ErrorAccumulator:
    """Quantization error for one tensor, accumulated over row blocks so a
    tensor never has to be in RAM whole to be measured.

    rel_fro_error = ||W - W_hat||_F / ||W||_F over the whole tensor.
    max_abs_error_over_scale = max over elements of |W - W_hat| / scale_row.
    Rounding bounds that at 0.5, so a value above 0.5 means a bug, not noise.

    Elementwise relative error |W - W_hat| / |W| is deliberately not used:
    it explodes for near-zero weights that contribute nothing to the output.
    """

    sq_error: float = 0.0
    sq_norm: float = 0.0
    max_abs_error_over_scale: float = 0.0

    def add(self, original: np.ndarray, q: np.ndarray, scales: np.ndarray) -> None:
        w = np.asarray(original, dtype=np.float64)
        err = w - dequantize_rowwise_int8(q, scales).astype(np.float64)
        self.sq_error += float(np.sum(err * err))
        self.sq_norm += float(np.sum(w * w))
        nonzero = scales > 0
        if nonzero.any():
            ratio = np.abs(err[nonzero]) / scales[nonzero].astype(np.float64)[:, None]
            self.max_abs_error_over_scale = max(self.max_abs_error_over_scale, float(ratio.max()))

    def merge(self, other: ErrorAccumulator) -> None:
        self.sq_error += other.sq_error
        self.sq_norm += other.sq_norm
        self.max_abs_error_over_scale = max(self.max_abs_error_over_scale, other.max_abs_error_over_scale)

    @property
    def rel_fro_error(self) -> float:
        return float(np.sqrt(self.sq_error / self.sq_norm)) if self.sq_norm > 0 else 0.0

    def summary(self) -> dict[str, float]:
        return {
            "rel_fro_error": self.rel_fro_error,
            "max_abs_error_over_scale": self.max_abs_error_over_scale,
        }
