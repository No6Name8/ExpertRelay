"""Compute backends: the part of the forward pass that does the arithmetic.

The interface is deliberately small: the three operations that hold all
the model's FLOPs. Everything else (norms, RoPE, activations, routing,
KV-cache bookkeeping) is cheap elementwise numpy in runtime.qwen_moe and
is shared by every backend.

  int8_linear  x @ (Q * scale)^T (+ bias), int8 weights, per-row scales
  linear       x @ W^T (+ bias), f32 weights (router, shared_expert_gate)
  attention    causal softmax attention over the KV cache

Backends take and return numpy arrays. That's the contract that keeps the
store, the expert sources, the cache and the future predictor/Manager
independent of which backend runs: they only ever deal in numpy int8
arrays from disk, never in backend tensors.

Available backends (selected by expertrelay.manager.backend_selection):
  numpy      runtime.int8_linear's blocked kernel + numpy/BLAS. Default on CPU.
  mindspore  MindSpore f32 ops (runtime.backend_mindspore). The path for
             Huawei Ascend. On CPU it's correct but slower; see
             docs/limitations.md for the measured comparison.
"""

from __future__ import annotations

import importlib
from typing import Protocol

import numpy as np

from expertrelay.runtime.int8_linear import int8_linear

# Imported lazily by name: loading MindSpore costs hundreds of MB of RAM,
# which a numpy-only run on an 8 GB machine shouldn't pay.
_BACKENDS = {
    "numpy": "expertrelay.runtime.backends:NumpyBackend",
    "mindspore": "expertrelay.runtime.backend_mindspore:MindSporeBackend",
}
BACKEND_NAMES = tuple(_BACKENDS)


class Backend(Protocol):
    name: str

    def int8_linear(
        self, x: np.ndarray, q: np.ndarray, scales: np.ndarray, bias: np.ndarray | None = None
    ) -> np.ndarray:
        """x [n, in] f32, q [out, in] int8, scales [out] f32 -> [n, out] f32."""
        ...

    def linear(self, x: np.ndarray, w: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
        """x [n, in] f32, w [out, in] f32 -> [n, out] f32."""
        ...

    def attention(
        self, q: np.ndarray, keys: np.ndarray, values: np.ndarray, allowed: np.ndarray
    ) -> np.ndarray:
        """q [n, heads, d], keys/values [k, heads, d], allowed [n, k] bool
        (True where a query may attend to a key) -> [n, heads, d]. Scaled by
        d**-0.5, softmax in f32."""
        ...


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class NumpyBackend:
    name = "numpy"

    def int8_linear(
        self, x: np.ndarray, q: np.ndarray, scales: np.ndarray, bias: np.ndarray | None = None
    ) -> np.ndarray:
        return int8_linear(x, q, scales, bias)

    def linear(self, x: np.ndarray, w: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
        y = x @ w.T
        return y + bias if bias is not None else y

    def attention(
        self, q: np.ndarray, keys: np.ndarray, values: np.ndarray, allowed: np.ndarray
    ) -> np.ndarray:
        scores = np.einsum("qhd,khd->hqk", q, keys) * np.float32(q.shape[-1] ** -0.5)
        scores = np.where(allowed[None, :, :], scores, np.float32(-np.inf))
        return np.einsum("hqk,khd->qhd", _softmax(scores), values)


def make_backend(name: str, device: str | None = None) -> Backend:
    if name not in _BACKENDS:
        raise ValueError(f"unknown backend {name!r}; available: {', '.join(BACKEND_NAMES)}")
    module, cls = _BACKENDS[name].split(":")
    backend_cls = getattr(importlib.import_module(module), cls)
    return backend_cls(device) if device is not None else backend_cls()
