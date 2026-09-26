"""MindSpore f32 compute backend: the path for Huawei Ascend hardware.

Every matmul and the attention softmax run as MindSpore ops. int8 weights
are cast to f32 inside MindSpore, and the per-row scale is applied to the
output (the same factoring as the numpy kernel, so both backends compute
the same function).

Honest scope, also in docs/limitations.md:
  - Tested on CPU only (MindSpore 2.7.1 CPU wheel); this machine has no
    Ascend device. `device` is passed straight to mindspore.set_device, so
    "Ascend" is one argument away, but it has never been run there.
  - Arrays cross the numpy <-> MindSpore boundary on every call. That's what
    keeps the rest of the system backend-independent, and it's fine on CPU,
    but on Ascend each call would copy host <-> device. A device-resident
    version (weights uploaded once, activations kept on the NPU) is needed
    before this is a fast Ascend path; today it's a correct one.
  - f32 only. f16 on MindSpore CPU measured slower and ~1000x less accurate
    (benchmarks/results/int8_kernels.json).
"""

from __future__ import annotations

import mindspore as ms
import numpy as np
from mindspore import ops


class MindSporeBackend:
    name = "mindspore"

    def __init__(self, device: str = "CPU"):
        ms.set_device(device)
        self.device = device

    def int8_linear(
        self, x: np.ndarray, q: np.ndarray, scales: np.ndarray, bias: np.ndarray | None = None
    ) -> np.ndarray:
        w = ops.cast(ms.Tensor(q), ms.float32)
        y = ops.matmul(ms.Tensor(x), w.T) * ms.Tensor(scales)
        if bias is not None:
            y = y + ms.Tensor(bias)
        return y.asnumpy()

    def linear(self, x: np.ndarray, w: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
        y = ops.matmul(ms.Tensor(x), ms.Tensor(w).T)
        if bias is not None:
            y = y + ms.Tensor(bias)
        return y.asnumpy()

    def attention(
        self, q: np.ndarray, keys: np.ndarray, values: np.ndarray, allowed: np.ndarray
    ) -> np.ndarray:
        qt = ops.transpose(ms.Tensor(q), (1, 0, 2))  # [heads, n, d]
        kt = ops.transpose(ms.Tensor(keys), (1, 2, 0))  # [heads, d, k]
        vt = ops.transpose(ms.Tensor(values), (1, 0, 2))  # [heads, k, d]
        scores = ops.bmm(qt, kt) * float(q.shape[-1] ** -0.5)  # [heads, n, k]
        scores = ops.masked_fill(scores, ms.Tensor(~allowed)[None], float("-inf"))
        out = ops.bmm(ops.softmax(scores, axis=-1), vt)  # [heads, n, d]
        return ops.transpose(out, (1, 0, 2)).asnumpy()
