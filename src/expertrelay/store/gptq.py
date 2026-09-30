"""Convert a GPTQ release's 4-bit linear layers into store.int4's encoding,
losslessly.

GPTQ: E. Frantar, S. Ashkboos, T. Hoefler, D. Alistarh, "GPTQ: Accurate
Post-Training Quantization for Generative Pre-trained Transformers", ICLR
2023, arXiv:2210.17323. The weights come already quantized in the release
(Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4); nothing here runs GPTQ itself.

Checkpoint format (AutoGPTQ's "gptq" format, i.e. v1; the release's
quantization_config has no checkpoint_format field, which means v1), per
linear layer with in_features I, out_features O, group size G:
  qweight int32 [I / 8, O]   8 codes per int32 along the INPUT dim; code
                             of input row i sits in bits 4*(i % 8) of
                             qweight[i // 8, o]
  qzeros  int32 [I / G, O/8] 8 zero points per int32 along the OUTPUT dim,
                             same bit order, stored MINUS ONE (v1)
  scales  float16 [I / G, O]
  g_idx   int32 [I]          group of each input row (i // G unless desc_act)
  bias    float16 [O]
  W[o, i] = (code[i, o] - (stored_zero[g_idx[i], o] + 1)) * scales[g_idx[i], o]

The release is symmetric (sym=true, desc_act=false), so every stored zero
is 7 (zero point 8) and g_idx is i // G; with those, W = (code - 8) * scale,
which is exactly store.int4's encoding (nibble - 8, times a float16 group
scale): the conversion only re-packs the codes along the other dimension
and transposes the scales, and changes no value. Anything else (another
zero point, a non-trivial g_idx, a non-zero bias) is refused rather than
converted approximately; store.int4 has no per-group zero points or
permutation because this release doesn't need them.
"""

from __future__ import annotations

import numpy as np

from expertrelay.store.int4 import GPTQ_ZERO

GPTQ_TENSORS = ("qweight", "qzeros", "scales", "g_idx", "bias")


def unpack_int32_nibbles(packed: np.ndarray, axis: int) -> np.ndarray:
    """int32 array with 8 4-bit codes per element (lowest bits first) ->
    uint8 codes, expanded 8x along `axis`."""
    u = np.ascontiguousarray(packed).view(np.uint32)
    codes = np.stack([(u >> (4 * k)) & 0xF for k in range(8)], axis=axis + 1).astype(np.uint8)
    shape = list(u.shape)
    shape[axis] *= 8
    return codes.reshape(shape)


def gptq_to_int4(
    qweight: np.ndarray,
    qzeros: np.ndarray,
    scales: np.ndarray,
    g_idx: np.ndarray,
    bias: np.ndarray | None,
    group: int,
) -> tuple[np.ndarray, np.ndarray]:
    """One GPTQ linear -> (packed uint8 [O, I / 2], float16 scales [O, I / G]) in
    store.int4's encoding. Raises ValueError if the layer isn't the symmetric,
    in-order, bias-free kind this conversion is exact for."""
    in_dim, out_dim = qweight.shape[0] * 8, qweight.shape[1]
    if qzeros.shape != (in_dim // group, out_dim // 8) or scales.shape != (in_dim // group, out_dim):
        raise ValueError(f"shapes don't match group size {group}: {qzeros.shape}, {scales.shape}")
    if not np.array_equal(g_idx, np.arange(in_dim) // group):
        raise ValueError("g_idx is not i // group (desc_act?): not supported")
    stored_zeros = unpack_int32_nibbles(qzeros, axis=1)
    if not (stored_zeros == GPTQ_ZERO - 1).all():
        raise ValueError(f"zero points {np.unique(stored_zeros) + 1} are not all {GPTQ_ZERO}")
    if bias is not None and np.any(bias):
        raise ValueError("non-zero bias: not supported")
    codes = unpack_int32_nibbles(qweight, axis=0).T  # [O, I]; code - 8 is the value, as in store.int4
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return np.ascontiguousarray(packed, dtype=np.uint8), np.ascontiguousarray(scales.T, dtype=np.float16)
