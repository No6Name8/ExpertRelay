"""Weight-only int4 for routed experts: symmetric, round-to-nearest,
group-wise scales along the input dimension.

Method. Each row of an [out, in] weight matrix is cut into groups of G
consecutive input columns (G = 128 or 64); each group gets one scale:
    scale_g = max_{j in g} |W[r, j]| / 7      (stored as float16)
    Q[r, j] = clip(round(W[r, j] / scale_g), -7, 7)
    W_hat[r, j] = Q[r, j] * scale_g
Plain round-to-nearest (RTN), no calibration data, no error compensation:
the weights alone decide the result.
  - Fine-grained, group-wise weight scales for transformer LLMs: Z. Yao,
    R. Y. Aminabadi, M. Zhang, X. Wu, C. Li, Y. He, "ZeroQuant: Efficient
    and Affordable Post-Training Quantization for Large-Scale
    Transformers", NeurIPS 2022, arXiv:2206.01861.
  - Small independently quantized blocks as the main lever for 4-bit
    round-to-nearest: T. Dettmers, L. Zettlemoyer, "The case for 4-bit
    precision: k-bit Inference Scaling Laws", ICML 2023 (PMLR 202),
    arXiv:2212.09720.
  - NOT used: GPTQ (E. Frantar, S. Ashkboos, T. Hoefler, D. Alistarh,
    "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained
    Transformers", ICLR 2023, arXiv:2210.17323), which corrects rounding
    error with second-order information from calibration data; and AWQ
    (J. Lin et al., "AWQ: Activation-aware Weight Quantization for LLM
    Compression and Acceleration", MLSys 2024, arXiv:2306.00978), which
    rescales salient channels using activation statistics. Both need
    calibration data and usually beat round-to-nearest at 4 bits;
    round-to-nearest is the baseline they are compared against.

Details that matter for reproducing it:
  - Symmetric, no zero point: 15 levels, -7..7 (the 16th code is unused),
    so the grid is symmetric and negation is exact.
  - The scale is computed in float32, stored in float16, and the weights are
    quantized with the STORED float16 scale. If rounding to float16 made the
    scale too small to reach the group's largest weight, it is moved up to
    the next float16 value, so nothing is ever clipped and
    |W - W_hat| <= scale / 2 holds for every weight (tests check it).
  - Rounding is round-half-to-even (np.rint).
  - An all-zero group has scale 0 and dequantizes to exact zeros.
  - Packing: two values per byte, value + 8 in 1..15; the even input column
    in the low 4 bits, the odd column in the high 4 bits.

Record (one per expert, like the int8 store's, store.layout):
  [gate packed][up packed][down packed][gate scales f16][up scales f16][down scales f16][zero pad]
padded to a multiple of 4096 bytes, so one unbuffered read loads one expert.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from expertrelay.store.unbuffered_io import ALIGNMENT, align_up

INT4_MAX = 7
FORMAT = "int4_groupwise_symmetric_rtn"


def quantize_groupwise_int4(w: np.ndarray, group: int) -> tuple[np.ndarray, np.ndarray]:
    """[rows, cols] float -> (int8 values in -7..7 [rows, cols], float16 scales [rows, cols // group])."""
    w = np.asarray(w, dtype=np.float32)
    rows, cols = w.shape
    if cols % group or group % 2:
        raise ValueError(f"{cols} columns don't split into groups of {group} (even)")
    g = w.reshape(rows, cols // group, group)
    absmax = np.abs(g).max(axis=2)
    scales = (absmax / INT4_MAX).astype(np.float16)
    # never let float16 rounding shrink a scale below what the largest weight needs
    while True:
        short = scales.astype(np.float32) * INT4_MAX < absmax
        if not short.any():
            break
        scales[short] = np.nextafter(scales[short], np.float16(np.inf))
    divisor = np.where(scales > 0, scales.astype(np.float32), np.float32(1.0))[:, :, None]
    q = np.clip(np.rint(g / divisor), -INT4_MAX, INT4_MAX).astype(np.int8)
    return q.reshape(rows, cols), scales


def pack_int4(q: np.ndarray) -> np.ndarray:
    """int8 values in -7..7 [rows, cols] -> uint8 [rows, cols // 2]."""
    u = (q.astype(np.int16) + 8).astype(np.uint8)
    return (u[:, 0::2] | (u[:, 1::2] << 4)).astype(np.uint8)


def unpack_int4(packed: np.ndarray) -> np.ndarray:
    """Inverse of pack_int4."""
    rows, half = packed.shape
    q = np.empty((rows, half * 2), dtype=np.int8)
    q[:, 0::2] = (packed & 0x0F).astype(np.int8) - 8
    q[:, 1::2] = (packed >> 4).astype(np.int8) - 8
    return q


def dequantize_groupwise_int4(packed: np.ndarray, scales: np.ndarray, group: int) -> np.ndarray:
    q = unpack_int4(packed).astype(np.float32)
    return q * np.repeat(scales.astype(np.float32), group, axis=1)


def quant_error(w: np.ndarray, packed: np.ndarray, scales: np.ndarray, group: int) -> dict[str, float]:
    """rel_fro_error = ||W - W_hat||_F / ||W||_F; max_abs_error_over_scale is
    bounded by 0.5 when nothing is clipped (the int8 store reports the same
    two numbers, store.quantize.ErrorAccumulator)."""
    w = np.asarray(w, dtype=np.float64)
    err = w - dequantize_groupwise_int4(packed, scales, group).astype(np.float64)
    per_elem_scale = np.repeat(scales.astype(np.float64), group, axis=1)
    nonzero = per_elem_scale > 0
    return {
        "rel_fro_error": float(np.sqrt(np.sum(err * err) / np.sum(w * w))) if np.any(w) else 0.0,
        "max_abs_error_over_scale": float((np.abs(err[nonzero]) / per_elem_scale[nonzero]).max())
        if nonzero.any()
        else 0.0,
        "sq_error": float(np.sum(err * err)),
        "sq_norm": float(np.sum(w * w)),
    }


@dataclass(frozen=True)
class Int4RecordLayout:
    """Byte layout of one int4 expert record; `matrices` order is the on-disk order."""

    matrices: tuple[tuple[str, tuple[int, int]], ...]
    group_size: int

    @classmethod
    def from_config(cls, config: dict, group_size: int) -> Int4RecordLayout:
        inter, hidden = config["moe_intermediate_size"], config["hidden_size"]
        return cls(
            matrices=(
                ("gate_proj", (inter, hidden)),
                ("up_proj", (inter, hidden)),
                ("down_proj", (hidden, inter)),
            ),
            group_size=group_size,
        )

    @property
    def shapes(self) -> dict[str, tuple[int, int]]:
        return dict(self.matrices)

    def weight_offset(self, name: str) -> int:
        offset = 0
        for n, (rows, cols) in self.matrices:
            if n == name:
                return offset
            offset += rows * cols // 2
        raise KeyError(name)

    @property
    def weights_bytes(self) -> int:
        return sum(rows * cols // 2 for _, (rows, cols) in self.matrices)

    def scales_offset(self, name: str) -> int:
        offset = self.weights_bytes
        for n, (rows, cols) in self.matrices:
            if n == name:
                return offset
            offset += rows * (cols // self.group_size) * 2
        raise KeyError(name)

    @property
    def scales_bytes(self) -> int:
        return sum(rows * (cols // self.group_size) * 2 for _, (rows, cols) in self.matrices)

    @property
    def record_size(self) -> int:
        return align_up(self.weights_bytes + self.scales_bytes)

    def to_dict(self) -> dict:
        return {
            "format": FORMAT,
            "group_size": self.group_size,
            "matrices": [{"name": n, "shape": list(s)} for n, s in self.matrices],
            "weights_bytes": self.weights_bytes,
            "scales_bytes": self.scales_bytes,
            "scales_dtype": "float16",
            "record_size": self.record_size,
            "alignment": ALIGNMENT,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Int4RecordLayout:
        if d.get("format") != FORMAT:
            raise ValueError(f"not an int4 layout: {d.get('format')!r}")
        return cls(
            matrices=tuple((m["name"], tuple(m["shape"])) for m in d["matrices"]), group_size=d["group_size"]
        )


def serialize_int4_record(
    layout: Int4RecordLayout, quantized: dict[str, tuple[np.ndarray, np.ndarray]]
) -> bytearray:
    """{matrix: (packed uint8 [rows, cols/2], scales float16 [rows, cols/G])} -> one padded record."""
    record = bytearray(layout.record_size)
    for name, (rows, cols) in layout.matrices:
        packed, scales = quantized[name]
        if packed.shape != (rows, cols // 2) or packed.dtype != np.uint8:
            raise ValueError(
                f"{name}: packed {packed.dtype}{packed.shape}, expected uint8{(rows, cols // 2)}"
            )
        if scales.shape != (rows, cols // layout.group_size):
            raise ValueError(f"{name}: scales {scales.shape}, expected {(rows, cols // layout.group_size)}")
        w_off, s_off = layout.weight_offset(name), layout.scales_offset(name)
        record[w_off : w_off + packed.nbytes] = packed.tobytes()
        s = scales.astype("<f2")
        record[s_off : s_off + s.nbytes] = s.tobytes()
    return record


def parse_int4_record(layout: Int4RecordLayout, buf) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Inverse of serialize_int4_record, as views into `buf`."""
    out = {}
    for name, (rows, cols) in layout.matrices:
        packed = np.frombuffer(buf, dtype=np.uint8, count=rows * cols // 2, offset=layout.weight_offset(name))
        groups = cols // layout.group_size
        scales = np.frombuffer(buf, dtype="<f2", count=rows * groups, offset=layout.scales_offset(name))
        out[name] = (packed.reshape(rows, cols // 2), scales.reshape(rows, groups))
    return out
