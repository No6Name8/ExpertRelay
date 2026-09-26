"""Expert-usage traces: what the router did, for every token at every layer.

Written by the runtime when tracing is on (runtime.generate --trace-dir),
read by bench/phase3_analysis.py and cache/simulator.py. The on-disk
format is specified in docs/trace-format.md; keep the two in sync.

Tracing must not change model outputs. The recorder only receives values
the forward pass already computed, plus one extra computation that is used
for nothing but the trace: the Fate-style cross-layer prediction.

Fate-style prediction. Z. Fang, Z. Hong, Y. Huang, Y. Lyu, W. Chen, Y. Yu,
F. Yu, Z. Zheng, "Fate: Fast Edge Inference of Mixture-of-Experts Models via
Cross-Layer Gate", arXiv:2502.12224 (2025); ACM Web Conference 2026,
doi:10.1145/3774904.3792527. Fate's observation is that the gate INPUT of
layer L (the hidden state after layer L's post-attention norm, which layer
L's router sees) is close enough to layer L+1's gate input that applying
layer L+1's router to it predicts layer L+1's experts before layer L+1
runs. We record that prediction for every layer >= 1. We do not implement
Fate's prefetching system or its caching strategy, only the predictor.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

MAGIC = b"ERTRACE1"
FORMAT_VERSION = 1
NO_EXPERT = 255  # "no value" for u8 expert-id fields
PHASE_PREFILL, PHASE_DECODE = 0, 1


def record_dtype(num_experts: int, top_k: int) -> np.dtype:
    """One record per (token, layer). Packed, little-endian, no padding."""
    if num_experts >= NO_EXPERT:
        raise ValueError(f"u8 expert ids hold at most {NO_EXPERT - 1} experts, got {num_experts}")
    return np.dtype(
        [
            ("position", "<u4"),  # token position in the sequence (0 = first prompt token)
            ("token", "<u4"),  # the token id at that position
            ("layer", "u1"),
            ("phase", "u1"),  # PHASE_PREFILL or PHASE_DECODE
            ("experts", "u1", (top_k,)),  # chosen experts, highest router weight first
            ("weights", "<f4", (top_k,)),  # their router softmax probabilities (not renormalized)
            ("logits", "<f2", (num_experts,)),  # full router logits for this layer
            (
                "fate_experts",
                "u1",
                (top_k,),
            ),  # Fate-style predicted top-k for this layer; NO_EXPERT at layer 0
            (
                "fate_logits",
                "<f2",
                (num_experts,),
            ),  # this layer's router on the previous layer's gate input; 0 at layer 0
            (
                "prev_token_experts",
                "u1",
                (top_k,),
            ),  # previous token's experts at this layer; NO_EXPERT if none
        ]
    )


def _top_k_desc(values: np.ndarray, k: int) -> np.ndarray:
    part = np.argpartition(-values, k - 1, axis=-1)[:, :k]
    order = np.argsort(-np.take_along_axis(values, part, axis=-1), axis=-1, kind="stable")
    return np.take_along_axis(part, order, axis=-1)


class ExpertTraceWriter:
    """Collects one sequence's records and appends them to a trace file after
    every forward call, so an interrupted run keeps everything already done."""

    def __init__(self, path: Path, *, num_layers: int, num_experts: int, top_k: int, metadata: dict):
        self.path = Path(path)
        self.num_layers, self.num_experts, self.top_k = num_layers, num_experts, top_k
        self.dtype = record_dtype(num_experts, top_k)
        header = {
            "format_version": FORMAT_VERSION,
            "num_layers": num_layers,
            "num_experts": num_experts,
            "top_k": top_k,
            "record_itemsize": self.dtype.itemsize,
            **metadata,
        }
        blob = json.dumps(header, ensure_ascii=False).encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "wb") as f:
            f.write(MAGIC + struct.pack("<I", len(blob)) + blob)
        self._prev_experts = np.full((num_layers, top_k), NO_EXPERT, dtype=np.uint8)
        self._pending_fate: np.ndarray | None = None  # logits predicted for the NEXT layer
        self._call: np.ndarray | None = None

    def begin(self, positions: np.ndarray, tokens: np.ndarray, phase: int) -> None:
        n = len(positions)
        self._call = np.zeros(n * self.num_layers, dtype=self.dtype)
        self._call["position"] = np.repeat(positions, self.num_layers)
        self._call["token"] = np.repeat(tokens, self.num_layers)
        self._call["layer"] = np.tile(np.arange(self.num_layers), n)
        self._call["phase"] = phase
        self._pending_fate = None

    def layer(
        self,
        layer: int,
        logits: np.ndarray,
        selected: np.ndarray,
        weights: np.ndarray,
        next_layer_fate_logits: np.ndarray | None,
    ) -> None:
        """Called once per layer per forward call. Arrays are [n, ...] over the
        tokens of this call. next_layer_fate_logits is layer L+1's router
        applied to layer L's gate input (None at the last layer)."""
        rows = self._call[layer :: self.num_layers]  # a view: this layer's record for each token
        rows["experts"] = selected
        rows["weights"] = weights
        rows["logits"] = logits
        if self._pending_fate is None:
            rows["fate_experts"] = NO_EXPERT
        else:
            rows["fate_logits"] = self._pending_fate
            rows["fate_experts"] = _top_k_desc(self._pending_fate, self.top_k)
        prev = np.concatenate([self._prev_experts[layer][None], selected[:-1]]).astype(np.uint8)
        rows["prev_token_experts"] = prev
        self._prev_experts[layer] = selected[-1]
        self._pending_fate = next_layer_fate_logits

    def end(self) -> None:
        with open(self.path, "ab") as f:
            f.write(self._call.tobytes())
        self._call = None


def read_trace(path: Path) -> tuple[dict, np.ndarray]:
    """(header, records). Raises if the file is truncated mid-record: a
    partial trace from an interrupted run must be rerun, not half-analyzed."""
    data = Path(path).read_bytes()
    if data[:8] != MAGIC:
        raise ValueError(f"{path}: not an expert trace (bad magic)")
    (hlen,) = struct.unpack("<I", data[8:12])
    header = json.loads(data[12 : 12 + hlen].decode("utf-8"))
    if header["format_version"] != FORMAT_VERSION:
        raise ValueError(f"{path}: unsupported trace format_version {header['format_version']}")
    dtype = record_dtype(header["num_experts"], header["top_k"])
    body = data[12 + hlen :]
    if len(body) % dtype.itemsize:
        raise ValueError(f"{path}: truncated ({len(body)} bytes is not a whole number of records)")
    return header, np.frombuffer(body, dtype=dtype)
