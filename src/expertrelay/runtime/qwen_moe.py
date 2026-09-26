"""Qwen1.5-MoE-A2.7B (HF model_type "qwen2_moe") forward pass in numpy, from
the int8 expert store.

The arithmetic goes through a swappable compute backend (runtime.backends:
numpy by default on CPU, MindSpore f32 for Ascend). This module holds the
model structure and the cheap elementwise parts shared by every backend.

The math follows Hugging Face transformers' Qwen2Moe implementation, which
is also what the reference check compares against (bench.reference_check):
  - RMSNorm in f32
  - RoPE tables computed in f32 from inv_freq = 1 / theta^(2i/d), rotate_half
    convention
  - router: softmax over all experts in f32, then top-k. norm_topk_prob is
    False for this model, so the top-k weights are NOT renormalized
  - shared expert added with a sigmoid gate
  - experts run in ascending expert-id order, each on all of its tokens at
    once, like HF's loop. In prefill that also means each expert is loaded
    once per forward call, not once per token
Weights are int8 with per-row scales (store.quantize); activations and
accumulation are f32.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from expertrelay.runtime.backends import Backend, NumpyBackend
from expertrelay.runtime.expert_trace import PHASE_DECODE, PHASE_PREFILL, ExpertTraceWriter
from expertrelay.runtime.weights import EMBEDDING, ExpertSource, Int8Matrix, ResidentWeights


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    num_experts: int
    top_k: int
    norm_topk_prob: bool
    rms_norm_eps: float
    rope_theta: float

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @classmethod
    def from_hf(cls, c: dict) -> ModelConfig:
        if c.get("use_sliding_window"):
            raise NotImplementedError("sliding-window attention is not implemented")
        if c.get("mlp_only_layers") or c.get("decoder_sparse_step", 1) != 1:
            raise NotImplementedError("dense (non-MoE) decoder layers are not implemented")
        return cls(
            vocab_size=c["vocab_size"],
            hidden_size=c["hidden_size"],
            num_layers=c["num_hidden_layers"],
            num_heads=c["num_attention_heads"],
            num_kv_heads=c["num_key_value_heads"],
            num_experts=c["num_experts"],
            top_k=c["num_experts_per_tok"],
            norm_topk_prob=c["norm_topk_prob"],
            rms_norm_eps=c["rms_norm_eps"],
            rope_theta=_rope_theta(c),
        )


def _rope_theta(c: dict) -> float:
    """Checkpoint config.json has a top-level rope_theta; transformers 5.x
    configs nest it in rope_parameters. Only plain (unscaled) RoPE is
    implemented, so any scaling type is rejected rather than ignored."""
    params = c.get("rope_parameters") or {}
    scaling = c.get("rope_scaling") or {}
    kind = params.get("rope_type", scaling.get("rope_type", scaling.get("type", "default")))
    if kind != "default":
        raise NotImplementedError(f"RoPE type {kind!r} is not implemented")
    return float(c["rope_theta"] if "rope_theta" in c else params["rope_theta"])


class KVCache:
    def __init__(self, config: ModelConfig, max_seq: int):
        shape = (config.num_layers, max_seq, config.num_kv_heads, config.head_dim)
        self.k = np.zeros(shape, dtype=np.float32)
        self.v = np.zeros(shape, dtype=np.float32)
        self.length = 0
        self.max_seq = max_seq

    @property
    def nbytes(self) -> int:
        return self.k.nbytes + self.v.nbytes

    @staticmethod
    def bytes_for(config: ModelConfig, max_seq: int) -> int:
        return 2 * config.num_layers * max_seq * config.num_kv_heads * config.head_dim * 4


@dataclass
class ForwardTrace:
    """Optional per-call capture, for the reference check and tests."""

    hidden_after_layer: list[np.ndarray] = field(default_factory=list)  # [n, hidden] per layer
    selected_experts: list[np.ndarray] = field(default_factory=list)  # [n, top_k] per layer
    final_hidden: np.ndarray | None = None


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    variance = np.mean(x * x, axis=-1, keepdims=True)
    return weight * (x * (1.0 / np.sqrt(variance + eps)))


def rope_tables(positions: np.ndarray, head_dim: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """cos/sin [n, head_dim] in f32, computed the way HF's rotary embedding does."""
    inv_freq = (
        1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.int64).astype(np.float32) / head_dim))
    ).astype(np.float32)
    freqs = positions.astype(np.float32)[:, None] * inv_freq[None, :]
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def _rotate_half(x: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _top_k_desc(probs: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest per row, largest first (torch.topk order)."""
    part = np.argpartition(-probs, k - 1, axis=-1)[:, :k]
    order = np.argsort(-np.take_along_axis(probs, part, axis=-1), axis=-1, kind="stable")
    return np.take_along_axis(part, order, axis=-1)


@dataclass
class StepTimings:
    """Where one forward call's wall time went."""

    total_seconds: float = 0.0
    expert_read_seconds: float = 0.0
    expert_loads: int = 0
    expert_bytes: int = 0

    @property
    def compute_seconds(self) -> float:
        return self.total_seconds - self.expert_read_seconds


class QwenMoe:
    def __init__(
        self,
        config: ModelConfig,
        resident: ResidentWeights,
        experts: ExpertSource,
        backend: Backend | None = None,
    ):
        self.c = config
        self.resident = resident
        self.experts = experts
        self.backend = backend or NumpyBackend()

    def _w(self, name: str):
        return self.resident[name]

    def _linear(self, name: str, x: np.ndarray, bias: str | None = None) -> np.ndarray:
        w = self._w(name)
        b = self._w(bias) if bias else None
        if isinstance(w, Int8Matrix):
            return self.backend.int8_linear(x, w.q, w.scales, b)
        return self.backend.linear(x, w, b)

    def _expert_linear(self, w: Int8Matrix, x: np.ndarray) -> np.ndarray:
        return self.backend.int8_linear(x, w.q, w.scales)

    def _attention(
        self, layer: int, h: np.ndarray, cos: np.ndarray, sin: np.ndarray, cache: KVCache
    ) -> np.ndarray:
        c, p = self.c, f"model.layers.{layer}.self_attn."
        n, hd = h.shape[0], c.head_dim
        q = self._linear(p + "q_proj.weight", h, p + "q_proj.bias").reshape(n, c.num_heads, hd)
        k = self._linear(p + "k_proj.weight", h, p + "k_proj.bias").reshape(n, c.num_kv_heads, hd)
        v = self._linear(p + "v_proj.weight", h, p + "v_proj.bias").reshape(n, c.num_kv_heads, hd)
        q = q * cos[:, None, :] + _rotate_half(q) * sin[:, None, :]
        k = k * cos[:, None, :] + _rotate_half(k) * sin[:, None, :]

        start = cache.length
        end = start + n
        if end > cache.max_seq:
            raise ValueError(f"sequence length {end} exceeds the KV cache ({cache.max_seq})")
        cache.k[layer, start:end] = k
        cache.v[layer, start:end] = v
        keys, values = cache.k[layer, :end], cache.v[layer, :end]
        if c.num_kv_heads != c.num_heads:
            rep = c.num_heads // c.num_kv_heads
            keys, values = np.repeat(keys, rep, axis=1), np.repeat(values, rep, axis=1)

        # causal: query at absolute position start+i sees keys 0..start+i
        allowed = np.arange(end)[None, :] <= (start + np.arange(n))[:, None]
        out = self.backend.attention(q, keys, values, allowed).reshape(n, c.hidden_size)
        return self._linear(p + "o_proj.weight", out)

    def _moe(
        self,
        layer: int,
        h: np.ndarray,
        timings: StepTimings,
        trace: ForwardTrace | None,
        expert_trace: ExpertTraceWriter | None,
    ) -> np.ndarray:
        c, p = self.c, f"model.layers.{layer}.mlp."
        n = h.shape[0]
        router_logits = self._linear(p + "gate.weight", h)
        probs = _softmax(router_logits)
        selected = _top_k_desc(probs, c.top_k)
        weights = np.take_along_axis(probs, selected, axis=-1)
        if c.norm_topk_prob:
            weights = weights / weights.sum(axis=-1, keepdims=True)
        if trace is not None:
            trace.selected_experts.append(selected.copy())
        if expert_trace is not None:
            # Fate-style prediction (see runtime.expert_trace for the citation):
            # the NEXT layer's router applied to THIS layer's gate input. Only
            # the trace uses it; it doesn't feed back into the forward pass.
            fate = (
                self._linear(f"model.layers.{layer + 1}.mlp.gate.weight", h)
                if layer + 1 < c.num_layers
                else None
            )
            expert_trace.layer(layer, router_logits, selected, weights, fate)

        routed = np.zeros((n, c.hidden_size), dtype=np.float32)
        for expert in np.unique(selected):
            token_idx, slot = np.nonzero(selected == expert)
            before_s, before_b = self.experts.stats.read_seconds, self.experts.stats.bytes_read
            w = self.experts.load(layer, int(expert))
            timings.expert_read_seconds += self.experts.stats.read_seconds - before_s
            timings.expert_bytes += self.experts.stats.bytes_read - before_b
            timings.expert_loads += 1
            x = h[token_idx]
            y = self._expert_linear(
                w["down_proj"],
                _silu(self._expert_linear(w["gate_proj"], x)) * self._expert_linear(w["up_proj"], x),
            )
            np.add.at(routed, token_idx, y * weights[token_idx, slot][:, None])
            del w  # nothing kept: the next use of this expert reads it again

        shared = self._linear(
            p + "shared_expert.down_proj.weight",
            _silu(self._linear(p + "shared_expert.gate_proj.weight", h))
            * self._linear(p + "shared_expert.up_proj.weight", h),
        )
        gate = _sigmoid(self._linear(p + "shared_expert_gate.weight", h))
        return routed + gate * shared

    def forward(
        self,
        token_ids: np.ndarray,
        cache: KVCache,
        *,
        all_logits: bool = False,
        trace: ForwardTrace | None = None,
        expert_trace: ExpertTraceWriter | None = None,
    ) -> tuple[np.ndarray, StepTimings]:
        """Run `token_ids` (appended after what's already in `cache`).
        Returns logits for the last position (or all positions) and timings."""
        t0 = time.perf_counter()
        timings = StepTimings()
        c = self.c
        token_ids = np.asarray(token_ids, dtype=np.int64)
        emb = self._w(EMBEDDING)
        x = (
            emb.dequantized_rows(token_ids)
            if isinstance(emb, Int8Matrix)
            else np.asarray(emb[token_ids], np.float32)
        )

        positions = cache.length + np.arange(len(token_ids))
        if expert_trace is not None:
            expert_trace.begin(positions, token_ids, PHASE_PREFILL if cache.length == 0 else PHASE_DECODE)
        cos, sin = rope_tables(positions, c.head_dim, c.rope_theta)
        for layer in range(c.num_layers):
            p = f"model.layers.{layer}."
            h = rms_norm(x, self._w(p + "input_layernorm.weight"), c.rms_norm_eps)
            x = x + self._attention(layer, h, cos, sin, cache)
            h = rms_norm(x, self._w(p + "post_attention_layernorm.weight"), c.rms_norm_eps)
            x = x + self._moe(layer, h, timings, trace, expert_trace)
            if trace is not None:
                trace.hidden_after_layer.append(x.copy())
        cache.length += len(token_ids)
        if expert_trace is not None:
            expert_trace.end()

        final = rms_norm(x if all_logits else x[-1:], self._w("model.norm.weight"), c.rms_norm_eps)
        if trace is not None:
            trace.final_hidden = final.copy()
        logits = self._linear("lm_head.weight", final)
        timings.total_seconds = time.perf_counter() - t0
        return (logits if all_logits else logits[0]), timings
