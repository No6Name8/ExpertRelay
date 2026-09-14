"""A reduced, hand-rolled MindSpore re-implementation of Qwen1.5-MoE-A2.7B's
architecture (HF model_type "qwen2_moe"), sized down to a handful of layers
and experts so it fits and runs on an 8GB CPU machine.

Parameter names deliberately follow MindFormers' naming vocabulary (see
`decoder.layers.N.self_attention...`, `mlp.router`, `gating`/`hidden`/
`linear_fc2` for the SwiGLU projections, `output_layer`, etc. -- lifted from
the real mindformers/models/qwen3_moe/utils.py weight_mapping) so the
checkpoint this produces reads like a MindFormers-style checkpoint.

One deliberate simplification vs. real MindFormers: MindFormers fuses Q/K/V
into one `linear_qkv` matrix and stacks all experts into two big `weight1`/
`weight2` tensors (see mindformers/checkpoint/converter/convert_op.py) for
its tensor-parallel / expert-parallel training kernels. That fusion is an
internal performance detail for large-scale distributed *training*; it's
irrelevant to what we're validating here (does the conversion + device-split
pipeline work end-to-end), so this model keeps attention and experts UNFUSED
-- one Linear per Q/K/V/O and per expert projection. See docs/setup-notes.md
for the full list of what's real vs. simplified.

The MoE layer's expert dispatch is split-aware: each MoELayer is given a
`local_expert_ids` set. Tokens routed to a local expert are computed in
this process; tokens routed to any other expert id call out through a
`remote_expert_fn(layer_idx, expert_id, hidden_states) -> np.ndarray`
callback -- this is the exact seam the two-process TCP split hooks into.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Set

import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter


def _np32(x) -> Tensor:
    return Tensor(np.asarray(x, dtype=np.float32))


class Linear(nn.Cell):
    """Plain y = x @ W^T (+ b), weight stored HF/PyTorch-style as [out, in]."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.weight = Parameter(ops.zeros((out_features, in_features), ms.float32), name="weight")
        self.bias = Parameter(ops.zeros((out_features,), ms.float32), name="bias") if bias else None

    def construct(self, x):
        y = ops.matmul(x, self.weight.T)
        if self.bias is not None:
            y = y + self.bias
        return y


class RMSNorm(nn.Cell):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = Parameter(ops.ones((hidden_size,), ms.float32), name="weight")
        self.eps = eps

    def construct(self, x):
        variance = ops.mean(ops.square(x), axis=-1, keep_dims=True)
        x = x * ops.rsqrt(variance + self.eps)
        return x * self.weight


class WordEmbeddings(nn.Cell):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.weight = Parameter(ops.zeros((vocab_size, hidden_size), ms.float32), name="weight")

    def construct(self, token_ids):
        return ops.gather(self.weight, token_ids, 0)


class Embedding(nn.Cell):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.word_embeddings = WordEmbeddings(vocab_size, hidden_size)

    def construct(self, token_ids):
        return self.word_embeddings(token_ids)


def _rotary_tables(seq_len: int, head_dim: int, theta: float):
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    t = np.arange(seq_len, dtype=np.float64)
    freqs = np.outer(t, inv_freq)  # (seq, head_dim/2)
    emb = np.concatenate([freqs, freqs], axis=-1)  # (seq, head_dim)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def _rotate_half(x):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return ops.concat([-x2, x1], axis=-1)


class SelfAttention(nn.Cell):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.linear_q = Linear(hidden_size, hidden_size, bias=True)
        self.linear_k = Linear(hidden_size, hidden_size, bias=True)
        self.linear_v = Linear(hidden_size, hidden_size, bias=True)
        self.linear_proj = Linear(hidden_size, hidden_size, bias=False)

    def construct(self, x, cos, sin, causal_mask):
        b, s, h = x.shape
        nh, hd = self.num_heads, self.head_dim

        q = self.linear_q(x).reshape(b, s, nh, hd).transpose(0, 2, 1, 3)
        k = self.linear_k(x).reshape(b, s, nh, hd).transpose(0, 2, 1, 3)
        v = self.linear_v(x).reshape(b, s, nh, hd).transpose(0, 2, 1, 3)

        cos_b = cos.reshape(1, 1, s, hd)
        sin_b = sin.reshape(1, 1, s, hd)
        q = q * cos_b + _rotate_half(q) * sin_b
        k = k * cos_b + _rotate_half(k) * sin_b

        scores = ops.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(hd)
        scores = scores + causal_mask
        probs = ops.softmax(scores, axis=-1)
        out = ops.matmul(probs, v)  # (b, nh, s, hd)
        out = out.transpose(0, 2, 1, 3).reshape(b, s, h)
        return self.linear_proj(out)


class SwiGLUExpert(nn.Cell):
    """gate/up/down MLP, named `gating`/`hidden`/`linear_fc2` per MindFormers' convention."""

    def __init__(self, hidden_size: int, ffn_size: int):
        super().__init__()
        self.gating = Linear(hidden_size, ffn_size, bias=False)
        self.hidden = Linear(hidden_size, ffn_size, bias=False)
        self.linear_fc2 = Linear(ffn_size, hidden_size, bias=False)
        self.act = nn.SiLU()

    def construct(self, x):
        return self.linear_fc2(self.act(self.gating(x)) * self.hidden(x))


class MoELayer(nn.Cell):
    """Router + shared expert + locally-hosted routed experts.

    `local_expert_ids` are the only expert ids this process actually holds
    weights for. Any selected expert id NOT in that set is dispatched via
    `remote_expert_fn` -- set by the coordinator process to a function that
    round-trips the request over the TCP socket to whichever process does
    hold it.
    """

    def __init__(
        self,
        hidden_size: int,
        moe_ffn_size: int,
        shared_ffn_size: int,
        expert_ids: List[int],
        local_expert_ids: Set[int],
        top_k: int,
    ):
        super().__init__()
        self.expert_ids = expert_ids  # full logical expert-id list this router chooses among
        self.local_expert_ids = set(local_expert_ids)
        self.top_k = top_k
        self.router = Linear(hidden_size, len(expert_ids), bias=False)
        self.shared_expert = SwiGLUExpert(hidden_size, shared_ffn_size)
        self.shared_expert_gate = Linear(hidden_size, 1, bias=False)
        self.experts: Dict[int, SwiGLUExpert] = {}
        for eid in expert_ids:
            if eid in self.local_expert_ids:
                cell = SwiGLUExpert(hidden_size, moe_ffn_size)
                setattr(self, f"expert_{eid}", cell)
                self.experts[eid] = cell

        self.remote_expert_fn: Optional[Callable[[int, int, np.ndarray], np.ndarray]] = None
        self.layer_idx: int = -1
        self.remote_calls = 0
        self.local_calls = 0

    def construct(self, x):
        b, s, h = x.shape
        flat = x.reshape(b * s, h)
        flat_np = flat.asnumpy()

        logits = self.router(flat)
        weights = ops.softmax(logits, axis=-1).asnumpy()  # (tokens, num_selected_experts) -- see docs: NOT renormalized (norm_topk_prob=False in real config)
        topk_idx = np.argsort(-weights, axis=-1)[:, : self.top_k]  # (tokens, top_k) indices into expert_ids

        out_np = np.zeros_like(flat_np)
        num_tokens = flat_np.shape[0]
        for t in range(num_tokens):
            token_vec = flat_np[t : t + 1]
            for slot in topk_idx[t]:
                eid = self.expert_ids[slot]
                w = weights[t, slot]
                if eid in self.experts:
                    self.local_calls += 1
                    expert_out = self.experts[eid](Tensor(token_vec, ms.float32)).asnumpy()
                else:
                    self.remote_calls += 1
                    if self.remote_expert_fn is None:
                        raise RuntimeError(
                            f"expert {eid} is not local and no remote_expert_fn is set "
                            f"(layer {self.layer_idx})"
                        )
                    expert_out = self.remote_expert_fn(self.layer_idx, int(eid), token_vec)
                out_np[t] += w * expert_out[0]

        routed_out = Tensor(out_np, ms.float32).reshape(b, s, h)

        shared_out = self.shared_expert(x)
        shared_gate = ops.sigmoid(self.shared_expert_gate(x))
        combined = routed_out + shared_gate * shared_out
        return combined


class DecoderLayer(nn.Cell):
    def __init__(self, hidden_size, num_heads, moe_ffn_size, shared_ffn_size,
                 expert_ids, local_expert_ids, top_k, rms_eps):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, rms_eps)
        self.self_attention = SelfAttention(hidden_size, num_heads)
        self.pre_mlp_layernorm = RMSNorm(hidden_size, rms_eps)
        self.mlp = MoELayer(hidden_size, moe_ffn_size, shared_ffn_size, expert_ids, local_expert_ids, top_k)

    def construct(self, x, cos, sin, causal_mask):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attention(x, cos, sin, causal_mask)
        x = residual + x

        residual = x
        x = self.pre_mlp_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x


class Decoder(nn.Cell):
    def __init__(self, num_layers, hidden_size, num_heads, moe_ffn_size, shared_ffn_size,
                 expert_ids, local_expert_ids_per_layer, top_k, rms_eps):
        super().__init__()
        self.layers = nn.CellList([
            DecoderLayer(
                hidden_size, num_heads, moe_ffn_size, shared_ffn_size,
                expert_ids, local_expert_ids_per_layer[i], top_k, rms_eps,
            )
            for i in range(num_layers)
        ])
        for i, layer in enumerate(self.layers):
            layer.mlp.layer_idx = i
        self.final_layernorm = RMSNorm(hidden_size, rms_eps)

    def construct(self, x, cos, sin, causal_mask):
        for layer in self.layers:
            x = layer(x, cos, sin, causal_mask)
        return self.final_layernorm(x)


class ExpertShardLayer(nn.Cell):
    """Holds just the expert FFNs assigned to this process, for one decoder layer."""

    def __init__(self, hidden_size: int, moe_ffn_size: int, expert_ids: List[int]):
        super().__init__()
        self.experts: Dict[int, SwiGLUExpert] = {}
        for eid in expert_ids:
            cell = SwiGLUExpert(hidden_size, moe_ffn_size)
            setattr(self, f"expert_{eid}", cell)
            self.experts[eid] = cell

    def construct(self, expert_id, x):
        # Not used directly by MindSpore graph tracing -- the TCP server calls
        # into self.experts[eid] directly (see expert_server.py). Kept for
        # completeness / potential batched-graph use later.
        return self.experts[int(expert_id)](x)


class RemoteExpertShard(nn.Cell):
    """Everything a 'Process B'-style device needs: no embedding, no attention,
    no lm_head -- just the expert FFNs it was assigned, addressable by
    (layer_idx, expert_id). This is the whole of what a remote expert host
    needs to hold in RAM.
    """

    def __init__(self, hidden_size: int, moe_ffn_size: int, expert_ids_per_layer: Dict[int, List[int]]):
        super().__init__()
        self.shard_layers = nn.CellList([
            ExpertShardLayer(hidden_size, moe_ffn_size, expert_ids_per_layer.get(i, []))
            for i in sorted(expert_ids_per_layer.keys())
        ])
        self._layer_index = {i: pos for pos, i in enumerate(sorted(expert_ids_per_layer.keys()))}

    def run_expert(self, layer_idx: int, expert_id: int, x_np: np.ndarray) -> np.ndarray:
        pos = self._layer_index[layer_idx]
        layer = self.shard_layers[pos]
        out = layer.experts[expert_id](Tensor(x_np, ms.float32))
        return out.asnumpy()


class ReducedQwenMoe(nn.Cell):
    """Top-level model: embedding -> decoder -> output_layer (lm_head)."""

    def __init__(self, cfg: dict, local_expert_ids_per_layer: List[Set[int]],
                 remote_expert_fn: Optional[Callable] = None):
        super().__init__()
        self.cfg = cfg
        self.embedding = Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.decoder = Decoder(
            num_layers=cfg["num_layers"],
            hidden_size=cfg["hidden_size"],
            num_heads=cfg["num_attention_heads"],
            moe_ffn_size=cfg["moe_intermediate_size"],
            shared_ffn_size=cfg["shared_expert_intermediate_size"],
            expert_ids=cfg["expert_ids"],
            local_expert_ids_per_layer=local_expert_ids_per_layer,
            top_k=cfg["top_k"],
            rms_eps=cfg["rms_norm_eps"],
        )
        self.output_layer = Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)
        if remote_expert_fn is not None:
            self.set_remote_expert_fn(remote_expert_fn)

    def set_remote_expert_fn(self, fn):
        for layer in self.decoder.layers:
            layer.mlp.remote_expert_fn = fn

    def call_stats(self):
        stats = []
        for i, layer in enumerate(self.decoder.layers):
            stats.append({"layer": i, "local_calls": layer.mlp.local_calls, "remote_calls": layer.mlp.remote_calls})
        return stats

    def construct(self, token_ids):
        b, s = token_ids.shape
        hd = self.cfg["hidden_size"] // self.cfg["num_attention_heads"]
        cos_np, sin_np = _rotary_tables(s, hd, self.cfg["rope_theta"])
        cos, sin = Tensor(cos_np), Tensor(sin_np)

        mask_np = np.triu(np.full((s, s), -1e9, dtype=np.float32), k=1)
        causal_mask = Tensor(mask_np).reshape(1, 1, s, s)

        x = self.embedding(token_ids)
        x = self.decoder(x, cos, sin, causal_mask)
        return self.output_layer(x)
