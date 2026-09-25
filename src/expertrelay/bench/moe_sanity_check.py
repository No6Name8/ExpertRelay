"""Minimal MoE forward-pass + greedy-decode sanity check on MindSpore CPU.

Not a pretrained model -- this is a small, randomly-initialized decoder-only
transformer with a Mixture-of-Experts feed-forward block (top-k router over
N experts). Purpose: prove the MoE mechanics (routing, expert dispatch,
combine) run correctly end-to-end on this machine's MindSpore/CPU install
before attempting any real checkpoint conversion or device-splitting work.
"""

from __future__ import annotations

import mindspore as ms
import numpy as np
from mindspore import Tensor, nn, ops

VOCAB_SIZE = 64
HIDDEN_SIZE = 32
NUM_EXPERTS = 4
TOP_K = 2
NUM_LAYERS = 2
SEQ_LEN = 8


class Router(nn.Cell):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int):
        super().__init__()
        self.gate = nn.Dense(hidden_size, num_experts)
        self.top_k = top_k
        self.num_experts = num_experts

    def construct(self, x: Tensor):
        logits = self.gate(x)
        weights = ops.softmax(logits, axis=-1)
        topk_weights, topk_idx = ops.top_k(weights, self.top_k)
        topk_weights = topk_weights / topk_weights.sum(axis=-1, keepdims=True)
        return topk_weights, topk_idx


class Expert(nn.Cell):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Dense(hidden_size, hidden_size * 2)
        self.act = nn.GELU()
        self.fc2 = nn.Dense(hidden_size * 2, hidden_size)

    def construct(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class MoELayer(nn.Cell):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int):
        super().__init__()
        self.router = Router(hidden_size, num_experts, top_k)
        self.experts = nn.CellList([Expert(hidden_size) for _ in range(num_experts)])
        self.num_experts = num_experts
        self.top_k = top_k

    def construct(self, x: Tensor) -> Tensor:
        # x: (batch, seq, hidden)
        b, s, h = x.shape
        flat = x.reshape(b * s, h)
        topk_weights, topk_idx = self.router(flat)  # (b*s, top_k)

        out = ops.zeros_like(flat)
        for e in range(self.num_experts):
            expert_out = self.experts[e](flat)  # (b*s, h), computed for all tokens (dense sanity check)
            mask = (topk_idx == e).astype(ms.float32)  # (b*s, top_k)
            token_weight = (mask * topk_weights).sum(axis=-1, keepdims=True)  # (b*s, 1)
            out = out + expert_out * token_weight
        return out.reshape(b, s, h)


class TinyMoEDecoder(nn.Cell):
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int, num_experts: int, top_k: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.moe_layers = nn.CellList([MoELayer(hidden_size, num_experts, top_k) for _ in range(num_layers)])
        self.norm = nn.LayerNorm([hidden_size])
        self.head = nn.Dense(hidden_size, vocab_size)

    def construct(self, token_ids: Tensor) -> Tensor:
        x = self.embed(token_ids)
        for layer in self.moe_layers:
            x = x + layer(x)  # residual
        x = self.norm(x)
        return self.head(x)


def main() -> None:
    ms.set_context(device_target="CPU")
    ms.set_seed(0)

    model = TinyMoEDecoder(VOCAB_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_EXPERTS, TOP_K)
    model.set_train(False)

    prompt = Tensor(np.random.randint(0, VOCAB_SIZE, size=(1, SEQ_LEN)), ms.int32)
    print("Prompt token ids:", prompt.asnumpy().tolist())

    generated = prompt
    for _ in range(5):
        logits = model(generated)
        next_logits = logits[:, -1, :]
        next_token = ops.argmax(next_logits, dim=-1).reshape(1, 1).astype(ms.int32)
        generated = ops.concat([generated, next_token], axis=1)

    print("Generated token ids:", generated.asnumpy().tolist())
    print("MoE sanity check PASSED: forward pass + greedy decode ran without error.")


if __name__ == "__main__":
    main()
