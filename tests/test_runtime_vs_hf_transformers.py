"""Our numpy forward pass vs. Hugging Face transformers' Qwen2MoE code.

A tiny random Qwen2MoE goes through the real store builder (so int8). The
DEQUANTIZED int8 weights are then loaded into HF's Qwen2MoeForCausalLM,
so both sides compute the same model in f32 and any difference comes from
the implementation, not from quantization. Covers grouped KV heads, RoPE,
routing (softmax then top-k, no renormalization), the gated shared expert,
and KV-cache decoding vs. HF's full recompute.

This is the automated, small-scale version of bench/reference_check.py,
which compares the real 24-layer model against the original bf16 weights.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from store_helpers import build_test_store, random_checkpoint  # noqa: E402

from expertrelay.runtime.generate import model_config  # noqa: E402
from expertrelay.runtime.qwen_moe import KVCache, QwenMoe  # noqa: E402
from expertrelay.runtime.weights import Int8Matrix, RamExpertSource, ResidentWeights  # noqa: E402

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="store finalize verifies via unbuffered reads"
)

HF_CONFIG = {
    "vocab_size": 96,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,  # grouped-query: exercises the KV head repeat
    "num_experts": 6,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 16,
    "shared_expert_intermediate_size": 24,
    "norm_topk_prob": False,
    "rope_theta": 10000.0,
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 64,
    "decoder_sparse_step": 1,
    "mlp_only_layers": [],
    "use_sliding_window": False,
    "tie_word_embeddings": False,
}
TOKENS = [5, 17, 3, 88, 41, 41, 9, 60, 2, 33]


def _dequant(t) -> np.ndarray:
    return t.q.astype(np.float32) * t.scales[:, None] if isinstance(t, Int8Matrix) else np.asarray(t)


@pytest.fixture(scope="module")
def both_models(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    store = tmp_path_factory.mktemp("hfstore")
    hf_config = transformers.Qwen2MoeConfig(**HF_CONFIG)
    hf_config._attn_implementation = "eager"
    config_dict = hf_config.to_dict()
    arrays = random_checkpoint(config_dict, seed=0)
    build_test_store(store, config_dict, arrays, mp)
    mp.undo()

    resident = ResidentWeights.load(store)
    experts = RamExpertSource(store)
    ours = QwenMoe(model_config(store), resident, experts)

    torch.manual_seed(0)
    hf = transformers.Qwen2MoeForCausalLM(hf_config).eval().float()
    state = {}
    for name, value in hf.state_dict().items():
        if ".mlp.experts." in name:
            layer = int(name.split(".")[2])
            per_expert = []
            for e in range(HF_CONFIG["num_experts"]):
                w = experts.load(layer, e)
                if name.endswith("gate_up_proj"):  # HF fuses [gate; up] along the output dim
                    per_expert.append(np.concatenate([_dequant(w["gate_proj"]), _dequant(w["up_proj"])]))
                else:
                    per_expert.append(_dequant(w["down_proj"]).copy())
            state[name] = torch.from_numpy(np.stack(per_expert))
        else:
            state[name] = torch.from_numpy(_dequant(resident[name]).reshape(value.shape).copy())
    hf.load_state_dict(state, strict=True)
    return ours, hf


def _hf_logits(hf, tokens: list[int]) -> np.ndarray:
    with torch.no_grad():
        return hf(torch.tensor([tokens])).logits[0].numpy()


def test_prefill_logits_match_hf(both_models):
    ours, hf = both_models
    cache = KVCache(ours.c, max_seq=32)
    logits, _ = ours.forward(np.array(TOKENS), cache, all_logits=True)
    expected = _hf_logits(hf, TOKENS)
    np.testing.assert_allclose(logits, expected, rtol=1e-4, atol=1e-4)
    np.testing.assert_array_equal(logits.argmax(-1), expected.argmax(-1))


@pytest.mark.parametrize("kernel", ["blocked", "fused"])
def test_kv_cache_decode_matches_hf_full_recompute(both_models, kernel):
    """Decode is where the kernels differ (the fused one sums in another
    order); both must match transformers to the same tolerance."""
    from expertrelay.runtime import int8_linear

    if kernel == "fused":
        pytest.importorskip("numba")
    before = int8_linear.current_kernel()
    int8_linear.set_kernel(kernel)
    try:
        _decode_matches_hf(both_models)
    finally:
        int8_linear.set_kernel(before)


def _decode_matches_hf(both_models):
    ours, hf = both_models
    cache = KVCache(ours.c, max_seq=32)
    seq = TOKENS[:4]
    logits, _ = ours.forward(np.array(seq), cache)
    for _ in range(5):  # greedy: each step feeds ONE token through the cache
        nxt = int(np.argmax(logits))
        expected_last = _hf_logits(hf, seq)[-1]
        np.testing.assert_allclose(logits, expected_last, rtol=1e-4, atol=1e-4)
        assert nxt == int(np.argmax(expected_last))
        seq = seq + [nxt]
        logits, _ = ours.forward(np.array([nxt]), cache)
