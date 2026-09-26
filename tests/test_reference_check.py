"""Tests for the reference-check harness itself (bench/reference_check.py).

The real check compares our runtime with HF transformers on the 28.6 GB
bf16 checkpoint. Before trusting its numbers, the harness has to be right:
  1. Its layer-by-layer driver (lazy expert banks, per-layer loading) must
     reproduce transformers' own full-model forward pass exactly.
  2. Its comparison must report small errors and high agreement for a
     correct int8 model, and must catch a wrong one.
Both run on a tiny random Qwen2MoE saved in real HF checkpoint format.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
safetensors_torch = pytest.importorskip("safetensors.torch")

from store_helpers import build_test_store, random_checkpoint  # noqa: E402

from expertrelay.bench.reference_check import (  # noqa: E402
    Bf16Checkpoint,
    compare,
    hf_reference_forward,
    lm_head_study,
)
from expertrelay.runtime.generate import model_config  # noqa: E402
from expertrelay.runtime.qwen_moe import ForwardTrace, KVCache, QwenMoe  # noqa: E402
from expertrelay.runtime.weights import ResidentWeights, UnbufferedExpertSource  # noqa: E402

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="store finalize verifies via unbuffered reads"
)

HF_CONFIG = {
    "vocab_size": 80,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
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
SEQUENCE = [1, 7, 42, 3, 3, 19, 64, 5, 27, 11, 50, 2]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """(checkpoint dir in HF format, int8 store built from the same weights)."""
    config = transformers.Qwen2MoeConfig(**HF_CONFIG)
    arrays = random_checkpoint(config.to_dict(), seed=3)
    # round to bf16 first, so the checkpoint and the store see identical inputs
    arrays = {k: torch.from_numpy(v).to(torch.bfloat16).float().numpy() for k, v in arrays.items()}

    ckpt_dir = tmp_path_factory.mktemp("ckpt")
    safetensors_torch.save_file(
        {k: torch.from_numpy(v).to(torch.bfloat16).contiguous() for k, v in arrays.items()},
        str(ckpt_dir / "model.safetensors"),
    )
    (ckpt_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(arrays, "model.safetensors")})
    )
    config.save_pretrained(ckpt_dir)

    store_dir = tmp_path_factory.mktemp("store")
    mp = pytest.MonkeyPatch()
    build_test_store(store_dir, config.to_dict(), arrays, mp)
    mp.undo()
    return ckpt_dir, store_dir


def test_layer_by_layer_driver_reproduces_transformers_full_forward(tiny):
    ckpt_dir, _ = tiny
    ref = hf_reference_forward(Bf16Checkpoint(ckpt_dir), [SEQUENCE])[0]

    full = transformers.Qwen2MoeForCausalLM.from_pretrained(
        ckpt_dir, dtype=torch.float32, attn_implementation="eager"
    )
    with torch.no_grad():
        out = full(torch.tensor([SEQUENCE]), output_hidden_states=True)
    np.testing.assert_allclose(ref.logits, out.logits[0].numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ref.final_hidden, out.hidden_states[-1][0].numpy(), rtol=1e-5, atol=1e-5)
    assert len(ref.hidden_after_layer) == HF_CONFIG["num_hidden_layers"]
    assert all(s.shape == (len(SEQUENCE), HF_CONFIG["num_experts_per_tok"]) for s in ref.selected_experts)


def _ours(store_dir) -> tuple[ForwardTrace, np.ndarray]:
    model = QwenMoe(
        model_config(store_dir), ResidentWeights.load(store_dir), UnbufferedExpertSource(store_dir)
    )
    trace = ForwardTrace()
    logits, _ = model.forward(np.array(SEQUENCE), KVCache(model.c, 32), all_logits=True, trace=trace)
    model.experts.close()
    return trace, logits


def test_comparison_of_correct_int8_model_shows_small_error(tiny):
    ckpt_dir, store_dir = tiny
    ckpt = Bf16Checkpoint(ckpt_dir)
    ref = hf_reference_forward(ckpt, [SEQUENCE])[0]
    trace, logits = _ours(store_dir)

    result = compare(trace, logits, ref, prompt_len=6)
    # int8 weights: ~1% reconstruction error per tensor, so a few % downstream at most
    assert max(result["hidden_rel_error_per_layer"]) < 0.05
    assert result["logits_rel_error"] < 0.05
    assert result["top1_agreement_all_positions"] >= 0.8
    study = lm_head_study(ckpt, trace, logits, ref)
    assert set(study) >= {"top1_agreement_int8_lm_head", "top1_agreement_fp16_lm_head"}


def test_comparison_catches_a_wrong_model(tiny):
    ckpt_dir, store_dir = tiny
    ref = hf_reference_forward(Bf16Checkpoint(ckpt_dir), [SEQUENCE])[0]
    trace, logits = _ours(store_dir)
    trace.hidden_after_layer[1] = trace.hidden_after_layer[1] * 1.5  # a broken layer 1
    shuffled = logits[:, np.random.default_rng(0).permutation(logits.shape[1])]  # a broken lm_head

    result = compare(trace, shuffled, ref, prompt_len=6)
    assert result["hidden_rel_error_per_layer"][1] > 0.3
    assert result["top1_agreement_all_positions"] < 0.5
