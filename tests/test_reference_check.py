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


def test_reference_resumes_after_a_crash_with_identical_result(tiny, tmp_path, monkeypatch):
    import expertrelay.bench.reference_check as rc

    ckpt_dir, _ = tiny
    sequences = [SEQUENCE, SEQUENCE[:7]]
    uninterrupted = rc.hf_reference_forward(
        Bf16Checkpoint(ckpt_dir), sequences, keep_layers=False, logits=False
    )

    load_layer = rc._load_layer

    def crash_at_layer_2(ckpt, config, layer):
        if layer == 2:
            raise KeyboardInterrupt
        return load_layer(ckpt, config, layer)

    monkeypatch.setattr(rc, "_load_layer", crash_at_layer_2)
    with pytest.raises(KeyboardInterrupt):
        rc.hf_reference_forward(
            Bf16Checkpoint(ckpt_dir), sequences, keep_layers=False, logits=False, resume_dir=tmp_path
        )
    assert [p.name for p in tmp_path.iterdir()] == ["after_layer_01.pt"]
    loaded: list[int] = []
    monkeypatch.setattr(
        rc, "_load_layer", lambda c, cfg, layer: loaded.append(layer) or load_layer(c, cfg, layer)
    )
    resumed = rc.hf_reference_forward(
        Bf16Checkpoint(ckpt_dir), sequences, keep_layers=False, logits=False, resume_dir=tmp_path
    )
    assert loaded == [2]
    for a, b in zip(uninterrupted, resumed, strict=True):
        np.testing.assert_array_equal(a.final_hidden, b.final_hidden)
    with pytest.raises(SystemExit):  # a checkpoint for other sequences is never used
        rc.hf_reference_forward(
            Bf16Checkpoint(ckpt_dir), [SEQUENCE], keep_layers=False, logits=False, resume_dir=tmp_path
        )


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


def test_kl_per_position():
    from expertrelay.bench.reference_check import kl_per_position

    a = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    np.testing.assert_allclose(kl_per_position(a, a), [0.0, 0.0], atol=1e-12)
    p = np.array([0.5, 0.5])
    q = np.array([0.9, 0.1])
    expected = float(np.sum(p * np.log(p / q)))
    assert kl_per_position(np.log(p)[None], np.log(q)[None])[0] == pytest.approx(expected)


@pytest.mark.parametrize("kernel", ["blocked", "fused"])
def test_token_by_token_variant_covers_the_same_positions(tiny, kernel):
    """The decode variants feed the continuation one token at a time; they
    must produce one logits row and one hidden row per position, close to
    the single-prefill result (same weights, different summation order)."""
    from pathlib import Path

    from expertrelay.bench.reference_check import our_traces
    from expertrelay.runtime import int8_linear
    from expertrelay.runtime.generate import RuntimeConfig

    if kernel == "fused":
        pytest.importorskip("numba")
    _, store_dir = tiny
    rt = RuntimeConfig(memory_budget_gb=100, max_seq=32, max_new_tokens=6, prompts_file=Path("unused"),
                       store_dir=store_dir)  # fmt: skip
    before = int8_linear.current_kernel()
    try:
        ((prefill_trace, prefill_logits),) = our_traces(store_dir, rt, [SEQUENCE], [6], "prefill", "blocked")
        ((decode_trace, decode_logits),) = our_traces(store_dir, rt, [SEQUENCE], [6], "decode", kernel)
    finally:
        int8_linear.set_kernel(before)
    assert decode_logits.shape == prefill_logits.shape == (len(SEQUENCE), HF_CONFIG["vocab_size"])
    np.testing.assert_allclose(decode_logits, prefill_logits, rtol=1e-4, atol=1e-4)
    for a, b in zip(decode_trace.hidden_after_layer, prefill_trace.hidden_after_layer, strict=True):
        assert a.shape == b.shape
    np.testing.assert_array_equal(decode_trace.selected_experts[0], prefill_trace.selected_experts[0])
