"""Smoke test for expertrelay.store.convert_qwen_moe's pure/offline logic
(no network -- see test_store_fetch_hf_tensors.py's docstring for why)."""

from __future__ import annotations

from expertrelay.store.convert_qwen_moe import build_wanted_tensor_names


def test_build_wanted_tensor_names_counts_and_contents():
    names = build_wanted_tensor_names(num_layers=2, expert_ids=[0, 1, 2, 3])

    # 3 global tensors + 2 layers * (14 non-expert + 4 experts * 3 = 26) = 3 + 52 = 55
    assert len(names) == 55
    assert len(names) == len(set(names))  # no duplicates

    assert "model.embed_tokens.weight" in names
    assert "lm_head.weight" in names
    assert "model.norm.weight" in names
    assert "model.layers.0.self_attn.q_proj.weight" in names
    assert "model.layers.1.mlp.experts.3.down_proj.weight" in names
    # only the requested layers/experts, nothing beyond them
    assert "model.layers.2.input_layernorm.weight" not in names
    assert "model.layers.0.mlp.experts.4.gate_proj.weight" not in names


def test_build_wanted_tensor_names_scales_with_fewer_experts():
    names_4 = build_wanted_tensor_names(num_layers=1, expert_ids=[0, 1, 2, 3])
    names_2 = build_wanted_tensor_names(num_layers=1, expert_ids=[0, 1])
    assert len(names_2) < len(names_4)
