"""Build a real expert store from in-memory checkpoint arrays, for tests.

Runs the actual store builder (quantize, serialize, write, journal,
finalize, verify). Only the network fetch is replaced by a stub that
serves the arrays.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from expertrelay.store import build_store
from expertrelay.store.fetch_hf_tensors import TensorLocation
from expertrelay.store.layout import ExpertRecordLayout, parse_expert_tensor_name, plan_resident_layout


def random_checkpoint(config: dict, seed: int = 0) -> dict[str, np.ndarray]:
    """Checkpoint-format (per-expert, HF names) random weights for a Qwen2MoE config."""
    rng = np.random.default_rng(seed)
    h, i_moe, i_sh = (
        config["hidden_size"],
        config["moe_intermediate_size"],
        config["shared_expert_intermediate_size"],
    )
    heads, kv, vocab = config["num_attention_heads"], config["num_key_value_heads"], config["vocab_size"]
    hd = h // heads

    def w(*shape, scale=0.1):
        return rng.normal(0, scale, shape).astype(np.float32)

    arrays = {
        "model.embed_tokens.weight": w(vocab, h, scale=1.0),
        "lm_head.weight": w(vocab, h),
        "model.norm.weight": 1 + w(h),
    }
    for layer in range(config["num_hidden_layers"]):
        p = f"model.layers.{layer}."
        arrays |= {
            p + "input_layernorm.weight": 1 + w(h),
            p + "post_attention_layernorm.weight": 1 + w(h),
            p + "self_attn.q_proj.weight": w(heads * hd, h),
            p + "self_attn.q_proj.bias": w(heads * hd),
            p + "self_attn.k_proj.weight": w(kv * hd, h),
            p + "self_attn.k_proj.bias": w(kv * hd),
            p + "self_attn.v_proj.weight": w(kv * hd, h),
            p + "self_attn.v_proj.bias": w(kv * hd),
            p + "self_attn.o_proj.weight": w(h, heads * hd),
            p + "mlp.gate.weight": w(config["num_experts"], h, scale=1.0),
            p + "mlp.shared_expert.gate_proj.weight": w(i_sh, h),
            p + "mlp.shared_expert.up_proj.weight": w(i_sh, h),
            p + "mlp.shared_expert.down_proj.weight": w(h, i_sh),
            p + "mlp.shared_expert_gate.weight": w(1, h),
        }
        for e in range(config["num_experts"]):
            arrays |= {
                p + f"mlp.experts.{e}.gate_proj.weight": w(i_moe, h),
                p + f"mlp.experts.{e}.up_proj.weight": w(i_moe, h),
                p + f"mlp.experts.{e}.down_proj.weight": w(h, i_moe),
            }
    return arrays


def build_test_store(
    store_dir: Path, config: dict, arrays: dict[str, np.ndarray], monkeypatch
) -> build_store.Plan:
    locations = {
        n: TensorLocation(name=n, url="fake://", dtype="BF16", shape=a.shape, data_start=0, nbytes=a.size * 2)
        for n, a in arrays.items()
    }
    resident_names = sorted(
        (n for n in arrays if not parse_expert_tensor_name(n)), key=build_store._resident_order_key
    )
    plan = build_store.Plan(
        repo_id="test/tiny-qwen2-moe",
        revision="0" * 40,
        config=config,
        num_layers=config["num_hidden_layers"],
        num_experts=config["num_experts"],
        layout=ExpertRecordLayout.from_config(config),
        resident=plan_resident_layout([(n, arrays[n].shape) for n in resident_names]),
        locations=locations,
    )

    def fake_fetch_rows(loc: TensorLocation, row_start: int = 0, row_end: int | None = None) -> np.ndarray:
        return arrays[loc.name][row_start:row_end].copy()

    monkeypatch.setattr(build_store, "fetch_rows", fake_fetch_rows)
    store_dir.mkdir(parents=True, exist_ok=True)
    build_store._preallocate(store_dir / build_store.EXPERTS_BIN, plan.experts_bin_bytes)
    build_store._preallocate(store_dir / build_store.RESIDENT_BIN, plan.resident_bin_bytes)
    journal = store_dir / build_store.JOURNAL
    for entry in plan.resident:
        build_store.append_journal(journal, build_store.build_resident(plan, store_dir, entry))
    for layer in range(plan.num_layers):
        for e in range(plan.num_experts):
            build_store.append_journal(journal, build_store.build_expert(plan, store_dir, layer, e))
    experts, resident, _ = build_store.verified_journal(store_dir, build_store.read_journal(journal))
    manifest = {
        "source": {"repo_id": plan.repo_id, "revision": plan.revision, "config": config},
        "expert_layout": plan.layout.to_dict(),
        "runs": [{"peak_rss_mb": 0.0}],
    }
    build_store.finalize(plan, store_dir, manifest, experts, resident)
    return plan
