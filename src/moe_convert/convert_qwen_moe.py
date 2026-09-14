"""Generic HF (Qwen1.5-MoE / qwen2_moe) -> reduced MindSpore checkpoint converter.

Not hardcoded to one size: pass --num-layers and --expert-ids to pick any
truncated slice of the real Qwen1.5-MoE-A2.7B checkpoint that fits in RAM.
Only the bytes for the requested tensors are downloaded (see
fetch_hf_tensors.py) -- picking fewer layers/experts directly shrinks both
the download and the resulting checkpoint size.

Produces, under --out-dir:
  process_a.ckpt   -- embedding, attention, norms, shared expert, router,
                       lm_head, and whichever experts are assigned to A
  process_b.ckpt    -- only the experts assigned to B (RemoteExpertShard)
  manifest.json     -- real config values used + the expert/layer assignment,
                       so both processes and the demo script agree on shapes
                       without re-deriving them

Usage:
  python -m moe_convert.convert_qwen_moe \
      --num-layers 2 --expert-ids 0,1,2,3 \
      --process-a-experts 0,1 --process-b-experts 2,3 --top-k 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

import numpy as np
import mindspore as ms
from mindspore import Tensor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from moe_convert.fetch_hf_tensors import fetch_index, fetch_tensors  # noqa: E402
from moe_model import ReducedQwenMoe, RemoteExpertShard  # noqa: E402

DEFAULT_REPO = "Qwen/Qwen1.5-MoE-A2.7B"


def fetch_config(repo_id: str, revision: str = "main") -> dict:
    url = f"https://huggingface.co/{repo_id}/resolve/{revision}/config.json"
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.loads(resp.read())


def build_wanted_tensor_names(num_layers: int, expert_ids: list[int]) -> list[str]:
    names = ["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"]
    for i in range(num_layers):
        p = f"model.layers.{i}."
        names += [
            p + "input_layernorm.weight",
            p + "self_attn.q_proj.weight", p + "self_attn.q_proj.bias",
            p + "self_attn.k_proj.weight", p + "self_attn.k_proj.bias",
            p + "self_attn.v_proj.weight", p + "self_attn.v_proj.bias",
            p + "self_attn.o_proj.weight",
            p + "post_attention_layernorm.weight",
            p + "mlp.gate.weight",
            p + "mlp.shared_expert.gate_proj.weight",
            p + "mlp.shared_expert.up_proj.weight",
            p + "mlp.shared_expert.down_proj.weight",
            p + "mlp.shared_expert_gate.weight",
        ]
        for eid in expert_ids:
            names += [
                p + f"mlp.experts.{eid}.gate_proj.weight",
                p + f"mlp.experts.{eid}.up_proj.weight",
                p + f"mlp.experts.{eid}.down_proj.weight",
            ]
    return names


def _set(net, name: str, arr: np.ndarray):
    params = dict(net.parameters_and_names())
    if name not in params:
        raise KeyError(f"model has no parameter named {name!r}")
    expected = tuple(params[name].shape)
    if tuple(arr.shape) != expected:
        raise ValueError(f"{name}: shape {arr.shape} != expected {expected}")
    params[name].set_data(Tensor(arr, ms.float32))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default=DEFAULT_REPO)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--expert-ids", default="0,1,2,3", help="comma-separated real expert indices (0-59) to keep")
    ap.add_argument("--process-a-experts", default="0,1", help="subset of --expert-ids hosted by process A")
    ap.add_argument("--process-b-experts", default="2,3", help="subset of --expert-ids hosted by process B")
    ap.add_argument("--top-k", type=int, default=2, help="experts-per-token for the REDUCED router (real model uses 4 of 60)")
    ap.add_argument("--out-dir", default="models/reduced_qwen_moe")
    args = ap.parse_args()

    expert_ids = [int(x) for x in args.expert_ids.split(",")]
    a_experts = [int(x) for x in args.process_a_experts.split(",")]
    b_experts = [int(x) for x in args.process_b_experts.split(",")]
    assert set(a_experts) | set(b_experts) == set(expert_ids), "process-a/b experts must partition --expert-ids"
    assert set(a_experts) & set(b_experts) == set(), "process-a/b experts must not overlap"

    print(f"Fetching config.json for {args.repo_id} ...")
    hf_cfg = fetch_config(args.repo_id, args.revision)

    cfg = {
        "vocab_size": hf_cfg["vocab_size"],
        "hidden_size": hf_cfg["hidden_size"],
        "num_attention_heads": hf_cfg["num_attention_heads"],
        "moe_intermediate_size": hf_cfg["moe_intermediate_size"],
        "shared_expert_intermediate_size": hf_cfg["shared_expert_intermediate_size"],
        "rms_norm_eps": hf_cfg["rms_norm_eps"],
        "rope_theta": hf_cfg["rope_theta"],
        "num_layers": args.num_layers,
        "expert_ids": expert_ids,
        "top_k": args.top_k,
        # kept for the manifest / documentation, not used at runtime:
        "real_num_hidden_layers": hf_cfg["num_hidden_layers"],
        "real_num_experts": hf_cfg["num_experts"],
        "real_num_experts_per_tok": hf_cfg["num_experts_per_tok"],
    }

    wanted = build_wanted_tensor_names(args.num_layers, expert_ids)
    print(f"Need {len(wanted)} real tensors from {args.repo_id} "
          f"({args.num_layers} layers, experts {expert_ids}) ...")

    index = fetch_index(args.repo_id, args.revision)

    def progress(done, total, name, nbytes):
        print(f"  [{done:4d}/{total}] {name}  ({nbytes/1e6:.2f} MB)", flush=True)

    hf_tensors = fetch_tensors(args.repo_id, wanted, revision=args.revision, index=index, progress_cb=progress)
    total_mb = sum(a.nbytes for a in hf_tensors.values()) / 1e6
    print(f"Fetched {len(hf_tensors)} tensors, {total_mb:.1f} MB total (fp32 in RAM).")

    # ---- Process A: full base model + its local experts ----
    net_a = ReducedQwenMoe(cfg, local_expert_ids_per_layer=[set(a_experts)] * args.num_layers)

    _set(net_a, "embedding.word_embeddings.weight", hf_tensors["model.embed_tokens.weight"])
    _set(net_a, "output_layer.weight", hf_tensors["lm_head.weight"])
    _set(net_a, "decoder.final_layernorm.weight", hf_tensors["model.norm.weight"])

    for i in range(args.num_layers):
        p = f"model.layers.{i}."
        d = f"decoder.layers.{i}."
        _set(net_a, d + "input_layernorm.weight", hf_tensors[p + "input_layernorm.weight"])
        _set(net_a, d + "self_attention.linear_q.weight", hf_tensors[p + "self_attn.q_proj.weight"])
        _set(net_a, d + "self_attention.linear_q.bias", hf_tensors[p + "self_attn.q_proj.bias"])
        _set(net_a, d + "self_attention.linear_k.weight", hf_tensors[p + "self_attn.k_proj.weight"])
        _set(net_a, d + "self_attention.linear_k.bias", hf_tensors[p + "self_attn.k_proj.bias"])
        _set(net_a, d + "self_attention.linear_v.weight", hf_tensors[p + "self_attn.v_proj.weight"])
        _set(net_a, d + "self_attention.linear_v.bias", hf_tensors[p + "self_attn.v_proj.bias"])
        _set(net_a, d + "self_attention.linear_proj.weight", hf_tensors[p + "self_attn.o_proj.weight"])
        _set(net_a, d + "pre_mlp_layernorm.weight", hf_tensors[p + "post_attention_layernorm.weight"])

        full_router = hf_tensors[p + "mlp.gate.weight"]  # [60, hidden]
        sliced_router = full_router[expert_ids, :]  # keep only our chosen experts' rows, same order as expert_ids
        _set(net_a, d + "mlp.router.weight", sliced_router)

        _set(net_a, d + "mlp.shared_expert.gating.weight", hf_tensors[p + "mlp.shared_expert.gate_proj.weight"])
        _set(net_a, d + "mlp.shared_expert.hidden.weight", hf_tensors[p + "mlp.shared_expert.up_proj.weight"])
        _set(net_a, d + "mlp.shared_expert.linear_fc2.weight", hf_tensors[p + "mlp.shared_expert.down_proj.weight"])
        _set(net_a, d + "mlp.shared_expert_gate.weight", hf_tensors[p + "mlp.shared_expert_gate.weight"])

        for eid in a_experts:
            _set(net_a, d + f"mlp.expert_{eid}.gating.weight", hf_tensors[p + f"mlp.experts.{eid}.gate_proj.weight"])
            _set(net_a, d + f"mlp.expert_{eid}.hidden.weight", hf_tensors[p + f"mlp.experts.{eid}.up_proj.weight"])
            _set(net_a, d + f"mlp.expert_{eid}.linear_fc2.weight", hf_tensors[p + f"mlp.experts.{eid}.down_proj.weight"])

    # ---- Process B: only its assigned experts ----
    b_expert_ids_per_layer = {i: b_experts for i in range(args.num_layers)}
    net_b = RemoteExpertShard(cfg["hidden_size"], cfg["moe_intermediate_size"], b_expert_ids_per_layer)
    for i in range(args.num_layers):
        p = f"model.layers.{i}."
        for eid in b_experts:
            _set(net_b, f"shard_layers.{i}.expert_{eid}.gating.weight", hf_tensors[p + f"mlp.experts.{eid}.gate_proj.weight"])
            _set(net_b, f"shard_layers.{i}.expert_{eid}.hidden.weight", hf_tensors[p + f"mlp.experts.{eid}.up_proj.weight"])
            _set(net_b, f"shard_layers.{i}.expert_{eid}.linear_fc2.weight", hf_tensors[p + f"mlp.experts.{eid}.down_proj.weight"])

    os.makedirs(args.out_dir, exist_ok=True)
    a_path = os.path.join(args.out_dir, "process_a.ckpt")
    b_path = os.path.join(args.out_dir, "process_b.ckpt")
    ms.save_checkpoint(net_a, a_path)
    ms.save_checkpoint(net_b, b_path)

    manifest = {
        "source_repo": args.repo_id,
        "revision": args.revision,
        "config": cfg,
        "process_a": {"checkpoint": "process_a.ckpt", "local_expert_ids": a_experts},
        "process_b": {"checkpoint": "process_b.ckpt", "local_expert_ids": b_experts},
    }
    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    a_size = os.path.getsize(a_path) / 1e6
    b_size = os.path.getsize(b_path) / 1e6
    print(f"Wrote {a_path} ({a_size:.1f} MB), {b_path} ({b_size:.1f} MB), {manifest_path}")


if __name__ == "__main__":
    main()
