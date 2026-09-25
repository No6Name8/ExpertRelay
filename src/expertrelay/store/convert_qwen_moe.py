"""Generic HF (Qwen1.5-MoE / qwen2_moe) -> reduced MindSpore checkpoint converter.

Not hardcoded to one size: pass --num-layers and --expert-ids to pick any
truncated slice of the real Qwen1.5-MoE-A2.7B checkpoint that fits in RAM.
Only the bytes for the requested tensors are downloaded (see
fetch_hf_tensors.py) -- picking fewer layers/experts directly shrinks both
the download and the resulting checkpoint size.

Produces, under --out-dir:
  process_a.ckpt    -- embedding, attention, norms, shared expert, router,
                        lm_head, and whichever experts are assigned to A
  process_b.ckpt    -- only the experts assigned to B (RemoteExpertShard)
  manifest.json     -- real config values used + the expert/layer assignment,
                        so both processes and the demo script agree on shapes
                        without re-deriving them

This truncation is a documented, load-bearing simplification -- see
docs/limitations.md -- not a hidden shortcut: the manifest records exactly
which real layers/experts were kept, and every consumer reads shapes from
it rather than assuming the real model's full size.

Usage:
  python -m expertrelay.store.convert_qwen_moe \
      --num-layers 2 --expert-ids 0,1,2,3 \
      --process-a-experts 0,1 --process-b-experts 2,3 --top-k 2
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import mindspore as ms
import numpy as np
from mindspore import Tensor

from expertrelay.benchmarking import peak_process_rss_mb
from expertrelay.memory_budget import enforce_ram_budget
from expertrelay.paths import DEFAULT_MODEL_DIR
from expertrelay.runtime.moe_model import ReducedQwenMoe, RemoteExpertShard
from expertrelay.store.fetch_hf_tensors import fetch_index, fetch_tensors

DEFAULT_REPO = "Qwen/Qwen1.5-MoE-A2.7B"


def fetch_config(repo_id: str, revision: str = "main") -> dict:
    """Fetch the real HF config.json live -- architecture dimensions are
    never hardcoded here, only the truncation choices (layers/experts) are."""
    url = f"https://huggingface.co/{repo_id}/resolve/{revision}/config.json"
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.loads(resp.read())


def build_wanted_tensor_names(num_layers: int, expert_ids: list[int]) -> list[str]:
    """The exact set of real HF tensor names needed for a `num_layers`/`expert_ids` slice."""
    names = ["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"]
    for i in range(num_layers):
        p = f"model.layers.{i}."
        names += [
            p + "input_layernorm.weight",
            p + "self_attn.q_proj.weight",
            p + "self_attn.q_proj.bias",
            p + "self_attn.k_proj.weight",
            p + "self_attn.k_proj.bias",
            p + "self_attn.v_proj.weight",
            p + "self_attn.v_proj.bias",
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


def _set(net: ms.nn.Cell, name: str, arr: np.ndarray) -> None:
    """Copy a fetched real weight into a named parameter, failing loudly on any shape mismatch."""
    params = dict(net.parameters_and_names())
    if name not in params:
        raise KeyError(f"model has no parameter named {name!r}")
    expected = tuple(params[name].shape)
    if tuple(arr.shape) != expected:
        raise ValueError(f"{name}: shape {arr.shape} != expected {expected}")
    params[name].set_data(Tensor(arr, ms.float32))


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default=DEFAULT_REPO)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument(
        "--expert-ids", default="0,1,2,3", help="comma-separated real expert indices (0-59) to keep"
    )
    ap.add_argument("--process-a-experts", default="0,1", help="subset of --expert-ids hosted by process A")
    ap.add_argument("--process-b-experts", default="2,3", help="subset of --expert-ids hosted by process B")
    ap.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="experts-per-token for the REDUCED router (real model uses 4 of 60)",
    )
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument(
        "--max-ram-gb",
        type=float,
        default=6.0,
        help="refuse to proceed if the fetched tensors would exceed this (fp32, in RAM)",
    )
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()

    expert_ids = [int(x) for x in args.expert_ids.split(",")]
    a_experts = [int(x) for x in args.process_a_experts.split(",")]
    b_experts = [int(x) for x in args.process_b_experts.split(",")]
    assert set(a_experts) | set(b_experts) == set(expert_ids), (
        "process-a/b experts must partition --expert-ids"
    )
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
    print(
        f"Need {len(wanted)} real tensors from {args.repo_id} "
        f"({args.num_layers} layers, experts {expert_ids}) ..."
    )

    index = fetch_index(args.repo_id, args.revision)

    def progress(done: int, total: int, name: str, nbytes: int) -> None:
        print(f"  [{done:4d}/{total}] {name}  ({nbytes / 1e6:.2f} MB)", flush=True)

    hf_tensors = fetch_tensors(
        args.repo_id, wanted, revision=args.revision, index=index, progress_cb=progress
    )
    total_bytes = sum(a.nbytes for a in hf_tensors.values())
    print(f"Fetched {len(hf_tensors)} tensors, {total_bytes / 1e6:.1f} MB total (fp32 in RAM).")
    # Gate BEFORE the heavier step (building two full nn.Cell graphs + saving
    # checkpoints, which roughly doubles peak RAM vs. the raw tensors alone).
    enforce_ram_budget(total_bytes, args.max_ram_gb, "converting this slice")

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
        sliced_router = full_router[
            expert_ids, :
        ]  # keep only our chosen experts' rows, same order as expert_ids
        _set(net_a, d + "mlp.router.weight", sliced_router)

        _set(
            net_a, d + "mlp.shared_expert.gating.weight", hf_tensors[p + "mlp.shared_expert.gate_proj.weight"]
        )
        _set(net_a, d + "mlp.shared_expert.hidden.weight", hf_tensors[p + "mlp.shared_expert.up_proj.weight"])
        _set(
            net_a,
            d + "mlp.shared_expert.linear_fc2.weight",
            hf_tensors[p + "mlp.shared_expert.down_proj.weight"],
        )
        _set(net_a, d + "mlp.shared_expert_gate.weight", hf_tensors[p + "mlp.shared_expert_gate.weight"])

        for eid in a_experts:
            _set(
                net_a,
                d + f"mlp.expert_{eid}.gating.weight",
                hf_tensors[p + f"mlp.experts.{eid}.gate_proj.weight"],
            )
            _set(
                net_a,
                d + f"mlp.expert_{eid}.hidden.weight",
                hf_tensors[p + f"mlp.experts.{eid}.up_proj.weight"],
            )
            _set(
                net_a,
                d + f"mlp.expert_{eid}.linear_fc2.weight",
                hf_tensors[p + f"mlp.experts.{eid}.down_proj.weight"],
            )

    # ---- Process B: only its assigned experts ----
    b_expert_ids_per_layer = dict.fromkeys(range(args.num_layers), b_experts)
    net_b = RemoteExpertShard(cfg["hidden_size"], cfg["moe_intermediate_size"], b_expert_ids_per_layer)
    for i in range(args.num_layers):
        p = f"model.layers.{i}."
        for eid in b_experts:
            _set(
                net_b,
                f"shard_layers.{i}.expert_{eid}.gating.weight",
                hf_tensors[p + f"mlp.experts.{eid}.gate_proj.weight"],
            )
            _set(
                net_b,
                f"shard_layers.{i}.expert_{eid}.hidden.weight",
                hf_tensors[p + f"mlp.experts.{eid}.up_proj.weight"],
            )
            _set(
                net_b,
                f"shard_layers.{i}.expert_{eid}.linear_fc2.weight",
                hf_tensors[p + f"mlp.experts.{eid}.down_proj.weight"],
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    a_path = args.out_dir / "process_a.ckpt"
    b_path = args.out_dir / "process_b.ckpt"
    ms.save_checkpoint(net_a, str(a_path))
    ms.save_checkpoint(net_b, str(b_path))

    manifest = {
        "source_repo": args.repo_id,
        "revision": args.revision,
        "config": cfg,
        "process_a": {"checkpoint": "process_a.ckpt", "local_expert_ids": a_experts},
        "process_b": {"checkpoint": "process_b.ckpt", "local_expert_ids": b_experts},
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(
        f"Wrote {a_path} ({a_path.stat().st_size / 1e6:.1f} MB), "
        f"{b_path} ({b_path.stat().st_size / 1e6:.1f} MB), {manifest_path}"
    )
    print(f"Peak RSS this process: {peak_process_rss_mb():.1f} MB")


if __name__ == "__main__":
    main()
