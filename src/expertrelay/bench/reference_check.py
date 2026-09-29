"""Our int8 runtime vs. Hugging Face transformers on the ORIGINAL bf16 weights.

    python -m expertrelay.bench.reference_check

For each prompt in the fixed prompt set:
  1. Our runtime greedily generates the continuation (same as Phase 2).
  2. The whole sequence (prompt + continuation) is run through our model
     once more in one prefill, capturing the hidden state after every layer,
     the final hidden state, the logits at every position, and the experts
     the router picked.
  3. The same sequence goes through HF transformers' Qwen2MoE, one decoder
     layer at a time, so the 28.6 GB bf16 model never has to fit in RAM.
     Layer weights come from the original safetensors shards, upcast bf16 ->
     f32 (exact), and are dropped after the layer.

Compared: relative error of the hidden state after each layer, of the final
hidden state and of the logits; top-1 agreement (at every position, does
argmax of our logits equal argmax of the reference logits?); KL divergence
KL(reference || ours) of the next-token distributions, in nats; and routing
agreement (does the router pick the same top-k experts?).

Also measured: the same comparison with lm_head in fp16 instead of int8, so
disagreement caused by the int8 lm_head alone can be separated out.

HF's code runs unmodified, including its expert loop. Only the storage of
the experts differs. HF keeps all 60 experts of a layer in one fused tensor
(~1 GB even in bf16, more than this machine has free). This script sets that
attribute to an object that loads expert e from the shards when HF's own
forward code indexes it (`self.gate_up_proj[e]`).

Needs the dev extras (torch, transformers==5.17.0, safetensors) and the
bf16 checkpoint: `python -m expertrelay.store.download_checkpoint`.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import transformers
from safetensors import safe_open
from transformers import Qwen2MoeConfig
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen2_moe.modeling_qwen2_moe import (
    Qwen2MoeDecoderLayer,
    Qwen2MoeRMSNorm,
    Qwen2MoeRotaryEmbedding,
)

from expertrelay import REPO_ROOT
from expertrelay.benchmarking import append_benchmark_record, base_record, peak_process_rss_mb
from expertrelay.manager.profile import collect_machine_profile
from expertrelay.paths import BENCHMARK_RESULTS_DIR
from expertrelay.runtime.generate import DEFAULT_RUNTIME_CONFIG, RuntimeConfig, generate, load_model
from expertrelay.runtime.qwen_moe import ForwardTrace, KVCache
from expertrelay.store.download_checkpoint import checkpoint_dir
from expertrelay.store.tokenizer import load_tokenizer

DOC = REPO_ROOT / "docs" / "reference-check.md"
LM_HEAD_BLOCK_ROWS = 8192


# ---------------------------------------------------------------------------
# The bf16 checkpoint, read lazily
# ---------------------------------------------------------------------------


class Bf16Checkpoint:
    """Tensors from local safetensors shards, one at a time, as f32.

    At most MAX_OPEN_SHARDS shard files are open at once, least recently
    used closed first. On Windows, safetensors' torch loader maps each ~4 GB
    shard in a way that counts against the system commit limit; with every
    shard open the full check crashed (access violation in
    torch/storage.py) the first time it read from a 6th shard, twice, while
    the same tensors read fine on their own. The model is read in layer
    order, so two open shards are enough."""

    MAX_OPEN_SHARDS = 2

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        index = json.loads((self.directory / "model.safetensors.index.json").read_text())
        self._shard_of = index["weight_map"]
        self._open: OrderedDict[str, object] = OrderedDict()

    def _handle(self, name: str):
        shard = self._shard_of[name]
        if shard in self._open:
            self._open.move_to_end(shard)
        else:
            while len(self._open) >= self.MAX_OPEN_SHARDS:
                self._open.popitem(last=False)
                gc.collect()  # drop the closed shard's mapping now, not whenever
            self._open[shard] = safe_open(str(self.directory / shard), framework="pt")
        return self._open[shard]

    def get(self, name: str) -> torch.Tensor:
        return self._handle(name).get_tensor(name).float()

    def rows(self, name: str, start: int, stop: int) -> torch.Tensor:
        return self._handle(name).get_slice(name)[start:stop].float()

    def config(self) -> Qwen2MoeConfig:
        config = Qwen2MoeConfig.from_pretrained(self.directory)
        config._attn_implementation = "eager"
        return config


class _LazyExpertBank:
    """Stands in for HF's fused expert parameter. `bank[e]` returns expert e's
    weight in the exact layout HF's fused tensor would hold: gate_up is
    [gate; up] stacked on the output dim, down as-is. Only the most recently
    used expert is kept: HF's loop reads gate_up[e] then down[e] for one
    expert before moving to the next."""

    def __init__(self, ckpt: Bf16Checkpoint, layer: int, which: str):
        self._ckpt, self._layer, self._which = ckpt, layer, which
        self._cached: tuple[int, torch.Tensor] | None = None

    def __getitem__(self, expert) -> torch.Tensor:
        e = int(expert)
        if self._cached is None or self._cached[0] != e:
            p = f"model.layers.{self._layer}.mlp.experts.{e}."
            if self._which == "gate_up":
                w = torch.cat([self._ckpt.get(p + "gate_proj.weight"), self._ckpt.get(p + "up_proj.weight")])
            else:
                w = self._ckpt.get(p + "down_proj.weight")
            self._cached = (e, w)
        return self._cached[1]


def _load_layer(ckpt: Bf16Checkpoint, config: Qwen2MoeConfig, layer: int) -> Qwen2MoeDecoderLayer:
    with torch.device("meta"):
        module = Qwen2MoeDecoderLayer(config, layer)
    experts = module.mlp.experts
    del experts.gate_up_proj, experts.down_proj  # the 1 GB fused parameters
    module = module.to_empty(device="cpu")
    prefix = f"model.layers.{layer}."
    state = {name: ckpt.get(prefix + name) for name in module.state_dict()}
    module.load_state_dict(state, strict=True)
    experts.gate_up_proj = _LazyExpertBank(ckpt, layer, "gate_up")
    experts.down_proj = _LazyExpertBank(ckpt, layer, "down")
    return module.eval()


@dataclass
class ReferenceTrace:
    hidden_after_layer: list[np.ndarray] = field(default_factory=list)
    selected_experts: list[np.ndarray] = field(default_factory=list)
    final_hidden: np.ndarray | None = None
    logits: np.ndarray | None = None


def hf_reference_forward(ckpt: Bf16Checkpoint, sequences: list[list[int]]) -> list[ReferenceTrace]:
    """HF transformers, layer by layer, all sequences through each layer
    before the next layer's weights are loaded."""
    config = ckpt.config()
    rotary = Qwen2MoeRotaryEmbedding(config=config)
    traces = [ReferenceTrace() for _ in sequences]
    states, extras = [], []
    with torch.no_grad():
        for ids in sequences:
            h = torch.cat([ckpt.rows("model.embed_tokens.weight", t, t + 1) for t in ids])[None]
            position_ids = torch.arange(len(ids))[None]
            mask = create_causal_mask(
                config=config,
                inputs_embeds=h,
                attention_mask=None,
                past_key_values=None,
                position_ids=position_ids,
                allow_is_causal_skip=False,
            )
            states.append(h)
            extras.append(
                {
                    "attention_mask": mask,
                    "position_ids": position_ids,
                    "position_embeddings": rotary(h, position_ids),
                }
            )
        for layer in range(config.num_hidden_layers):
            module = _load_layer(ckpt, config, layer)
            picked: list[torch.Tensor] = []
            hook = module.mlp.gate.register_forward_hook(
                lambda _m, _i, out, picked=picked: picked.append(out[2])
            )
            for i, trace in enumerate(traces):
                states[i] = module(states[i], **extras[i])
                trace.hidden_after_layer.append(states[i][0].numpy().copy())
                trace.selected_experts.append(picked[-1].numpy().copy())
            hook.remove()
            del module
            gc.collect()
            print(f"  reference layer {layer + 1}/{config.num_hidden_layers}", flush=True)

        norm = Qwen2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        norm.weight.data = ckpt.get("model.norm.weight")
        for i, trace in enumerate(traces):
            trace.final_hidden = norm(states[i])[0].numpy()
            trace.logits = lm_head_logits(ckpt, trace.final_hidden, dtype=torch.float32)
    return traces


def lm_head_logits(ckpt: Bf16Checkpoint, hidden: np.ndarray, dtype: torch.dtype) -> np.ndarray:
    """hidden @ lm_head^T, lm_head read in row blocks from the shards and
    rounded to `dtype` first (f32 = the original bf16 values exactly;
    f16 = what an fp16 lm_head in the store would hold)."""
    h = torch.from_numpy(np.ascontiguousarray(hidden, dtype=np.float32))
    vocab = ckpt.config().vocab_size
    blocks = []
    for r0 in range(0, vocab, LM_HEAD_BLOCK_ROWS):
        w = ckpt.rows("lm_head.weight", r0, min(vocab, r0 + LM_HEAD_BLOCK_ROWS)).to(dtype).float()
        blocks.append((h @ w.T).numpy())
    return np.concatenate(blocks, axis=-1)


# ---------------------------------------------------------------------------
# Our side, and the comparison
# ---------------------------------------------------------------------------


def our_traces(
    store_dir: Path, rt: RuntimeConfig, sequences: list[list[int]]
) -> list[tuple[ForwardTrace, np.ndarray]]:
    model, _ = load_model(store_dir, "unbuffered", rt.max_seq, rt.memory_budget_gb)
    out = []
    for ids in sequences:
        trace = ForwardTrace()
        logits, _ = model.forward(np.asarray(ids), KVCache(model.c, rt.max_seq), all_logits=True, trace=trace)
        out.append((trace, logits))
    model.experts.close()
    return out


def kl_per_position(ref_logits: np.ndarray, our_logits: np.ndarray) -> np.ndarray:
    """KL(P_ref || P_ours) at each position, nats, from logits, in float64."""

    def log_softmax(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float64)
        x = x - x.max(axis=-1, keepdims=True)
        return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))

    lp, lq = log_softmax(ref_logits), log_softmax(our_logits)
    return (np.exp(lp) * (lp - lq)).sum(axis=-1)


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.astype(np.float64) - b) / np.linalg.norm(b.astype(np.float64)))


def compare(ours: ForwardTrace, our_logits: np.ndarray, ref: ReferenceTrace, prompt_len: int) -> dict:
    ours_top1, ref_top1 = our_logits.argmax(-1), ref.logits.argmax(-1)
    generated = slice(prompt_len - 1, len(ours_top1) - 1)  # positions whose next token was generated
    routing = []
    for mine, theirs in zip(ours.selected_experts, ref.selected_experts, strict=True):
        overlap = [len(set(a) & set(b)) / len(a) for a, b in zip(mine.tolist(), theirs.tolist(), strict=True)]
        same = [set(a) == set(b) for a, b in zip(mine.tolist(), theirs.tolist(), strict=True)]
        routing.append({"same_set_fraction": float(np.mean(same)), "mean_overlap": float(np.mean(overlap))})
    return {
        "positions": len(ours_top1),
        "hidden_rel_error_per_layer": [
            _rel(o, r) for o, r in zip(ours.hidden_after_layer, ref.hidden_after_layer, strict=True)
        ],
        "final_hidden_rel_error": _rel(ours.final_hidden, ref.final_hidden),
        "logits_rel_error": _rel(our_logits, ref.logits),
        "top1_agreement_all_positions": float(np.mean(ours_top1 == ref_top1)),
        "top1_agreement_generated_positions": float(np.mean(ours_top1[generated] == ref_top1[generated])),
        "generated_positions": int(len(ours_top1[generated])),
        "kl": _kl_summary(kl_per_position(ref.logits, our_logits)),
        "routing_per_layer": routing,
    }


def _kl_summary(kl: np.ndarray) -> dict:
    return {
        "mean": float(kl.mean()),
        "median": float(np.median(kl)),
        "max": float(kl.max()),
        "sum": float(kl.sum()),
        "positions": int(len(kl)),
    }


def lm_head_study(
    ckpt: Bf16Checkpoint, ours: ForwardTrace, our_logits: np.ndarray, ref: ReferenceTrace
) -> dict:
    """Is int8 lm_head a cause of disagreement? Recompute our logits from OUR
    final hidden state with an fp16 lm_head and compare both against the
    reference, and against each other (flips caused by lm_head alone)."""
    fp16_logits = lm_head_logits(ckpt, ours.final_hidden, dtype=torch.float16)
    ref_top1 = ref.logits.argmax(-1)
    int8_top1, fp16_top1 = our_logits.argmax(-1), fp16_logits.argmax(-1)
    return {
        "top1_agreement_int8_lm_head": float(np.mean(int8_top1 == ref_top1)),
        "top1_agreement_fp16_lm_head": float(np.mean(fp16_top1 == ref_top1)),
        "positions_where_lm_head_precision_changes_top1": int(np.sum(int8_top1 != fp16_top1)),
        "logits_rel_error_int8_lm_head": _rel(our_logits, ref.logits),
        "logits_rel_error_fp16_lm_head": _rel(fp16_logits, ref.logits),
        "kl_fp16_lm_head": _kl_summary(kl_per_position(ref.logits, fp16_logits)),
        "positions": int(len(ref_top1)),
    }


def aggregate(per_prompt: list[dict]) -> dict:
    layers = len(per_prompt[0]["comparison"]["hidden_rel_error_per_layer"])
    total = sum(p["comparison"]["positions"] for p in per_prompt)

    def weighted(key: str) -> float:
        return sum(p["comparison"][key] * p["comparison"]["positions"] for p in per_prompt) / total

    return {
        "hidden_rel_error_per_layer_mean": [
            statistics.fmean(p["comparison"]["hidden_rel_error_per_layer"][i] for p in per_prompt)
            for i in range(layers)
        ],
        "routing_same_set_per_layer_mean": [
            statistics.fmean(p["comparison"]["routing_per_layer"][i]["same_set_fraction"] for p in per_prompt)
            for i in range(layers)
        ],
        "final_hidden_rel_error_mean": statistics.fmean(
            p["comparison"]["final_hidden_rel_error"] for p in per_prompt
        ),
        "logits_rel_error_mean": statistics.fmean(p["comparison"]["logits_rel_error"] for p in per_prompt),
        "top1_agreement_all_positions": weighted("top1_agreement_all_positions"),
        "top1_agreement_generated_positions": sum(
            p["comparison"]["top1_agreement_generated_positions"] * p["comparison"]["generated_positions"]
            for p in per_prompt
        )
        / sum(p["comparison"]["generated_positions"] for p in per_prompt),
        "generated_positions": sum(p["comparison"]["generated_positions"] for p in per_prompt),
        "kl_mean": sum(p["comparison"]["kl"]["sum"] for p in per_prompt) / total,
        "kl_max": max(p["comparison"]["kl"]["max"] for p in per_prompt),
        "top1_agreement_all_positions_fp16_lm_head": sum(
            p["lm_head"]["top1_agreement_fp16_lm_head"] * p["lm_head"]["positions"] for p in per_prompt
        )
        / total,
        "kl_mean_fp16_lm_head": sum(p["lm_head"]["kl_fp16_lm_head"]["sum"] for p in per_prompt) / total,
        "positions": total,
        "lm_head_changes_top1_positions": sum(
            p["lm_head"]["positions_where_lm_head_precision_changes_top1"] for p in per_prompt
        ),
    }


def format_markdown(record: dict, json_path: Path) -> str:
    agg, prompts = record["aggregate"], record["per_prompt"]
    lines = [
        "# Reference check: int8 runtime vs. HF transformers on the original bf16 weights",
        "",
        f"Generated by `python -m expertrelay.bench.reference_check` at {record['timestamp']} "
        f"(commit {record['git_commit']}{', dirty tree' if record['git_dirty'] else ''}). "
        f"Raw data: `{json_path.resolve().relative_to(REPO_ROOT).as_posix()}`. Do not edit by hand.",
        "",
        "Reference: `transformers` "
        f"{record['config']['transformers_version']} Qwen2MoE, run one decoder layer at a time on the original "
        "bf16 weights upcast to f32. Ours: the int8 store, numpy runtime. Same token sequences (prompt + our "
        "greedy continuation), one prefill each.",
        "",
        "## Summary",
        "",
        "| | |",
        "|---|---|",
        f"| Top-1 agreement, all {agg['positions']} positions | **{agg['top1_agreement_all_positions']:.1%}** |",
        f"| Top-1 agreement, the {agg['generated_positions']} generated positions | "
        f"{agg['top1_agreement_generated_positions']:.1%} |",
        f"| KL(reference \\|\\| ours), mean / max over positions (nats) | {agg['kl_mean']:.4f} / {agg['kl_max']:.4f} |",
        f"| With lm_head in fp16 instead of int8: top-1 agreement / mean KL | "
        f"{agg['top1_agreement_all_positions_fp16_lm_head']:.1%} / {agg['kl_mean_fp16_lm_head']:.4f} |",
        f"| Final hidden state, relative error (mean over prompts) | {agg['final_hidden_rel_error_mean']:.2%} |",
        f"| Logits, relative error (mean over prompts) | {agg['logits_rel_error_mean']:.2%} |",
        f"| Positions where int8 vs. fp16 lm_head changes the top-1 token | {agg['lm_head_changes_top1_positions']} |",
        "",
        "## Per prompt",
        "",
        "| Prompt | Positions | Top-1 agreement (all) | Top-1 agreement (generated) | Logits rel. error | "
        "KL mean / max | Top-1 with int8 / fp16 lm_head |",
        "|---|---|---|---|---|---|---|",
        *[
            f"| {p['id']} | {p['comparison']['positions']} | {p['comparison']['top1_agreement_all_positions']:.1%} | "
            f"{p['comparison']['top1_agreement_generated_positions']:.1%} | "
            f"{p['comparison']['logits_rel_error']:.2%} | "
            f"{p['comparison']['kl']['mean']:.4f} / {p['comparison']['kl']['max']:.4f} | "
            f"{p['lm_head']['top1_agreement_int8_lm_head']:.1%} / "
            f"{p['lm_head']['top1_agreement_fp16_lm_head']:.1%} |"
            for p in prompts
        ],
        "",
        "## Per layer (mean over prompts)",
        "",
        "| Layer | Hidden state rel. error | Same top-k expert set |",
        "|---|---|---|",
        *[
            f"| {i} | {e:.3%} | {r:.1%} |"
            for i, (e, r) in enumerate(
                zip(
                    agg["hidden_rel_error_per_layer_mean"],
                    agg["routing_same_set_per_layer_mean"],
                    strict=True,
                )
            )
        ],
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_RUNTIME_CONFIG)
    ap.add_argument(
        "--checkpoint-dir", type=Path, default=None, help="default: models/hf/<repo>@<store revision>"
    )
    ap.add_argument("--out", type=Path, default=BENCHMARK_RESULTS_DIR / "reference_check.json")
    args = ap.parse_args()

    rt = RuntimeConfig.load(args.config)
    source = json.loads((rt.store_dir / "store.json").read_text())["source"]
    ckpt_dir = args.checkpoint_dir or checkpoint_dir(source["repo_id"], source["revision"])
    prompts = json.loads(rt.prompts_file.read_text(encoding="utf-8"))["prompts"]
    record = base_record(
        label="reference check vs HF transformers bf16",
        seed=0,
        model={
            "store": rt.store_dir.relative_to(REPO_ROOT).as_posix(),
            "source": source["revision"],
            "reference_checkpoint": ckpt_dir.relative_to(REPO_ROOT).as_posix(),
        },
        config={
            "max_new_tokens": rt.max_new_tokens,
            "prompts": prompts,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
        },
        machine=collect_machine_profile(measure_disk=False),
    )

    tokenizer = load_tokenizer(rt.store_dir)
    model, _ = load_model(rt.store_dir, "unbuffered", rt.max_seq, rt.memory_budget_gb)
    sequences, prompt_lens = [], []
    for p in prompts:
        ids = tokenizer.encode(p["text"]).ids
        generated, _ = generate(model, ids, rt.max_new_tokens, rt.max_seq)
        sequences.append(ids + generated)
        prompt_lens.append(len(ids))
        print(f"generated for {p['id']}", flush=True)
    model.experts.close()
    del model
    gc.collect()

    ours = our_traces(rt.store_dir, rt, sequences)
    gc.collect()
    ckpt = Bf16Checkpoint(ckpt_dir)
    refs = hf_reference_forward(ckpt, sequences)

    per_prompt = []
    for p, n, (trace, logits), ref in zip(prompts, prompt_lens, ours, refs, strict=True):
        per_prompt.append(
            {
                "id": p["id"],
                "prompt_tokens": n,
                "comparison": compare(trace, logits, ref, n),
                "lm_head": lm_head_study(ckpt, trace, logits, ref),
            }
        )
    record["per_prompt"] = per_prompt
    record["aggregate"] = aggregate(per_prompt)
    record["peak_rss_mb"] = peak_process_rss_mb()
    append_benchmark_record(args.out, record)
    DOC.write_text(format_markdown(record, args.out), encoding="utf-8")
    print(json.dumps(record["aggregate"], indent=1))


if __name__ == "__main__":
    main()
