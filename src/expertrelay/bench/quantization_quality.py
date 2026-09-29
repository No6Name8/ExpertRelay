"""What does 4-bit cost? int8 and int4 experts vs. the ORIGINAL bf16 model,
on the same prompts and positions.

    python -m expertrelay.bench.quantization_quality
    python -m expertrelay.bench.quantization_quality --report   # redo the doc section from saved JSON

Prompts: the 4 Phase 2 prompts (benchmarks/prompts/phase2.json) and 24
more, 4 per category (benchmarks/prompts/quality24.json: English, Modern
Standard Arabic, Gulf Arabic, code, math, chat-style), all as raw text for
the base model.

Positions. Each prompt is continued once, greedily, by the int8 model (32
new tokens); every model is then run over the same prompt + continuation
("teacher forcing"), so all precisions are scored on exactly the same
positions. Our models run as the runtime does: the prompt in one prefill,
then one token at a time with the fused decode kernel; the output
projection (lm_head, int8 in every store) is applied one position at a
time, as in decoding. The reference is Hugging Face transformers on the
original bf16 weights, one decoder layer at a time (bench.reference_check).
The continuation comes from the int8 model, so the contexts are int8's
choices; each model is still scored against bf16's next-token choice at
every position.

Per position: top-1 agreement (our most likely next token == bf16's) and
KL(bf16 || ours) in nats. Reported per prompt, per category, overall.

PASS (fixed before any result was seen): overall top-1 agreement >= 96%
AND every category of the 24-prompt set >= 90% ("no category collapses").

Writes benchmarks/results/quantization_quality.json and a section of
docs/int4-experts.md.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
import transformers

from expertrelay import REPO_ROOT
from expertrelay.bench.reference_check import (
    Bf16Checkpoint,
    hf_reference_forward,
    kl_per_position,
    lm_head_logits,
)
from expertrelay.benchmarking import append_benchmark_record, base_record, peak_process_rss_mb
from expertrelay.manager.profile import collect_machine_profile
from expertrelay.paths import BENCHMARK_RESULTS_DIR, MODELS_ROOT
from expertrelay.runtime.generate import DEFAULT_RUNTIME_CONFIG, RuntimeConfig, generate, load_model
from expertrelay.runtime.int8_linear import set_kernel
from expertrelay.runtime.qwen_moe import ForwardTrace, KVCache
from expertrelay.store.download_checkpoint import checkpoint_dir
from expertrelay.store.tokenizer import load_tokenizer

STORES = {
    "int8": "qwen1.5-moe-a2.7b-int8",
    "int4_g128": "qwen1.5-moe-a2.7b-int4g128",
    "int4_g64": "qwen1.5-moe-a2.7b-int4g64",
}
PROMPT_FILES = {
    "phase2": REPO_ROOT / "benchmarks" / "prompts" / "phase2.json",
    "quality24": REPO_ROOT / "benchmarks" / "prompts" / "quality24.json",
}
PASS_OVERALL = 0.96
PASS_CATEGORY = 0.90
OUT = BENCHMARK_RESULTS_DIR / "quantization_quality.json"


def load_prompts() -> list[dict]:
    out = []
    for set_name, path in PROMPT_FILES.items():
        for p in json.loads(path.read_text(encoding="utf-8"))["prompts"]:
            category = p.get("category") or f"phase2_{p['id']}"
            out.append({"set": set_name, "id": p["id"], "category": category, "text": p["text"]})
    return out


def final_hidden_states(
    store_dir: Path, rt: RuntimeConfig, sequences, prompt_lens
) -> tuple[list[np.ndarray], object, object]:
    """Our final hidden state at every position, run as the runtime runs."""
    model, _ = load_model(store_dir, "unbuffered", rt.max_seq, rt.memory_budget_gb)
    set_kernel("fused")
    out = []
    for ids, n in zip(sequences, prompt_lens, strict=True):
        cache = KVCache(model.c, rt.max_seq)
        rows = []
        for feed in [ids[:n], *([t] for t in ids[n:])]:
            trace = ForwardTrace()
            model.forward(np.asarray(feed), cache, all_logits=True, trace=trace)
            rows.append(trace.final_hidden)
        out.append(np.concatenate(rows))
    lm_head = model.resident["lm_head.weight"]
    model.experts.close()
    return out, lm_head, model.backend


def our_logits(hidden: np.ndarray, lm_head, backend) -> np.ndarray:
    """lm_head one position at a time: the decode path (fused kernel)."""
    return np.concatenate(
        [backend.int8_linear(hidden[i : i + 1], lm_head.q, lm_head.scales) for i in range(len(hidden))]
    )


def score(ref: np.ndarray, ours: np.ndarray, prompt_len: int) -> dict:
    top_ref, top_ours = ref.argmax(-1), ours.argmax(-1)
    kl = kl_per_position(ref, ours)
    gen = slice(prompt_len - 1, len(top_ref) - 1)
    return {
        "positions": int(len(top_ref)),
        "agree": int(np.sum(top_ref == top_ours)),
        "generated_positions": int(len(top_ref[gen])),
        "generated_agree": int(np.sum(top_ref[gen] == top_ours[gen])),
        "kl_sum": float(kl.sum()),
        "kl_max": float(kl.max()),
    }


def aggregate(rows: list[dict]) -> dict:
    pos = sum(r["positions"] for r in rows)
    gen = sum(r["generated_positions"] for r in rows)
    return {
        "prompts": len(rows),
        "positions": pos,
        "top1_agreement": sum(r["agree"] for r in rows) / pos,
        "top1_agreement_generated": sum(r["generated_agree"] for r in rows) / gen,
        "kl_mean": sum(r["kl_sum"] for r in rows) / pos,
        "kl_max": max(r["kl_max"] for r in rows),
    }


def verdict(by_precision: dict) -> dict:
    out = {}
    for name, v in by_precision.items():
        cats = {c: a for c, a in v["by_category"].items() if not c.startswith("phase2_")}
        worst = min(cats, key=lambda c: cats[c]["top1_agreement"])
        passed = (
            v["overall"]["top1_agreement"] >= PASS_OVERALL and cats[worst]["top1_agreement"] >= PASS_CATEGORY
        )
        out[name] = {
            "pass": passed,
            "overall_top1": v["overall"]["top1_agreement"],
            "worst_category": worst,
            "worst_category_top1": cats[worst]["top1_agreement"],
        }
    return out


def format_markdown(record: dict) -> str:
    a = record["analysis"]
    names = list(a["by_precision"])
    cats = [c for c in a["by_precision"][names[0]]["by_category"] if not c.startswith("phase2_")]
    lines = [
        "## Quality vs. the original bf16 model",
        "",
        f"Generated by `python -m expertrelay.bench.quantization_quality` at {record['timestamp']} "
        f"(commit {record['git_commit'][:7]}{', dirty tree' if record['git_dirty'] else ''}) from "
        "`benchmarks/results/quantization_quality.json`. Do not edit by hand.",
        "",
        f"{a['prompts']} prompts (4 from Phase 2 + 24, 4 per category), each continued by the int8 model "
        f"for 32 tokens; every model scored on the same {a['positions']} positions against bf16's next "
        "token. Top-1 = our most likely next token equals bf16's. KL(bf16 || ours) in nats per position.",
        "",
        f"**Pass rule, fixed before the results:** overall top-1 >= {PASS_OVERALL:.0%} and every category "
        f">= {PASS_CATEGORY:.0%}.",
        "",
        "| precision | top-1, all positions | top-1, generated | mean KL | max KL | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for n in names:
        o, v = a["by_precision"][n]["overall"], a["verdict"][n]
        lines.append(
            f"| {n} | **{o['top1_agreement']:.1%}** | {o['top1_agreement_generated']:.1%} | {o['kl_mean']:.4f} | "
            f"{o['kl_max']:.3f} | {'PASS' if v['pass'] else 'FAIL'} (worst: {v['worst_category']} "
            f"{v['worst_category_top1']:.1%}) |"
        )
    lines += [
        "",
        "Per category (24-prompt set), top-1 agreement / mean KL:",
        "",
        "| category | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for c in cats:
        cells = []
        for n in names:
            x = a["by_precision"][n]["by_category"][c]
            cells.append(f"{x['top1_agreement']:.1%} / {x['kl_mean']:.4f}")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    lines += ["", "Arabic only (both varieties of the 24-prompt set, plus Phase 2's Arabic prompt):", ""]
    lines += ["| precision | top-1 | generated | mean KL | max KL |", "|---|---|---|---|---|"]
    for n in names:
        x = a["by_precision"][n]["arabic"]
        lines.append(
            f"| {n} | {x['top1_agreement']:.1%} | {x['top1_agreement_generated']:.1%} | {x['kl_mean']:.4f} | "
            f"{x['kl_max']:.3f} |"
        )
    p2 = a["by_precision"]["int8"]["phase2"]
    lines += [
        "",
        f"Consistency check: int8 on the 4 Phase 2 prompts alone: {p2['top1_agreement']:.1%} "
        "(the reference check's earlier result on the same sequences: 98.2%).",
    ]
    return "\n".join(lines) + "\n"


def write_doc(record: dict) -> Path:
    from expertrelay.bench.int4_report import upsert_section

    return upsert_section("quality", format_markdown(record))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        print(f"wrote {write_doc(json.loads(OUT.read_text(encoding='utf-8'))[-1])}")
        return

    rt = RuntimeConfig.load(DEFAULT_RUNTIME_CONFIG)
    int8_dir = MODELS_ROOT / STORES["int8"]
    source = json.loads((int8_dir / "store.json").read_text())["source"]
    ckpt_dir = checkpoint_dir(source["repo_id"], source["revision"])
    prompts = load_prompts()
    record = base_record(
        label="quantization quality: int8 and int4 experts vs bf16",
        seed=0,
        model={
            "stores": STORES,
            "source": source["revision"],
            "reference": ckpt_dir.relative_to(REPO_ROOT).as_posix(),
        },
        config={
            "prompt_files": {k: v.relative_to(REPO_ROOT).as_posix() for k, v in PROMPT_FILES.items()},
            "prompts": [{k: p[k] for k in ("set", "id", "category")} for p in prompts],
            "max_new_tokens": rt.max_new_tokens,
            "pass_overall": PASS_OVERALL,
            "pass_category": PASS_CATEGORY,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
        },
        machine=collect_machine_profile(measure_disk=False),
    )

    tokenizer = load_tokenizer(int8_dir)
    model, _ = load_model(int8_dir, "unbuffered", rt.max_seq, rt.memory_budget_gb)
    set_kernel("fused")
    sequences, prompt_lens = [], []
    for p in prompts:
        ids = tokenizer.encode(p["text"]).ids
        generated, _ = generate(model, ids, rt.max_new_tokens, rt.max_seq)
        sequences.append(ids + generated)
        prompt_lens.append(len(ids))
    model.experts.close()
    del model
    gc.collect()
    print(f"continued {len(sequences)} prompts", flush=True)

    ours, lm_head, backend = {}, None, None
    for name, store in STORES.items():
        ours[name], lm_head, backend = final_hidden_states(MODELS_ROOT / store, rt, sequences, prompt_lens)
        gc.collect()
        print(f"ours: {name}", flush=True)

    ckpt = Bf16Checkpoint(ckpt_dir)
    refs = hf_reference_forward(ckpt, sequences, keep_layers=False, logits=False)

    rows = {name: [] for name in STORES}
    for i, (p, n) in enumerate(zip(prompts, prompt_lens, strict=True)):
        ref = lm_head_logits(ckpt, refs[i].final_hidden, dtype=torch.float32)
        for name in STORES:
            rows[name].append(
                {
                    "id": p["id"],
                    "set": p["set"],
                    "category": p["category"],
                    **score(ref, our_logits(ours[name][i], lm_head, backend), n),
                }
            )
        print(f"scored {p['id']}", flush=True)

    by_precision = {}
    for name, rs in rows.items():
        cats = sorted({r["category"] for r in rs})
        by_precision[name] = {
            "overall": aggregate(rs),
            "by_category": {c: aggregate([r for r in rs if r["category"] == c]) for c in cats},
            "arabic": aggregate([r for r in rs if r["category"] in ("ar_msa", "ar_gulf", "phase2_ar")]),
            "phase2": aggregate([r for r in rs if r["set"] == "phase2"]),
            "per_prompt": rs,
        }
    record["analysis"] = {
        "prompts": len(prompts),
        "positions": by_precision["int8"]["overall"]["positions"],
        "sequences": sequences,
        "prompt_lens": prompt_lens,
        "by_precision": by_precision,
        "verdict": verdict(by_precision),
    }
    record["peak_rss_mb"] = peak_process_rss_mb()
    append_benchmark_record(OUT, record)
    print(f"saved {OUT}; wrote {write_doc(record)}")
    print(json.dumps(record["analysis"]["verdict"], indent=1))


if __name__ == "__main__":
    main()
