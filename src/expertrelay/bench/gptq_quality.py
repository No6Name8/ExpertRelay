"""Step B3 quality: int8, int4 RTN and int4 GPTQ experts vs the ORIGINAL
bf16 Chat model, same prompts and positions, resumable.

    python -m expertrelay.bench.gptq_quality

The method is bench.quantization_quality's (Step B2), applied to the Chat
model:
  - the same 28 prompts (benchmarks/prompts/phase2.json + quality24.json),
    each wrapped in the Chat model's chat template (store.chat_template,
    as the runtime does for Chat stores);
  - each prompt continued once, greedily, for 32 tokens by the int8 Chat
    store with the fused kernel; every model is then run over the same
    prompt + continuation (teacher forcing), the prompt in one prefill and
    the continuation one token at a time, as the runtime decodes;
  - reference: Hugging Face transformers on the bf16 Chat weights, one
    decoder layer at a time (bench.reference_check);
  - per position: top-1 agreement with bf16 and KL(bf16 || ours).
The three stores share one resident part (hard-linked from the int8 Chat
store; store.json "resident_from" is checked), so they differ only in the
routed experts, and their lm_head is the same int8 matrix.

PASS (fixed in the Step B3 request before any result): overall top-1
agreement >= 96% AND every category of the 24-prompt set >= 90% (Step B2's
rule, bench.quantization_quality.verdict).

Resumable: every finished piece is saved at once under
models/work/gptq_quality/ (continuations per prompt, our final hidden
states per store and prompt, the reference's state after every layer, the
scores per prompt); a rerun skips what is saved. The directory's
settings.json fingerprints the stores, prompts and reference; a mismatch
stops the run instead of mixing results. Writes
benchmarks/results/gptq_quality.json.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import transformers

from expertrelay import REPO_ROOT
from expertrelay.bench.quantization_quality import (
    PASS_CATEGORY,
    PASS_OVERALL,
    PROMPT_FILES,
    aggregate,
    load_prompts,
    our_logits,
    score,
    verdict,
)
from expertrelay.bench.reference_check import Bf16Checkpoint, hf_reference_forward, lm_head_logits
from expertrelay.benchmarking import append_benchmark_record, base_record, peak_process_rss_mb
from expertrelay.manager.profile import collect_machine_profile
from expertrelay.paths import BENCHMARK_RESULTS_DIR, MODELS_ROOT
from expertrelay.runtime.generate import DEFAULT_RUNTIME_CONFIG, RuntimeConfig, generate, load_model
from expertrelay.runtime.int8_linear import set_kernel
from expertrelay.runtime.qwen_moe import ForwardTrace, KVCache
from expertrelay.store.chat_template import ChatTemplate
from expertrelay.store.download_checkpoint import checkpoint_dir
from expertrelay.store.tokenizer import load_tokenizer

STORES = {
    "int8": "qwen1.5-moe-a2.7b-chat-int8",
    "int4_rtn_g128": "qwen1.5-moe-a2.7b-chat-int4g128-rtn",
    "int4_gptq_g128": "qwen1.5-moe-a2.7b-chat-int4g128-gptq",
}
WORK = MODELS_ROOT / "work" / "gptq_quality"
OUT = BENCHMARK_RESULTS_DIR / "gptq_quality.json"
ARABIC = ("ar_msa", "ar_gulf", "phase2_ar")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save_npy(path: Path, a: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, a)
    os.replace(tmp, path)


def _append_line(path: Path, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _read_lines(path: Path) -> dict[int, dict]:
    """Complete lines by prompt index; a line torn by a crash is ignored (and redone)."""
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[d["index"]] = d
    return out


def settings(prompts: list[dict], rt: RuntimeConfig, ref_dir: Path) -> dict:
    return {
        "stores": {
            name: {
                "store": store,
                "experts_index_sha256": _sha256(MODELS_ROOT / store / "experts_index.json"),
                "resident_index_sha256": _sha256(MODELS_ROOT / store / "resident_index.json"),
            }
            for name, store in STORES.items()
        },
        "prompt_files_sha256": {k: _sha256(v) for k, v in PROMPT_FILES.items()},
        "prompts": [p["id"] for p in prompts],
        "max_new_tokens": rt.max_new_tokens,
        "reference": ref_dir.relative_to(REPO_ROOT).as_posix(),
    }


def check_work_dir(current: dict) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / "settings.json"
    if path.exists():
        if json.loads(path.read_text()) != current:
            raise SystemExit(f"{path} is from other stores/prompts; move {WORK} away to start over")
    else:
        path.write_text(json.dumps(current, indent=1))


def check_same_resident() -> None:
    base = MODELS_ROOT / STORES["int8"]
    for store in STORES.values():
        d = MODELS_ROOT / store
        if d != base and json.loads((d / "store.json").read_text())["resident_from"]["store"] != base.name:
            raise SystemExit(f"{store} doesn't share {base.name}'s resident weights")


def continuations(prompts: list[dict], rt: RuntimeConfig) -> tuple[list[list[int]], list[int]]:
    path = WORK / "sequences.jsonl"
    done = _read_lines(path)
    missing = [i for i in range(len(prompts)) if i not in done]
    if missing:
        store = MODELS_ROOT / STORES["int8"]
        tokenizer, template = load_tokenizer(store), ChatTemplate.for_store(store)
        model, _ = load_model(store, "unbuffered", rt.max_seq, rt.memory_budget_gb)
        set_kernel("fused")
        for i in missing:
            ids = tokenizer.encode(template.user_prompt(prompts[i]["text"])).ids
            generated, _ = generate(model, ids, rt.max_new_tokens, rt.max_seq)
            done[i] = {"index": i, "id": prompts[i]["id"], "prompt_len": len(ids), "ids": ids + generated}
            _append_line(path, done[i])
            print(f"continued {prompts[i]['id']}", flush=True)
        model.experts.close()
        del model
        gc.collect()
    return [done[i]["ids"] for i in range(len(prompts))], [done[i]["prompt_len"] for i in range(len(prompts))]


def our_hidden_states(
    name: str, rt: RuntimeConfig, sequences: list[list[int]], prompt_lens: list[int]
) -> None:
    """Final hidden state at every position, run as the runtime runs; one file per prompt."""
    missing = [i for i in range(len(sequences)) if not (WORK / name / f"{i:02d}.npy").exists()]
    if not missing:
        return
    model, _ = load_model(MODELS_ROOT / STORES[name], "unbuffered", rt.max_seq, rt.memory_budget_gb)
    set_kernel("fused")
    for i in missing:
        ids, n = sequences[i], prompt_lens[i]
        cache = KVCache(model.c, rt.max_seq)
        rows = []
        for feed in [ids[:n], *([t] for t in ids[n:])]:
            trace = ForwardTrace()
            model.forward(np.asarray(feed), cache, all_logits=True, trace=trace)
            rows.append(trace.final_hidden)
        _save_npy(WORK / name / f"{i:02d}.npy", np.concatenate(rows))
        print(f"ours {name}: {i + 1}/{len(sequences)}", flush=True)
    model.experts.close()
    del model
    gc.collect()


def reference_hidden_states(ckpt: Bf16Checkpoint, sequences: list[list[int]]) -> None:
    if all((WORK / "bf16" / f"{i:02d}.npy").exists() for i in range(len(sequences))):
        return
    traces = hf_reference_forward(
        ckpt, sequences, keep_layers=False, logits=False, resume_dir=WORK / "bf16_layers"
    )
    for i, t in enumerate(traces):
        _save_npy(WORK / "bf16" / f"{i:02d}.npy", t.final_hidden)


def score_all(
    prompts: list[dict], prompt_lens: list[int], ckpt: Bf16Checkpoint, rt: RuntimeConfig
) -> list[dict]:
    path = WORK / "scores.jsonl"
    done = _read_lines(path)
    missing = [i for i in range(len(prompts)) if i not in done]
    if missing:
        # the lm_head is the same int8 matrix in all three stores (shared resident part)
        model, _ = load_model(MODELS_ROOT / STORES["int8"], "unbuffered", rt.max_seq, rt.memory_budget_gb)
        lm_head, backend = model.resident["lm_head.weight"], model.backend
        model.experts.close()
        set_kernel("fused")
        for i in missing:
            ref = lm_head_logits(ckpt, np.load(WORK / "bf16" / f"{i:02d}.npy"), dtype=torch.float32)
            p = prompts[i]
            rows = {
                name: {
                    "id": p["id"],
                    "set": p["set"],
                    "category": p["category"],
                    **score(
                        ref,
                        our_logits(np.load(WORK / name / f"{i:02d}.npy"), lm_head, backend),
                        prompt_lens[i],
                    ),
                }
                for name in STORES
            }
            done[i] = {"index": i, "rows": rows}
            _append_line(path, done[i])
            print(f"scored {p['id']}", flush=True)
        del model, lm_head
        gc.collect()
    return [done[i]["rows"] for i in range(len(prompts))]


def analyse(per_prompt: list[dict]) -> dict:
    by_precision = {}
    for name in STORES:
        rs = [r[name] for r in per_prompt]
        cats = sorted({r["category"] for r in rs})
        by_precision[name] = {
            "overall": aggregate(rs),
            "by_category": {c: aggregate([r for r in rs if r["category"] == c]) for c in cats},
            "arabic": aggregate([r for r in rs if r["category"] in ARABIC]),
            "phase2": aggregate([r for r in rs if r["set"] == "phase2"]),
            "per_prompt": rs,
        }
    return by_precision


def main() -> None:
    rt = RuntimeConfig.load(DEFAULT_RUNTIME_CONFIG)
    source = json.loads((MODELS_ROOT / STORES["int8"] / "store.json").read_text())["source"]
    ref_dir = checkpoint_dir(source["repo_id"], source["revision"])
    prompts = load_prompts()
    check_same_resident()
    current = settings(prompts, rt, ref_dir)
    check_work_dir(current)

    sequences, prompt_lens = continuations(prompts, rt)
    for name in STORES:
        our_hidden_states(name, rt, sequences, prompt_lens)
    ckpt = Bf16Checkpoint(ref_dir)
    reference_hidden_states(ckpt, sequences)
    per_prompt = score_all(prompts, prompt_lens, ckpt, rt)

    by_precision = analyse(per_prompt)
    record = base_record(
        label="Step B3 quality: int8, int4 RTN and int4 GPTQ experts vs bf16 Chat",
        seed=0,
        model={"stores": STORES, "source": source["revision"], "reference": current["reference"]},
        config={
            "prompt_files": {k: v.relative_to(REPO_ROOT).as_posix() for k, v in PROMPT_FILES.items()},
            "prompts": [{k: p[k] for k in ("set", "id", "category")} for p in prompts],
            "prompt_format": "chat_template",
            "continuation_by": "int8",
            "max_new_tokens": rt.max_new_tokens,
            "pass_overall": PASS_OVERALL,
            "pass_category": PASS_CATEGORY,
            "work_settings": current,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
        },
        machine=collect_machine_profile(measure_disk=False),
    )
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
    print(f"saved {OUT}")
    print(json.dumps(record["analysis"]["verdict"], indent=1))


if __name__ == "__main__":
    main()
