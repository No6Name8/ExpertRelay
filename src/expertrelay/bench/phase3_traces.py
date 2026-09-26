"""Phase 3, step 1: record expert-usage traces for the fixed prompt set.

    python -m expertrelay.bench.phase3_traces                       # base store (configs/phase3.json)
    python -m expertrelay.bench.phase3_traces --store-dir models/qwen1.5-moe-a2.7b-chat-int8
    python -m expertrelay.bench.phase3_traces --estimate-only

One trace file per prompt, in models/traces/<store directory name>/ (see
docs/trace-format.md), plus a small <prompt id>.done.json per finished
prompt. The run is resumable: prompts with a .done.json are skipped, so an
interrupted overnight run picks up where it stopped. Prompts are run
round-robin across categories, so a partial run still covers every
category.

Prompt format depends on the store: the base model gets the text as-is;
a Chat model (source repo ending in "-Chat") gets it wrapped in its own
chat template (Qwen1.5 ChatML, verified against the pinned revision's
tokenizer_config.json). Recorded in every trace header and the run record.

The runtime estimate uses Phase 2's measured speeds
(benchmarks/results/phase2_baselines.json, setup C). The actual runtime is
recorded next to it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from itertools import zip_longest
from pathlib import Path

from expertrelay import REPO_ROOT
from expertrelay.benchmarking import append_benchmark_record, base_record, peak_process_rss_mb
from expertrelay.manager.profile import collect_machine_profile
from expertrelay.paths import BENCHMARK_RESULTS_DIR, MODELS_ROOT
from expertrelay.runtime.expert_trace import ExpertTraceWriter
from expertrelay.runtime.generate import RuntimeConfig, generate, load_model
from expertrelay.store.tokenizer import load_tokenizer

DEFAULT_CONFIG = REPO_ROOT / "configs" / "phase3.json"
PHASE2_RESULTS = BENCHMARK_RESULTS_DIR / "phase2_baselines.json"
TRACES_ROOT = MODELS_ROOT / "traces"
# Qwen1.5-Chat's template (tokenizer_config.json chat_template at the pinned
# revision), specialized to one user turn with the default system prompt.
CHATML = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
)


def trace_dir_for(store_dir: Path) -> Path:
    return TRACES_ROOT / Path(store_dir).name


def prompt_format_for(store_source: dict) -> str:
    return "chatml" if store_source["repo_id"].endswith("-Chat") else "plain"


def format_prompt(text: str, fmt: str) -> str:
    return CHATML.format(text=text) if fmt == "chatml" else text


def round_robin(prompts: list[dict]) -> list[dict]:
    by_cat: dict[str, list[dict]] = {}
    for p in prompts:
        by_cat.setdefault(p["category"], []).append(p)
    return [p for group in zip_longest(*by_cat.values()) for p in group if p is not None]


def measured_speeds() -> dict:
    """Seconds per prompt token (prefill) and per generated token (decode),
    from the latest Phase 2 run of setup C."""
    record = json.loads(PHASE2_RESULTS.read_text(encoding="utf-8"))[-1]
    c = next(r for r in record["results"] if r["id"] == "C")
    prefill = [p["steps"][0] for p in c["run"]["prompts"]]
    return {
        "prefill_s_per_token": sum(s["total_seconds"] for s in prefill) / sum(s["tokens"] for s in prefill),
        "decode_s_per_token": 1 / c["metrics"]["decode_tokens_per_s"],
        "source": f"{PHASE2_RESULTS.relative_to(REPO_ROOT).as_posix()} @ {record['git_commit'][:7]}",
    }


def estimate_seconds(prompt_lengths: list[int], max_new_tokens: int, speeds: dict) -> float:
    return sum(
        n * speeds["prefill_s_per_token"] + (max_new_tokens - 1) * speeds["decode_s_per_token"]
        for n in prompt_lengths
    )


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--store-dir", type=Path, default=None, help="default: store_dir from the config")
    ap.add_argument("--estimate-only", action="store_true")
    args = ap.parse_args()

    rt = RuntimeConfig.load(args.config)
    if args.store_dir is not None:
        rt = replace(rt, store_dir=args.store_dir.resolve())
    source = json.loads((rt.store_dir / "store.json").read_text())["source"]
    fmt = prompt_format_for(source)
    prompts = round_robin(json.loads(rt.prompts_file.read_text(encoding="utf-8"))["prompts"])
    tokenizer = load_tokenizer(rt.store_dir)
    encoded = {p["id"]: tokenizer.encode(format_prompt(p["text"], fmt)).ids for p in prompts}
    too_long = [pid for pid, ids in encoded.items() if len(ids) + rt.max_new_tokens > rt.max_seq]
    if too_long:
        raise SystemExit(f"prompts too long for max_seq={rt.max_seq}: {too_long}")

    speeds = measured_speeds()
    estimate = estimate_seconds([len(v) for v in encoded.values()], rt.max_new_tokens, speeds)
    print(
        f"{len(prompts)} prompts x {rt.max_new_tokens} new tokens, format {fmt}: "
        f"estimated {estimate / 3600:.1f} h from {speeds['source']}",
        flush=True,
    )
    if args.estimate_only:
        return

    out_dir = trace_dir_for(rt.store_dir)
    record = base_record(
        label="phase3 expert traces",
        seed=0,
        model={"store": rt.store_dir.name, "repo_id": source["repo_id"], "revision": source["revision"]},
        config={
            **{k: str(v) if isinstance(v, Path) else v for k, v in asdict(rt).items()},
            "prompt_format": fmt,
            "trace_dir": out_dir.relative_to(REPO_ROOT).as_posix(),
        },
        machine=collect_machine_profile(measure_disk=False),
        estimate={"seconds": estimate, **speeds},
    )
    model, load_info = load_model(rt.store_dir, "unbuffered", rt.max_seq, rt.memory_budget_gb)
    started = time.perf_counter()
    done_now = 0
    for i, p in enumerate(prompts, start=1):
        done_file = out_dir / f"{p['id']}.done.json"
        if done_file.exists():
            continue
        t0 = time.perf_counter()
        writer = ExpertTraceWriter(
            out_dir / f"{p['id']}.ert",
            num_layers=model.c.num_layers,
            num_experts=model.c.num_experts,
            top_k=model.c.top_k,
            metadata={
                "prompt_id": p["id"],
                "category": p["category"],
                "language": p["language"],
                "prompt_format": fmt,
                "store": rt.store_dir.name,
                "source_revision": source["revision"],
                "max_new_tokens": rt.max_new_tokens,
            },
        )
        generated, steps = generate(
            model, encoded[p["id"]], rt.max_new_tokens, rt.max_seq, expert_trace=writer
        )
        seconds = time.perf_counter() - t0
        done_file.write_text(
            json.dumps(
                {
                    "id": p["id"],
                    "category": p["category"],
                    "prompt_tokens": len(encoded[p["id"]]),
                    "generated_ids": generated,
                    "generated_text": tokenizer.decode(generated),
                    "seconds": seconds,
                    "prefill_seconds": steps[0].total_seconds,
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        done_now += 1
        elapsed = time.perf_counter() - started
        print(
            f"[{i}/{len(prompts)}] {p['id']}: {seconds:.0f} s, run elapsed {elapsed / 3600:.2f} h", flush=True
        )
    model.experts.close()

    done = [json.loads((out_dir / f"{p['id']}.done.json").read_text(encoding="utf-8")) for p in prompts]
    record["results"] = {
        "prompts_done": len(done),
        "prompts_run_this_session": done_now,
        "actual_seconds_total": sum(d["seconds"] for d in done),
        "estimated_seconds_total": estimate,
        "trace_bytes_total": sum((out_dir / f"{p['id']}.ert").stat().st_size for p in prompts),
        "load": load_info,
        "per_prompt": [
            {k: d[k] for k in ("id", "category", "prompt_tokens", "seconds", "prefill_seconds")} for d in done
        ],
        "peak_rss_mb": peak_process_rss_mb(),
    }
    append_benchmark_record(BENCHMARK_RESULTS_DIR / f"phase3_trace_run_{rt.store_dir.name}.json", record)
    print(f"all {len(done)} prompts traced in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
