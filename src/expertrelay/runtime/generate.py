"""Greedy generation from the int8 store, with every step instrumented.

    python -m expertrelay.runtime.generate --prompt "The capital of France is"
    python -m expertrelay.runtime.generate --all-prompts --json-out out.json --source mmap

For every forward call (the prompt prefill, then one call per new token) it
records wall time, time spent reading experts from disk, expert bytes read,
and bytes the OS actually read from physical disks. Compute time is wall
time minus expert read time. With `--source mmap` the reads happen as page
faults inside the matmuls, so they land in "compute" and only the device
byte count shows them.

Generation is greedy and always produces exactly max_new_tokens: EOS does
not stop it, so every run does the same amount of work.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from expertrelay import REPO_ROOT
from expertrelay.benchmarking import peak_process_rss_mb
from expertrelay.manager.backend_selection import BackendChoice, select_backend
from expertrelay.manager.profile import collect_machine_profile, device_read_bytes, measure_memory
from expertrelay.memory_budget import enforce_ram_budget
from expertrelay.runtime.backends import BACKEND_NAMES, make_backend
from expertrelay.runtime.expert_trace import ExpertTraceWriter
from expertrelay.runtime.int8_linear import BLOCK_ROWS_PREFILL
from expertrelay.runtime.qwen_moe import KVCache, ModelConfig, QwenMoe
from expertrelay.runtime.weights import (
    EMBEDDING,
    RESIDENT_INDEX,
    ExpertSource,
    MmapExpertSource,
    RamExpertSource,
    ResidentWeights,
    UnbufferedExpertSource,
)
from expertrelay.store.layout import read_resident_index
from expertrelay.store.tokenizer import load_tokenizer

DEFAULT_RUNTIME_CONFIG = REPO_ROOT / "configs" / "runtime.json"
SOURCES: dict[str, type[ExpertSource]] = {
    "unbuffered": UnbufferedExpertSource,
    "mmap": MmapExpertSource,
    "ram": RamExpertSource,
}
# Interpreter + numpy + tokenizers, measured on the dev machine before any
# weights are loaded (~150 MB), rounded up.
PROCESS_BASELINE_BYTES = 200 * 2**20
# Importing MindSpore and running a first op: measured 195 MB on the dev
# machine, rounded up. Only paid when the MindSpore backend is selected.
MINDSPORE_BACKEND_BYTES = 256 * 2**20


@dataclass(frozen=True)
class RuntimeConfig:
    memory_budget_gb: float
    max_seq: int
    max_new_tokens: int
    prompts_file: Path
    store_dir: Path

    @classmethod
    def load(cls, path: Path = DEFAULT_RUNTIME_CONFIG) -> RuntimeConfig:
        d = json.loads(Path(path).read_text())
        return cls(
            memory_budget_gb=d["memory_budget_gb"],
            max_seq=d["max_seq"],
            max_new_tokens=d["max_new_tokens"],
            prompts_file=REPO_ROOT / d["prompts_file"],
            store_dir=REPO_ROOT / d["store_dir"],
        )


def model_config(store_dir: Path) -> ModelConfig:
    return ModelConfig.from_hf(json.loads((Path(store_dir) / "store.json").read_text())["source"]["config"])


def estimate_ram_bytes(
    store_dir: Path, config: ModelConfig, max_seq: int, source: str, backend: str = "numpy"
) -> int:
    """Upper-bound estimate of this process's RAM for a run, used to enforce
    the budget BEFORE loading anything. Every term is computed from real
    shapes, none are guessed except the fixed interpreter baseline."""
    store_dir = Path(store_dir)
    entries = read_resident_index(store_dir / RESIDENT_INDEX)
    resident = 0 if source == "mmap" else sum(e.region_size for e in entries if e.name != EMBEDDING)
    manifest = json.loads((store_dir / "store.json").read_text())
    record = manifest["expert_layout"]["record_size"]
    experts = {
        "unbuffered": record,  # one aligned read buffer; experts are used as views into it
        "mmap": 0,  # page cache, not process-private memory
        "ram": config.num_layers * config.num_experts * record,
    }[source]
    widest = max(config.hidden_size, max(e.shape[-1] for e in entries if len(e.shape) == 2))
    tallest = max(e.shape[0] for e in entries if len(e.shape) == 2 and e.name != EMBEDDING)
    scratch = BLOCK_ROWS_PREFILL * widest * 4
    # prefill of max_seq tokens: a few [n, widest] f32 intermediates, attention
    # scores [heads, n, n], and logits for the last position only
    activations = max_seq * widest * 4 * 6 + config.num_heads * max_seq * max_seq * 4 + tallest * 4
    return (
        PROCESS_BASELINE_BYTES
        + (MINDSPORE_BACKEND_BYTES if backend == "mindspore" else 0)
        + resident
        + experts
        + KVCache.bytes_for(config, max_seq)
        + scratch
        + activations
    )


def load_model(
    store_dir: Path,
    source: str,
    max_seq: int,
    budget_gb: float | None,
    backend: BackendChoice | None = None,
) -> tuple[QwenMoe, dict]:
    """budget_gb=None disables enforcement. Only the "normal load" baseline
    does that, precisely to show what happens without it. backend=None lets
    the Manager choose from this machine's profile."""
    choice = backend or select_backend(collect_machine_profile(measure_disk=False))
    config = model_config(store_dir)
    estimate = estimate_ram_bytes(store_dir, config, max_seq, source, choice.name)
    if budget_gb is not None:
        enforce_ram_budget(estimate, budget_gb, f"running with --source {source} --backend {choice.name}")
    t = time.perf_counter()
    resident = ResidentWeights.load(store_dir, mmap_everything=(source == "mmap"))
    experts = SOURCES[source](store_dir)
    compute = make_backend(choice.name, choice.device)
    info = {
        "source": source,
        "backend": asdict(choice),
        "estimated_ram_bytes": estimate,
        "budget_gb": budget_gb,
        "resident_ram_bytes": resident.ram_bytes,
        "load_seconds": time.perf_counter() - t,
    }
    return QwenMoe(config, resident, experts, compute), info


@dataclass
class StepRecord:
    kind: str  # "prefill" or "decode"
    tokens: int
    total_seconds: float
    expert_read_seconds: float
    compute_seconds: float
    expert_loads: int
    expert_bytes: int
    device_bytes: int


def generate(
    model: QwenMoe,
    prompt_ids: list[int],
    max_new_tokens: int,
    max_seq: int,
    expert_trace: ExpertTraceWriter | None = None,
) -> tuple[list[int], list[StepRecord]]:
    """Greedy, fixed length. With `expert_trace`, every forward call's router
    decisions are recorded (runtime.expert_trace); outputs are unchanged."""
    cache = KVCache(model.c, max_seq)
    steps: list[StepRecord] = []
    generated: list[int] = []
    feed = prompt_ids
    for i in range(max_new_tokens):
        dev0 = device_read_bytes()
        logits, t = model.forward(np.asarray(feed), cache, expert_trace=expert_trace)
        steps.append(
            StepRecord(
                kind="prefill" if i == 0 else "decode",
                tokens=len(feed),
                total_seconds=t.total_seconds,
                expert_read_seconds=t.expert_read_seconds,
                compute_seconds=t.compute_seconds,
                expert_loads=t.expert_loads,
                expert_bytes=t.expert_bytes,
                device_bytes=device_read_bytes() - dev0,
            )
        )
        nxt = int(np.argmax(logits))
        generated.append(nxt)
        feed = [nxt]
    return generated, steps


def run_prompts(
    store_dir: Path,
    source: str,
    rt: RuntimeConfig,
    prompts: list[dict],
    budget_gb: float | None,
    backend: BackendChoice | None = None,
) -> dict:
    """Load once, generate for every prompt, return everything as plain data."""
    free_at_start = measure_memory().available_bytes
    model, info = load_model(store_dir, source, rt.max_seq, budget_gb, backend)
    tokenizer = load_tokenizer(store_dir)
    results = []
    for p in prompts:
        prompt_ids = tokenizer.encode(p["text"]).ids
        generated, steps = generate(model, prompt_ids, rt.max_new_tokens, rt.max_seq)
        results.append(
            {
                "id": p["id"],
                "prompt_ids": prompt_ids,
                "generated_ids": generated,
                "generated_text": tokenizer.decode(generated),
                "steps": [asdict(s) for s in steps],
            }
        )
        print(f"[{source}] {p['id']}: {tokenizer.decode(generated)!r}", flush=True)
    model.experts.close()
    return {
        "load": info,
        "ram_available_at_start_bytes": free_at_start,
        "prompts": results,
        "peak_rss_mb": peak_process_rss_mb(),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_RUNTIME_CONFIG)
    ap.add_argument("--store-dir", type=Path, default=None, help="default: store_dir from the config")
    ap.add_argument("--source", choices=sorted(SOURCES), default="unbuffered")
    ap.add_argument(
        "--backend",
        choices=["auto", *BACKEND_NAMES],
        default="auto",
        help="compute backend; auto = the Manager chooses from this machine's profile",
    )
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--all-prompts", action="store_true", help="run every prompt in the config's prompt set")
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument(
        "--no-budget", action="store_true", help="disable RAM budget enforcement (normal-load baseline)"
    )
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    rt = RuntimeConfig.load(args.config)
    if args.max_new_tokens is not None:
        rt = RuntimeConfig(**{**asdict(rt), "max_new_tokens": args.max_new_tokens})
    store_dir = args.store_dir or rt.store_dir
    if args.all_prompts:
        prompts = json.loads(rt.prompts_file.read_text(encoding="utf-8"))["prompts"]
    elif args.prompt is not None:
        prompts = [{"id": "cli", "text": args.prompt}]
    else:
        ap.error("give --prompt or --all-prompts")

    backend = (
        None if args.backend == "auto" else BackendChoice(args.backend, None, "chosen on the command line")
    )
    out = run_prompts(
        store_dir, args.source, rt, prompts, None if args.no_budget else rt.memory_budget_gb, backend
    )
    if args.json_out:
        args.json_out.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    for p in out["prompts"]:
        decode = [s for s in p["steps"] if s["kind"] == "decode"]
        tps = len(decode) / sum(s["total_seconds"] for s in decode) if decode else float("nan")
        print(f"{p['id']}: first token {p['steps'][0]['total_seconds']:.2f}s, decode {tps:.2f} tok/s")
    print(f"peak RSS {out['peak_rss_mb']:.0f} MB")


if __name__ == "__main__":
    main()
