"""Greedy generation from the int8 store, with every step instrumented.

    python -m expertrelay.runtime.generate --prompt "The capital of France is"
    python -m expertrelay.runtime.generate --all-prompts --json-out out.json --source mmap
    python -m expertrelay.runtime.generate --config configs/runtime_cache.json --source cached \
        --prompt "..." [--no-prefetch] [--pin-layer0] [--cache-gb 1.66] [--prefetch-k 8]

For every forward call (the prompt prefill, then one call per new token) it
records wall time, time spent reading experts from disk, expert bytes read,
and bytes the OS actually read from physical disks. Compute time is wall
time minus expert read time. With `--source mmap` the reads happen as page
faults inside the matmuls, so they land in "compute" and only the device
byte count shows them.

Prompts from a Chat store are wrapped in the model's own chat template
(store.chat_template) unless `--no-chat-template` asks for raw text.

With `--source cached` experts stay in an LRU cache in RAM (budget
`expert_cache_gb`), optionally with every layer-0 expert pinned, and the
prefetcher loads the Fate-guessed experts for the next layer in the
background (`--prefetch/--no-prefetch`: the demo's on/off switch). Each
step then also records cache hits, prefetch hits, waits for in-flight
prefetches, wasted and cancelled prefetches, and per-layer attention / MoE
compute / blocked-on-reads time.

Generation is greedy and always produces exactly max_new_tokens: EOS does
not stop it, so every run does the same amount of work.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from expertrelay import REPO_ROOT
from expertrelay.benchmarking import peak_process_rss_mb
from expertrelay.cache.expert_cache import CachedExpertSource, slots_for
from expertrelay.manager.backend_selection import BackendChoice, select_backend
from expertrelay.manager.profile import collect_machine_profile, device_read_bytes, measure_memory
from expertrelay.memory_budget import enforce_ram_budget
from expertrelay.predictor.prefetch_policy import PrefetchPolicy, load_calibrator
from expertrelay.runtime.backends import BACKEND_NAMES, make_backend
from expertrelay.runtime.expert_trace import ExpertTraceWriter
from expertrelay.runtime.int8_linear import BLOCK_ROWS_PREFILL
from expertrelay.runtime.int8_linear import KERNELS as INT8_KERNELS
from expertrelay.runtime.int8_linear import current_kernel as current_int8_kernel
from expertrelay.runtime.int8_linear import set_kernel as set_int8_kernel
from expertrelay.runtime.qwen_moe import KVCache, ModelConfig, QwenMoe, StepTimings
from expertrelay.runtime.weights import (
    EMBEDDING,
    RESIDENT_INDEX,
    ExpertSource,
    MmapExpertSource,
    RamExpertSource,
    ReadLog,
    ResidentWeights,
    UnbufferedExpertSource,
)
from expertrelay.store.chat_template import ChatTemplate, is_chat_store
from expertrelay.store.layout import read_resident_index
from expertrelay.store.tokenizer import load_tokenizer

DEFAULT_RUNTIME_CONFIG = REPO_ROOT / "configs" / "runtime.json"
SOURCES: dict[str, type[ExpertSource]] = {
    "unbuffered": UnbufferedExpertSource,
    "mmap": MmapExpertSource,
    "ram": RamExpertSource,
    "cached": CachedExpertSource,
}
# Interpreter + numpy + tokenizers, measured on the dev machine before any
# weights are loaded (~150 MB), rounded up.
PROCESS_BASELINE_BYTES = 200 * 2**20
# Importing MindSpore and running a first op: measured 195 MB on the dev
# machine, rounded up. Only paid when the MindSpore backend is selected.
MINDSPORE_BACKEND_BYTES = 256 * 2**20
# numba plus the compiled fused int8 kernel: measured 83 MB on the dev
# machine, rounded up. Only paid when the fused kernel is selected.
FUSED_KERNEL_BYTES = 128 * 2**20


@dataclass(frozen=True)
class RuntimeConfig:
    """The cache/prefetch fields only matter with --source cached, and are
    optional in the file, so the Phase 2/3 configs load unchanged."""

    memory_budget_gb: float
    max_seq: int
    max_new_tokens: int
    prompts_file: Path
    store_dir: Path
    expert_cache_gb: float = 0.0
    prefetch: bool = False
    prefetch_k: int = 8
    prefetch_min_probability: float = 0.0
    prefetch_calibration: Path | None = None
    pin_layer0: bool = False
    io_threads: int = 1
    # None = the store decides (store.chat_template.is_chat_store): Chat stores
    # get their chat template, base stores raw text
    chat_template: bool | None = None
    int8_kernel: str = "fused"  # runtime.int8_linear: "fused" (numba) or "blocked"
    prefill_pipelining: bool = True  # --source cached: read a layer's experts in parallel in prefill
    prefetch_adaptive: bool = False  # cap prefetch reads at what fits in the measured window
    prefetch_calibration_kind: str = "rank"  # "rank" (per-rank maps) or "single"
    log_reads: bool = False  # record every expert read (runtime.weights.ReadLog)

    @classmethod
    def load(cls, path: Path = DEFAULT_RUNTIME_CONFIG) -> RuntimeConfig:
        d = json.loads(Path(path).read_text())
        calibration = d.get("prefetch_calibration")
        return cls(
            memory_budget_gb=d["memory_budget_gb"],
            max_seq=d["max_seq"],
            max_new_tokens=d["max_new_tokens"],
            prompts_file=REPO_ROOT / d["prompts_file"],
            store_dir=REPO_ROOT / d["store_dir"],
            expert_cache_gb=d.get("expert_cache_gb", 0.0),
            prefetch=d.get("prefetch", False),
            prefetch_k=d.get("prefetch_k", 8),
            prefetch_min_probability=d.get("prefetch_min_probability", 0.0),
            prefetch_calibration=REPO_ROOT / calibration if calibration else None,
            pin_layer0=d.get("pin_layer0", False),
            io_threads=d.get("io_threads", 1),
            chat_template=d.get("chat_template"),
            int8_kernel=d.get("int8_kernel", "fused"),
            prefill_pipelining=d.get("prefill_pipelining", True),
            prefetch_adaptive=d.get("prefetch_adaptive", False),
            prefetch_calibration_kind=d.get("prefetch_calibration_kind", "rank"),
            log_reads=d.get("log_reads", False),
        )

    def cache_slots_needed(self, config: ModelConfig) -> tuple[int, int]:
        """(pinned, free) slots the cache must hold. Free = one layer's experts
        in use + one prefetch batch + 1. Counted even with prefetch off, so
        the switch can be turned on mid-run."""
        pinned = config.num_experts if self.pin_layer0 else 0
        return pinned, config.top_k + self.prefetch_k + 1


def model_config(store_dir: Path) -> ModelConfig:
    return ModelConfig.from_hf(json.loads((Path(store_dir) / "store.json").read_text())["source"]["config"])


def estimate_ram_bytes(
    store_dir: Path,
    config: ModelConfig,
    max_seq: int,
    source: str,
    backend: str = "numpy",
    expert_cache_gb: float = 0.0,
    int8_kernel: str = "blocked",
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
        # the slot pool, allocated once; reads go straight into slots
        "cached": slots_for(round(expert_cache_gb * 1e9), record) * record,
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
        + (FUSED_KERNEL_BYTES if backend == "numpy" and int8_kernel == "fused" else 0)
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
    rt: RuntimeConfig | None = None,
) -> tuple[QwenMoe, dict]:
    """budget_gb=None disables enforcement. Only the "normal load" baseline
    does that, precisely to show what happens without it. backend=None lets
    the Manager choose from this machine's profile. `rt` carries the cache
    and prefetch settings; required for source "cached"."""
    choice = backend or select_backend(collect_machine_profile(measure_disk=False))
    config = model_config(store_dir)
    if source == "cached" and rt is None:
        raise ValueError("source 'cached' needs the runtime config's cache settings")
    cache_gb = rt.expert_cache_gb if source == "cached" else 0.0
    kernel = rt.int8_kernel if rt else RuntimeConfig.int8_kernel
    estimate = estimate_ram_bytes(store_dir, config, max_seq, source, choice.name, cache_gb, kernel)
    if budget_gb is not None:
        enforce_ram_budget(estimate, budget_gb, f"running with --source {source} --backend {choice.name}")
    info: dict = {"source": source}
    if choice.name == "numpy":
        try:
            set_int8_kernel(kernel)
        except ImportError:  # numba missing: the blocked kernel still works
            set_int8_kernel("blocked")
        info["int8_kernel"] = current_int8_kernel()
    t = time.perf_counter()
    resident = ResidentWeights.load(store_dir, mmap_everything=(source == "mmap"))
    prefetcher = None
    if source == "cached":
        pinned, free = rt.cache_slots_needed(config)
        experts = CachedExpertSource(
            store_dir,
            capacity_bytes=round(
                cache_gb * 1e9
            ),  # to the byte: int() would truncate 1.2486574 GB into one slot less
            pinned=[(0, e) for e in range(pinned)],
            io_threads=rt.io_threads,
            min_free_slots=free,
        )
        calibrator, provenance = (
            load_calibrator(rt.prefetch_calibration, rt.prefetch_calibration_kind)
            if rt.prefetch_calibration
            else (None, None)
        )
        prefetcher = PrefetchPolicy(rt.prefetch_k, rt.prefetch_min_probability, calibrator)
        info["cache"] = {
            "budget_gb": cache_gb,
            "slots": experts.num_slots,
            "bytes": experts.capacity_bytes,
            "pinned_layer0": rt.pin_layer0,
            "io_threads": rt.io_threads,
            "prefetch": rt.prefetch,
            "prefetch_k": rt.prefetch_k,
            "prefetch_min_probability": rt.prefetch_min_probability,
            "prefetch_adaptive": rt.prefetch_adaptive,
            "prefill_pipelining": rt.prefill_pipelining,
            "calibration": provenance,
            # a calibration fitted on another store's traces is a simplification
            "calibration_store_matches": provenance is None or provenance["store"] == Path(store_dir).name,
        }
    else:
        experts = SOURCES[source](store_dir)
    compute = make_backend(choice.name, choice.device)
    info.update(
        {
            "backend": asdict(choice),
            "estimated_ram_bytes": estimate,
            "budget_gb": budget_gb,
            "resident_ram_bytes": resident.ram_bytes,
            "load_seconds": time.perf_counter() - t,
        }
    )
    model = QwenMoe(config, resident, experts, compute, prefetcher)
    model.prefetch_enabled = bool(rt and rt.prefetch and prefetcher is not None)
    model.prefetch_adaptive = bool(rt and rt.prefetch_adaptive)
    model.prefill_pipelining = bool(rt and rt.prefill_pipelining)
    if rt and rt.log_reads and hasattr(experts, "read_log"):
        experts.read_log = ReadLog()
    return model, info


@dataclass
class StepRecord:
    """One forward call: StepTimings' fields (see runtime.qwen_moe), plus
    what the OS says was read from physical disks."""

    kind: str  # "prefill" or "decode"
    tokens: int
    total_seconds: float
    expert_read_seconds: float
    compute_seconds: float
    expert_loads: int
    expert_bytes: int
    device_bytes: int
    demand_reads: int = 0
    cache_hits: int = 0
    prefetch_hits: int = 0
    prefetch_waits: int = 0
    prefetches_issued: int = 0
    prefetches_skipped_low_confidence: int = 0
    prefetch_reads: int = 0
    prefetch_bytes: int = 0
    prefetches_wasted: int = 0
    prefetches_cancelled: int = 0
    prefetches_skipped_budget: int = 0
    attention_seconds: list[float] | None = None
    moe_compute_seconds: list[float] | None = None
    read_wait_seconds: list[float] | None = None

    @classmethod
    def from_timings(cls, kind: str, tokens: int, t: StepTimings, device_bytes: int) -> StepRecord:
        fields = {k: v for k, v in asdict(t).items() if k in cls.__dataclass_fields__}
        return cls(
            kind=kind, tokens=tokens, compute_seconds=t.compute_seconds, device_bytes=device_bytes, **fields
        )


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
            StepRecord.from_timings(
                "prefill" if i == 0 else "decode", len(feed), t, device_read_bytes() - dev0
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
    model, info = load_model(store_dir, source, rt.max_seq, budget_gb, backend, rt)
    tokenizer = load_tokenizer(store_dir)
    use_template = rt.chat_template if rt.chat_template is not None else is_chat_store(store_dir)
    template = ChatTemplate.for_store(store_dir) if use_template else None
    results = []
    log = getattr(model.experts, "read_log", None)
    for p in prompts:
        prompt_ids = tokenizer.encode(template.user_prompt(p["text"]) if template else p["text"]).ids
        first_read = len(log.entries) if log else None
        generated, steps = generate(model, prompt_ids, rt.max_new_tokens, rt.max_seq)
        results.append(
            {
                "id": p["id"],
                "prompt_ids": prompt_ids,
                "generated_ids": generated,
                "generated_text": tokenizer.decode(generated),
                "steps": [asdict(s) for s in steps],
                "read_log_range": [first_read, len(log.entries)] if log else None,
            }
        )
        print(f"[{source}] {p['id']}: {tokenizer.decode(generated)!r}", flush=True)
    model.experts.close()
    return {
        "load": info,
        "prompt_format": "chat_template" if template else "raw",
        "ram_available_at_start_bytes": free_at_start,
        "prompts": results,
        # (start s, seconds, idle gap before s, reads already in flight, kind, layer, expert)
        "read_log": [list(e) for e in log.entries] if log else None,
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
    cache = ap.add_argument_group("expert cache and prefetch (--source cached; defaults from the config)")
    cache.add_argument("--cache-gb", type=float, default=None, help="RAM for cached experts")
    cache.add_argument(
        "--prefetch", action=argparse.BooleanOptionalAction, default=None, help="the prediction on/off switch"
    )
    cache.add_argument("--prefetch-k", type=int, default=None)
    cache.add_argument("--prefetch-min-probability", type=float, default=None)
    cache.add_argument("--pin-layer0", action=argparse.BooleanOptionalAction, default=None)
    cache.add_argument("--io-threads", type=int, default=None)
    ap.add_argument("--int8-kernel", choices=list(INT8_KERNELS), default=None)
    ap.add_argument("--log-reads", action=argparse.BooleanOptionalAction, default=None)
    cache.add_argument("--prefill-pipelining", action=argparse.BooleanOptionalAction, default=None)
    cache.add_argument("--prefetch-adaptive", action=argparse.BooleanOptionalAction, default=None)
    cache.add_argument("--calibration-kind", choices=["rank", "single"], default=None)
    ap.add_argument(
        "--chat-template",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="wrap prompts in the model's chat template (default: yes for Chat stores, no for base stores)",
    )
    args = ap.parse_args()

    rt = RuntimeConfig.load(args.config)
    overrides = {
        "max_new_tokens": args.max_new_tokens,
        "expert_cache_gb": args.cache_gb,
        "prefetch": args.prefetch,
        "prefetch_k": args.prefetch_k,
        "prefetch_min_probability": args.prefetch_min_probability,
        "pin_layer0": args.pin_layer0,
        "io_threads": args.io_threads,
        "chat_template": args.chat_template,
        "int8_kernel": args.int8_kernel,
        "log_reads": args.log_reads,
        "prefill_pipelining": args.prefill_pipelining,
        "prefetch_adaptive": args.prefetch_adaptive,
        "prefetch_calibration_kind": args.calibration_kind,
    }
    rt = replace(rt, **{k: v for k, v in overrides.items() if v is not None})
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
