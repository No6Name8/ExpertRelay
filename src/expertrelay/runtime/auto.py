"""--auto: the Manager measures this machine and configures the runtime.

    python -m expertrelay.runtime.generate --auto --all-prompts --store-dir models/qwen1.5-moe-a2.7b-chat-int8
    python -m expertrelay.runtime.generate --auto --fast ...               # GPTQ int4 experts
    python -m expertrelay.runtime.generate --auto --budget-file budget.txt # live memory budget

Startup, in order:
  1. machine profile (free RAM, cores, accelerators), before anything is
     loaded, so free RAM is what the run can really use;
  2. read probe on the store (manager.probe: one expert read with 1 and
     with 2 in flight);
  3. load the resident weights with no expert cache, and run a few decode
     tokens of a fixed probe prompt to measure compute per layer
     (probe_compute; reads are excluded from those numbers);
  4. manager.policy.decide: every setting, each with its reason;
  5. build the expert cache and prefetcher it chose, and start.
The probe's tokens are discarded and its reads bypass the cache (no
cache exists yet), so it doesn't warm anything the run then uses, and the
adaptive prefetcher's window estimates start empty, as in a hand-set run.

Live memory budget (LiveBudget): between tokens, the budget can be
changed (from a file the demo's slider writes, or set()); the cache is
resized to what the new budget and the RAM free right now allow, by the
same rule as at startup. Outputs don't depend on the cache size, so the
tokens stay the same.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from expertrelay.cache.expert_cache import CachedExpertSource, CacheTooSmall
from expertrelay.manager.backend_selection import select_backend
from expertrelay.manager.policy import (
    PREFETCH_ADAPTIVE,
    PREFETCH_OFF,
    ComputeProbe,
    Decisions,
    StoreFacts,
    cache_slots_for,
    decide,
    safety_margin_bytes,
)
from expertrelay.manager.probe import probe_reads
from expertrelay.manager.profile import collect_machine_profile, measure_memory
from expertrelay.memory_budget import enforce_ram_budget
from expertrelay.paths import BENCHMARK_RESULTS_DIR
from expertrelay.runtime.generate import (
    RuntimeConfig,
    configure_prefetch,
    estimate_ram_bytes,
    load_model,
    make_cached_source,
    model_config,
)
from expertrelay.runtime.qwen_moe import KVCache, QwenMoe
from expertrelay.runtime.weights import UnbufferedExpertSource

PROBE_TOKENS = 4  # decode tokens measured (after one discarded warm-up token)
PROBE_PROMPT_IDS = [785, 6722, 315, 9625, 374]  # fixed ids; any short prompt works, these are never shown
GPTQ_QUALITY = BENCHMARK_RESULTS_DIR / "gptq_quality.json"


def fast_store_for(store_dir: Path) -> Path:
    """The GPTQ int4 sibling of an int8 store (store.build_int4_store's naming)."""
    fast = Path(store_dir).with_name(Path(store_dir).name.replace("int8", "int4g128") + "-gptq")
    if not fast.exists():
        raise SystemExit(f"--fast needs the GPTQ int4 store {fast}; build it with store.build_int4_store")
    return fast


def fast_quality_note() -> str:
    if not GPTQ_QUALITY.exists():
        return "not measured on this checkout"
    v = json.loads(GPTQ_QUALITY.read_text(encoding="utf-8"))[-1]["analysis"]["verdict"]
    g, i = v["int4_gptq_g128"], v["int8"]
    return (
        f"top-1 agreement with bf16 {g['overall_top1']:.1%} vs int8's {i['overall_top1']:.1%}, worst category "
        f"{g['worst_category']} {g['worst_category_top1']:.1%}: {'PASS' if g['pass'] else 'FAIL'} against the "
        "96% / 90% quality bar (benchmarks/results/gptq_quality.json)"
    )


def calibration_for(store_dir: Path) -> Path | None:
    path = BENCHMARK_RESULTS_DIR / f"prefetch_calibration_{Path(store_dir).name}.json"
    return path if path.exists() else None


def top_guess_hit_rate(store_dir: Path) -> float | None:
    """How often the rank-1 prefetch guess is used, on the calibration's test prompts."""
    path = calibration_for(store_dir)
    if path is None:
        return None
    record = json.loads(path.read_text(encoding="utf-8"))[-1]
    return record["test_rank_summary"][0]["observed_picked"]


def probe_compute(model: QwenMoe, tokens: int = PROBE_TOKENS) -> ComputeProbe:
    """Compute per decode token and per layer window, from `tokens` decode
    steps after one warm-up step; read waits are excluded by StepTimings."""
    prompt = [i % model.c.vocab_size for i in PROBE_PROMPT_IDS]
    cache = KVCache(model.c, len(prompt) + tokens + 2)
    logits, _ = model.forward(np.asarray(prompt), cache)
    nxt = int(np.argmax(logits))
    logits, _ = model.forward(np.asarray([nxt]), cache)  # warm-up, discarded
    per_token, windows = [], []
    for _ in range(tokens):
        nxt = int(np.argmax(logits))
        logits, t = model.forward(np.asarray([nxt]), cache)
        per_token.append(t.compute_seconds)
        att, moe = t.attention_seconds, t.moe_compute_seconds
        windows.append([moe[layer] + att[layer + 1] for layer in range(len(att) - 1)])
    return ComputeProbe(
        tokens=tokens,
        compute_seconds_per_token=statistics.median(per_token),
        layer_window_seconds=[statistics.median(w[i] for w in windows) for i in range(len(windows[0]))],
    )


def store_facts(store_dir: Path, rt: RuntimeConfig, backend: str) -> StoreFacts:
    config = model_config(store_dir)
    record = json.loads((Path(store_dir) / "store.json").read_text())["expert_layout"]["record_size"]
    return StoreFacts(
        name=Path(store_dir).name,
        record_bytes=record,
        num_layers=config.num_layers,
        num_experts=config.num_experts,
        top_k=config.top_k,
        base_ram_bytes=estimate_ram_bytes(
            store_dir, config, rt.max_seq, "cached", backend, 0.0, rt.int8_kernel
        ),
        top_guess_hit_rate=top_guess_hit_rate(store_dir),
    )


def runtime_config_for(d: Decisions, rt: RuntimeConfig, store_dir: Path) -> RuntimeConfig:
    return replace(
        rt,
        expert_cache_gb=d.cache_bytes / 1e9,
        prefetch=d.prefetch != PREFETCH_OFF,
        prefetch_k=d.prefetch_k,
        prefetch_min_probability=d.min_probability,
        prefetch_calibration=calibration_for(store_dir) if d.prefetch == PREFETCH_ADAPTIVE else None,
        prefetch_adaptive=d.prefetch == PREFETCH_ADAPTIVE,
        pin_layer0=d.pin_layer0,
        io_threads=d.io_threads,
    )


def auto_load(
    store_dir: Path, rt: RuntimeConfig, budget_gb: float | None, fast: bool = False
) -> tuple[QwenMoe, dict, Decisions, Path, RuntimeConfig]:
    """(model, load info, decisions, the store actually used, the runtime config they amount to)."""
    t0 = time.perf_counter()
    profile = collect_machine_profile(measure_disk=False)
    if fast:
        store_dir = fast_store_for(store_dir)
    backend = select_backend(profile)
    facts = store_facts(store_dir, rt, backend.name)
    reads = probe_reads(store_dir)
    model, info = load_model(store_dir, "unbuffered", rt.max_seq, budget_gb, backend, rt)
    compute = probe_compute(model)
    decisions = decide(
        profile,
        facts,
        reads,
        compute,
        None if budget_gb is None else round(budget_gb * 1e9),
        fast,
        fast_quality_note() if fast else None,
    )
    model.experts.close()
    model._window[:] = np.nan  # start like a hand-set run: no window measured yet
    chosen = runtime_config_for(decisions, rt, store_dir)
    if decisions.use_cache:
        if budget_gb is not None:
            estimate = facts.base_ram_bytes + decisions.cache_bytes
            enforce_ram_budget(estimate, budget_gb, "--auto with the Manager's cache")
        model.experts, prefetcher, info["cache"] = make_cached_source(store_dir, chosen, model.c)
        info["source"] = "cached"
    else:
        model.experts, prefetcher = UnbufferedExpertSource(store_dir), None
        info["source"] = "unbuffered"
    configure_prefetch(model, chosen, prefetcher)
    info["manager"] = decisions.to_dict()
    info["manager_seconds"] = time.perf_counter() - t0
    return model, info, decisions, store_dir, chosen


@dataclass
class LiveBudget:
    """Applies a memory budget changed while running, between tokens.

    The budget comes from `file` (a number of GB; applied when its text changes)
    or set(). The cache gets what the new budget allows after the rest of
    the process (facts.base_ram_bytes), and never more than the RAM free
    right now plus what the cache already holds, minus the safety margin:
    the same rule as at startup."""

    model: QwenMoe
    facts: StoreFacts
    total_ram_bytes: int
    file: Path | None = None
    events: list[dict] = field(default_factory=list)
    _budget_gb: float | None = None
    _last_text: str | None = None
    _tokens: int = 0

    def set(self, budget_gb: float) -> None:
        self._budget_gb = budget_gb

    def _poll_file(self) -> None:
        """Re-read every time and compare contents: on Windows a file's mtime
        only advances with the ~15.6 ms system tick, so two quick slider
        moves could share one mtime and the second would be missed."""
        if self.file is None:
            return
        try:
            text = self.file.read_text().strip()
        except OSError:  # missing, or being rewritten by the slider right now: next token
            return
        if text == self._last_text:
            return
        self._last_text = text
        try:
            self._budget_gb = float(text)
        except ValueError:
            self.events.append({"token": self._tokens, "error": f"unreadable budget file {self.file}"})

    def __call__(self) -> None:
        self._tokens += 1
        self._poll_file()
        experts = self.model.experts
        if self._budget_gb is None or not isinstance(experts, CachedExpertSource):
            return
        budget, self._budget_gb = self._budget_gb, None
        room = (
            measure_memory().available_bytes
            + experts.capacity_bytes
            - safety_margin_bytes(self.total_ram_bytes)
        )
        slots, why = cache_slots_for(room, round(budget * 1e9) - self.facts.base_ram_bytes, self.facts)
        event = {"token": self._tokens, "budget_gb": budget, "slots_before": experts.num_slots}
        if not slots:  # the cache can't go away mid-run: the smallest one it can run with
            slots, why = experts.min_slots, f"the smallest cache: {why}"
        try:
            event["slots_after"] = experts.resize(slots * self.facts.record_bytes)
            event["reason"] = why
        except CacheTooSmall as e:
            event["slots_after"], event["reason"] = experts.num_slots, f"refused: {e}"
        self.events.append(event)
        print(
            f"[budget] {budget:.2f} GB: cache {event['slots_before']} -> {event['slots_after']} experts",
            flush=True,
        )
