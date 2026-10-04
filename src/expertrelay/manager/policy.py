"""The Manager's decisions: from the machine profile and the startup probe
to every runtime setting, each with the reason it was made.

Pure functions of plain numbers, so they are tested with made-up machines
(tests/test_manager_policy.py). The rules, and the measurements behind
their thresholds:

cache size   all the RAM that is free at startup, minus what the rest of
             the process needs (resident weights, KV cache, scratch:
             runtime.generate.estimate_ram_bytes) and a safety margin of
             max(512 MiB, 5% of total RAM) left to the OS so it never
             pages; capped by the memory budget if one is given, and by
             the whole store. Below the minimum the cache can run with
             (one layer's experts + one prefetch batch + 1, plus layer 0
             if pinned), no cache: one read per expert use. The Phase 4+5
             to B3 benchmarks ran with ~0.4 GB left free and never paged.
I/O threads  2 if two reads in flight gave >= 10% more throughput than one
             in the probe (read_diagnosis measured +25% on the dev
             machine's NVMe), else 1.
prefetch     reads_fit = the median per-layer window (this layer's MoE
             compute + the next layer's attention, the time a prefetch
             issued at this layer has before the next layer needs its
             experts) / the time per read with the chosen I/O threads.
               reads_fit >= 8 (a whole top-8 guess fits): fixed top-8,
                 every guess can be read in time;
               otherwise, break-even of the single most confident guess:
                 it saves p x min(reads_fit, 1) reads of waiting (p = how
                 often the top guess is used, from the store's prefetch
                 calibration: 0.94 for the int8 Chat store) and costs at
                 most (1 - p) x 1 read when wrong (the disk was busy with
                 it). If saving >= cost: adaptive, reads per layer capped at
                 what fits, least confident guesses dropped (threshold
                 0.05 on the calibrated probability); else off. Without a
                 calibration p = 0.5, i.e. a whole read must fit.
             A prefetch that can't finish before its layer still helps:
             the layer waits only for the rest of the read. This rule
             replaced a first "reads_fit >= 1" cutoff before any
             validation run, after the first --auto smoke run measured
             windows of ~0.8 reads on the dev machine: the regime where
             Step B3 measured adaptive prefetch beating cache only.
pin layer 0  layer 0 has no Fate guess (no previous layer), so it can't be
             prefetched; pinning all its experts makes it always hit, at
             the cost of num_experts slots for the other 23 layers. Pinned
             only when that is at most 25% of the cache.
backend      manager.backend_selection (Ascend -> MindSpore, else numpy).
precision    int8, always, unless --fast asks for the GPTQ int4 experts;
             the reason then carries their measured quality cost.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field

from expertrelay.manager.backend_selection import BackendChoice, select_backend
from expertrelay.manager.probe import ReadProbe
from expertrelay.manager.profile import MachineProfile

MIN_MARGIN_BYTES = 512 * 2**20
MARGIN_SHARE_OF_TOTAL = 0.05
TWO_THREADS_MIN_GAIN = 1.10
PREFETCH_K = 8
ADAPTIVE_MIN_PROBABILITY = 0.05
PIN_MAX_SHARE = 0.25
UNCALIBRATED_TOP_GUESS_HIT_RATE = 0.5
PREFETCH_OFF, PREFETCH_ADAPTIVE, PREFETCH_TOP_K = "off", "adaptive", "top_k"


@dataclass(frozen=True)
class ComputeProbe:
    """Measured by the runtime on a few decode tokens, with no cache and no
    prefetch, so read waits are excluded from every number."""

    tokens: int
    compute_seconds_per_token: float
    layer_window_seconds: list[float]  # per layer L < last: MoE compute of L + attention of L+1


@dataclass(frozen=True)
class StoreFacts:
    name: str
    record_bytes: int
    num_layers: int
    num_experts: int
    top_k: int
    base_ram_bytes: int  # the process without any expert cache
    top_guess_hit_rate: float | None  # P(rank-1 guess is used), from the calibration; None: no calibration

    @property
    def has_calibration(self) -> bool:
        return self.top_guess_hit_rate is not None


@dataclass(frozen=True)
class Decisions:
    cache_bytes: int
    cache_slots: int
    io_threads: int
    prefetch: str
    prefetch_k: int
    min_probability: float
    pin_layer0: bool
    backend: BackendChoice
    precision: str
    reasons: dict[str, str]
    inputs: dict = field(default_factory=dict)

    @property
    def use_cache(self) -> bool:
        return self.cache_slots > 0

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        lines = ["Manager decisions:"]
        lines += [f"  {k:<10} {v}" for k, v in self.reasons.items()]
        return "\n".join(lines)


def safety_margin_bytes(total_ram_bytes: int) -> int:
    return max(MIN_MARGIN_BYTES, int(MARGIN_SHARE_OF_TOTAL * total_ram_bytes))


def min_cache_slots(top_k: int, num_experts: int, pin_layer0: bool) -> int:
    """As runtime.generate.RuntimeConfig.cache_slots_needed."""
    return (num_experts if pin_layer0 else 0) + top_k + PREFETCH_K + 1


def cache_slots_for(room_bytes: int, budget_room_bytes: int | None, store: StoreFacts) -> tuple[int, str]:
    """Whole slots in `room_bytes` (and in the budget's room, if any), at
    most the whole store; 0 when below the minimum the cache needs."""
    room = room_bytes if budget_room_bytes is None else min(room_bytes, budget_room_bytes)
    limit = (
        "free RAM" if budget_room_bytes is None or room_bytes <= budget_room_bytes else "the memory budget"
    )
    slots = min(max(room, 0) // store.record_bytes, store.num_layers * store.num_experts)
    need = min_cache_slots(store.top_k, store.num_experts, pin_layer0=False)
    if slots < need:
        return 0, f"{limit} leaves room for {slots} experts, fewer than the {need} a cache needs"
    return int(slots), f"limited by {limit}"


def decide(
    profile: MachineProfile,
    store: StoreFacts,
    reads: ReadProbe,
    compute: ComputeProbe,
    budget_bytes: int | None = None,
    fast: bool = False,
    fast_quality: str | None = None,
) -> Decisions:
    reasons: dict[str, str] = {}
    gb = 1e9

    # cache size
    margin = safety_margin_bytes(profile.memory.total_bytes)
    room = profile.memory.available_bytes - margin - store.base_ram_bytes
    budget_room = None if budget_bytes is None else budget_bytes - store.base_ram_bytes
    slots, why = cache_slots_for(room, budget_room, store)
    reasons["cache"] = (
        f"{slots} experts ({slots * store.record_bytes / gb:.2f} GB), {why}: "
        f"{profile.memory.available_bytes / gb:.2f} GB free - {store.base_ram_bytes / gb:.2f} GB for the rest "
        f"of the process - {margin / gb:.2f} GB safety margin"
        + (f"; budget {budget_bytes / gb:.2f} GB" if budget_bytes is not None else "")
    )

    # I/O threads
    gain = reads.two_in_flight_gain
    cores = profile.cpu.logical_cores or 1
    io_threads = 2 if slots and gain >= TWO_THREADS_MIN_GAIN and cores >= 2 else 1
    reasons["io_threads"] = (
        f"{io_threads}: 2 reads in flight gave {gain:.2f}x the throughput of 1 "
        f"({reads.seconds_per_read_1 * 1e3:.1f} -> {reads.seconds_per_read_2 * 1e3:.1f} ms per read); "
        f"2 threads need >= {TWO_THREADS_MIN_GAIN:.2f}x"
        + ("" if slots else " (no cache: reads happen one at a time in the forward pass)")
    )

    # prefetch
    per_read = reads.seconds_per_read_2 if io_threads == 2 else reads.seconds_per_read_1
    window = statistics.median(compute.layer_window_seconds)
    reads_fit = window / per_read
    measured = (
        f"per-layer window {window * 1e3:.1f} ms (compute {compute.compute_seconds_per_token:.3f} s/token) / "
        f"{per_read * 1e3:.1f} ms per read = {reads_fit:.1f} reads fit"
    )
    min_probability = 0.0
    p = store.top_guess_hit_rate if store.has_calibration else UNCALIBRATED_TOP_GUESS_HIT_RATE
    saving, cost = p * min(reads_fit, 1.0), 1.0 - p
    break_even = (
        f"top guess used {p:.0%}: saves {saving:.2f} reads of waiting vs costs <= {cost:.2f} when wrong"
    )
    if not slots:
        prefetch, why = PREFETCH_OFF, "no cache to prefetch into"
    elif reads_fit >= PREFETCH_K:
        prefetch, why = PREFETCH_TOP_K, f"a whole top-{PREFETCH_K} guess fits, read every guess"
    elif saving >= cost:  # a tie prefetches: without a calibration, one whole read fitting is enough
        prefetch = PREFETCH_ADAPTIVE
        why = f"fewer than {PREFETCH_K} fit, read only the most confident guesses that fit ({break_even})"
        if store.has_calibration:
            min_probability = ADAPTIVE_MIN_PROBABILITY
            why += f"; skip guesses below {min_probability} calibrated probability"
        else:
            why += "; no calibration for this store, so no confidence threshold"
    else:
        prefetch, why = PREFETCH_OFF, f"a wrong guess would cost more than a right one saves ({break_even})"
    reasons["prefetch"] = f"{prefetch}: {why} ({measured})"

    # pin layer 0
    pin = bool(slots) and store.num_experts <= PIN_MAX_SHARE * slots
    reasons["pin_layer0"] = (
        f"{'yes' if pin else 'no'}: layer 0's {store.num_experts} experts would take "
        f"{store.num_experts / slots:.0%} of the cache (pinned only at <= {PIN_MAX_SHARE:.0%})"
        if slots
        else "no: no cache"
    )

    backend = select_backend(profile)
    reasons["backend"] = f"{backend.name}: {backend.reason}"
    precision = "int4_gptq" if fast else "int8"
    reasons["precision"] = (
        f"int4 GPTQ experts (--fast). Measured quality cost: {fast_quality}"
        if fast
        else "int8 (the default; int4 GPTQ only with --fast)"
    )

    return Decisions(
        cache_bytes=slots * store.record_bytes,
        cache_slots=slots,
        io_threads=io_threads,
        prefetch=prefetch,
        prefetch_k=PREFETCH_K,
        min_probability=min_probability,
        pin_layer0=pin,
        backend=backend,
        precision=precision,
        reasons=reasons,
        inputs={
            "free_ram_bytes": profile.memory.available_bytes,
            "total_ram_bytes": profile.memory.total_bytes,
            "logical_cores": profile.cpu.logical_cores,
            "accelerators": profile.accelerators,
            "safety_margin_bytes": margin,
            "budget_bytes": budget_bytes,
            "store": asdict(store),
            "read_probe": asdict(reads),
            "compute_probe": asdict(compute),
            "reads_fit_per_layer": reads_fit,
        },
    )
