"""The Manager's decisions on made-up machines (manager.policy): no hardware
is touched. The numbers are shaped like the dev machine's (8.67 MB int8
records, ~4 ms per read) and then pushed to the regimes each rule is for."""

from __future__ import annotations

from dataclasses import replace

import pytest

from expertrelay.manager.policy import (
    PREFETCH_ADAPTIVE,
    PREFETCH_OFF,
    PREFETCH_TOP_K,
    ComputeProbe,
    StoreFacts,
    decide,
    safety_margin_bytes,
)
from expertrelay.manager.probe import ReadProbe
from expertrelay.manager.profile import MemoryInfo

GB = 10**9
STORE = StoreFacts(
    name="test-int8",
    record_bytes=8_671_232,
    num_layers=24,
    num_experts=60,
    top_k=4,
    base_ram_bytes=2 * GB,
    top_guess_hit_rate=0.94,
)
FAST_DISK = ReadProbe(reads_per_depth=24, seconds_per_read_1=0.004, seconds_per_read_2=0.003, record_bytes=1)
SLOW_DISK = ReadProbe(reads_per_depth=24, seconds_per_read_1=0.040, seconds_per_read_2=0.039, record_bytes=1)


def compute(seconds_per_token: float) -> ComputeProbe:
    per_layer = seconds_per_token / 24
    return ComputeProbe(
        tokens=4, compute_seconds_per_token=seconds_per_token, layer_window_seconds=[per_layer] * 23
    )


SLOW_COMPUTE, FAST_COMPUTE = compute(1.0), compute(0.09)


def machine(profile, free_gb: float, total_gb: float = 8.0):
    return replace(
        profile, memory=MemoryInfo(total_bytes=int(total_gb * GB), available_bytes=int(free_gb * GB))
    )


def test_slow_compute_fast_disk_prefetches_every_guess(fake_profile):
    d = decide(machine(fake_profile, 4.0), STORE, FAST_DISK, SLOW_COMPUTE)
    assert d.prefetch == PREFETCH_TOP_K and d.min_probability == 0.0
    assert d.io_threads == 2  # 4 -> 3 ms per read is a 1.33x gain
    assert "fits" in d.reasons["prefetch"]


def test_fast_compute_fast_disk_is_adaptive(fake_profile):
    d = decide(machine(fake_profile, 4.0), STORE, FAST_DISK, FAST_COMPUTE)
    assert d.prefetch == PREFETCH_ADAPTIVE and d.min_probability == 0.05


def test_fast_compute_slow_disk_is_cache_only_or_adaptive(fake_profile):
    d = decide(machine(fake_profile, 4.0), STORE, SLOW_DISK, FAST_COMPUTE)
    assert d.prefetch in (PREFETCH_OFF, PREFETCH_ADAPTIVE)
    # 3.75 ms window / 40 ms per read = 0.094 reads: saves 0.94 x 0.094 = 0.088 > costs 0.06
    assert d.prefetch == PREFETCH_ADAPTIVE
    assert d.io_threads == 1  # 2 in flight gained only 1.03x
    assert d.use_cache


def test_very_slow_disk_turns_prefetch_off(fake_profile):
    hdd = ReadProbe(reads_per_depth=24, seconds_per_read_1=0.100, seconds_per_read_2=0.100, record_bytes=1)
    d = decide(machine(fake_profile, 4.0), STORE, hdd, FAST_COMPUTE)
    # 3.75 ms / 100 ms = 0.0375 reads: saves 0.035 < costs 0.06
    assert d.prefetch == PREFETCH_OFF and d.use_cache
    assert "cost more" in d.reasons["prefetch"]


def test_without_calibration_a_whole_read_must_fit(fake_profile):
    store = replace(STORE, top_guess_hit_rate=None)
    short = decide(machine(fake_profile, 4.0), store, FAST_DISK, compute(0.06))  # 2.5 ms window, 3 ms reads
    assert short.prefetch == PREFETCH_OFF


def test_no_calibration_means_no_threshold(fake_profile):
    d = decide(
        machine(fake_profile, 4.0), replace(STORE, top_guess_hit_rate=None), FAST_DISK, compute(0.15)
    )  # 6.25 ms window: 2 reads fit
    assert d.prefetch == PREFETCH_ADAPTIVE and d.min_probability == 0.0


def test_low_ram_shrinks_the_cache_then_drops_it(fake_profile):
    roomy = decide(machine(fake_profile, 4.0), STORE, FAST_DISK, FAST_COMPUTE)
    tight = decide(machine(fake_profile, 3.0), STORE, FAST_DISK, FAST_COMPUTE)
    none = decide(machine(fake_profile, 2.6), STORE, FAST_DISK, FAST_COMPUTE)
    assert roomy.cache_slots > tight.cache_slots > 0
    margin = safety_margin_bytes(8 * GB)
    assert tight.cache_slots == (3 * GB - margin - STORE.base_ram_bytes) // STORE.record_bytes
    assert tight.cache_bytes + STORE.base_ram_bytes + margin <= 3 * GB  # the OS keeps its margin
    assert none.cache_slots == 0 and not none.use_cache and none.prefetch == PREFETCH_OFF
    assert "fewer than" in none.reasons["cache"]


def test_memory_budget_caps_the_cache(fake_profile):
    d = decide(machine(fake_profile, 6.0), STORE, FAST_DISK, FAST_COMPUTE, budget_bytes=3 * GB)
    assert d.cache_slots == (3 * GB - STORE.base_ram_bytes) // STORE.record_bytes
    assert "memory budget" in d.reasons["cache"]


def test_cache_never_exceeds_the_whole_store(fake_profile):
    d = decide(machine(fake_profile, 64.0, total_gb=64.0), STORE, FAST_DISK, FAST_COMPUTE)
    assert d.cache_slots == 24 * 60


def test_layer0_pinned_only_when_it_is_a_small_share(fake_profile):
    small = decide(machine(fake_profile, 4.0), STORE, FAST_DISK, FAST_COMPUTE)
    big = decide(machine(fake_profile, 8.0, total_gb=16.0), STORE, FAST_DISK, FAST_COMPUTE)
    assert not small.pin_layer0  # 60 of ~170 slots
    assert big.cache_slots >= 4 * 60 and big.pin_layer0


def test_ascend_profile_chooses_mindspore(fake_profile):
    cpu = decide(machine(fake_profile, 4.0), STORE, FAST_DISK, FAST_COMPUTE)
    npu = decide(replace(machine(fake_profile, 4.0), accelerators=["ascend"]), STORE, FAST_DISK, FAST_COMPUTE)
    assert cpu.backend.name == "numpy" and npu.backend.name == "mindspore"


def test_precision_is_int8_unless_fast(fake_profile):
    d = decide(machine(fake_profile, 4.0), STORE, FAST_DISK, FAST_COMPUTE)
    f = decide(
        machine(fake_profile, 4.0), STORE, FAST_DISK, FAST_COMPUTE, fast=True, fast_quality="91.5% FAIL"
    )
    assert d.precision == "int8" and f.precision == "int4_gptq"
    assert "91.5% FAIL" in f.reasons["precision"]


@pytest.mark.parametrize("free_gb", [2.6, 4.0])
def test_every_decision_has_a_reason_and_the_record_is_plain_data(fake_profile, free_gb):
    import json

    d = decide(machine(fake_profile, free_gb), STORE, FAST_DISK, SLOW_COMPUTE)
    assert set(d.reasons) == {"cache", "io_threads", "prefetch", "pin_layer0", "backend", "precision"}
    assert all(d.reasons.values())
    json.dumps(d.to_dict())
    assert d.summary().startswith("Manager decisions:")
