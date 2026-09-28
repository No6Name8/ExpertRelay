"""The Phase 4 benchmark's bookkeeping: metrics from per-step records and
the token comparison."""

from __future__ import annotations

import pytest

from expertrelay.bench import phase4_benchmark
from expertrelay.bench.phase4_benchmark import agreement, cache_metrics


def step(**kw) -> dict:
    s = {
        "expert_loads": 4,
        "cache_hits": 0,
        "prefetch_hits": 0,
        "prefetch_waits": 0,
        "demand_reads": 0,
        "prefetches_issued": 0,
        "prefetch_reads": 0,
        "prefetches_wasted": 0,
        "prefetches_cancelled": 0,
        "prefetches_skipped_low_confidence": 0,
        "attention_seconds": [0.01, 0.02],
        "moe_compute_seconds": [0.03, 0.04],
        "read_wait_seconds": [0.0, 0.005],
    }
    s.update(kw)
    return s


def test_cache_metrics_count_decode_steps_only():
    run = {
        "prompts": [
            {
                "steps": [
                    step(expert_loads=40, demand_reads=40),  # prefill: ignored
                    step(
                        cache_hits=1, prefetch_hits=2, demand_reads=1, prefetch_reads=3, prefetches_wasted=1
                    ),
                    step(cache_hits=3, prefetch_waits=1, prefetch_reads=2),
                ]
            }
        ]
    }
    m = cache_metrics(run)
    assert m["decode_hit_rate"] == pytest.approx(6 / 8)
    assert m["decode_prefetch_hit_rate"] == pytest.approx(2 / 8)
    assert m["decode_prefetch_wait_rate"] == pytest.approx(1 / 8)
    assert m["decode_demand_reads_per_token"] == pytest.approx(0.5)
    assert m["decode_prefetch_reads_per_token"] == pytest.approx(2.5)
    assert m["decode_prefetches_wasted_per_token"] == pytest.approx(0.5)
    assert m["per_layer_ms"]["read_wait"] == pytest.approx([0.0, 5.0])


def test_agreement_against_today_and_phase2(monkeypatch):
    def result(sid: str, rnd: int, ids: list[list[int]]) -> dict:
        return {
            "id": sid,
            "round": rnd,
            "status": "ok",
            "run": {"prompts": [{"generated_ids": x} for x in ids]},
        }

    monkeypatch.setattr(phase4_benchmark, "phase2_c_tokens", lambda: [[1, 2], [3]])
    out = agreement(
        [
            result("C", 0, [[1, 2], [3]]),
            result("pf8", 0, [[1, 2], [3]]),
            result("pf4", 0, [[1, 2], [4]]),
            {"id": "cache", "round": 0, "status": "failed"},
        ]
    )
    assert out["pf8#0"] == {"same_as_todays_C": True, "same_as_phase2_C": True}
    assert out["pf4#0"] == {"same_as_todays_C": False, "same_as_phase2_C": False}
    assert "cache#0" not in out
