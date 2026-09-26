"""Prediction-aware cache simulation on hand-made event streams."""

from __future__ import annotations

import numpy as np
import pytest

from expertrelay.cache.predictive import Access, Predict, SimResult, overlapped_decode_seconds, simulate
from expertrelay.cache.simulator import simulate_lru
from expertrelay.predictor.offline import ReuseModel


def acc(layer: int, experts: list[int], pos: int, decode: bool = True) -> Access:
    ex = np.array(experts)
    return Access(layer, np.unique(ex), decode, np.array([pos]), ex[None, :])


def alternating_reuse() -> ReuseModel:
    """Fitted on an expert that's used every other position: never reused at gap 1."""
    used = np.zeros((40, 2), dtype=bool)
    used[::2, 0] = True
    used[1::2, 1] = True
    rm = ReuseModel(min_count=1)
    rm.add(used, first_target=0)
    return rm.fit()


def test_lru_without_prefetch_matches_the_plain_simulator():
    rng = np.random.default_rng(0)
    num_layers, num_experts = 3, 4
    keys = rng.integers(0, num_layers * num_experts, size=400)
    events = [acc(int(k) // num_experts, [int(k) % num_experts], i) for i, k in enumerate(keys)]
    for cap in (4, 5, 7, 12):
        res = simulate(events, num_layers=num_layers, num_experts=num_experts, capacity=cap, policy="lru")
        assert res.hits.tolist() == simulate_lru(keys, cap).astype(int).tolist()


@pytest.mark.parametrize("policy", ["lru", "predictive"])
def test_perfect_prediction_with_prefetch_always_hits(policy):
    rng = np.random.default_rng(1)
    num_layers, num_experts = 4, 6
    events = []
    for pos in range(30):
        for layer in range(num_layers):
            e = int(rng.integers(num_experts))
            probs = np.full(num_experts, 0.01, dtype=np.float32)
            probs[e] = 0.99
            events += [Predict(layer, probs), acc(layer, [e], pos)]
    res = simulate(
        events,
        num_layers=num_layers,
        num_experts=num_experts,
        capacity=num_experts + 1,
        policy=policy,
        prefetch_k=1,
        reuse=alternating_reuse(),
    )
    assert res.decode_hit_rate() == 1.0
    assert res.prefetches_wasted == 0
    assert res.prefetches_used == res.prefetch_reads.sum()


def test_predictive_eviction_keeps_the_expert_about_to_be_used():
    # Layer 2's expert 0 is the least recently used, but predicted for layer 2's next visit.
    events = [
        acc(2, [0], 0),
        acc(0, [0], 0),
        acc(1, [0], 0),
        Predict(2, np.array([0.99, 0.01], dtype=np.float32)),
        acc(0, [1], 1),  # a miss with a full cache: something must go
        acc(2, [0], 1),
    ]
    kw = {"num_layers": 3, "num_experts": 2, "capacity": 3, "reuse": alternating_reuse()}
    lru = simulate(events, policy="lru", **kw)
    pred = simulate(events, policy="predictive", **kw)
    assert lru.hits.tolist() == [0, 0, 0, 0, 0]
    assert pred.hits.tolist() == [0, 0, 0, 0, 1]


@pytest.mark.parametrize(("cached_prob", "prefetched"), [(0.6, 0), (0.3, 1)])
def test_predictive_prefetch_only_when_it_beats_the_victim(cached_prob, prefetched):
    # A reuse table that rates everything 1.0, so the only evictable keys worth
    # less are layer 0's cached experts, at `cached_prob`. Expert 2 is predicted
    # at 0.5: it's prefetched only if that beats them.
    always = ReuseModel(min_count=1)
    always.add(np.ones((20, 2), dtype=bool), first_target=0)
    always.fit()
    events = [acc(layer, [e], e) for e in (0, 1) for layer in range(3)]
    events += [
        Predict(0, np.array([cached_prob, cached_prob, 0.5], dtype=np.float32)),
        acc(0, [2], 2),
    ]
    res = simulate(
        events, num_layers=3, num_experts=3, capacity=6, policy="predictive", prefetch_k=1, reuse=always
    )
    assert res.prefetch_reads[-1] == prefetched
    assert res.hits[-1] == prefetched


def test_capacity_must_fit_a_layer_plus_the_prefetch():
    with pytest.raises(ValueError):
        simulate([], num_layers=2, num_experts=4, capacity=5, policy="lru", prefetch_k=2)
    with pytest.raises(ValueError):
        simulate([], num_layers=2, num_experts=4, capacity=8, policy="predictive")


def test_overlapped_decode_seconds_by_hand():
    res = SimResult(
        layer=np.array([0, 1, 0, 1]),
        decode=np.array([False, True, True, True]),
        accesses=np.array([10, 4, 4, 4]),
        hits=np.array([0, 4, 2, 3]),
        prefetch_reads=np.array([0, 3, 1, 0]),
    )
    # window 1.0 s, read 0.5 s. Decode visits:
    #   layer 1: prefetch 3 reads = 1.5 s -> 0.5 s spill; 0 misses -> 1.5 s
    #   layer 0: prefetch 0.5 s fits; 2 misses -> 1.0 + 1.0 = 2.0 s
    #   layer 1: no prefetch; 1 miss -> 1.5 s
    seconds, tokens = overlapped_decode_seconds(res, compute_s_per_layer=1.0, read_s=0.5)
    assert seconds == pytest.approx(5.0)
    assert tokens == 1
