"""Cache simulator (cache.simulator) on hand-made sequences with known answers,
plus property checks against a brute-force reference."""

from __future__ import annotations

import numpy as np
import pytest

from expertrelay.cache.simulator import (
    INFINITE,
    lru_hit_counts,
    lru_stack_distances,
    next_use_indices,
    simulate_belady,
    simulate_lfu,
    simulate_lru,
    simulate_pinned,
)

# The classic page-replacement example (Silberschatz, Galvin & Gagne,
# "Operating System Concepts", Ch. 10): 20 references, 3 frames.
# Published answers: OPT 9 faults, LRU 12 faults (so 11 and 8 hits).
TEXTBOOK = np.array([7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1])


def test_belady_textbook_answer():
    assert simulate_belady(TEXTBOOK, 3).sum() == 20 - 9


def test_lru_textbook_answer_both_methods():
    assert simulate_lru(TEXTBOOK, 3).sum() == 20 - 12
    assert lru_hit_counts(lru_stack_distances(TEXTBOOK, 8), [3])[0] == 20 - 12


def test_stack_distances_by_hand():
    # a b c a b b d a
    stream = np.array([0, 1, 2, 0, 1, 1, 3, 0])
    expected = [INFINITE, INFINITE, INFINITE, 2, 2, 0, INFINITE, 2]
    assert lru_stack_distances(stream, 4).tolist() == expected


def test_lfu_by_hand():
    # capacity 2: when c arrives, a has been used twice and b once, so b is evicted
    stream = np.array([0, 0, 1, 2, 0, 1])
    # a miss, a hit, b miss, c miss (evict b: freq1 vs a freq2), a hit, b miss
    assert simulate_lfu(stream, 2).tolist() == [False, True, False, False, True, False]


def test_lfu_tie_breaks_on_least_recent():
    # capacity 2, a b c: a and b both freq 1, a older -> evict a; then a misses, b hits
    stream = np.array([0, 1, 2, 1, 0])
    assert simulate_lfu(stream, 2).tolist() == [False, False, False, True, False]


def test_pinned_keys_always_hit():
    stream = np.array([5, 1, 2, 3, 5, 4, 5])
    # capacity 2, half pinned -> key 5 pinned; the other slot is LRU over 1,2,3,4 (no reuse)
    assert simulate_pinned(stream, 2, hot_keys=[5, 1], pinned_fraction=0.5).tolist() == [
        True,
        False,
        False,
        False,
        True,
        False,
        True,
    ]


@pytest.mark.parametrize("policy", [simulate_lru, simulate_lfu, simulate_belady])
def test_edge_capacities(policy):
    stream = np.array([1, 2, 1, 3, 2, 1])
    assert policy(stream, 0).sum() == 0
    assert policy(stream, 10).sum() == len(stream) - 3  # only the 3 first-time loads miss


def test_properties_on_random_streams():
    rng = np.random.default_rng(0)
    for trial in range(20):
        num_keys = int(rng.integers(3, 30))
        # skewed popularity, so caching actually matters
        p = rng.dirichlet(np.full(num_keys, 0.3))
        stream = rng.choice(num_keys, size=400, p=p)
        caps = [0, 1, 2, 5, 10, num_keys]
        fast = lru_hit_counts(lru_stack_distances(stream, num_keys), caps)
        nu = next_use_indices(stream)
        for cap, h in zip(caps, fast, strict=True):
            lru = simulate_lru(stream, cap).sum()
            assert h == lru, f"trial {trial}: stack-distance LRU != direct LRU at C={cap}"
            best = simulate_belady(stream, cap, nu).sum()
            assert best >= lru and best >= simulate_lfu(stream, cap).sum(), (
                f"trial {trial}: Belady not optimal at C={cap}"
            )
        # LRU is a stack algorithm: more capacity never hurts
        assert list(fast) == sorted(fast)
