"""Offline expert-cache simulator: replay a stream of expert accesses and
count hits at a given cache size, for several eviction policies.

An access stream is a 1-D array of integer keys (layer * num_experts +
expert), in the order the runtime would load them. All experts are the
same size, so capacity is counted in experts. Every access is a hit or a
miss; a miss loads the expert (evicting one if the cache is full). Each
simulate_* function returns one boolean per access (True = hit), so hit
rates can be split by phase or by any other per-access label.

Policies:
  lru       evict the least recently used. Computed for ALL capacities in
            one pass from LRU stack distances: LRU is a stack algorithm, so
            an access hits at capacity C iff fewer than C distinct keys were
            touched since its previous access. R. L. Mattson, J. Gecsei,
            D. R. Slutz, I. L. Traiger, "Evaluation techniques for storage
            hierarchies", IBM Systems Journal 9(2), 1970.
  lfu       evict the key with the fewest accesses so far (counting its whole
            history, including before it was evicted); ties go to the least
            recently used.
  pinned    a fixed "hot set" occupies part of the cache permanently
            (preloaded, so always a hit); the rest is LRU. The hot set must
            be chosen from DIFFERENT data than the stream being replayed, or
            the result is an optimistic cheat.
  belady    evict the key whose next use is farthest in the future. Optimal
            for equal-size items; an upper bound no real cache can reach,
            because it needs the future. L. A. Belady, "A study of replacement
            algorithms for a virtual-storage computer", IBM Systems Journal
            5(2), 1966.
"""

from __future__ import annotations

import heapq
from collections import OrderedDict

import numpy as np

INFINITE = np.iinfo(np.int64).max


def lru_stack_distances(stream: np.ndarray, num_keys: int) -> np.ndarray:
    """For each access, how many DISTINCT other keys were accessed since the
    previous access to the same key (INFINITE for a first access). An access
    is an LRU hit at capacity C exactly when its distance is < C."""
    last = np.full(num_keys, -1, dtype=np.int64)
    out = np.empty(len(stream), dtype=np.int64)
    for i, key in enumerate(stream):
        prev = last[key]
        out[i] = INFINITE if prev < 0 else np.count_nonzero(last > prev)
        last[key] = i
    return out


def lru_hit_counts(distances: np.ndarray, capacities: list[int]) -> np.ndarray:
    """Hit count at each capacity, from lru_stack_distances (pass a subset
    of the distances, e.g. decode accesses only, to count just those)."""
    finite = np.sort(distances[distances != INFINITE])
    return np.searchsorted(finite, np.asarray(capacities), side="left")


def simulate_lru(stream: np.ndarray, capacity: int) -> np.ndarray:
    """Direct LRU simulation, used by the pinned policy and as the reference
    the stack-distance method is tested against."""
    cache: OrderedDict[int, None] = OrderedDict()
    hits = np.zeros(len(stream), dtype=bool)
    for i, key in enumerate(stream.tolist()):
        if key in cache:
            hits[i] = True
            cache.move_to_end(key)
        elif capacity > 0:
            if len(cache) >= capacity:
                cache.popitem(last=False)
            cache[key] = None
    return hits


def simulate_lfu(stream: np.ndarray, capacity: int) -> np.ndarray:
    hits = np.zeros(len(stream), dtype=bool)
    if capacity <= 0:
        return hits
    freq: dict[int, int] = {}
    last_use: dict[int, int] = {}
    cached: set[int] = set()
    heap: list[tuple[int, int, int]] = []  # (freq, last_use, key), lazily invalidated
    for i, key in enumerate(stream.tolist()):
        freq[key] = freq.get(key, 0) + 1
        last_use[key] = i
        if key in cached:
            hits[i] = True
        else:
            if len(cached) >= capacity:
                while True:
                    f, t, victim = heapq.heappop(heap)
                    if victim in cached and freq[victim] == f and last_use[victim] == t:
                        cached.remove(victim)
                        break
            cached.add(key)
        heapq.heappush(heap, (freq[key], i, key))
    return hits


def simulate_pinned(
    stream: np.ndarray, capacity: int, hot_keys: list[int], pinned_fraction: float
) -> np.ndarray:
    """Pin the first `round(capacity * pinned_fraction)` of `hot_keys` (hottest
    first); the rest of the capacity is LRU over the unpinned keys."""
    n_pinned = min(len(hot_keys), round(capacity * pinned_fraction))
    is_pinned = np.isin(stream, np.asarray(hot_keys[:n_pinned], dtype=stream.dtype))
    hits = is_pinned.copy()
    hits[~is_pinned] = simulate_lru(stream[~is_pinned], capacity - n_pinned)
    return hits


def next_use_indices(stream: np.ndarray) -> np.ndarray:
    """For each access, the index of the next access to the same key (INFINITE if none)."""
    nxt = np.full(len(stream), INFINITE, dtype=np.int64)
    seen: dict[int, int] = {}
    for i in range(len(stream) - 1, -1, -1):
        key = int(stream[i])
        nxt[i] = seen.get(key, INFINITE)
        seen[key] = i
    return nxt


def simulate_belady(stream: np.ndarray, capacity: int, next_use: np.ndarray | None = None) -> np.ndarray:
    hits = np.zeros(len(stream), dtype=bool)
    if capacity <= 0:
        return hits
    if next_use is None:
        next_use = next_use_indices(stream)
    cached: dict[int, int] = {}  # key -> its next use
    heap: list[tuple[int, int]] = []  # (-next_use, key), lazily invalidated
    for i, key in enumerate(stream.tolist()):
        nu = int(next_use[i])
        if key in cached:
            hits[i] = True
        elif len(cached) >= capacity:
            while True:
                neg, victim = heapq.heappop(heap)
                if cached.get(victim) == -neg:
                    del cached[victim]
                    break
        cached[key] = nu
        heapq.heappush(heap, (-nu, key))
    return hits
