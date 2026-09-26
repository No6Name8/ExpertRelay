"""Expert-cache simulation with predictions: prefetching and
prediction-aware eviction. Offline, over recorded traces; nothing here is
in the runtime yet (docs/limitations.md).

The replay is a list of events in the order the runtime would meet them:

  Access(layer, experts, ...)  the router's picks at one layer for one
                               forward call; each expert is a demand access
                               (hit if cached, else loaded).
  Predict(layer, probs)        a prediction for that layer's NEXT visit,
                               available one layer earlier (Fate-style,
                               arXiv:2502.12224). With prefetch_k > 0, the
                               top-k predicted experts are loaded now.

Policies (capacity in experts, all equal-size):
  lru         evict the least recently touched. A prefetched expert counts
              as touched when it's loaded, and a predicted expert that's
              already cached is touched too (it's about to be used).
  predictive  evict the expert least likely to be picked at its layer's
              next visit: the calibrated prediction for the layer that has
              one, a ReuseModel estimate (gap since last use, recent
              count) for every other layer. Ties go to least recently
              touched. A prefetch only happens if the predicted
              probability beats the eviction victim's.

The current access's own experts, and the experts being prefetched
together, are never evicted to make room for each other; this needs
capacity >= experts per layer + prefetch_k, which simulate() enforces.

With prefetching, a policy can beat Belady's hit rate: Belady is optimal
only among caches that load on demand. The extra reads it takes to do so
(including wasted prefetches) are counted, and they cost time in the
projection (bench/phase35_prediction.py).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from expertrelay.predictor.offline import RECENT_WINDOW, ReuseModel


@dataclass
class Access:
    layer: int
    experts: np.ndarray  # unique expert ids, ascending (load order)
    decode: bool
    positions: np.ndarray  # [n] token positions this call covered (global, increasing)
    position_experts: np.ndarray  # [n, K] experts picked at each of those positions


@dataclass
class Predict:
    layer: int
    probs: np.ndarray  # [E] P(picked at the layer's next visit); ranks the prefetch too


@dataclass
class SimResult:
    """Per Access event, in order."""

    layer: np.ndarray
    decode: np.ndarray
    accesses: np.ndarray
    hits: np.ndarray
    prefetch_reads: np.ndarray  # reads the Predict just before this visit issued
    prefetches_used: int = 0  # prefetched experts that were then accessed before eviction
    prefetches_wasted: int = 0  # prefetched experts evicted without being accessed
    extra: dict = field(default_factory=dict)

    def decode_hit_rate(self) -> float:
        return float(self.hits[self.decode].sum() / max(self.accesses[self.decode].sum(), 1))


class _Cache:
    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        capacity: int,
        policy: str,
        reuse: ReuseModel | None,
    ) -> None:
        self.L, self.E, self.capacity, self.policy, self.reuse = (
            num_layers,
            num_experts,
            capacity,
            policy,
            reuse,
        )
        n = num_layers * num_experts
        self.cached = np.zeros(n, dtype=bool)
        self.n_cached = 0
        self.stamp = np.zeros(n, dtype=np.int64)
        self.clock = 1
        self.prefetched = np.zeros(n, dtype=bool)  # loaded by a prefetch, not accessed since
        self.used, self.wasted = 0, 0
        self.last_used = np.full((num_layers, num_experts), -1, dtype=np.int64)
        self.recent = np.zeros((num_layers, num_experts), dtype=np.int64)
        self.history: list[deque] = [deque(maxlen=RECENT_WINDOW) for _ in range(num_layers)]
        self.next_pos = np.zeros(num_layers, dtype=np.int64)
        self.pred = np.zeros((num_layers, num_experts), dtype=np.float32)
        self.pred_active = np.zeros(num_layers, dtype=bool)

    def touch(self, key: int) -> None:
        self.stamp[key] = self.clock
        self.clock += 1

    def priority(self) -> np.ndarray:
        """Lower = evicted first. Only used by the predictive policy (LRU reads
        the live stamps)."""
        gap = np.where(self.last_used >= 0, self.next_pos[:, None] - self.last_used, -1)
        p = self.reuse.lookup(gap, self.recent)
        p[self.pred_active] = self.pred[self.pred_active]
        return p.reshape(-1)

    def victim(self, protected: np.ndarray, pri: np.ndarray | None) -> int | None:
        cand = np.flatnonzero(self.cached & ~protected)
        if not len(cand):
            return None
        if pri is None:
            return int(cand[np.argmin(self.stamp[cand])])
        pc = pri[cand]
        tied = cand[pc == pc.min()]
        return int(tied[np.argmin(self.stamp[tied])])

    def evict(self, key: int) -> None:
        self.cached[key] = False
        self.n_cached -= 1
        if self.prefetched[key]:
            self.prefetched[key] = False
            self.wasted += 1

    def insert(self, key: int) -> None:
        self.cached[key] = True
        self.n_cached += 1
        self.touch(key)

    def access(self, ev: Access) -> int:
        keys = ev.layer * self.E + ev.experts.astype(np.int64)
        protected = np.zeros_like(self.cached)
        protected[keys] = True
        pri = self.priority() if self.policy == "predictive" else None
        hits = 0
        for key in keys.tolist():
            if self.cached[key]:
                hits += 1
                if self.prefetched[key]:
                    self.prefetched[key] = False
                    self.used += 1
            elif self.capacity > 0:
                if self.n_cached >= self.capacity:
                    self.evict(self.victim(protected, pri))
                self.insert(key)
            self.touch(key)
        layer = ev.layer
        for pos, exps in zip(ev.positions.tolist(), ev.position_experts, strict=True):
            self.last_used[layer, exps] = pos
            self.history[layer].append(exps)
        self.recent[layer] = np.bincount(
            np.concatenate(self.history[layer]).astype(np.int64), minlength=self.E
        )
        self.next_pos[layer] = int(ev.positions[-1]) + 1
        self.pred_active[layer] = False
        return hits

    def predict(self, ev: Predict, prefetch_k: int) -> int:
        if self.policy == "predictive":
            self.pred[ev.layer] = ev.probs
            self.pred_active[ev.layer] = True
        if prefetch_k <= 0 or self.capacity <= 0:
            return 0
        top = np.argsort(-ev.probs, kind="stable")[:prefetch_k]
        keys = ev.layer * self.E + top.astype(np.int64)
        protected = np.zeros_like(self.cached)
        protected[keys] = True
        reads = 0
        if self.policy == "lru":
            # least likely first, so the most likely ends up most recently touched
            for key in keys[::-1].tolist():
                if not self.cached[key]:
                    if self.n_cached >= self.capacity:
                        self.evict(self.victim(protected, None))
                    self.insert(key)
                    self.prefetched[key] = True
                    reads += 1
                self.touch(key)
            return reads
        pri = self.priority()
        for e, key in zip(top.tolist(), keys.tolist(), strict=True):
            if self.cached[key]:
                continue
            if self.n_cached >= self.capacity:
                v = self.victim(protected, pri)
                if v is None or pri[v] >= ev.probs[e]:
                    break  # candidates are in descending probability: none later can win either
                self.evict(v)
            self.insert(key)
            self.prefetched[key] = True
            reads += 1
        return reads


def simulate(
    events: list[Access | Predict],
    *,
    num_layers: int,
    num_experts: int,
    capacity: int,
    policy: str,
    prefetch_k: int = 0,
    reuse: ReuseModel | None = None,
) -> SimResult:
    if policy not in ("lru", "predictive"):
        raise ValueError(f"unknown policy {policy!r}")
    if policy == "predictive" and reuse is None:
        raise ValueError("the predictive policy needs a fitted ReuseModel")
    if 0 < capacity < num_experts + prefetch_k:
        raise ValueError(f"capacity {capacity} < experts per layer + prefetch_k ({num_experts + prefetch_k})")
    cache = _Cache(num_layers, num_experts, capacity, policy, reuse)
    pending = np.zeros(num_layers, dtype=np.int64)
    layer, decode, accesses, hits, pf = [], [], [], [], []
    for ev in events:
        if isinstance(ev, Predict):
            pending[ev.layer] += cache.predict(ev, prefetch_k)
            continue
        layer.append(ev.layer)
        decode.append(ev.decode)
        accesses.append(len(ev.experts))
        hits.append(cache.access(ev))
        pf.append(int(pending[ev.layer]))
        pending[ev.layer] = 0
    return SimResult(
        layer=np.asarray(layer, dtype=np.int64),
        decode=np.asarray(decode, dtype=bool),
        accesses=np.asarray(accesses, dtype=np.int64),
        hits=np.asarray(hits, dtype=np.int64),
        prefetch_reads=np.asarray(pf, dtype=np.int64),
        prefetches_used=cache.used,
        prefetches_wasted=cache.wasted,
    )


def overlapped_decode_seconds(
    result: SimResult, compute_s_per_layer: float, read_s: float
) -> tuple[float, int]:
    """PROJECTED decode time (seconds, decode tokens) from a simulation.

    Per decode layer visit: its compute window, plus the demand misses read
    one at a time before it can run, plus any part of the prefetch for it
    that didn't fit inside the previous layer's compute window. One read at
    a time (queue depth 1, as the runtime reads today), prefetches ahead of
    demand reads in the queue. Assumes the whole previous layer's compute
    is available to hide the prefetch; in the real forward pass the
    prediction is ready only after that layer's attention, so this is
    optimistic about the window (docs/limitations.md).
    """
    d = result.decode
    misses = (result.accesses - result.hits)[d]
    spill = np.maximum(0.0, result.prefetch_reads[d] * read_s - compute_s_per_layer)
    seconds = float(np.sum(compute_s_per_layer + spill + misses * read_s))
    tokens = int(np.sum(d & (result.layer == 0)))
    return seconds, tokens
