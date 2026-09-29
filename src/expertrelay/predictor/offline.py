"""Offline expert predictors, fitted and scored on recorded traces only.

Nothing here runs inside the runtime: these are the Phase 3.5 building
blocks for deciding WHAT a prefetcher and a prediction-aware cache should
use, before any of it is built into the forward pass
(bench/phase35_prediction.py drives them; docs/limitations.md says what that
study can and can't show). Everything is plain numpy, per CLAUDE.md's
backend rule.

Shapes: E = experts per layer, K = experts the router picks (top-k). A
"used" matrix is [T, E] bool: which experts one layer picked at each of T
consecutive token positions of one sequence.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Scores and recall
# ---------------------------------------------------------------------------


def used_matrix(experts: np.ndarray, num_experts: int) -> np.ndarray:
    """[T, K] expert ids -> [T, E] bool."""
    out = np.zeros((len(experts), num_experts), dtype=bool)
    np.put_along_axis(out, experts.astype(np.int64), True, axis=1)
    return out


def recency_scores(used: np.ndarray, window: int, decay: float) -> np.ndarray:
    """r[t, e] = sum over j = 1..window of decay**(j-1) * used[t-j, e].

    Only positions BEFORE t count: at prediction time the current token's
    routing is exactly what's unknown.
    """
    out = np.zeros(used.shape, dtype=np.float32)
    for j in range(1, min(window, len(used) - 1) + 1):
        out[j:] += np.float32(decay ** (j - 1)) * used[:-j]
    return out


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """[N, E] -> [N, k] indices of the k highest scores (unordered). Ties go to
    the lower expert id, so results are deterministic."""
    order = np.argsort(-scores, axis=1, kind="stable")
    return order[:, :k]


def topk_hits(scores: np.ndarray, truth: np.ndarray, k: int) -> np.ndarray:
    """How many of each row's true experts ([N, K]) the top-k of `scores` catches."""
    guess = top_k(scores, k)
    return (truth[:, :, None] == guess[:, None, :]).any(axis=2).sum(axis=1)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def pool_adjacent_violators(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted least-squares nondecreasing fit to y (already ordered by the
    score). M. Ayer, H. D. Brunk, G. M. Ewing, W. T. Reid, E. Silverman, "An
    empirical distribution function for sampling with incomplete
    information", Annals of Mathematical Statistics 26(4), 1955."""
    means: list[float] = []
    weights: list[float] = []
    sizes: list[int] = []
    for yi, wi in zip(y.tolist(), w.tolist(), strict=True):
        means.append(yi)
        weights.append(wi)
        sizes.append(1)
        while len(means) > 1 and means[-2] > means[-1]:
            wt = weights[-2] + weights[-1]
            m = (means[-2] * weights[-2] + means[-1] * weights[-1]) / wt if wt > 0 else means[-1]
            size = sizes[-2] + sizes[-1]
            del means[-1], weights[-1], sizes[-1]
            means[-1], weights[-1], sizes[-1] = m, wt, size
    return np.repeat(np.asarray(means), sizes)


class IsotonicCalibrator:
    """Monotone map from a score to P(expert is picked), fitted by isotonic
    regression. B. Zadrozny, C. Elkan, "Transforming classifier scores into
    accurate multiclass probability estimates", KDD 2002.

    Fitted on up to millions of (score, label) pairs, so the scores are first
    grouped into `num_bins` equal-count bins and PAV runs on the bin means
    (weighted by bin size). The result is a step function over bin edges.
    """

    def __init__(self, num_bins: int = 1000) -> None:
        self.num_bins = num_bins
        self.edges: np.ndarray | None = None
        self.values: np.ndarray | None = None

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> IsotonicCalibrator:
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        labels = np.asarray(labels, dtype=np.float64).reshape(-1)
        edges = np.unique(np.quantile(scores, np.linspace(0, 1, self.num_bins + 1)[1:-1]))
        bins = np.searchsorted(edges, scores, side="right")
        count = np.bincount(bins, minlength=len(edges) + 1).astype(np.float64)
        total = np.bincount(bins, weights=labels, minlength=len(edges) + 1)
        keep = count > 0
        means = np.where(keep, total / np.maximum(count, 1), 0.0)
        fitted = np.empty(len(count))
        fitted[keep] = pool_adjacent_violators(means[keep], count[keep])
        # an empty bin (possible only at the ends with heavy ties) takes its neighbour's value
        fitted[~keep] = np.interp(np.flatnonzero(~keep), np.flatnonzero(keep), fitted[keep])
        self.edges, self.values = edges, fitted
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if self.edges is None or self.values is None:
            raise RuntimeError("fit() first")
        return self.values[np.searchsorted(self.edges, scores, side="right")].astype(np.float32)

    def to_dict(self) -> dict:
        if self.edges is None or self.values is None:
            raise RuntimeError("fit() first")
        return {"num_bins": self.num_bins, "edges": self.edges.tolist(), "values": self.values.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> IsotonicCalibrator:
        cal = cls(d["num_bins"])
        cal.edges = np.asarray(d["edges"], dtype=np.float64)
        cal.values = np.asarray(d["values"], dtype=np.float64)
        if len(cal.values) != len(cal.edges) + 1 or (np.diff(cal.values) < 0).any():
            raise ValueError(
                "not a fitted isotonic calibrator (values must be one longer than edges, nondecreasing)"
            )
        return cal


def ranks_desc(probs: np.ndarray) -> np.ndarray:
    """0-based rank of each entry along the last axis, highest first; ties
    keep index order (the order predictor.offline.top_k and the prefetch
    policy use)."""
    order = np.argsort(-probs, axis=-1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(probs.shape[-1]), axis=-1)
    return ranks


class RankedIsotonicCalibrator:
    """One isotonic map per rank of the guess (rank 1 = the expert with the
    highest router probability), ranks >= max_rank sharing the last one.

    A single map (IsotonicCalibrator) is right on average over all experts
    but not per rank: the same raw probability means more for the top guess
    than for the eighth, because what matters is how it compares with the
    other candidates of that token. Conditioning on rank captures that.
    Each map is fitted with the same isotonic regression (Zadrozny & Elkan,
    KDD 2002)."""

    def __init__(self, max_rank: int = 12, num_bins: int = 200):
        self.max_rank = max_rank
        self.num_bins = num_bins
        self.per_rank: list[IsotonicCalibrator] = []

    def _bucket(self, ranks: np.ndarray) -> np.ndarray:
        return np.minimum(ranks, self.max_rank - 1)

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> RankedIsotonicCalibrator:
        """probs, labels: [tokens, candidates] (all of a token's experts, any order)."""
        bucket = self._bucket(ranks_desc(probs))
        self.per_rank = [
            IsotonicCalibrator(self.num_bins).fit(probs[bucket == r], labels[bucket == r])
            for r in range(self.max_rank)
        ]
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        """probs: [..., candidates] of one token each, any order."""
        if not self.per_rank:
            raise RuntimeError("fit() first")
        probs = np.asarray(probs)
        bucket = self._bucket(ranks_desc(probs))
        out = np.empty(probs.shape, dtype=np.float32)
        for r, cal in enumerate(self.per_rank):
            m = bucket == r
            if m.any():
                out[m] = cal.predict(probs[m])
        return out

    def to_dict(self) -> dict:
        return {
            "kind": "rank_isotonic",
            "max_rank": self.max_rank,
            "per_rank": [c.to_dict() for c in self.per_rank],
        }

    @classmethod
    def from_dict(cls, d: dict) -> RankedIsotonicCalibrator:
        cal = cls(d["max_rank"])
        cal.per_rank = [IsotonicCalibrator.from_dict(x) for x in d["per_rank"]]
        if len(cal.per_rank) != cal.max_rank:
            raise ValueError("one isotonic map per rank expected")
        return cal


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, num_bins: int = 20) -> float:
    """Equal-width-bin ECE: sum over bins of (bin share) * |observed rate -
    mean predicted probability|. M. P. Naeini, G. F. Cooper, M. Hauskrecht,
    "Obtaining well calibrated probabilities using Bayesian binning", AAAI
    2015; C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger, "On calibration of
    modern neural networks", ICML 2017."""
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    bins = np.minimum((probs * num_bins).astype(np.int64), num_bins - 1)
    count = np.bincount(bins, minlength=num_bins)
    gap = np.abs(
        np.bincount(bins, weights=labels, minlength=num_bins)
        - np.bincount(bins, weights=probs, minlength=num_bins)
    )
    return float(gap.sum() / max(count.sum(), 1))


def reliability_table(probs: np.ndarray, labels: np.ndarray, num_bins: int = 20) -> list[dict]:
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    bins = np.minimum((probs * num_bins).astype(np.int64), num_bins - 1)
    out = []
    for b in range(num_bins):
        m = bins == b
        if m.any():
            out.append(
                {
                    "lo": b / num_bins,
                    "hi": (b + 1) / num_bins,
                    "count": int(m.sum()),
                    "mean_predicted": float(probs[m].mean()),
                    "observed": float(labels[m].mean()),
                }
            )
    return out


# ---------------------------------------------------------------------------
# Two layers ahead: expert-to-expert transitions
# ---------------------------------------------------------------------------


class TransitionModel:
    """P(expert j picked at the target layer | expert i picked at the source
    layer), counted over tokens. Used for predicting further ahead than the
    Fate-style cross-layer gate allows from recorded traces: that needs the
    hidden state, which the traces don't store (see docs/limitations.md).

    Additive (Laplace) smoothing `alpha` keeps unseen pairs above zero."""

    def __init__(self, num_experts: int, alpha: float = 1.0) -> None:
        self.num_experts = num_experts
        self.alpha = alpha
        self.pair = np.zeros((num_experts, num_experts), dtype=np.float64)
        self.src = np.zeros(num_experts, dtype=np.float64)
        self.dst_marginal = np.zeros(num_experts, dtype=np.float64)

    def add(self, src_used: np.ndarray, dst_used: np.ndarray) -> None:
        """[T, E] bool each, same tokens."""
        s, d = src_used.astype(np.float64), dst_used.astype(np.float64)
        self.pair += s.T @ d
        self.src += s.sum(axis=0)
        self.dst_marginal += d.sum(axis=0)

    def conditional(self) -> np.ndarray:
        """[E_src, E_dst] conditional probabilities."""
        prior = (self.dst_marginal + self.alpha) / (self.dst_marginal.sum() + self.alpha * self.num_experts)
        return (self.pair + self.alpha * prior * self.num_experts) / (
            self.src[:, None] + self.alpha * self.num_experts
        )

    def score(self, src_weights: np.ndarray) -> np.ndarray:
        """[N, E_src] weights over source experts (one-hot picks, or a predicted
        distribution) -> [N, E_dst] log-probability-like scores."""
        return np.log(src_weights @ self.conditional() + 1e-12).astype(np.float32)


# ---------------------------------------------------------------------------
# Reuse model: P(used at a layer's next visit | its recent history)
# ---------------------------------------------------------------------------

GAP_EXACT = 16  # gaps 1..16 get a bucket each
GAP_BUCKETS = GAP_EXACT + 3  # then 17-32, 33-64, 65+
NEVER = GAP_BUCKETS  # never used before: its own bucket
RECENT_WINDOW = 8


def gap_bucket(gap: np.ndarray) -> np.ndarray:
    """gap = next visit - last use (>= 1), or < 0 for "never used"."""
    g = np.asarray(gap, dtype=np.int64)
    b = np.where(
        g <= GAP_EXACT, g - 1, np.where(g <= 32, GAP_EXACT, np.where(g <= 64, GAP_EXACT + 1, GAP_EXACT + 2))
    )
    return np.where(g < 1, NEVER, b)


class ReuseModel:
    """How likely an expert is to be picked at its layer's next visit, given
    how many visits ago it was last picked (gap) and how often it was picked
    in the last RECENT_WINDOW visits (count). A table of observed rates,
    fitted on tuning prompts; thin cells fall back to the gap-only rate.

    This is what a prediction-aware cache uses for every layer the
    cross-layer predictor can't see yet."""

    def __init__(self, min_count: int = 200) -> None:
        self.min_count = min_count
        self.events = np.zeros((NEVER + 1, RECENT_WINDOW + 1), dtype=np.float64)
        self.total = np.zeros_like(self.events)
        self.table: np.ndarray | None = None

    def add(self, used: np.ndarray, first_target: int) -> None:
        """One sequence at one layer: [T, E] bool. Targets are positions >=
        first_target (e.g. only generated tokens); history includes all."""
        num_experts = used.shape[1]
        last = np.full(num_experts, -1, dtype=np.int64)
        recent = np.zeros(num_experts, dtype=np.int64)
        for t in range(len(used)):
            if t >= max(first_target, 1):
                gb = gap_bucket(np.where(last >= 0, t - last, -1))
                c = np.minimum(recent, RECENT_WINDOW)
                np.add.at(self.total, (gb, c), 1.0)
                np.add.at(self.events, (gb, c), used[t].astype(np.float64))
            last[used[t]] = t
            recent += used[t]
            if t >= RECENT_WINDOW:
                recent -= used[t - RECENT_WINDOW]

    def fit(self) -> ReuseModel:
        overall = self.events.sum() / max(self.total.sum(), 1.0)
        by_gap = np.where(
            self.total.sum(axis=1) > 0,
            self.events.sum(axis=1) / np.maximum(self.total.sum(axis=1), 1.0),
            overall,
        )
        cell = self.events / np.maximum(self.total, 1.0)
        self.table = np.where(self.total >= self.min_count, cell, by_gap[:, None]).astype(np.float32)
        return self

    def lookup(self, gap: np.ndarray, recent: np.ndarray) -> np.ndarray:
        if self.table is None:
            raise RuntimeError("fit() first")
        return self.table[gap_bucket(gap), np.minimum(recent, RECENT_WINDOW)]
