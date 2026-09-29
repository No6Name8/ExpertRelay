"""Which experts to prefetch for the next layer, from the Fate-style guess.

Z. Fang et al., "Fate: Fast Edge Inference of Mixture-of-Experts Models via
Cross-Layer Gate", arXiv:2502.12224 (2025): layer L+1's router applied to
layer L's gate input predicts layer L+1's experts before layer L+1 runs.
The runtime computes those router logits (it owns the compute backend);
this module only turns them into a list of experts, in plain numpy.

The guess is the top-k experts by router probability. Guesses whose
calibrated probability of being picked is below `min_probability` are
dropped: a read that is almost never used still occupies the disk and a
cache slot. The calibration maps the raw router softmax probability of
one expert to P(that expert is among the top-k picked): isotonic
regression (B. Zadrozny, C. Elkan, KDD 2002) fitted on held-out traces by
bench/fit_prefetch_calibration.py, one map per guess rank
(predictor.offline.RankedIsotonicCalibrator) when the record has it, else
a single map (IsotonicCalibrator).

With several tokens in one call (prefill), each expert's score is its
highest probability over the tokens. The calibration was fitted per token
(decode), so applying it to that maximum is a heuristic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from expertrelay.predictor.offline import IsotonicCalibrator, RankedIsotonicCalibrator, softmax


@dataclass(frozen=True)
class PrefetchChoice:
    experts: list[int]
    skipped_low_confidence: int


@dataclass
class PrefetchPolicy:
    top_k: int
    min_probability: float = 0.0
    calibrator: IsotonicCalibrator | RankedIsotonicCalibrator | None = None

    def choose(self, router_logits: np.ndarray) -> PrefetchChoice:
        """router_logits: [n_tokens, E] (or [E]) for the layer being predicted."""
        probs = softmax(np.atleast_2d(np.asarray(router_logits, dtype=np.float32))).max(axis=0)
        order = np.argsort(-probs, kind="stable")[: self.top_k]
        if self.calibrator is None or self.min_probability <= 0.0:
            return PrefetchChoice([int(e) for e in order], 0)
        keep = self.calibrator.predict(probs[order]) >= self.min_probability
        return PrefetchChoice([int(e) for e in order[keep]], int((~keep).sum()))


def load_calibrator(
    path: Path, kind: str = "rank"
) -> tuple[IsotonicCalibrator | RankedIsotonicCalibrator, dict]:
    """From the last record of a bench/fit_prefetch_calibration.py results
    file: (calibrator, where it came from). kind "rank": the rank-aware map
    if the record has one, else the single map; "single": the single map."""
    if kind not in ("rank", "single"):
        raise ValueError(f"unknown calibration kind {kind!r}")
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    d = records[-1] if isinstance(records, list) else records
    provenance = {
        "file": Path(path).name,
        "git_commit": d.get("git_commit"),
        "timestamp": d.get("timestamp"),
        "store": (d.get("model") or {}).get("store"),
        "kind": "rank_isotonic" if kind == "rank" and "rank_calibrator" in d else "isotonic",
    }
    if kind == "rank" and "rank_calibrator" in d:
        return RankedIsotonicCalibrator.from_dict(d["rank_calibrator"]), provenance
    return IsotonicCalibrator.from_dict(d["calibrator"]), provenance
