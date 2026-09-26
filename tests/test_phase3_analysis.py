"""Phase 3 analysis functions on hand-made traces with known answers."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from expertrelay.bench.phase3_analysis import (  # noqa: E402
    Trace,
    access_stream,
    fate_accuracy,
    popularity,
    reuse_distances,
    temporal_locality,
)
from expertrelay.cache.simulator import INFINITE  # noqa: E402
from expertrelay.runtime.expert_trace import (  # noqa: E402
    NO_EXPERT,
    PHASE_DECODE,
    PHASE_PREFILL,
    record_dtype,
)

L, E, K = 2, 6, 2


def make_trace(rows: list[tuple[int, int, int, list[int]]], category: str = "en") -> Trace:
    """rows: (position, layer, phase, experts). Everything else defaulted."""
    rec = np.zeros(len(rows), dtype=record_dtype(E, K))
    for i, (pos, layer, phase, experts) in enumerate(rows):
        rec[i]["position"], rec[i]["layer"], rec[i]["phase"] = pos, layer, phase
        rec[i]["experts"] = experts
        rec[i]["fate_experts"] = NO_EXPERT
        rec[i]["prev_token_experts"] = NO_EXPERT
    header = {"num_layers": L, "num_experts": E, "top_k": K, "category": category}
    return Trace(header, rec)


def test_access_stream_follows_runtime_load_order():
    # prefill of 2 tokens, then 1 decode token
    t = make_trace(
        [
            (0, 0, PHASE_PREFILL, [3, 1]),
            (0, 1, PHASE_PREFILL, [0, 2]),
            (1, 0, PHASE_PREFILL, [1, 5]),
            (1, 1, PHASE_PREFILL, [2, 4]),
            (2, 0, PHASE_DECODE, [5, 0]),
            (2, 1, PHASE_DECODE, [4, 3]),
        ]
    )
    keys, phases = access_stream(t)
    # prefill loads each layer's UNIQUE experts once, ascending; decode loads its own top-k
    assert keys.tolist() == [1, 3, 5, 6 + 0, 6 + 2, 6 + 4, 0, 5, 6 + 3, 6 + 4]
    assert phases.tolist() == [PHASE_PREFILL] * 6 + [PHASE_DECODE] * 4


def test_reuse_distance_counts_tokens_per_layer():
    t = make_trace(
        [
            (0, 0, PHASE_PREFILL, [1, 2]),
            (1, 0, PHASE_DECODE, [1, 3]),  # 1 last used at 0 -> 1; 3 first use
            (2, 0, PHASE_DECODE, [4, 5]),  # both first use
            (3, 0, PHASE_DECODE, [2, 3]),  # 2 last at 0 -> 3; 3 last at 1 -> 2
        ]
    )
    assert reuse_distances([t]).tolist() == [1, INFINITE, INFINITE, INFINITE, 3, 2]


def test_temporal_locality_and_fate_recall():
    t = make_trace([(0, 1, PHASE_DECODE, [1, 2]), (1, 1, PHASE_DECODE, [2, 3])])
    t.rec["prev_token_experts"][1] = [1, 2]  # token 1 shares expert 2 with token 0
    loc = temporal_locality(t.rec, K)
    assert loc["records"] == 1 and loc["mean_shared_fraction"] == 0.5 and loc["all_shared"] == 0.0

    t.rec["fate_experts"][0] = [1, 2]  # both right
    t.rec["fate_experts"][1] = [5, 3]  # one right
    t.rec["fate_logits"][:] = 0
    acc = fate_accuracy(t.rec, K)
    assert acc["recall_at_k"] == pytest.approx(0.75)
    assert acc["hits_distribution"] == [0.0, 0.5, 0.5]


def test_popularity_top10pct_share():
    # layer 0: expert 0 picked in every token; E=6 -> top 10% = 1 expert
    rows = [(p, 0, PHASE_DECODE, [0, 1 + p % 5]) for p in range(10)]
    pop = popularity([make_trace(rows)], None)
    assert pop["top10pct_experts"] == 1
    assert pop["top10pct_share_per_layer"][0] == pytest.approx(10 / 20)
