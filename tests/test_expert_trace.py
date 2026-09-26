"""Expert-usage traces (runtime.expert_trace): tracing must not change the
model's output, and the format must round-trip exactly."""

from __future__ import annotations

import sys

import numpy as np
import pytest
from store_helpers import build_test_store, random_checkpoint

from expertrelay.runtime.expert_trace import (
    NO_EXPERT,
    PHASE_DECODE,
    PHASE_PREFILL,
    ExpertTraceWriter,
    read_trace,
    record_dtype,
)
from expertrelay.runtime.generate import generate, model_config
from expertrelay.runtime.qwen_moe import ForwardTrace, KVCache, QwenMoe
from expertrelay.runtime.weights import RamExpertSource, ResidentWeights

CONFIG = {
    "vocab_size": 64,
    "hidden_size": 32,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 16,
    "shared_expert_intermediate_size": 24,
    "norm_topk_prob": False,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
}
PROMPT = [3, 9, 27, 17, 51]
NEW = 6


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    if sys.platform != "win32":
        pytest.skip("store finalize verifies via unbuffered reads")
    mp = pytest.MonkeyPatch()
    d = tmp_path_factory.mktemp("store")
    build_test_store(d, CONFIG, random_checkpoint(CONFIG, seed=5), mp)
    mp.undo()
    return d


def _model(store) -> QwenMoe:
    return QwenMoe(model_config(store), ResidentWeights.load(store), RamExpertSource(store))


def _writer(path, **meta) -> ExpertTraceWriter:
    return ExpertTraceWriter(path, num_layers=4, num_experts=8, top_k=2, metadata={"prompt_id": "t", **meta})


def test_tracing_does_not_change_tokens(store, tmp_path):
    plain, _ = generate(_model(store), PROMPT, NEW, max_seq=32)
    traced, _ = generate(_model(store), PROMPT, NEW, max_seq=32, expert_trace=_writer(tmp_path / "t.ert"))
    assert traced == plain


def test_trace_contents(store, tmp_path):
    path = tmp_path / "t.ert"
    generated, _ = generate(_model(store), PROMPT, NEW, max_seq=32, expert_trace=_writer(path, extra="x"))
    header, rec = read_trace(path)
    assert header["extra"] == "x" and header["num_layers"] == 4 and header["top_k"] == 2

    tokens = len(PROMPT) + NEW - 1  # the last generated token is never fed back in
    assert len(rec) == tokens * 4
    assert list(rec["token"][rec["layer"] == 0]) == PROMPT + generated[:-1]
    assert set(rec["phase"][rec["position"] < len(PROMPT)]) == {PHASE_PREFILL}
    assert set(rec["phase"][rec["position"] >= len(PROMPT)]) == {PHASE_DECODE}

    # router weights are sorted descending and are probabilities
    assert np.all(rec["weights"][:, 0] >= rec["weights"][:, 1])
    assert np.all((rec["weights"] > 0) & (rec["weights"] < 1))

    # no Fate prediction for layer 0; one for every other layer
    assert np.all(rec["fate_experts"][rec["layer"] == 0] == NO_EXPERT)
    assert np.all(rec["fate_experts"][rec["layer"] > 0] < 8)

    # previous-token experts: none at position 0, else the same layer at position - 1
    for layer in range(4):
        rows = rec[rec["layer"] == layer]
        assert np.all(rows["prev_token_experts"][0] == NO_EXPERT)
        np.testing.assert_array_equal(rows["prev_token_experts"][1:], rows["experts"][:-1])


def test_trace_experts_match_the_forward_pass(store, tmp_path):
    """The recorded choices are exactly what the model routed to."""
    path = tmp_path / "t.ert"
    model = _model(store)
    trace = ForwardTrace()
    model.forward(np.array(PROMPT), KVCache(model.c, 16), trace=trace, expert_trace=_writer(path))
    _, rec = read_trace(path)
    for layer in range(4):
        np.testing.assert_array_equal(rec["experts"][rec["layer"] == layer], trace.selected_experts[layer])


def test_round_trip_and_truncation(tmp_path):
    path = tmp_path / "t.ert"
    w = _writer(path)
    w.begin(np.array([0, 1]), np.array([7, 8]), PHASE_PREFILL)
    rng = np.random.default_rng(0)
    for layer in range(4):
        logits = rng.normal(size=(2, 8)).astype(np.float32)
        sel = np.argsort(-logits, axis=1)[:, :2]
        w.layer(
            layer,
            logits,
            sel,
            np.full((2, 2), 0.25, np.float32),
            rng.normal(size=(2, 8)) if layer < 3 else None,
        )
    w.end()
    header, rec = read_trace(path)
    assert rec.dtype == record_dtype(8, 2) and len(rec) == 8
    assert header["record_itemsize"] == rec.dtype.itemsize

    path.write_bytes(path.read_bytes()[:-3])  # an interrupted write
    with pytest.raises(ValueError, match="truncated"):
        read_trace(path)


def test_rejects_non_trace_files(tmp_path):
    bad = tmp_path / "x.ert"
    bad.write_bytes(b"NOTATRACE" + b"\0" * 20)
    with pytest.raises(ValueError, match="magic"):
        read_trace(bad)
