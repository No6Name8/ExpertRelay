"""Step C bookkeeping (bench.fair_test, bench.resumable): the run order,
resuming after an interruption, the token check and the llama.cpp metric
definitions. The runs themselves need the real store and llama.cpp."""

from __future__ import annotations

import pytest

import expertrelay.bench.fair_test as ft
from expertrelay.bench.resumable import RunSet


def test_order_interleaves_every_setup_and_runs_normal_load_last():
    order = ft.plan()
    assert order[-1] == "A#0" and len(order) == 3 * 5 + 1
    for r in range(3):
        rnd = [x for x in order if x.endswith(f"#{r}") and not x.startswith("A")]
        assert sorted(x[0] for x in rnd) == list("BCDEF")
    firsts = [next(x for x in order if x.endswith(f"#{r}"))[0] for r in range(3)]
    assert len(set(firsts)) == 3  # a different setup opens each round


def test_runset_resumes_and_keeps_failed_attempts(tmp_path):
    runs = RunSet(tmp_path / "w", {"a": 1})
    runs.start("t")
    runs.save("x#0", {"status": "ok", "v": 1})
    runs.save("y#0", {"status": "failed: boom"})
    again = RunSet(tmp_path / "w", {"a": 1})
    assert again.start("t") == 1
    assert again.done("x#0") and not again.done("y#0")
    again.prepare("y#0")
    again.save("y#0", {"status": "ok"})
    assert [f["status"] for f in again.failed_attempts()] == ["failed: boom"]
    assert [s["index"] for s in again.sessions()] == [0, 1]
    assert again.load("y#0")["session"] == 1
    with pytest.raises(SystemExit):
        RunSet(tmp_path / "w", {"a": 2}).start("t")


def test_token_check_flags_a_differing_run():
    def run(tokens):
        return {"status": "ok", "run": {"prompts": [{"generated_ids": tokens}]}}

    results = {"B#0": run([1, 2]), "C#0": run([1, 2]), "D#0": run([1, 3]), "A#0": {"status": "timeout"}}
    check = ft.token_check(results, ["B#0", "C#0", "D#0", "A#0"])
    assert check == {"B#0": True, "C#0": True, "D#0": False, "reference": "B#0"}


def test_llama_metrics_use_the_same_definitions_as_expertrelay():
    runs = [
        {
            "decode_tokens": 63,
            "decode_s": 6.3,
            "ttft_s": 1.0,
            "server_timings": {"predicted_per_second": 11, "prompt_ms": 900},
        },
        {"decode_tokens": 63, "decode_s": 12.6, "ttft_s": 3.0, "server_timings": None},
    ]
    m = ft.llama_metrics(runs)
    assert m["decode_tokens_per_s"] == pytest.approx(126 / 18.9)  # tokens 2..64 over their time, all prompts
    assert m["time_to_first_token_mean_s"] == 2.0
    assert m["server_predicted_per_second_mean"] == 11


def test_llama_logprobs_asks_every_prefix_and_refuses_a_changed_prompt(monkeypatch):
    import numpy as np

    import expertrelay.bench.llamacpp_quality as lq

    asked = []

    def fake_request(prefix, vocab):
        asked.append(list(prefix))
        lp = np.full(vocab, -10.0)
        lp[prefix[-1] % vocab] = -0.01
        return lp, len(prefix)

    monkeypatch.setattr(lq, "_request", fake_request)
    rows = lq.llama_logprobs([3, 1, 4], vocab=8)
    assert asked == [[3], [3, 1], [3, 1, 4]] and rows.shape == (3, 8)
    assert list(rows.argmax(-1)) == [3, 1, 4]
    monkeypatch.setattr(
        lq, "_request", lambda prefix, vocab: (np.zeros(vocab), len(prefix) + 1)
    )  # a BOS added
    with pytest.raises(RuntimeError):
        lq.llama_logprobs([3, 1], vocab=8)
