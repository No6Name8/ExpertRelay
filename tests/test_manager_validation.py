"""Bookkeeping of the Phase 6 validation run (bench.manager_validation):
resume after an interruption in a new session, and the scoring rule
fixed before the run. The runs themselves are faked (they need the real
store)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import expertrelay.bench.manager_validation as mv

SPEED = {"auto": 2.0, "nocache": 1.8, "cache": 2.0, "adaptive": 2.1, "top8": 1.9}


@pytest.fixture
def fake(tmp_path, monkeypatch):
    monkeypatch.setattr(mv, "WORK", tmp_path / "work")
    monkeypatch.setattr(mv, "OUT", tmp_path / "out.json")
    monkeypatch.setattr(mv, "settings", lambda: {"setups": ["x"]})
    monkeypatch.setattr(mv, "fit_cache", lambda rt, gb: {"used_gb": 1.25})
    for name in ("cache_metrics", "disk_per_token", "read_log_summary"):
        monkeypatch.setattr(mv, name, lambda run: {})
    state = SimpleNamespace(calls=[], crash_at=None)

    def run_one(setup, config, timeout):
        state.calls.append(setup["id"])
        if len(state.calls) == state.crash_at:
            raise KeyboardInterrupt
        kind = setup["id"].split("_", 1)[1]
        run = {
            "prompts": [{"generated_ids": [1, 2, 3]}],
            "load": {"source": "unbuffered" if kind == "nocache" else "cached"},
            "manager": {"cache_slots": 140, "prefetch": "adaptive"} if kind == "auto" else None,
        }
        return {**setup, "status": "ok", "run": run, "metrics": {"decode_tokens_per_s": SPEED[kind]}}

    monkeypatch.setattr(mv, "run_one", run_one)
    return state


def test_resumes_in_a_new_session_and_scores(fake):
    total = len(mv.plan())
    assert total == 20  # 2 regimes x 5 setups x 2 rounds
    fake.crash_at = 7
    with pytest.raises(KeyboardInterrupt):
        mv.main()
    mv.main()
    assert len(fake.calls) == 7 + (total - 6)
    record = json.loads(mv.OUT.read_text())[-1]
    assert record["interrupted"] and [len(s["runs"]) for s in record["sessions"]] == [6, total - 6]
    assert all(record["agreement"].values())
    for regime in ("fast", "slow"):
        s = record["score"][regime]
        assert s["best_hand_set"] == "adaptive" and s["auto_modes"] == ["adaptive"]
        assert s["picked_best"] is True
        assert s["auto_over_best"] == pytest.approx(2.0 / 2.1)


def test_auto_modes_map_the_managers_choice():
    assert mv.auto_mode({"manager": {"cache_slots": 0, "prefetch": "off"}}) == "no cache"
    assert mv.auto_mode({"manager": {"cache_slots": 9, "prefetch": "off"}}) == "cache only"
    assert mv.auto_mode({"manager": {"cache_slots": 9, "prefetch": "top_k"}}) == "top-8"
