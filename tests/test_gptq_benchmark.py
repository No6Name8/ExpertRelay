"""Resume bookkeeping of the Step B3 speed run: an interrupted run continues
in a new session with only the unfinished runs, a failed run is kept and
redone, and the record says which session ran what. The runs themselves
are faked (they need the real stores)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import expertrelay.bench.gptq_benchmark as gb


@pytest.fixture
def fake(tmp_path, monkeypatch):
    monkeypatch.setattr(gb, "WORK", tmp_path / "work")
    monkeypatch.setattr(gb, "OUT", tmp_path / "out.json")
    monkeypatch.setattr(gb, "settings", lambda: {"setups": ["x"]})
    monkeypatch.setattr(gb, "fit_cache", lambda rt, gb_: {"used_gb": 1.25})
    for name in ("cache_metrics", "disk_per_token", "read_log_summary"):
        monkeypatch.setattr(gb, name, lambda run: {})
    state = SimpleNamespace(calls=[], behaviour={})

    def run_one(setup, config, timeout):
        state.calls.append(setup["id"])
        behaviour = state.behaviour.get(len(state.calls))
        if behaviour == "crash":
            raise KeyboardInterrupt
        status = "failed (exit code 1)" if behaviour == "fail" else "ok"
        return {**setup, "status": status, "run": {"prompts": [{"generated_ids": [1, 2]}]}}

    monkeypatch.setattr(gb, "run_one", run_one)
    return state


def test_interrupted_run_resumes_in_a_new_session(fake):
    total = len(gb.plan())
    fake.behaviour = {3: "fail", 5: "crash"}
    with pytest.raises(KeyboardInterrupt):
        gb.main()
    assert len(list((gb.WORK / "runs").glob("*.json"))) == 4  # 4 saved, one of them failed
    gb.main()
    # session 1 does every run not saved as ok: the failed one, the crashed one, the rest
    assert len(fake.calls) == 5 + (total - 3)
    record = json.loads(gb.OUT.read_text())[-1]
    assert record["interrupted"] is True
    assert [len(s["runs"]) for s in record["sessions"]] == [3, total - 3]
    assert len(record["failed_attempts"]) == 1 and record["failed_attempts"][0]["session"] == 0
    assert all(r["status"] == "ok" for r in record["results"]) and len(record["results"]) == total
    assert all(record["agreement"].values())


def test_saved_work_from_other_settings_is_refused(fake, monkeypatch):
    gb.main()
    monkeypatch.setattr(gb, "settings", lambda: {"setups": ["y"]})
    with pytest.raises(SystemExit):
        gb.main()
