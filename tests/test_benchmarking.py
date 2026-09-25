"""Smoke test for expertrelay.benchmarking: the shared metadata every
benchmark record must carry, and the append-to-JSON-file helper."""

from __future__ import annotations

import json
from pathlib import Path

from expertrelay.benchmarking import append_benchmark_record, base_record, machine_profile


def test_base_record_has_required_fields():
    record = base_record(label="unit-test", seed=0, model={"m": 1}, config={"c": 2}, extra_field=3)
    for key in ("label", "timestamp", "git_commit", "git_dirty", "machine", "seed", "model", "config"):
        assert key in record
    assert record["extra_field"] == 3
    assert record["seed"] == 0


def test_machine_profile_has_expected_keys():
    profile = machine_profile()
    for key in ("platform", "python_version", "mindspore_version", "cpu_count_logical", "total_ram_gb"):
        assert key in profile


def test_append_benchmark_record_creates_and_appends(tmp_path: Path):
    out_path = tmp_path / "results" / "demo.json"
    append_benchmark_record(out_path, {"n": 1})
    append_benchmark_record(out_path, {"n": 2})

    records = json.loads(out_path.read_text())
    assert [r["n"] for r in records] == [1, 2]
