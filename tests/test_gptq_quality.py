"""Resume bookkeeping of the Step B3 quality run (the run itself needs the
real stores and bf16 checkpoint; its method is tested in
test_quantization_quality.py and test_reference_check.py)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

import expertrelay.bench.gptq_quality as gq  # noqa: E402


def test_torn_last_line_is_ignored_and_redone(tmp_path):
    path = tmp_path / "scores.jsonl"
    gq._append_line(path, {"index": 0, "x": 1})
    gq._append_line(path, {"index": 1, "x": 2})
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"index": 2, "x"')  # crash mid-write
    assert sorted(gq._read_lines(path)) == [0, 1]


def test_saved_arrays_are_complete_or_absent(tmp_path):
    target = tmp_path / "int8" / "03.npy"
    gq._save_npy(target, np.arange(6.0).reshape(2, 3))
    np.testing.assert_array_equal(np.load(target), np.arange(6.0).reshape(2, 3))
    assert [p.name for p in target.parent.iterdir()] == ["03.npy"]


def test_work_dir_from_other_settings_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(gq, "WORK", tmp_path / "work")
    gq.check_work_dir({"a": 1})
    gq.check_work_dir({"a": 1})
    with pytest.raises(SystemExit):
        gq.check_work_dir({"a": 2})
