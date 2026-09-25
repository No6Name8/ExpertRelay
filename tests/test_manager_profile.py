"""Tests for expertrelay.manager.profile's hardware-independent logic:
chunk sizing, offset generation, parameter validation, parsing, timing math,
serialization and formatting.

The parts that do need real hardware (PowerShell drive query, registry CPU
name, the unbuffered write/read loop) are exercised by running
`python -m expertrelay.manager.profile`, whose output is committed to
benchmarks/results/ and docs/machine-profile.md.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from expertrelay.manager.profile import (
    EXPERT_BYTES_FP16,
    EXPERT_BYTES_INT8,
    IO_ALIGNMENT,
    MachineProfile,
    ReadSpeed,
    format_markdown,
    format_summary,
    output_path_for,
    parse_windows_drive_json,
    random_aligned_offsets,
    sequential_offsets,
    validate_disk_test_params,
)


def test_expert_sizes_match_qwen_moe_expert_and_are_io_aligned():
    # gate + up + down projections, each 1408 x 2048
    assert EXPERT_BYTES_INT8 == 3 * 1408 * 2048
    assert EXPERT_BYTES_FP16 == 2 * EXPERT_BYTES_INT8
    assert EXPERT_BYTES_INT8 % IO_ALIGNMENT == 0
    assert EXPERT_BYTES_FP16 % IO_ALIGNMENT == 0


def test_sequential_offsets_are_back_to_back_and_fit():
    offsets = sequential_offsets(file_bytes=10 * 4096, chunk_bytes=3 * 4096)
    assert offsets == [0, 3 * 4096, 6 * 4096]  # a 4th chunk would overrun the file
    assert offsets[-1] + 3 * 4096 <= 10 * 4096


def test_random_offsets_are_aligned_in_bounds_and_seeded():
    file_bytes, chunk = 2**30, EXPERT_BYTES_FP16
    offsets = random_aligned_offsets(file_bytes, chunk, count=500, seed=7)

    assert len(offsets) == 500
    assert all(o % IO_ALIGNMENT == 0 for o in offsets)
    assert all(o >= 0 and o + chunk <= file_bytes for o in offsets)
    assert offsets == random_aligned_offsets(file_bytes, chunk, count=500, seed=7)
    assert offsets != random_aligned_offsets(file_bytes, chunk, count=500, seed=8)


def test_random_offsets_can_reach_the_last_valid_slot():
    # a file exactly one alignment unit larger than the chunk has two slots: 0 and 4096
    offsets = random_aligned_offsets(file_bytes=2 * 4096, chunk_bytes=4096, count=200, seed=0)
    assert set(offsets) == {0, 4096}


@pytest.mark.parametrize(
    ("file_bytes", "chunks"),
    [
        (2**30 + 1, (EXPERT_BYTES_INT8,)),  # unaligned file size
        (2**30, (EXPERT_BYTES_INT8 + 1,)),  # unaligned chunk
        (4096, (8192,)),  # chunk bigger than file
    ],
)
def test_validate_disk_test_params_rejects_bad_input(file_bytes, chunks):
    with pytest.raises(ValueError):
        validate_disk_test_params(file_bytes, chunks)


def test_validate_disk_test_params_accepts_defaults():
    validate_disk_test_params(2**30, (EXPERT_BYTES_INT8, EXPERT_BYTES_FP16))


def test_parse_windows_drive_json_real_shape():
    # captured from this repo's dev machine
    text = '{"Model": "CS3160PGFCV100DCAFF", "BusType": "NVMe", "MediaType": "SSD"}'
    assert parse_windows_drive_json(text) == ("CS3160PGFCV100DCAFF", "NVMe", "SSD")


@pytest.mark.parametrize(
    "text",
    ["", "not json", "[1, 2]", '{"Model": "", "BusType": "Unspecified", "MediaType": null}'],
)
def test_parse_windows_drive_json_unknowns_are_none(text):
    assert parse_windows_drive_json(text) == (None, None, None)


def test_read_speed_from_timings():
    r = ReadSpeed.from_timings(
        pattern="random",
        chunk_bytes=1_000_000,
        reads_per_run=100,
        seconds_per_run=[0.1, 0.2, 0.05],
        device_bytes_read_per_run=[100_000_000, 100_500_000, 100_000_000],
    )
    assert r.bytes_per_run == 100_000_000
    assert r.mb_per_s_per_run == pytest.approx([1000.0, 500.0, 2000.0])
    assert r.median_mb_per_s == pytest.approx(1000.0)
    assert r.cache_bypass_verified


def test_read_speed_flags_a_cache_served_run():
    # one run read almost nothing from the device: it came from the file
    # cache, so the whole measurement must not be reported as verified
    r = ReadSpeed.from_timings(
        pattern="sequential",
        chunk_bytes=1_000_000,
        reads_per_run=100,
        seconds_per_run=[0.1, 0.01],
        device_bytes_read_per_run=[100_000_000, 4096],
    )
    assert not r.cache_bypass_verified


def test_profile_to_dict_is_json_serializable(fake_profile: MachineProfile):
    data = json.loads(json.dumps(fake_profile.to_dict()))
    assert data["cpu"]["model"] == "Test CPU 9000"
    assert data["disk_read"]["results"][0]["pattern"] == "sequential"


def test_output_path_sanitizes_hostname(tmp_path: Path):
    assert output_path_for("my host/01", tmp_path) == tmp_path / "machine_profile_my_host_01.json"


def test_format_summary_has_key_numbers(fake_profile: MachineProfile):
    text = format_summary(fake_profile)
    assert "Test CPU 9000" in text
    assert "NVMe SSD" in text
    assert "8.00 GB total" in text
    assert "sequential" in text and "random" in text


def test_format_summary_marks_unknowns(fake_profile: MachineProfile):
    profile = replace(fake_profile, cpu=replace(fake_profile.cpu, model=None), disk_read=None)
    text = format_summary(profile)
    assert "unknown" in text
    assert "Disk reads" not in text


def test_format_markdown_has_table_and_caveat(fake_profile: MachineProfile, tmp_path: Path):
    md = format_markdown(fake_profile, timestamp="2026-01-01T00:00:00Z", json_path=tmp_path / "x.json")
    assert md.startswith("# Machine profile: test-host")
    assert "| sequential | 8.65 MB |" in md
    assert "| random | 17.30 MB |" in md
    assert "SLC cache" in md
