"""The store builder and reader end to end, offline, on a tiny synthetic model.

The network fetch is replaced by a stub that serves in-memory arrays, so
the real code runs for everything else: quantize, serialize, write at
offset, journal, resume verification, index writing, and the one-read
loader. The real download is exercised by running
`python -m expertrelay.store.build_store` (see docs/expert-store.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from expertrelay.store import build_store, unbuffered_io
from expertrelay.store.expert_reader import ExpertStoreReader
from expertrelay.store.fetch_hf_tensors import TensorLocation
from expertrelay.store.layout import (
    EXPERT_MATRICES,
    ExpertRecordLayout,
    expert_tensor_name,
    plan_resident_layout,
    read_resident_index,
)
from expertrelay.store.quantize import dequantize_rowwise_int8, quantize_rowwise_int8

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="unbuffered I/O is Windows-only")

NUM_LAYERS, NUM_EXPERTS = 2, 3
CONFIG = {
    "moe_intermediate_size": 8,
    "hidden_size": 16,
    "num_hidden_layers": NUM_LAYERS,
    "num_experts": NUM_EXPERTS,
}


def _fake_model() -> tuple[build_store.Plan, dict[str, np.ndarray]]:
    rng = np.random.default_rng(0)
    layout = ExpertRecordLayout.from_config(CONFIG)
    source: dict[str, np.ndarray] = {}
    for layer in range(NUM_LAYERS):
        for e in range(NUM_EXPERTS):
            for m, shape in layout.matrices:
                source[expert_tensor_name(layer, e, m)] = rng.normal(0, 0.02, shape).astype(np.float32)
    source["model.embed_tokens.weight"] = rng.normal(0, 1, (40, 16)).astype(np.float32)
    source["model.layers.0.self_attn.q_proj.bias"] = rng.normal(0, 1, (16,)).astype(np.float32)
    source["model.layers.0.mlp.gate.weight"] = rng.normal(0, 1, (NUM_EXPERTS, 16)).astype(np.float32)

    locations = {
        name: TensorLocation(
            name=name, url="fake://", dtype="BF16", shape=a.shape, data_start=0, nbytes=a.size * 2
        )
        for name, a in source.items()
    }
    resident_names = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.bias",
        "model.layers.0.mlp.gate.weight",
    ]
    plan = build_store.Plan(
        repo_id="fake/repo",
        revision="0" * 40,
        config=CONFIG,
        num_layers=NUM_LAYERS,
        num_experts=NUM_EXPERTS,
        layout=layout,
        resident=plan_resident_layout([(n, source[n].shape) for n in resident_names]),
        locations=locations,
    )
    return plan, source


@pytest.fixture
def fake(monkeypatch):
    plan, source = _fake_model()

    def fake_fetch_rows(loc: TensorLocation, row_start: int = 0, row_end: int | None = None) -> np.ndarray:
        return source[loc.name][row_start:row_end].copy()

    monkeypatch.setattr(build_store, "fetch_rows", fake_fetch_rows)
    # 5 rows of 16 per block: the 40-row embedding is built from 8 separate blocks
    monkeypatch.setattr(build_store, "MAX_BLOCK_ELEMENTS", 5 * 16)
    return plan, source


def _build_everything(plan: build_store.Plan, store_dir: Path) -> None:
    build_store._preallocate(store_dir / build_store.EXPERTS_BIN, plan.experts_bin_bytes)
    build_store._preallocate(store_dir / build_store.RESIDENT_BIN, plan.resident_bin_bytes)
    journal = store_dir / build_store.JOURNAL
    for entry in plan.resident:
        build_store.append_journal(journal, build_store.build_resident(plan, store_dir, entry))
    for layer in range(NUM_LAYERS):
        for e in range(NUM_EXPERTS):
            build_store.append_journal(journal, build_store.build_expert(plan, store_dir, layer, e))


def _finalize(plan, store_dir) -> dict:
    experts, resident, _ = build_store.verified_journal(
        store_dir, build_store.read_journal(store_dir / build_store.JOURNAL)
    )
    return build_store.finalize(plan, store_dir, {"runs": [{"peak_rss_mb": 1.0}]}, experts, resident)


@windows_only
def test_one_read_loads_one_complete_expert(fake, tmp_path, monkeypatch):
    plan, source = fake
    _build_everything(plan, tmp_path)
    _finalize(plan, tmp_path)

    calls = []
    real_read_at = unbuffered_io.read_at

    def counting_read_at(handle, address, offset, nbytes):
        calls.append((offset, nbytes))
        return real_read_at(handle, address, offset, nbytes)

    monkeypatch.setattr(unbuffered_io, "read_at", counting_read_at)
    with ExpertStoreReader(tmp_path) as reader:
        expert = reader.read(1, 2)

    assert calls == [(reader.entry(1, 2).offset, plan.layout.record_size)]  # exactly one read, whole record
    for m in EXPERT_MATRICES:
        original = source[expert_tensor_name(1, 2, m)]
        q, scales = quantize_rowwise_int8(original)
        np.testing.assert_array_equal(expert.q[m], q)  # weights AND scales came from that one read
        np.testing.assert_array_equal(expert.scales[m], scales)
        assert np.all(np.abs(expert.dequantize(m) - original) <= scales[:, None] * (0.5 + 1e-6))


@windows_only
def test_finalize_verifies_and_summarizes(fake, tmp_path):
    plan, _ = fake
    _build_everything(plan, tmp_path)
    summary = _finalize(plan, tmp_path)
    assert summary["verification"]["ok"]
    assert summary["verification"]["experts_checked"] == NUM_LAYERS * NUM_EXPERTS
    stats = summary["expert_quant_error"]
    assert stats["experts"] == NUM_LAYERS * NUM_EXPERTS
    assert 0 < stats["mean_rel_fro_error"] <= stats["max_rel_fro_error"] < 0.05
    assert stats["max_abs_error_over_scale"] <= 0.5 + 1e-6


@windows_only
def test_verify_store_detects_a_flipped_byte(fake, tmp_path):
    plan, _ = fake
    _build_everything(plan, tmp_path)
    _finalize(plan, tmp_path)
    with open(tmp_path / build_store.EXPERTS_BIN, "r+b") as f:
        f.seek(plan.layout.record_size * 4 + 10)  # inside expert (1, 1)
        f.write(b"\x7f")
    result = build_store.verify_store(tmp_path)
    assert not result["ok"] and result["experts_bad"] == [(1, 1)]


def test_resident_tensors_round_trip_including_multi_block(fake, tmp_path):
    plan, source = fake
    _build_everything(plan, tmp_path)
    data = (tmp_path / build_store.RESIDENT_BIN).read_bytes()
    embed, bias, router = plan.resident

    rows, cols = embed.shape
    q = np.frombuffer(data, np.int8, rows * cols, embed.offset).reshape(rows, cols)
    scales = np.frombuffer(data, "<f4", rows, embed.scales_offset)
    expected_q, expected_s = quantize_rowwise_int8(source[embed.name])
    np.testing.assert_array_equal(q, expected_q)  # 8 blocks reassemble into the whole-tensor result
    np.testing.assert_array_equal(scales, expected_s)
    np.testing.assert_allclose(
        dequantize_rowwise_int8(q, scales), source[embed.name], atol=float(scales.max())
    )

    for fp32 in (bias, router):  # stay exact
        stored = np.frombuffer(data, "<f4", int(np.prod(fp32.shape)), fp32.offset).reshape(fp32.shape)
        np.testing.assert_array_equal(stored, source[fp32.name])


def test_resume_rebuilds_only_what_fails_its_hash(fake, tmp_path):
    plan, _ = fake
    _build_everything(plan, tmp_path)
    with open(tmp_path / build_store.EXPERTS_BIN, "r+b") as f:
        f.seek(plan.layout.record_size * 1 + 100)  # inside expert (0, 1)
        f.write(b"\xff\xff")

    experts, resident, dropped = build_store.verified_journal(
        tmp_path, build_store.read_journal(tmp_path / build_store.JOURNAL)
    )
    assert dropped == 1
    assert set(experts) == {(layer, e) for layer in range(NUM_LAYERS) for e in range(NUM_EXPERTS)} - {(0, 1)}
    assert len(resident) == len(plan.resident)


def test_journal_drops_torn_last_line_but_rejects_mid_file_corruption(tmp_path):
    journal = tmp_path / "j.jsonl"
    build_store.append_journal(journal, {"kind": "expert", "n": 1})
    build_store.append_journal(journal, {"kind": "expert", "n": 2})
    with open(journal, "a") as f:
        f.write('{"kind": "exp')  # crash mid-append
    assert [r["n"] for r in build_store.read_journal(journal)] == [1, 2]

    journal.write_text('{"n": 1}\nGARBAGE\n{"n": 3}\n')
    with pytest.raises(ValueError):
        build_store.read_journal(journal)


def test_resident_index_written_by_finalize_matches_plan_offsets(fake, tmp_path):
    if sys.platform != "win32":
        pytest.skip("finalize verifies experts via unbuffered reads")
    plan, _ = fake
    _build_everything(plan, tmp_path)
    _finalize(plan, tmp_path)
    written = read_resident_index(tmp_path / build_store.RESIDENT_INDEX)
    assert [(e.name, e.offset) for e in written] == [(e.name, e.offset) for e in plan.resident]
    assert all(e.sha256 for e in written)


@windows_only
def test_store_report_is_generated_from_the_manifest(fake, tmp_path):
    plan, _ = fake
    _build_everything(plan, tmp_path)
    manifest = {
        "runs": [
            {"started": "t0"},
            {
                "started": "t1",
                "finished": "t2",
                "items_built": 9,
                "bytes_downloaded": 10**9,
                "peak_rss_mb": 250.0,
                "failures": [],
            },
        ],
        "source": {"repo_id": plan.repo_id, "revision": plan.revision},
        "expert_layout": plan.layout.to_dict(),
        "quantization": build_store.QUANTIZATION_METHOD,
        "num_layers": NUM_LAYERS,
        "num_experts": NUM_EXPERTS,
    }
    experts, resident, _ = build_store.verified_journal(
        tmp_path, build_store.read_journal(tmp_path / build_store.JOURNAL)
    )
    build_store.finalize(plan, tmp_path, manifest, experts, resident)

    report = build_store.format_store_report(manifest)
    assert f"pinned at commit `{plan.revision}`" in report
    assert "interrupted (hard kill)" in report  # the crashed run is shown, not hidden
    assert f"{plan.layout.record_size:,} bytes" in report
    assert "Mismatches: 0." in report


def test_store_report_refuses_an_incomplete_store():
    with pytest.raises(ValueError):
        build_store.format_store_report({"complete": False, "summary": None})
