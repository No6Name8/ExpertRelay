"""Store format (expertrelay.store.layout): record layout, alignment of every
offset, serialize/parse, index read/write, resident planning."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from expertrelay.store.fetch_hf_tensors import TensorLocation, row_byte_range
from expertrelay.store.layout import (
    KIND_FP32,
    KIND_INT8,
    ExpertIndexEntry,
    ExpertRecordLayout,
    expert_offset,
    parse_expert_record,
    parse_expert_tensor_name,
    plan_resident_layout,
    read_expert_index,
    read_resident_index,
    resident_kind,
    serialize_expert_record,
    write_expert_index,
    write_resident_index,
)
from expertrelay.store.quantize import quantize_rowwise_int8
from expertrelay.store.unbuffered_io import ALIGNMENT, align_up

QWEN_CONFIG = {"moe_intermediate_size": 1408, "hidden_size": 2048, "num_hidden_layers": 24, "num_experts": 60}


def test_qwen_expert_record_layout():
    layout = ExpertRecordLayout.from_config(QWEN_CONFIG)
    assert layout.shapes == {"gate_proj": (1408, 2048), "up_proj": (1408, 2048), "down_proj": (2048, 1408)}
    assert layout.weights_bytes == 3 * 1408 * 2048 == 8_650_752
    assert layout.scales_bytes == (1408 + 1408 + 2048) * 4 == 19_456
    # scales sit immediately after the weights, not on their own boundary
    assert layout.scales_offset("gate_proj") == layout.weights_bytes
    assert layout.record_size == align_up(8_650_752 + 19_456) == 8_671_232
    assert layout.record_size % ALIGNMENT == 0


def test_every_expert_offset_in_the_full_model_is_aligned_and_disjoint():
    layout = ExpertRecordLayout.from_config(QWEN_CONFIG)
    offsets = [expert_offset(layer, e, 60, layout.record_size) for layer in range(24) for e in range(60)]
    assert len(set(offsets)) == 1440
    assert all(o % ALIGNMENT == 0 for o in offsets)
    assert sorted(offsets) == [i * layout.record_size for i in range(1440)]  # back to back, no gaps/overlap


def _tiny_layout() -> ExpertRecordLayout:
    return ExpertRecordLayout.from_config({"moe_intermediate_size": 8, "hidden_size": 16})


def _tiny_quantized(seed: int = 0):
    rng = np.random.default_rng(seed)
    return {
        name: quantize_rowwise_int8(rng.normal(size=shape).astype(np.float32))
        for name, shape in _tiny_layout().matrices
    }


def test_serialize_parse_round_trip_and_zero_padding():
    layout = _tiny_layout()
    quantized = _tiny_quantized()
    record = serialize_expert_record(layout, quantized)
    assert len(record) == layout.record_size == 4096
    assert not any(record[layout.weights_bytes + layout.scales_bytes :])  # padding is zeros

    parsed = parse_expert_record(layout, bytes(record))
    for name, (q, s) in quantized.items():
        np.testing.assert_array_equal(parsed[name][0], q)
        np.testing.assert_array_equal(parsed[name][1], s)


def test_serialize_rejects_wrong_shape():
    quantized = _tiny_quantized()
    q, s = quantized["gate_proj"]
    quantized["gate_proj"] = (q.T.copy(), s)
    with pytest.raises(ValueError):
        serialize_expert_record(_tiny_layout(), quantized)


def _entry(layer: int, expert: int, layout: ExpertRecordLayout) -> ExpertIndexEntry:
    return ExpertIndexEntry(
        layer=layer,
        expert=expert,
        offset=expert_offset(layer, expert, 2, layout.record_size),
        size=layout.record_size,
        shapes={n: list(s) for n, s in layout.matrices},
        sha256="ab" * 32,
        quant_error={"expert": {"rel_fro_error": 0.01, "max_abs_error_over_scale": 0.5}},
    )


def test_expert_index_write_read_round_trip(tmp_path: Path):
    layout = _tiny_layout()
    entries = [_entry(layer, e, layout) for layer in range(2) for e in range(2)]
    write_expert_index(tmp_path / "idx.json", layout, list(reversed(entries)))
    layout2, entries2 = read_expert_index(tmp_path / "idx.json")
    assert layout2 == layout
    assert entries2 == entries  # written sorted by offset


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda e: replace(e, offset=e.offset + 512),  # misaligned
        lambda e: replace(e, size=e.size - 4096),  # not a full record
    ],
)
def test_expert_index_read_rejects_layout_violations(tmp_path: Path, corrupt):
    layout = _tiny_layout()
    entries = [_entry(0, 0, layout), corrupt(_entry(0, 1, layout))]
    write_expert_index(tmp_path / "idx.json", layout, entries)
    with pytest.raises(ValueError):
        read_expert_index(tmp_path / "idx.json")


def test_expert_index_read_rejects_duplicates(tmp_path: Path):
    layout = _tiny_layout()
    e = _entry(0, 0, layout)
    write_expert_index(tmp_path / "idx.json", layout, [e, replace(e, offset=e.offset + layout.record_size)])
    with pytest.raises(ValueError):
        read_expert_index(tmp_path / "idx.json")


@pytest.mark.parametrize(
    ("name", "shape", "kind"),
    [
        ("model.embed_tokens.weight", (151936, 2048), KIND_INT8),
        ("lm_head.weight", (151936, 2048), KIND_INT8),
        ("model.layers.3.self_attn.q_proj.weight", (2048, 2048), KIND_INT8),
        ("model.layers.3.mlp.shared_expert.down_proj.weight", (2048, 5632), KIND_INT8),
        ("model.layers.3.self_attn.q_proj.bias", (2048,), KIND_FP32),
        ("model.layers.3.input_layernorm.weight", (2048,), KIND_FP32),
        ("model.layers.3.mlp.gate.weight", (60, 2048), KIND_FP32),  # router stays exact
        ("model.layers.3.mlp.shared_expert_gate.weight", (1, 2048), KIND_FP32),
    ],
)
def test_resident_kind(name, shape, kind):
    assert resident_kind(name, shape) == kind


def test_resident_plan_offsets_aligned_and_index_round_trip(tmp_path: Path):
    plan = plan_resident_layout(
        [
            ("a.weight", (10, 7)),
            ("a.bias", (10,)),
            ("model.layers.0.mlp.gate.weight", (3, 7)),
            ("b.weight", (5000, 3)),
        ]
    )
    assert all(e.offset % ALIGNMENT == 0 and e.region_size % ALIGNMENT == 0 for e in plan)
    for prev, cur in zip(plan, plan[1:], strict=False):
        assert cur.offset == prev.offset + prev.region_size
    int8 = plan[0]
    assert int8.kind == KIND_INT8 and int8.weight_bytes == 70 and int8.scales_offset == int8.offset + 70

    write_resident_index(tmp_path / "r.json", plan)
    assert read_resident_index(tmp_path / "r.json") == plan


def test_parse_expert_tensor_name():
    assert parse_expert_tensor_name("model.layers.23.mlp.experts.59.down_proj.weight") == (
        23,
        59,
        "down_proj",
    )
    assert parse_expert_tensor_name("model.layers.0.mlp.shared_expert.down_proj.weight") is None


def test_row_byte_range_for_row_blocks():
    loc = TensorLocation(name="t", url="u", dtype="BF16", shape=(100, 8), data_start=1000, nbytes=1600)
    assert loc.row_bytes == 16
    assert row_byte_range(loc, 0, 100) == (1000, 2599)  # whole tensor
    assert row_byte_range(loc, 10, 20) == (1160, 1319)
    with pytest.raises(ValueError):
        row_byte_range(loc, 90, 101)
