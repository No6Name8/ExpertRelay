"""GPTQ int4: the conversion from a GPTQ release, RTN on GPTQ's grid, the
kernel over GPTQ's full code range, and a resumable GPTQ store build.

The real release is too large for a test; the conversion on the real
weights is checked on sampled experts by bench.gptq_conversion_check
(results in benchmarks/results/gptq_conversion_check.json), against the
same independent formula these tests use."""

from __future__ import annotations

import importlib.util
import json
import sys

import numpy as np
import pytest
from store_helpers import build_test_store, random_checkpoint

from expertrelay.bench.gptq_conversion_check import reference_dequantize
from expertrelay.runtime.int4_linear import int4_linear
from expertrelay.store.gptq import gptq_to_int4
from expertrelay.store.int4 import (
    QUANTIZER_GPTQ,
    QUANTIZER_RTN_GPTQ_GRID,
    Int4RecordLayout,
    dequantize_groupwise_int4,
    pack_int4,
    quantize_groupwise_int4_gptq_grid,
)

needs_numba = pytest.mark.skipif(importlib.util.find_spec("numba") is None, reason="numba not installed")
windows_only = pytest.mark.skipif(sys.platform != "win32", reason="unbuffered expert reads are Windows-only")


def pack_gptq_int32(codes: np.ndarray, axis: int) -> np.ndarray:
    """Independent GPTQ packer, element by element: 8 codes per int32 along
    `axis`, the first in the lowest 4 bits."""
    codes = np.moveaxis(codes, axis, 0)
    out = np.zeros((codes.shape[0] // 8, *codes.shape[1:]), dtype=np.uint32)
    for i in range(codes.shape[0]):
        out[i // 8] |= codes[i].astype(np.uint32) << np.uint32(4 * (i % 8))
    return np.moveaxis(out, 0, axis).view(np.int32)


def gptq_layer(out_dim: int, in_dim: int, group: int, seed: int, zero: int = 8) -> dict[str, np.ndarray]:
    """A random layer in the release's format (v1: zero points stored minus one)."""
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 16, (in_dim, out_dim))
    codes[0, 0], codes[1, 0] = 0, 15  # both ends of the code range
    return {
        "qweight": pack_gptq_int32(codes, axis=0),
        "qzeros": pack_gptq_int32(np.full((in_dim // group, out_dim), zero - 1), axis=1),
        "scales": rng.uniform(0.001, 0.05, (in_dim // group, out_dim)).astype(np.float16),
        "g_idx": (np.arange(in_dim) // group).astype(np.int32),
        "bias": np.zeros(out_dim, dtype=np.float16),
    }


def test_conversion_matches_the_documented_formula_exactly():
    t = gptq_layer(48, 64, 16, seed=0)
    packed, scales = gptq_to_int4(*t.values(), group=16)
    assert packed.shape == (48, 32) and scales.shape == (48, 4)
    ours = dequantize_groupwise_int4(packed, scales, 16)
    np.testing.assert_array_equal(
        ours, reference_dequantize(t["qweight"], t["qzeros"], t["scales"], t["g_idx"])
    )
    # element by element, from the raw int32s
    q, z, s = t["qweight"].view(np.uint32), t["qzeros"].view(np.uint32), t["scales"]
    for o, i in [(0, 0), (0, 1), (47, 63), (13, 37)]:
        code = (int(q[i // 8, o]) >> (4 * (i % 8))) & 0xF
        zero = ((int(z[i // 16, o // 8]) >> (4 * (o % 8))) & 0xF) + 1
        assert ours[o, i] == np.float32(code - zero) * np.float32(s[i // 16, o])


def test_reference_formula_honours_zero_points_and_g_idx():
    # the reference is general, so it can catch a wrong assumption in the converter
    t = gptq_layer(8, 16, 8, seed=1, zero=3)
    t["g_idx"] = t["g_idx"][::-1].copy()
    w = reference_dequantize(t["qweight"], t["qzeros"], t["scales"], t["g_idx"])
    codes = (t["qweight"].view(np.uint32)[0, 0] & 0xF).astype(np.int32)
    assert w[0, 0] == np.float32(codes - 3) * np.float32(t["scales"][1, 0])


@pytest.mark.parametrize(
    "change",
    [
        lambda t: t.update(qzeros=gptq_layer(48, 64, 16, 0, zero=7)["qzeros"]),
        lambda t: t.update(g_idx=np.roll(t["g_idx"], 1)),
        lambda t: t["bias"].__setitem__(3, 0.5),
    ],
    ids=["zero point 7", "desc_act order", "bias"],
)
def test_conversion_refuses_what_it_cannot_represent(change):
    t = gptq_layer(48, 64, 16, seed=0)
    change(t)
    with pytest.raises(ValueError):
        gptq_to_int4(*t.values(), group=16)


def gptq_quantizer_reference(w: np.ndarray) -> tuple[np.ndarray, np.float16]:
    """GPTQ's Quantizer (sym, 4 bits) on one group, transcribed scalar by scalar."""
    xmin, xmax = min(float(w.min()), 0.0), max(float(w.max()), 0.0)
    xmax = max(abs(xmin), xmax)
    if xmin < 0:
        xmin = -xmax
    if xmin == 0 and xmax == 0:
        xmin, xmax = -1.0, 1.0
    scale = np.float16((xmax - xmin) / 15)
    codes = np.clip(np.rint(w / np.float32(scale)) + 8, 0, 15)
    return (codes - 8).astype(np.int8), scale


def test_rtn_on_gptq_grid_matches_the_gptq_quantizer():
    rng = np.random.default_rng(2)
    w = rng.standard_t(3, (6, 64)).astype(np.float32) * 0.02
    w[1, :16] = np.abs(w[1, :16])  # no negative weight: GPTQ's xmin = 0 case
    w[2, 16:32] = 0  # all-zero group
    q, s = quantize_groupwise_int4_gptq_grid(w, 16)
    assert q.min() >= -8 and q.max() <= 7 and s.dtype == np.float16
    for r in range(6):
        for g in range(4):
            ref_q, ref_s = gptq_quantizer_reference(w[r, 16 * g : 16 * (g + 1)])
            assert s[r, g] == ref_s
            np.testing.assert_array_equal(q[r, 16 * g : 16 * (g + 1)], ref_q)
    np.testing.assert_array_equal(dequantize_groupwise_int4(pack_int4(q), s, 16)[2, 16:32], 0.0)
    assert (q == -8).any()  # uses the code the -7..7 grid leaves out


@pytest.mark.parametrize("kernel", [pytest.param("fused", marks=needs_numba), "blocked"])
@pytest.mark.parametrize("n", [1, 4, 20])
def test_int4_kernels_match_float64_on_gptq_weights(kernel, n):
    t = gptq_layer(304, 128, 32, seed=3)
    packed, s = gptq_to_int4(*t.values(), group=32)
    rng = np.random.default_rng(4)
    x = rng.normal(0, 1, (n, 128)).astype(np.float32)
    exact = (
        x.astype(np.float64)
        @ reference_dequantize(t["qweight"], t["qzeros"], t["scales"], t["g_idx"]).astype(np.float64).T
    )
    np.testing.assert_allclose(int4_linear(x, packed, s, 32, kernel=kernel), exact, rtol=1e-5, atol=1e-5)


def test_layout_records_quantizer_and_old_layouts_default():
    config = {"moe_intermediate_size": 16, "hidden_size": 32}
    layout = Int4RecordLayout.from_config(config, 8, QUANTIZER_GPTQ)
    assert Int4RecordLayout.from_dict(layout.to_dict()) == layout
    old = {k: v for k, v in layout.to_dict().items() if k != "quantizer"}
    assert Int4RecordLayout.from_dict(old).quantizer == "rtn_absmax7"
    with pytest.raises(ValueError):
        Int4RecordLayout.from_config(config, 8, "awq")


CONFIG = {
    "vocab_size": 64,
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 16,
    "shared_expert_intermediate_size": 24,
    "norm_topk_prob": False,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
}
GROUP = 8


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    """A tiny bf16 checkpoint, the int8 store built from it, and a GPTQ-format
    release of the same model (random codes, the release's format)."""
    torch = pytest.importorskip("torch")
    st = pytest.importorskip("safetensors.torch")
    from safetensors.numpy import save_file

    arrays = random_checkpoint(CONFIG, seed=5)
    arrays = {k: torch.from_numpy(v).to(torch.bfloat16).float().numpy() for k, v in arrays.items()}
    bf16 = tmp_path_factory.mktemp("bf16")
    st.save_file(
        {k: torch.from_numpy(v).to(torch.bfloat16) for k, v in arrays.items()}, str(bf16 / "m.safetensors")
    )
    (bf16 / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(arrays, "m.safetensors")})
    )
    int8 = tmp_path_factory.mktemp("int8")
    mp = pytest.MonkeyPatch()
    build_test_store(int8, CONFIG, arrays, mp)
    mp.undo()

    gptq = tmp_path_factory.mktemp("gptq")
    tensors, seed = {}, 10
    for layer in range(CONFIG["num_hidden_layers"]):
        for e in range(CONFIG["num_experts"]):
            for name, (rows, cols) in Int4RecordLayout.from_config(CONFIG, GROUP).matrices:
                seed += 1
                for k, v in gptq_layer(rows, cols, GROUP, seed).items():
                    tensors[f"model.layers.{layer}.mlp.experts.{e}.{name}.{k}"] = v
    save_file(tensors, str(gptq / "g.safetensors"))
    (gptq / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, "g.safetensors")})
    )
    quant = {"quant_method": "gptq", "bits": 4, "group_size": GROUP, "sym": True, "desc_act": False}
    (gptq / "config.json").write_text(json.dumps({**CONFIG, "quantization_config": quant}))
    return bf16, int8, gptq, tensors


@windows_only
def test_gptq_build_is_exact_and_resumes(checkpoints, tmp_path):
    from expertrelay.store.build_int4_store import JOURNAL, build
    from expertrelay.store.expert_reader import EXPERTS_BIN, ExpertStoreReader
    from expertrelay.store.layout import parse_expert_record

    bf16, int8, gptq, tensors = checkpoints
    out = tmp_path / "gptq-store"
    first = build(int8, bf16, out, GROUP, QUANTIZER_GPTQ, gptq)
    assert first["bad"] == [] and first["resumed_records"] == 0
    with ExpertStoreReader(out) as r:
        assert r.layout.quantizer == QUANTIZER_GPTQ
        for x in r.entries():
            rec = parse_expert_record(r.layout, r.read_raw(x.layer, x.expert))
            for name, _ in r.layout.matrices:
                p = f"model.layers.{x.layer}.mlp.experts.{x.expert}.{name}."
                ref = reference_dequantize(
                    *(tensors[p + k] for k in ("qweight", "qzeros", "scales", "g_idx"))
                )
                np.testing.assert_array_equal(dequantize_groupwise_int4(*rec[name], GROUP), ref)
            del rec
        record_size, victim = r.layout.record_size, r.entries()[5]

    # a crash that tore one record and lost the index: the rerun redoes just that record
    with open(out / EXPERTS_BIN, "r+b") as f:
        f.seek(victim.offset + 100)
        f.write(b"\xff" * 8)
    (out / "experts_index.json").unlink()
    second = build(int8, bf16, out, GROUP, QUANTIZER_GPTQ, gptq)
    total = CONFIG["num_hidden_layers"] * CONFIG["num_experts"]
    assert second["bad"] == [] and second["resumed_records"] == total - 1
    assert (out / EXPERTS_BIN).stat().st_size == total * record_size
    assert len((out / JOURNAL).read_text().splitlines()) == 1 + total + 1

    # a journal from other settings is never mixed in
    with pytest.raises(SystemExit):
        build(int8, bf16, out, GROUP, QUANTIZER_RTN_GPTQ_GRID)


@windows_only
def test_rtn_gptq_grid_build_quantizes_bf16(checkpoints, tmp_path):
    from expertrelay.store.build_int4_store import build
    from expertrelay.store.expert_reader import ExpertStoreReader
    from expertrelay.store.layout import parse_expert_record

    bf16, int8, _, _ = checkpoints
    out = tmp_path / "rtn-store"
    assert build(int8, bf16, out, GROUP, QUANTIZER_RTN_GPTQ_GRID)["bad"] == []
    arrays = random_checkpoint(CONFIG, seed=5)
    with ExpertStoreReader(out) as r:
        x = r.entries()[3]
        packed, s = parse_expert_record(r.layout, r.read_raw(x.layer, x.expert))["down_proj"]
        import torch

        w = torch.from_numpy(arrays[f"model.layers.{x.layer}.mlp.experts.{x.expert}.down_proj.weight"])
        q, ref_s = quantize_groupwise_int4_gptq_grid(w.to(torch.bfloat16).float().numpy(), GROUP)
        np.testing.assert_array_equal(packed, pack_int4(q))
        np.testing.assert_array_equal(s, ref_s)
        del packed, s
