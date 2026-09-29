"""int4 expert store: quantization, packing, record layout, the int4
kernels, the local bf16 reader, and a full build on a tiny checkpoint run
through the real runtime."""

from __future__ import annotations

import importlib.util
import json
import sys

import numpy as np
import pytest
from store_helpers import build_test_store, random_checkpoint

from expertrelay.runtime import int8_linear
from expertrelay.runtime.int4_linear import int4_linear
from expertrelay.store.int4 import (
    Int4RecordLayout,
    dequantize_groupwise_int4,
    pack_int4,
    parse_int4_record,
    quant_error,
    quantize_groupwise_int4,
    serialize_int4_record,
    unpack_int4,
)

needs_numba = pytest.mark.skipif(importlib.util.find_spec("numba") is None, reason="numba not installed")
windows_only = pytest.mark.skipif(sys.platform != "win32", reason="unbuffered expert reads are Windows-only")


def test_quantization_error_is_at_most_half_a_step_and_nothing_clips():
    rng = np.random.default_rng(0)
    w = rng.standard_t(3, (64, 256)).astype(np.float32) * 0.02  # heavy tails, like real weights
    q, s = quantize_groupwise_int4(w, 32)
    assert q.min() >= -7 and q.max() <= 7 and s.dtype == np.float16 and s.shape == (64, 8)
    err = np.abs(w - dequantize_groupwise_int4(pack_int4(q), s, 32))
    step = np.repeat(s.astype(np.float32), 32, axis=1)
    assert (err <= step / 2 * (1 + 1e-6)).all()
    assert quant_error(w, pack_int4(q), s, 32)["max_abs_error_over_scale"] <= 0.5 + 1e-6


def test_float16_rounding_never_shrinks_a_scale_into_clipping():
    # 7 * float16(max/7) < max for this value: the scale must be bumped up
    w = np.zeros((1, 8), dtype=np.float32)
    w[0, 0] = 0.100006
    q, s = quantize_groupwise_int4(w, 8)
    assert float(s[0, 0]) * 7 >= w[0, 0]
    assert abs(q[0, 0]) <= 7


def test_all_zero_group_is_exact():
    w = np.zeros((2, 16), dtype=np.float32)
    w[1, 8:] = 1.0
    q, s = quantize_groupwise_int4(w, 8)
    assert s[0, 0] == 0 and s[0, 1] == 0 and s[1, 0] == 0
    back = dequantize_groupwise_int4(pack_int4(q), s, 8)
    np.testing.assert_array_equal(back[0], 0.0)  # all-zero groups come back exactly
    np.testing.assert_array_equal(back[1, :8], 0.0)
    assert np.abs(back[1, 8:] - 1.0).max() <= float(s[1, 1]) / 2


def test_pack_unpack_round_trip():
    q = np.random.default_rng(1).integers(-7, 8, (5, 12)).astype(np.int8)
    packed = pack_int4(q)
    assert packed.shape == (5, 6) and packed.dtype == np.uint8
    np.testing.assert_array_equal(unpack_int4(packed), q)
    assert (packed[0, 0] & 0x0F) == q[0, 0] + 8 and (packed[0, 0] >> 4) == q[0, 1] + 8  # even col low


def test_bad_group_size_is_rejected():
    with pytest.raises(ValueError):
        quantize_groupwise_int4(np.zeros((2, 30), np.float32), 8)


def test_record_layout_and_round_trip():
    config = {"moe_intermediate_size": 16, "hidden_size": 32}
    layout = Int4RecordLayout.from_config(config, 8)
    assert layout.weights_bytes == 3 * 16 * 32 // 2
    assert layout.scales_bytes == (16 * 4 + 16 * 4 + 32 * 2) * 2
    assert layout.record_size % 4096 == 0
    assert Int4RecordLayout.from_dict(layout.to_dict()) == layout
    rng = np.random.default_rng(2)
    quantized = {}
    for name, (rows, cols) in layout.matrices:
        q, s = quantize_groupwise_int4(rng.normal(0, 0.1, (rows, cols)).astype(np.float32), 8)
        quantized[name] = (pack_int4(q), s)
    back = parse_int4_record(layout, bytes(serialize_int4_record(layout, quantized)))
    for name in quantized:
        np.testing.assert_array_equal(back[name][0], quantized[name][0])
        np.testing.assert_array_equal(back[name][1], quantized[name][1])


@pytest.mark.parametrize("kernel", [pytest.param("fused", marks=needs_numba), "blocked"])
@pytest.mark.parametrize("n", [1, 4, 20])
def test_int4_kernels_match_float64(kernel, n):
    rng = np.random.default_rng(3)
    q, s = quantize_groupwise_int4(rng.normal(0, 0.05, (300, 128)).astype(np.float32), 32)
    packed = pack_int4(q)
    x = rng.normal(0, 1, (n, 128)).astype(np.float32)
    bias = rng.normal(0, 1, 300).astype(np.float32)
    exact = x.astype(np.float64) @ dequantize_groupwise_int4(packed, s, 32).astype(np.float64).T
    y = int4_linear(x, packed, s, 32, bias, kernel=kernel)
    np.testing.assert_allclose(y, exact + bias, rtol=1e-5, atol=1e-5)


def test_local_reader_matches_safetensors(tmp_path):
    torch = pytest.importorskip("torch")
    st = pytest.importorskip("safetensors.torch")
    from expertrelay.store.safetensors_local import LocalCheckpoint

    t = {"a": torch.randn(3, 4).to(torch.bfloat16), "b": torch.randn(5).to(torch.bfloat16)}
    st.save_file({"a": t["a"]}, str(tmp_path / "s1.safetensors"))
    st.save_file({"b": t["b"]}, str(tmp_path / "s2.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "s1.safetensors", "b": "s2.safetensors"}})
    )
    ck = LocalCheckpoint(tmp_path)
    for name in ("a", "b", "a"):
        np.testing.assert_array_equal(ck.get(name), t[name].float().numpy())


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


@pytest.fixture(scope="module")
def int4_store(tmp_path_factory):
    """A tiny bf16 checkpoint, the int8 store built from it, and the int4
    store built from the bf16 checkpoint (group size 8)."""
    torch = pytest.importorskip("torch")
    st = pytest.importorskip("safetensors.torch")
    from expertrelay.store.build_int4_store import build

    arrays = random_checkpoint(CONFIG, seed=5)
    arrays = {k: torch.from_numpy(v).to(torch.bfloat16).float().numpy() for k, v in arrays.items()}
    ckpt = tmp_path_factory.mktemp("bf16")
    st.save_file(
        {k: torch.from_numpy(v).to(torch.bfloat16) for k, v in arrays.items()}, str(ckpt / "m.safetensors")
    )
    (ckpt / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(arrays, "m.safetensors")})
    )
    int8_dir = tmp_path_factory.mktemp("int8")
    mp = pytest.MonkeyPatch()
    build_test_store(int8_dir, CONFIG, arrays, mp)
    mp.undo()
    out = tmp_path_factory.mktemp("int4")
    result = build(int8_dir, ckpt, out, 8)
    return out, arrays, result


@windows_only
def test_build_writes_verified_records_with_errors(int4_store):
    from expertrelay.store.expert_reader import ExpertStoreReader

    out, arrays, result = int4_store
    assert result["bad"] == []
    assert (out / "resident.bin").exists() and (out / "store.json").exists()
    with ExpertStoreReader(out) as r:
        assert isinstance(r.layout, Int4RecordLayout)
        assert all(r.verify(x.layer, x.expert) for x in r.entries())
        e = r.entries()[0]
        assert 0 < e.quant_error["expert"]["rel_fro_error"] < 0.2
        packed, scales = (
            a.copy() for a in parse_int4_record(r.layout, r.read_raw(e.layer, e.expert))["up_proj"]
        )
        w = arrays[f"model.layers.{e.layer}.mlp.experts.{e.expert}.up_proj.weight"]
        q, s = quantize_groupwise_int4(w, 8)
        np.testing.assert_array_equal(packed, pack_int4(q))  # straight from bf16
        np.testing.assert_array_equal(scales, s)


@windows_only
@pytest.mark.parametrize("kernel", [pytest.param("fused", marks=needs_numba), "blocked"])
def test_int4_store_runs_and_cache_changes_nothing(int4_store, kernel):
    from expertrelay.cache.expert_cache import CachedExpertSource
    from expertrelay.predictor.prefetch_policy import PrefetchPolicy
    from expertrelay.runtime.generate import generate, model_config
    from expertrelay.runtime.qwen_moe import QwenMoe
    from expertrelay.runtime.weights import RamExpertSource, ResidentWeights, UnbufferedExpertSource
    from expertrelay.store.expert_reader import ExpertStoreReader

    out, _, _ = int4_store
    before = int8_linear.current_kernel()
    int8_linear.set_kernel(kernel)
    try:
        prompt = [3, 9, 27, 17, 51]
        ref = QwenMoe(model_config(out), ResidentWeights.load(out), RamExpertSource(out))
        ref_tokens, _ = generate(ref, prompt, 8, 32)
        with ExpertStoreReader(out) as r:
            record = r.layout.record_size
        for source in (
            UnbufferedExpertSource(out),
            CachedExpertSource(out, capacity_bytes=5 * record, io_threads=2, min_free_slots=5),
        ):
            m = QwenMoe(model_config(out), ResidentWeights.load(out), source, prefetcher=PrefetchPolicy(2))
            m.prefetch_adaptive = m.prefill_pipelining = isinstance(source, CachedExpertSource)
            tokens, _ = generate(m, prompt, 8, 32)
            source.close()
            assert tokens == ref_tokens
    finally:
        int8_linear.set_kernel(before)
