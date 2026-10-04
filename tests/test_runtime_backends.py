"""Compute backends (runtime.backends): numpy and MindSpore f32.

- Every primitive of every backend matches float64 math.
- Both backends generate the same greedy tokens on a test model built
  through the real store builder, with logits within 1e-4.
- The Manager picks MindSpore only when an Ascend NPU is present.
- The store, expert sources, cache, predictor and Manager never import a
  compute backend or MindSpore: they must not depend on which one runs.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest
from store_helpers import build_test_store, random_checkpoint

from expertrelay.manager.backend_selection import select_backend
from expertrelay.manager.profile import MachineProfile, detect_accelerators
from expertrelay.runtime.backends import BACKEND_NAMES, make_backend
from expertrelay.runtime.generate import generate, model_config
from expertrelay.runtime.qwen_moe import KVCache, QwenMoe
from expertrelay.runtime.weights import RamExpertSource, ResidentWeights
from expertrelay.store.quantize import quantize_rowwise_int8

pytest.importorskip("mindspore")


@pytest.fixture(scope="module", params=BACKEND_NAMES)
def backend(request):
    return make_backend(request.param)


def test_int8_linear_matches_float64(backend):
    rng = np.random.default_rng(0)
    q, s = quantize_rowwise_int8(rng.normal(0, 0.05, (300, 96)).astype(np.float32))
    x = rng.normal(0, 1, (5, 96)).astype(np.float32)
    bias = rng.normal(0, 1, 300).astype(np.float32)
    exact = x.astype(np.float64) @ (q.astype(np.float64) * s[:, None]).T + bias
    np.testing.assert_allclose(backend.int8_linear(x, q, s, bias), exact, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(backend.int8_linear(x, q, s), exact - bias, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("n", [1, 5])  # decode-sized and prefill-sized inputs
def test_int4_linear_matches_float64(backend, n):
    from expertrelay.store.int4 import dequantize_groupwise_int4, pack_int4, quantize_groupwise_int4

    rng = np.random.default_rng(1)
    q, s = quantize_groupwise_int4(rng.normal(0, 0.05, (300, 96)).astype(np.float32), 32)
    packed = pack_int4(q)
    x = rng.normal(0, 1, (n, 96)).astype(np.float32)
    exact = x.astype(np.float64) @ dequantize_groupwise_int4(packed, s, 32).astype(np.float64).T
    np.testing.assert_allclose(backend.int4_linear(x, packed, s, 32), exact, rtol=1e-5, atol=1e-5)


def test_linear_matches_float64(backend):
    rng = np.random.default_rng(1)
    w = rng.normal(0, 0.1, (7, 32)).astype(np.float32)
    x = rng.normal(0, 1, (4, 32)).astype(np.float32)
    np.testing.assert_allclose(backend.linear(x, w), x.astype(np.float64) @ w.T, rtol=1e-5, atol=1e-5)


def test_attention_matches_float64_with_causal_mask(backend):
    # 3 new queries at positions 5..7 over 8 cached keys: a prefill chunk after a cache
    rng = np.random.default_rng(2)
    n, k, heads, d = 3, 8, 4, 16
    q, keys, values = (rng.normal(0, 1, (m, heads, d)).astype(np.float32) for m in (n, k, k))
    allowed = np.arange(k)[None, :] <= (5 + np.arange(n))[:, None]

    scores = np.einsum("qhd,khd->hqk", q.astype(np.float64), keys) / np.sqrt(d)
    scores = np.where(allowed[None], scores, -np.inf)
    probs = np.exp(scores - scores.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    exact = np.einsum("hqk,khd->qhd", probs, values)

    out = backend.attention(q, keys, values, allowed)
    assert out.shape == (n, heads, d) and out.dtype == np.float32
    np.testing.assert_allclose(out, exact, rtol=1e-5, atol=1e-5)


CONFIG = {
    "vocab_size": 64,
    "hidden_size": 32,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 16,
    "shared_expert_intermediate_size": 24,
    "norm_topk_prob": False,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
}


@pytest.mark.skipif(sys.platform != "win32", reason="store finalize verifies via unbuffered reads")
def test_both_backends_generate_the_same_tokens(tmp_path, monkeypatch):
    build_test_store(tmp_path, CONFIG, random_checkpoint(CONFIG, seed=4), monkeypatch)
    prompt = [3, 9, 27, 17, 51, 8]
    runs = {}
    for name in BACKEND_NAMES:
        model = QwenMoe(
            model_config(tmp_path),
            ResidentWeights.load(tmp_path),
            RamExpertSource(tmp_path),
            make_backend(name),
        )
        logits, _ = model.forward(np.array(prompt), KVCache(model.c, 32), all_logits=True)
        tokens, _ = generate(model, prompt, max_new_tokens=10, max_seq=32)
        runs[name] = (logits, tokens)

    (np_logits, np_tokens), (ms_logits, ms_tokens) = runs["numpy"], runs["mindspore"]
    assert ms_tokens == np_tokens
    # different matmul kernels, so close rather than bit-identical
    np.testing.assert_allclose(ms_logits, np_logits, rtol=1e-4, atol=1e-4)


def test_manager_picks_mindspore_only_with_ascend(fake_profile: MachineProfile):
    cpu = select_backend(fake_profile)
    assert (cpu.name, cpu.device) == ("numpy", None)
    npu = select_backend(replace(fake_profile, accelerators=["ascend"]))
    assert (npu.name, npu.device) == ("mindspore", "Ascend")
    assert cpu.reason and npu.reason


def test_ascend_detection():
    assert detect_accelerators(env={}, exists=lambda p: False) == []
    assert detect_accelerators(env={"ASCEND_HOME_PATH": "/opt/ascend"}, exists=lambda p: False) == ["ascend"]
    assert detect_accelerators(env={}, exists=lambda p: p == "/dev/davinci0") == ["ascend"]


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError):
        make_backend("tpu")


def test_backend_independent_modules_never_load_a_backend():
    """Imported in a fresh interpreter: these modules must not pull in
    runtime.backends, either backend, or MindSpore."""
    modules = [
        "expertrelay.store.build_store",
        "expertrelay.store.expert_reader",
        "expertrelay.runtime.weights",
        "expertrelay.cache",
        "expertrelay.cache.expert_cache",
        "expertrelay.cache.predictive",
        "expertrelay.predictor",
        "expertrelay.predictor.offline",
        "expertrelay.predictor.prefetch_policy",
        "expertrelay.manager.profile",
        "expertrelay.manager.backend_selection",
        "expertrelay.manager.probe",
        "expertrelay.manager.policy",
    ]
    code = (
        "import sys\n"
        + "".join(f"import {m}\n" for m in modules)
        + "bad = [m for m in ('expertrelay.runtime.backends', 'expertrelay.runtime.backend_mindspore', "
        "'mindspore') if m in sys.modules]\n"
        "print(','.join(bad))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "", f"backend-independent modules loaded: {out.stdout.strip()}"


def test_numpy_backend_does_not_load_mindspore():
    code = (
        "import sys\n"
        "from expertrelay.runtime.backends import make_backend\n"
        "make_backend('numpy')\n"
        "print('mindspore' in sys.modules)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
