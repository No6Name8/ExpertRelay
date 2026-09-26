"""Smoke test for expertrelay.store.fetch_hf_tensors's pure/offline logic.

Deliberately does NOT hit the network: this suite must stay deterministic
and runnable without internet access (see CLAUDE.md's "tests are
deterministic" rule). The network-dependent path (actually fetching real
Qwen1.5-MoE-A2.7B tensors) is exercised manually and documented in
docs/setup-notes.md / docs/limitations.md, not in automated tests.
"""

from __future__ import annotations

import numpy as np

from expertrelay.store import fetch_hf_tensors
from expertrelay.store.fetch_hf_tensors import _bf16_bytes_to_fp32, _resolve_url


def test_bf16_to_fp32_known_values():
    # bf16 is the top 16 bits of fp32. 1.0f32 = 0x3F800000 -> bf16 = 0x3F80.
    # -2.0f32 = 0xC0000000 -> bf16 = 0xC000.
    raw = np.array([0x3F80, 0xC000], dtype="<u2").tobytes()
    out = _bf16_bytes_to_fp32(raw, shape=(2,))
    np.testing.assert_allclose(out, [1.0, -2.0])
    assert out.dtype == np.float32


def test_resolve_url_format():
    url = _resolve_url("Qwen/Qwen1.5-MoE-A2.7B", "model.safetensors.index.json", "main")
    assert url == "https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B/resolve/main/model.safetensors.index.json"


def test_hf_token_is_sent_only_when_set(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert fetch_hf_tensors._auth_headers() == {}
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    assert fetch_hf_tensors._auth_headers() == {"Authorization": "Bearer hf_test"}
