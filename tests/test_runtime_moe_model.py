"""Smoke test for expertrelay.runtime.moe_model: it builds and runs a
forward pass without raising, and shapes come out right. Split-vs-unsplit
numerical correctness is covered separately in test_split_correctness.py.
"""

from __future__ import annotations

import mindspore as ms
import numpy as np

from expertrelay.runtime.moe_model import ReducedQwenMoe

CFG = {
    "vocab_size": 16,
    "hidden_size": 8,
    "num_attention_heads": 2,
    "moe_intermediate_size": 4,
    "shared_expert_intermediate_size": 8,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "num_layers": 1,
    "expert_ids": [0, 1],
    "top_k": 1,
}


def test_forward_pass_shape():
    ms.set_context(device_target="CPU")
    net = ReducedQwenMoe(CFG, local_expert_ids_per_layer=[{0, 1}])
    token_ids = ms.Tensor(np.array([[1, 2, 3, 4, 5]], dtype=np.int32))

    logits = net(token_ids)

    assert logits.shape == (1, 5, CFG["vocab_size"])
    assert np.isfinite(logits.asnumpy()).all()


def test_missing_remote_expert_fn_raises():
    """If a token routes to a non-local expert and no remote_expert_fn is
    wired up, that must fail loudly rather than silently produce wrong output."""
    # Router weights start at zero (see Linear.__init__), so logits tie and
    # argsort's stable ordering always ranks expert 0 first -- making expert 0
    # not-local is what deterministically forces a (missing) remote dispatch.
    ms.set_context(device_target="CPU")
    net = ReducedQwenMoe(CFG, local_expert_ids_per_layer=[{1}])  # expert 0 is not local, no remote fn set
    token_ids = ms.Tensor(np.array([[1, 2, 3]], dtype=np.int32))

    try:
        net(token_ids)
    except RuntimeError as e:
        assert "not local" in str(e)
    else:
        raise AssertionError("expected RuntimeError for an unreachable expert")
