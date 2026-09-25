"""Smoke test for expertrelay.runtime.expert_server: build a tiny checkpoint
+ manifest on disk (standing in for a real converted one), load it back with
load_net(), and confirm run_expert() produces correctly-shaped, correct output.
"""

from __future__ import annotations

import json
from pathlib import Path

import mindspore as ms
import numpy as np

from expertrelay.runtime.expert_server import load_net
from expertrelay.runtime.moe_model import RemoteExpertShard


def test_load_net_and_run_expert(tmp_path: Path):
    ms.set_context(device_target="CPU")
    hidden_size, ffn_size = 6, 3

    built = RemoteExpertShard(hidden_size, ffn_size, expert_ids_per_layer={0: [5, 7]})
    for _name, param in built.parameters_and_names():
        param.set_data(ms.Tensor(np.random.RandomState(0).uniform(-0.1, 0.1, param.shape).astype(np.float32)))

    ckpt_path = tmp_path / "process_b.ckpt"
    ms.save_checkpoint(built, str(ckpt_path))

    manifest = {
        "source_repo": "test/tiny",
        "revision": "main",
        "config": {
            "hidden_size": hidden_size,
            "moe_intermediate_size": ffn_size,
            "num_layers": 1,
        },
        "process_b": {"checkpoint": "process_b.ckpt", "local_expert_ids": [5, 7]},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    net = load_net(tmp_path, manifest)

    x = np.random.RandomState(1).uniform(-1, 1, (1, hidden_size)).astype(np.float32)
    out = net.run_expert(0, 5, x)

    assert out.shape == (1, hidden_size)
    assert np.isfinite(out).all()

    # Loading must have actually taken effect, not left the shard at its
    # zero-initialized default -- rerunning the SAME weights directly should
    # match exactly.
    expected = built.shard_layers[0].experts[5](ms.Tensor(x, ms.float32)).asnumpy()
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)
