"""Smoke tests for expertrelay.manager.coordinator: checkpoint loading
round-trips correctly, and RemoteExpertClient speaks net_proto correctly
against a real (if trivial) TCP server.
"""

from __future__ import annotations

import socket
import threading
from pathlib import Path

import mindspore as ms
import numpy as np

from expertrelay.manager.coordinator import RemoteExpertClient, load_net
from expertrelay.runtime.moe_model import ReducedQwenMoe
from expertrelay.runtime.net_proto import array_to_payload, payload_to_array, recv_msg, send_msg

CFG = {
    "vocab_size": 10,
    "hidden_size": 8,  # must give an even head_dim (hidden_size / num_attention_heads) for RoPE's rotate_half
    "num_attention_heads": 2,
    "moe_intermediate_size": 3,
    "shared_expert_intermediate_size": 6,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "num_layers": 1,
    "expert_ids": [0, 1],
    "top_k": 1,
}


def test_load_net_roundtrip(tmp_path: Path):
    ms.set_context(device_target="CPU")
    built = ReducedQwenMoe(CFG, local_expert_ids_per_layer=[{0, 1}])
    for _name, param in built.parameters_and_names():
        param.set_data(ms.Tensor(np.random.RandomState(0).uniform(-0.1, 0.1, param.shape).astype(np.float32)))

    ckpt_path = tmp_path / "process_a.ckpt"
    ms.save_checkpoint(built, str(ckpt_path))

    manifest = {
        "source_repo": "test/tiny",
        "revision": "main",
        "config": CFG,
        "process_a": {"checkpoint": "process_a.ckpt", "local_expert_ids": [0, 1]},
        "process_b": {"checkpoint": "process_b.ckpt", "local_expert_ids": []},
    }

    net = load_net(tmp_path, manifest)
    token_ids = ms.Tensor(np.array([[1, 2, 3]], dtype=np.int32))

    expected = built(token_ids).asnumpy()
    actual = net(token_ids).asnumpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_remote_expert_client_speaks_net_proto():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    seen_cmds = []

    def run_server():
        conn, _ = server.accept()
        with conn:
            while True:
                header, payload = recv_msg(conn)
                seen_cmds.append(header.get("cmd"))
                if header.get("cmd") == "shutdown":
                    return
                x = payload_to_array(header, payload)
                resp_header, resp_payload = array_to_payload(x + 100)
                send_msg(conn, resp_header, resp_payload)

    t = threading.Thread(target=run_server, daemon=True)
    t.start()

    client = RemoteExpertClient("127.0.0.1", port)
    out = client(0, 3, np.array([[1.0, 2.0]], dtype=np.float32))
    np.testing.assert_array_equal(out, np.array([[101.0, 102.0]], dtype=np.float32))
    assert client.calls == 1
    assert client.wire_seconds > 0

    client.shutdown()
    t.join(timeout=5)
    server.close()

    assert seen_cmds == ["run_expert", "shutdown"]
