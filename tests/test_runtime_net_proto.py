"""Smoke test for expertrelay.runtime.net_proto: a real loopback TCP round
trip must reproduce the sent array exactly, byte for byte.
"""

from __future__ import annotations

import socket
import threading

import numpy as np

from expertrelay.runtime.net_proto import array_to_payload, payload_to_array, recv_msg, send_msg


def test_array_payload_roundtrip():
    arr = np.arange(12, dtype=np.float32).reshape(3, 4) * 1.5 - 3
    header, payload = array_to_payload(arr)
    out = payload_to_array(header, payload)
    np.testing.assert_array_equal(arr, out)


def test_send_recv_over_real_tcp_socket():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    received = {}

    def run_server():
        conn, _ = server.accept()
        with conn:
            header, payload = recv_msg(conn)
            received["header"] = header
            received["payload"] = payload
            echo_header, echo_payload = array_to_payload(payload_to_array(header, payload) * 2)
            send_msg(conn, echo_header, echo_payload)

    t = threading.Thread(target=run_server, daemon=True)
    t.start()

    client = socket.create_connection(("127.0.0.1", port), timeout=5)
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    header, payload = array_to_payload(arr)
    header["cmd"] = "run_expert"
    header["layer"] = 0
    header["expert_id"] = 2
    send_msg(client, header, payload)

    resp_header, resp_payload = recv_msg(client)
    result = payload_to_array(resp_header, resp_payload)

    t.join(timeout=5)
    client.close()
    server.close()

    np.testing.assert_array_equal(result, arr * 2)
    assert received["header"]["layer"] == 0
    assert received["header"]["expert_id"] == 2
