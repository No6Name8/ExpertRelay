"""Process B: a TCP server that holds ONLY the expert FFNs assigned to it.

This is the actual "device split" -- run this on a separate process (and,
later, a separate machine) that owns a subset of the MoE experts. It never
sees a token id, an embedding, or attention weights; it only ever receives
"here is a hidden-state vector, run expert E of layer L on it" requests and
returns the result.

To move this from a same-machine simulation to a real second device: copy
models/reduced_qwen_moe/{process_b.ckpt,manifest.json} to that machine and
run this same script there with --host 0.0.0.0 (or a specific interface)
and --port. Then point the coordinator at that machine's IP instead of
127.0.0.1. No code changes -- see docs/setup-notes.md.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time

import mindspore as ms

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moe_model import RemoteExpertShard  # noqa: E402
from net_proto import recv_msg, send_msg, array_to_payload, payload_to_array  # noqa: E402


def load_net(model_dir: str, manifest: dict) -> RemoteExpertShard:
    cfg = manifest["config"]
    b_info = manifest["process_b"]
    num_layers = cfg["num_layers"]
    expert_ids_per_layer = {i: b_info["local_expert_ids"] for i in range(num_layers)}
    net = RemoteExpertShard(cfg["hidden_size"], cfg["moe_intermediate_size"], expert_ids_per_layer)
    ckpt_path = os.path.join(model_dir, b_info["checkpoint"])
    param_dict = ms.load_checkpoint(ckpt_path)
    ms.load_param_into_net(net, param_dict)
    return net


def serve(host: str, port: int, model_dir: str):
    ms.set_context(device_target="CPU")

    with open(os.path.join(model_dir, "manifest.json")) as f:
        manifest = json.load(f)

    print(f"[expert_server] loading {manifest['process_b']['checkpoint']} "
          f"(experts {manifest['process_b']['local_expert_ids']}) ...", flush=True)
    net = load_net(model_dir, manifest)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    print(f"[expert_server] listening on {host}:{port}", flush=True)
    print("EXPERT_SERVER_READY", flush=True)

    n_requests = 0
    total_compute_s = 0.0
    try:
        while True:
            conn, addr = srv.accept()
            print(f"[expert_server] coordinator connected from {addr}", flush=True)
            with conn:
                while True:
                    try:
                        header, payload = recv_msg(conn)
                    except ConnectionError:
                        break
                    if header.get("cmd") == "shutdown":
                        print(f"[expert_server] shutdown requested. "
                              f"served {n_requests} requests, "
                              f"{total_compute_s*1000:.1f} ms total compute.", flush=True)
                        return
                    x = payload_to_array(header, payload)
                    t0 = time.perf_counter()
                    out = net.run_expert(header["layer"], header["expert_id"], x)
                    total_compute_s += time.perf_counter() - t0
                    n_requests += 1
                    resp_header, resp_payload = array_to_payload(out)
                    send_msg(conn, resp_header, resp_payload)
            print(f"[expert_server] coordinator disconnected "
                  f"({n_requests} requests served so far)", flush=True)
    finally:
        srv.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=50051)
    ap.add_argument("--model-dir", default="models/reduced_qwen_moe")
    args = ap.parse_args()
    serve(args.host, args.port, args.model_dir)
