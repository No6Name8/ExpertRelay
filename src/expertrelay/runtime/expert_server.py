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
import socket
import sys
import time
from pathlib import Path

import mindspore as ms

from expertrelay.memory_budget import enforce_ram_budget
from expertrelay.paths import DEFAULT_MODEL_DIR
from expertrelay.runtime.moe_model import RemoteExpertShard
from expertrelay.runtime.net_proto import array_to_payload, payload_to_array, recv_msg, send_msg

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_net(model_dir: Path, manifest: dict, max_ram_gb: float = 6.0) -> RemoteExpertShard:
    cfg = manifest["config"]
    b_info = manifest["process_b"]
    num_layers = cfg["num_layers"]
    expert_ids_per_layer = dict.fromkeys(range(num_layers), b_info["local_expert_ids"])
    net = RemoteExpertShard(cfg["hidden_size"], cfg["moe_intermediate_size"], expert_ids_per_layer)
    ckpt_path = model_dir / b_info["checkpoint"]
    # Checkpoint file size (fp32 on disk) is a reasonable proxy for the RAM
    # loading it will use -- gate BEFORE ms.load_checkpoint, not after.
    enforce_ram_budget(ckpt_path.stat().st_size, max_ram_gb, f"loading {ckpt_path.name}")
    param_dict = ms.load_checkpoint(str(ckpt_path))
    ms.load_param_into_net(net, param_dict)
    return net


def serve(host: str, port: int, model_dir: Path, max_ram_gb: float = 6.0) -> None:
    ms.set_context(device_target="CPU")

    manifest = json.loads((model_dir / "manifest.json").read_text())

    print(
        f"[expert_server] loading {manifest['process_b']['checkpoint']} "
        f"(experts {manifest['process_b']['local_expert_ids']}) ...",
        flush=True,
    )
    net = load_net(model_dir, manifest, max_ram_gb)

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
                        print(
                            f"[expert_server] shutdown requested. "
                            f"served {n_requests} requests, "
                            f"{total_compute_s * 1000:.1f} ms total compute.",
                            flush=True,
                        )
                        return
                    x = payload_to_array(header, payload)
                    t0 = time.perf_counter()
                    out = net.run_expert(header["layer"], header["expert_id"], x)
                    total_compute_s += time.perf_counter() - t0
                    n_requests += 1
                    resp_header, resp_payload = array_to_payload(out)
                    send_msg(conn, resp_header, resp_payload)
            print(
                f"[expert_server] coordinator disconnected ({n_requests} requests served so far)", flush=True
            )
    finally:
        srv.close()


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=50051)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument(
        "--max-ram-gb", type=float, default=6.0, help="refuse to load a checkpoint larger than this"
    )
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    serve(args.host, args.port, args.model_dir, args.max_ram_gb)


if __name__ == "__main__":
    main()
