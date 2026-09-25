"""Process A: the coordinator (today's minimal Manager).

Holds the embedding, attention, norms, shared expert, router, and lm_head --
plus whichever routed experts are assigned to it locally. For any token
whose router picks an expert NOT held locally, it opens (well, reuses) a
TCP connection to the expert host ("Process B") and asks it to run that
expert, exactly like the real distributed-inference plan this repo is
building towards.

Run expert_server first, then this script pointed at its --host/--port.
To point Process B at a real second machine later: change --expert-host to
that machine's LAN/VPN IP and make sure --port is reachable there -- nothing
else in this file changes. See docs/setup-notes.md.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

import mindspore as ms
import numpy as np
import tokenizers

from expertrelay.benchmarking import append_benchmark_record, base_record, peak_process_rss_mb
from expertrelay.memory_budget import enforce_ram_budget
from expertrelay.paths import DEFAULT_MODEL_DIR
from expertrelay.runtime.moe_model import ReducedQwenMoe
from expertrelay.runtime.net_proto import array_to_payload, payload_to_array, recv_msg, send_msg

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_net(model_dir: Path, manifest: dict, max_ram_gb: float = 6.0) -> ReducedQwenMoe:
    cfg = manifest["config"]
    a_info = manifest["process_a"]
    local_ids = set(a_info["local_expert_ids"])
    net = ReducedQwenMoe(cfg, local_expert_ids_per_layer=[local_ids] * cfg["num_layers"])
    ckpt_path = model_dir / a_info["checkpoint"]
    enforce_ram_budget(ckpt_path.stat().st_size, max_ram_gb, f"loading {ckpt_path.name}")
    param_dict = ms.load_checkpoint(str(ckpt_path))
    ms.load_param_into_net(net, param_dict)
    return net


class RemoteExpertClient:
    """Wraps the TCP connection to the expert host; tracks call count/latency."""

    def __init__(self, host: str, port: int):
        self.sock = socket.create_connection((host, port), timeout=30)
        self.calls = 0
        self.wire_seconds = 0.0

    def __call__(self, layer_idx: int, expert_id: int, x_np: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        header, payload = array_to_payload(x_np)
        header.update({"cmd": "run_expert", "layer": layer_idx, "expert_id": expert_id})
        send_msg(self.sock, header, payload)
        resp_header, resp_payload = recv_msg(self.sock)
        out = payload_to_array(resp_header, resp_payload)
        self.wire_seconds += time.perf_counter() - t0
        self.calls += 1
        return out

    def shutdown(self) -> None:
        send_msg(self.sock, {"cmd": "shutdown"})
        self.sock.close()


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--tokenizer", type=Path, default=None, help="defaults to <model-dir>/tokenizer.json")
    ap.add_argument("--expert-host", default="127.0.0.1")
    ap.add_argument("--expert-port", type=int, default=50051)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="recorded in the benchmark; greedy decoding has no other randomness",
    )
    ap.add_argument(
        "--benchmark-out",
        type=Path,
        default=None,
        help="if set, append a JSON record of this run's timing here (e.g. benchmarks/results/foo.json)",
    )
    ap.add_argument("--label", default="local-two-process-simulation")
    ap.add_argument(
        "--max-ram-gb", type=float, default=6.0, help="refuse to load a checkpoint larger than this"
    )
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model_dir / "tokenizer.json"

    ms.set_context(device_target="CPU")
    ms.set_seed(args.seed)

    manifest = json.loads((args.model_dir / "manifest.json").read_text())

    print("[coordinator] loading process_a checkpoint ...", flush=True)
    net = load_net(args.model_dir, manifest, args.max_ram_gb)

    print(f"[coordinator] connecting to expert host {args.expert_host}:{args.expert_port} ...", flush=True)
    remote = RemoteExpertClient(args.expert_host, args.expert_port)
    net.set_remote_expert_fn(remote)

    tok = tokenizers.Tokenizer.from_file(str(args.tokenizer))
    prompt_ids = tok.encode(args.prompt).ids
    print(f"[coordinator] prompt: {args.prompt!r} -> {prompt_ids}", flush=True)

    generated = list(prompt_ids)
    t_start = time.perf_counter()
    for _ in range(args.max_new_tokens):
        token_ids = ms.Tensor(np.array([generated], dtype=np.int32))
        logits = net(token_ids)
        next_id = int(np.argmax(logits.asnumpy()[0, -1, :]))
        generated.append(next_id)
    elapsed = time.perf_counter() - t_start

    new_tokens = len(generated) - len(prompt_ids)
    tokens_per_sec = new_tokens / elapsed if elapsed > 0 else float("nan")

    text = tok.decode(generated)
    print(f"[coordinator] generated ids: {generated}")
    print(f"[coordinator] decoded text: {text!r}")
    print(f"[coordinator] {new_tokens} new tokens in {elapsed:.3f}s -> {tokens_per_sec:.3f} tok/s")
    print(
        f"[coordinator] remote (Process B) calls: {remote.calls}, "
        f"total wire time {remote.wire_seconds * 1000:.1f} ms "
        f"({remote.wire_seconds / elapsed * 100:.1f}% of wall time)"
    )
    for s in net.call_stats():
        print(
            f"[coordinator]   layer {s['layer']}: local_calls={s['local_calls']} remote_calls={s['remote_calls']}"
        )

    if args.benchmark_out:
        record = base_record(
            label=args.label,
            seed=args.seed,
            model={"source_repo": manifest["source_repo"], "revision": manifest["revision"]},
            config=manifest["config"],
            prompt=args.prompt,
            new_tokens=new_tokens,
            elapsed_seconds=elapsed,
            tokens_per_second=tokens_per_sec,
            remote_calls=remote.calls,
            remote_wire_seconds=remote.wire_seconds,
            per_layer_calls=net.call_stats(),
            process_a_experts=manifest["process_a"]["local_expert_ids"],
            process_b_experts=manifest["process_b"]["local_expert_ids"],
            peak_rss_mb=peak_process_rss_mb(),
        )
        append_benchmark_record(args.benchmark_out, record)
        print(f"[coordinator] appended benchmark record to {args.benchmark_out}")

    remote.shutdown()


if __name__ == "__main__":
    main()
