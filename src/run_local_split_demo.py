"""Orchestrates the local two-process split simulation end-to-end:

  1. Launch expert_server.py ("Process B") as a subprocess, wait for it to
     report ready.
  2. Run coordinator.py ("Process A") in this process, generating tokens
     against the real (reduced) checkpoint, with expert calls for
     non-local experts round-tripping over a real TCP socket to Process B.
  3. Record a tokens/sec benchmark entry, clearly labeled as a
     single-machine simulation with a reduced expert count.
  4. Tear down Process B.

Both processes talk over 127.0.0.1 here. To run this for real across two
machines: start expert_server.py directly on the second machine (skip step
1 here) and pass --expert-host <that machine's IP> to coordinator.py (skip
this orchestration script and call coordinator.py directly). See
docs/setup-notes.md for the exact list of what changes.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default="models/reduced_qwen_moe")
    ap.add_argument("--port", type=int, default=50051)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument("--benchmark-out", default="../benchmarks/local_split_simulation.json")
    args = ap.parse_args()

    server_cmd = [
        sys.executable, "expert_server.py",
        "--host", "127.0.0.1", "--port", str(args.port),
        "--model-dir", args.model_dir,
    ]
    print(f"[demo] starting expert_server: {' '.join(server_cmd)}", flush=True)
    server = subprocess.Popen(server_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    try:
        ready = False
        for line in server.stdout:
            print(f"[expert_server] {line}", end="", flush=True)
            if "EXPERT_SERVER_READY" in line:
                ready = True
                break
        if not ready:
            raise RuntimeError("expert_server exited before signaling ready")

        coord_cmd = [
            sys.executable, "coordinator.py",
            "--model-dir", args.model_dir,
            "--expert-host", "127.0.0.1", "--expert-port", str(args.port),
            "--prompt", args.prompt,
            "--max-new-tokens", str(args.max_new_tokens),
            "--benchmark-out", args.benchmark_out,
            "--label", "single-machine simulation, reduced expert count",
        ]
        print(f"[demo] starting coordinator: {' '.join(coord_cmd)}", flush=True)
        result = subprocess.run(coord_cmd)
        if result.returncode != 0:
            raise RuntimeError(f"coordinator exited with code {result.returncode}")
    finally:
        print("[demo] tearing down expert_server ...", flush=True)
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
        # drain any remaining output
        if server.stdout:
            for line in server.stdout:
                print(f"[expert_server] {line}", end="", flush=True)

    print("[demo] done.", flush=True)


if __name__ == "__main__":
    main()
