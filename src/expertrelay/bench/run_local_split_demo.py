"""Orchestrates the local two-process split simulation end-to-end:

  1. Launch expertrelay.runtime.expert_server ("Process B") as a subprocess,
     wait for it to report ready.
  2. Run expertrelay.manager.coordinator ("Process A") in a subprocess,
     generating tokens against the real (reduced) checkpoint, with expert
     calls for non-local experts round-tripping over a real TCP socket to
     Process B.
  3. Record a tokens/sec benchmark entry, clearly labeled as a
     single-machine simulation with a reduced expert count.
  4. Tear down Process B.

Both processes talk over 127.0.0.1 here. To run this for real across two
machines: start expertrelay.runtime.expert_server directly on the second
machine (skip step 1 here) and pass --expert-host <that machine's IP> to
expertrelay.manager.coordinator (skip this orchestration script and call
the coordinator directly). See docs/setup-notes.md for the exact list of
what changes.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from expertrelay.paths import BENCHMARK_RESULTS_DIR, DEFAULT_MODEL_DIR


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--port", type=int, default=50051)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--benchmark-out", type=Path, default=BENCHMARK_RESULTS_DIR / "local_split_simulation.json"
    )
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()

    server_cmd = [
        sys.executable,
        "-m",
        "expertrelay.runtime.expert_server",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--model-dir",
        str(args.model_dir),
    ]
    print(f"[demo] starting expert_server: {' '.join(server_cmd)}", flush=True)
    server = subprocess.Popen(
        server_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )

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
            sys.executable,
            "-m",
            "expertrelay.manager.coordinator",
            "--model-dir",
            str(args.model_dir),
            "--expert-host",
            "127.0.0.1",
            "--expert-port",
            str(args.port),
            "--prompt",
            args.prompt,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--seed",
            str(args.seed),
            "--benchmark-out",
            str(args.benchmark_out),
            "--label",
            "single-machine simulation, reduced expert count",
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
