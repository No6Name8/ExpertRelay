"""Phase 2 baselines: three ways to get expert weights, same forward code,
same prompts (benchmarks/prompts/phase2.json), greedy decoding.

    python -m expertrelay.bench.phase2_baselines

  A  normal load   read the whole store into RAM (--source ram), with the
                   memory budget deliberately disabled. The store is 14.4 GB
                   and this machine has 8.3 GB, so this is expected to fail;
                   the point is to record HOW.
  B  OS paging     memory-map the whole store (--source mmap) and let Windows
                   page it in and out.
  C  ours          resident weights in RAM, each routed expert read on demand
                   with one unbuffered read and dropped after use, no cache
                   (--source unbuffered).

Each baseline runs in its own subprocess with a timeout. The parent polls
the child's peak memory, because a child that crashes can't report its
own. Results go to benchmarks/results/phase2_baselines.json, and
docs/phase2-baselines.md is generated from that record.

Run order is C, B, A: A may push the whole machine into heavy paging, which
would distort anything measured after it.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from expertrelay import REPO_ROOT
from expertrelay.benchmarking import append_benchmark_record, base_record, process_peak_rss_mb
from expertrelay.manager.profile import collect_machine_profile, device_read_bytes, measure_memory
from expertrelay.paths import BENCHMARK_RESULTS_DIR
from expertrelay.runtime.generate import DEFAULT_RUNTIME_CONFIG, RuntimeConfig

BASELINES = [
    {"id": "C", "name": "ours: unbuffered on-demand reads, no cache", "source": "unbuffered", "budget": True},
    {"id": "B", "name": "OS virtual memory: memory-mapped store", "source": "mmap", "budget": True},
    {"id": "A", "name": "normal load: whole store into RAM", "source": "ram", "budget": False},
]
DEFAULT_TIMEOUTS = {"A": 900, "B": 3600, "C": 3600}
DOC = REPO_ROOT / "docs" / "phase2-baselines.md"


def run_one(baseline: dict, config_path: Path, timeout_s: float) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        out_json = Path(tmp) / "out.json"
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "expertrelay.runtime.generate",
            "--all-prompts",
            "--config",
            str(config_path),
            "--source",
            baseline["source"],
            "--json-out",
            str(out_json),
        ]
        if not baseline["budget"]:
            cmd.append("--no-budget")
        print(f"[{baseline['id']}] {' '.join(cmd[3:])}", flush=True)
        ram_free_before = measure_memory().available_bytes
        dev0, t0 = device_read_bytes(), time.perf_counter()
        stderr_path = Path(tmp) / "stderr.txt"
        with open(stderr_path, "w", encoding="utf-8") as err:
            child = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=err, cwd=REPO_ROOT)
            peak, status = 0.0, None
            while child.poll() is None:
                if time.perf_counter() - t0 > timeout_s:
                    child.kill()
                    child.wait()
                    status = f"timeout after {timeout_s:.0f} s (killed)"
                    break
                sample = process_peak_rss_mb(child.pid)
                peak = max(peak, sample or 0.0)
                time.sleep(0.5)
        wall = time.perf_counter() - t0
        result = {
            **baseline,
            "wall_seconds": wall,
            "returncode": child.returncode,
            "ram_available_before_bytes": ram_free_before,
            "device_bytes_read_total": device_read_bytes() - dev0,
            "peak_rss_mb_polled": peak,
            "stderr_tail": stderr_path.read_text(encoding="utf-8", errors="replace").splitlines()[-15:],
        }
        if status is None:
            status = (
                "ok"
                if child.returncode == 0 and out_json.exists()
                else f"failed (exit code {child.returncode})"
            )
        result["status"] = status
        if status == "ok":
            result["run"] = json.loads(out_json.read_text(encoding="utf-8"))
            result["metrics"] = summarize(result["run"])
        print(f"[{baseline['id']}] {status}, {wall:.0f} s, peak {peak:.0f} MB", flush=True)
        return result


def summarize(run: dict) -> dict:
    prefill = [p["steps"][0] for p in run["prompts"]]
    decode = [s for p in run["prompts"] for s in p["steps"][1:]]
    decode_s = sum(s["total_seconds"] for s in decode)
    read_s = sum(s["expert_read_seconds"] for s in decode)
    return {
        "time_to_first_token_s": {p["id"]: p["steps"][0]["total_seconds"] for p in run["prompts"]},
        "time_to_first_token_mean_s": statistics.fmean(s["total_seconds"] for s in prefill),
        "decode_tokens": len(decode),
        "decode_tokens_per_s": len(decode) / decode_s,
        "decode_expert_read_s_per_token": read_s / len(decode),
        "decode_compute_s_per_token": (decode_s - read_s) / len(decode),
        "decode_read_fraction": read_s / decode_s,
        "decode_device_mb_per_token": statistics.fmean(s["device_bytes"] for s in decode) / 1e6,
        "decode_expert_mb_per_token": statistics.fmean(s["expert_bytes"] for s in decode) / 1e6,
        "peak_rss_mb_self_reported": run["peak_rss_mb"],
        "load_seconds": run["load"]["load_seconds"],
    }


def token_agreement(results: list[dict]) -> dict:
    """Correctness rule on the real store: every baseline that finished must
    have produced exactly the same tokens for every prompt."""
    ok = [r for r in results if r["status"] == "ok"]
    if len(ok) < 2:
        return {"compared": [r["id"] for r in ok], "identical": None}
    ref = ok[0]
    identical = all(
        [p["generated_ids"] for p in r["run"]["prompts"]]
        == [p["generated_ids"] for p in ref["run"]["prompts"]]
        for r in ok[1:]
    )
    return {"compared": [r["id"] for r in ok], "identical": identical}


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def format_markdown(record: dict, json_path: Path) -> str:
    rows = []
    for r in record["results"]:
        m = r.get("metrics")
        if m:
            rows.append(
                f"| {r['id']} | {r['name']} | {m['decode_tokens_per_s']:.2f} | {m['time_to_first_token_mean_s']:.1f} | "
                f"{r['peak_rss_mb_polled']:.0f} | {r['device_bytes_read_total'] / 1e9:.1f} | "
                f"{m['decode_device_mb_per_token']:.0f} | "
                + (
                    f"{m['decode_expert_read_s_per_token']:.2f} / {m['decode_compute_s_per_token']:.2f}"
                    if r["source"] == "unbuffered"
                    else f"n/a / {m['decode_compute_s_per_token']:.2f} (reads inside)"
                )
                + f" | {r['status']} |"
            )
        else:
            rows.append(
                f"| {r['id']} | {r['name']} | - | - | {r['peak_rss_mb_polled']:.0f} | "
                f"{r['device_bytes_read_total'] / 1e9:.1f} | - | - | {r['status']} |"
            )
    failures = [r for r in record["results"] if r["status"] != "ok"]
    ours = next((r for r in record["results"] if r["source"] == "unbuffered" and r.get("run")), None)
    lines = [
        "# Phase 2 baselines",
        "",
        f"Generated by `python -m expertrelay.bench.phase2_baselines` at {record['timestamp']} "
        f"(commit {record['git_commit']}{', dirty tree' if record['git_dirty'] else ''}). Raw data: "
        f"`{_repo_relative(json_path)}`. Do not edit by hand.",
        "",
        f"Model: full Qwen1.5-MoE-A2.7B (24 layers, 60 experts, top-4), int8 store. "
        f"{len(record['config']['prompts'])} prompts, {record['config']['max_new_tokens']} new tokens each, greedy. "
        f"RAM free when the benchmark started: {record['machine']['memory']['available_bytes'] / 1e9:.2f} GB "
        f"of {record['machine']['memory']['total_bytes'] / 1e9:.2f} GB.",
        "",
        "| | Setup | Decode tok/s | First token (s, mean) | Peak RAM (MB) | Disk read, total (GB) | "
        "Disk read per decode token (MB) | Per decode token: disk read / compute (s) | Status |",
        "|---|---|---|---|---|---|---|---|---|",
        *rows,
        "",
        f"Same tokens from every setup that finished: **{record['token_agreement']['identical']}** "
        f"(compared: {', '.join(record['token_agreement']['compared']) or 'none'}).",
        "",
        "With memory mapping (B) the disk reads happen as page faults inside the matrix multiplies, so they "
        "can't be timed apart from compute; its disk-read column shows bytes only.",
    ]
    for r in failures:
        lines += [
            "",
            f"## How {r['id']} ({r['name']}) failed",
            "",
            f"Status: {r['status']}. "
            f"Peak RAM before it ended: {r['peak_rss_mb_polled']:.0f} MB. Last lines of stderr:",
            "",
            "```",
            *r["stderr_tail"],
            "```",
        ]
    if ours:
        lines += ["", "## Sample outputs (setup C)", ""]
        for p in ours["run"]["prompts"]:
            lines += [f"**{p['id']}**: `{json.dumps(p['generated_text'], ensure_ascii=False)}`", ""]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_RUNTIME_CONFIG)
    ap.add_argument("--only", nargs="*", choices=[b["id"] for b in BASELINES], default=None)
    ap.add_argument("--timeout-scale", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=BENCHMARK_RESULTS_DIR / "phase2_baselines.json")
    args = ap.parse_args()

    rt = RuntimeConfig.load(args.config)
    prompts = json.loads(rt.prompts_file.read_text(encoding="utf-8"))["prompts"]
    # Provenance (commit, dirty flag, timestamp, machine) is captured BEFORE
    # the runs: it describes the code and machine state that produced them.
    record = base_record(
        label="phase2 baselines",
        seed=0,
        model={
            "store": str(rt.store_dir.relative_to(REPO_ROOT).as_posix()),
            "source": json.loads((rt.store_dir / "store.json").read_text())["source"]["revision"],
        },
        config={
            "memory_budget_gb": rt.memory_budget_gb,
            "max_seq": rt.max_seq,
            "max_new_tokens": rt.max_new_tokens,
            "prompts": prompts,
            "timeouts_s": DEFAULT_TIMEOUTS,
        },
        machine=collect_machine_profile(measure_disk=False),
    )
    selected = [b for b in BASELINES if args.only is None or b["id"] in args.only]
    results = [run_one(b, args.config, DEFAULT_TIMEOUTS[b["id"]] * args.timeout_scale) for b in selected]
    record["results"] = results
    record["token_agreement"] = token_agreement(results)
    append_benchmark_record(args.out, record)
    DOC.write_text(format_markdown(record, args.out), encoding="utf-8")
    print(f"saved {args.out} and {DOC}")


if __name__ == "__main__":
    main()
