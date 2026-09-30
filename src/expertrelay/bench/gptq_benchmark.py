"""Step B3 speed: int8 vs GPTQ int4 experts on the Chat model, same
session, same settings, resumable.

    python -m expertrelay.bench.gptq_benchmark

Phase 2 prompts in the Chat template (4 x 32 new tokens, greedy), fused
kernel, 2 I/O threads, pipelined prefill. For each precision:

  nocache   one read per expert use, no cache: the reference for the
            token-identity check within the precision
  cache     1.25 GB cache, no prediction
  adaptive  1.25 GB cache + adaptive prefetch (threshold 0.05)

Same cache size in GB for both precisions. Each precision's prefetch uses
the calibration fitted on its OWN store's traces: int8 Chat's
(configs/runtime_cache_chat.json), and GPTQ's from a short trace of its
own (configs/runtime_cache_chat_gptq.json; bench.phase3_traces
--prompts-per-category 4, then bench.fit_prefetch_calibration). Every
setup runs twice (the second round in reverse order). Read logs are
summarized; the full logs are not kept.

Resumable: each run's result is saved under models/work/gptq_speed/ the
moment it finishes, and a rerun does only the runs not yet saved. Every
start of this script is a "session", recorded with its machine profile and
free RAM, and every run records the session it ran in, so a speed run
finished in a second clean session says so in the results. A failed run is
kept (as a failed attempt) and redone. When all runs are saved, the record
goes to benchmarks/results/gptq_benchmark.json.

Run it on a clean machine: browsers and VS Code closed, nothing else heavy
running (no downloads, traces or builds).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

from expertrelay import REPO_ROOT
from expertrelay.bench.int4_benchmark import agreement, disk_per_token
from expertrelay.bench.phase2_baselines import run_one
from expertrelay.bench.phase4_benchmark import cache_metrics, fit_cache
from expertrelay.bench.stepb_benchmark import read_log_summary
from expertrelay.benchmarking import append_benchmark_record, base_record
from expertrelay.manager.profile import collect_machine_profile
from expertrelay.paths import BENCHMARK_RESULTS_DIR, MODELS_ROOT
from expertrelay.runtime.generate import DEFAULT_RUNTIME_CONFIG, RuntimeConfig

CACHE_GB = 1.25
TIMEOUT_S = 3600
ROUNDS = 2
PRECISIONS = {
    "int8": ("qwen1.5-moe-a2.7b-chat-int8", REPO_ROOT / "configs" / "runtime_cache_chat.json"),
    "gptq": ("qwen1.5-moe-a2.7b-chat-int4g128-gptq", REPO_ROOT / "configs" / "runtime_cache_chat_gptq.json"),
}
WORK = MODELS_ROOT / "work" / "gptq_speed"
OUT = BENCHMARK_RESULTS_DIR / "gptq_benchmark.json"


def setups() -> list[dict]:
    common = ["--int8-kernel", "fused", "--log-reads"]
    cached = [*common, "--io-threads", "2", "--prefill-pipelining", "--no-pin-layer0"]
    out = []
    for precision, (store, config) in PRECISIONS.items():
        sd = ["--store-dir", str(MODELS_ROOT / store)]
        out += [
            {"id": f"{precision}_nocache", "precision": precision, "name": f"{precision}, no cache",
             "source": "unbuffered", "budget": True, "cache": False, "store": store,
             "config": DEFAULT_RUNTIME_CONFIG, "args": [*common, *sd]},
            {"id": f"{precision}_cache", "precision": precision, "name": f"{precision}, cache {CACHE_GB} GB",
             "source": "cached", "budget": True, "cache": True, "store": store, "config": config,
             "args": [*cached, *sd, "--no-prefetch"]},
            {"id": f"{precision}_adaptive", "precision": precision,
             "name": f"{precision}, cache {CACHE_GB} GB + adaptive prefetch",
             "source": "cached", "budget": True, "cache": True, "store": store, "config": config,
             "args": [*cached, *sd, "--prefetch", "--prefetch-adaptive", "--prefetch-min-probability", "0.05"]},
        ]  # fmt: skip
    return out


def plan() -> list[tuple[int, dict]]:
    s = setups()
    return [(rnd, x) for rnd in range(ROUNDS) for x in (s if rnd % 2 == 0 else s[::-1])]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def settings() -> dict:
    """What the saved runs depend on; a mismatch means they can't be combined with new ones."""
    out = {"cache_gb": CACHE_GB, "rounds": ROUNDS, "setups": [_portable(s) for s in setups()]}
    for precision, (store, config) in PRECISIONS.items():
        calibration = RuntimeConfig.load(config).prefetch_calibration
        out[precision] = {
            "experts_index_sha256": _sha256(MODELS_ROOT / store / "experts_index.json"),
            "config_sha256": _sha256(config),
            "calibration_sha256": _sha256(calibration),
        }
    return out


def _portable(s: dict) -> dict:
    def conv(v):
        if isinstance(v, Path):
            return v.relative_to(REPO_ROOT).as_posix()
        if isinstance(v, list):
            return [conv(x) for x in v]
        if isinstance(v, str) and v.startswith(str(REPO_ROOT)):
            return Path(v).relative_to(REPO_ROOT).as_posix()
        return v

    return {k: conv(v) for k, v in s.items()}


def run_path(setup_id: str, rnd: int) -> Path:
    return WORK / "runs" / f"{setup_id}#{rnd}.json"


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj), encoding="utf-8")
    os.replace(tmp, path)


def start_session(current: dict) -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    saved = WORK / "settings.json"
    if saved.exists():
        if json.loads(saved.read_text()) != current:
            raise SystemExit(f"{saved} is from other stores/configs; move {WORK} away to start over")
    else:
        saved.write_text(json.dumps(current, indent=1))
    sessions = sorted(WORK.glob("session_*.json"))
    index = len(sessions)
    machine = collect_machine_profile(measure_disk=False)
    _write_json(
        WORK / f"session_{index}.json",
        base_record(label=f"gptq speed session {index}", seed=0, model=None, config={}, machine=machine),
    )
    print(f"session {index}: {machine.memory.available_bytes / 1e9:.2f} GB RAM free", flush=True)
    return index


def main() -> None:
    current = settings()
    session = start_session(current)
    for rnd, s in plan():
        path = run_path(s["id"], rnd)
        if path.exists() and json.loads(path.read_text(encoding="utf-8"))["status"] == "ok":
            continue
        if path.exists():  # a failed attempt: keep it, then redo the run
            failed = WORK / "failed" / f"{path.stem}.session{session}.json"
            failed.parent.mkdir(parents=True, exist_ok=True)
            os.replace(path, failed)
        s = dict(s)
        fit = None
        if s["cache"]:
            fit = fit_cache(
                replace(RuntimeConfig.load(s["config"]), store_dir=MODELS_ROOT / s["store"]), CACHE_GB
            )
            s["args"] = [*s["args"], "--cache-gb", f"{fit['used_gb']:.9f}"]
        r = run_one(s, s["config"], TIMEOUT_S)
        r = {**_portable(r), "round": rnd, "session": session, "cache_fit": fit}
        if r["status"] == "ok":
            r["cache"] = cache_metrics(r["run"])
            r["disk"] = disk_per_token(r["run"])
            r["read_log_summary"] = read_log_summary(r["run"])
            r["run"]["read_log"] = None  # summarized above; the full log would make the file huge
        _write_json(path, r)

    results = [json.loads(run_path(s["id"], rnd).read_text(encoding="utf-8")) for rnd, s in plan()]
    if any(r["status"] != "ok" for r in results):
        raise SystemExit("some runs failed; rerun this script in a clean session to redo them")
    sessions = [json.loads(p.read_text()) for p in sorted(WORK.glob("session_*.json"))]
    failed = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((WORK / "failed").glob("*.json"))]
    record = base_record(
        label="int8 vs GPTQ int4 experts, Chat model: speed",
        seed=0,
        model={p: {"store": st, "config": _portable({"c": c})["c"]} for p, (st, c) in PRECISIONS.items()},
        config={
            "prompts": json.loads(
                RuntimeConfig.load(DEFAULT_RUNTIME_CONFIG).prompts_file.read_text(encoding="utf-8")
            )["prompts"],
            "prompt_format": "chat_template",
            "cache_gb": CACHE_GB,
            "rounds": ROUNDS,
            "setups": current["setups"],
            "work_settings": current,
        },
        machine=collect_machine_profile(measure_disk=False),
    )
    record["sessions"] = [
        {
            "index": i,
            "timestamp": x["timestamp"],
            "git_commit": x["git_commit"],
            "git_dirty": x["git_dirty"],
            "ram_available_bytes": x["machine"]["memory"]["available_bytes"],
            "runs": [f"{r['id']}#{r['round']}" for r in results if r["session"] == i],
            "machine": x["machine"],
        }
        for i, x in enumerate(sessions)
    ]
    record["interrupted"] = len({r["session"] for r in results}) > 1
    record["failed_attempts"] = [
        {k: f.get(k) for k in ("id", "round", "session", "status", "stderr_tail")} for f in failed
    ]
    record["results"] = results
    record["agreement"] = agreement(results)
    append_benchmark_record(OUT, record)
    print(f"saved {OUT}")
    print(json.dumps(record["agreement"], indent=1))


if __name__ == "__main__":
    main()
