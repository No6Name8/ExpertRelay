"""Phase 6 validation: does --auto pick the best configuration?

    python -m expertrelay.bench.manager_validation

Chat model (int8), Phase 2 prompts in the Chat template (4 x 32 new
tokens, greedy), two compute regimes:
  fast  the fused int8 kernel (~0.1 s of compute per token)
  slow  the old blocked numpy kernel (~1 s of compute per token)
In each regime, --auto against every hand-set configuration:
  nocache   one read per expert use
  cache     1.25 GB cache, no prediction
  adaptive  1.25 GB cache + adaptive prefetch (threshold 0.05)
  top8      1.25 GB cache + fixed top-8 prefetch (no threshold)
Hand-set runs use 2 I/O threads, pipelined prefill and no layer-0 pinning
(the settings of the Step B / B3 benchmarks). Every setup runs twice, the
second round in reverse order.

Scoring, fixed before any run:
  - "picked the best": --auto's prefetch mode (no cache / cache only /
    adaptive / top-8) is the mode of the hand-set configuration with the
    highest mean decode tokens/s in that regime;
  - "how close": --auto's mean decode tokens/s / that best configuration's.
  - tokens: every run must produce the same tokens as the regime's
    no-cache run.

Resumable, like bench.gptq_benchmark: each run is saved under
models/work/manager_validation/ the moment it finishes, a rerun does only
the unfinished runs, every start is a recorded session. When all runs are
saved, the record goes to benchmarks/results/manager_validation.json.

Run on a clean machine: browsers and VS Code closed, nothing else heavy.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
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
from expertrelay.runtime.generate import RuntimeConfig

STORE = "qwen1.5-moe-a2.7b-chat-int8"
CONFIG = REPO_ROOT / "configs" / "runtime_cache_chat.json"
CACHE_GB = 1.25
TIMEOUT_S = 3600
ROUNDS = 2
REGIMES = {"fast": "fused", "slow": "blocked"}
WORK = MODELS_ROOT / "work" / "manager_validation"
OUT = BENCHMARK_RESULTS_DIR / "manager_validation.json"
# hand-set setup -> its prefetch mode, in the Manager's terms
MODES = {"nocache": "no cache", "cache": "cache only", "adaptive": "adaptive", "top8": "top-8"}


def setups() -> list[dict]:
    out = []
    for regime, kernel in REGIMES.items():
        common = ["--int8-kernel", kernel, "--log-reads", "--store-dir", str(MODELS_ROOT / STORE)]
        cached = [*common, "--io-threads", "2", "--prefill-pipelining", "--no-pin-layer0"]
        base = {"precision": regime, "regime": regime, "budget": True}
        out += [
            {**base, "id": f"{regime}_auto", "name": "--auto", "source": "cached", "cache": False, "args": [*common, "--auto"]},
            {**base, "id": f"{regime}_nocache", "name": "no cache", "source": "unbuffered", "cache": False, "args": common},
            {**base, "id": f"{regime}_cache", "name": f"cache {CACHE_GB} GB", "source": "cached", "cache": True,
             "args": [*cached, "--no-prefetch"]},
            {**base, "id": f"{regime}_adaptive", "name": f"cache {CACHE_GB} GB + adaptive prefetch", "source": "cached",
             "cache": True, "args": [*cached, "--prefetch", "--prefetch-adaptive", "--prefetch-min-probability", "0.05"]},
            {**base, "id": f"{regime}_top8", "name": f"cache {CACHE_GB} GB + top-8 prefetch", "source": "cached",
             "cache": True, "args": [*cached, "--prefetch", "--no-prefetch-adaptive", "--prefetch-k", "8",
                                     "--prefetch-min-probability", "0"]},
        ]  # fmt: skip
    return out


def plan() -> list[tuple[int, dict]]:
    out = []
    for regime in REGIMES:
        s = [x for x in setups() if x["regime"] == regime]
        out += [(rnd, x) for rnd in range(ROUNDS) for x in (s if rnd % 2 == 0 else s[::-1])]
    return out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable(v):
    if isinstance(v, Path):
        return v.relative_to(REPO_ROOT).as_posix()
    if isinstance(v, str) and v.startswith(str(REPO_ROOT)):
        return Path(v).relative_to(REPO_ROOT).as_posix()
    if isinstance(v, list):
        return [_portable(x) for x in v]
    if isinstance(v, dict):
        return {k: _portable(x) for k, x in v.items()}
    return v


def settings() -> dict:
    """What the saved runs depend on; a mismatch means they can't be combined with new ones."""
    calibration = RuntimeConfig.load(CONFIG).prefetch_calibration
    return {
        "setups": [_portable(s) for s in setups()],
        "rounds": ROUNDS,
        "cache_gb": CACHE_GB,
        "experts_index_sha256": _sha256(MODELS_ROOT / STORE / "experts_index.json"),
        "config_sha256": _sha256(CONFIG),
        "calibration_sha256": _sha256(calibration),
        "manager_code_sha256": {
            m: _sha256(REPO_ROOT / "src" / "expertrelay" / m)
            for m in ("manager/policy.py", "manager/probe.py", "runtime/auto.py")
        },
    }


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj), encoding="utf-8")
    os.replace(tmp, path)


def run_path(setup_id: str, rnd: int) -> Path:
    return WORK / "runs" / f"{setup_id}#{rnd}.json"


def start_session(current: dict) -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    saved = WORK / "settings.json"
    if saved.exists():
        if json.loads(saved.read_text()) != current:
            raise SystemExit(f"{saved} is from other code/configs; move {WORK} away to start over")
    else:
        saved.write_text(json.dumps(current, indent=1))
    index = len(list(WORK.glob("session_*.json")))
    machine = collect_machine_profile(measure_disk=False)
    _write_json(
        WORK / f"session_{index}.json",
        base_record(
            label=f"manager validation session {index}", seed=0, model=None, config={}, machine=machine
        ),
    )
    print(f"session {index}: {machine.memory.available_bytes / 1e9:.2f} GB RAM free", flush=True)
    return index


def auto_mode(run: dict) -> str:
    m = run["manager"]
    if not m["cache_slots"]:
        return "no cache"
    return {"off": "cache only", "adaptive": "adaptive", "top_k": "top-8"}[m["prefetch"]]


def score(results: list[dict]) -> dict:
    out = {}
    for regime in REGIMES:
        rs = [r for r in results if r["regime"] == regime]
        means = {
            s: statistics.fmean(r["metrics"]["decode_tokens_per_s"] for r in rs if r["id"] == f"{regime}_{s}")
            for s in ("auto", *MODES)
        }
        best = max(MODES, key=lambda s: means[s])
        modes = sorted({auto_mode(r["run"]) for r in rs if r["id"] == f"{regime}_auto"})
        out[regime] = {
            "mean_tokens_per_s": means,
            "best_hand_set": best,
            "best_hand_set_mode": MODES[best],
            "auto_modes": modes,
            "picked_best": modes == [MODES[best]],
            "auto_over_best": means["auto"] / means[best],
        }
    return out


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
            fit = fit_cache(replace(RuntimeConfig.load(CONFIG), store_dir=MODELS_ROOT / STORE), CACHE_GB)
            s["args"] = [*s["args"], "--cache-gb", f"{fit['used_gb']:.9f}"]
        r = run_one(s, CONFIG, TIMEOUT_S)
        r = {**_portable(r), "round": rnd, "session": session, "cache_fit": fit}
        if r["status"] == "ok":
            if r["run"]["load"]["source"] == "cached":
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
        label="Phase 6: --auto vs hand-set configurations, fast and slow compute",
        seed=0,
        model={"store": STORE, "config": _portable(CONFIG)},
        config={
            "regimes": REGIMES,
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
        }
        for i, x in enumerate(sessions)
    ]
    record["interrupted"] = len({r["session"] for r in results}) > 1
    record["failed_attempts"] = [
        {k: f.get(k) for k in ("id", "round", "session", "status", "stderr_tail")} for f in failed
    ]
    record["results"] = results
    record["agreement"] = agreement(results)
    record["score"] = score(results)
    append_benchmark_record(OUT, record)
    print(f"saved {OUT}")
    print(json.dumps(record["score"], indent=1))


if __name__ == "__main__":
    main()
