"""Step C: the fair test. ExpertRelay against OS paging, a normal load and
llama.cpp, same machine, same prompts, same settings. Resumable.

    python -m expertrelay.bench.fair_test            # step 3: setups A-F
    python -m expertrelay.bench.fair_test --sweep    # step 5: ExpertRelay memory sweep

Workload: the 24 prompts of benchmarks/prompts/quality24.json (4 per
category: English, MSA, Gulf Arabic, code, math, chat), each in the Chat
model's chat template, 64 new tokens, greedy, every token generated (EOS
doesn't stop any engine). Prompt token ids come from ExpertRelay's
tokenizer + template and are given to llama.cpp as ids, so every engine
sees exactly the same prompt.

Setups (step 3):
  A  normal load: the whole int8 store into RAM (--source ram, budget off),
     1 run, timeout 15 min: expected to fail on 8 GB; records how
  B  OS paging: the int8 store memory-mapped (--source mmap), the SAME fused
     kernel as ExpertRelay
  C  ExpertRelay, no cache (--source unbuffered, fused kernel)
  D  ExpertRelay --auto (the Manager's decisions are recorded per run)
  E  llama.cpp b11146, Q4_K_M GGUF
  F  llama.cpp b11146, Q8_0 GGUF
B-F run 3 times each, round r in the order B..F rotated by r, so drift over
the session falls on every setup; A runs last (it can push the machine
into heavy paging, which would distort what follows).

llama.cpp settings: the official Windows CPU build (bench.llamacpp_setup),
llama-server with -t 12 (= ExpertRelay's numba threads), default memory
mapping, context 256, one slot; per request: the prompt as token ids,
n_predict 64, temperature 0 and top_k 1 (greedy), ignore_eos, and
cache_prompt false so no prompt prefix is reused from the previous prompt
(ExpertRelay recomputes every prompt too). One server per run: started
cold, all 24 prompts, stopped.

Metrics, the same definitions for every engine:
  decode tokens/s   tokens 2..64 / the time from token 1 to token 64
                    (ExpertRelay: its per-step timings; llama.cpp: the
                    client-side arrival time of each streamed token; the
                    server's own timings are recorded next to it)
  time to first token  prompt processing + the first token, per prompt
  peak RAM          the process's peak working set, polled every 0.5 s
                    (counts memory-mapped file pages, as Task Manager does)
  disk read         bytes the OS read from physical disks during the run
                    (all processes; the machine is otherwise idle)
  free RAM          before every run
Every run reads from a cold process, but the OS file cache is NOT emptied
between runs (that needs admin rights): memory-mapped setups (B, E, F) may
find part of their file still cached from an earlier run. The disk-read
column shows how much each run really read.

Memory sweep (step 5): ExpertRelay with expert caches of 0, 0.5, 1.0, 1.5
and 2.0 GB (2 I/O threads; adaptive prefetch with the cache, as --auto
chooses on this machine), 1 run each, skipped (and recorded) when the RAM
free at its start can't hold it with the 0.54 GB safety margin. Tokens must
be identical at every size.

Results: benchmarks/results/fair_test.json and fair_test_sweep.json.
Run on a clean machine: browsers and VS Code closed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

from expertrelay import REPO_ROOT
from expertrelay.bench.llamacpp_setup import BIN_DIR, GGUF, THREADS
from expertrelay.bench.phase2_baselines import run_one
from expertrelay.bench.resumable import RunSet, portable
from expertrelay.benchmarking import append_benchmark_record, base_record, process_memory_mb
from expertrelay.manager.policy import safety_margin_bytes
from expertrelay.manager.profile import collect_machine_profile, device_read_bytes, measure_memory
from expertrelay.paths import BENCHMARK_RESULTS_DIR, MODELS_ROOT
from expertrelay.runtime.generate import RuntimeConfig, estimate_ram_bytes, model_config
from expertrelay.store.chat_template import ChatTemplate
from expertrelay.store.tokenizer import load_tokenizer

STORE = MODELS_ROOT / "qwen1.5-moe-a2.7b-chat-int8"
CONFIG = REPO_ROOT / "configs" / "stepc.json"
SWEEP_CONFIG = REPO_ROOT / "configs" / "stepc_sweep.json"
ROUNDS = 3
SWEEP_GB = [0.0, 0.5, 1.0, 1.5, 2.0]
PORT = 8089
CONTEXT = 256
TIMEOUT_S = {"A": 900, "B": 3 * 3600, "C": 3600, "D": 3600, "E": 3 * 3600, "F": 3 * 3600}
PER_LAYER = ("attention_seconds", "moe_compute_seconds", "read_wait_seconds")
OUT = BENCHMARK_RESULTS_DIR / "fair_test.json"
OUT_SWEEP = BENCHMARK_RESULTS_DIR / "fair_test_sweep.json"
WORK = MODELS_ROOT / "work" / "fair_test"
WORK_SWEEP = MODELS_ROOT / "work" / "fair_test_sweep"
FUSED = ["--int8-kernel", "fused", "--store-dir", str(STORE)]
SETUPS = {
    "A": {"name": "normal load (int8, whole store in RAM)", "engine": "expertrelay", "source": "ram",
          "budget": False, "args": FUSED},
    "B": {"name": "OS paging (int8 store memory-mapped, fused kernel)", "engine": "expertrelay", "source": "mmap",
          "budget": True, "args": FUSED},
    "C": {"name": "ExpertRelay, no cache", "engine": "expertrelay", "source": "unbuffered", "budget": True,
          "args": FUSED},
    "D": {"name": "ExpertRelay --auto", "engine": "expertrelay", "source": "cached", "budget": True,
          "args": [*FUSED, "--auto"]},
    "E": {"name": "llama.cpp Q4_K_M", "engine": "llama.cpp", "gguf": "Q4_K_M"},
    "F": {"name": "llama.cpp Q8_0", "engine": "llama.cpp", "gguf": "Q8_0"},
}  # fmt: skip


def plan() -> list[str]:
    base = ["B", "C", "D", "E", "F"]
    return [f"{s}#{r}" for r in range(ROUNDS) for s in base[r % 5 :] + base[: r % 5]] + ["A#0"]


def prompts() -> list[dict]:
    rt = RuntimeConfig.load(CONFIG)
    tokenizer, template = load_tokenizer(STORE), ChatTemplate.for_store(STORE)
    out = []
    for p in json.loads(rt.prompts_file.read_text(encoding="utf-8"))["prompts"]:
        out.append({**p, "prompt_ids": tokenizer.encode(template.user_prompt(p["text"])).ids})
    return out


class _PeakPoller(threading.Thread):
    def __init__(self, pid: int):
        super().__init__(daemon=True)
        self.pid, self.peak, self.private, self._stop = pid, 0.0, 0.0, threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            sample = process_memory_mb(self.pid)
            if sample:
                self.peak, self.private = max(self.peak, sample[0]), max(self.private, sample[1])
            self._stop.wait(0.5)

    def stop(self) -> None:
        self._stop.set()
        self.join()


def _post_stream(prompt_ids: list[int], n_predict: int) -> dict:
    body = {
        "prompt": prompt_ids,
        "n_predict": n_predict,
        "temperature": 0.0,
        "top_k": 1,
        "ignore_eos": True,
        "cache_prompt": False,
        "stream": True,
        "return_tokens": True,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/completion",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    arrivals, tokens, final = [], [], {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            chunk = json.loads(line[5:])
            got = chunk.get("tokens") or []
            now = time.perf_counter() - t0
            arrivals += [now] * len(got)
            tokens += got
            if chunk.get("stop"):
                final = chunk
    return {"tokens": tokens, "arrivals": arrivals, "timings": final.get("timings"), "final": final}


def _wait_healthy(proc: subprocess.Popen, timeout_s: float) -> float:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited with {proc.returncode} while loading")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                if r.status == 200:
                    return time.perf_counter() - t0
        except OSError:
            pass
        time.sleep(0.5)
    raise TimeoutError("llama-server did not become healthy")


def run_llama(setup: dict, prompt_set: list[dict], timeout_s: float) -> dict:
    gguf = GGUF[setup["gguf"]]
    cmd = [str(BIN_DIR / "llama-server.exe"), "-m", str(gguf), "-t", str(THREADS), "-c", str(CONTEXT), "-np", "1",
           "--port", str(PORT), "--host", "127.0.0.1"]  # fmt: skip
    result = {
        **setup,
        "command": portable(cmd),
        "ram_available_before_bytes": measure_memory().available_bytes,
    }
    dev0, t0 = device_read_bytes(), time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "server.log"
        with open(log, "w", encoding="utf-8") as logf:
            proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=REPO_ROOT)
            poller = _PeakPoller(proc.pid)
            poller.start()
            try:
                result["load_seconds"] = _wait_healthy(proc, timeout_s)
                runs = []
                for p in prompt_set:
                    if time.perf_counter() - t0 > timeout_s:
                        raise TimeoutError(f"timeout after {timeout_s:.0f} s")
                    r = _post_stream(p["prompt_ids"], RuntimeConfig.load(CONFIG).max_new_tokens)
                    a = r["arrivals"]
                    runs.append(
                        {
                            "id": p["id"],
                            "category": p["category"],
                            "prompt_ids": p["prompt_ids"],
                            "generated_ids": r["tokens"],
                            "ttft_s": a[0],
                            "decode_s": a[-1] - a[0],
                            "decode_tokens": len(a) - 1,
                            "server_timings": r["timings"],
                        }
                    )
                    print(
                        f"[{setup['gguf']}] {p['id']}: {len(a) - 1} tokens in {a[-1] - a[0]:.1f} s",
                        flush=True,
                    )
                result["status"] = "ok"
                result["prompts"] = runs
            except (RuntimeError, TimeoutError, OSError) as e:
                result["status"] = f"failed: {e}"
            finally:
                proc.terminate()
                try:
                    proc.wait(30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                poller.stop()
        result["server_log_tail"] = log.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
    result.update(
        wall_seconds=time.perf_counter() - t0,
        device_bytes_read_total=device_read_bytes() - dev0,
        peak_rss_mb_polled=poller.peak,
        peak_private_mb_polled=poller.private,
    )
    if result["status"] == "ok":
        result["metrics"] = llama_metrics(result["prompts"])
    return result


def llama_metrics(runs: list[dict]) -> dict:
    tokens = sum(r["decode_tokens"] for r in runs)
    timings = [r["server_timings"] for r in runs if r["server_timings"]]
    return {
        "decode_tokens_per_s": tokens / sum(r["decode_s"] for r in runs),
        "time_to_first_token_mean_s": statistics.fmean(r["ttft_s"] for r in runs),
        "decode_tokens": tokens,
        "server_predicted_per_second_mean": statistics.fmean(t["predicted_per_second"] for t in timings)
        if timings
        else None,
        "server_prompt_ms_mean": statistics.fmean(t["prompt_ms"] for t in timings) if timings else None,
    }


def run_expertrelay(setup: dict, config: Path, timeout_s: float) -> dict:
    r = run_one({**setup, "id": setup["key"]}, config, timeout_s)
    if r["status"] == "ok":  # metrics are computed; per-layer timings would make the file huge
        for p in r["run"]["prompts"]:
            for step in p["steps"]:
                for k in PER_LAYER:
                    step.pop(k, None)
    return r


def token_check(results: dict[str, dict], keys: list[str]) -> dict:
    """ExpertRelay's setups run the same int8 model: identical tokens required."""
    ok = {
        k: [p["generated_ids"] for p in results[k]["run"]["prompts"]]
        for k in keys
        if results[k]["status"] == "ok"
    }
    ref_key = next(iter(ok), None)
    return {k: v == ok[ref_key] for k, v in ok.items()} | {"reference": ref_key}


def main_runs() -> None:
    prompt_set = prompts()
    settings = {
        "plan": plan(),
        "setups": portable(SETUPS),
        "config": json.loads(CONFIG.read_text()),
        "prompt_ids": [p["prompt_ids"] for p in prompt_set],
        "gguf": json.loads((GGUF["Q4_K_M"].parent / "manifest.json").read_text()),
        "threads": THREADS,
        "context": CONTEXT,
    }
    runs = RunSet(WORK, settings)
    runs.start("fair test")
    for run_id in plan():
        if runs.done(run_id):
            continue
        runs.prepare(run_id)
        key = run_id.split("#")[0]
        setup = {**SETUPS[key], "key": key}
        if setup["engine"] == "llama.cpp":
            r = run_llama(setup, prompt_set, TIMEOUT_S[key])
        else:
            r = run_expertrelay(setup, CONFIG, TIMEOUT_S[key])
        runs.save(run_id, r)
    results = {run_id: runs.load(run_id) for run_id in plan()}
    unfinished = [k for k, r in results.items() if r["status"] != "ok" and not k.startswith("A")]
    if unfinished:
        raise SystemExit(f"runs not ok: {unfinished}; rerun in a clean session to redo them")
    record = base_record(
        label="Step C fair test: ExpertRelay vs OS paging, normal load and llama.cpp",
        seed=0,
        model={"store": STORE.name, "gguf": settings["gguf"]},
        config={k: v for k, v in settings.items() if k != "prompt_ids"}
        | {"prompts": [p["id"] for p in prompt_set]},
        machine=collect_machine_profile(measure_disk=False),
    )
    record["sessions"] = runs.sessions()
    record["interrupted"] = len({r["session"] for r in results.values()}) > 1
    record["failed_attempts"] = runs.failed_attempts()
    record["results"] = results
    ours = [k for k in results if k[0] in "ABCD"]
    record["tokens_identical_expertrelay"] = token_check(results, ours)
    append_benchmark_record(OUT, record)
    print(f"saved {OUT}")


def sweep_fits(cache_gb: float) -> tuple[bool, dict]:
    rt = RuntimeConfig.load(SWEEP_CONFIG)
    config = model_config(STORE)
    need = estimate_ram_bytes(STORE, config, rt.max_seq, "cached", "numpy", cache_gb, "fused")
    free = measure_memory().available_bytes
    margin = safety_margin_bytes(measure_memory().total_bytes)
    return need + margin <= free, {"needed_bytes": need, "free_bytes": free, "margin_bytes": margin}


def main_sweep() -> None:
    settings = {"sizes_gb": SWEEP_GB, "config": json.loads(SWEEP_CONFIG.read_text())}
    runs = RunSet(WORK_SWEEP, settings)
    runs.start("fair test memory sweep")
    cached = [*FUSED, "--io-threads", "2", "--prefill-pipelining", "--no-pin-layer0", "--prefetch",
              "--prefetch-adaptive", "--prefetch-min-probability", "0.05"]  # fmt: skip
    for gb in SWEEP_GB:
        run_id = f"cache_{gb:.1f}gb"
        if runs.done(run_id):
            continue
        runs.prepare(run_id)
        fits, check = sweep_fits(gb)
        if not fits:
            runs.save(
                run_id, {"status": "ok", "skipped": "not enough free RAM", "ram_check": check, "cache_gb": gb}
            )
            continue
        if gb == 0:
            setup = {"key": run_id, "name": "no cache", "source": "unbuffered", "budget": True, "args": FUSED}
        else:
            setup = {"key": run_id, "name": f"cache {gb} GB + adaptive prefetch", "source": "cached",
                     "budget": True, "args": [*cached, "--cache-gb", f"{gb:.9f}"]}  # fmt: skip
        r = run_expertrelay(setup, SWEEP_CONFIG, 3600)
        runs.save(run_id, {**r, "ram_check": check, "cache_gb": gb})
    results = {f"cache_{gb:.1f}gb": runs.load(f"cache_{gb:.1f}gb") for gb in SWEEP_GB}
    record = base_record(
        label="Step C memory sweep: ExpertRelay at expert-cache sizes 0-2 GB",
        seed=0,
        model={"store": STORE.name},
        config=settings,
        machine=collect_machine_profile(measure_disk=False),
    )
    record["sessions"] = runs.sessions()
    record["failed_attempts"] = runs.failed_attempts()
    record["results"] = results
    ran = [k for k, r in results.items() if "skipped" not in r]
    record["tokens_identical"] = token_check(results, ran)
    append_benchmark_record(OUT_SWEEP, record)
    print(f"saved {OUT_SWEEP}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", action="store_true", help="step 5: the memory sweep instead of setups A-F")
    args = ap.parse_args()
    main_sweep() if args.sweep else main_runs()


if __name__ == "__main__":
    main()
