"""Step C quality: llama.cpp Q4_K_M and Q8_0 vs the bf16 Chat model, at the
same positions as ExpertRelay's int8 (Step B3), teacher-forced. Resumable.

    python -m expertrelay.bench.llamacpp_quality

Positions and reference are Step B3's (bench.gptq_quality, work files in
models/work/gptq_quality/): the same 28 prompts in the Chat template, each
continued for 32 tokens by ExpertRelay's int8 store, and the bf16 Chat
model's final hidden state at every position (Hugging Face transformers,
layer by layer). The bf16 logits come from those states and the bf16
lm_head; ExpertRelay int8's scores at these positions are B3's
(benchmarks/results/gptq_quality.json).

llama.cpp's distribution at every position comes from the pinned build's
own llama-server (bench.llamacpp_setup): for each prefix of each sequence,
a /completion request with the prefix as token ids, n_predict 1 and
n_probs = the whole vocabulary returns the log-probability of every token
at the prefix's last position, computed by the server from its logits.
Requests extend the previous prefix by one token with cache_prompt on, so
the server evaluates one new token per request on its KV cache.
Deviation, recorded in docs/limitations.md: every position, the prompt's
included, is evaluated one token at a time, while ExpertRelay and the
bf16 reference process the prompt in one batch; batch and single-token
kernels can round differently.

Scores per position, as in bench.quantization_quality: top-1 agreement
with bf16 and KL(bf16 || llama.cpp) in nats over the whole vocabulary.
Same pass rule as before (overall >= 96%, every category >= 90%).

Each prompt's scores are saved as they finish (models/work/llamacpp_quality/);
writes benchmarks/results/llamacpp_quality.json.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

from expertrelay.bench.fair_test import PORT, _wait_healthy
from expertrelay.bench.gptq_quality import ARABIC, STORES, _append_line, _read_lines
from expertrelay.bench.gptq_quality import WORK as B3_WORK
from expertrelay.bench.llamacpp_setup import BIN_DIR, GGUF, THREADS
from expertrelay.bench.quantization_quality import aggregate, load_prompts, score, verdict
from expertrelay.bench.reference_check import Bf16Checkpoint, lm_head_logits
from expertrelay.benchmarking import append_benchmark_record, base_record, peak_process_rss_mb
from expertrelay.manager.profile import collect_machine_profile
from expertrelay.paths import BENCHMARK_RESULTS_DIR, MODELS_ROOT
from expertrelay.store.download_checkpoint import checkpoint_dir

MODELS = ("Q4_K_M", "Q8_0")
WORK = MODELS_ROOT / "work" / "llamacpp_quality"
OUT = BENCHMARK_RESULTS_DIR / "llamacpp_quality.json"
CONTEXT = 256


def _request(prefix: list[int], vocab: int) -> tuple[np.ndarray, int]:
    body = {
        "prompt": prefix,
        "n_predict": 1,
        "n_probs": vocab,
        "temperature": 0.0,
        "top_k": 1,
        "cache_prompt": True,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/completion",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    d = json.loads(urllib.request.urlopen(req, timeout=3600).read())
    logprobs = np.full(vocab, -np.inf, dtype=np.float64)
    for x in d["completion_probabilities"][0]["top_logprobs"]:
        logprobs[x["id"]] = x["logprob"]
    return logprobs, d["tokens_evaluated"]


def llama_logprobs(ids: list[int], vocab: int) -> np.ndarray:
    """[len(ids), vocab]: row i = log P(next token | ids[:i+1])."""
    rows = []
    for n in range(1, len(ids) + 1):
        lp, evaluated = _request(ids[:n], vocab)
        if evaluated != n:  # the server must see exactly our tokens (no BOS added, nothing dropped)
            raise RuntimeError(f"server evaluated {evaluated} tokens for a {n}-token prefix")
        rows.append(lp)
    return np.stack(rows)


def main() -> None:
    sequences = {d["index"]: d for d in _read_lines(B3_WORK / "sequences.jsonl").values()}
    prompts = load_prompts()
    source = json.loads((MODELS_ROOT / STORES["int8"] / "store.json").read_text())["source"]
    ckpt = Bf16Checkpoint(checkpoint_dir(source["repo_id"], source["revision"]))
    vocab = ckpt.config().vocab_size
    WORK.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        path = WORK / f"{model}.jsonl"
        done = _read_lines(path)
        if len(done) == len(prompts):
            continue
        cmd = [str(BIN_DIR / "llama-server.exe"), "-m", str(GGUF[model]), "-t", str(THREADS), "-c", str(CONTEXT),
               "-np", "1", "--port", str(PORT), "--host", "127.0.0.1"]  # fmt: skip
        with tempfile.TemporaryDirectory() as tmp, open(Path(tmp) / "server.log", "w") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            try:
                _wait_healthy(proc, 1800)
                for i, p in enumerate(prompts):
                    if i in done:
                        continue
                    t = time.perf_counter()
                    ids, n = sequences[i]["ids"], sequences[i]["prompt_len"]
                    ours = llama_logprobs(ids, vocab)
                    ref = lm_head_logits(
                        ckpt, np.load(B3_WORK / "bf16" / f"{i:02d}.npy"), dtype=torch.float32
                    )
                    row = {"id": p["id"], "set": p["set"], "category": p["category"], **score(ref, ours, n)}
                    _append_line(path, {"index": i, "row": row})
                    print(
                        f"{model} {p['id']}: {row['agree']}/{row['positions']} ({time.perf_counter() - t:.0f} s)",
                        flush=True,
                    )
            finally:
                proc.terminate()
                proc.wait(60)

    b3 = json.loads((BENCHMARK_RESULTS_DIR / "gptq_quality.json").read_text(encoding="utf-8"))[-1]
    by_precision = {"expertrelay_int8": b3["analysis"]["by_precision"]["int8"]}
    for model in MODELS:
        rows = [d["row"] for _, d in sorted(_read_lines(WORK / f"{model}.jsonl").items())]
        cats = sorted({r["category"] for r in rows})
        by_precision[f"llamacpp_{model}"] = {
            "overall": aggregate(rows),
            "by_category": {c: aggregate([r for r in rows if r["category"] == c]) for c in cats},
            "arabic": aggregate([r for r in rows if r["category"] in ARABIC]),
            "per_prompt": rows,
        }
    record = base_record(
        label="Step C quality: llama.cpp Q4_K_M / Q8_0 vs bf16 Chat at ExpertRelay's positions",
        seed=0,
        model={"gguf": json.loads((GGUF["Q4_K_M"].parent / "manifest.json").read_text()), "positions_from": "gptq_quality"},
        config={"method": "llama-server n_probs = vocab, prefix by prefix", "expertrelay_int8_from": {
            "file": "benchmarks/results/gptq_quality.json", "git_commit": b3["git_commit"], "timestamp": b3["timestamp"]}},
        machine=collect_machine_profile(measure_disk=False),
    )  # fmt: skip
    record["analysis"] = {
        "prompts": len(prompts),
        "positions": by_precision["expertrelay_int8"]["overall"]["positions"],
        "by_precision": by_precision,
        "verdict": verdict(by_precision),
    }
    record["peak_rss_mb"] = peak_process_rss_mb()
    append_benchmark_record(OUT, record)
    print(json.dumps(record["analysis"]["verdict"], indent=1))


if __name__ == "__main__":
    main()
