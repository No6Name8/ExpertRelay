"""Why llama.cpp Q8_0 disagrees with bf16 at early prompt positions: a
position-level check on one sequence.

    python -m expertrelay.bench.llamacpp_position_check

bench.llamacpp_quality found llama.cpp Q8_0 agreeing with bf16 at 98% of
the generated positions but much less at prompt positions, with 3-4
misses per prompt even inside the fixed chat-template text, where
ExpertRelay's int8 almost never misses. This records, for one sequence
(Step B3's prompt index 5, en_02):
  - every position where llama.cpp Q8_0's top-1 differs from bf16's, with
    the top-3 probabilities of bf16, ExpertRelay int8 and llama.cpp Q8_0;
  - llama.cpp's top-3 at those positions computed three ways: prefix by
    prefix on the KV cache (as llamacpp_quality does), the whole prefix in
    one batch with no cache reuse, and with an f32 KV cache (-ctk/-ctv f32)
    instead of the default f16;
  - the GGUF's model metadata next to the Hugging Face config.
If batch and incremental agree, the measurement method is not the cause;
if f32 KV changes nothing, KV precision is not the cause.

Writes benchmarks/results/llamacpp_position_check.json.
"""

from __future__ import annotations

import json
import subprocess
import urllib.request

import numpy as np
import torch

from expertrelay.bench.fair_test import PORT, _wait_healthy
from expertrelay.bench.gptq_quality import STORES, our_logits
from expertrelay.bench.gptq_quality import WORK as B3_WORK
from expertrelay.bench.llamacpp_quality import llama_logprobs
from expertrelay.bench.llamacpp_setup import BIN_DIR, GGUF, SRC_DIR, THREADS
from expertrelay.bench.reference_check import Bf16Checkpoint, lm_head_logits
from expertrelay.benchmarking import append_benchmark_record, base_record
from expertrelay.manager.profile import collect_machine_profile
from expertrelay.paths import BENCHMARK_RESULTS_DIR, MODELS_ROOT
from expertrelay.runtime.generate import load_model
from expertrelay.runtime.int8_linear import set_kernel
from expertrelay.store.download_checkpoint import checkpoint_dir
from expertrelay.store.tokenizer import load_tokenizer

INDEX = 5
OUT = BENCHMARK_RESULTS_DIR / "llamacpp_position_check.json"


def _top3(row: np.ndarray, tok) -> list[list]:
    p = np.exp(row - row.max())
    p /= p.sum()
    return [[tok.decode([int(t)]), float(p[t])] for t in np.argsort(-p)[:3]]


def _server(extra: list[str]) -> subprocess.Popen:
    cmd = [str(BIN_DIR / "llama-server.exe"), "-m", str(GGUF["Q8_0"]), "-t", str(THREADS), "-c", "256", "-np", "1",
           "--port", str(PORT), "--host", "127.0.0.1", *extra]  # fmt: skip
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    _wait_healthy(proc, 1800)
    return proc


def _batch_top3(prefix: list[int], tok) -> list[list]:
    body = {
        "prompt": prefix,
        "n_predict": 1,
        "n_probs": 3,
        "temperature": 0.0,
        "top_k": 1,
        "cache_prompt": False,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/completion",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    d = json.loads(urllib.request.urlopen(req, timeout=3600).read())
    return [
        [x["token"], float(np.exp(x["logprob"]))]
        for x in d["completion_probabilities"][0]["top_logprobs"][:3]
    ]


def _gguf_metadata() -> dict:
    import sys

    sys.path.insert(0, str(SRC_DIR / "gguf-py"))
    from gguf import GGUFReader

    out = {}
    for k, f in GGUFReader(GGUF["Q8_0"]).fields.items():
        if k.startswith("qwen2moe.") and f.data:
            out[k] = f.parts[f.data[0]].tolist()
    return out


def main() -> None:
    seq = [json.loads(line) for line in (B3_WORK / "sequences.jsonl").read_text().splitlines()][INDEX]
    ids, n = seq["ids"], seq["prompt_len"]
    store = MODELS_ROOT / STORES["int8"]
    tok = load_tokenizer(store)
    source = json.loads((store / "store.json").read_text())["source"]
    ckpt = Bf16Checkpoint(checkpoint_dir(source["repo_id"], source["revision"]))
    ref = lm_head_logits(ckpt, np.load(B3_WORK / "bf16" / f"{INDEX:02d}.npy"), dtype=torch.float32)
    model, _ = load_model(store, "unbuffered", 128, None)
    set_kernel("fused")
    int8 = our_logits(
        np.load(B3_WORK / "int8" / f"{INDEX:02d}.npy"), model.resident["lm_head.weight"], model.backend
    )
    model.experts.close()

    proc = _server([])
    try:
        q8 = llama_logprobs(ids, ref.shape[1])
        differ = [i for i in range(len(ids)) if int(ref[i].argmax()) != int(q8[i].argmax())]
        batch = {i: _batch_top3(ids[: i + 1], tok) for i in differ}
    finally:
        proc.terminate()
        proc.wait(60)
    proc = _server(["-ctk", "f32", "-ctv", "f32"])
    try:
        f32kv = {i: _batch_top3(ids[: i + 1], tok) for i in differ}
    finally:
        proc.terminate()
        proc.wait(60)

    rows = [
        {
            "position": i,
            "in_prompt": i < n - 1,
            "context_token": tok.decode([ids[i]]),
            "bf16_top3": _top3(ref[i], tok),
            "int8_top3": _top3(int8[i], tok),
            "q8_incremental_top3": _top3(q8[i], tok),
            "q8_batch_top3": batch[i],
            "q8_batch_f32_kv_top3": f32kv[i],
        }
        for i in differ
    ]
    hf = json.loads((checkpoint_dir(source["repo_id"], source["revision"]) / "config.json").read_text())
    record = base_record(
        label="llama.cpp Q8_0 vs bf16: positions where top-1 differs (one sequence)",
        seed=0,
        model={"gguf": GGUF["Q8_0"].name, "sequence": seq["id"], "prompt_len": n, "positions": len(ids)},
        config={"index": INDEX},
        machine=collect_machine_profile(measure_disk=False),
    )
    record["differing_positions"] = rows
    record["gguf_metadata"] = _gguf_metadata()
    record["hf_config"] = {k: hf.get(k) for k in ("rope_theta", "rms_norm_eps", "num_experts", "num_experts_per_tok",
                                                   "norm_topk_prob", "moe_intermediate_size")}  # fmt: skip
    append_benchmark_record(OUT, record)
    for r in rows:
        print(r["position"], r["context_token"], r["bf16_top3"][0], r["q8_incremental_top3"][0], r["q8_batch_top3"][0],
              r["q8_batch_f32_kv_top3"][0])  # fmt: skip


if __name__ == "__main__":
    main()
