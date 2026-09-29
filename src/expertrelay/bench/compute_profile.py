"""Where does the compute time of one generated token go?

    python -m expertrelay.bench.compute_profile            # base store, Phase 2 prompts
    python -m expertrelay.bench.compute_profile --kernel fused

Runs the real model on the Phase 2 prompt set with the numpy backend and
the no-cache expert source, and times every backend call by what it
computes:

  attn_proj       q/k/v/o projections (int8)
  attention_op    scores, softmax, weighted sum over the KV cache
  router          this layer's router (f32)
  shared_expert   the shared expert (int8) and its gate (f32)
  routed_experts  the routed experts (int8)
  lm_head         the output projection (int8, 151,936 x 2,048)
  embedding       the input row lookup

Each int8 multiply is split into its int8 -> f32 conversion and the matrix
multiply itself (with the fused kernel there is no separate conversion).
Whatever compute time is left is "other": norms, RoPE, activations,
routing bookkeeping (top-k, scatter-add) and Python overhead.

Expert reads are excluded: they happen in the expert source, outside the
backend, and are timed there. The timers wrap each call with
perf_counter, adding well under a millisecond per token.

Writes benchmarks/results/compute_profile.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from expertrelay import REPO_ROOT
from expertrelay.benchmarking import append_benchmark_record, base_record, peak_process_rss_mb
from expertrelay.manager.backend_selection import BackendChoice
from expertrelay.manager.profile import collect_machine_profile
from expertrelay.paths import BENCHMARK_RESULTS_DIR
from expertrelay.runtime import int8_linear as kernels
from expertrelay.runtime.backends import NumpyBackend
from expertrelay.runtime.generate import DEFAULT_RUNTIME_CONFIG, RuntimeConfig, generate, load_model
from expertrelay.runtime.weights import Int8Matrix
from expertrelay.store.tokenizer import load_tokenizer


class _Clock:
    def __init__(self) -> None:
        self.category = "other_call"
        self.seconds: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)

    def add(self, key: str, seconds: float) -> None:
        self.seconds[key] += seconds

    def reset(self) -> dict[str, float]:
        out = dict(self.seconds)
        self.seconds.clear()
        return out


class ProfilingBackend(NumpyBackend):
    """The numpy backend with every call timed. The blocked kernel is
    re-stated here with a timer around the conversion and one around the
    multiply; the fused kernel is timed as a whole."""

    def __init__(self, clock: _Clock, kernel: str):
        self.clock, self.kernel = clock, kernel

    def int8_linear(self, x, q, scales, bias=None):
        c = self.clock.category
        self.clock.calls[c] += 1
        if self.kernel == "fused" and kernels.use_fused(x.shape[0]):
            t = time.perf_counter()
            out = kernels.int8_linear(x, q, scales, bias, kernel="fused")
            self.clock.add(c + ":fused", time.perf_counter() - t)
            return out
        n = x.shape[0]
        out_dim, in_dim = q.shape
        block = kernels.block_rows_for(n)
        out = np.empty((n, out_dim), dtype=np.float32)
        for r0 in range(0, out_dim, block):
            r1 = min(out_dim, r0 + block)
            t0 = time.perf_counter()
            w = kernels._scratch_block(r1 - r0, in_dim)
            np.copyto(w, q[r0:r1], casting="unsafe")
            t1 = time.perf_counter()
            np.matmul(x, w.T, out=out[:, r0:r1])
            t2 = time.perf_counter()
            self.clock.add(c + ":convert", t1 - t0)
            self.clock.add(c + ":matmul", t2 - t1)
        t = time.perf_counter()
        out *= scales
        if bias is not None:
            out += bias
        self.clock.add(c + ":scale", time.perf_counter() - t)
        return out

    def linear(self, x, w, bias=None):
        t = time.perf_counter()
        out = super().linear(x, w, bias)
        self.clock.add(self.clock.category + ":f32", time.perf_counter() - t)
        return out

    def attention(self, q, keys, values, allowed):
        t = time.perf_counter()
        out = super().attention(q, keys, values, allowed)
        self.clock.add("attention_op", time.perf_counter() - t)
        return out


def _category(name: str) -> str:
    if "self_attn." in name:
        return "attn_proj"
    if "shared_expert" in name:
        return "shared_expert"
    if name.endswith("mlp.gate.weight"):
        return "router"
    if name == "lm_head.weight":
        return "lm_head"
    return "other_call"


def instrument(model, clock: _Clock) -> None:
    linear, expert_linear = model._linear, model._expert_linear

    def timed_linear(name, x, bias=None):
        clock.category = _category(name)
        return linear(name, x, bias)

    def timed_expert_linear(w, x):
        clock.category = "routed_experts"
        return expert_linear(w, x)

    model._linear = timed_linear
    model._expert_linear = timed_expert_linear
    emb = model.resident["model.embed_tokens.weight"]
    if isinstance(emb, Int8Matrix):
        original = emb.dequantized_rows

        def rows(r):
            t = time.perf_counter()
            out = original(r)
            clock.add("embedding", time.perf_counter() - t)
            return out

        object.__setattr__(emb, "dequantized_rows", rows)


def summarize(per_step: list[dict], compute_s: list[float]) -> dict:
    keys = sorted({k for s in per_step for k in s})
    mean = {k: statistics.fmean(s.get(k, 0.0) for s in per_step) for k in keys}
    total = statistics.fmean(compute_s)
    timed = sum(mean.values())
    groups = defaultdict(float)
    for k, v in mean.items():
        groups[k.split(":")[0]] += v
    parts = defaultdict(float)
    for k, v in mean.items():
        if ":" in k:
            parts[k.split(":")[1]] += v
    return {
        "compute_s_per_token": total,
        "by_call_s": mean,
        "by_group_s": dict(groups),
        "by_kind_s": dict(parts),
        "other_s": total - timed,
        "steps": len(per_step),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kernel", choices=["blocked", "fused"], default="blocked")
    ap.add_argument("--config", type=Path, default=DEFAULT_RUNTIME_CONFIG)
    args = ap.parse_args()

    rt = RuntimeConfig.load(args.config)
    prompts = json.loads(rt.prompts_file.read_text(encoding="utf-8"))["prompts"]
    record = base_record(
        label=f"compute profile per decode token ({args.kernel} int8 kernel)",
        seed=0,
        model={"store": rt.store_dir.relative_to(REPO_ROOT).as_posix()},
        config={
            "kernel": args.kernel,
            "prompts": [p["id"] for p in prompts],
            "max_new_tokens": rt.max_new_tokens,
        },
        machine=collect_machine_profile(measure_disk=False),
    )
    kernels.set_kernel(args.kernel)
    clock = _Clock()
    model, _ = load_model(
        rt.store_dir, "unbuffered", rt.max_seq, rt.memory_budget_gb, BackendChoice("numpy", None, "profile")
    )
    model.backend = ProfilingBackend(clock, args.kernel)
    instrument(model, clock)
    tokenizer = load_tokenizer(rt.store_dir)
    per_step, compute_s, read_s = [], [], []
    for p in prompts:
        ids = tokenizer.encode(p["text"]).ids
        clock.reset()
        # generate() runs prefill + decode; per-step clocks are split below
        steps_clock: list[dict] = []
        original_forward = model.forward

        def forward(*a, _orig=original_forward, _log=steps_clock, **kw):
            clock.reset()
            out = _orig(*a, **kw)
            _log.append(clock.reset())
            return out

        model.forward = forward
        _, steps = generate(model, ids, rt.max_new_tokens, rt.max_seq)
        model.forward = original_forward
        for s, c in zip(steps[1:], steps_clock[1:], strict=True):  # decode steps only
            per_step.append(c)
            compute_s.append(s.compute_seconds)
            read_s.append(s.expert_read_seconds)
        print(f"{p['id']}: done", flush=True)
    model.experts.close()
    record["profile"] = summarize(per_step, compute_s)
    record["profile"]["expert_read_s_per_token"] = statistics.fmean(read_s)
    record["peak_rss_mb"] = peak_process_rss_mb()
    out = BENCHMARK_RESULTS_DIR / "compute_profile.json"
    append_benchmark_record(out, record)
    print(json.dumps(record["profile"], indent=1))


if __name__ == "__main__":
    main()
