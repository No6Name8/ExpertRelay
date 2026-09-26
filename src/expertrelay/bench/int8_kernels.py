"""Benchmark the realistic ways to compute y = x @ W_hat^T on this CPU, where
W_hat = Q * scale (int8 weights, one f32 scale per output row).

    python -m expertrelay.bench.int8_kernels

Candidates:
  numpy_dequant      materialize W_hat = Q.astype(f32) * s, then BLAS sgemm
  numpy_factored     y = (x @ Q.astype(f32)^T) * s. Per-row scales factor out
                     of the dot product, so the weights never get multiplied
  numpy_blocked      (the shipped kernel) the factored form, converting Q a block of rows at a
                     time into a reused, cache-sized f32 buffer. Never
                     materializes the whole f32 matrix and never allocates
                     per call
  mindspore_fp32     int8 -> f32 cast and matmul as MindSpore CPU ops, scale after
  mindspore_fp16     int8 -> f16 cast and matmul as MindSpore CPU ops, scale in f32 after
  numpy_blocked_rowsN  numpy_blocked with a fixed block size N, to pick the block sizes

Error is measured against float64 math on the same int8 weights, so it's the
error of the compute path alone (quantization error is a separate thing,
measured by the store build). Weights and inputs are seeded random data,
not model weights. Kernel speed doesn't depend on the values. The shapes are
the real Qwen1.5-MoE-A2.7B ones: attention, one routed expert, the shared
expert, and lm_head. seq=1 is a decode step; seq=32 is a prefill chunk.
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from expertrelay.benchmarking import append_benchmark_record, base_record, peak_process_rss_mb
from expertrelay.manager.profile import collect_machine_profile
from expertrelay.paths import BENCHMARK_RESULTS_DIR
from expertrelay.runtime.int8_linear import BLOCK_ROWS_DECODE, BLOCK_ROWS_PREFILL, int8_linear
from expertrelay.store.quantize import quantize_rowwise_int8

SHAPES = {
    "attn_q (2048x2048)": (2048, 2048),
    "expert_gate (1408x2048)": (1408, 2048),
    "expert_down (2048x1408)": (2048, 1408),
    "shared_gate (5632x2048)": (5632, 2048),
    "lm_head (151936x2048)": (151936, 2048),
}
SEQ_LENS = (1, 32)
BLOCK_ROWS_SWEEP = (128, 256, 512, 1024, 2048)
# Materializing a full f32 copy of lm_head is 1.24 GB, more than this machine
# has free. Candidates that do that are skipped for it, and the skip is
# recorded, not hidden.
FULL_MATERIALIZE_LIMIT_BYTES = 256 * 2**20


def _numpy_dequant(x, q, s):
    return x @ (q.astype(np.float32) * s[:, None]).T


def _numpy_factored(x, q, s):
    return (x @ q.astype(np.float32).T) * s


def _numpy_blocked(x, q, s):
    return int8_linear(x, q, s)


def _blocked_with(block_rows: int) -> Callable:
    def fn(x, q, s):
        return int8_linear(x, q, s, block_rows=block_rows)

    return fn


def _mindspore_candidates() -> dict[str, Callable]:
    import mindspore as ms
    from mindspore import ops

    ms.set_device("CPU")

    def fp32(x, q, s):
        qt = ops.cast(ms.Tensor(q), ms.float32)
        return (ops.matmul(ms.Tensor(x), qt.T) * ms.Tensor(s)).asnumpy()

    def fp16(x, q, s):
        qt = ops.cast(ms.Tensor(q), ms.float16)
        y = ops.matmul(ms.Tensor(x.astype(np.float16)), qt.T)
        return (ops.cast(y, ms.float32) * ms.Tensor(s)).asnumpy()

    return {"mindspore_fp32": fp32, "mindspore_fp16": fp16}


def _time(fn: Callable, repeats: int) -> float:
    fn()  # warm-up: first-call allocation, MindSpore kernel selection
    samples = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t)
    return statistics.median(samples)


def run(repeats: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    candidates: dict[str, Callable] = {
        "numpy_dequant": _numpy_dequant,
        "numpy_factored": _numpy_factored,
        "numpy_blocked": _numpy_blocked,
        **{f"numpy_blocked_rows{b}": _blocked_with(b) for b in BLOCK_ROWS_SWEEP},
        **_mindspore_candidates(),
    }
    rows_out = []
    for shape_name, (out_dim, in_dim) in SHAPES.items():
        q, s = quantize_rowwise_int8(rng.normal(0, 0.02, (out_dim, in_dim)).astype(np.float32))
        full_f32_bytes = out_dim * in_dim * 4
        for seq in SEQ_LENS:
            x = rng.normal(0, 1, (seq, in_dim)).astype(np.float32)
            exact = x.astype(np.float64) @ (q.astype(np.float64) * s.astype(np.float64)[:, None]).T
            for name, fn in candidates.items():
                row = {"shape": shape_name, "seq": seq, "candidate": name}
                if not name.startswith("numpy_blocked") and full_f32_bytes > FULL_MATERIALIZE_LIMIT_BYTES:
                    row["skipped"] = f"materializes {full_f32_bytes / 1e9:.2f} GB of f32 weights"
                    rows_out.append(row)
                    continue
                y = fn(x, q, s)
                row["rel_error_vs_fp64"] = float(np.linalg.norm(y - exact) / np.linalg.norm(exact))
                row["median_ms"] = _time(lambda fn=fn, x=x, q=q, s=s: fn(x, q, s), repeats) * 1e3
                rows_out.append(row)
                print(
                    f"{shape_name:<24} seq={seq:<3} {name:<15} {row['median_ms']:9.2f} ms  "
                    f"err {row['rel_error_vs_fp64']:.2e}",
                    flush=True,
                )
    return rows_out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=BENCHMARK_RESULTS_DIR / "int8_kernels.json")
    args = ap.parse_args()
    results = run(args.repeats, args.seed)
    record = base_record(
        label="int8 linear kernels",
        seed=args.seed,
        model=None,
        config={
            "repeats": args.repeats,
            "seq_lens": list(SEQ_LENS),
            "block_rows_decode": BLOCK_ROWS_DECODE,
            "block_rows_prefill": BLOCK_ROWS_PREFILL,
            "shapes": {k: list(v) for k, v in SHAPES.items()},
        },
        machine=collect_machine_profile(measure_disk=False),
        results=results,
        peak_rss_mb=peak_process_rss_mb(),
    )
    append_benchmark_record(args.out, record)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
