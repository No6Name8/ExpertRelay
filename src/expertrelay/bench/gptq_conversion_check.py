"""Check the GPTQ int4 store against the GPTQ release, on sampled experts.

    python -m expertrelay.bench.gptq_conversion_check --samples 48

For each sampled (layer, expert) and each of its three matrices, the
weights are computed twice:
  - ours: the store record (as the runtime reads it), unpacked and scaled
    by store.int4 (value = nibble - 8, times the float16 group scale);
  - reference: straight from the release's raw tensors with the documented
    GPTQ formula, implemented here independently of store.gptq, in its
    general form (per-group zero points, g_idx gather), so it would also
    catch a wrong assumption about the zero points or the group order:
        W[o, i] = (code[i, o] - (stored_zero[g_idx[i], o] + 1)) * scales[g_idx[i], o]
    (AutoGPTQ "gptq" (v1) checkpoint format: zero points stored minus one).
They must be bit-identical in float32. The sample is seeded, and always
includes the first and last expert of the first and last layer. Writes
benchmarks/results/gptq_conversion_check.json.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from expertrelay.benchmarking import append_benchmark_record, base_record, peak_process_rss_mb
from expertrelay.manager.profile import collect_machine_profile
from expertrelay.paths import BENCHMARK_RESULTS_DIR, MODELS_ROOT
from expertrelay.store.build_int4_store import GPTQ_REPO, GPTQ_REVISION
from expertrelay.store.download_checkpoint import checkpoint_dir
from expertrelay.store.expert_reader import ExpertStoreReader
from expertrelay.store.int4 import dequantize_groupwise_int4
from expertrelay.store.layout import parse_expert_record
from expertrelay.store.safetensors_local import LocalCheckpoint

RESULTS = BENCHMARK_RESULTS_DIR / "gptq_conversion_check.json"


def reference_dequantize(
    qweight: np.ndarray, qzeros: np.ndarray, scales: np.ndarray, g_idx: np.ndarray
) -> np.ndarray:
    """The documented GPTQ formula -> float32 W [out, in]. Deliberately a
    different route from store.gptq: shifts broadcast over a shift vector,
    per-element zero and scale gathered through g_idx."""
    shifts = np.arange(0, 32, 4, dtype=np.uint32)
    w_codes = (qweight.astype(np.uint32)[:, None, :] >> shifts[None, :, None]) & 0xF  # [in/8, 8, out]
    w_codes = w_codes.reshape(-1, qweight.shape[1]).astype(np.int32)  # [in, out]
    z_codes = (qzeros.astype(np.uint32)[:, :, None] >> shifts[None, None, :]) & 0xF  # [groups, out/8, 8]
    zeros = z_codes.reshape(qzeros.shape[0], -1).astype(np.int32) + 1  # [groups, out]
    g = g_idx.astype(np.int64)
    w = (w_codes - zeros[g]).astype(np.float32) * scales[g].astype(np.float32)
    return w.T


def sample_experts(layers: int, experts: int, n: int, seed: int) -> list[tuple[int, int]]:
    corners = {(0, 0), (0, experts - 1), (layers - 1, 0), (layers - 1, experts - 1)}
    rest = sorted({(la, e) for la in range(layers) for e in range(experts)} - corners)
    return sorted(corners | set(random.Random(seed).sample(rest, max(0, n - len(corners)))))


def check(store: Path, gptq_dir: Path, samples: int, seed: int) -> dict:
    gptq = LocalCheckpoint(gptq_dir)
    rows = []
    with ExpertStoreReader(store) as reader:
        layout = reader.layout
        cfg = json.loads((store / "store.json").read_text())["source"]["config"]
        for layer, expert in sample_experts(cfg["num_hidden_layers"], cfg["num_experts"], samples, seed):
            parsed = parse_expert_record(layout, reader.read_raw(layer, expert))
            for name, _ in layout.matrices:
                packed, scales = parsed[name]
                ours = dequantize_groupwise_int4(packed, scales, layout.group_size)
                p = f"model.layers.{layer}.mlp.experts.{expert}.{name}."
                ref = reference_dequantize(
                    *(gptq.get(p + k) for k in ("qweight", "qzeros", "scales", "g_idx"))
                )
                rows.append(
                    {
                        "layer": layer,
                        "expert": expert,
                        "matrix": name,
                        "shape": list(ref.shape),
                        "identical": bool(ref.shape == ours.shape and np.array_equal(ref, ours)),
                        "max_abs_diff": float(np.abs(ref - ours).max()) if ref.shape == ours.shape else None,
                    }
                )
            del parsed
    return {
        "matrices_checked": len(rows),
        "experts_checked": len({(r["layer"], r["expert"]) for r in rows}),
        "all_identical": all(r["identical"] for r in rows),
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", type=Path, default=MODELS_ROOT / "qwen1.5-moe-a2.7b-chat-int4g128-gptq")
    ap.add_argument("--gptq-dir", type=Path, default=checkpoint_dir(GPTQ_REPO, GPTQ_REVISION))
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    record = base_record(
        label="GPTQ store vs independent GPTQ dequantization (sampled experts)",
        seed=args.seed,
        model={"store": args.store.name, "gptq_repo": GPTQ_REPO, "gptq_revision": GPTQ_REVISION},
        config={"samples": args.samples},
        machine=collect_machine_profile(measure_disk=False),
    )
    record["summary"] = check(args.store.resolve(), args.gptq_dir.resolve(), args.samples, args.seed)
    record["peak_rss_mb"] = peak_process_rss_mb()
    append_benchmark_record(RESULTS, record)
    s = record["summary"]
    print(
        f"{s['matrices_checked']} matrices of {s['experts_checked']} experts: all identical = {s['all_identical']}"
    )
    if not s["all_identical"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
