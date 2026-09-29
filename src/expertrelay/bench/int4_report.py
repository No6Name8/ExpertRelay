"""docs/int4-experts.md: the int4 expert stores, their quality and speed.

    python -m expertrelay.bench.int4_report          # redo the stores section from the build records

The doc is a fixed introduction (method and citations, the same as
store/int4.py) followed by generated sections, each owned by one script
and rebuilt from its results file:
  1_stores   store.build_int4_store  -> benchmarks/results/int4_store_build.json
  2_quality  bench.quantization_quality -> benchmarks/results/quantization_quality.json
  3_speed    bench.int4_benchmark    -> benchmarks/results/int4_benchmark.json
"""

from __future__ import annotations

import json
from pathlib import Path

from expertrelay import REPO_ROOT
from expertrelay.bench.phase3_analysis import APPENDED_SECTIONS_MARKER, upsert_doc_section
from expertrelay.paths import BENCHMARK_RESULTS_DIR

DOC = REPO_ROOT / "docs" / "int4-experts.md"
BUILD_RESULTS = BENCHMARK_RESULTS_DIR / "int4_store_build.json"
SECTION_NAMES = {"stores": "1_stores", "quality": "2_quality", "speed": "3_speed"}

HEAD = """# 4-bit experts (int4): what they save and what they cost

The routed experts are what the runtime reads from the SSD for every
token; everything else stays in RAM. This compares the int8 expert store
with two int4 versions of it, built from the ORIGINAL bf16 weights, not
from the int8 store (`store/build_int4_store.py`). The resident part
(attention, shared expert, router, norms, embedding, lm_head) is the int8
store's, byte for byte, in all three.

**Method** (`store/int4.py`): weight-only, symmetric round-to-nearest (RTN),
one float16 scale per group of 128 (or 64) consecutive input columns,
4-bit values -7..7, no zero point. The scale is stored in float16 and the
weights are quantized with the stored value, moved up one float16 step when
rounding would otherwise clip, so every weight is within half a step. No
calibration data is used.

- Group-wise weight scales: Z. Yao, R. Y. Aminabadi, M. Zhang, X. Wu, C. Li,
  Y. He, "ZeroQuant: Efficient and Affordable Post-Training Quantization for
  Large-Scale Transformers", NeurIPS 2022, arXiv:2206.01861.
- Small independently quantized blocks as the main lever for 4-bit RTN:
  T. Dettmers, L. Zettlemoyer, "The case for 4-bit precision: k-bit
  Inference Scaling Laws", ICML 2023 (PMLR 202, pp. 7750-7774),
  arXiv:2212.09720.
- **Not used:** GPTQ (E. Frantar, S. Ashkboos, T. Hoefler, D. Alistarh,
  "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained
  Transformers", ICLR 2023, arXiv:2210.17323) and AWQ (J. Lin, J. Tang,
  H. Tang, S. Yang, W.-M. Chen, W.-C. Wang, G. Xiao, X. Dang, C. Gan, S. Han,
  "AWQ: Activation-aware Weight Quantization for LLM Compression and
  Acceleration", MLSys 2024, arXiv:2306.00978). Both use calibration data
  and usually beat RTN at 4 bits; the numbers below are for plain RTN.
"""


def upsert_section(name: str, text: str) -> Path:
    if not DOC.exists():
        DOC.write_text(HEAD + "\n" + APPENDED_SECTIONS_MARKER + "\n", encoding="utf-8")
    upsert_doc_section(DOC, SECTION_NAMES[name], text)
    return DOC


def _pct(x: float) -> str:
    return f"{x:.2%}"


def stores_section(records: list[dict]) -> str:
    latest = {}
    for r in records:
        latest[r["config"]["group_size"]] = r
    lines = [
        "## The stores",
        "",
        "Generated from `benchmarks/results/int4_store_build.json` (latest record per group size). "
        "Reconstruction error per expert = ||W - W_hat|| / ||W|| over its three matrices; max error / step "
        "is bounded by 0.5 when nothing clips.",
        "",
        "| store | record (bytes) | vs int8 | experts total | error per expert: mean / median / max | "
        "max error / step | records verified | build commit |",
        "|---|---|---|---|---|---|---|---|",
    ]
    int8 = None
    for g in sorted(latest, reverse=True):
        s = latest[g]["summary"]
        int8 = s["int8_for_comparison"]
        e = s["quant_error"]
        lines.append(
            f"| {s['store']} (group {g}) | {s['record_size']:,} | {s['record_size'] / int8['record_size']:.1%} | "
            f"{s['experts_bin_bytes'] / 1e9:.2f} GB | {_pct(e['rel_fro_error_mean'])} / "
            f"{_pct(e['rel_fro_error_median'])} / {_pct(e['rel_fro_error_max'])} | "
            f"{e['max_abs_error_over_scale']:.4f} | {s['records_verified'] - len(s['bad_records'])} / "
            f"{s['records_verified']} | {latest[g]['git_commit'][:7]} |"
        )
    if int8:
        e = int8["quant_error"]
        lines.append(
            f"| int8 store, for comparison (row-wise scales) | {int8['record_size']:,} | 100% | "
            f"{int8['experts_bin_bytes'] / 1e9:.2f} GB | {_pct(e['rel_fro_error_mean'])} / "
            f"{_pct(e['rel_fro_error_median'])} / {_pct(e['rel_fro_error_max'])} | "
            f"{e['max_abs_error_over_scale']:.4f} | - | - |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    print(f"wrote {upsert_section('stores', stores_section(json.loads(BUILD_RESULTS.read_text())))}")


if __name__ == "__main__":
    main()
