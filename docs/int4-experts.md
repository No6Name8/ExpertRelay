# 4-bit experts (int4): what they save and what they cost

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

<!-- appended sections: kept when this doc is regenerated -->

<!-- section:1_stores -->
## The stores

Generated from `benchmarks/results/int4_store_build.json` (latest record per group size). Reconstruction error per expert = ||W - W_hat|| / ||W|| over its three matrices; max error / step is bounded by 0.5 when nothing clips.

| store | record (bytes) | vs int8 | experts total | error per expert: mean / median / max | max error / step | records verified | build commit |
|---|---|---|---|---|---|---|---|
| qwen1.5-moe-a2.7b-int4g128 (group 128) | 4,460,544 | 51.4% | 6.42 GB | 11.79% / 11.78% / 12.35% | 0.5000 | 1440 / 1440 | 59dd078 |
| qwen1.5-moe-a2.7b-int4g64 (group 64) | 4,595,712 | 53.0% | 6.62 GB | 10.81% / 10.80% / 11.17% | 0.5000 | 1440 / 1440 | b977146 |
| int8 store, for comparison (row-wise scales) | 8,671,232 | 100% | 12.49 GB | 0.83% / 0.83% / 1.03% | 0.5000 | - | - |
<!-- /section:1_stores -->

