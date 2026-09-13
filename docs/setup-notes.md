# Setup notes — MindSpore verification & MoE model survey

Date: 2026-09-13

## 1. MindSpore installation check

- **Version:** 2.7.1 (Python 3.10.0)
- **Backend:** CPU only. The installed wheel does not support Ascend or GPU —
  attempting `ms.set_context(device_target="Ascend")` or `"GPU"` raises
  `Unsupported device target ... This process only supports one of the ['CPU']`.
  To get Ascend support we'd need the Ascend-specific wheel (built with
  `-e Ascend`) plus the CANN toolkit installed, which this machine doesn't have.
- **Sanity check:** ran a basic `ops.matmul` on two 2x2 tensors on the CPU
  backend — returned the correct result (`[[19,22],[43,50]]`, sum `134.0`).
  MindSpore's core tensor ops work correctly on this machine.
- **Hardware constraints found during this session:** 8.3 GB total system RAM,
  ~0.8 GB free at time of testing, 12 logical CPUs. This matters a lot for
  what's feasible below.

## 2. MoE models in MindSpore's own ecosystem

- **MindSpore Hub:** no MoE models. Hub's pretrained models are concentrated
  in image classification, object detection, semantic segmentation, and
  recommendation — no LLM/MoE checkpoints there.
- **MindFormers (MindSpore Transformers):** this is where MoE support
  actually lives. It has **native architecture code** (not just docs) for:
  - `deepseek3` — DeepSeek-V3 (671B total / 37B active)
  - `qwen2_moe` — Qwen1.5/Qwen2 MoE family (e.g. Qwen1.5-MoE-A2.7B: 14.3B
    total / 2.7B active)
  - `qwen3_moe` — Qwen3 MoE family
  - MindFormers docs also list DeepSeek-V2 (236B, Sparse LLM) as supported.
  - No genuinely small (<5B total) MoE architecture is natively supported —
    everything on the list is a real frontier-scale model.
- **MindOne** (mindspore-lab/mindone): mentioned in search results as
  supporting `qwen3_omni_moe` and `qwen3_vl_moe`, but these are
  multimodal/omni variants, not a smaller/simpler entry point.
- **Conclusion:** MindSpore's own ecosystem has MoE *architecture* code, but
  no pretrained MoE *checkpoints* ready to just download and run — everything
  requires bringing your own (converted) weights, and the models it does
  support are all large.

## 3. Small open-source MoE candidates for conversion

Two realistic candidates, ranked by conversion feasibility rather than by
raw size (both are "small" only relative to frontier MoE, not relative to
this machine):

1. **Qwen1.5-MoE-A2.7B** (14.3B total params, 2.7B activated, 60 experts,
   4 routed + shared experts per token). **Best candidate** because
   MindFormers already ships `qwen2_moe` architecture code — the routing,
   expert-FFN, and shared-expert logic don't need to be written from
   scratch. Conversion work would be:
   - Download HF safetensors weights.
   - Run/adapt MindFormers' existing `convert_weight.py`-style script for
     the Qwen2 family to remap HF tensor names → MindSpore parameter names
     (attention/MLP naming conventions differ between HF `transformers` and
     MindFormers).
   - Match the `qwen2_moe` config (hidden size, expert count, top-k, shared
     expert flag) to Qwen1.5-MoE-A2.7B's actual config.
   - Validate numerically (e.g. compare logits on a few tokens against the
     HF reference implementation) since expert-index/weight-normalization
     conventions are a common source of silent bugs in MoE ports.
   - **Blocker on this machine:** 14.3B params is ~28 GB in fp16 or ~14 GB
     in int8 — far beyond the 8.3 GB total RAM available here.

2. **DeepSeek-V2-Lite** (15.7B total, 2.4B active, 64 routed + 2 shared
   experts). MindFormers has `deepseek3` code for the newer V3 architecture,
   which is closely related but not identical to V2's MoE layer design
   (V3 adds changes like different load-balancing and routing details), so
   this would need real architecture adaptation, not just a weight-name
   remap — higher effort than option 1.
   - **Same RAM blocker** as above (15.7B total).

A third option considered and **rejected**: OLMoE-1B-7B (7B total / 1B
active, fully open, Apache 2.0) is smaller in total params than either
option above, but MindFormers has no existing OLMoE architecture code, so
the MoE routing/dispatch logic would need to be written from scratch — more
engineering effort for a model that still doesn't fit in 8 GB RAM anyway
(7B params ≈ 14 GB fp16 / ~7 GB int8, i.e. still at or over this machine's
total memory with nothing else running).

**Net assessment:** every real pretrained MoE checkpoint we could find,
including the smallest genuinely open ones, needs more RAM than this
machine has free. Converting one here would hit an out-of-memory wall
before we could even validate correctness.

## 4. Single-device inference attempt

Given the RAM ceiling above, downloading/converting a real checkpoint in
this session wasn't viable. Instead, to still validate the mechanics before
committing to a specific model, I wrote a small **native MindSpore MoE
sanity check**: [`src/moe_sanity_check.py`](../src/moe_sanity_check.py).

- Randomly-initialized (untrained) decoder: embedding → 2× MoE layers
  (router + top-k dispatch over 4 experts, each a 2-layer FFN) → LayerNorm →
  output head. Vocab size 64, hidden size 32 — tiny on purpose.
- Ran a forward pass and 5 steps of greedy decoding on CPU.
- **Result: ran successfully, no errors.** Output token ids were e.g.
  `[41, 42, 58, 31, 56, 1, 1, 39, 36, 23, 36, 23, 36]` — the repeating
  `36, 23, 36, 23...` tail is the expected behavior of greedy decoding on an
  *untrained* model (it collapses into a short cycle), not a bug.
- **What this confirms:** MindSpore's `nn.Dense`, `nn.Embedding`,
  `ops.top_k`, `ops.softmax`, and control flow needed for MoE routing +
  expert dispatch + combine all work correctly together on this CPU install,
  and a generation loop (KV-cache-free, naive re-forward-pass-per-token)
  runs end to end.
- **What this does NOT confirm:** nothing about real MoE checkpoint loading,
  weight conversion correctness, or linguistically coherent output — that
  requires a real trained model, which we couldn't fit on this machine in
  this session (see §3).

## What worked / what didn't

**Worked:**
- MindSpore 2.7.1 CPU backend, basic tensor ops.
- Native MindSpore MoE routing/dispatch/generation mechanics (toy model).
- Identifying that MindFormers has ready-made `qwen2_moe` / `qwen3_moe` /
  `deepseek3` architecture code as a real head start for future conversion
  work.

**Didn't work / blocked:**
- No pretrained MoE checkpoint (from MindSpore's ecosystem or converted from
  HF) could be loaded and run on this machine — every realistic candidate
  needs more RAM than is available (8.3 GB total, ~0.8 GB free).
- Ascend/GPU backends aren't usable with the currently installed wheel.

## Suggested next steps

- Test on a machine with more RAM (16 GB+) or the actual target Ascend
  hardware (Atlas 200I DK A2) before attempting real checkpoint conversion.
- If Qwen1.5-MoE-A2.7B is still the target, start the weight-conversion
  script now (it doesn't require the full model resident in RAM if done
  shard-by-shard) even before a bigger machine is available, so it's ready
  once resources allow a load test.
