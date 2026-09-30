# Limitations — what's real, what's simplified, what's a placeholder

This is the single source of truth for the current state of ExpertRelay.
If a claim elsewhere in the repo (README, docstrings, commit messages)
seems to conflict with this file, this file wins — file an issue against
the other one. See `docs/setup-notes.md` for the narrative history of how
each of these came to be.

## Package status

| Package | Status |
|---|---|
| `expertrelay.store` | Implemented: the full-model int8 expert store (`build_store`, `expert_reader`), selective HF fetch, pinned tokenizer, and a downloader for the original checkpoint (reference check only) |
| `expertrelay.runtime` | Implemented: the full 24-layer Qwen1.5-MoE-A2.7B forward pass from the int8 store, with KV cache and greedy generation, on a swappable compute backend (numpy with a fused numba int8 kernel for decode, MindSpore f32). Single device |
| `expertrelay.manager` | The machine profile and compute-backend selection. **No multi-device Manager exists** (see "Device split" below) |
| `expertrelay.cache` | `expert_cache`: the runtime's LRU expert cache with a fixed RAM budget, optional pinned layer 0 and background prefetch (`--source cached`). **Built and tested, not yet benchmarked** (see "Expert cache and prefetcher"). Offline simulators (`simulator`, `predictive`) replay recorded traces. RAM and one SSD only: no networked devices |
| `expertrelay.predictor` | `prefetch_policy`: the runtime prefetcher's choice (top-k of the Fate-style guess, low-confidence guesses skipped). **Built and tested, not yet benchmarked.** Offline tools (`offline`: scoring, calibration, reuse model), fitted on recorded traces. Without `--source cached`, reads are still synchronous and reactive |
| `expertrelay.bench` | `int8_kernels` (compute-path choice), `phase2_baselines` (normal load vs. OS paging vs. ours), `reference_check` (vs. HF transformers, bf16) |

## Device split: removed for now

The earlier two-process split (a coordinator and an expert server over a
TCP socket) ran on the reduced 2-layer fp32 MindSpore model. Phase 2
replaced that model with the full int8 runtime, and nothing else used the
split, so it was deleted rather than left as dead code (last present at
commit `93e5a7a`). **There is currently no networked-device path.** It
will be rebuilt on the int8 runtime, where an `ExpertSource` backed by a
remote device is the natural seam (`runtime/weights.py`).

## Compute backends: numpy on CPU, MindSpore for Ascend

The arithmetic runs behind a small interface (`runtime/backends.py`: int8
linear, f32 linear, attention) with two implementations:

- **numpy** (`NumpyBackend`): the blocked int8 kernel plus numpy/BLAS.
  Default on CPU.
- **MindSpore f32** (`runtime/backend_mindspore.py`): every matmul and the
  attention softmax as MindSpore ops. The path for Huawei Ascend.

The Manager picks one from the machine profile
(`manager/backend_selection.py`): an Ascend NPU means MindSpore, CPU only
means numpy. The choice and its reason go into every run's record. The
store, the expert sources, the cache, the predictor and the Manager only
handle plain numpy int8 arrays and never import a backend or MindSpore; a
test enforces this with a fresh-interpreter import check.

Both backends are tested:
- each primitive against float64 math;
- the same greedy tokens on a test model built through the real store
  builder, with logits within 1e-4 (different matmul kernels, so close
  rather than bit-identical).

In a one-off, unrecorded spot check on the real 24-layer store, both
produced the same 6 tokens, with MindSpore about 10x slower end to end
(0.07 vs. 0.72 tok/s). Only the kernel measurement below is a recorded
benchmark.

**What the MindSpore backend is not yet:**
- It has never run on Ascend: this machine has no NPU, only the CPU wheel.
  `device` goes straight to `mindspore.set_device`, so "Ascend" is one
  argument away, but untested.
- Arrays cross numpy <-> MindSpore on every call. On CPU that's harmless,
  but on an NPU it would copy host <-> device each time. A fast Ascend path
  needs weights uploaded once and activations kept on the device. Today
  it's a correct path, not a fast one.
- Elementwise work (norms, RoPE, activations, routing) stays in numpy for
  both backends. Routing in particular: the same router logits pick the
  same experts whichever backend computed them.

### Why numpy is the CPU default: the measurement

`bench/int8_kernels.py` timed every realistic way to compute an int8 x f32
matmul on this CPU (i5-12450H), on
the model's real shapes. Decode step (one token), median ms, and relative
error vs. float64 on the same int8 weights:

| Shape | numpy blocked (chosen) | numpy factored | numpy dequantize | MindSpore f32 | MindSpore f16 |
|---|---|---|---|---|---|
| attention 2048x2048 | **1.5** | 4.2 | 8.8 | 19.6 | 23.3 |
| expert gate 1408x2048 | **1.1** | 2.9 | 5.8 | 11.4 | 17.6 |
| expert down 2048x1408 | **1.2** | 4.9 | 5.9 | 7.1 | 16.0 |
| shared expert 5632x2048 | **4.3** | 10.9 | 22.7 | 56.5 | 88.9 |
| lm_head 151936x2048 | **118** | � | � | � | � |
| relative error | 2e-7 to 3e-7 | 2e-7 to 3e-7 | 2e-7 to 3e-7 | 4e-7 to 9e-7 | **3e-4** |

Source: `benchmarks/results/int8_kernels.json`, commit `90b5e9e`, clean tree.
"�" means skipped: those paths need a 1.24 GB f32 copy of lm_head.

- "numpy blocked" converts int8 rows to f32 a block at a time into a
  reused, cache-sized buffer, and applies the per-row scale to the output
  rather than to the weights (`runtime/int8_linear.py`). The block size is
  128 rows for decode (fastest on all five shapes; 1 MB fits the L2 cache)
  and 512 for prefill.
- For a 32-token prefill it isn't uniformly fastest: plain "factored" edges
  it out on the expert down-projection (4.1 vs. 4.7 ms). It's still chosen
  because decode dominates generation time, and it's the only option that can compute lm_head at all: the others need a
  1.24 GB f32 copy of it, more RAM than this machine has free.
- MindSpore f32 CPU ops were 6-13x slower on decode. MindSpore f16 was
  slower still, with ~1000x the error. These are MindSpore's CPU kernels on
  this machine; on Ascend hardware the choice would need re-measuring.
- MindSpore is imported only when its backend is selected (it costs
  ~195 MB of RAM just to load; the budget estimate adds it then).

## The runtime vs. the original model

- **Architecture checked against Hugging Face transformers' own
  Qwen2MoE code, automatically.** `tests/test_runtime_vs_hf_transformers.py`
  builds a tiny random Qwen2MoE through the real store builder, loads the
  dequantized int8 weights into `transformers.Qwen2MoeForCausalLM`, and
  requires our logits to match HF's to 1e-4, both for prefill and for
  token-by-token KV-cache decoding. The tiny model uses grouped KV heads
  (2 KV for 4 query heads) to exercise that path, which the real model
  (16/16) doesn't need.
- **Real model vs. bf16: `bench/reference_check.py`, results in
  `docs/reference-check.md`.** Layer-by-layer comparison of our int8
  hidden states and logits against HF transformers on the ORIGINAL bf16
  weights, same input (checkpoint shards sha256-verified against the
  Hub). Measured 2026-09-29 on the 4 Phase 2 prompts, 223 positions: the
  int8 model's top next token equals bf16's at 98.2% of positions (97.7%
  of the 128 generated ones); mean KL(bf16 || int8) 0.0022 nats, max
  0.040. That is a small sample: 4 prompts, each continued with OUR
  greedy tokens (teacher-forced), so it measures per-step agreement, not
  whether long free-running generations stay identical. The HF side reads
  the shards two at a time: with all eight open it crashed on Windows
  (see `bench/reference_check.py`).
- **int8 changes outputs compared to bf16.** Weight-only, symmetric,
  per-output-channel int8 (Krishnamoorthi 2018; the weight side of Dettmers
  et al. 2022 LLM.int8(), without its outlier decomposition) is lossy.
  Reconstruction error per tensor is in `docs/expert-store.md`: `lm_head`
  is the worst at 1.92%, routed experts average 0.83%. The int8 `lm_head`
  isn't what changes answers: with `lm_head` in fp16 the agreement is
  97.8% (vs 98.2%), mean KL 0.0015; the two differ in their top token at
  3 of 223 positions, in both directions. So `lm_head` stays int8.

## Phase 2 baselines: what the numbers mean

Full table: `docs/phase2-baselines.md` (generated from
`benchmarks/results/phase2_baselines.json`, two runs at `90b5e9e` and
`5406ea5`; the second, clean run is the one shown).

- **Ours (C): 0.70 decode tok/s, 12.5 s to first token, 1.7 GB working
  set.** Per decode token: 0.47 s reading 834 MB of experts (96 records)
  from the SSD, 0.95 s computing, so **compute-bound**, about 2:1. Prefill
  is where most expert reads happen: an 18-token prompt reads ~900
  expert records (7.8 GB), because across 18 tokens x 4 picks almost every
  expert in every layer gets used. That's why the first token takes
  10-16 s.
- **OS paging (B): 0.41 tok/s, 30 s to first token.** Likely causes, not
  separately measured: page faults bring the file in a few pages at a time
  and stall the matmul that touched them, versus one 8.7 MB read per expert
  in C; and in B the resident weights are memory-mapped too, so they can be
  evicted and re-read like the experts.
- **Normal load (A): never finished loading.** Windows let it commit
  14.6 GB (the whole store) but kept only 4.7 GB resident. The rest went to
  the page file, and the machine spent the whole 15-minute timeout paging:
  451 GB of disk reads system-wide. That is the failure mode on an 8 GB
  machine: not an out-of-memory error, but thrashing with no progress.
- **Correctness on the real model:** C and B produced identical tokens for
  all 4 prompts x 32 tokens, and so did the second run of each.
- **Output quality isn't evaluated.** The sample outputs are fluent in
  English, code and math, but the Arabic sample states a wrong date for
  Cairo and repeats itself. That's plausible for a base model under greedy
  decoding; the int8 model agrees with bf16 on the next token at 98.2% of
  positions on these prompts (`docs/reference-check.md`). No quality
  benchmark (accuracy on a task set) has been run.
- **Timing noise:** the runs share the machine with VS Code, Defender and
  a background download. Free RAM at start was 4.2 GB (run 1) and 3.0 GB
  (run 2); decode speed differed by 4% between runs.

## Phase 3: expert-usage traces and cache simulation

Results: `docs/phase3-analysis.md` (base store) and
`docs/phase3-analysis-qwen1.5-moe-a2.7b-chat-int8.md` (Chat store), both
generated from `benchmarks/results/phase3_analysis_<store>.json` and
`phase35_prediction_<store>.json`; the two compared in
`docs/base-vs-chat.md`. The Chat runs wrap every prompt in the ChatML
template (a system prompt, 36 prompt tokens on average vs 17), so a
difference between the two is a difference between the two setups, not
the effect of fine-tuning alone. Each store has its own prefetch
calibration (see "Expert cache and prefetcher"). What these
results are and aren't:

- **Measured:** which experts the int8 model's router picked, for 96 fixed
  prompts x 128 greedy tokens (`runtime/expert_trace.py`, format in
  `docs/trace-format.md`). Tracing is tested not to change any token.
- **Routing of the int8 model, not bf16.** int8 can change which experts get
  picked (see "The int8 expert store"). On the Phase 2 prompts the same
  top-4 set is picked 90-97% of the time per layer
  (`docs/reference-check.md`); the traces' routing statistics are the int8
  model's.
- **Greedy decoding, fixed length, EOS ignored.** Past a natural ending,
  the base model under greedy decoding often repeats itself, and
  repetition probably inflates temporal locality and reuse compared with
  sampled, naturally-ending text. The prompts are 16 per category, all
  written for this repo: a small, specific sample, not a corpus.
- **Simulated, not measured:** cache hit rates. The simulator replays
  exactly what the runtime loads (each layer's unique experts per forward
  call), but no cache exists in the runtime yet. Assumptions: all experts
  the same size; capacity counted in experts (x 8.67 MB of RAM each); cache
  warm across prompts; "all prompts" = two held-out halves of the run
  (even/odd prompts), each replayed in run order, with every policy seeing
  the same two streams. The pinned policy's hot set always comes from the
  OTHER half (or, per category, the other categories); pinned experts still
  pay their first load. Belady needs the future: an upper bound, not a
  buildable policy.
- **Fixed: the halves were whole categories.** Until commit c14de18,
  "every other prompt in run order" put {en, ar_gulf, math} in one half
  and {ar_msa, code, chat} in the other, because the run cycles through
  the 6 categories. The halves are now odd/even prompt numbers within
  every category (`phase3_analysis.prompt_halves`), and the analysis was
  rerun (the last record in `phase3_analysis_<store>.json`). It made almost
  no difference: the "LRU + pinned hot set" decode hit rate went from
  4.8 / 24.2 / 37.7 / 63.6% to 4.9 / 24.2 / 37.7 / 63.7% at 96 / 192 /
  384 / 720 experts, still below plain LRU (13.4 / 25.2 / 40.4 / 66.2%).
  The other rows moved by at most 0.1 point.
- **Projected, not measured:** decode tokens/s at each cache size = Phase
  2's measured compute time per token + simulated misses x Phase 2's
  measured read time per expert. It assumes reads and compute don't
  overlap, and the same machine load as Phase 2. Every table and chart that
  shows it is labeled a projection.
- **Fate-style predictor: accuracy only.** The trace records Fate's
  cross-layer guess (Fang et al., arXiv:2502.12224), and the analysis
  measures how much of the true top-4 it catches. Nothing acts on it yet:
  no prefetching exists, so no speedup from prediction has been measured.

## Phase 3.5: prediction and caching, simulated

Results: section 7 of `docs/phase3-analysis.md`, from
`benchmarks/results/phase35_prediction_<store>.json`
(`bench/phase35_prediction.py`). Everything above about the traces applies.
In addition:

- **Held out within each category.** Odd-numbered prompts of every
  category tune (predictor weights, isotonic calibration, the reuse table,
  the choice of "best" combination); even-numbered prompts are the only
  ones reported. 48 + 48 prompts: small, so differences of a point or two
  between policies are within noise.
- **Decode tokens only for prediction.** In prefill, a layer's picks for
  every prompt token come out of one call, so there is nothing "recent" to
  use ahead of it. The simulated cache still replays prefill loads (it
  stays warm through them); only decode hits are reported.
- **Two layers ahead is not Fate.** Fate's gate two layers ahead needs the
  hidden state, which the traces don't record. The two-ahead numbers use
  expert-to-expert transition tables instead (from the picks two layers
  back, or Fate's one-ahead guess pushed one step further), so they show
  how much a SIMPLE two-ahead predictor loses, not how Fate itself would
  do. Measuring Fate two ahead needs a trace re-run that records router
  L+2 applied to layer L's gate input.
- **Layer 0** has no previous layer: its prediction is popularity (from the
  tuning prompts) plus recency, made during the previous token's last
  layer.
- **Simulated policies, not built ones.** "Prediction-aware eviction"
  scores every cached expert by its chance of being picked at its layer's
  next visit: the calibrated prediction for the next layer, a reuse table
  (gap since last use, recent count) for all other layers. Prefetching
  loads the top-k predicted experts for the next layer. A prefetch that
  isn't used costs a read and a cache slot; both are counted. With
  prefetching, the hit rate can exceed Belady's, which is optimal only
  for caches that load on demand.
- **Projected, not measured, and optimistic about overlap.** Compute per
  layer = Phase 2's 0.954 s per token / 24, the same for every layer
  (attention and expert work aren't separated). Reads go one at a time.
  A prefetch for layer L+1 is hidden behind ALL of layer L's compute; in
  the real forward pass the prediction is ready only after layer L's
  attention, so the real window is shorter. Two read times: Phase 2's
  end-to-end runtime figure (4.9 ms per expert) and the raw uncached SSD
  benchmark (2.08 ms, `docs/machine-profile.md`), which the runtime
  doesn't reach today. Prefill time isn't modeled; the projection is
  decode speed.

## 4-bit experts (Step B2): half the bytes, too much quality lost

`docs/int4-experts.md`, from `benchmarks/results/int4_store_build.json`,
`quantization_quality.json` and `int4_benchmark.json`; store metadata
(sha256 of every record) in `benchmarks/stores/`.

- **Method: plain round-to-nearest only.** Symmetric 4-bit, float16 scale
  per group of 128 or 64 input columns, quantized from the original bf16
  weights, routed experts only (everything resident stays int8). No
  calibration data, no GPTQ or AWQ; those usually do much better at 4 bits,
  so these numbers are what RTN costs, not what 4-bit costs.
- **Size:** a record is 51.4% (group 128) or 53.0% (group 64) of int8's.
  Reconstruction error per expert: 11.8% / 10.8% mean, vs 0.83% for int8.
- **Quality: FAILS the pass rule** (fixed before the run: overall top-1
  agreement with bf16 >= 96% and every category >= 90%). Top-1 agreement:
  int8 97.8%, int4-g128 87.7%, int4-g64 89.6%; mean KL 0.0027 vs 0.19 /
  0.17 nats. Arabic suffers most (Gulf Arabic 81.1% / 83.2%, MSA 84.8% /
  89.2%); code least (93.2% / 95.9%).
- **Speed: +45%.** Same session, VS Code and browsers closed, Phase 2
  prompts, 1.25 GB cache for both: best int8 setup 2.41 tok/s (cache +
  adaptive prefetch), best int4-g64 setup 3.49 tok/s. Expert bytes read
  per token: 668-676 MB (int8, cached) vs 302-358 MB (int4); the cache
  holds 143 int8 experts vs 271 int4 ones. Compute per token rises
  0.09 -> 0.12-0.14 s (unpacking 4-bit values). Cache and prefetch on vs
  off gave identical tokens within each precision, in all 12 runs.
- **Verdict so far:** round-to-nearest int4 buys +45% speed at a cost of
  8 points of next-token agreement (13 for Arabic). Not good enough to
  use; the next candidates are GPTQ/AWQ-style 4-bit, or int4 only for
  the least-used experts.
- **The int4 speed runs used the base int8 store's prefetch calibration**
  (flagged in each run's `calibration_store_matches`); int4 routing
  differs slightly, so its adaptive prefetch is somewhat mis-calibrated.
- **Cache size was one expert short** in the Phase 4+5, Step B and B2
  benchmarks: the size was passed with 6 decimals (1.248657 GB), so the
  cache held 143 int8 experts instead of 144 (and 271 int4 ones instead
  of 272). Under 1%; fixed for future runs.
- **Sample:** 28 prompts (4 Phase 2 + 4 per category), 1,478 positions,
  each prompt continued by the int8 model and every model scored on those
  same positions (teacher forcing). Per-step agreement, not identity of
  long free-running generations.
- **Build memory:** the int4 builds peaked at 4.0-4.5 GB of process memory,
  against a 0.5 GB estimate. The difference is pages of the memory-mapped
  bf16 shards, which Windows counts in the process's working set but can
  drop at any time; the builder's own allocations stay small. The estimate
  doesn't count mapped file pages.

## Faster compute and what it changed (Step B)

`docs/stepb-benchmark.md` (`bench/stepb_benchmark.py`) and
`benchmarks/results/compute_profile.json` (`bench/compute_profile.py`).

- **Where compute went, before:** 0.99 s per generated token, 76% of it
  converting int8 weights to float32 before BLAS multiplied them (0.75 s),
  20% the multiplies, 3% everything else (norms, activations, routing,
  Python). By component: shared expert 0.35 s, routed experts 0.32 s,
  attention projections 0.16 s, lm_head 0.12 s.
- **The fused int8 kernel** (`runtime.int8_linear`, numba) multiplies the
  int8 weights directly for inputs of up to 4 rows. Compute per generated
  token: 0.99 -> 0.09 s in the profile, 1.14 -> 0.12 s in the benchmark.
  It is now the default. It also serves prefill whenever an expert gets
  4 tokens or fewer, so prefill got faster too. It sums in a different
  order; the reference check measured the same 98.2% agreement with bf16
  as the blocked kernel, token by token (`docs/reference-check.md`).
  Needs numba (a new dependency); without it the blocked kernel is used.
- **Reads are now the bottleneck.** With the fused kernel a layer's
  compute (the window a prefetch can hide behind) is ~4 ms, while one
  read takes ~4.2 ms alone and ~7 ms when two share the drive. Prefetching
  8 experts per layer can't finish in time: 29% of expert uses waited for
  an unfinished prefetch, and the extra reads (136 per token) slowed
  generation below no prefetching at all (1.61 vs 1.81 tok/s).
- **Adaptive prefetching** (reads per layer capped at window x I/O
  threads / read time) cuts prefetch reads from 136 to 30 per token and
  wasted ones from 72 to 2.5, and is faster than fixed top-8 (1.82 vs
  1.61 tok/s). Threshold 0.1 was not better than 0.05 (1.63 vs 1.82; one
  of its runs had a burst of very slow reads). The fastest setup measured
  was the cache with no prediction at all (1.99 tok/s): with compute this
  fast, prediction mostly adds disk traffic. It is now the configs'
  default to prefetch adaptively; whether to prefetch at all is open.
- **Pipelined prefill** (a layer's experts queued on the I/O threads as
  soon as its router has picked them): time to first token 8.5 s without
  it (fused kernel, no cache) vs 5.6-6.1 s with it.
- **"Idle drive is slower" (why the cache alone didn't help in Phase 4+5):
  only partly.** In the no-cache run (blocked kernel), reads after a 20-50
  ms idle gap took 6.6 ms (median) vs 4.3 ms after 1-5 ms; but in the
  cache-only run the same gaps showed no penalty (4.4 ms). And this time
  the cache alone DID help, in proportion to the reads it saved (0.61 ->
  0.67 tok/s, time blocked on reads -20% for -20% reads). Phase 4+5's
  "no gain" did not reproduce, so it was most likely that session's
  conditions.
- **This session was slower than Phase 4+5's:** the unchanged no-cache
  setup ran at 0.61 tok/s (compute 1.14 s/token) vs 0.71 (0.92 s) on
  2026-09-28. Phase 4+5 ran from a plain terminal with VS Code closed; this
  benchmark ran with VS Code open. Compare setups within one benchmark,
  not across them.
- 4 prompts x 32 tokens, base store only, one cache size (1.25 GB).

## Expert cache and prefetcher (Phase 4+5)

`--source cached` (`cache/expert_cache.py`, `predictor/prefetch_policy.py`,
hooks in `runtime/qwen_moe.py`), settings in `configs/runtime_cache.json`
or on the command line. What is and isn't established:

- **Measured once, on a small prompt set.** `docs/phase4-benchmark.md`
  (`bench/phase4_benchmark.py`): the 4 Phase 2 prompts x 32 tokens, each
  setup twice, 1.25 GB cache, 2 I/O threads. Decode speed went from 0.71
  tok/s (no cache) to 0.90 (prefetch top-8) and 0.95 (top-8 + layer 0
  pinned). That is 124 generated tokens per run, on one machine, run from
  a plain terminal with VS Code and other apps closed (3.35 GB of RAM free
  at the start); the two runs of each setup agree to
  within 0.04 tok/s. Not yet measured: other prompts, the Chat store,
  longer generations, other cache sizes.
- **Correctness on the real model:** all 10 runs gave tokens identical to
  that session's no-cache run and to Phase 2's recorded setup C. Tokens,
  not logits: the benchmark doesn't record logits. On the tiny synthetic
  model, cache and prefetcher on or off, cache smaller than one layer,
  layer 0 pinned, one or two I/O threads and the switch flipped
  mid-generation all give logits bit-identical to the all-in-RAM reference
  (`tests/test_runtime_correctness.py`).
- **Half the prefetched reads are wasted** at top-8 (81 of 166 reads per
  token). They're hidden behind compute today, but they are real SSD
  traffic and would compete with anything else using the disk.
- **Compute rises slightly with prefetching** (0.92 -> 0.97 s per token):
  the background reads and the extra router product share the CPU and
  memory bandwidth.
- **Cache without prefetch didn't help:** 19.7% hits cut reads from 96 to
  77 per token, but the time blocked on reads stayed at ~0.48 s. Reads
  became sparser, and pauses between reads are what makes them slow (see
  "The disk benchmark overstates what expert reads get"); that's the
  likely reason, not yet isolated.
- **Memory:** the cache is a fixed pool of whole expert slots
  (`expert_cache_gb` / 8.67 MB, rounded down), allocated once and counted
  in the process estimate that the budget is checked against. With the
  1.66 GB cache in `configs/runtime_cache.json` the estimate is 3.51 GB,
  under that config's 3.6 GB budget. That is a lot of an 8 GB machine
  with other work running; the benchmark must record free RAM.
- **Prefetch timing:** the guess for layer L+1 is computed right after
  layer L's router, not earlier. Background reads use their own file
  handle per I/O thread (1 by default); the forward pass's own demand
  reads run alongside, so up to two reads can be in flight.
- **Confidence threshold:** from a rank-aware isotonic calibration (one
  map per guess rank, `predictor.offline.RankedIsotonicCalibrator`) fitted
  on each store's own tuning-prompt traces
  (`bench/fit_prefetch_calibration.py`; `configs/runtime_cache.json` for
  base, `runtime_cache_chat.json` for Chat). It replaced a single map that
  was right on average but off by up to ~12 points per rank (e.g. the 3rd
  guess: 74% picked, 62% predicted). **This changes what the 0.05
  threshold does:** with the rank-aware map it skips 1.9% of top-8
  prefetches (base) instead of 8.1%, so the Phase 4+5 benchmark, which ran
  with the single map, isn't exactly today's configuration. A run whose
  calibration came from another store is flagged in its `load` info
  (`calibration_store_matches`). Chat runs now get the model's own chat
  template (`store.chat_template`, checked against transformers), the same
  text the Chat traces were recorded with, so the Chat calibration matches
  what the runtime feeds it. In prefill, each expert's score is its
  best probability over the prompt's tokens, and the per-token calibration
  is applied to that: a heuristic.
- **Layer 0** has no prefetch (no layer before it). `pin_layer0` keeps all
  60 of its experts in RAM (0.52 GB of the cache).
- **Per-layer timing** brackets synchronous calls with `perf_counter`:
  meaningful for the numpy backend. "Attention" includes the input norm;
  "MoE compute" includes the router, the Fate guess, the routed experts and
  the shared expert, minus time blocked on reads.
- Windows only, like the unbuffered reader.

## Correctness rule, as applied now

CLAUDE.md: output must match the reference token-for-token. Since the int8
store, the reference is **the int8 model with every expert held in RAM and
no cache or prediction**. bf16 is not the reference for that rule, because
int8 alone already changes tokens; that cost is measured separately (see
the reference check above).

- `tests/test_runtime_correctness.py`: on a tiny model built through the
  real store builder, experts read on demand (ours), fully in RAM
  (reference), and memory-mapped all give **bit-identical** logits and the
  same greedy tokens.
- On the real 24-layer store, `bench/phase2_baselines.py` records every
  setup's generated tokens and checks they're identical across the setups
  that finish.

## Resident weights: one exception to "in RAM"

The resident part of the store (attention, shared expert, router, norms,
lm_head: 1.56 GB) is read into RAM once. **The embedding table (311 MB) is
not:** it's memory-mapped, and only the rows for actual tokens get paged in
(one 2 KB row per token). On this 8 GB machine, with VS Code and the rest
of the dev environment running, only about 1-2 GB is actually free, so the
311 MB matters. `runtime/weights.py` documents this; the budget estimate
accounts for it.

## Generation

- **Greedy only.** No sampling, temperature, or top-p.
- **Fixed length:** exactly `max_new_tokens` (config) per prompt. EOS does
  not stop generation, so every benchmark run does the same work.
- **Base model, no chat template.** Prompts are plain text to continue.
  The Chat model's store is being built for the demo, but Phase 2 runs on
  the base store.
- **One sequence at a time.** No batching across prompts.

## Performance: what's still naive

- **Reads and compute don't overlap** with `--source unbuffered` (the
  Phase 2 baseline, by design). `--source cached` with prefetch on
  overlaps them; not measured yet.
- **No expert is kept between tokens** with `--source unbuffered`: an
  expert picked by two consecutive tokens is read from disk twice. The
  cached source keeps them.
- **The disk benchmark overstates what expert reads get.** The machine
  profile's 2.08 ms per 8.65 MB read comes from a 1 GiB file it writes
  seconds before reading; it still measures 4.0-4.2 GB/s (profile rerun,
  2026-09-28). Real store records, read with the same call, take ~4.2 ms
  back to back (2.0 GB/s) from `experts.bin`, and the same from a fresh
  contiguous copy made that day (`bench/read_diagnosis.py`,
  `benchmarks/results/read_diagnosis.json`). So neither fragmentation nor
  file age explains the gap. The most likely cause: the small test file
  sits in the SSD's fast write cache, while 12.5 GB of experts sit in its
  slower main flash (the 12.5 GB copy wrote at only 95 MB/s, which is what
  such a drive does once that cache is full). The flash type isn't in the
  profile, so this is an inference. What the diagnosis measured directly:
  - **Pauses cost the most:** bursts of 4 reads with 40 ms gaps, like the
    runtime's pattern, average 6.1-6.3 ms per read (median 4.8-4.9, p90
    10-13 ms) against 4.3 ms back to back. Phase 2's 4.9 ms per read sits
    between the two.
  - **Two reads in flight:** 2.5 GB/s total instead of 2.0 (+25%, both
    rounds, both files); four in flight gives no more.
  - **Records split across extents:** about 5% slower (+0.2-0.3 ms).
    `experts.bin` has 227 extents; 182 of 1,440 records are split.
  - **Fresh contiguous copy:** no clear gain (1-3%), so it wasn't swapped
    in. It's kept as `experts.bin.new`.
  Consequences: projections should use store-read times, not the
  profile's fresh-file number; the runtime uses two I/O threads
  (`io_threads: 2`); and keeping reads flowing (prefetching) avoids the
  pause penalty.
- **Python-level loop over experts and layers** around BLAS calls. Fine for
  measuring where time goes; it has per-call overhead a compiled runtime
  wouldn't.

## The int8 expert store (`expertrelay.store.build_store`)

The full Qwen1.5-MoE-A2.7B, all 24 layers and 1,440 routed experts, stored
on the SSD in int8 at a pinned Hugging Face revision. Format and numbers:
`docs/expert-store.md`.

- **Kept in fp32, deliberately:** norms, biases, the router (`mlp.gate`)
  and `shared_expert_gate`, about 12 MB total. That keeps the router itself
  from adding quantization error to routing decisions. It does NOT make
  expert selection identical to bf16: the router's input is the hidden
  state produced by the int8 layers before it, so which experts get picked
  can still differ from the bf16 model.
- **Source integrity:** Hugging Face publishes a sha256 per shard file, not
  per tensor, so byte ranges can't be checked against it without
  downloading whole shards. What is checked: every range response has
  exactly the requested length (a truncated or Range-ignoring response is
  rejected and retried), transport is TLS, and the revision is pinned to a
  commit hash. The store's own sha256s protect everything after the
  download.
- **Windows only** for the one-read loader (`expert_reader`), which uses
  `FILE_FLAG_NO_BUFFERING`. So the "ours" runtime path is Windows-only too.
  The mmap and in-RAM expert sources are portable.

## Test suite constraints

Tests must be deterministic and must not require network access. The
Hugging Face fetch path is therefore only tested offline (byte-level bf16
decoding, row ranges, name generation). Tests that use unbuffered reads are
skipped on non-Windows platforms. The HF-equivalence test needs the dev
extras (`torch`, `transformers`) and is skipped without them.

## Memory budget

`expertrelay.memory_budget.enforce_ram_budget` is a real, enforced gate
(raises `MemoryBudgetExceeded`). It runs before the store build starts
(`--max-ram-gb`, default 1.5 GB) and before the runtime loads any weights
(`memory_budget_gb` in `configs/runtime.json`, default 2.5 GB). It gates on
an estimate computed from real shapes (`runtime.generate
.estimate_ram_bytes`), not on measured RSS. Measured peak RSS is recorded in
every benchmark (`expertrelay.benchmarking`). The normal-load baseline runs
with the gate disabled on purpose, to show what happens without it.

## Machine profile (`expertrelay.manager.profile`)

- **Windows-only for the hardware-specific parts.** CPU model (registry),
  drive model/bus/media type (PowerShell `Get-Disk`/`Get-PhysicalDisk`), and
  the uncached disk-read test (Win32 `FILE_FLAG_NO_BUFFERING` via ctypes)
  are implemented for Windows only. On other platforms the descriptive
  fields are `None` (unknown, never guessed) and the disk-read test raises
  `NotImplementedError`. RAM, core counts, free space and software versions
  are portable (psutil / stdlib). Porting the read test to Linux (`O_DIRECT`)
  is needed before profiling a Linux device such as an Atlas board.
- **Windows file cache: bypassed, and verified per run.** Every timed run
  records the OS physical-disk read counters and is marked
  `cache_bypass_verified` only if the device served at least the bytes the
  run requested. A one-off control on the dev machine read a 536 MB file
  twice: the cached pass registered 0 MB of device reads, the
  `NO_BUFFERING` pass registered exactly 536.3 MB.
- **SSD-internal caching: NOT bypassed.** The test file is written right
  before it's read, so it most likely sits in the drive's SLC write cache.
  That can read faster than data that has aged into TLC/QLC NAND. Treat the
  numbers as an upper bound for loading cold experts that have sat on disk
  for a while.
- **"Random" means large reads at random offsets**, one expert-sized chunk
  (8.65 or 17.30 MB) at a random 4 KiB-aligned offset. That's the realistic
  pattern for loading one expert. It is not a 4K random-IOPS test.
- **Single-threaded, queue depth 1**, like the runtime's expert loads today.

## Downloads

The original bf16 checkpoint (reference check) and the Chat model's
weights (demo store) are fetched over this machine's ~0.5-2 MB/s link, so
each takes many hours. `store/download_checkpoint.py` runs with
`HF_HUB_DISABLE_XET=1`: the default Xet backend stalled after ~730 MB, with
no bytes written for 5+ minutes, while plain HTTP resumed immediately.
