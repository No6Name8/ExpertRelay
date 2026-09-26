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
| `expertrelay.runtime` | Implemented: the full 24-layer Qwen1.5-MoE-A2.7B forward pass from the int8 store, with KV cache and greedy generation, on a swappable compute backend (numpy, MindSpore f32). Single device |
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
  weights, same input. It needs those weights on disk, and they're
  still downloading at the time of writing (28.6 GB). Until that file
  exists, the real-model accuracy of the int8 runtime is **unmeasured**.
- **int8 changes outputs compared to bf16.** Weight-only, symmetric,
  per-output-channel int8 (Krishnamoorthi 2018; the weight side of Dettmers
  et al. 2022 LLM.int8(), without its outlier decomposition) is lossy.
  Reconstruction error per tensor is in `docs/expert-store.md`: `lm_head`
  is the worst at 1.92%, routed experts average 0.83%.

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
  decoding, and it's from int8 weights whose effect vs. bf16 is still
  pending (reference check, below). No quality benchmark has been run.
- **Timing noise:** the runs share the machine with VS Code, Defender and
  a background download. Free RAM at start was 4.2 GB (run 1) and 3.0 GB
  (run 2); decode speed differed by 4% between runs.

## Phase 3: expert-usage traces and cache simulation

Results: `docs/phase3-analysis.md` (base store; generated from
`benchmarks/results/phase3_analysis_<store>.json`). What they are and aren't:

- **Measured:** which experts the int8 model's router picked, for 96 fixed
  prompts x 128 greedy tokens (`runtime/expert_trace.py`, format in
  `docs/trace-format.md`). Tracing is tested not to change any token.
- **Routing of the int8 model, not bf16.** int8 can change which experts get
  picked (see "The int8 expert store"). How much is part of the pending
  reference check.
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
- **Known issue: the even/odd halves are whole categories.** The run
  cycles through the 6 categories, so taking every other prompt in run
  order puts {en, ar_gulf, math} in one half and {ar_msa, code, chat} in
  the other. The pinned policy's "all prompts" hot set therefore comes from
  three OTHER categories, which is stricter than intended and likely
  understates it. Not rerun yet; Phase 3.5 (below) splits within each
  category instead.
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

## Expert cache and prefetcher (Phase 4+5): built, not yet measured

`--source cached` (`cache/expert_cache.py`, `predictor/prefetch_policy.py`,
hooks in `runtime/qwen_moe.py`), settings in `configs/runtime_cache.json`
or on the command line. What is and isn't established:

- **No speed has been measured.** Tokens/s, hit rates on the real model,
  and the per-layer attention / MoE / read-wait split are instrumented
  (`generate.StepRecord`) but have not been run on the 24-layer model.
  Every speed figure for the cache so far is a projection from traces
  (`docs/phase3-analysis.md`, sections 7-8).
- **Correctness is tested on the tiny synthetic model only:** cache and
  prefetcher on or off, cache smaller than one layer, layer 0 pinned, one
  or two I/O threads, the switch flipped mid-generation, all give logits
  bit-identical to the all-in-RAM reference
  (`tests/test_runtime_correctness.py`). On the real model the same check
  (tokens with the cache and prefetcher on vs. off) is still to be run with
  the benchmarks.
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
- **Confidence threshold:** from the isotonic calibration fitted on the
  BASE model's tuning-prompt traces
  (`bench/fit_prefetch_calibration.py`). A Chat-store run reuses it until
  a Chat calibration is fitted; the run's `load` info flags this
  (`calibration_store_matches`). In prefill, each expert's score is its
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

- **Reads and compute don't overlap.** Each expert is read, then used, then
  the next one is read. That's the no-cache, no-prediction baseline by
  design; overlapping them is what the predictor is for.
- **No expert is kept between tokens.** An expert picked by two
  consecutive tokens is read from disk twice. That's the no-cache baseline
  by design.
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
