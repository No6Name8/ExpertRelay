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
| `expertrelay.runtime` | Implemented: the full 24-layer Qwen1.5-MoE-A2.7B forward pass from the int8 store, with KV cache and greedy generation. Single device |
| `expertrelay.manager` | Only the machine profile. **No multi-device Manager exists** (see "Device split" below) |
| `expertrelay.cache` | **Placeholder only.** Every routed expert is read from disk each time it's picked and dropped after use |
| `expertrelay.predictor` | **Placeholder only.** Expert reads are synchronous and reactive: the router decides, then the read happens, then compute |
| `expertrelay.bench` | `int8_kernels` (compute-path choice), `phase2_baselines` (normal load vs. OS paging vs. ours), `reference_check` (vs. HF transformers, bf16) |

## Device split: removed for now

The earlier two-process split (a coordinator and an expert server over a
TCP socket) ran on the reduced 2-layer fp32 MindSpore model. Phase 2
replaced that model with the full int8 runtime, and nothing else used the
split, so it was deleted rather than left as dead code (last present at
commit `93e5a7a`). **There is currently no networked-device path.** It
will be rebuilt on the int8 runtime, where an `ExpertSource` backed by a
remote device is the natural seam (`runtime/weights.py`).

## The runtime computes in numpy, not MindSpore

This is a measured choice, and it contradicts "built on MindSpore" for the
CPU path, so it's stated up front. `bench/int8_kernels.py` timed every
realistic way to compute an int8 x f32 matmul on this CPU (i5-12450H), on
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
- MindSpore is still a dependency (the kernel benchmark uses it), but the
  runtime imports none of it.

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
