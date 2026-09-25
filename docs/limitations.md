# Limitations — what's real, what's simplified, what's a placeholder

This is the single source of truth for the current state of ExpertRelay.
If a claim elsewhere in the repo (README, docstrings, commit messages)
seems to conflict with this file, this file wins — file an issue against
the other one. See `docs/setup-notes.md` for the narrative history of how
each of these came to be.

## Package status

| Package | Status |
|---|---|
| `expertrelay.store` | Implemented: selective HF safetensors fetch, the full-model int8 expert store (`build_store`, `expert_reader`), and the older reduced MindSpore checkpoint converter (`convert_qwen_moe`). Nothing loads the int8 store into the model yet |
| `expertrelay.runtime` | Implemented: hand-rolled MoE model, wire protocol, expert-serving process |
| `expertrelay.manager` | Minimal implementation: static two-device expert dispatch. **Not yet** the general hot/cold, multi-device Manager described in README |
| `expertrelay.cache` | **Placeholder only** — no eviction, no hot/cold placement, no runtime cache. Today's "placement" is a static assignment fixed at conversion time (`--process-a-experts`/`--process-b-experts`) |
| `expertrelay.predictor` | **Placeholder only** — no prediction of any kind. Every expert call is synchronous and reactive: the router decides, then (and only then) the request goes out |
| `expertrelay.bench` | Implemented: `moe_sanity_check` (tiny, self-contained), `run_local_split_demo` (orchestrates the two-process simulation) |

## The model: hand-rolled, not `mindformers`

The `mindformers` PyPI package is **not installed and not a dependency** —
`pyproject.toml` lists only `mindspore`, `numpy`, `tokenizers`, `psutil`.
`expertrelay.runtime.moe_model` is a from-scratch MindSpore implementation
of Qwen1.5-MoE-A2.7B's architecture (HF `model_type: qwen2_moe`).

Why: MindFormers does not actually ship a `qwen2_moe` architecture (an
earlier note in `docs/setup-notes.md` claiming it did was wrong and has
been corrected there). What it has is `qwen3_moe` and `deepseek3` — neither
is Qwen1.5-MoE-A2.7B's real architecture. Rather than force-fit a different
model family's code, this repo implements qwen2_moe directly.

Parameter names follow MindFormers' naming *vocabulary* (`decoder.layers.N
.self_attention...`, `mlp.router`, `gating`/`hidden`/`linear_fc2`, lifted
from the real `mindformers/models/qwen3_moe/utils.py`) so a checkpoint from
this repo reads like a MindFormers-style checkpoint — but it is not
loadable by MindFormers' own model classes, and no code here imports
`mindformers`.

**Deliberate simplification:** real MindFormers fuses Q/K/V into one
`linear_qkv` matrix and stacks all experts into two `weight1`/`weight2`
tensors (see `mindformers/checkpoint/converter/convert_op.py`) for its
tensor-parallel / expert-parallel *training* kernels. This repo keeps
attention and experts **unfused** — one `Linear` per Q/K/V/O and per
expert projection — because that fusion is a training-performance detail
orthogonal to what's being validated here (does the split/dispatch/
conversion pipeline work). Labeled in `moe_model.py`'s module docstring.

## The reduced model

Every real run in this repo so far uses a truncated slice of the real
Qwen1.5-MoE-A2.7B checkpoint (fetched live from Hugging Face — the weight
*values* are real, only the *quantity* kept is reduced):

- **2 of the real 24 layers.**
- **4 of the real 60 experts** (ids 0,1,2,3 in the validated run; the
  `expertrelay.store.convert_qwen_moe` CLI takes `--num-layers`/
  `--expert-ids` generically, so this isn't hardcoded — just what fits in
  8GB RAM on the dev machine so far).
- **Router softmax computed over the kept experts only** (4, not the real
  60) — the real per-expert router weight *rows* are used, but the
  denominator is smaller than the real model's, so routing-weight
  magnitudes differ from what the real 60-expert model would produce.
  Labeled in `MoELayer.construct`'s comment.
- **`top_k` overridden to 2** (real model: 4). With only 4 total experts
  loaded, "top-4 of 4" is "always use all of them" — it wouldn't exercise
  local/remote dispatch at all. Labeled in `convert_qwen_moe.py`'s CLI help
  and module docstring.
- **Consequence for output quality:** the real `lm_head` was trained to
  decode a residual stream that has been through 24 layers; feeding it one
  that's been through 2 produces real vocabulary tokens (confirmed: the
  prompt echoes back correctly) but not fluent continuations. This is
  expected, not a bug — see the full example in `docs/setup-notes.md` §3.

## The int8 expert store (`expertrelay.store.build_store`)

The full Qwen1.5-MoE-A2.7B, all 24 layers and 1,440 routed experts, stored
on the SSD in int8 at a pinned Hugging Face revision. Format and numbers:
`docs/expert-store.md`.

- **int8 changes the model's outputs compared to bf16.** Weight-only,
  symmetric, per-output-channel int8 (Krishnamoorthi 2018; the weight side
  of Dettmers et al. 2022 LLM.int8(), without its outlier decomposition) is
  lossy. Measured per-matrix reconstruction error is in
  `docs/expert-store.md`. A model computed from these weights will not
  reproduce bf16 logits exactly, and greedy decoding can pick different
  tokens.
- **What the correctness rule compares against, from here on.** CLAUDE.md's
  rule (ExpertRelay output must match the reference token-for-token) now
  uses as its reference **this same int8 model, computed with every expert
  resident and no cache or prediction**. Caching, prefetching, prediction
  and device splits must not change a single token relative to that. They
  move bytes around; they don't change arithmetic. bf16 is NOT the
  reference for that rule, because int8 alone would already break it.
- **The quality cost of int8 vs. bf16 is a separate, not-yet-done
  measurement.** It needs the bf16 model evaluated side by side (e.g.
  perplexity on a fixed text set, plus a token-agreement rate). The bf16
  model is 28.6 GB and doesn't fit in this machine's 8 GB RAM, so that
  evaluation waits for a bigger machine or a streamed evaluator.
  Reconstruction error (above) is a proxy, not a quality measurement.
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
  `FILE_FLAG_NO_BUFFERING`. The build itself is portable Python, but its
  final verification pass uses the loader.
- **Not wired into inference yet.** `runtime.moe_model` still runs the
  reduced fp32 MindSpore checkpoint. Loading int8 experts from this store
  into a forward pass is the next phase.

## The device split

- **Real**: two separate OS processes, a real TCP socket, a real
  length-prefixed request/response protocol (`expertrelay.runtime
  .net_proto`). `expertrelay.runtime.expert_server` (Process B) never
  loads or sees anything but its assigned expert weights.
- **Simulated**: both processes have so far only been run on
  `127.0.0.1` on the same machine — not a real second device. Moving
  Process B to a real machine needs zero code changes (`--host 0.0.0.0` on
  the server, `--expert-host <real IP>` on the coordinator) — see
  `docs/setup-notes.md` §6 for the exact steps, not yet executed.
- **Not batched**: each token's each selected expert is dispatched in its
  own request, one at a time, in a Python `for` loop
  (`MoELayer.construct`). Fine for validating correctness; will pay real
  per-call network latency once Process B is on an actual second machine.
  No KV cache either — every generation step reruns the full sequence.

## Correctness testing

`tests/test_split_correctness.py` automates CLAUDE.md's correctness rule
(split output must match a non-split reference token-for-token) — but
against a **tiny synthetic model** (hidden_size=8, 2 layers, 4 experts,
random weights), not the real 24-layer/60-expert Qwen1.5-MoE-A2.7B. The
real-model version of this check (does splitting the ACTUAL converted
checkpoint change the output vs. loading all 4 kept experts in one
process) has been validated **manually**, once, and is documented — not
re-run automatically — in `docs/setup-notes.md` §3, because it needs a
live download and 3GB+ RAM, which an automated test suite that must stay
fast and network-free (see below) can't assume.

## Test suite constraints

Tests must be deterministic and must not require network access — several
tests in `tests/test_store_fetch_hf_tensors.py` and
`tests/test_store_convert_qwen_moe.py` therefore only exercise the *offline*
logic (byte-level bf16 decoding, tensor-name generation), not an actual
Hugging Face fetch. The network path is real code, exercised manually
during actual conversion runs, not covered by `pytest`.

## Memory budget

`expertrelay.memory_budget.enforce_ram_budget` is a real, enforced gate
(raises `MemoryBudgetExceeded`, not just a logged warning) called before
`expertrelay.store.convert_qwen_moe` proceeds past fetching tensors, and
before `expertrelay.runtime.expert_server` / `expertrelay.manager
.coordinator` load a checkpoint — all default to a 6GB ceiling
(`--max-ram-gb`), leaving headroom under this dev machine's 8GB. It gates
on an estimate (fetched-tensor bytes, or checkpoint file size) BEFORE the
memory-heavy step, not on measured peak RSS after the fact — the latter is
`expertrelay.benchmarking.peak_process_rss_mb`, recorded in every benchmark
record but not itself a gate.

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
  for a while. Getting around this means writing past the SLC cache (often
  tens of GB) on every profile run, which isn't worth the time or the SSD
  wear.
- **"Random" means large reads at random offsets**, one expert-sized chunk
  (8.65 or 17.30 MB) at a random 4 KiB-aligned offset. That's the realistic
  pattern for loading one expert. It is not a 4K random-IOPS test, and the
  two numbers shouldn't be compared.
- **Single-threaded, queue depth 1.** Reads are synchronous, one at a time.
  NVMe drives can go faster with several requests in flight, so this is a
  lower bound on what an async/multi-queue loader could get. It is also the
  correct baseline for today's one-expert-at-a-time `MoELayer`.

## Benchmarks

See `benchmarks/README.md` for per-file caveats. The short version: current
numbers (tokens/sec on the local two-process simulation) are a
correctness/plumbing baseline, not a throughput measurement — no KV cache,
no batching, loopback "network." Don't compare them to a future real
two-machine number without re-reading those caveats.
