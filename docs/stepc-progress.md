# Step C progress (the big fair test)

Updated after every finished item. If anything stops, rerun the command
listed under "Next": every step resumes where it stopped.

## Done

- **0. Disk check** (2026-10-04): 51 GB free on C:. The GGUF files need
  ~24 GB at the end (Q4_K_M ~8.8 GB + Q8_0 ~15.3 GB); converting needs a
  bf16 GGUF (~28.6 GB) first, so the peak is ~37 GB: it fits, nothing has
  to be deleted.
- **Baseline choices (step 1, before any download):**
  - **No official Qwen GGUF exists** for Qwen1.5-MoE-A2.7B-Chat (the Hub
    has no `Qwen/Qwen1.5-MoE-A2.7B-Chat-GGUF`; only third-party uploads).
    So the GGUFs are converted from the verified bf16 Chat weights
    (`Qwen/Qwen1.5-MoE-A2.7B-Chat` @ ec052fda) with llama.cpp's own
    converter and quantizer.
  - **llama.cpp pinned to build b11146** (commit 7fe450e1), the build the
    latest stable release v0.5.0 (2026-09-23) points to; Windows CPU x64
    zip `llama-b11146-bin-win-cpu-x64.zip`, sha256 14cf1303....
  - Threads: 12 for llama.cpp, the same as ExpertRelay's numba kernels (12
    logical cores).

- **1. llama.cpp setup and GGUFs** (2026-10-04):
  `benchmarks/results/llamacpp_setup.json` (exact commands, sizes, sha256).
  Binary zip sha256 matches GitHub's digest; `llama-cli --version`:
  0.5.0-dev (build 11146, commit 7fe450e19). Files in `models/gguf/`:
  - bf16 GGUF 28.64 GB (converter, 561 s), used for Q4_K_M, then deleted;
  - **Q4_K_M 9.50 GB** (llama-quantize, 609 s). The experts' down
    projections have rows of 1408, not divisible by K-quants' 256, so
    llama.cpp stores those tensors as q5_0 / q8_0 instead (its own
    fallback, logged by llama-quantize);
  - **Q8_0 15.23 GB** (converter `--outtype q8_0`, 757 s).
  - Free disk after: 26 GB.
- **Checks before the runs:** llama-server b11146 takes our prompt token
  ids as given (no BOS added), streams one token per event, and returns
  log-probabilities for all 151,936 tokens with n_probs = vocab (they sum
  to 1.000), so exact top-1 and KL vs bf16 are measurable with the pinned
  build itself.
- **Code for steps 3-5** (committed): `bench/fair_test.py` (A-F, 3
  interleaved rounds; `--sweep`), `bench/resumable.py`,
  `bench/llamacpp_quality.py`, configs `configs/stepc.json`,
  `configs/stepc_sweep.json`; tests in `tests/test_fair_test.py`.

## In progress

- **4. llama.cpp quality** (`python -m expertrelay.bench.llamacpp_quality`,
  running in the background; not a speed measurement).
