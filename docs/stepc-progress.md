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

## In progress

- **1. llama.cpp setup and GGUF conversion.**
