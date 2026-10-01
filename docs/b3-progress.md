# Step B3 progress (GPTQ int4 on the Chat model)

Updated after every finished item. If anything stops, rerun the command
listed under "Next" — every step resumes where it stopped.

Pass rule (fixed before any result): overall top-1 agreement vs bf16 Chat
≥ 96% and every category ≥ 90%.

## Done

- **0. Disk check** (2026-09-30): 94.25 GB free on C:. Nothing from B3 was
  on disk yet: `models/hf/` empty; the only Chat store is
  `models/qwen1.5-moe-a2.7b-chat-int8`. HF_TOKEN not set (public repos).
  - GPTQ repo `Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4` @
    `81b132adfae58e03b96ae1ed1f0d578d0cc4d09a`: 8.43 GB (3 shards).
  - bf16 `Qwen/Qwen1.5-MoE-A2.7B-Chat` @
    `ec052fda178e241c7c443468d2fa1db6618996be`: 28.64 GB (8 shards).
  - GPTQ config: bits 4, group_size 128, desc_act false, sym true,
    damp_percent 0.01, true_sequential true.

- **Code for steps 1-6** (committed and pushed; 287 tests pass):
  - step 1: `store/download_checkpoint.py` (`--retries`),
    `store/verify_checkpoint.py` (9e6fe76)
  - steps 2-4: `store/gptq.py`, `store/int4.py` (quantizer field, RTN on
    GPTQ's grid), resumable `store/build_int4_store.py`,
    `bench/gptq_conversion_check.py`, tests (ae13a04)
  - step 5: `bench/gptq_quality.py`, resumable bf16 reference (acaa134)
  - step 6: `bench/gptq_benchmark.py`, `phase3_traces --prompts-per-category`,
    `configs/runtime_cache_chat_gptq.json` (7b24ff6)
- **Facts checked on the release before coding** (range reads of 8
  experts): every stored zero nibble is 7, so the zero point is 8 (v1 +1);
  g_idx = i // 128; expert biases all 0; codes 0..15 all used. Sources
  checked: GPTQ reference `quant.py` (Quantizer.find_params, quantize) and
  AutoGPTQ `qlinear_cuda_old.py` (pack: `zeros -= 1`; forward:
  `zeros = zeros + 1`).

## Done (step 1)

- **1a. GPTQ download: done and verified** (2026-09-30): all 9 files match
  the Hub's hashes (`benchmarks/results/checkpoint_verification.json`).
- **1b. bf16 Chat download: done and verified** (2026-10-01; paused once
  at the user's request and resumed without loss): all 15 files match the
  Hub's hashes (`benchmarks/results/checkpoint_verification.json`).
  59 GB free afterwards.

- **2. GPTQ int4 Chat store** (2026-10-01): `models/qwen1.5-moe-a2.7b-chat-int4g128-gptq`,
  1440/1440 records sha256-verified, record 4,460,544 bytes (51.4% of int8).
  Build record: `benchmarks/results/int4_store_build.json`; metadata:
  `benchmarks/stores/qwen1.5-moe-a2.7b-chat-int4g128-gptq/`.
- **4 (check). Conversion vs independent GPTQ formula:** 144 matrices of 48
  sampled experts bit-identical (`benchmarks/results/gptq_conversion_check.json`).

## In progress

- **3. RTN int4 Chat store (GPTQ grid, g128).**

## Next (each command resumes if rerun)

    $env:HF_HUB_DISABLE_XET=1
    python -m expertrelay.store.download_checkpoint --repo-id Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4 --revision 81b132adfae58e03b96ae1ed1f0d578d0cc4d09a --retries 50
    python -m expertrelay.store.verify_checkpoint --repo-id Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4 --revision 81b132adfae58e03b96ae1ed1f0d578d0cc4d09a
    python -m expertrelay.store.download_checkpoint --repo-id Qwen/Qwen1.5-MoE-A2.7B-Chat --revision ec052fda178e241c7c443468d2fa1db6618996be --retries 50
    python -m expertrelay.store.verify_checkpoint --repo-id Qwen/Qwen1.5-MoE-A2.7B-Chat --revision ec052fda178e241c7c443468d2fa1db6618996be

Then:

    python -m expertrelay.store.build_int4_store --int8-store models/qwen1.5-moe-a2.7b-chat-int8 --group-size 128 --quantizer gptq
    python -m expertrelay.bench.gptq_conversion_check
    python -m expertrelay.store.build_int4_store --int8-store models/qwen1.5-moe-a2.7b-chat-int8 --group-size 128 --quantizer rtn_gptq_grid
    python -m expertrelay.bench.gptq_quality
    python -m expertrelay.bench.phase3_traces --store-dir models/qwen1.5-moe-a2.7b-chat-int4g128-gptq --prompts-per-category 4
    python -m expertrelay.bench.fit_prefetch_calibration --store-dir models/qwen1.5-moe-a2.7b-chat-int4g128-gptq
    # clean machine, plain PowerShell:
    python -m expertrelay.bench.gptq_benchmark
