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

## In progress

- **1. Downloads + hash verification.** Code: `store/download_checkpoint.py`
  (`--retries`), `store/verify_checkpoint.py`. Outputs:
  `models/hf/<repo>@<commit>/`, verification in
  `benchmarks/results/checkpoint_verification.json`.

## Next

    $env:HF_HUB_DISABLE_XET=1
    python -m expertrelay.store.download_checkpoint --repo-id Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4 --revision 81b132adfae58e03b96ae1ed1f0d578d0cc4d09a --retries 50
    python -m expertrelay.store.verify_checkpoint --repo-id Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4 --revision 81b132adfae58e03b96ae1ed1f0d578d0cc4d09a
    python -m expertrelay.store.download_checkpoint --repo-id Qwen/Qwen1.5-MoE-A2.7B-Chat --revision ec052fda178e241c7c443468d2fa1db6618996be --retries 50
    python -m expertrelay.store.verify_checkpoint --repo-id Qwen/Qwen1.5-MoE-A2.7B-Chat --revision ec052fda178e241c7c443468d2fa1db6618996be
