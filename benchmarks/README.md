# Benchmarks

Every benchmark script writes its raw results as a JSON list to
`benchmarks/results/<name>.json` (never edited by hand), with each record
carrying at minimum: `git_commit`, `git_dirty`, `machine` (platform, CPU,
RAM total and free, drive, MindSpore version), `seed`, `model`, `config`,
and `timestamp` -- see `CLAUDE.md`. Readable tables in `docs/` are generated
from these files by the same scripts, never hand-edited.

| File | Script | What |
|---|---|---|
| `machine_profile_<host>.json` | `expertrelay.manager.profile` | RAM, CPU, drive, uncached SSD read speed at expert-sized chunks |
| `expert_store_build_<host>.json` | `expertrelay.store.build_store` | int8 store sizes, quantization error, peak RAM of the build |
| `int8_kernels.json` | `expertrelay.bench.int8_kernels` | speed and error of each int8 matmul option on real shapes |
| `phase2_baselines.json` | `expertrelay.bench.phase2_baselines` | full 24-layer model: normal load vs. OS paging vs. ours |
| `reference_check.json` | `expertrelay.bench.reference_check` | our int8 model vs. HF transformers on the original bf16 weights |
| `local_split_simulation.json` | *(removed)* | historical, see below |

Prompts are fixed and live in `prompts/`.

## local_split_simulation.json (historical)

Produced by `run_local_split_demo`, which ran a 2-layer / 4-expert reduced
fp32 slice of the model as two processes talking over 127.0.0.1. That code
was removed in Phase 2 once the full 24-layer int8 runtime replaced it; it
lives in git history (last present at commit `93e5a7a`). The records are
kept as history, not as a current result. The reduced model and the current
one are too different to compare.
