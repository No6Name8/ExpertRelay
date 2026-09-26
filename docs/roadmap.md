# ExpertRelay roadmap

The official project plan. Created and led by Abdullah Ali Alanazi (see
`AUTHORS`). Status is maintained here; what each finished phase actually
delivered, with its caveats, is in `docs/limitations.md`.

Status legend: **done** · **in progress** · **pending**

---

## Step A: Authorship (before onboarding any teammate)

Goal: public, dated recognition of Abdullah Ali Alanazi as the originator.
No patent is planned.

| # | Action | Status |
|---|---|---|
| A1 | Publish a concept note on Zenodo in Abdullah's name only: public, with a dated DOI | pending |
| A2 | Connect the GitHub repo to Zenodo, so every tagged release gets its own dated DOI | pending |
| A3 | The repo stays on Abdullah's account (github.com/No6Name8), with him as sole owner | done (currently private) |
| A4 | A signed one-page team agreement before anyone sees the code or docs | pending |
| A5 | Register Abdullah as team leader | pending |
| A6 | Check the competition's IP rules and PSU's IP policy | pending |
| A7 | Full paper on arXiv once there are real results | pending |

**Claims must be precise.** Expert caching and expert prediction are
prior work, not ours (e.g. Fate, arXiv:2502.12224; see README "Background
reading"). What's new is the combination: **a mix of devices and SSDs
treated as one virtual memory for MoE models, with a Manager deciding
placement, eviction and prediction across the network.** Every public
claim (concept note, paper, video, competition entry) is worded that way.

---

## Phases

| Phase | What | Status |
|---|---|---|
| 0 | **Machine check.** Typed machine profile: RAM, CPU, drive, uncached SSD read speed at expert size, verified against the OS disk counters | done |
| 1 | **Expert store.** Full Qwen1.5-MoE-A2.7B in int8 on the SSD (14.36 GB), one aligned read per expert, pinned revision, every sha256 verified | done |
| 2 | **First full run.** All 24 layers on the 8 GB laptop at 0.70 tok/s; baselines (normal load never finishes loading; OS paging 0.41 tok/s); numpy/MindSpore backends | done; the int8-vs-bf16 reference check is pending (bf16 download) |
| 3 | **Expert-usage study and cache simulator.** Trace recorder, 96-prompt set, usage analysis, LRU/LFU/pinned/Belady simulator | in progress: code done and tested; base-store trace run underway; Chat-store run and analysis to follow |
| 4 | **Virtual memory cache.** A real RAM cache of experts in the runtime, within the memory budget, with the policy chosen from Phase 3 | pending |
| 5 | **Prediction, with an on/off switch.** Read predicted experts early, overlapping with compute; tokens must be identical with it on and off | pending |
| 6 | **Manager lite.** One component that decides placement and eviction across RAM and SSD from the machine profile (compute-backend selection already lives there) | pending |
| 7 | **Demo kit.** Chat store (building), demo script, benchmark sheets generated from result files | pending |
| 8 | **Friend-PC day.** Run with a second machine over the LAN, measured | pending |

### After Phase 8

| Stage | What | Status |
|---|---|---|
| Two PCs | A permanent two-machine setup, experts placed across both by the Manager | pending |
| Swarm | N devices joining and leaving; experts rebalanced as they do | pending |
| Datacenter simulator | Simulate large clusters from real traces and measured device profiles | pending |
| One-command install | Install and run ExpertRelay with a single command | pending |

---

## Demo video: 10 segments

**Rule: only show what really works. Every number on screen comes from a
file in `benchmarks/results/`, and the file and commit are shown with
it.** If a segment's phase isn't done when filming, the segment is cut or
explicitly labeled "planned". Nothing is mocked up.

| # | Segment | Source of what's shown | Ready? |
|---|---|---|---|
| 1 | The problem: a 14.4 GB MoE model vs. an 8 GB laptop | `machine_profile_<host>.json`, `expert_store_build_<host>.json` | ready |
| 2 | The normal way fails: loading the whole model thrashes without finishing | `phase2_baselines.json` (setup A) | ready |
| 3 | ExpertRelay runs the full model on that laptop, live, with its tok/s and RAM | `phase2_baselines.json` (setup C) + live run | ready |
| 4 | Same answer as the reference: identical tokens across setups; accuracy vs. bf16 | `phase2_baselines.json`, `reference_check.json` | partly (reference check pending) |
| 5 | Inside the model: which experts it uses, and how predictably | `phase3_analysis_<store>.json` charts | after Phase 3 |
| 6 | Virtual memory: the expert cache and its measured speedup | Phase 4 results | after Phase 4 |
| 7 | Prediction switched on and off: speed changes, tokens don't | Phase 5 results | after Phase 5 |
| 8 | The Manager deciding where experts live | Phase 6 results | after Phase 6 |
| 9 | Two machines working as one | Phase 8 results | after Phase 8 |
| 10 | Where it goes next (swarm, datacenter), labeled as plan, and credits | this roadmap, `AUTHORS` | ready (as plan) |
