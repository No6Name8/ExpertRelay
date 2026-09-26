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
| A6 | Check the IP rules of the Huawei ICT Competition, Innovation Track (Middle East & Central Asia) | pending |
| A7 | Full paper on arXiv once there are real results | pending |

**Claims must be precise.** Expert caching and expert prediction are
prior work, not ours (e.g. Fate, arXiv:2502.12224; see README "Background
reading"). What's new is the combination: **a mix of devices and SSDs
treated as one virtual memory for MoE models, with a Manager deciding
placement, eviction and prediction across the network.** Every public
claim (concept note, paper, video, Huawei ICT Competition, Innovation Track (Middle East & Central Asia) entry) is worded that way.

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
it.** Parts that aren't finished when filming appear only as animations
clearly labeled "in development", with no measured numbers on them.

| # | Segment | Benchmark file(s) behind what's shown | Ready? |
|---|---|---|---|
| 1 | **The wall.** The 8 GB PC tries to load the model the normal way and fails | `phase2_baselines.json` (setup A: committed 14.6 GB, 4.7 GB resident, never finished loading) | ready |
| 2 | **Why it matters.** Models are growing faster than hardware | none yet: needs a data file of cited public model sizes and hardware memory figures | pending |
| 3 | **The 8 GB PC alone.** ~14 GB of model files next to 8 GB of RAM; network cable unplugged, airplane mode on; Task Manager on screen the whole time (RAM under budget, disk busy, network at 0); ExpertRelay runs the model; the answer matches the 32 GB PC's word for word | `expert_store_build_<host>.json`, `phase2_baselines.json` (setup C), a new offline-run record, a 32 GB PC comparison record | partly: runs fully offline today; the 32 GB PC comparison needs Phase 8; the demo uses the Chat store (building) |
| 4 | **Memory slider.** RAM budget dragged from 32 GB down to 4 GB during generation; it slows smoothly and never crashes. Live brain map of experts: Arabic, English and code light up different experts | Phase 4 cache results (budget sweep); `phase3_analysis_<store>.json` (expert usage per category) | pending: needs the Phase 4 cache with a live-adjustable budget; brain-map data comes from Phase 3 |
| 5 | **Prediction on/off switch**, live, with a prediction-accuracy counter | Phase 5 results; `phase3_analysis_<store>.json` (Fate-style predictor accuracy) | pending (Phase 5) |
| 6 | **The Manager scans both PCs** and assigns roles automatically | Phase 6/8 results; `machine_profile_<host>.json` per PC | pending (Phases 6 and 8) |
| 7 | **Two PCs.** The 32 GB PC fails alone on a bigger model; together they run it; per-PC "experts computed" counters show both working; pull the network cable and it recovers | two-PC results | pending (two-PC stage) |
| 8 | **Moving the wall.** The 32 GB PC runs a ~235B MoE model from its 2 TB SSD: normal load fails, naive streaming crawls, ExpertRelay is faster | results on a ~235B MoE store | pending (needs a store for the larger model, and the 32 GB PC) |
| 9 | **Datacenter simulator** results (100+ mixed chips) | datacenter-simulator results | pending |
| 10 | **Scoreboard and impact** | the result files above; only what's finished at filming time | pending (grows as the phases finish) |
