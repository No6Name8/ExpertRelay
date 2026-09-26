# ExpertRelay

"Virtual memory for AI": MoE expert caching, prediction, and prefetching
across RAM, SSD, and networked devices. MindSpore backend for Ascend,
numpy backend on CPU, selected by the Manager based on the machine
(`runtime/backends.py`, `manager/backend_selection.py`; the measured reason
is in `docs/limitations.md`). Primary dev machine: Windows, 8GB RAM, no
GPU/Ascend — every design choice here has to work under that constraint
first, then scale up.

Read `docs/limitations.md` before trusting any claim about what this repo
can currently do — it's the single source of truth for real vs. simplified
vs. not-yet-implemented.

## Rules for all work in this repo

**No fake results.** Any simplification — fewer layers, a mocked component,
a reduced expert count, a synthetic weight instead of a real one — must be
labeled in a code comment AND recorded in `docs/limitations.md`. If you're
not sure whether something counts as a simplification, write it down; that
costs nothing, and a reviewer finding an undisclosed one costs everything.

**Benchmarks are scripts, not spreadsheets.** Every benchmark is a script
in the repo that writes raw JSON to `benchmarks/results/`, including at
minimum: git commit hash, machine profile (platform, CPU count, RAM,
MindSpore version), model, config, seed, and timestamp. Use
`expertrelay.benchmarking` for this — don't hand-roll the metadata dict
again. Charts are generated only from those JSON files, never hand-edited
or eyeballed into existence.

**Correctness rule.** ExpertRelay's output must match the reference (the
full model held in RAM, no splitting/caching/prediction) token-for-token
under greedy decoding. Since the int8 store, the reference is the **int8**
model computed that way, not bf16: int8 changes outputs on its own, and its
quality cost vs. bf16 is measured separately (see `docs/limitations.md`). Enforce this with tests wherever possible. When the
real reference is too large/slow/network-dependent for an automated test
(e.g. the actual 24-layer/60-expert Qwen1.5-MoE-A2.7B), the test's
reference is a smaller synthetic model exercising the same code path — and
the manual, real-model validation this narrower test stands in for must be
documented (see `tests/test_runtime_correctness.py` for the pattern, and
`docs/limitations.md` for where the real-model check currently lives).

**Tests.** pytest for all logic in `store/`, `cache/`, `predictor/`,
`manager/`, `runtime/`. Tests must be deterministic — no un-seeded
randomness, no live network calls (offline logic gets tested; the
network-dependent path gets documented instead, see
`tests/test_store_fetch_hf_tensors.py`'s docstring for the pattern).

**Code style.**
- Type hints and docstrings everywhere reasonable. Docstrings explain WHY
  (a non-obvious constraint, a deliberate simplification, a workaround),
  not what the code already says via good naming.
- `ruff` for lint and format — run both before committing.
- `pathlib.Path` for paths, always anchored to `expertrelay.REPO_ROOT`
  (see `expertrelay/paths.py`). No relative-to-CWD or relative-to-`__file__`
  path hacks.
- No dead code, no commented-out code. Delete it or don't write it.

**Citations.** Any method taken from a paper (e.g. Fate's cross-layer gate
prediction) must cite the paper in a comment at the point it's implemented
— not just in a README reference list.

**Nothing but the runtime knows the compute backend.** The store, expert
sources, cache, predictor and Manager deal only in plain numpy arrays and
backend *names*. They never import `runtime.backends`, a backend, or
MindSpore. `tests/test_runtime_backends.py` enforces this. Any new compute
operation goes into the backend interface and gets an implementation and a
test for every backend.

**Machine facts come from one place.** RAM, CPU, drive, and disk-speed
information comes only from `expertrelay.manager.profile`. Don't call
psutil/platform/PowerShell for machine facts anywhere else; extend the
profile instead.

**Memory budget.** Enforce a memory budget in code wherever a component
loads weights or caches data, and measure peak RAM in every benchmark
(`expertrelay.benchmarking.peak_process_rss_mb`).

**Git.** Commit after each completed step with a clear message, and push.
Don't batch unrelated changes into one commit.

## Package layout

```
src/expertrelay/
  store/      persistence: fetching + converting expert weights (HF Hub -> local checkpoint)
  cache/      expert placement + eviction: the runtime's RAM cache, and offline simulators (see limitations.md)
  predictor/  predicting which expert will be needed next: the prefetch policy, offline tools (see limitations.md)
  manager/    machine profile, compute-backend selection; later: device placement and dispatch
  runtime/    the MoE forward pass and the compute backends (numpy, MindSpore)
  bench/      benchmark and demo scripts -- each writes to benchmarks/results/
```

## Not covered by this file

Anything about the current state of the code — what's implemented, what's
a stub, what's been validated and how — belongs in `docs/limitations.md`
and `docs/setup-notes.md`, not here. This file is standing instructions;
those are living status.
