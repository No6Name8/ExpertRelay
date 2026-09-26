# ExpertRelay

Created by Abdullah Ali Alanazi.

Concept note: [doi:10.5281/zenodo.22980838](https://doi.org/10.5281/zenodo.22980838)
(Zenodo, https://zenodo.org/records/22980838).

Virtual memory for AI: a Manager that understands every device in a
cluster, keeps hot experts in RAM and cold experts on SSD or a networked
peer, predicts which expert a Mixture-of-Experts (MoE) model will need
next, and starts moving it there before the router asks for it. Built on
MindSpore.

## The problem

Large MoE models only activate a small subset of "expert" subnetworks per
token — which is why they can, in principle, be split across multiple weak
devices (phones, mini PCs, dev boards) instead of needing one expensive
machine. But naive multi-device setups (layer-splitting RPC, on-demand
streaming) load an expert only *after* the router picks it — every layer
stalls waiting on a network or storage link before it can compute. And
today that's usually a hardcoded, static split: expert 0-29 on device A,
30-59 on device B, forever.

## The idea

Treat expert weights the way an OS treats memory pages. A **Manager**
tracks every device in the cluster (its RAM, its SSD, its link speed to
every other device) and where every expert currently lives. Hot experts —
the ones being routed to often — stay resident in RAM; cold ones page out
to SSD or a networked peer. A **predictor** watches routing patterns and
tries to guess which expert a future token/layer will need *before* the
router officially decides, so the Manager can start moving that expert
across the link early — overlapping transfer with compute instead of doing
them sequentially. A **swarm** of these devices, not a fixed pair, is the
end target: any device can hold any expert, and the Manager routes each
request to wherever that expert currently lives.

## Why it matters

Turns "buy one expensive AI accelerator" into "network a few cheap
machines you already own" — with the scheduling smart enough to make that
actually fast. Benefits cost-constrained institutions (schools, clinics,
small labs), privacy-sensitive users who need fully local inference, and
the existing mini-PC clustering community (Strix Halo, Atlas boards)
that's already doing this manually with cruder tooling.

## Status

Early. What's real right now: the full Qwen1.5-MoE-A2.7B (24 layers,
1,440 routed experts) stored on the SSD in int8, and a runtime that
generates from it on an 8 GB laptop. Resident weights stay in RAM, and each
routed expert is read from disk with one unbuffered read when the router
picks it, then dropped. There's no cache yet, no prediction (every read is
reactive), and only one device: the earlier two-process network split was
removed with the reduced model it ran on. **See `docs/limitations.md`
before trusting any claim about what currently works.** It's the
maintained source of truth for real vs. simplified vs. not-yet-built, and
this README will drift out of date faster than that file will.

## Structure

```
src/expertrelay/
  store/      fetch + convert expert weights (HF Hub -> local MindSpore checkpoint)
  cache/      hot/cold placement + eviction across RAM/SSD/network  [placeholder]
  predictor/  predict which expert is needed next, ahead of the router  [placeholder]
  manager/    understands every device, dispatches each request to the right one
  runtime/    the MoE forward pass, wire protocol, expert-serving process
  bench/      benchmark and demo scripts -> benchmarks/results/*.json
tests/        pytest, deterministic, no live network calls
docs/         limitations.md (current state), setup-notes.md (history)
benchmarks/   results/*.json (raw) + README.md (how to read them)
```

## Getting started

```bash
pip install -e ".[dev]"
pytest                      # fast, deterministic, no network/GPU needed
ruff check . && ruff format --check .
```

Running the real (network-dependent, RAM-heavy) pipeline — converting a
slice of real Qwen1.5-MoE-A2.7B weights and running the two-process split
against them — is documented in `docs/setup-notes.md`, not repeated here
since it needs a live download and several GB of RAM.

## Background reading

Builds on ideas from recent MoE offloading/caching research (Fate, APEX,
HOBBIT, SlimCaching) and community work on multi-device LLM clustering
(llama.cpp RPC, Strix Halo clusters), applied specifically to cross-device
scheduling rather than single-machine caching. Any method adapted from a
specific paper is cited at its point of use in code, per `CLAUDE.md`.

## Licence

- **Code:** all rights reserved, Copyright 2026 Abdullah Ali Alanazi. An
  open-source licence will be chosen at publication.
- **Documentation and the concept note:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Prompts written for this project** (`benchmarks/prompts/`): CC BY 4.0.
  Each prompt file records the source and licence of its prompts.

Created by Abdullah Ali Alanazi; see [AUTHORS](AUTHORS).

## Third-party

- **Model weights are not included in this repository.** They're downloaded
  from Hugging Face, at pinned commits, under the Tongyi Qianwen licence:
  [Qwen1.5-MoE-A2.7B](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B/blob/main/LICENSE)
  and [Qwen1.5-MoE-A2.7B-Chat](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat/blob/main/LICENSE).
  Converted int8 stores made from them fall under the same licence.
- **MindSpore, numpy and PyTorch** (and the other packages listed in
  `pyproject.toml`) are used under their own licences. None of their code
  is included here.
