# ExpertRelay

Predictive expert prefetching for distributed Mixture-of-Experts (MoE) inference across weak, low-cost hardware clusters.

## The problem

Large MoE models only activate a small subset of "expert" subnetworks per token, which is why they can be split across multiple weak devices (phones, mini PCs, dev boards) instead of needing one expensive machine. But current multi-device inference setups (naive layer-splitting RPC, on-demand streaming) load an expert only after the router picks it — every layer stalls waiting on the network or storage link before it can compute.

## The idea

Predict which expert the next token is likely to need before the router officially decides, and start transferring that expert across the link early — overlapping data transfer with compute instead of doing them sequentially. The prediction layer sits above the transport, so it works the same way whether devices are linked over USB4/Thunderbolt, Ethernet, or HarmonyOS's Distributed Soft Bus.

## Why it matters

Turns "buy one expensive AI accelerator" into "network a few cheap machines you already own" — with the scheduling smart enough to make that actually fast. Benefits cost-constrained institutions (schools, clinics, small labs), privacy-sensitive users who need fully local inference, and the existing mini-PC clustering community (Strix Halo, Atlas boards) that's already doing this manually with cruder tooling.

## Status

Early development — targeting a working benchmark comparing predictive prefetching vs. naive on-demand loading across a real multi-device split, on both Huawei hardware (Atlas 200I DK A2 / Ascend NPU) and a generic cluster (e.g. consumer mini PCs).

## Structure

- `src/` — implementation (predictor, scheduler, transport hooks)
- `benchmarks/` — baseline vs. prefetch comparison results
- `docs/` — design notes, competition write-up materials

## Background reading

Builds on ideas from recent MoE offloading/caching research (Fate, APEX, HOBBIT, SlimCaching) and community work on multi-device LLM clustering (llama.cpp RPC, Strix Halo clusters), applied specifically to cross-device scheduling rather than single-machine caching.
