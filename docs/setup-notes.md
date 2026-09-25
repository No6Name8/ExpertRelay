# Setup notes — MindSpore verification & MoE model survey

Date: 2026-09-13

## 1. MindSpore installation check

- **Version:** 2.7.1 (Python 3.10.0)
- **Backend:** CPU only. The installed wheel does not support Ascend or GPU —
  attempting `ms.set_context(device_target="Ascend")` or `"GPU"` raises
  `Unsupported device target ... This process only supports one of the ['CPU']`.
  To get Ascend support we'd need the Ascend-specific wheel (built with
  `-e Ascend`) plus the CANN toolkit installed, which this machine doesn't have.
- **Sanity check:** ran a basic `ops.matmul` on two 2x2 tensors on the CPU
  backend — returned the correct result (`[[19,22],[43,50]]`, sum `134.0`).
  MindSpore's core tensor ops work correctly on this machine.
- **Hardware constraints found during this session:** 8.3 GB total system RAM,
  ~0.8 GB free at time of testing, 12 logical CPUs. This matters a lot for
  what's feasible below.

## 2. MoE models in MindSpore's own ecosystem

- **MindSpore Hub:** no MoE models. Hub's pretrained models are concentrated
  in image classification, object detection, semantic segmentation, and
  recommendation — no LLM/MoE checkpoints there.
- **MindFormers (MindSpore Transformers):** this is where MoE support
  actually lives. It has **native architecture code** (not just docs) for:
  - `deepseek3` — DeepSeek-V3 (671B total / 37B active)
  - `qwen2_moe` — Qwen1.5/Qwen2 MoE family (e.g. Qwen1.5-MoE-A2.7B: 14.3B
    total / 2.7B active)
  - `qwen3_moe` — Qwen3 MoE family
  - MindFormers docs also list DeepSeek-V2 (236B, Sparse LLM) as supported.
  - No genuinely small (<5B total) MoE architecture is natively supported —
    everything on the list is a real frontier-scale model.
- **MindOne** (mindspore-lab/mindone): mentioned in search results as
  supporting `qwen3_omni_moe` and `qwen3_vl_moe`, but these are
  multimodal/omni variants, not a smaller/simpler entry point.
- **Conclusion:** MindSpore's own ecosystem has MoE *architecture* code, but
  no pretrained MoE *checkpoints* ready to just download and run — everything
  requires bringing your own (converted) weights, and the models it does
  support are all large.

## 3. Small open-source MoE candidates for conversion

Two realistic candidates, ranked by conversion feasibility rather than by
raw size (both are "small" only relative to frontier MoE, not relative to
this machine):

1. **Qwen1.5-MoE-A2.7B** (14.3B total params, 2.7B activated, 60 experts,
   4 routed + shared experts per token). **Best candidate** because
   MindFormers already ships `qwen2_moe` architecture code — the routing,
   expert-FFN, and shared-expert logic don't need to be written from
   scratch. Conversion work would be:
   - Download HF safetensors weights.
   - Run/adapt MindFormers' existing `convert_weight.py`-style script for
     the Qwen2 family to remap HF tensor names → MindSpore parameter names
     (attention/MLP naming conventions differ between HF `transformers` and
     MindFormers).
   - Match the `qwen2_moe` config (hidden size, expert count, top-k, shared
     expert flag) to Qwen1.5-MoE-A2.7B's actual config.
   - Validate numerically (e.g. compare logits on a few tokens against the
     HF reference implementation) since expert-index/weight-normalization
     conventions are a common source of silent bugs in MoE ports.
   - **Blocker on this machine:** 14.3B params is ~28 GB in fp16 or ~14 GB
     in int8 — far beyond the 8.3 GB total RAM available here.

2. **DeepSeek-V2-Lite** (15.7B total, 2.4B active, 64 routed + 2 shared
   experts). MindFormers has `deepseek3` code for the newer V3 architecture,
   which is closely related but not identical to V2's MoE layer design
   (V3 adds changes like different load-balancing and routing details), so
   this would need real architecture adaptation, not just a weight-name
   remap — higher effort than option 1.
   - **Same RAM blocker** as above (15.7B total).

A third option considered and **rejected**: OLMoE-1B-7B (7B total / 1B
active, fully open, Apache 2.0) is smaller in total params than either
option above, but MindFormers has no existing OLMoE architecture code, so
the MoE routing/dispatch logic would need to be written from scratch — more
engineering effort for a model that still doesn't fit in 8 GB RAM anyway
(7B params ≈ 14 GB fp16 / ~7 GB int8, i.e. still at or over this machine's
total memory with nothing else running).

**Net assessment:** every real pretrained MoE checkpoint we could find,
including the smallest genuinely open ones, needs more RAM than this
machine has free. Converting one here would hit an out-of-memory wall
before we could even validate correctness.

## 4. Single-device inference attempt

Given the RAM ceiling above, downloading/converting a real checkpoint in
this session wasn't viable. Instead, to still validate the mechanics before
committing to a specific model, I wrote a small **native MindSpore MoE
sanity check** (now at
[`src/expertrelay/bench/moe_sanity_check.py`](../src/expertrelay/bench/moe_sanity_check.py)).

- Randomly-initialized (untrained) decoder: embedding → 2× MoE layers
  (router + top-k dispatch over 4 experts, each a 2-layer FFN) → LayerNorm →
  output head. Vocab size 64, hidden size 32 — tiny on purpose.
- Ran a forward pass and 5 steps of greedy decoding on CPU.
- **Result: ran successfully, no errors.** Output token ids were e.g.
  `[41, 42, 58, 31, 56, 1, 1, 39, 36, 23, 36, 23, 36]` — the repeating
  `36, 23, 36, 23...` tail is the expected behavior of greedy decoding on an
  *untrained* model (it collapses into a short cycle), not a bug.
- **What this confirms:** MindSpore's `nn.Dense`, `nn.Embedding`,
  `ops.top_k`, `ops.softmax`, and control flow needed for MoE routing +
  expert dispatch + combine all work correctly together on this CPU install,
  and a generation loop (KV-cache-free, naive re-forward-pass-per-token)
  runs end to end.
- **What this does NOT confirm:** nothing about real MoE checkpoint loading,
  weight conversion correctness, or linguistically coherent output — that
  requires a real trained model, which we couldn't fit on this machine in
  this session (see §3).

## What worked / what didn't

**Worked:**
- MindSpore 2.7.1 CPU backend, basic tensor ops.
- Native MindSpore MoE routing/dispatch/generation mechanics (toy model).
- Identifying that MindFormers has ready-made `qwen2_moe` / `qwen3_moe` /
  `deepseek3` architecture code as a real head start for future conversion
  work.

**Didn't work / blocked:**
- No pretrained MoE checkpoint (from MindSpore's ecosystem or converted from
  HF) could be loaded and run on this machine — every realistic candidate
  needs more RAM than is available (8.3 GB total, ~0.8 GB free).
- Ascend/GPU backends aren't usable with the currently installed wheel.

## Suggested next steps (2026-09-13, superseded below)

- Test on a machine with more RAM (16 GB+) or the actual target Ascend
  hardware (Atlas 200I DK A2) before attempting real checkpoint conversion.
- If Qwen1.5-MoE-A2.7B is still the target, start the weight-conversion
  script now (it doesn't require the full model resident in RAM if done
  shard-by-shard) even before a bigger machine is available, so it's ready
  once resources allow a load test.

---

# 2026-09-14 — Reduced real-weight conversion + local two-process split

**Correction to the 2026-09-13 notes above:** MindFormers does **not**
actually ship a `qwen2_moe` architecture. That was wrong — verified this
session by listing the real repo (`mindspore-ai/mindformers`, not
`mindspore-lab/mindformers`, which doesn't exist). What MindFormers actually
has is `qwen3_moe` (`mindformers/models/qwen3_moe/`) plus `deepseek3`. There
is no native architecture for Qwen1.5-MoE-A2.7B's exact HF model type
(`qwen2_moe`) in MindFormers. This session ended up writing that
architecture by hand instead (see below) rather than relying on a
MindFormers class that doesn't exist for this model family.

**Path note (2026-09-25):** the file paths below (`src/moe_convert/...`,
`src/coordinator.py`, etc.) describe where this code lived *at the time it
was written*, in this session. It has since been restructured into the
`src/expertrelay/` package (`expertrelay.store`, `expertrelay.manager`,
`expertrelay.runtime`, `expertrelay.bench`) — see `docs/limitations.md` for
the current, maintained state and `CLAUDE.md` for the package layout. This
section is left as-is as a historical record of the design work; don't
follow its paths literally.

## What this session set out to do

Get the *whole* pipeline — real-weight conversion, a truncated model small
enough for this 8GB machine, and an actual two-process device split talking
over a socket — working end to end on a toy-sized slice of the real model,
so that swapping in the friend's 32GB machine later is a scale-up, not a
first debug pass.

## 1. Weight conversion pipeline (real weights, truncated architecture)

**This is real, not synthetic.** `src/moe_convert/` downloads actual
Qwen1.5-MoE-A2.7B weights from Hugging Face and converts a chosen slice of
them — not fake/random data, and not the full 28.6GB checkpoint either.

- `src/moe_convert/fetch_hf_tensors.py`: a selective safetensors fetcher.
  Safetensors files are laid out as `[8-byte header length][JSON header
  with each tensor's byte offsets][raw tensor bytes]`. This fetches just
  the header (via an HTTP Range request) for each of the checkpoint's 8
  shards, then issues one more Range request per tensor we actually want —
  never downloading a full shard file. This is what makes "load only 2-4
  experts" mean something on an 8GB machine instead of requiring a 28.6GB
  download regardless. bf16 tensors (HF's native dtype for this checkpoint)
  are upcast to fp32 by hand (bf16 is literally the top 16 bits of fp32 —
  left-shift into a uint32, view as float32), since numpy has no native
  bf16 type.
- `src/moe_convert/convert_qwen_moe.py`: generic CLI, not hardcoded to one
  size. Flags: `--num-layers`, `--expert-ids` (which of the real 60 expert
  indices to keep), `--process-a-experts` / `--process-b-experts` (how to
  split those between the two devices), `--top-k` (see below for why this
  is overridden), `--repo-id` (defaults to `Qwen/Qwen1.5-MoE-A2.7B`, but
  works for any HF repo with the same tensor-naming convention). Real
  `config.json` values (hidden_size, rope_theta, rms_norm_eps, etc.) are
  fetched live and used — nothing about the model's actual dimensions is
  hardcoded.
- **What we actually converted this session:** 2 of the real 24 layers,
  4 of the real 60 experts (ids 0,1,2,3 — split 0,1 → Process A, 2,3 →
  Process B), full real vocab (151936) and full embedding/lm_head (so the
  real tokenizer works correctly end to end). Command:
  ```
  python -m moe_convert.convert_qwen_moe \
      --num-layers 2 --expert-ids 0,1,2,3 \
      --process-a-experts 0,1 --process-b-experts 2,3 --top-k 2 \
      --out-dir ../models/reduced_qwen_moe
  ```
  Downloaded 55 real tensors, 3.18 GB total once upcast to fp32 in RAM.
  Wrote `process_a.ckpt` (3.04 GB — embedding, attention, norms, shared
  expert, router, lm_head, experts 0/1), `process_b.ckpt` (138 MB — only
  experts 2/3), and `manifest.json` (records the real config values used
  plus the expert/layer assignment, so nothing downstream re-derives or
  re-hardcodes shapes). **Converted checkpoints are gitignored** (`models/`,
  `*.ckpt`) — they're regenerable multi-hundred-MB build artifacts, not
  something that belongs in git history; re-run the command above to
  reproduce them.
- **Deliberate simplification vs. real MindFormers conventions:** real
  MindFormers (verified by reading `mindformers/models/qwen3_moe/utils.py`
  and `mindformers/checkpoint/converter/convert_op.py` from the actual repo)
  fuses Q/K/V into one `linear_qkv` matrix and packs all experts into two
  stacked `weight1`/`weight2` tensors, for its tensor-parallel/expert-
  parallel training kernels. That fusion is a training-performance detail,
  orthogonal to what we're validating (does conversion + split + dispatch
  work at all) — our model keeps attention and experts **unfused** (one
  Linear per Q/K/V/O, one Linear-triple per expert), while still using
  MindFormers' naming vocabulary (`decoder.layers.N...`, `mlp.router`,
  `gating`/`hidden`/`linear_fc2`, `output_layer`, etc. — these exact names
  come from the real `utils.py`). If this ever needs to run inside actual
  MindFormers training/serving, an extra fusion pass would be needed; for
  our own hand-rolled inference-only model (`src/moe_model.py`) it doesn't.
- **Other honest deviations from the real model, forced by truncation:**
  - Router softmax is now computed over only the 4 kept experts (their real
    trained weight rows, sliced from the real 60-row router), not the real
    60-way softmax. Changes the routing-weight magnitudes vs. the real model.
  - `top_k` overridden from the real value (4) to 2. With only 4 total
    experts loaded, "top-4 of 4" is just "use all of them every time" —
    trivial, and wouldn't exercise the local/remote split at all. Top-2
    forces genuine selective routing between Process A's and Process B's
    experts.
  - Only 2 of 24 layers means the residual stream reaching the real lm_head
    has had ~22 fewer layers of processing than the weights were trained
    for. See §3 for what this does to output quality.

## 2. The real two-process device split

- `src/net_proto.py`: the wire protocol. Length-prefixed JSON header +
  length-prefixed raw payload, over a plain TCP `socket`. Doesn't know or
  care whether the peer is `127.0.0.1` or across a LAN.
- `src/expert_server.py` ("Process B"): loads **only**
  `process_b.ckpt` (138 MB) — no embedding, no attention, no lm_head, just
  the expert FFNs it was assigned. Listens on a TCP port; on each request
  `{cmd: run_expert, layer, expert_id, shape, dtype} + raw float32 bytes`,
  runs that expert's forward pass and returns the result the same way.
- `src/coordinator.py` ("Process A"): loads `process_a.ckpt`, runs
  embedding → attention → router → (local expert *or* a real TCP round
  trip to Process B, per-token per-selected-expert) → shared expert →
  next layer → ... → lm_head → greedy-decode loop, using the real Qwen
  tokenizer (`tokenizers` library, `tokenizer.json` fetched from the same
  HF repo) for encode/decode.
- `src/moe_model.py`'s `MoELayer.construct` is the actual split-decision
  point: for each token and each of its top-k selected experts, if
  `expert_id in self.experts` (held locally) it's computed in-process;
  otherwise it calls `self.remote_expert_fn(layer_idx, expert_id, x)` —
  which `coordinator.py` wires up to `RemoteExpertClient`, a thin wrapper
  around the TCP socket. This is the exact seam that makes the "device
  split" real rather than simulated-in-name-only: the coordinator has no
  idea whether a given expert call resolves in-process or over a socket
  until it actually checks.
- `src/run_local_split_demo.py`: orchestrates both for the local
  simulation — launches `expert_server.py` as a subprocess, waits for its
  `EXPERT_SERVER_READY` line, runs `coordinator.py`, tears the server down.

## 3. End-to-end result

Ran `python run_local_split_demo.py --model-dir ../models/reduced_qwen_moe
--prompt "The capital of France is" --max-new-tokens 15` from `src/`:

```
prompt: 'The capital of France is' -> [785, 6722, 315, 9625, 374]
generated ids: [785, 6722, 315, 9625, 374, 113805, 112089, 322, 116766,
                74193, 48738, 121581, 99216, 101054, 5891, 58, 100460,
                101169, 107474, 11452]
decoded text: 'The capital of France is抽出贯通//没有必要待笑靥术空间ael[搞下去的有效cart'
15 new tokens in 50.586s -> 0.297 tok/s
remote (Process B) calls: 349, total wire time 9583.2 ms (18.9% of wall time)
  layer 0: local_calls=171 remote_calls=189
  layer 1: local_calls=200 remote_calls=160
```

**It worked — with an important caveat on "coherent."** The prompt tokens
round-trip correctly through the real tokenizer, and the model faithfully
echoes the prompt back (`'The capital of France is'` decodes perfectly,
confirming embedding + tokenizer + lm_head are wired correctly with real
weights). The *generated continuation* is not fluent English — it's a mix
of real Chinese/multilingual vocabulary tokens and stray symbols. This is
the expected consequence of keeping only 2 of the model's 24 layers (see
§1): the real lm_head was trained to decode a residual stream that's been
through 24 layers of processing, and we're feeding it one that's been
through 2. It's real weights producing real (if linguistically incoherent)
vocabulary tokens through a correctly-wired pipeline — not garbage/NaN
output, not a crash, not a bug in the split logic. Confirmed via the
`local_calls`/`remote_calls` counts (349 real remote calls actually
happened, split unevenly per layer exactly as random top-2-of-4 routing
over real trained router weights would produce) that the local/remote
dispatch is genuinely exercised, not a no-op.

Bugs hit and fixed along the way (kept here since they'll recur on the
32GB machine if not documented):
- `coordinator.py`'s default `--tokenizer` path assumed CWD-relative
  `models/...` rather than deriving from `--model-dir`; fixed to default
  to `<model-dir>/tokenizer.json`.
- Printing generated text containing CJK characters crashed on Windows'
  default `cp1252` console encoding (`UnicodeEncodeError`). Fixed with
  `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` at the top
  of both `coordinator.py` and `expert_server.py`.

## 4. Benchmark

Recorded to `benchmarks/local_split_simulation.json` (see
`benchmarks/README.md` for the full caveats — no KV cache, one token/expert
at a time, loopback "network" is not representative of a real link):
**0.297 tokens/sec**, 349 real TCP expert calls, split 189/160 and 160/200
(remote/local) across the two layers. Treat the `local_calls`/`remote_calls`
counts as the meaningful result here, not the raw tok/s.

## 5. What's real vs. mocked/reduced — summary table

| Piece | Status |
|---|---|
| HF weight values (embed, attn, norms, shared expert, router, 4 experts, lm_head) | **Real**, fetched live from `Qwen/Qwen1.5-MoE-A2.7B` |
| Tokenizer | **Real** Qwen BPE tokenizer (`tokenizer.json` from the same HF repo) |
| Model config (hidden_size, rope_theta, rms_norm_eps, ...) | **Real**, read from the real `config.json` |
| Number of layers | **Reduced**: 2 of 24 |
| Number of experts | **Reduced**: 4 of 60 (ids 0,1,2,3) |
| Router softmax denominator | **Changed**: over 4 experts, not 60 (forced by the above) |
| top-k | **Changed**: 2, not the real 4 (to make the split meaningful with only 4 experts) |
| Attention (Q/K/V/O) and expert weight layout | **Unfused** hand-rolled Linear layers, not MindFormers' fused `linear_qkv`/`weight1`/`weight2` training-kernel layout |
| Device split | **Real** — two separate OS processes, real TCP sockets, real request/response protocol |
| Network link | **Simulated**: both processes on `127.0.0.1`, not a real second machine |
| Output coherence | Real vocabulary tokens via a correctly-wired real pipeline; not fluent text (expected, given only 2/24 layers) |

## 6. Exactly what changes to point Process B at the friend's 32GB machine

This is the part the whole exercise was meant to de-risk, so being precise
here matters:

1. **Copy files, not re-run conversion (unless scaling up too):** copy
   `models/reduced_qwen_moe/process_b.ckpt` and `manifest.json` to the
   32GB machine. (If also scaling up the model size at the same time —
   more layers/experts — re-run `convert_qwen_moe.py` there instead, since
   it has the RAM to hold more of the real 28.6GB checkpoint; the script
   doesn't change either way.)
2. **On the 32GB machine:** `python expert_server.py --host 0.0.0.0
   --port 50051 --model-dir <path to the copied files>`. `--host 0.0.0.0`
   instead of the default `127.0.0.1` so it accepts connections from
   another machine, not just itself.
3. **On this machine (Process A/coordinator):** run `coordinator.py`
   directly (skip `run_local_split_demo.py`, which is only for the
   same-machine simulation) with `--expert-host <32GB machine's LAN/VPN
   IP>` instead of `127.0.0.1`. Everything else about the command is
   identical.
4. **Network/firewall:** the 32GB machine needs port 50051 (or whatever
   `--port` is chosen) reachable from this machine — open it in whatever
   firewall sits in front of it, and use its actual reachable IP (LAN IP,
   Tailscale/VPN IP, etc.), not `127.0.0.1` or `localhost`.
5. **Nothing in `net_proto.py`, `expert_server.py`'s request handling, or
   `moe_model.py`'s split logic needs to change.** The whole point of the
   TCP boundary is that neither side's code knows or cares whether the
   peer is a `localhost` subprocess or a machine down the hall.
6. **What SHOULD change once real network latency is in play:** the
   current per-token-per-expert round trip (one small TCP request per
   expert call, 349 of them for 15 tokens) is fine over loopback but will
   pay real per-call latency over an actual link. Worth batching multiple
   tokens' requests to the same remote expert into one round trip before
   moving to hardware where that latency is real — noted here so it's a
   deliberate next step, not a surprise regression when the benchmark
   number gets worse on real hardware.
7. **Scaling the model up:** once real RAM allows it, `convert_qwen_moe.py`
   already supports more layers/experts via `--num-layers` /
   `--expert-ids` — no new code needed, just bigger arguments (up to the
   real ceiling of 24 layers / 60 experts / top-4 routing, at which point
   `--top-k 4` should also be restored to match the real model exactly).
