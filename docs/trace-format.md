# Expert-usage trace format (`.ert`, version 1)

Written by `expertrelay.runtime.expert_trace.ExpertTraceWriter` (via
`python -m expertrelay.bench.phase3_traces`), read by `read_trace` in the
same module. One file per prompt, in `models/traces/<store directory
name>/<prompt id>.ert`, next to a `<prompt id>.done.json` that marks the
prompt as finished and holds its generated tokens and timing. A trace
without its `.done.json` belongs to an interrupted prompt and is ignored
by the analysis. The next trace run redoes that prompt.

## Layout

```
offset  size  field
0       8     magic: ASCII "ERTRACE1"
8       4     header length H, u32 little-endian
12      H     header: UTF-8 JSON object
12+H    ...   records: N fixed-size records, back to back, no padding
```

A file whose record area isn't a whole number of records is rejected as
truncated.

### Header (JSON)

| key | meaning |
|---|---|
| `format_version` | `1` |
| `num_layers`, `num_experts`, `top_k` | model shape; they fix the record layout |
| `record_itemsize` | bytes per record (check against the layout below) |
| `prompt_id`, `category`, `language` | from `benchmarks/prompts/phase3.json` |
| `prompt_format` | `plain` (text as-is) or `chatml` (wrapped in the Chat model's template) |
| `store`, `source_revision` | which int8 store and which pinned Hugging Face commit |
| `max_new_tokens` | generation length of the run |

### Record (one per token per layer)

Little-endian, packed. `E` = `num_experts`, `K` = `top_k`. For Qwen1.5-MoE-A2.7B
(E = 60, K = 4) a record is 278 bytes, so one token across 24 layers is 6.7 KB.

| field | type | meaning |
|---|---|---|
| `position` | u32 | token position in the sequence, 0 = first prompt token |
| `token` | u32 | token id at that position |
| `layer` | u8 | decoder layer |
| `phase` | u8 | 0 = prefill (the prompt, one forward call), 1 = decode (one call per generated token) |
| `experts` | u8[K] | chosen experts, highest router weight first |
| `weights` | f32[K] | their router softmax probabilities over all E experts (not renormalized, as the model uses them) |
| `logits` | f16[E] | full router logits |
| `fate_experts` | u8[K] | Fate-style prediction for this layer; 255 at layer 0 |
| `fate_logits` | f16[E] | this layer's router applied to the PREVIOUS layer's gate input; all 0 at layer 0 |
| `prev_token_experts` | u8[K] | the previous position's `experts` at this layer; 255 at position 0 |

Records are ordered by position, then layer. Each forward call appends its
records as one block, so an interrupted run leaves whole calls behind.

## The Fate-style prediction

Z. Fang et al., "Fate: Fast Edge Inference of Mixture-of-Experts Models via
Cross-Layer Gate", arXiv:2502.12224 (2025); ACM Web Conference 2026. At
layer L, the gate input (the hidden state after layer L's post-attention
RMSNorm, which layer L's router reads) is fed to layer L+1's router. Its
top-K guesses layer L+1's experts before layer L+1 runs. The trace stores
that guess in layer L+1's record. It's computed only for the trace and
never feeds back into the forward pass. We record only the predictor, not
Fate's prefetching system or its caching strategy.

## Reading one

```python
from expertrelay.runtime.expert_trace import read_trace

header, rec = read_trace("models/traces/qwen1.5-moe-a2.7b-int8/en_01.ert")
decode = rec[rec["phase"] == 1]
print(header["prompt_id"], len(decode) // header["num_layers"], "generated tokens")
```
