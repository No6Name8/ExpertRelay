# Concept note: verified facts

The fact sheet behind the public concept note. Every number here is copied
from a file in this repo, and that file is named next to it. A number that
isn't in this sheet shouldn't be in the concept note. Where a figure is
derived rather than read directly, the arithmetic is shown.

Results files record the commit of the code that produced them. That commit
is often older than the one that committed the results, and both are given
here. Several results files hold more than one record (one per run); each
source below names the record by its position in the file, its timestamp
and its commit.

---

## 1. Machine

Source: `benchmarks/results/machine_profile_THE-BEAST.json`, **record 1
of 2** (timestamp 2026-09-25T06:23:24Z, code commit `fb084d1`, with a dirty
working tree). Record 2 is a rerun on 2026-09-28 (commit `87e5425`) that
measured 4,027-4,210 MB/s on the same test; the table uses record 1. Its
fresh-file read speed is NOT the speed of reading the model's experts,
which is ~2.0 GB/s (`benchmarks/results/read_diagnosis.json`, see
`docs/limitations.md`).

| Fact | Value |
|---|---|
| CPU | 12th Gen Intel Core i5-12450H, 8 cores / 12 threads |
| RAM, total | 8,300,781,568 bytes (≈ 8.3 GB) |
| RAM, available when measured | 2,035,322,880 bytes (≈ 2.0 GB) |
| Drive | CS3160PGFCV100DCAFF NVMe SSD, 510,263,291,904 bytes |
| GPU / NPU | none used; the runtime runs on the CPU (numpy backend) |

Uncached SSD reads. Method: Win32 `FILE_FLAG_NO_BUFFERING`, 1 GiB test file,
3 runs, OS-cache bypass verified for every run. Same source file.

| Read pattern | Block size | Throughput |
|---|---|---|
| Sequential | 8.65 MB (one expert) | 4289.3 MB/s |
| Random | 8.65 MB | 4149.5 MB/s |
| Sequential | 17.30 MB | 4348.1 MB/s |
| Random | 17.30 MB | 4382.2 MB/s |

## 2. Model and store

**Model.** Qwen1.5-MoE-A2.7B (Qwen/Qwen1.5-MoE-A2.7B), pinned to revision
`1a758c50ecb6350748b9ce0a99d2352fd9fc11c9`. Source:
`benchmarks/results/expert_store_build_THE-BEAST.json`, **record 1 of 2**
(the base model: timestamp 2026-09-26T02:14:34Z, code commit `86f796e`).
Record 2 is the Chat model's store build (Qwen/Qwen1.5-MoE-A2.7B-Chat,
2026-09-26T19:30:15Z); no number here comes from it. Every
`expert_store_build_THE-BEAST.json` below means record 1.

| Fact | Value | Source |
|---|---|---|
| Layers | 24 | `docs/expert-store.md`, `docs/phase2-baselines.md` |
| Routed experts per layer | 60, so 1,440 in total | `docs/expert-store.md`; the 1,440 count is also in `expert_store_build_THE-BEAST.json` (verification) |
| Experts active per token per layer | top-4 routed (`docs/phase2-baselines.md`), plus the always-on shared expert, which is kept in RAM (`docs/limitations.md`, "resident part") and has its own gate (`docs/expert-store.md`) | as given |
| Original checkpoint size (bf16) | 28,631,568,384 bytes (≈ 28.6 GB) | `expert_store_build_THE-BEAST.json` |
| Total parameters | ≈ 14.3 billion (derived: 28,631,568,384 bytes ÷ 2 bytes per bf16 value = 14,315,784,192) | derived from `expert_store_build_THE-BEAST.json` |
| Active parameters per token | ≈ 2.69 billion (derived: 1,858,701,312 resident + 96 × 8,650,752 routed-expert weights = 2,689,173,504). This matches the "A2.7B" in the model's name. | derived from `expert_store_build_THE-BEAST.json` |

**Store** (int8, on the SSD). Source:
`benchmarks/results/expert_store_build_THE-BEAST.json`, record 1 of 2 (code
commit `86f796e`; results committed in `93e5a7a`).

| Fact | Value |
|---|---|
| Total store size | 14,358,396,928 bytes (≈ 14.36 GB) |
| Routed experts | 12,486,574,080 bytes |
| Resident part (attention, norms, router, shared experts, embedding, lm_head) | 1,871,822,848 bytes |
| One expert record | 8,671,232 bytes (8,650,752 weights + 19,456 scales, padded to 4096-byte alignment), read with one unbuffered read |
| Quantization | int8, symmetric, per output row (absmax); weight-only |
| Expert reconstruction error | mean 0.8300%, median 0.8273%, max 1.0286% (layer 3, expert 33) |
| Resident int8 tensors (170) | mean 0.984%, max 1.92% (`lm_head`) |
| Integrity check | 1,440 experts and 339 resident tensors checked against their sha256; 0 bad |
| Peak RAM while building the store | 469 MB |

## 3. Phase 2 results: first full run

Source: `benchmarks/results/phase2_baselines.json`, **record 2 of 2** (code
commit `5406ea5`, clean tree; timestamp 2026-09-26T12:08:35Z). Record 1 is
the first run (commit `90b5e9e`, 2026-09-26T11:40:56Z). Written up in
`docs/phase2-baselines.md`.

**Workload:**
- 4 prompts (English, Arabic, Python code, math) with 18, 18, 21 and 38
  prompt tokens. Prompt text: `benchmarks/prompts/phase2.json`.
- 32 new tokens per prompt, greedy decoding, so 124 decode tokens per
  setup.
- Free RAM at the start of the run: 2.98 GB.

| Setup | Result | Decode speed | Time to first token | Peak working set | Committed memory | Device reads, total |
|---|---|---|---|---|---|---|
| **A**: normal load (whole model into RAM) | never finished loading; killed at the 900 s timeout | none | none | 4,661 MB | 14,608 MB | 450.9 GB |
| **B**: memory-mapped, OS paging | ran | 0.406 tok/s | 30.3 s | 4,431 MB | 557 MB | 113.6 GB |
| **C**: ExpertRelay (int8 store, one read per expert) | ran | **0.702 tok/s** | **12.5 s** | **1,685 MB** | 2,101 MB | 139.5 GB |

**Per decode token, setup C:**
- 833.8 MB read from the device, of which 832.4 MB are experts.
- 0.471 s reading and 0.954 s computing, so reads are 0.33 of the time.

For setup B, 591.4 MB is read from the device per decode token.

**Identical tokens:**
- Setups B and C produced identical output tokens for all 4 prompts ×
  32 tokens.
- This held in both recorded runs: the first run (runtime commit
  `90b5e9e`) and the second.

Source: `benchmarks/results/phase2_baselines.json`, where both records
have `token_agreement.identical: true` for C vs. B; also
`docs/limitations.md`.

What this does **not** show:
- C has no cache and no prediction yet. Every expert is read when the router
  picks it, then dropped.
- The speed is on this one 8 GB laptop.
- See `docs/limitations.md`.

## 4. Correctness

**Verified: automated tests, all passing.** They use small synthetic
models, with the same code path as the real model.

| What | Test |
|---|---|
| Our runtime matches Hugging Face transformers' own Qwen2-MoE implementation on a tiny model, to 1e-4 on the logits, for both prefill and token-by-token KV-cache decoding | `tests/test_runtime_vs_hf_transformers.py` |
| Output is bit-identical whichever way experts are loaded (unbuffered reads, memory-mapped, all in RAM) | `tests/test_runtime_correctness.py` |
| The numpy and MindSpore backends produce the same tokens | `tests/test_runtime_backends.py` |
| The reference-check harness reproduces HF's full forward pass to 1e-5 | `tests/test_reference_check.py` |
| Recording expert traces doesn't change the output tokens | `tests/test_expert_trace.py` |

On the real model, B and C give identical tokens (section 3).

**Pending: the real-model accuracy of int8 vs. the original bf16 weights.**
- `bench/reference_check.py` compares our int8 hidden states and logits,
  layer by layer, against HF transformers on the original bf16 weights.
- It needs the 28.6 GB bf16 checkpoint on disk, and that download hasn't
  finished.
- Until it runs, the real-model accuracy of the int8 runtime is
  **unmeasured**. Source: `docs/limitations.md`.
- No quality benchmark has been run either. Source: `docs/limitations.md`.

## 5. Dates

From the git history of this repo.

| Event | Commit | Date |
|---|---|---|
| First commit | `27c270de0ef30c2f3f58b9ced5459086d186b50c` | 2026-09-13 13:19:59 +0300 |
| Int8 store built (results committed) | `93e5a7a` | 2026-09-26 05:16:26 +0300 |
| Full 24-layer runtime | `90b5e9e` | 2026-09-26 14:39:33 +0300 |
| Phase 2 results committed | `89a273b6187c22c3fcaf3f9c4643b05a2e9bb333` | 2026-09-26 15:36:30 +0300 |

## 6. References

Checked on 2026-09-26 against arXiv abstract pages, or the project's own
page where there is no paper. Notes on each:

- **APEX: ambiguous. Choose one before citing.** Two different works use
  the name:
  - (a) Kanani, Badawi, Ogras, *APEX: Adaptive Expert Prefetching for
    Memory-Efficient Edge MoE Inference*, arXiv:2608.11688 (Aug 2026).
  - (b) *MoE-APEX: An Efficient MoE Inference System with Adaptive
    Precision Expert Offloading*, ASPLOS 2026.
  - (b)'s title and DOI (10.1145/3779212.3790187) were confirmed via
    search, but the ACM page itself refused access (HTTP 403).
  - (b)'s author list (the same eight authors as HOBBIT) comes only from
    search results; it was not read on the ACM page. It looks like the
    conference version of HOBBIT, but that is **unverified**.
  - Both entries are below. Delete the one you don't mean.
- **MoE-Infinity:** the title changed between arXiv versions. The entry
  uses the current title. v1 (Jan 2024) was "MoE-Infinity:
  Activation-Aware Expert Offloading for Efficient MoE Serving".
- **Flash-MoE** is a GitHub project, not a paper:
  - Author: Dan Woods (danveloper). Repo created 2026-03-18 (GitHub API).
  - The repo contains a paper PDF, `paper/flash_moe.pdf`, which was
    **not read**.
  - It's a different work from "FlashMoE" (arXiv:2601.17063), a paper on
    ML-based cache replacement.
- **Qwen1.5-MoE** has no paper. The citation is the Qwen team's blog post
  (2024-03-28), plus the pinned model revision.
- **DAOP:** submitted to arXiv on 2024-12-16 and published at DATE 2025.
  The page range isn't verified, so the entry has none.
- **SlimCaching:** accepted by IEEE Transactions on Mobile Computing (per
  its arXiv page). Volume and pages aren't verified, so it's cited as the
  arXiv preprint.

```bibtex
@inproceedings{fang2026fate,
  title     = {Fate: Fast Edge Inference of Mixture-of-Experts Models via Cross-Layer Gate},
  author    = {Fang, Zhiyuan and Hong, Zicong and Huang, Yuegui and Lyu, Yufeng and Chen, Wuhui and Yu, Yue and Yu, Fan and Zheng, Zibin},
  booktitle = {Proceedings of the ACM Web Conference 2026 (WWW '26)},
  year      = {2026},
  doi       = {10.1145/3774904.3792527},
  eprint    = {2502.12224},
  archivePrefix = {arXiv},
  note      = {arXiv v1 submitted 2025-02-17}
}

% APEX, candidate (a). AMBIGUOUS: confirm this is the intended "APEX".
@misc{kanani2026apex,
  title         = {APEX: Adaptive Expert Prefetching for Memory-Efficient Edge MoE Inference},
  author        = {Kanani, Alish and Badawi, Layan and Ogras, Umit Y.},
  year          = {2026},
  eprint        = {2608.11688},
  archivePrefix = {arXiv}
}

% APEX, candidate (b). AMBIGUOUS, and PARTLY UNVERIFIED: title and DOI confirmed
% via search; the ACM page was not readable (HTTP 403), and the author list
% is from search results only.
@inproceedings{tang2026moeapex,
  title     = {MoE-APEX: An Efficient MoE Inference System with Adaptive Precision Expert Offloading},
  author    = {Tang, Peng and Liu, Jiacheng and Hou, Xiaofeng and Pu, Yifei and Wang, Jing and Heng, Pheng-Ann and Li, Chao and Guo, Minyi},
  booktitle = {Proceedings of the 31st ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS '26)},
  year      = {2026},
  doi       = {10.1145/3779212.3790187}
}

@misc{tang2024hobbit,
  title         = {HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference},
  author        = {Tang, Peng and Liu, Jiacheng and Hou, Xiaofeng and Pu, Yifei and Wang, Jing and Heng, Pheng-Ann and Li, Chao and Guo, Minyi},
  year          = {2024},
  eprint        = {2411.01433},
  archivePrefix = {arXiv}
}

@misc{xue2024moeinfinity,
  title         = {MoE-Infinity: Efficient MoE Inference on Personal Machines with Sparsity-Aware Expert Cache},
  author        = {Xue, Leyang and Fu, Yao and Lu, Zhan and Mai, Luo and Marina, Mahesh},
  year          = {2024},
  eprint        = {2401.14361},
  archivePrefix = {arXiv}
}

@misc{eliseev2023mixtraloffloading,
  title         = {Fast Inference of Mixture-of-Experts Language Models with Offloading},
  author        = {Eliseev, Artyom and Mazur, Denis},
  year          = {2023},
  eprint        = {2312.17238},
  archivePrefix = {arXiv}
}

@misc{li2025primacpp,
  title         = {Prima.cpp: Fast 30-70B LLM Inference on Heterogeneous and Low-Resource Home Clusters},
  author        = {Li, Zonghang and Li, Tao and Feng, Wenjiao and Xiao, Rongxing and She, Jianshu and Huang, Hong and Guizani, Mohsen and Yu, Hongfang and Ho, Qirong and Xiang, Wei and Liu, Xue},
  year          = {2025},
  eprint        = {2504.08791},
  archivePrefix = {arXiv}
}

@inproceedings{zhang2025daop,
  title         = {DAOP: Data-Aware Offloading and Predictive Pre-Calculation for Efficient MoE Inference},
  author        = {Zhang, Yujie and Aggarwal, Shivam and Mitra, Tulika},
  booktitle     = {Design, Automation and Test in Europe Conference (DATE)},
  year          = {2025},
  eprint        = {2501.10375},
  archivePrefix = {arXiv}
}

@misc{chen2025slimcaching,
  title         = {SlimCaching: Edge Caching of Mixture-of-Experts for Distributed Inference},
  author        = {Chen, Qian and Chen, Xianhao and Huang, Kaibin},
  year          = {2025},
  eprint        = {2507.06567},
  archivePrefix = {arXiv},
  note          = {Accepted by IEEE Transactions on Mobile Computing}
}

@misc{li2026moespac,
  title         = {MoE-SpAc: Efficient MoE Inference Based on Speculative Activation Utility in Heterogeneous Edge Scenarios},
  author        = {Li, Shuhuai and Lin, Jianghao and Ge, Dongdong and Ye, Yinyu},
  year          = {2026},
  eprint        = {2603.09983},
  archivePrefix = {arXiv}
}

@inproceedings{alizadeh2024llminaflash,
  title         = {LLM in a flash: Efficient Large Language Model Inference with Limited Memory},
  author        = {Alizadeh, Keivan and Mirzadeh, Iman and Belenko, Dmitry and Khatamifard, Karen and Cho, Minsik and Del Mundo, Carlo C. and Rastegari, Mohammad and Farajtabar, Mehrdad},
  booktitle     = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL)},
  year          = {2024},
  eprint        = {2312.11514},
  archivePrefix = {arXiv}
}

% Software project, not a paper. Its paper/flash_moe.pdf was not read.
@misc{woods2026flashmoe,
  title        = {Flash-MoE: Running a big model on a small laptop},
  author       = {Woods, Dan},
  year         = {2026},
  howpublished = {\url{https://github.com/danveloper/flash-moe}},
  note         = {GitHub repository, created 2026-03-18}
}

@misc{qwen2024qwen15moe,
  title        = {Qwen1.5-MoE: Matching 7B Model Performance with 1/3 Activated Parameters},
  author       = {{Qwen Team}},
  year         = {2024},
  month        = mar,
  howpublished = {\url{https://qwenlm.github.io/blog/qwen-moe/}},
  note         = {Blog post, 2024-03-28. Model used: Qwen/Qwen1.5-MoE-A2.7B, revision 1a758c50ecb6350748b9ce0a99d2352fd9fc11c9}
}
```
