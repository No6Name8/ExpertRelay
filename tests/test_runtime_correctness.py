"""CLAUDE.md's correctness rule, for the runtime: where expert bytes come
from must not change a single bit of the output.

Reference: every expert held in RAM (RamExpertSource) with resident weights
in RAM, the "full model in RAM, no cache, no prediction" setup. Compared:
experts read on demand with one unbuffered read each (UnbufferedExpertSource,
ours), and everything memory-mapped (the OS-paging baseline). Greedy
decoding must pick the same tokens, and the logits must be bit-identical,
because the arithmetic is the same and only the byte source differs.

The expert cache and the background prefetcher (cache.expert_cache,
predictor.prefetch_policy) are held to the same rule: on or off, at any
cache size, pinned or not, with any number of I/O threads, the output must
be bit-identical to the reference.

Uses a tiny random model through the real store builder. The same property
on the real 24-layer store is checked by bench/phase2_baselines.py, which
records every baseline's generated tokens side by side.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from store_helpers import build_test_store, random_checkpoint

from expertrelay.cache.expert_cache import CachedExpertSource
from expertrelay.predictor.prefetch_policy import PrefetchPolicy
from expertrelay.runtime.generate import generate, model_config
from expertrelay.runtime.qwen_moe import KVCache, QwenMoe
from expertrelay.runtime.weights import (
    EMBEDDING,
    MmapExpertSource,
    RamExpertSource,
    ResidentWeights,
    UnbufferedExpertSource,
)
from expertrelay.store.expert_reader import ExpertStoreReader

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="unbuffered expert reads are Windows-only")

CONFIG = {
    "vocab_size": 64,
    "hidden_size": 32,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 16,
    "shared_expert_intermediate_size": 24,
    "norm_topk_prob": False,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
}


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    d = tmp_path_factory.mktemp("store")
    build_test_store(d, CONFIG, random_checkpoint(CONFIG, seed=1), mp)
    mp.undo()
    return d


def _model(store, source_cls, mmap_resident=False) -> QwenMoe:
    return QwenMoe(
        model_config(store), ResidentWeights.load(store, mmap_everything=mmap_resident), source_cls(store)
    )


def test_all_expert_sources_give_bit_identical_generation(store):
    prompt = [3, 9, 27, 17, 51]
    outputs = {}
    for name, cls, mmap_resident in (
        ("ram (reference)", RamExpertSource, False),
        ("unbuffered (ours)", UnbufferedExpertSource, False),
        ("mmap (OS paging)", MmapExpertSource, True),
    ):
        model = _model(store, cls, mmap_resident)
        cache = KVCache(model.c, max_seq=32)
        prefill_logits, _ = model.forward(np.array(prompt), cache, all_logits=True)
        tokens, steps = generate(model, prompt, max_new_tokens=8, max_seq=32)
        outputs[name] = (prefill_logits, tokens)
        model.experts.close()
        assert sum(s.expert_loads for s in steps) > 0

    ref_logits, ref_tokens = outputs["ram (reference)"]
    for name, (logits, tokens) in outputs.items():
        np.testing.assert_array_equal(logits, ref_logits, err_msg=name)  # bit-identical, not "close"
        assert tokens == ref_tokens, name


def test_unbuffered_source_reads_exactly_one_record_per_expert_use(store):
    model = _model(store, UnbufferedExpertSource)
    _, steps = generate(model, [1, 2, 3], max_new_tokens=3, max_seq=16)
    record = model.experts.layout.record_size
    for s in steps:
        assert s.expert_bytes == s.expert_loads * record
    # decode: one token, so exactly top_k experts per layer, each loaded once
    assert all(
        s.expert_loads == CONFIG["num_hidden_layers"] * CONFIG["num_experts_per_tok"] for s in steps[1:]
    )
    model.experts.close()


def test_embedding_stays_memory_mapped_everything_else_in_ram(store):
    resident = ResidentWeights.load(store)
    assert isinstance(resident[EMBEDDING].q.base, np.memmap)  # a view straight onto the mapped file
    q_proj = resident["model.layers.0.self_attn.q_proj.weight"]
    assert not isinstance(q_proj.q.base, np.memmap)  # read into process memory
    assert resident.ram_bytes > 0


def _cached_model(store, *, slots: int, prefetch_k: int, pin_layer0: bool, io_threads: int) -> QwenMoe:
    config = model_config(store)
    with ExpertStoreReader(store) as r:
        record = r.layout.record_size
    source = CachedExpertSource(
        store,
        capacity_bytes=slots * record,
        pinned=[(0, e) for e in range(config.num_experts)] if pin_layer0 else [],
        io_threads=io_threads,
        min_free_slots=config.top_k + prefetch_k + 1,
    )
    return QwenMoe(config, ResidentWeights.load(store), source, prefetcher=PrefetchPolicy(prefetch_k))


def _reference(store, prompt: list[int], new_tokens: int):
    model = _model(store, RamExpertSource)
    logits, _ = model.forward(np.array(prompt), KVCache(model.c, max_seq=32), all_logits=True)
    tokens, _ = generate(model, prompt, max_new_tokens=new_tokens, max_seq=32)
    return logits, tokens


@pytest.mark.parametrize(
    ("slots", "prefetch_k", "pin_layer0", "io_threads"),
    [
        (5, 2, False, 1),  # smaller than one layer's 8 experts: constant eviction
        (8, 4, False, 2),
        (15, 4, True, 1),  # all of layer 0 pinned
        (24, 4, False, 1),  # everything fits
    ],
)
@pytest.mark.parametrize("prefetch_on", [True, False])
def test_cache_and_prefetcher_give_bit_identical_generation(
    store, slots, prefetch_k, pin_layer0, io_threads, prefetch_on
):
    prompt = [3, 9, 27, 17, 51]
    ref_logits, ref_tokens = _reference(store, prompt, 8)
    model = _cached_model(
        store, slots=slots, prefetch_k=prefetch_k, pin_layer0=pin_layer0, io_threads=io_threads
    )
    model.prefetch_enabled = prefetch_on
    logits, _ = model.forward(np.array(prompt), KVCache(model.c, max_seq=32), all_logits=True)
    tokens, steps = generate(model, prompt, max_new_tokens=8, max_seq=32)
    model.experts.close()
    np.testing.assert_array_equal(logits, ref_logits)  # bit-identical, not "close"
    assert tokens == ref_tokens
    issued = sum(s.prefetches_issued for s in steps)
    assert (issued > 0) == prefetch_on
    for s in steps:
        uses = s.cache_hits + s.prefetch_hits + s.prefetch_waits + s.demand_reads
        assert uses == s.expert_loads


def test_prediction_switch_can_flip_mid_generation(store):
    prompt = [5, 1, 60, 33]
    _, ref_tokens = _reference(store, prompt, 8)
    model = _cached_model(store, slots=8, prefetch_k=4, pin_layer0=False, io_threads=1)
    cache = KVCache(model.c, max_seq=32)
    feed, tokens = prompt, []
    for i in range(8):
        model.prefetch_enabled = i % 2 == 0
        logits, _ = model.forward(np.array(feed), cache)
        tokens.append(int(np.argmax(logits)))
        feed = [tokens[-1]]
    model.experts.close()
    assert tokens == ref_tokens


def test_per_layer_timings_cover_every_layer(store):
    model = _cached_model(store, slots=8, prefetch_k=4, pin_layer0=False, io_threads=1)
    _, steps = generate(model, [1, 2, 3], max_new_tokens=3, max_seq=16)
    model.experts.close()
    layers = CONFIG["num_hidden_layers"]
    for s in steps:
        assert len(s.attention_seconds) == len(s.moe_compute_seconds) == len(s.read_wait_seconds) == layers
        assert min(s.attention_seconds + s.moe_compute_seconds + s.read_wait_seconds) >= 0
        assert sum(s.attention_seconds + s.moe_compute_seconds + s.read_wait_seconds) <= s.total_seconds
        assert sum(s.read_wait_seconds) == pytest.approx(s.expert_read_seconds)
