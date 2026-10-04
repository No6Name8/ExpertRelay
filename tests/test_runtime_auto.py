"""--auto (runtime.auto) on a tiny store: the correctness rule holds
whatever the Manager decides. Tokens must equal the reference (every
expert in RAM, no cache, no prediction) with the Manager's own choice, with
each prefetch mode, I/O thread count, pinning and cache size forced, with
no cache at all, and with the memory budget changed while generating."""

from __future__ import annotations

import sys
from dataclasses import replace

import pytest
from store_helpers import build_test_store, random_checkpoint

import expertrelay.runtime.auto as auto
from expertrelay.cache.expert_cache import CachedExpertSource
from expertrelay.manager.backend_selection import BackendChoice
from expertrelay.manager.policy import PREFETCH_ADAPTIVE, PREFETCH_OFF, PREFETCH_TOP_K, Decisions
from expertrelay.runtime import int8_linear
from expertrelay.runtime.generate import RuntimeConfig, generate, model_config
from expertrelay.runtime.qwen_moe import QwenMoe
from expertrelay.runtime.weights import RamExpertSource, ResidentWeights, UnbufferedExpertSource

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
PROMPT = [3, 9, 27, 17, 51]
NEW_TOKENS = 10


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    d = tmp_path_factory.mktemp("store")
    build_test_store(d, CONFIG, random_checkpoint(CONFIG, seed=7), mp)
    mp.undo()
    return d


@pytest.fixture(autouse=True)
def blocked_kernel():
    """load_model sets the process-wide kernel; numba isn't needed for this
    property, and other tests get the kernel back."""
    before = int8_linear.current_kernel()
    yield
    int8_linear.set_kernel(before)


@pytest.fixture(scope="module")
def reference(store) -> list[int]:
    model = QwenMoe(model_config(store), ResidentWeights.load(store), RamExpertSource(store))
    tokens, _ = generate(model, PROMPT, NEW_TOKENS, 64)
    return tokens


def rt_for(store) -> RuntimeConfig:
    return RuntimeConfig(
        memory_budget_gb=1.0,
        max_seq=64,
        max_new_tokens=NEW_TOKENS,
        prompts_file=store / "unused.json",
        store_dir=store,
        int8_kernel="blocked",
    )


def forced(slots: int, prefetch: str, io_threads: int = 1, pin: bool = False) -> Decisions:
    return Decisions(
        cache_bytes=0,  # filled from the record size below
        cache_slots=slots,
        io_threads=io_threads,
        prefetch=prefetch,
        prefetch_k=8,
        min_probability=0.0,
        pin_layer0=pin,
        backend=BackendChoice("numpy", None, "test"),
        precision="int8",
        reasons={"test": "forced"},
    )


def test_auto_with_the_managers_own_choice(store, reference):
    model, info, decisions, used, _ = auto.auto_load(store, rt_for(store), None)
    tokens, _ = generate(model, PROMPT, NEW_TOKENS, 64)
    model.experts.close()
    assert tokens == reference
    assert used == store and info["manager"]["reasons"]
    assert decisions.prefetch in (PREFETCH_OFF, PREFETCH_ADAPTIVE, PREFETCH_TOP_K)
    assert info["manager"]["inputs"]["compute_probe"]["layer_window_seconds"]


@pytest.mark.parametrize(
    "choice",
    [
        forced(11, PREFETCH_OFF),
        forced(12, PREFETCH_ADAPTIVE, io_threads=2),
        forced(24, PREFETCH_TOP_K, io_threads=2, pin=True),
        forced(11, PREFETCH_TOP_K, io_threads=1),
        forced(0, PREFETCH_OFF),
    ],
    ids=["cache only", "adaptive 2 threads", "top-8 pinned, whole store", "top-8 smallest cache", "no cache"],
)
def test_tokens_identical_whatever_the_manager_chooses(store, reference, monkeypatch, choice):
    record = auto.store_facts(store, rt_for(store), "numpy").record_bytes
    choice = replace(choice, cache_bytes=choice.cache_slots * record)
    monkeypatch.setattr(auto, "decide", lambda *a, **k: choice)
    model, info, _, _, _ = auto.auto_load(store, rt_for(store), None)
    assert isinstance(model.experts, CachedExpertSource) == (choice.cache_slots > 0)
    if choice.cache_slots:
        assert model.experts.num_slots == choice.cache_slots
        assert model.prefetch_enabled == (choice.prefetch != PREFETCH_OFF)
        assert model.prefetch_adaptive == (choice.prefetch == PREFETCH_ADAPTIVE)
    else:
        assert isinstance(model.experts, UnbufferedExpertSource)
    tokens, _ = generate(model, PROMPT, NEW_TOKENS, 64)
    model.experts.close()
    assert tokens == reference


def test_memory_budget_changed_while_generating(store, reference, monkeypatch, tmp_path):
    record = auto.store_facts(store, rt_for(store), "numpy").record_bytes
    monkeypatch.setattr(
        auto, "decide", lambda *a, **k: replace(forced(24, PREFETCH_TOP_K, 2), cache_bytes=24 * record)
    )
    model, _, _, _, _ = auto.auto_load(store, rt_for(store), None)
    facts = auto.store_facts(store, rt_for(store), "numpy")
    budget_file = tmp_path / "budget.txt"
    live = auto.LiveBudget(model, facts, total_ram_bytes=8 * 10**9, file=budget_file)
    plan = {2: facts.base_ram_bytes + 13 * record, 5: facts.base_ram_bytes + 3 * record}

    def slider():
        if live._tokens + 1 in plan:  # the demo's slider writes the file; LiveBudget re-reads it
            budget_file.write_text(f"{plan[live._tokens + 1] / 1e9:.12f}")
        live()

    tokens, _ = generate(model, PROMPT, NEW_TOKENS, 64, between_tokens=slider)
    assert tokens == reference
    sizes = [(e["slots_before"], e["slots_after"]) for e in live.events]
    assert sizes == [(24, 13), (13, 11)]  # then the smallest cache it can run with
    assert "smallest" in live.events[1]["reason"]
    live.set(1.0)  # back up: grows again, still identical afterwards
    live()
    assert model.experts.num_slots == 24  # capped at the whole store
    model.experts.close()
