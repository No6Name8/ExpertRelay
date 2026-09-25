"""The correctness rule from CLAUDE.md, scoped to what's automatable here:
ExpertRelay's split execution must match a non-split reference token-for-
token under greedy decoding.

The real target (splitting real Qwen1.5-MoE-A2.7B weights must match the
full model in RAM) needs a network download and 3GB+ of RAM -- unsuitable
for a fast, deterministic pytest run. That check is done manually and
documented in docs/setup-notes.md / docs/limitations.md. What we CAN and
DO automate here: for the same (tiny, synthetic) weights, does routing some
experts through a remote_expert_fn instead of holding them locally change
the answer? It must not -- that is the actual invariant the device split
depends on, independent of model size.
"""

from __future__ import annotations

import mindspore as ms
import numpy as np

from expertrelay.runtime.moe_model import ReducedQwenMoe, RemoteExpertShard

TINY_CFG = {
    "vocab_size": 20,
    "hidden_size": 8,
    "num_attention_heads": 2,
    "moe_intermediate_size": 4,
    "shared_expert_intermediate_size": 8,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "num_layers": 2,
    "expert_ids": [0, 1, 2, 3],
    "top_k": 2,
}


def _seed_all_params(net: ms.nn.Cell, base_seed: int) -> dict[str, np.ndarray]:
    """Deterministically fill every parameter with small pseudo-random values,
    keyed by name so a second net can be given identical values for its
    identically-named parameters."""
    values: dict[str, np.ndarray] = {}
    for i, (name, param) in enumerate(sorted(net.parameters_and_names())):
        rng = np.random.RandomState(base_seed + i)
        arr = rng.uniform(-0.2, 0.2, size=tuple(param.shape)).astype(np.float32)
        param.set_data(ms.Tensor(arr, ms.float32))
        values[name] = arr
    return values


def _copy_matching(net: ms.nn.Cell, values: dict[str, np.ndarray]) -> None:
    for name, param in net.parameters_and_names():
        if name in values:
            param.set_data(ms.Tensor(values[name], ms.float32))


def _copy_into_remote_shard(shard: RemoteExpertShard, values: dict[str, np.ndarray]) -> None:
    """RemoteExpertShard names its parameters `shard_layers.{i}.expert_{eid}.*`,
    not `decoder.layers.{i}.mlp.expert_{eid}.*` like the full model -- map
    explicitly rather than relying on name equality (which would silently
    leave every remote expert at its zero-initialized default)."""
    for name, param in shard.parameters_and_names():
        # name looks like "shard_layers.<i>.expert_<eid>.<proj>.weight"
        _, i_str, expert_part, proj, weight = name.split(".")
        eid = expert_part.removeprefix("expert_")
        source_name = f"decoder.layers.{i_str}.mlp.expert_{eid}.{proj}.{weight}"
        param.set_data(ms.Tensor(values[source_name], ms.float32))


def test_split_execution_matches_unsplit_reference():
    ms.set_context(device_target="CPU")

    # Reference: every expert local, no remote dispatch at all.
    net_full = ReducedQwenMoe(TINY_CFG, local_expert_ids_per_layer=[{0, 1, 2, 3}] * TINY_CFG["num_layers"])
    full_values = _seed_all_params(net_full, base_seed=0)

    # Split: experts {0,1} local to the "coordinator", {2,3} live only on a
    # separate RemoteExpertShard, reached via a plain function call standing
    # in for the TCP round trip (net_proto's wire format is tested on its own
    # in test_runtime_net_proto.py -- this test is about dispatch correctness).
    net_split = ReducedQwenMoe(TINY_CFG, local_expert_ids_per_layer=[{0, 1}] * TINY_CFG["num_layers"])
    _copy_matching(net_split, full_values)

    remote_shard = RemoteExpertShard(
        hidden_size=TINY_CFG["hidden_size"],
        moe_ffn_size=TINY_CFG["moe_intermediate_size"],
        expert_ids_per_layer={i: [2, 3] for i in range(TINY_CFG["num_layers"])},
    )
    _copy_into_remote_shard(remote_shard, full_values)

    def remote_expert_fn(layer_idx: int, expert_id: int, x_np: np.ndarray) -> np.ndarray:
        return remote_shard.run_expert(layer_idx, expert_id, x_np)

    net_split.set_remote_expert_fn(remote_expert_fn)

    token_ids = ms.Tensor(np.arange(12, dtype=np.int32).reshape(1, 12) % TINY_CFG["vocab_size"])

    logits_full = net_full(token_ids).asnumpy()
    logits_split = net_split(token_ids).asnumpy()

    np.testing.assert_allclose(logits_full, logits_split, rtol=1e-5, atol=1e-5)

    # Greedy decoding must pick the SAME token ids from both, not just close logits.
    ids_full = np.argmax(logits_full, axis=-1)
    ids_split = np.argmax(logits_split, axis=-1)
    np.testing.assert_array_equal(ids_full, ids_split)

    # And the split path must have been genuinely exercised (both local AND
    # remote dispatch happened at least once) -- otherwise this test would
    # trivially pass without ever touching remote_expert_fn.
    stats = net_split.call_stats()
    assert sum(s["local_calls"] for s in stats) > 0
    assert sum(s["remote_calls"] for s in stats) > 0
