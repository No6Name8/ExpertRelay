"""File extents, the contiguous rewrite of experts.bin, and the read
diagnosis's timing loop, on small files and a tiny store."""

from __future__ import annotations

import hashlib
import sys

import numpy as np
import pytest
from store_helpers import build_test_store, random_checkpoint

from expertrelay.bench.read_diagnosis import run_test
from expertrelay.store.expert_reader import EXPERTS_BIN, EXPERTS_INDEX
from expertrelay.store.file_extents import Extent, cluster_size_for, file_extents, parse_queryextents, pieces
from expertrelay.store.layout import read_expert_index
from expertrelay.store.rewrite_experts import bad_records, copy_unbuffered

windows_only = pytest.mark.skipif(
    sys.platform != "win32", reason="unbuffered I/O and fsutil are Windows-only"
)

SAMPLE = """VCN: 0x0        Clusters: 0xb        LCN: 0x5c67bdc
VCN: 0xb        Clusters: 0x1a       LCN: 0x6170724
VCN: 0x25       Clusters: 0x28       LCN: -1
"""


def test_parse_queryextents():
    assert parse_queryextents(SAMPLE) == [(0x0, 0xB), (0xB, 0x1A), (0x25, 0x28)]


def test_cluster_size_from_file_size():
    assert cluster_size_for(0x4D * 4096, 0x4D) == 4096
    assert cluster_size_for(0x4D * 4096 - 100, 0x4D) == 4096  # last cluster partly used
    with pytest.raises(ValueError):
        cluster_size_for(3000, 4)  # 4 x 512 is too small, 4 x 1024 leaves more than a cluster over


def test_pieces():
    ex = [Extent(0, 100), Extent(100, 50), Extent(150, 1000)]
    assert pieces(ex, 0, 100) == 1
    assert pieces(ex, 90, 20) == 2
    assert pieces(ex, 10, 500) == 3
    assert pieces(ex, 150, 1000) == 1


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    config = {
        "vocab_size": 64,
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 16,
        "shared_expert_intermediate_size": 24,
        "norm_topk_prob": False,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
    }
    mp = pytest.MonkeyPatch()
    d = tmp_path_factory.mktemp("store")
    build_test_store(d, config, random_checkpoint(config, seed=4), mp)
    mp.undo()
    return d


@windows_only
def test_copy_is_byte_identical_and_verifies(store, tmp_path):
    layout, entries = read_expert_index(store / EXPERTS_INDEX)
    dst = tmp_path / "copy.bin"
    copy_unbuffered(store / EXPERTS_BIN, dst, chunk=2 * 4096)  # small chunks: many iterations
    assert (
        hashlib.sha256(dst.read_bytes()).digest()
        == hashlib.sha256((store / EXPERTS_BIN).read_bytes()).digest()
    )
    assert bad_records(dst, entries, layout.record_size) == []
    assert sum(e.length for e in file_extents(dst)) >= dst.stat().st_size


@windows_only
def test_verification_catches_a_changed_byte(store, tmp_path):
    layout, entries = read_expert_index(store / EXPERTS_INDEX)
    data = bytearray((store / EXPERTS_BIN).read_bytes())
    victim = entries[3]
    data[victim.offset + 17] ^= 0xFF
    bad = tmp_path / "bad.bin"
    bad.write_bytes(bytes(data))
    assert bad_records(bad, entries, layout.record_size) == [(victim.layer, victim.expert)]


@windows_only
@pytest.mark.parametrize("test", ["back_to_back", "gapped", "qd2", "qd4"])
def test_read_diagnosis_reads_every_record_once(store, test):
    _, entries = read_expert_index(store / EXPERTS_INDEX)
    sample = [entries[i] for i in np.random.default_rng(0).permutation(len(entries))]
    r = run_test(store / EXPERTS_BIN, sample, test, [1] * len(sample))
    assert r["reads"] == len(sample)
    assert r["queue_depth"] == (int(test[2:]) if test.startswith("qd") else 1)
    assert r["by_pieces"]["1"]["reads"] == len(sample)
    assert r["mb_per_s"] > 0
