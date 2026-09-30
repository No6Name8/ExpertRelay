"""Offline tests for checkpoint verification.

The Hub listing itself (hub_files) is a live network call and isn't tested
here; it's exercised by every real download, whose result is recorded in
benchmarks/results/checkpoint_verification.json. The hashing that the
check relies on is tested against known values.
"""

from __future__ import annotations

import hashlib

import expertrelay.store.verify_checkpoint as vc


def test_file_hashes_match_sha256_and_git_blob_id(tmp_path):
    path = tmp_path / "f.json"
    path.write_bytes(b"hello\n")
    sha256, blob = vc.file_hashes(path)
    assert sha256 == hashlib.sha256(b"hello\n").hexdigest()
    # `git hash-object` of "hello\n"
    assert blob == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_verify_flags_bad_and_missing_files(tmp_path, monkeypatch):
    (tmp_path / "a.safetensors").write_bytes(b"weights")
    (tmp_path / "b.json").write_bytes(b"{}")
    remote = {
        "a.safetensors": {"size": 7, "oid": "x", "lfs": {"oid": hashlib.sha256(b"weights").hexdigest()}},
        "b.json": {"size": 2, "oid": "0" * 40},
        "c.json": {"size": 1, "oid": "0" * 40},
        "README.md": {"size": 1, "oid": "0" * 40},
    }
    monkeypatch.setattr(vc, "hub_files", lambda repo_id, revision: remote)
    report = vc.verify("r", "rev", tmp_path, ["*.safetensors", "*.json"])
    assert report["bad"] == ["b.json"]
    assert report["missing"] == ["c.json"]
    assert [f["file"] for f in report["files"]] == ["a.safetensors", "b.json"]
