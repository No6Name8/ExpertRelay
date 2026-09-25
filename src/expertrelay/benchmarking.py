"""Shared benchmark-recording utilities.

Every benchmark in this repo writes its raw results as JSON to
benchmarks/results/, including the git commit hash, a machine profile,
model/config, seed, and timestamp -- see CLAUDE.md's benchmarking rule.
This module is the one place that assembles that common metadata, so each
benchmark script doesn't re-implement (and inevitably drift from) it.

Machine facts come only from expertrelay.manager.profile. base_record takes
the profile as a required argument instead of importing and calling that
module itself: the profile CLI uses base_record to save its own results, so
importing in both directions would be a circular import. The type-only
import below gives us the type check without the runtime cycle.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import psutil

from expertrelay import REPO_ROOT

if TYPE_CHECKING:
    from expertrelay.manager.profile import MachineProfile


def git_commit_hash() -> str | None:
    """The commit this benchmark ran at, or None if not in a git checkout
    (e.g. a source tarball) -- callers must not assume this is always set."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def git_is_dirty() -> bool | None:
    """Whether the working tree had uncommitted changes when this benchmark
    ran, or None if that can't be determined -- a benchmark tied to a commit
    hash that was actually run against a dirty tree is a common way for
    "reproduce this" to quietly fail; record it rather than hide it."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return bool(result.stdout.strip())
    except Exception:
        return None


def peak_process_rss_mb() -> float:
    """Peak memory used by the CURRENT process so far, in MB.

    psutil has no portable peak-RSS accessor: Windows exposes `peak_wset`;
    everywhere else we fall back to the current RSS, which understates the
    true peak for a process whose memory has since been freed. Called at the
    end of a benchmark run (after the memory-heavy work), so the understatement
    is usually small -- but it IS an approximation, not a hard peak on POSIX.
    """
    info = psutil.Process().memory_info()
    peak_bytes = getattr(info, "peak_wset", None) or info.rss
    return peak_bytes / 1e6


def append_benchmark_record(path: Path, record: dict) -> None:
    """Append one JSON record to a list-of-records file, creating it if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.exists():
        existing = json.loads(path.read_text())
    existing.append(record)
    path.write_text(json.dumps(existing, indent=2))


def base_record(
    *,
    label: str,
    seed: int,
    model: dict | None,
    config: dict,
    machine: MachineProfile,
    **extra: Any,
) -> dict:
    """Assemble the fields every benchmark record must carry, per CLAUDE.md.

    `model` is None only for benchmarks that don't involve a model (e.g. the
    machine profile's own disk-speed test).
    """
    record = {
        "label": label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit_hash(),
        "git_dirty": git_is_dirty(),
        "machine": machine.to_dict(),
        "seed": seed,
        "model": model,
        "config": config,
    }
    record.update(extra)
    return record
