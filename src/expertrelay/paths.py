"""Centralized filesystem locations.

Nothing in this repo should hardcode a relative path that only resolves
correctly from one particular working directory (e.g. "the script's own
directory") -- every default path used by an entry point comes from here,
anchored to `expertrelay.REPO_ROOT`.
"""

from __future__ import annotations

from pathlib import Path

from expertrelay import REPO_ROOT

MODELS_ROOT: Path = REPO_ROOT / "models"
DEFAULT_EXPERT_STORE_DIR: Path = MODELS_ROOT / "qwen1.5-moe-a2.7b-int8"
BENCHMARK_RESULTS_DIR: Path = REPO_ROOT / "benchmarks" / "results"
