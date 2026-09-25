"""expertrelay.memory_budget must actually gate, not just report -- this is
the enforcement CLAUDE.md's memory-budget rule requires."""

from __future__ import annotations

import pytest

from expertrelay.memory_budget import MemoryBudgetExceeded, enforce_ram_budget


def test_within_budget_does_not_raise():
    enforce_ram_budget(1_000_000_000, max_gb=2.0, what="test")  # 1 GB under a 2 GB budget


def test_over_budget_raises_with_useful_message():
    with pytest.raises(MemoryBudgetExceeded, match="3.00 GB"):
        enforce_ram_budget(3_000_000_000, max_gb=2.0, what="loading foo.ckpt")
