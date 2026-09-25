"""Enforced memory-budget checks -- see CLAUDE.md's memory-budget rule.

This repo's whole premise is running MoE inference on RAM-constrained
devices, so "does this fit" has to be a real, enforced check at the point
weights are fetched or loaded, not just a number reported after the fact
(that's what expertrelay.benchmarking.peak_process_rss_mb is for -- a
measurement, not a gate).
"""

from __future__ import annotations


class MemoryBudgetExceeded(RuntimeError):
    pass


def enforce_ram_budget(estimated_bytes: int, max_gb: float, what: str) -> None:
    """Raise MemoryBudgetExceeded if `estimated_bytes` would exceed `max_gb`.

    `estimated_bytes` is a caller-supplied estimate (e.g. total fetched
    tensor bytes, or a checkpoint file's on-disk size, which for these
    fp32 checkpoints is a reasonable proxy for the RAM loading it will use)
    -- this gates an operation BEFORE it happens, it does not measure actual
    process RSS after the fact.
    """
    estimated_gb = estimated_bytes / 1e9
    if estimated_gb > max_gb:
        raise MemoryBudgetExceeded(
            f"{what} would use an estimated {estimated_gb:.2f} GB, over the {max_gb:.2f} GB budget "
            f"(--max-ram-gb) -- reduce --num-layers/--expert-ids or raise the budget explicitly"
        )
