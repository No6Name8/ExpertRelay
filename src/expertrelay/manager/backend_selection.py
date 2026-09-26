"""The Manager picks the compute backend from the machine profile.

Rule: an Ascend NPU -> MindSpore (its kernels are the reason to use
MindSpore at all); CPU only -> numpy, which measured 6-13x faster than
MindSpore's CPU ops on this model's shapes (benchmarks/results/
int8_kernels.json, docs/limitations.md). The reason is recorded alongside
the choice, so every benchmark record says why its backend ran.
"""

from __future__ import annotations

from dataclasses import dataclass

from expertrelay.manager.profile import MachineProfile


@dataclass(frozen=True)
class BackendChoice:
    name: str  # a key of runtime.backends.BACKEND_NAMES
    device: str | None  # passed to the backend; None where it has no notion of device
    reason: str


def select_backend(profile: MachineProfile) -> BackendChoice:
    if "ascend" in profile.accelerators:
        return BackendChoice(
            "mindspore", "Ascend", "Ascend NPU detected: MindSpore is the backend for Ascend"
        )
    return BackendChoice(
        "numpy", None, "CPU only: the numpy int8 kernel measured 6-13x faster than MindSpore CPU ops"
    )
