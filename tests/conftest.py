from __future__ import annotations

import pytest

from expertrelay.manager.profile import (
    EXPERT_BYTES_FP16,
    EXPERT_BYTES_INT8,
    CpuInfo,
    DiskReadBenchmark,
    DriveInfo,
    MachineProfile,
    MemoryInfo,
    ReadSpeed,
    SoftwareInfo,
)


@pytest.fixture
def fake_profile() -> MachineProfile:
    """A hand-built profile with made-up numbers, for testing code that
    consumes a profile without touching real hardware."""
    return MachineProfile(
        hostname="test-host",
        memory=MemoryInfo(total_bytes=8_000_000_000, available_bytes=2_000_000_000),
        cpu=CpuInfo(model="Test CPU 9000", physical_cores=4, logical_cores=8),
        drive=DriveInfo(
            path="C:\\repo",
            model="TestDisk 1TB",
            bus_type="NVMe",
            media_type="SSD",
            total_bytes=1_000_000_000_000,
            free_bytes=100_000_000_000,
        ),
        software=SoftwareInfo(platform="TestOS-1.0", python_version="3.10.0", mindspore_version="2.7.1"),
        disk_read=DiskReadBenchmark(
            method="test",
            test_file_bytes=2**30,
            repeats=2,
            seed=0,
            results=[
                ReadSpeed.from_timings(
                    pattern="sequential",
                    chunk_bytes=EXPERT_BYTES_INT8,
                    reads_per_run=10,
                    seconds_per_run=[0.1, 0.2],
                    device_bytes_read_per_run=[10 * EXPERT_BYTES_INT8] * 2,
                ),
                ReadSpeed.from_timings(
                    pattern="random",
                    chunk_bytes=EXPERT_BYTES_FP16,
                    reads_per_run=10,
                    seconds_per_run=[0.4, 0.4],
                    device_bytes_read_per_run=[10 * EXPERT_BYTES_FP16] * 2,
                ),
            ],
        ),
    )
