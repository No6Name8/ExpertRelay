"""Measure this machine: RAM, CPU, the drive holding the repo, and real
(uncached) read speed from that drive at expert-sized chunks.

This is the single source of machine information in ExpertRelay. The
Manager will use it to decide what fits where; `expertrelay.benchmarking`
embeds it in every benchmark record. Nothing else should query RAM, CPU, or
disk facts on its own.

Two tiers, because the disk-speed test is expensive (it writes and reads a
1 GiB file) and most callers only need the cheap facts:

- `collect_machine_profile(measure_disk=False)`: RAM, CPU, drive identity and
  free space, software versions. Takes about a second (one PowerShell call).
- `collect_machine_profile(measure_disk=True)`: all of the above plus the
  measured read speeds.

Platform support: RAM, core counts, free space and software versions work
everywhere. CPU model, drive model/bus/media type, and the disk-speed test
are implemented for Windows only (the dev machine). Elsewhere the descriptive
fields are None ("unknown", never guessed) and the disk-speed test raises
NotImplementedError. See docs/limitations.md.

Run `python -m expertrelay.manager.profile` to measure and save a profile.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import re
import shutil
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import psutil

from expertrelay import REPO_ROOT
from expertrelay.benchmarking import append_benchmark_record, base_record, peak_process_rss_mb
from expertrelay.paths import BENCHMARK_RESULTS_DIR
from expertrelay.store import unbuffered_io

# One routed expert of Qwen1.5-MoE-A2.7B is three projections: gate and up
# are [moe_intermediate_size=1408, hidden_size=2048], down is the transpose.
# That exact byte count is what the Manager will move per expert, so we time
# reads of exactly that size (8.65 MB at int8, 17.30 MB at fp16, the "9MB /
# 17MB" figures before rounding) instead of round numbers.
QWEN_MOE_EXPERT_PARAMS = 3 * 1408 * 2048
EXPERT_BYTES_INT8 = QWEN_MOE_EXPERT_PARAMS * 1
EXPERT_BYTES_FP16 = QWEN_MOE_EXPERT_PARAMS * 2

# Unbuffered I/O alignment (see expertrelay.store.unbuffered_io). Both expert
# sizes above happen to be exact multiples of it (2112 and 4224 sectors), so
# no padding is needed.
IO_ALIGNMENT = unbuffered_io.ALIGNMENT

DEFAULT_TEST_FILE_BYTES = 1024 * 1024 * 1024  # 1 GiB
DEFAULT_REPEATS = 3
_WRITE_BLOCK_BYTES = 64 * 1024 * 1024
_FREE_SPACE_MARGIN_BYTES = 1024 * 1024 * 1024
_PROBE_FILE_PREFIX = ".disk_probe_"

DEFAULT_MARKDOWN_OUT = REPO_ROOT / "docs" / "machine-profile.md"


@dataclass(frozen=True)
class MemoryInfo:
    total_bytes: int
    available_bytes: int


@dataclass(frozen=True)
class CpuInfo:
    model: str | None
    physical_cores: int | None
    logical_cores: int | None


@dataclass(frozen=True)
class DriveInfo:
    path: str
    model: str | None
    bus_type: str | None  # e.g. "NVMe", "SATA", "USB"
    media_type: str | None  # e.g. "SSD", "HDD"
    total_bytes: int
    free_bytes: int


@dataclass(frozen=True)
class SoftwareInfo:
    platform: str
    python_version: str
    mindspore_version: str


@dataclass(frozen=True)
class ReadSpeed:
    """Timing for one (pattern, chunk size) combination, over several runs.

    MB/s is decimal (1 MB = 1e6 bytes), matching how drive vendors quote speed.

    `device_bytes_read_per_run` is how much the OS disk counters say was
    actually read from physical disks during each timed run. A run served
    from the file cache reads ~0 bytes from the device, so `cache_bypass_verified`
    (every run read at least `bytes_per_run` from the device) is direct
    evidence the MB/s figures are real disk reads. Background I/O from other
    processes can only raise the counter, so it can't make a genuinely
    uncached run fail this check.
    """

    pattern: str  # "sequential" or "random"
    chunk_bytes: int
    reads_per_run: int
    bytes_per_run: int
    seconds_per_run: list[float]
    mb_per_s_per_run: list[float]
    median_mb_per_s: float
    device_bytes_read_per_run: list[int]
    cache_bypass_verified: bool

    @classmethod
    def from_timings(
        cls,
        *,
        pattern: str,
        chunk_bytes: int,
        reads_per_run: int,
        seconds_per_run: list[float],
        device_bytes_read_per_run: list[int],
    ) -> ReadSpeed:
        bytes_per_run = chunk_bytes * reads_per_run
        mb_per_s = [bytes_per_run / 1e6 / s for s in seconds_per_run]
        return cls(
            pattern=pattern,
            chunk_bytes=chunk_bytes,
            reads_per_run=reads_per_run,
            bytes_per_run=bytes_per_run,
            seconds_per_run=seconds_per_run,
            mb_per_s_per_run=mb_per_s,
            median_mb_per_s=statistics.median(mb_per_s),
            device_bytes_read_per_run=device_bytes_read_per_run,
            cache_bypass_verified=all(dev >= bytes_per_run for dev in device_bytes_read_per_run),
        )


@dataclass(frozen=True)
class DiskReadBenchmark:
    method: str
    test_file_bytes: int
    repeats: int
    seed: int
    results: list[ReadSpeed]


@dataclass(frozen=True)
class MachineProfile:
    hostname: str
    memory: MemoryInfo
    cpu: CpuInfo
    drive: DriveInfo
    software: SoftwareInfo
    disk_read: DiskReadBenchmark | None

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Cheap facts
# --------------------------------------------------------------------------


def measure_memory() -> MemoryInfo:
    vm = psutil.virtual_memory()
    return MemoryInfo(total_bytes=vm.total, available_bytes=vm.available)


def measure_cpu() -> CpuInfo:
    return CpuInfo(
        model=_cpu_model_name(),
        physical_cores=psutil.cpu_count(logical=False),
        logical_cores=psutil.cpu_count(logical=True),
    )


def _cpu_model_name() -> str | None:
    # platform.processor() on Windows returns "Intel64 Family 6 Model 154
    # Stepping 3, GenuineIntel", not a model name. The registry has the
    # marketing name ("12th Gen Intel(R) Core(TM) i5-12450H").
    if sys.platform != "win32":
        return None
    import winreg

    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except OSError:
        return None


_WINDOWS_DRIVE_QUERY = (
    "$p = Get-Partition -DriveLetter {letter}; "
    "$d = Get-Disk -Number $p.DiskNumber; "
    "$pd = Get-PhysicalDisk | Where-Object DeviceId -eq ([string]$p.DiskNumber); "
    "[pscustomobject]@{{Model=$d.FriendlyName; BusType=[string]$d.BusType; "
    "MediaType=[string]$pd.MediaType}} | ConvertTo-Json"
)


def parse_windows_drive_json(text: str) -> tuple[str | None, str | None, str | None]:
    """(model, bus_type, media_type) from the PowerShell query's JSON output.

    Any field that's missing, empty, or "Unspecified" comes back as None.
    Storage Spaces reports "Unspecified" when it can't tell, and that is not
    a value we want to record as if it were real.
    """
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None, None, None
    if not isinstance(data, dict):
        return None, None, None

    def clean(key: str) -> str | None:
        value = data.get(key)
        if value is None:
            return None
        value = str(value).strip()
        return None if value in ("", "Unspecified") else value

    return clean("Model"), clean("BusType"), clean("MediaType")


def measure_drive(path: Path = REPO_ROOT) -> DriveInfo:
    usage = shutil.disk_usage(path)
    model = bus_type = media_type = None
    if sys.platform == "win32":
        letter = Path(path).resolve().drive.rstrip(":")
        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    _WINDOWS_DRIVE_QUERY.format(letter=letter),
                ],
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
            )
            model, bus_type, media_type = parse_windows_drive_json(result.stdout)
        except (OSError, subprocess.SubprocessError):
            pass  # leave as None: unknown, not guessed
    return DriveInfo(
        path=str(Path(path).resolve()),
        model=model,
        bus_type=bus_type,
        media_type=media_type,
        total_bytes=usage.total,
        free_bytes=usage.free,
    )


def measure_software() -> SoftwareInfo:
    return SoftwareInfo(
        platform=platform.platform(),
        python_version=platform.python_version(),
        # Read from package metadata, not `import mindspore`: importing the
        # framework just to get its version costs hundreds of MB of RAM,
        # which long-running callers (the store build) can't spare.
        mindspore_version=importlib.metadata.version("mindspore"),
    )


# --------------------------------------------------------------------------
# Disk read speed
# --------------------------------------------------------------------------
#
# Cache-avoidance method: unbuffered I/O (Win32 CreateFileW with
# FILE_FLAG_NO_BUFFERING), for both writing and reading the test file.
#
# Why not "a test file larger than RAM": on an 8 GB machine that means writing
# 9+ GB for every profile run. That's slow, and it spends SSD write endurance
# on a measurement. It also only avoids the cache statistically: Windows can
# still serve the tail of the file from the standby list.
#
# Why this is sufficient: with NO_BUFFERING the OS cache manager is bypassed
# entirely. The file is written with NO_BUFFERING + WRITE_THROUGH, so its
# pages never enter the Windows file cache, and every read is a real device
# read. The price is that sizes, offsets and buffer addresses must be
# sector-aligned; the primitives live in expertrelay.store.unbuffered_io.
#
# What this does NOT bypass: the SSD's own internal caching. A file that was
# just written probably still sits in the drive's SLC write cache, which can
# read faster than data that has aged into TLC/QLC NAND. Expert weights on a
# real deployment are usually old data, so these numbers are an upper bound
# on cold-expert load speed. Filling the SLC cache (often tens of GB) to get
# around this isn't practical here. Recorded in docs/limitations.md.


def sequential_offsets(file_bytes: int, chunk_bytes: int) -> list[int]:
    """Back-to-back, non-overlapping chunk offsets covering as much of the file as fits."""
    return list(range(0, file_bytes - chunk_bytes + 1, chunk_bytes))


def random_aligned_offsets(
    file_bytes: int, chunk_bytes: int, count: int, seed: int, alignment: int = IO_ALIGNMENT
) -> list[int]:
    """`count` seeded, aligned offsets where a full chunk fits inside the file."""
    highest_slot = (file_bytes - chunk_bytes) // alignment
    rng = np.random.default_rng(seed)
    return [int(slot) * alignment for slot in rng.integers(0, highest_slot + 1, size=count)]


def validate_disk_test_params(test_file_bytes: int, chunk_sizes: tuple[int, ...]) -> None:
    """Reject parameters that unbuffered I/O would fail on, before touching the disk."""
    if test_file_bytes % IO_ALIGNMENT:
        raise ValueError(f"test file size {test_file_bytes} is not a multiple of {IO_ALIGNMENT}")
    for chunk in chunk_sizes:
        if chunk % IO_ALIGNMENT:
            raise ValueError(f"chunk size {chunk} is not a multiple of {IO_ALIGNMENT}")
        if chunk > test_file_bytes:
            raise ValueError(f"chunk size {chunk} exceeds test file size {test_file_bytes}")


def _write_test_file(path: Path, size: int, seed: int) -> None:
    """Fill `path` with `size` bytes of incompressible data, bypassing the cache.

    Random bytes, not zeros: some SSD controllers compress or deduplicate, and
    a file of zeros would read back suspiciously fast.
    """
    rng = np.random.default_rng(seed)
    with (
        unbuffered_io.unbuffered_handle(path, write=True) as handle,
        unbuffered_io.aligned_buffer(_WRITE_BLOCK_BYTES) as (buf, addr),
    ):
        remaining = size
        while remaining:
            n = min(_WRITE_BLOCK_BYTES, remaining)
            buf[:n] = rng.bytes(n)
            unbuffered_io.write_sequential(handle, addr, n)
            remaining -= n


def _device_read_bytes() -> int:
    """Total bytes read from all physical disks since boot, per the OS counters."""
    return sum(c.read_bytes for c in psutil.disk_io_counters(perdisk=True).values())


def _time_reads(path: Path, chunk_bytes: int, offsets: list[int]) -> tuple[float, int]:
    """Read one chunk at each offset. Returns (seconds, device bytes read in that
    window). Open/close are outside the timed window."""
    with (
        unbuffered_io.unbuffered_handle(path, write=False) as handle,
        unbuffered_io.aligned_buffer(chunk_bytes) as (_buf, addr),
    ):
        device_before = _device_read_bytes()
        start = time.perf_counter()
        for offset in offsets:
            unbuffered_io.read_at(handle, addr, offset, chunk_bytes)
        seconds = time.perf_counter() - start
        return seconds, _device_read_bytes() - device_before


def _measure_pattern(
    path: Path, *, pattern: str, chunk_bytes: int, offsets_per_run: list[list[int]]
) -> ReadSpeed:
    timings = [_time_reads(path, chunk_bytes, offsets) for offsets in offsets_per_run]
    return ReadSpeed.from_timings(
        pattern=pattern,
        chunk_bytes=chunk_bytes,
        reads_per_run=len(offsets_per_run[0]),
        seconds_per_run=[seconds for seconds, _ in timings],
        device_bytes_read_per_run=[device for _, device in timings],
    )


def measure_disk_read(
    directory: Path = REPO_ROOT,
    *,
    test_file_bytes: int = DEFAULT_TEST_FILE_BYTES,
    chunk_sizes: tuple[int, ...] = (EXPERT_BYTES_INT8, EXPERT_BYTES_FP16),
    repeats: int = DEFAULT_REPEATS,
    seed: int = 0,
) -> DiskReadBenchmark:
    """Write a temporary test file in `directory`, time uncached sequential and
    random reads at each chunk size, then delete the file (even on failure).

    Random runs read as many chunks as the sequential run does, so both
    patterns move the same number of bytes and their MB/s are comparable.
    """
    if sys.platform != "win32":
        raise NotImplementedError("uncached disk read measurement is implemented for Windows only")
    validate_disk_test_params(test_file_bytes, chunk_sizes)
    free = shutil.disk_usage(directory).free
    if free < test_file_bytes + _FREE_SPACE_MARGIN_BYTES:
        raise OSError(
            f"only {free / 1e9:.1f} GB free in {directory}, need {test_file_bytes / 1e9:.1f} GB + margin"
        )

    path = Path(directory) / f"{_PROBE_FILE_PREFIX}{socket.gethostname()}.bin"
    results: list[ReadSpeed] = []
    try:
        _write_test_file(path, test_file_bytes, seed)
        for chunk in chunk_sizes:
            seq = sequential_offsets(test_file_bytes, chunk)
            results.append(
                _measure_pattern(
                    path, pattern="sequential", chunk_bytes=chunk, offsets_per_run=[seq] * repeats
                )
            )
            rand = [
                random_aligned_offsets(test_file_bytes, chunk, len(seq), seed + run) for run in range(repeats)
            ]
            results.append(_measure_pattern(path, pattern="random", chunk_bytes=chunk, offsets_per_run=rand))
    finally:
        path.unlink(missing_ok=True)

    return DiskReadBenchmark(
        method="win32 FILE_FLAG_NO_BUFFERING (write + read), page-aligned buffers",
        test_file_bytes=test_file_bytes,
        repeats=repeats,
        seed=seed,
        results=results,
    )


# --------------------------------------------------------------------------
# Assembly, formatting, CLI
# --------------------------------------------------------------------------


def collect_machine_profile(
    *,
    measure_disk: bool = False,
    test_file_bytes: int = DEFAULT_TEST_FILE_BYTES,
    repeats: int = DEFAULT_REPEATS,
    seed: int = 0,
) -> MachineProfile:
    return MachineProfile(
        hostname=socket.gethostname(),
        memory=measure_memory(),
        cpu=measure_cpu(),
        drive=measure_drive(REPO_ROOT),
        software=measure_software(),
        disk_read=(
            measure_disk_read(REPO_ROOT, test_file_bytes=test_file_bytes, repeats=repeats, seed=seed)
            if measure_disk
            else None
        ),
    )


def output_path_for(hostname: str, results_dir: Path = BENCHMARK_RESULTS_DIR) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", hostname)
    return results_dir / f"machine_profile_{safe}.json"


def _gb(n: int) -> str:
    return f"{n / 1e9:.2f} GB"


def _or_unknown(value: object) -> str:
    return "unknown" if value is None else str(value)


def format_summary(profile: MachineProfile) -> str:
    """Plain-text summary for the terminal."""
    m, c, d = profile.memory, profile.cpu, profile.drive
    lines = [
        f"Host:   {profile.hostname}",
        f"RAM:    {_gb(m.total_bytes)} total, {_gb(m.available_bytes)} available",
        f"CPU:    {_or_unknown(c.model)} ({_or_unknown(c.physical_cores)} cores / "
        f"{_or_unknown(c.logical_cores)} threads)",
        f"Drive:  {_or_unknown(d.model)} [{_or_unknown(d.bus_type)} {_or_unknown(d.media_type)}], "
        f"{_gb(d.free_bytes)} free of {_gb(d.total_bytes)}",
    ]
    if profile.disk_read:
        lines.append(f"Disk reads ({profile.disk_read.method}):")
        for r in profile.disk_read.results:
            lines.append(
                f"  {r.pattern:<10} {r.chunk_bytes / 1e6:6.2f} MB chunks: "
                f"{r.median_mb_per_s:8.1f} MB/s median of {len(r.mb_per_s_per_run)} runs, "
                f"cache bypass {'verified' if r.cache_bypass_verified else 'NOT VERIFIED'}"
            )
    return "\n".join(lines)


def format_markdown(profile: MachineProfile, *, timestamp: str, json_path: Path) -> str:
    """Readable copy for docs/. Generated from the same object that was saved
    as JSON, never edited by hand."""
    m, c, d, s = profile.memory, profile.cpu, profile.drive, profile.software
    try:
        source = json_path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        source = str(json_path)
    out = [
        f"# Machine profile: {profile.hostname}",
        "",
        f"Generated by `python -m expertrelay.manager.profile` at {timestamp}. Raw data: `{source}`.",
        "Do not edit by hand; re-run the command instead.",
        "",
        "| | |",
        "|---|---|",
        f"| RAM total | {_gb(m.total_bytes)} |",
        f"| RAM available at measurement time | {_gb(m.available_bytes)} |",
        f"| CPU | {_or_unknown(c.model)} |",
        f"| Cores / threads | {_or_unknown(c.physical_cores)} / {_or_unknown(c.logical_cores)} |",
        f"| Drive (holds the repo) | {_or_unknown(d.model)} |",
        f"| Drive type | {_or_unknown(d.bus_type)} {_or_unknown(d.media_type)} |",
        f"| Drive free / total | {_gb(d.free_bytes)} / {_gb(d.total_bytes)} |",
        f"| OS | {s.platform} |",
        f"| Python / MindSpore | {s.python_version} / {s.mindspore_version} |",
    ]
    if profile.disk_read:
        r0 = profile.disk_read
        out += [
            "",
            "## Uncached read speed",
            "",
            f"Method: {r0.method}. Test file {r0.test_file_bytes / 2**30:.0f} GiB, "
            f"{r0.repeats} runs per row, seed {r0.seed}. MB = 10^6 bytes.",
            f"Chunk sizes are one Qwen1.5-MoE-A2.7B routed expert: {EXPERT_BYTES_INT8:,} bytes at int8, "
            f"{EXPERT_BYTES_FP16:,} bytes at fp16.",
            "",
            "| Pattern | Chunk | Median MB/s | Per-run MB/s | ms per chunk (median) | Cache bypass |",
            "|---|---|---|---|---|---|",
        ]
        for r in r0.results:
            per_run = ", ".join(f"{v:.0f}" for v in r.mb_per_s_per_run)
            ms_per_chunk = r.chunk_bytes / (r.median_mb_per_s * 1e6) * 1e3
            verified = "verified" if r.cache_bypass_verified else "**NOT VERIFIED**"
            out.append(
                f"| {r.pattern} | {r.chunk_bytes / 1e6:.2f} MB | {r.median_mb_per_s:.0f} | {per_run} | "
                f"{ms_per_chunk:.2f} | {verified} |"
            )
        out += [
            "",
            "Cache bypass is checked, not assumed: for every timed run, the OS physical-disk read counters",
            "must show at least as many bytes read from the device as the run requested. A run served from",
            "the Windows file cache reads ~0 bytes from the device and would fail this check.",
            "",
            "Caveat: this bypasses the Windows file cache but not the SSD's own SLC cache. The test file",
            "was freshly written, so these are an upper bound for loading cold experts that have sat on",
            "disk for a while. See docs/limitations.md.",
        ]
    return "\n".join(out) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-disk", action="store_true", help="skip the (slow, 1 GiB write) disk read test")
    ap.add_argument("--test-file-mib", type=int, default=DEFAULT_TEST_FILE_BYTES // 2**20)
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    ap.add_argument("--seed", type=int, default=0, help="seeds the test file contents and random offsets")
    ap.add_argument(
        "--out", type=Path, default=None, help="default: benchmarks/results/machine_profile_<host>.json"
    )
    ap.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN_OUT)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    test_file_bytes = args.test_file_mib * 2**20

    profile = collect_machine_profile(
        measure_disk=not args.no_disk, test_file_bytes=test_file_bytes, repeats=args.repeats, seed=args.seed
    )
    record = base_record(
        label="machine profile",
        seed=args.seed,
        model=None,
        config={
            "measure_disk": not args.no_disk,
            "test_file_bytes": test_file_bytes,
            "repeats": args.repeats,
            "chunk_sizes": [EXPERT_BYTES_INT8, EXPERT_BYTES_FP16],
        },
        machine=profile,
        peak_rss_mb=peak_process_rss_mb(),
    )
    out_path = args.out or output_path_for(profile.hostname)
    append_benchmark_record(out_path, record)

    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text(format_markdown(profile, timestamp=record["timestamp"], json_path=out_path))

    print(format_summary(profile))
    print(f"\nSaved {out_path}")
    print(f"Saved {args.markdown_out}")


if __name__ == "__main__":
    main()
