"""Where a file's bytes sit on the volume: its extents (contiguous runs).

An expert record that spans two extents is read by the OS as two I/Os,
even though the runtime issues one ReadFile, so the extent map is part of
explaining read times (bench/read_diagnosis.py).

Read with `fsutil file queryextents`, which works without admin rights.
Its output is in clusters; the cluster size is recovered from the file's
own size (the smallest power of two for which the clusters cover the file
with less than one cluster to spare), so no volume query is needed.
Windows only.
"""

from __future__ import annotations

import bisect
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

_LINE = re.compile(r"VCN:\s*(0x[0-9a-fA-F]+)\s+Clusters:\s*(0x[0-9a-fA-F]+)\s+LCN:\s*(0x[0-9a-fA-F]+|-1)")


@dataclass(frozen=True)
class Extent:
    offset: int  # bytes into the file
    length: int  # bytes


def parse_queryextents(text: str) -> list[tuple[int, int]]:
    """(first cluster in the file, cluster count) per extent, in file order."""
    return [(int(v, 16), int(c, 16)) for v, c, _ in _LINE.findall(text)]


def cluster_size_for(file_size: int, total_clusters: int) -> int:
    size = 512
    while size * total_clusters < file_size:
        size *= 2
    if size * total_clusters - file_size >= size:
        raise ValueError(
            f"{total_clusters} clusters can't hold exactly {file_size} bytes at any cluster size"
        )
    return size


def file_extents(path: Path) -> list[Extent]:
    out = subprocess.run(
        ["fsutil", "file", "queryextents", str(path)], capture_output=True, text=True, check=True
    ).stdout
    runs = parse_queryextents(out)
    cluster = cluster_size_for(Path(path).stat().st_size, sum(c for _, c in runs))
    return [Extent(v * cluster, c * cluster) for v, c in runs]


def pieces(extents: list[Extent], offset: int, length: int) -> int:
    """How many extents the byte range [offset, offset + length) touches."""
    starts = [e.offset for e in extents]
    first = bisect.bisect_right(starts, offset) - 1
    last = bisect.bisect_right(starts, offset + length - 1) - 1
    return last - first + 1
