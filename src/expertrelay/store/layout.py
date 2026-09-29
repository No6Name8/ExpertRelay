"""On-disk format of the expert store.

A store directory holds:
  store.json            manifest: pinned source revision, model config,
                        quantization method, both layouts, build summary
  experts.bin           every routed expert, one fixed-size record each
  experts_index.json    per expert: layer, expert id, offset, size, shapes,
                        sha256, quantization error
  resident.bin          everything used on every token (attention, shared
                        expert, router, norms, embeddings, lm_head)
  resident_index.json   per tensor: name, kind, shape, offsets, sha256, error

Expert record (identical layout for every expert):
  [gate_proj int8][up_proj int8][down_proj int8][gate scales f32][up scales f32][down scales f32][zero pad]
The scales sit right after the weights, and the record is padded to a
multiple of 4096 bytes. Records are back to back, so every record starts on
a 4096-byte boundary. One unbuffered read of `record_size` bytes at
`offset` therefore loads one complete expert, weights and scales together.

An int4 store (store.int4) has the same files and rules; only the
expert record differs (packed 4-bit weights, float16 group scales), and
its index says so in layout["format"].

Resident tensors are laid out the same way (int8 weights, then f32 scales,
then pad), each starting on a 4096-byte boundary. Tensors kept in fp32 are
[f32 data][pad].
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from expertrelay.store.int4 import FORMAT as INT4_FORMAT
from expertrelay.store.int4 import Int4RecordLayout, parse_int4_record
from expertrelay.store.unbuffered_io import ALIGNMENT, align_up, is_aligned

INDEX_FORMAT_VERSION = 1
EXPERT_MATRICES = ("gate_proj", "up_proj", "down_proj")
_EXPERT_TENSOR_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)

KIND_INT8 = "int8_rowwise"
KIND_FP32 = "fp32"


@dataclass(frozen=True)
class ExpertRecordLayout:
    """Byte layout of one expert record. `matrices` is ordered: that order is the on-disk order."""

    matrices: tuple[tuple[str, tuple[int, int]], ...]

    @classmethod
    def from_config(cls, config: dict) -> ExpertRecordLayout:
        inter, hidden = config["moe_intermediate_size"], config["hidden_size"]
        return cls(
            matrices=(
                ("gate_proj", (inter, hidden)),
                ("up_proj", (inter, hidden)),
                ("down_proj", (hidden, inter)),
            )
        )

    @property
    def shapes(self) -> dict[str, tuple[int, int]]:
        return dict(self.matrices)

    def weight_offset(self, name: str) -> int:
        offset = 0
        for n, (rows, cols) in self.matrices:
            if n == name:
                return offset
            offset += rows * cols
        raise KeyError(name)

    @property
    def weights_bytes(self) -> int:
        return sum(rows * cols for _, (rows, cols) in self.matrices)

    def scales_offset(self, name: str) -> int:
        offset = self.weights_bytes
        for n, (rows, _cols) in self.matrices:
            if n == name:
                return offset
            offset += rows * 4
        raise KeyError(name)

    @property
    def scales_bytes(self) -> int:
        return sum(rows * 4 for _, (rows, _cols) in self.matrices)

    @property
    def record_size(self) -> int:
        return align_up(self.weights_bytes + self.scales_bytes)

    def to_dict(self) -> dict:
        return {
            "matrices": [{"name": n, "shape": list(s)} for n, s in self.matrices],
            "weights_bytes": self.weights_bytes,
            "scales_bytes": self.scales_bytes,
            "scales_dtype": "float32",
            "record_size": self.record_size,
            "alignment": ALIGNMENT,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ExpertRecordLayout:
        return cls(matrices=tuple((m["name"], tuple(m["shape"])) for m in d["matrices"]))


def serialize_expert_record(
    layout: ExpertRecordLayout, quantized: dict[str, tuple[np.ndarray, np.ndarray]]
) -> bytearray:
    """Pack {matrix name: (int8 weights, f32 scales)} into one padded record."""
    record = bytearray(layout.record_size)
    for name, shape in layout.matrices:
        q, scales = quantized[name]
        if q.shape != shape or q.dtype != np.int8 or scales.shape != (shape[0],):
            raise ValueError(f"{name}: got {q.dtype}{q.shape} / scales{scales.shape}, expected int8{shape}")
        w_off, s_off = layout.weight_offset(name), layout.scales_offset(name)
        record[w_off : w_off + q.nbytes] = q.tobytes()
        record[s_off : s_off + shape[0] * 4] = scales.astype("<f4").tobytes()
    return record


def parse_expert_record(
    layout: ExpertRecordLayout | Int4RecordLayout, buf
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Inverse of serialize_expert_record. Returns views into `buf`: copy them
    if `buf` is about to be reused. For an int4 layout: (packed uint8,
    float16 group scales) per matrix instead of (int8, float32 row scales)."""
    if isinstance(layout, Int4RecordLayout):
        return parse_int4_record(layout, buf)
    out = {}
    for name, (rows, cols) in layout.matrices:
        q = np.frombuffer(buf, dtype=np.int8, count=rows * cols, offset=layout.weight_offset(name))
        scales = np.frombuffer(buf, dtype="<f4", count=rows, offset=layout.scales_offset(name))
        out[name] = (q.reshape(rows, cols), scales)
    return out


def expert_slot(layer: int, expert: int, num_experts: int) -> int:
    return layer * num_experts + expert


def expert_offset(layer: int, expert: int, num_experts: int, record_size: int) -> int:
    return expert_slot(layer, expert, num_experts) * record_size


def parse_expert_tensor_name(name: str) -> tuple[int, int, str] | None:
    """(layer, expert, matrix) for a routed-expert weight name, else None."""
    m = _EXPERT_TENSOR_RE.match(name)
    return (int(m.group(1)), int(m.group(2)), m.group(3)) if m else None


def expert_tensor_name(layer: int, expert: int, matrix: str) -> str:
    return f"model.layers.{layer}.mlp.experts.{expert}.{matrix}.weight"


@dataclass(frozen=True)
class ExpertIndexEntry:
    layer: int
    expert: int
    offset: int
    size: int
    shapes: dict[str, list[int]]
    sha256: str
    quant_error: dict[str, dict[str, float]]  # per matrix: rel_fro_error, max_abs_error_over_scale


def write_expert_index(
    path: Path, layout: ExpertRecordLayout | Int4RecordLayout, entries: list[ExpertIndexEntry]
) -> None:
    doc = {
        "format_version": INDEX_FORMAT_VERSION,
        "layout": layout.to_dict(),
        "entries": [asdict(e) for e in sorted(entries, key=lambda e: e.offset)],
    }
    Path(path).write_text(json.dumps(doc, indent=1))


def layout_from_dict(d: dict) -> ExpertRecordLayout | Int4RecordLayout:
    return (
        Int4RecordLayout.from_dict(d) if d.get("format") == INT4_FORMAT else ExpertRecordLayout.from_dict(d)
    )


def read_expert_index(path: Path) -> tuple[ExpertRecordLayout | Int4RecordLayout, list[ExpertIndexEntry]]:
    """Load and validate an expert index. Validation is strict: an unaligned
    or wrong-sized entry would make the one-read-per-expert guarantee false."""
    doc = json.loads(Path(path).read_text())
    if doc.get("format_version") != INDEX_FORMAT_VERSION:
        raise ValueError(f"unsupported expert index format_version {doc.get('format_version')}")
    layout = layout_from_dict(doc["layout"])
    entries = [ExpertIndexEntry(**e) for e in doc["entries"]]
    seen: set[tuple[int, int]] = set()
    for e in entries:
        if not is_aligned(e.offset) or e.size != layout.record_size:
            raise ValueError(
                f"expert ({e.layer},{e.expert}): offset {e.offset} / size {e.size} breaks the layout"
            )
        if (e.layer, e.expert) in seen:
            raise ValueError(f"duplicate expert ({e.layer},{e.expert}) in index")
        seen.add((e.layer, e.expert))
    return layout, entries


def resident_kind(name: str, shape: tuple[int, ...]) -> str:
    """Which resident tensors become int8 and which stay fp32.

    int8: every 2-D weight matrix (attention q/k/v/o, shared expert, embedding,
    lm_head). fp32: norms and biases, as specified. Also fp32: the router
    (mlp.gate, 60 x 2048) and shared_expert_gate (1 x 2048). Together they're
    ~12 MB. Keeping them exact means the router adds no quantization error
    of its own to routing decisions. Selection can still differ from bf16,
    because the router's input comes out of the int8 layers before it.
    """
    if len(shape) != 2:
        return KIND_FP32
    if name.endswith(".mlp.gate.weight") or name.endswith(".mlp.shared_expert_gate.weight"):
        return KIND_FP32
    return KIND_INT8


@dataclass(frozen=True)
class ResidentEntry:
    name: str
    kind: str
    shape: list[int]
    offset: int
    weight_bytes: int
    scales_offset: int | None
    scales_bytes: int
    region_size: int
    sha256: str | None = None
    quant_error: dict[str, float] | None = None


def plan_resident_layout(tensors: list[tuple[str, tuple[int, ...]]]) -> list[ResidentEntry]:
    """Assign each resident tensor an aligned region, in the given order."""
    entries, offset = [], 0
    for name, shape in tensors:
        kind = resident_kind(name, shape)
        numel = int(np.prod(shape)) if shape else 1
        if kind == KIND_INT8:
            weight_bytes, scales_bytes = numel, shape[0] * 4
            scales_offset = offset + weight_bytes
        else:
            weight_bytes, scales_bytes, scales_offset = numel * 4, 0, None
        region = align_up(weight_bytes + scales_bytes)
        entries.append(
            ResidentEntry(
                name=name,
                kind=kind,
                shape=list(shape),
                offset=offset,
                weight_bytes=weight_bytes,
                scales_offset=scales_offset,
                scales_bytes=scales_bytes,
                region_size=region,
            )
        )
        offset += region
    return entries


def write_resident_index(path: Path, entries: list[ResidentEntry]) -> None:
    doc = {
        "format_version": INDEX_FORMAT_VERSION,
        "alignment": ALIGNMENT,
        "entries": [asdict(e) for e in entries],
    }
    Path(path).write_text(json.dumps(doc, indent=1))


def read_resident_index(path: Path) -> list[ResidentEntry]:
    doc = json.loads(Path(path).read_text())
    if doc.get("format_version") != INDEX_FORMAT_VERSION:
        raise ValueError(f"unsupported resident index format_version {doc.get('format_version')}")
    entries = [ResidentEntry(**e) for e in doc["entries"]]
    for e in entries:
        if not is_aligned(e.offset) or not is_aligned(e.region_size):
            raise ValueError(
                f"resident tensor {e.name}: offset {e.offset} / region {e.region_size} unaligned"
            )
    return entries


def sha256_file_region(path: Path, offset: int, size: int, chunk: int = 8 * 2**20) -> str:
    """sha256 of bytes [offset, offset + size) of a file, streamed."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(offset)
        remaining = size
        while remaining:
            data = f.read(min(chunk, remaining))
            if not data:
                raise OSError(f"{path}: hit end of file with {remaining} bytes of region unread")
            h.update(data)
            remaining -= len(data)
    return h.hexdigest()
