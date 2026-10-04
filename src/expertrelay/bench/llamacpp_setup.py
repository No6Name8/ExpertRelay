"""Step C baseline: llama.cpp, pinned, and the Qwen1.5-MoE-A2.7B-Chat GGUFs
converted from the verified bf16 weights. Resumable.

    python -m expertrelay.bench.llamacpp_setup

No official Qwen GGUF of this model exists on the Hub (checked
2026-10-04: no Qwen/Qwen1.5-MoE-A2.7B-Chat-GGUF; third-party uploads
only), so the GGUFs are made here with llama.cpp's own tools, from the
bf16 Chat checkpoint whose every file was checked against the Hub's hashes
(benchmarks/results/checkpoint_verification.json).

llama.cpp is pinned to build b11146 (commit 7fe450e1), the build the
stable release v0.5.0 points to. The official prebuilt Windows CPU x64
zip is downloaded and checked against the sha256 GitHub publishes for it;
the converter comes from the source archive of the same tag.

Steps (each skipped when its output exists and matches the sha256 this
script recorded for it in models/gguf/manifest.json):
  1. download + verify the binary zip, extract
  2. download the source archive of the same tag, extract
  3. bf16 GGUF:  convert_hf_to_gguf.py <bf16 Chat> --outtype bf16
  4. Q4_K_M:     llama-quantize <bf16 GGUF> Q4_K_M
  5. delete the bf16 GGUF (28.6 GB; only step 4 needs it)
  6. Q8_0:       convert_hf_to_gguf.py <bf16 Chat> --outtype q8_0
Q8_0 is made by the converter straight from the bf16 weights rather than
by llama-quantize from the bf16 GGUF: keeping the bf16 GGUF, Q4_K_M and
Q8_0 on disk together needs ~53 GB, more than the 51 GB free. Both tools
implement the same Q8_0 (round to nearest, one scale = max|w| / 127 per 32
weights); this is recorded in docs/limitations.md.

Writes the exact commands, versions, sizes and sha256 of every file to
benchmarks/results/llamacpp_setup.json.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

from expertrelay import REPO_ROOT
from expertrelay.benchmarking import append_benchmark_record, base_record
from expertrelay.manager.profile import collect_machine_profile
from expertrelay.paths import BENCHMARK_RESULTS_DIR, MODELS_ROOT
from expertrelay.store.download_checkpoint import checkpoint_dir

BUILD = "b11146"
COMMIT = "7fe450e19305b828c199d602c23a8337aaa1f03b"
STABLE_RELEASE = "v0.5.0"
BIN_ZIP = f"llama-{BUILD}-bin-win-cpu-x64.zip"
BIN_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{BUILD}/{BIN_ZIP}"
BIN_SHA256 = "14cf1303ca9ac3abd94816850532f9f9a69ac66fbaca3776fc6f9061c2fac1d1"  # GitHub's published digest
SRC_URL = f"https://github.com/ggml-org/llama.cpp/archive/{COMMIT}.tar.gz"
TOOLS = MODELS_ROOT / "tools" / f"llama.cpp-{BUILD}"
BIN_DIR = TOOLS / "bin"
SRC_DIR = TOOLS / f"llama.cpp-{COMMIT}"
GGUF_DIR = MODELS_ROOT / "gguf"
MANIFEST = GGUF_DIR / "manifest.json"
CHAT_REPO = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
CHAT_REVISION = "ec052fda178e241c7c443468d2fa1db6618996be"
NAME = "qwen1.5-moe-a2.7b-chat"
GGUF = {
    "bf16": GGUF_DIR / f"{NAME}-bf16.gguf",
    "Q4_K_M": GGUF_DIR / f"{NAME}-Q4_K_M.gguf",
    "Q8_0": GGUF_DIR / f"{NAME}-Q8_0.gguf",
}
THREADS = 12  # = ExpertRelay's numba threads on the dev machine (12 logical cores)
RESULTS = BENCHMARK_RESULTS_DIR / "llamacpp_setup.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(64 << 20):
            h.update(chunk)
    return h.hexdigest()


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def _record(key: str, path: Path, command: list[str] | None, seconds: float) -> dict:
    entry = {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path), "command": command,
             "seconds": seconds}  # fmt: skip
    m = _manifest()
    m[key] = entry
    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=1))
    return entry


def _done(key: str, path: Path) -> bool:
    entry = _manifest().get(key)
    if entry is None or not path.exists() or path.stat().st_size != entry["bytes"]:
        return False
    print(f"{key}: verifying {path.name} ...", flush=True)
    return sha256_file(path) == entry["sha256"]


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.replace(dest)


def _portable(cmd: list[str]) -> list[str]:
    return [c.replace(str(REPO_ROOT) + "\\", "").replace(str(REPO_ROOT), ".") for c in cmd]


def _run(cmd: list[str]) -> float:
    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    print("$ " + " ".join(_portable(cmd)), flush=True)
    t = time.perf_counter()
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    return time.perf_counter() - t


def binaries() -> dict:
    zip_path = TOOLS / BIN_ZIP
    if not (zip_path.exists() and sha256_file(zip_path) == BIN_SHA256):
        _download(BIN_URL, zip_path)
    got = sha256_file(zip_path)
    if got != BIN_SHA256:
        raise SystemExit(f"{BIN_ZIP}: sha256 {got} != published {BIN_SHA256}")
    if not (BIN_DIR / "llama-server.exe").exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(BIN_DIR)
    version = subprocess.run([str(BIN_DIR / "llama-cli.exe"), "--version"], capture_output=True, text=True)
    return {
        "zip": BIN_ZIP,
        "url": BIN_URL,
        "sha256": got,
        "version_output": (version.stdout + version.stderr).strip(),
    }


def source() -> dict:
    archive = TOOLS / f"llama.cpp-{COMMIT}.tar.gz"
    if not archive.exists():
        _download(SRC_URL, archive)
    if not (SRC_DIR / "convert_hf_to_gguf.py").exists():
        with tarfile.open(archive) as t:
            # no extractall(filter=...) on this Python 3.10: check paths by hand
            root = TOOLS.resolve()
            # only what the converter uses: the top-level scripts, conversion, gguf-py and
            # requirements (the whole tree has paths past Windows' 260-char limit)
            top = f"llama.cpp-{COMMIT}/"
            members = [
                m
                for m in t.getmembers()
                if (m.isfile() or m.isdir())
                and m.name.startswith(top)
                and (
                    "/" not in m.name[len(top) :]
                    or m.name[len(top) :].split("/")[0] in ("gguf-py", "requirements", "conversion")
                )
            ]
            for m in members:
                if root not in (root / m.name).resolve().parents and (root / m.name).resolve() != root:
                    raise SystemExit(f"unsafe path in {archive.name}: {m.name}")
            t.extractall(TOOLS, members=members)
    return {"url": SRC_URL, "commit": COMMIT, "archive_sha256": sha256_file(archive)}


def convert(outtype: str, key: str) -> dict:
    out = GGUF[key]
    if _done(key, out):
        return _manifest()[key]
    cmd = [sys.executable, str(SRC_DIR / "convert_hf_to_gguf.py"), str(checkpoint_dir(CHAT_REPO, CHAT_REVISION)),
           "--outfile", str(out), "--outtype", outtype]  # fmt: skip
    seconds = _run(cmd)
    return _record(key, out, _portable(cmd), seconds)


def quantize(key: str) -> dict:
    out = GGUF[key]
    if _done(key, out):
        return _manifest()[key]
    cmd = [str(BIN_DIR / "llama-quantize.exe"), str(GGUF["bf16"]), str(out), key, str(THREADS)]
    seconds = _run(cmd)
    return _record(key, out, _portable(cmd), seconds)


def main() -> None:
    record = base_record(
        label="llama.cpp baseline setup: pinned build, GGUFs converted from bf16 Chat",
        seed=0,
        model={"repo_id": CHAT_REPO, "revision": CHAT_REVISION},
        config={"build": BUILD, "commit": COMMIT, "stable_release": STABLE_RELEASE, "threads": THREADS},
        machine=collect_machine_profile(measure_disk=False),
    )
    record["binaries"] = binaries()
    record["source"] = source()
    files = {}
    if not _done("Q4_K_M", GGUF["Q4_K_M"]):
        files["bf16"] = convert("bf16", "bf16")
    files["Q4_K_M"] = quantize("Q4_K_M")
    if GGUF["bf16"].exists():
        GGUF["bf16"].unlink()  # 28.6 GB, only needed for Q4_K_M
    files["bf16"] = files.get("bf16") or _manifest().get("bf16")
    files["Q8_0"] = convert("q8_0", "Q8_0")
    record["files"] = files
    record["bf16_gguf_deleted_after_quantizing"] = True
    append_benchmark_record(RESULTS, record)
    print(json.dumps({k: (v["bytes"], v["sha256"][:12]) for k, v in files.items() if v}, indent=1))


if __name__ == "__main__":
    main()
