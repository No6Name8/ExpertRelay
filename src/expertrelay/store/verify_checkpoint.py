"""Check a downloaded checkpoint against the Hub's own hashes for its revision.

    python -m expertrelay.store.verify_checkpoint --repo-id Qwen/Qwen1.5-MoE-A2.7B-Chat --revision <commit>

The Hub lists every file of a revision with an id: for large (LFS) files
the sha256 of the content, for small files the git blob id (sha1 of
"blob <size>\\0" + content). Both are recomputed from the local files, so
every byte of every file is checked. Appends the result (file, size,
sha256, match) to benchmarks/results/checkpoint_verification.json, which
is also the record of exactly which bytes later results were computed from.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import urllib.request
from pathlib import Path

from expertrelay.benchmarking import append_benchmark_record, base_record
from expertrelay.manager.profile import collect_machine_profile
from expertrelay.paths import BENCHMARK_RESULTS_DIR
from expertrelay.store.fetch_hf_tensors import _auth_headers

RESULTS = BENCHMARK_RESULTS_DIR / "checkpoint_verification.json"


def hub_files(repo_id: str, revision: str) -> dict[str, dict]:
    url = f"https://huggingface.co/api/models/{repo_id}/tree/{revision}"
    req = urllib.request.Request(url, headers=_auth_headers())
    return {t["path"]: t for t in json.load(urllib.request.urlopen(req, timeout=60)) if t["type"] == "file"}


def file_hashes(path: Path) -> tuple[str, str]:
    """(sha256, git blob sha1) of a file, in one pass."""
    size = path.stat().st_size
    sha256, blob = hashlib.sha256(), hashlib.sha1(f"blob {size}\0".encode())
    with open(path, "rb") as f:
        while chunk := f.read(64 << 20):
            sha256.update(chunk)
            blob.update(chunk)
    return sha256.hexdigest(), blob.hexdigest()


def verify(repo_id: str, revision: str, directory: Path, patterns: list[str]) -> dict:
    remote = {
        p: t for p, t in hub_files(repo_id, revision).items() if any(fnmatch.fnmatch(p, x) for x in patterns)
    }
    files, bad, missing = [], [], []
    for name, t in sorted(remote.items()):
        local = Path(directory) / name
        if not local.exists():
            missing.append(name)
            continue
        sha256, blob = file_hashes(local)
        lfs = t.get("lfs")
        ok = (sha256 == lfs["oid"]) if lfs else (blob == t["oid"])
        ok = ok and local.stat().st_size == t["size"]
        files.append({"file": name, "bytes": local.stat().st_size, "sha256": sha256, "hub_id_matches": ok})
        if not ok:
            bad.append(name)
    return {"repo_id": repo_id, "revision": revision, "files": files, "bad": bad, "missing": missing}


def main() -> None:
    from expertrelay.store.download_checkpoint import ALLOW_PATTERNS, checkpoint_dir

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--revision", required=True, help="a commit hash")
    args = ap.parse_args()
    report = verify(args.repo_id, args.revision, checkpoint_dir(args.repo_id, args.revision), ALLOW_PATTERNS)
    record = base_record(
        label=f"checkpoint verification {args.repo_id}",
        seed=0,
        model={"repo_id": args.repo_id, "revision": args.revision},
        config={"patterns": ALLOW_PATTERNS},
        machine=collect_machine_profile(measure_disk=False),
    )
    record.update(report)
    append_benchmark_record(RESULTS, record)
    print(json.dumps({k: report[k] for k in ("bad", "missing")}), f"{len(report['files'])} files checked")
    if report["bad"] or report["missing"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
