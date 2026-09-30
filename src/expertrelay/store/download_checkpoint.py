"""Download an unmodified Hugging Face checkpoint (safetensors shards + JSON
files) at a pinned commit, resumable, and verify every file.

Used for the reference checks and for building stores from original
weights (bf16, GPTQ releases).

    python -m expertrelay.store.download_checkpoint --repo-id Qwen/Qwen1.5-MoE-A2.7B --revision <commit> --retries 30

Files land in models/hf/<repo>@<commit>/. huggingface_hub resumes partial
files, and skips files already complete, so rerunning after an
interruption continues where it stopped. With --retries, a dropped
connection (this link drops long transfers, docs/limitations.md) is
retried after a pause instead of ending the run. HF_TOKEN from the
environment is used if set (huggingface_hub reads it).

After downloading, every file is checked against the Hub's own hash for
that revision (store.verify_checkpoint); a file that fails is deleted and
downloaded again, up to --retries times.

Run with HF_HUB_DISABLE_XET=1 on this machine. With the default Xet
transfer backend (hf_xet), the download stalled after ~730 MB: the network
kept receiving, but neither the partial files nor the Xet chunk cache grew
for 5+ minutes. Plain HTTP resumed immediately at the link's normal speed.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from huggingface_hub import snapshot_download

from expertrelay.paths import MODELS_ROOT
from expertrelay.store.fetch_hf_tensors import resolve_revision

ALLOW_PATTERNS = ["*.safetensors", "*.json"]
RETRY_PAUSE_S = 30


def checkpoint_dir(repo_id: str, revision: str) -> Path:
    return MODELS_ROOT / "hf" / f"{repo_id.replace('/', '--')}@{revision}"


def download(repo_id: str, revision: str, retries: int = 0) -> Path:
    from expertrelay.store.verify_checkpoint import verify

    sha = resolve_revision(repo_id, revision)
    target = checkpoint_dir(repo_id, sha)
    for attempt in range(retries + 1):
        print(f"downloading {repo_id}@{sha} -> {target} (attempt {attempt + 1})", flush=True)
        try:
            snapshot_download(
                repo_id=repo_id,
                revision=sha,
                local_dir=target,
                allow_patterns=ALLOW_PATTERNS,
                max_workers=2,
            )
        except Exception as e:  # a dropped connection: resume on the next attempt
            print(f"attempt {attempt + 1} failed: {e!r}", flush=True)
            time.sleep(RETRY_PAUSE_S)
            continue
        report = verify(repo_id, sha, target, ALLOW_PATTERNS)
        if not report["bad"] and not report["missing"]:
            print(f"done and verified: {target}", flush=True)
            return target
        for name in report["bad"]:  # corrupt: fetch it again
            (target / name).unlink()
        print(f"verification failed for {report['bad'] + report['missing']}; retrying", flush=True)
    raise SystemExit(f"{repo_id}@{sha}: not complete and verified after {retries + 1} attempts")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument(
        "--revision", default="main", help="branch/tag/commit; resolved to a commit hash before download"
    )
    ap.add_argument("--retries", type=int, default=0, help="extra attempts after a failure")
    args = ap.parse_args()
    download(args.repo_id, args.revision, args.retries)


if __name__ == "__main__":
    main()
