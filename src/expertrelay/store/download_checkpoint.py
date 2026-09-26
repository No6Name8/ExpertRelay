"""Download an unmodified Hugging Face checkpoint (safetensors shards + JSON
files) at a pinned commit, resumable.

Used only for the reference check: the int8 store is lossy, so comparing
against the original model needs the original bf16 weights on disk. The
store build streams and discards them.

    python -m expertrelay.store.download_checkpoint --repo-id Qwen/Qwen1.5-MoE-A2.7B

Files land in models/hf/<repo>@<commit>/. huggingface_hub resumes partial
files; rerunning after an interruption continues where it stopped.

Run with HF_HUB_DISABLE_XET=1 on this machine. With the default Xet
transfer backend (hf_xet), the download stalled after ~730 MB: the network
kept receiving, but neither the partial files nor the Xet chunk cache grew
for 5+ minutes. Plain HTTP resumed immediately at the link's normal speed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from expertrelay.paths import MODELS_ROOT
from expertrelay.store.fetch_hf_tensors import resolve_revision


def checkpoint_dir(repo_id: str, revision: str) -> Path:
    return MODELS_ROOT / "hf" / f"{repo_id.replace('/', '--')}@{revision}"


def download(repo_id: str, revision: str) -> Path:
    sha = resolve_revision(repo_id, revision)
    target = checkpoint_dir(repo_id, sha)
    print(f"downloading {repo_id}@{sha} -> {target}", flush=True)
    snapshot_download(
        repo_id=repo_id,
        revision=sha,
        local_dir=target,
        allow_patterns=["*.safetensors", "*.json"],
        max_workers=2,
    )
    print(f"done: {target}", flush=True)
    return target


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument(
        "--revision", default="main", help="branch/tag/commit; resolved to a commit hash before download"
    )
    args = ap.parse_args()
    download(args.repo_id, args.revision)


if __name__ == "__main__":
    main()
