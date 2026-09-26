"""The tokenizer that goes with a store, fetched at the store's pinned revision.

Kept in the store directory next to the weights, so a store is
self-contained: the same commit hash pins both the weights and how text maps
to token ids.
"""

from __future__ import annotations

import json
from pathlib import Path

import tokenizers

from expertrelay.store.fetch_hf_tensors import fetch_file

TOKENIZER_FILE = "tokenizer.json"


def ensure_tokenizer(store_dir: Path) -> Path:
    path = Path(store_dir) / TOKENIZER_FILE
    if not path.exists():
        source = json.loads((Path(store_dir) / "store.json").read_text())["source"]
        path.write_bytes(fetch_file(source["repo_id"], TOKENIZER_FILE, source["revision"]))
    return path


def load_tokenizer(store_dir: Path) -> tokenizers.Tokenizer:
    return tokenizers.Tokenizer.from_file(str(ensure_tokenizer(store_dir)))
