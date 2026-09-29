# Store metadata

Copies of each expert store's `store.json` and `experts_index.json` (and
the int8 store's `resident_index.json`, which the int4 stores share: their
`resident.bin` is the int8 store's, hard-linked). The stores themselves
(`models/`) are too large for git and are rebuilt with
`store/build_store.py` (int8, from the Hub at the pinned revision) and
`store/build_int4_store.py` (int4, from the original bf16 checkpoint).
`experts_index.json` holds the sha256 of every expert record and its
quantization error, so a rebuilt store can be checked record by record.
