"""Predicting which expert a future token/layer will route to, ahead of the
router actually deciding, so the Manager can start moving it early.

Not implemented yet: today the router's decision (expertrelay.runtime
.qwen_moe.QwenMoe._moe) is only known at the moment it's needed, and every
expert read is synchronous: compute waits for it. There is no early read.
Prediction approaches from the literature (e.g. Fate's cross-layer gate
prediction) will land here; each such method must cite its source paper in
a code comment, per CLAUDE.md. See docs/limitations.md.
"""
