"""Predicting which expert a future token/layer will route to, ahead of the
router actually deciding, so the Manager can start moving it early.

Not in the runtime yet: today the router's decision (expertrelay.runtime
.qwen_moe.QwenMoe._moe) is only known at the moment it's needed, and every
expert read is synchronous: compute waits for it. There is no early read.
`offline` holds the tools fitted and scored on recorded traces (Phase 3.5)
to decide what the runtime predictor should be. Each method from the
literature cites its source paper in a code comment, per CLAUDE.md. See
docs/limitations.md.
"""
