"""Predicting which expert a future token/layer will route to, ahead of the
router actually deciding, so the Manager can start moving it early.

Not implemented yet: today the router's decision (expertrelay.runtime
.moe_model.MoELayer) is only known at the moment it's needed, and every
remote expert call is a synchronous request -- there is no early-send.
Prediction approaches from the literature (e.g. Fate's cross-layer gate
prediction) will land here; each such method must cite its source paper in
a code comment, per CLAUDE.md. See docs/limitations.md.
"""
