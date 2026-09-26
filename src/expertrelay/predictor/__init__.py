"""Predicting which expert a future token/layer will route to, ahead of the
router actually deciding, so the Manager can start moving it early.

`prefetch_policy`: what the runtime prefetches (--source cached with
prefetch on): the top-k of the Fate-style cross-layer guess, minus guesses
whose calibrated probability is too low. `offline` holds the tools fitted
and scored on recorded traces (Phase 3.5), including that calibration.
Each method from the literature cites its source paper in a code comment,
per CLAUDE.md. See docs/limitations.md.
"""
