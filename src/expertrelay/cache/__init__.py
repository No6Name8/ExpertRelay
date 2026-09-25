"""Hot/cold expert placement and eviction across RAM, SSD, and networked
devices -- the "virtual memory" part of ExpertRelay.

Not implemented yet: today, expert placement (which process holds which
expert) is a static assignment fixed at conversion time
(expertrelay.store.convert_qwen_moe --process-a-experts/--process-b-experts),
not a runtime cache with eviction. See docs/limitations.md.
"""
