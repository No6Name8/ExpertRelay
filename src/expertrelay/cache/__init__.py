"""Hot/cold expert placement and eviction across RAM, SSD, and networked
devices -- the "virtual memory" part of ExpertRelay.

Not implemented yet: today every routed expert is read from the SSD each
time the router picks it and dropped right after (runtime.weights
.UnbufferedExpertSource). Nothing is kept, so there is nothing to evict.
See docs/limitations.md.
"""
