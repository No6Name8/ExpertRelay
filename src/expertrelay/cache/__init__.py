"""Hot/cold expert placement and eviction across RAM, SSD, and networked
devices -- the "virtual memory" part of ExpertRelay.

Not in the runtime yet: today every routed expert is read from the SSD
each time the router picks it and dropped right after (runtime.weights
.UnbufferedExpertSource). Nothing is kept, so there is nothing to evict.
`simulator` and `predictive` replay recorded traces offline to choose the
policy the runtime cache should use. See docs/limitations.md.
"""
