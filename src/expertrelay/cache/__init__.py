"""Hot/cold expert placement and eviction across RAM, SSD, and networked
devices -- the "virtual memory" part of ExpertRelay.

`expert_cache`: the runtime's cache (--source cached): experts in RAM
under a fixed budget, LRU, optional pinned layer 0, background prefetch.
Single device: RAM and one SSD, no networked devices yet.
`simulator` and `predictive` replay recorded traces offline to compare
policies. See docs/limitations.md.
"""
