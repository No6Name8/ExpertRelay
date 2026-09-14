# Benchmarks

## local_split_simulation.json

**Single-machine simulation, reduced expert count.** Two Python processes on
this one 8GB laptop (`expert_server.py` = "Process B", `coordinator.py` =
"Process A") talking over a real TCP socket on `127.0.0.1`, running a
2-layer / 4-expert reduced slice of the real Qwen1.5-MoE-A2.7B weights (see
`docs/setup-notes.md` for exactly what's real vs. reduced).

**Do not compare this number to a future real two-machine benchmark without
re-reading the caveats below** -- it measures very different things:

- No KV cache: every generation step re-runs the full forward pass over the
  whole sequence so far. Tokens/sec gets *worse* as the sequence grows, not
  representative of a real serving setup.
- One token at a time, no batching, and each of a token's top-2 experts is
  dispatched one at a time (a Python `for` loop over experts, not real
  parallel dispatch) -- current numbers are a correctness/plumbing
  baseline, not a throughput ceiling.
- The remote calls hit `127.0.0.1` -- effectively free network latency
  (no real NIC hop, no real link bandwidth limits). A real second machine
  over LAN/USB4/etc. will look different in both directions: possibly
  slower per call (real network latency) but the two processes would then
  also be running on separate CPUs instead of time-slicing one 8GB machine.
- Only 2 of the real model's 24 layers and 4 of its 60 experts are present.

What it's actually good for: proving the local/remote expert dispatch
split, the TCP request/response protocol, and the conversion pipeline all
work correctly end-to-end before scaling up. The `remote_calls` /
`local_calls` counts per layer are the more meaningful numbers here than
the raw tokens/sec.
