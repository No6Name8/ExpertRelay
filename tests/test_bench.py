"""Smoke tests for expertrelay.bench.* -- moe_sanity_check runs standalone
(tiny, untrained, fast) and run_local_split_demo's arg parsing works.
Actually launching the two-process demo needs a real converted checkpoint
(see docs/setup-notes.md) and is exercised manually, not in this suite.
"""

from __future__ import annotations

from expertrelay.bench import moe_sanity_check, run_local_split_demo


def test_moe_sanity_check_runs_without_error(capsys):
    moe_sanity_check.main()
    out = capsys.readouterr().out
    assert "MoE sanity check PASSED" in out


def test_run_local_split_demo_arg_parser_defaults():
    parser = run_local_split_demo.build_arg_parser()
    args = parser.parse_args([])
    assert args.port == 50051
    assert args.max_new_tokens == 20
    assert str(args.benchmark_out).endswith("local_split_simulation.json")
