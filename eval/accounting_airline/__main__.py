"""``uv run python -m eval.accounting_airline --mode deterministic|standin|live``.

Deterministic mode drives the real PostgreSQL domain lane. ``standin`` runs the whole two-session
pipeline (checkpoints C0-C10) with the SCRIPTED STAND-IN agent; it never counts toward the live
gate. Authorized ``live`` executes two genuine native OMP conversations with two turns each,
at most ten minutes per turn, and retains private evidence even when a gate fails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import domain_lane
from .gates import BLOCKED, FAIL, NOT_RUN, PASS, Gate
from .report import LIVE_GATE, Report

# NOT_RUN/BLOCKED are never success: a lane that did not run cannot exit 0.
STATUS_EXIT = {PASS: 0, FAIL: 1, BLOCKED: 3, NOT_RUN: 3}


def _progress(gate: Gate) -> None:
    detail = f" - {gate.detail[:160]}" if gate.status != PASS and gate.detail else ""
    print(f"[{gate.status:<7}] {gate.name}{detail}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m eval.accounting_airline")
    parser.add_argument("--mode", choices=("deterministic", "standin", "live"), required=True)
    parser.add_argument("--runtime", help="runtime adapter for live mode (only 'omp' is verified)")
    parser.add_argument(
        "--authorize-live", action="store_true",
        help="explicit authorization to spend live-agent credits (live mode only)",
    )  # fmt: skip
    parser.add_argument(
        "--run-root-parent", type=Path, default=None,
        help="directory that will contain the owner-only run root (default: $TMPDIR)",
    )  # fmt: skip
    parser.add_argument(
        "--model",
        default="openai-codex/gpt-6-luna",
        help="explicit supported native model for live mode",
    )
    parser.add_argument(
        "--thinking",
        choices=("off", "minimal", "low", "medium", "high", "xhigh", "max", "auto"),
        default=None,
        help="native reasoning effort (fresh run default: low; resume: original pinned value)",
    )
    parser.add_argument(
        "--resume-live-root",
        type=Path,
        help="continue retained completed A with a fresh B; never repeat A",
    )
    return parser


def _live(args: argparse.Namespace) -> int:
    from .live_run import run_live

    if not args.authorize_live:
        verdict = Gate(
            "live.two_session_lab", NOT_RUN, "explicit authorization missing; pass --authorize-live"
        )
        _progress(verdict)
        root, rid = domain_lane.make_run_root(args.run_root_parent)
        domain_lane.write_private(
            root / "artifacts" / "live-lane.json",
            {"run_id": rid, "authorized": False, "gates": [verdict.as_dict()]},
        )
        return 3
    if args.resume_live_root is not None:
        from .live_resume import resume_live

        if args.runtime != "omp":
            _progress(Gate("live.runtime_binary", BLOCKED, "continuation requires --runtime omp"))
            return 3
        return resume_live(
            args.resume_live_root,
            lambda line: print(line, flush=True),
            authorized=True,
            model=args.model,
            thinking=args.thinking,
        )
    return run_live(
        args.run_root_parent,
        lambda line: print(line, flush=True),
        authorized=True,
        runtime=args.runtime or "",
        model=args.model,
        thinking=args.thinking or "low",
    )


def _standin_report_exit(report: Report) -> int:
    statuses = [report.gate_status(group) for group in report.gates if group != LIVE_GATE]
    if FAIL in statuses:
        return 1
    return 0 if all(status == PASS for status in statuses) else 3


def _standin(args: argparse.Namespace) -> int:
    from . import lab_run
    from .report import gate_table

    report, kept, teardown = lab_run.run_standin(
        args.run_root_parent, lambda line: print(line, flush=True)
    )
    print("\nCHECKPOINTS (runtime=scripted-standin; never counts toward the live gate)")
    print(report.checkpoints.table())
    print("\nGATES")
    print(gate_table(report))
    print(f"\nreport={kept / 'report.json'} lab_root_removed={teardown['root']['removed']}")
    return _standin_report_exit(report)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "standin":
        return _standin(args)
    if args.mode == "live":
        return _live(args)
    run_root, run_id = domain_lane.make_run_root(args.run_root_parent)
    print(f"run_id={run_id} run_root={run_root}", flush=True)
    report = domain_lane.run_lane(run_root, run_id, _progress)
    counts = {s: sum(1 for g in report.gates if g.status == s) for s in (PASS, FAIL, BLOCKED)}
    print(
        f"domain={report.domain_status} live=NOT_RUN(deterministic mode) gates={len(report.gates)} "
        f"pass={counts[PASS]} fail={counts[FAIL]} blocked={counts[BLOCKED]} "
        f"elapsed={report.elapsed_s:.1f}s report={run_root / 'artifacts' / 'domain-lane.json'}",
        flush=True,
    )
    return STATUS_EXIT.get(report.domain_status, 1)


if __name__ == "__main__":
    sys.exit(main())
