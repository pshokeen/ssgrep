"""Comparison / diff tooling for evaluation payloads.

``python -m eval.compare BASE CANDIDATE [--gates]`` prints a table of
per-metric deltas across every summary slice (``overall``, ``holdout``,
``class:*``) and, with ``--gates``, the five gate checks derived from the
baseline payload. The process exits 1 when any gate is breached and 0
otherwise, so the tool can gate CI-style comparisons::

    python -m eval.compare eval/results/baseline.json candidate.json --gates

Deltas are ``candidate - baseline`` per (slice, metric) cell, formatted to
four decimal places with a tolerance guard: deltas that round to zero at
that precision render as ``0.0000`` instead of a noisy signed epsilon, and
missing values render as ``n/a``. Gates reuse the runner's own derivation
and checking (:func:`eval.run_eval.derive_gates` /
:func:`eval.run_eval.check_gates`) so the diff tool can never disagree with
the runner about what a breach is.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from eval.run_eval import check_gates, derive_gates

# Display order mirrors eval.metrics.ALL_KEYS (the runner's summary keys).
METRIC_KEYS: tuple[str, ...] = (
    "ndcg@5",
    "ndcg@10",
    "rr@10",
    "rr@50",
    "p@5",
    "p@10",
    "r@50",
    "r@100",
)

# The summary slices the diff covers, in display order.
_SLICE_ORDER = ("overall", "holdout")


@dataclass(frozen=True)
class Comparison:
    """Deltas per (slice, metric) between two payloads' summaries."""

    base_arm: str | None
    candidate_arm: str | None
    slices: list[str]
    metrics: list[str]
    deltas: dict[str, dict[str, float | None]]


def _arm(payload: Mapping[str, object]) -> str | None:
    value = payload.get("arm")
    return value if isinstance(value, str) else None


def _aggregate(summary: Mapping[str, object], slice_name: str) -> Mapping[str, object]:
    value = summary.get(slice_name)
    return value if isinstance(value, Mapping) else {}


def _all_slices(
    base_summary: Mapping[str, object], candidate_summary: Mapping[str, object]
) -> list[str]:
    names = [name for name in _SLICE_ORDER if name in base_summary or name in candidate_summary]
    classes = sorted(
        {
            name
            for summary in (base_summary, candidate_summary)
            for name in summary
            if name.startswith("class:")
        }
    )
    return names + classes


def _metric_keys(
    base_summary: Mapping[str, object], candidate_summary: Mapping[str, object]
) -> list[str]:
    present = {
        metric
        for summary in (base_summary, candidate_summary)
        for aggregate in summary.values()
        if isinstance(aggregate, Mapping)
        for metric in METRIC_KEYS
        if metric in aggregate
    }
    return [metric for metric in METRIC_KEYS if metric in present]


def _delta(base_value: object, candidate_value: object) -> float | None:
    if base_value is None or candidate_value is None:
        return None
    if not isinstance(base_value, (int, float, str)) or not isinstance(
        candidate_value, (int, float, str)
    ):
        return None
    try:
        return float(candidate_value) - float(base_value)
    except (TypeError, ValueError):
        return None


def compare_payloads(base: Mapping[str, object], candidate: Mapping[str, object]) -> Comparison:
    """Deltas per (slice, metric); raises ValueError on a missing summary."""
    base_summary = base.get("summary")
    candidate_summary = candidate.get("summary")
    if not isinstance(base_summary, Mapping) or not isinstance(candidate_summary, Mapping):
        raise ValueError("both payloads must carry a 'summary' mapping")
    slices = _all_slices(base_summary, candidate_summary)
    metrics = _metric_keys(base_summary, candidate_summary)
    deltas = {
        slice_name: {
            metric: _delta(
                _aggregate(base_summary, slice_name).get(metric),
                _aggregate(candidate_summary, slice_name).get(metric),
            )
            for metric in metrics
        }
        for slice_name in slices
    }
    return Comparison(
        base_arm=_arm(base),
        candidate_arm=_arm(candidate),
        slices=slices,
        metrics=metrics,
        deltas=deltas,
    )


def _fmt_delta(delta: float | None) -> str:
    """Signed 4dp delta; ``n/a`` for missing; ``0.0000`` for sub-4dp noise."""
    if delta is None:
        return "    n/a"
    if abs(delta) < 5e-5:
        return "  0.0000"
    return f"{delta:+.4f}"


def _fmt_value(value: object) -> str:
    if value is None or not isinstance(value, (int, float, str)):
        return "n/a"
    return f"{float(value):.4f}"


def render_table(comparison: Comparison) -> str:
    """The delta table: one row per summary slice, one column per metric."""
    width = max((len(name) for name in comparison.slices), default=0) + 2
    lines = [f"{'slice':<{width}}" + "".join(f"{metric:>10}" for metric in comparison.metrics)]
    for slice_name in comparison.slices:
        row = f"{slice_name:<{width}}"
        row += "".join(
            f"{_fmt_delta(comparison.deltas[slice_name][metric]):>10}"
            for metric in comparison.metrics
        )
        lines.append(row)
    return "\n".join(lines)


def render_gates(gates: Mapping[str, object], checks: dict) -> str:
    """One line per gate check plus the overall verdict."""
    lines = ["gates (derived from baseline):"]
    for name, check in checks["checks"].items():
        boundary = check.get("floor", check.get("ceiling"))
        status = "PASS" if check["pass"] is True else ("n/a" if check["pass"] is None else "FAIL")
        lines.append(
            f"  {name:<14} value={_fmt_value(check['value'])}  "
            f"boundary={_fmt_value(boundary)}  {status}"
        )
    lines.append(f"GATES: {'PASS' if checks['all_pass'] else 'FAIL'}")
    return "\n".join(lines)


def _load_payload(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the process exit code (0 on success)."""
    parser = argparse.ArgumentParser(
        prog="eval.compare",
        description="Diff two evaluation payloads; optionally enforce baseline gates.",
    )
    parser.add_argument("base", type=Path, help="baseline result payload JSON")
    parser.add_argument("candidate", type=Path, help="candidate result payload JSON")
    parser.add_argument(
        "--gates",
        action="store_true",
        help="derive gates from BASE and check CANDIDATE against them",
    )
    args = parser.parse_args(argv)

    try:
        base = _load_payload(args.base)
        candidate = _load_payload(args.candidate)
        comparison = compare_payloads(base, candidate)
    except ValueError as error:
        print(f"usage error: {error}", file=sys.stderr)
        return 2

    print(f"baseline: {args.base} (arm={comparison.base_arm})")
    print(f"candidate: {args.candidate} (arm={comparison.candidate_arm})")
    print()
    print(render_table(comparison))

    if not args.gates:
        return 0
    try:
        gates = derive_gates(base)
        checks = check_gates(gates, candidate)
    except (KeyError, TypeError) as error:
        print(f"usage error: cannot derive gates from baseline: {error}", file=sys.stderr)
        return 2
    print()
    print(render_gates(gates, checks))
    return 0 if checks["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
