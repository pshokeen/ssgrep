"""A real collector process for the guarantee tests: run one cycle, die or pause at a hook.

``python -m tests.accounting.verify_driver collect <fault> <now> [--hold DIR] [--new-epoch]
[--bytes N] [--seconds N] [--repin] [--backfill SINCE UNTIL]``

* ``fault`` is one of the cycle's fault-hook step names (or ``none``). Without ``--hold`` the
  process SIGKILLs ITSELF right at that step (no cleanup, no ``finally``, lease descriptor
  dropped by the kernel); the step name is first written to ``<DIR>/died-at`` when ``--hold`` is
  absent and ``--marks DIR`` is given.
* With ``--hold DIR`` it creates ``DIR/reached`` at the step, then waits for ``DIR/release``
  (bounded) and carries on, so a test can act while the cycle is genuinely pending and holds
  the profile lease.
* ``now`` is the collection clock (``YYYY-MM-DDTHH:MM:SSZ``) so reference runs are reproducible.

The environment (data dir and runtime store roots) comes from the parent, exactly as for any
real ``ssgrep`` process; nothing here knows about the test's tmp paths.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from ssgrep.accounting.collector import collect_once
from ssgrep.accounting.collector_types import CollectOptions, CycleBudget

PROFILE = "personal-local"
WAIT_SECONDS = 120


def _hook(fault: str, hold: Path | None, marks: Path | None):
    def on_step(name: str) -> None:
        if name != fault:
            return
        if hold is not None:
            hold.mkdir(parents=True, exist_ok=True)
            (hold / "reached").write_text(name)
            deadline = time.monotonic() + WAIT_SECONDS
            while not (hold / "release").exists():
                if time.monotonic() > deadline:
                    raise SystemExit("verify_driver: never released")
                time.sleep(0.02)
            return
        if marks is not None:
            marks.mkdir(parents=True, exist_ok=True)
            (marks / "died-at").write_text(name)
        os.kill(os.getpid(), signal.SIGKILL)

    return on_step


def main(argv: list[str]) -> int:
    if argv[0] != "collect":
        raise SystemExit(f"unknown mode {argv[0]}")
    fault, now = argv[1], datetime.fromisoformat(argv[2]).astimezone(UTC)
    flags = argv[3:]
    hold = Path(flags[flags.index("--hold") + 1]) if "--hold" in flags else None
    marks = Path(flags[flags.index("--marks") + 1]) if "--marks" in flags else None
    default = CycleBudget()

    def number(flag: str, fallback: int) -> int:
        return int(flags[flags.index(flag) + 1]) if flag in flags else fallback

    budget = CycleBudget(
        seconds=number("--seconds", default.seconds), bytes=number("--bytes", default.bytes)
    )
    since = until = None
    if "--backfill" in flags:
        at = flags.index("--backfill")
        since, until = flags[at + 1], flags[at + 2]
    options = CollectOptions(
        since=since,
        until=until,
        backfill="--backfill" in flags,
        new_epoch="--new-epoch" in flags,
        repin_frequency="--repin" in flags,
        budget=budget,
    )
    result = collect_once(
        PROFILE, options, now=now, on_step=None if fault == "none" else _hook(fault, hold, marks)
    )
    print(
        json.dumps(
            {
                "state": result.state,
                "generation": result.report.generation if result.report else None,
                "snapshot_id": result.report.snapshot_id if result.report else None,
                "epoch": result.epoch,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
