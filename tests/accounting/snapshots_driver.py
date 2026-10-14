"""Subprocess helpers for snapshot recovery tests (real processes, real ``os._exit``).

``python -m tests.accounting.snapshots_driver publish <profile> <generation> <fault> [bulk]``
publishes the deterministic envelope ``make_envelope(generation, bulk)`` and dies with
``os._exit(9)`` at ``fault``: ``before`` (before any step), one of ``snapshots.STEPS`` (right
after that step), or ``none``.

``read <profile> <ready-file> <stop-file>`` hammers ``service.read_snapshot`` (creating
``ready-file`` after its first complete read) until ``stop-file`` exists, then prints
``<reads> <first> <last> <bad>``; ``bad`` counts torn, unverifiable or regressing reads.
``make_envelope`` is shared with the parent so it can compare exact bytes.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from ssgrep.accounting import service, snapshots
from ssgrep.accounting.ledger_contracts import SnapshotEnvelope
from tests.accounting.snapshot_builders import window_7d

CRASH_CODE = 9


def make_envelope(generation: int, bulk: int = 0, **overrides: Any) -> SnapshotEnvelope:
    """A complete valid envelope whose content (and size) depends on generation/bulk."""
    fields: dict[str, Any] = {
        "generation": generation,
        "counter_epoch": 1,
        "methodology_id": "m-test-v1",
        "policy_version": 1,
        "collected_at": f"2026-10-12T12:{generation % 60:02d}:00Z",
        "coverage_through": "2026-10-12T11:59:00Z",
        "source_manifest_digest": "b" * 16,
        "windows": {
            "7d": window_7d(),
            "30d": {"blob": f"{generation}:" + "x" * bulk, "seq": generation},
        },
        "coverage": {"partial": False},
        "observations": {"seq": generation},
        "estimates": {"representative_receipt": "rcpt-7f3a91"},
        "warnings": (),
    }
    fields.update(overrides)
    return snapshots.build_envelope(**fields)


def gen_name(generation: int, bulk: int = 0, **overrides: Any) -> str:
    """File name of the generation file for ``make_envelope(generation, bulk, **overrides)``."""
    envelope = make_envelope(generation, bulk, **overrides)
    return f"gen-{generation:010d}-{envelope.snapshot_id}.json"


def _publish(profile: str, generation: int, fault: str, bulk: int) -> None:
    def on_step(name: str) -> None:
        if name == fault:
            os._exit(CRASH_CODE)

    if fault == "before":
        os._exit(CRASH_CODE)
    snapshots.publish(profile, make_envelope(generation, bulk), on_step)


def _read(profile: str, ready: str, stop: str) -> None:
    reads = bad = 0
    first = last = 0
    while not os.path.exists(stop):
        envelope, failure, _ = service.read_snapshot(profile)
        if envelope is None:
            if failure is not None:
                bad += 1
            continue
        reads += 1
        try:
            verified = snapshots.snapshot_identity(envelope.__dict__) == envelope.snapshot_id
        except ValueError:
            verified = False
        if not verified or envelope.generation < last:
            bad += 1
        first = first or envelope.generation
        if reads == 1:
            open(ready, "w").close()
        last = max(last, envelope.generation)
    print(reads, first, last, bad)


def main(argv: list[str]) -> None:
    if argv[0] == "publish":
        _publish(argv[1], int(argv[2]), argv[3], int(argv[4]) if len(argv) > 4 else 0)
    elif argv[0] == "read":
        _read(argv[1], argv[2], argv[3])
    else:
        raise SystemExit(f"unknown mode {argv[0]}")


if __name__ == "__main__":
    main(sys.argv[1:])
