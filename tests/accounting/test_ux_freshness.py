"""10.4: stale-after-three-intervals, status fields, and reads that never change anything.

Fingerprints cover every file AND directory (mtime_ns, size, bytes) under the data root, so
a touched timestamp, a created lock file or a rewritten checkpoint is caught, not just a
changed byte. Every read command runs repeatedly through the real binary and in-process.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from ssgrep.accounting import acks, policy, service
from ssgrep.accounting.ledger_contracts import DestinationPolicy, Signal, SnapshotState
from ssgrep.accounting.service_status import read_status
from tests.accounting import collector_scene as scene
from tests.accounting.test_ux_support import NOW, Binary, collected_scene, fingerprint

COLLECTED = scene.at(hour=20)


@pytest.fixture
def collected(tmp_path: Path) -> Path:
    collected_scene(tmp_path)
    return tmp_path


@pytest.mark.parametrize(
    ("age", "stale"),
    [(0, False), (60, False), (179, False), (180, False), (181, True), (3600, True)],
)
def test_staleness_begins_after_exactly_three_intervals_of_180_seconds(
    collected: Path, age: int, stale: bool
) -> None:
    view = service.read_summary(now=COLLECTED + timedelta(seconds=age))
    assert view.stale is stale and view.age_seconds == age
    assert view.state is (SnapshotState.STALE if stale else SnapshotState.FRESH)
    assert view.has_values, "stale keeps the last good values"
    status = read_status("personal-local", now=COLLECTED + timedelta(seconds=age))
    assert status.stale is stale and status.stale_after_seconds == 180


def test_pause_is_a_separate_state_and_stale_age_continues_while_paused(collected: Path) -> None:
    policy.pause("personal-local")
    fresh = service.read_summary(now=COLLECTED + timedelta(seconds=10))
    old = service.read_summary(now=COLLECTED + timedelta(seconds=999))
    assert fresh.state is SnapshotState.PAUSED and not fresh.stale
    assert old.state is SnapshotState.PAUSED and old.stale, "paused AND stale are both visible"
    assert old.age_seconds == 999


def test_status_reports_coverage_lag_storage_and_both_freshness_clocks(collected: Path) -> None:
    status = read_status("personal-local", now=COLLECTED + timedelta(seconds=45))
    assert status.collection_age_seconds == 45
    assert status.delivery_age_seconds is None, "nothing delivered: unknown, not 45"
    assert status.lag.lag_bytes == 0 and status.lag.sources_total
    assert status.missing.missing_sources == 0
    assert status.storage.level in {"ok", "high", "critical"}
    assert {c.name for c in status.storage.components} >= {"snapshots", "receipts", "history"}
    runtimes = {r.runtime for r in status.lag.by_runtime}
    assert "claude" in runtimes


def test_status_json_exposes_the_same_fields_as_integers_and_nulls(collected: Path) -> None:
    body = Binary(collected).data("metrics", "status")["data"]
    assert body["snapshot"]["stale_after_seconds"] == 180
    assert isinstance(body["snapshot"]["age_seconds"], int)
    assert body["freshness"]["delivery"]["age_seconds"] is None
    assert body["lag"]["lag_bytes"] == 0 and body["lag"]["by_runtime"]
    assert body["storage_pressure"]["level"] in {"ok", "high", "critical"}
    assert body["missing_evidence"]["missing_sources"] == 0
    assert body["watermarks"]["count"] and body["watermarks"]["items"]
    assert "acknowledgment" in body["freshness"]["delivery"]


# ---- reads change nothing ----

READS = [
    ("savings",),
    ("savings", "--by", "task"),
    ("savings", "--runtime", "claude", "--since", "2026-10-01"),
    ("savings", "history"),
    ("metrics", "status"),
    ("metrics", "census", "--root", "{home}"),
]


def test_repeated_real_reads_leave_every_file_and_directory_untouched(collected: Path) -> None:
    run = Binary(collected)
    receipt = run.data("savings")["data"]["representative_receipt"]
    commands = [
        *READS,
        ("savings", "explain", receipt),
        ("savings", "explain", receipt, "--include-content"),
    ]
    root = collected / "data"
    before = fingerprint(root)
    state_before = fingerprint(collected / "claude")
    for _ in range(2):
        for args in commands:
            concrete = tuple(a.replace("{home}", str(collected / "home")) for a in args)
            for mode in ((), ("--json",)):
                done = run(*concrete, *mode)
                assert done.returncode == 0, (concrete, done.stderr[-300:])
    assert fingerprint(root) == before, "reading created, touched or rewrote an accounting file"
    assert fingerprint(collected / "claude") == state_before, "source transcripts were touched"


def test_reads_leave_delivery_acknowledgments_and_checkpoints_exactly_as_they_were(
    collected: Path,
) -> None:
    from dataclasses import replace

    dest = DestinationPolicy(
        "org-primary", "https://collector.example.test:4318", (Signal.METRICS,)
    )
    policy.save_profile(
        replace(
            policy.load_profile_policy("personal-local"),
            destinations=(dest,),
            enabled_exports=(("org-primary", "metrics"),),
        )
    )
    acks.update_delivery(
        "personal-local",
        "org-primary",
        Signal.METRICS,
        lambda d: acks.retry_scheduled(
            acks.set_counts(acks.collected(d, 1), pending_batches=2, unknown_batches=0),
            "2026-10-12T20:05:00Z",
            "network_error",
        ),
    )
    root = collected / "data" / "accounting" / "personal-local"
    watched = ("delivery", "collector", "snapshots", "receipts", "history")
    before = {name: fingerprint(root / name) for name in watched}
    run = Binary(collected)
    for args in (("savings",), ("metrics", "status"), ("savings", "history")):
        for _ in range(2):
            assert run(*args).returncode == 0 and run(*args, "--json").returncode == 0
    after = {name: fingerprint(root / name) for name in watched}
    assert after == before
    line = read_status("personal-local", now=NOW).delivery_lines[0]
    assert line.pending_batches == 2 and line.next_retry_utc == "2026-10-12T20:05:00Z"
