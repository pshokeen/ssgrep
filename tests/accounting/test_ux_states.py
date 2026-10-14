"""10.2: every named state names its cause and a REAL next command, with --profile preserved.

Snapshots, receipts and the ledger come from the real collector; delivery documents are written
by the real ``acks`` writer (the export module's own API); pause/enable go through the real
policy module. Assertions are about state, cause and an executable action, not prose.
"""

from __future__ import annotations

import shlex
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from ssgrep.accounting import acks, policy, service
from ssgrep.accounting.ledger_contracts import (
    DestinationPolicy,
    Signal,
    SnapshotState,
)
from ssgrep.accounting.service_status import read_status
from ssgrep.accounting.service_types import Condition, NextAction
from ssgrep.cli import savings_render
from ssgrep.cli.metrics_render import render_status
from tests.accounting import collector_scene as scene, verify_scenes as vs
from tests.accounting.test_ux_support import NOW, collected_scene

DEST = "org-primary"
ENDPOINT = "https://collector.example.test:4318"
STAMP = "2026-10-12T20:00:00Z"
COMMANDS = {
    "savings",
    "savings explain",
    "savings history",
    "metrics setup",
    "metrics status",
    "metrics collect",
    "metrics resume",
    "metrics export",
    "index",
}


def action_of(item: Any) -> NextAction:
    """The next action a condition or delivery line carries (fails loudly when it has none)."""
    assert item.action is not None, item
    return item.action


def conditions(view: Any) -> dict[str, Condition]:
    return {c.code: c for c in view.conditions}


def assert_executable(command: str, profile: str = "personal-local") -> None:
    """Parses as an ssgrep command whose subcommand exists and whose --profile is preserved."""
    words = shlex.split(command)
    assert words[0] == "ssgrep", command
    name = " ".join(w for w in words[1:3] if not w.startswith("-"))
    assert any(name == c or name.startswith(c.split()[0] + " " + c.split()[-1]) or
               name.split()[0] == c for c in COMMANDS), command  # fmt: skip
    assert not any(w.startswith("<") or w.endswith(">") for w in words), command
    if profile != "personal-local":
        assert words[-2:] == ["--profile", profile], command
    else:
        assert "--profile" not in words, command


def with_destination(*, enabled: bool = True, profile: str = "personal-local") -> None:
    dest = DestinationPolicy(DEST, ENDPOINT, (Signal.METRICS,), credential_ref="env:UX_TOKEN")
    current = policy.load_profile_policy(profile)
    policy.save_profile(
        replace(
            current,
            destinations=(dest,),
            enabled_exports=((DEST, "metrics"),) if enabled else (),
        )
    )


def write_delivery(change) -> None:
    acks.update_delivery("personal-local", DEST, Signal.METRICS, change)


@pytest.fixture
def collected(tmp_path: Path) -> Path:
    collected_scene(tmp_path)
    return tmp_path


# ---- readable states without numbers -----------------------------------------------------------


def test_disabled_names_setup_and_does_not_enable_anything(tmp_path: Path) -> None:
    view = service.read_summary(now=NOW)
    assert view.state is SnapshotState.DISABLED and not view.has_values
    assert set(conditions(view)) == {"disabled"}
    assert view.next_actions[0].command == "ssgrep metrics setup"
    assert not policy.load_profile_policy("personal-local").collection_enabled
    assert not (tmp_path / "data").exists(), "reading created no storage"


def test_enabled_without_snapshot_says_capture_on_collection_incomplete_and_shows_no_zero(
    tmp_path: Path,
) -> None:
    vs.enable()
    view = service.read_summary(now=NOW)
    assert view.state is SnapshotState.NO_SNAPSHOT and view.measured == ()
    assert view.next_actions[0].command == "ssgrep metrics collect"
    text = savings_render.render_summary(view, 80).plain
    assert "MEASURED" not in text and "ESTIMATED" not in text
    assert "Capture is enabled" in " ".join(text.split())


def test_every_condition_with_an_action_offers_a_parseable_command_with_the_profile(
    tmp_path: Path,
) -> None:
    policy.enable_collection("acme", {"profile_id": "acme", "roots": [str(tmp_path)]})
    view = service.read_summary("acme", now=NOW)
    assert view.conditions and all(c.action is not None for c in view.conditions)
    for condition in view.conditions:
        assert condition.action is not None
        assert_executable(condition.action.command, "acme")
    assert_executable(view.next_actions[0].command, "acme")


# ---- fresh states from a real collection ---------------------------------------------------------


def test_negative_partial_values_keep_their_signs_and_name_receipt_and_coverage_actions(
    collected: Path,
) -> None:
    view = service.read_summary(now=NOW)
    assert view.state is SnapshotState.FRESH and view.has_negative and view.partial
    found = conditions(view)
    assert {"negative", "partial", "index_unavailable"} <= set(found)
    assert (
        view.receipt_id
        and f"savings explain {view.receipt_id}" in action_of(found["negative"]).command
    )
    text = savings_render.render_summary(view, 40).plain
    assert "-133 tokens" in " ".join(text.split()) or "-133" in text
    assert not any(word in text.lower() for word in ("saved", "great", "congrat", "win", "!"))
    for condition in found.values():
        if condition.action:
            assert_executable(condition.action.command)


def test_index_unavailable_is_separate_from_usage_and_never_asks_for_exit_4(
    collected: Path,
) -> None:
    view = service.read_summary(now=NOW)
    index = conditions(view)["index_unavailable"]
    assert view.index == "unavailable" and dict(view.measured)["searches"].value == 4
    assert index.action is not None and index.action.command == "ssgrep index"
    status = read_status("personal-local", now=NOW)
    assert status.state is SnapshotState.FRESH, "status stays readable without the index"
    assert not (collected / "data" / "lancedb").exists(), "nothing built an index"


def test_no_traceable_reuse_shows_costs_and_exclusions_but_claims_nothing(tmp_path: Path) -> None:
    vs.enable(pin=True)
    vs.start_capture(day=12)
    vs.put(
        vs.projects(tmp_path),
        "lonely",
        [vs.claude_prompt("lonely", "hello", uuid="u", ts=vs.ts(12, 9))],
        scene.at(hour=10),
    )
    scene.journal_events(scene.execution("n-1", vs.ts(12, 9, 5), text="anything", transport="cli"))
    assert scene.collect(NOW).state == "published"
    view = service.read_summary(now=NOW)
    assert view.no_reuse and view.coverage["candidate_receipts"] == 0
    assert "no_reuse" in conditions(view)
    claimed = [q.value for r in view.scenarios for _, q in r.net if q.value is not None]
    assert all(v <= 0 for v in claimed), "no positive savings were invented"
    text = " ".join(savings_render.render_summary(view, 80).plain.split())
    assert "No traceable reuse was found" in text and "no savings are claimed" in text


# ---- time-dependent states ----------------------------------------------------------------------


def test_stale_keeps_last_good_values_with_age_and_collect_and_reads_do_not_refresh_it(
    collected: Path,
) -> None:
    later = NOW + timedelta(hours=3)
    first = service.read_summary(now=later)
    assert first.state is SnapshotState.STALE and first.stale and first.has_values
    assert first.age_seconds == int((later - scene.at(hour=20)).total_seconds())
    assert action_of(conditions(first)["stale"]).command == "ssgrep metrics collect"
    assert scene.current().collected_at == "2026-10-12T20:00:00Z"
    again = service.read_summary(now=later + timedelta(seconds=30))
    assert again.age_seconds == first.age_seconds + 30, "age keeps growing; a read resets nothing"
    assert dict(first.measured) == dict(again.measured)


def test_paused_keeps_history_and_names_timestamp_and_resume_without_collect(
    collected: Path,
) -> None:
    policy.pause("personal-local")
    view = service.read_summary(now=NOW + timedelta(hours=3))
    assert view.state is SnapshotState.PAUSED and view.has_values and view.stale
    paused = conditions(view)["paused"]
    assert action_of(paused).command == "ssgrep metrics resume"
    assert view.paused_at and view.paused_at in paused.cause
    assert conditions(view)["stale"].action is None, "collect would fail while paused"
    assert "ssgrep metrics collect" not in [a.command for a in view.next_actions]


def test_evidence_expired_keeps_totals_and_states_the_boundary(collected: Path) -> None:
    later = scene.at(hour=10) + timedelta(days=45)
    assert scene.collect(later).state == "published"
    view = service.read_summary(now=later)
    expired = conditions(view)["evidence_expired"]
    assert view.evidence_expired_before and view.frozen_roots
    assert view.evidence_expired_before in expired.cause
    assert action_of(expired).command == "ssgrep savings history"
    history = service.read_history()
    assert history.entries and not any(e.recomputable for e in history.entries)
    receipt = service.read_receipt("personal-local", view.receipt_id or "")
    assert receipt is None or not receipt.evidence_available


# ---- delivery states ----------------------------------------------------------------------------


def delivery_state(profile: str = "personal-local") -> dict[str, Any]:
    status = read_status(profile, now=NOW)
    assert len(status.delivery_lines) == 1
    return {"line": status.delivery_lines[0], "view": status}


def test_configured_but_disabled_and_enabled_but_unsent_are_distinct(collected: Path) -> None:
    with_destination(enabled=False)
    assert delivery_state()["line"].state == "disabled"
    with_destination(enabled=True)
    line = delivery_state()["line"]
    assert line.state == "waiting" and "nothing has been queued" in line.cause
    assert "--preview" in line.action.command


def test_offline_destination_shows_pending_and_next_retry_never_synced(collected: Path) -> None:
    with_destination()
    write_delivery(
        lambda d: acks.retry_scheduled(
            acks.set_counts(acks.collected(d, 1), pending_batches=3, unknown_batches=0),
            "2026-10-12T20:05:00Z",
            "network_error",
        )
    )
    line = delivery_state()["line"]
    assert line.state == "offline" and line.pending_batches == 3
    assert line.next_retry_utc == "2026-10-12T20:05:00Z"
    assert "Not synced" in line.cause and "local collection continues" in line.cause
    assert line.action.command.endswith("--check")
    assert_executable(line.action.command.replace(f"--destination {DEST}", f"--destination {DEST}"))
    summary = service.read_summary(now=NOW)
    assert "offline" in conditions(summary)
    text = " ".join(savings_render.render_summary(summary, 80).plain.split())
    assert "Not synced" in text and "synced." not in text.replace("Not synced", "")


def test_credentials_expired_names_rotation_and_never_prints_a_secret(
    collected: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UX_TOKEN", "sk-SECRET-VALUE-0001")
    with_destination()
    write_delivery(
        lambda d: acks.blocked(
            acks.set_counts(acks.collected(d, 1), pending_batches=2, unknown_batches=0),
            "unauthorized",
            "fp-1",
        )
    )
    view = read_status("personal-local", now=NOW)
    line = view.delivery_lines[0]
    assert line.state == "credentials_expired" and line.blocked_reason == "unauthorized"
    assert "Rotate the credential" in action_of(line).why
    rendered = render_status(view, 80).plain
    assert "sk-SECRET-VALUE-0001" not in rendered and "credentials expired" in rendered.lower()
    summary = service.read_summary(now=NOW)
    assert action_of(conditions(summary)["credentials_expired"]).command == action_of(line).command


def test_partial_export_shows_counts_and_is_never_called_acknowledged(collected: Path) -> None:
    with_destination()

    def partial(d: acks.Delivery) -> acks.Delivery:
        d = acks.collected(d, 1)
        d = acks.attempted(d, 1, STAMP)
        return acks.outcome(
            d,
            generation=1,
            snapshot_id="a1" * 32,
            index=0,
            total=2,
            kind="partial",
            now=STAMP,
            rejected=4,
        )

    write_delivery(partial)
    line = delivery_state()["line"]
    assert line.state == "partial_export" and line.rejected_points == 4
    assert line.acked_generation is None and "Not fully acknowledged" in line.cause


def test_collection_and_delivery_freshness_are_independent(collected: Path) -> None:
    with_destination()

    def acked(d: acks.Delivery) -> acks.Delivery:
        d = acks.collected(d, 1)
        d = acks.attempted(d, 1, "2026-10-12T19:00:00Z")
        return acks.outcome(
            d,
            generation=1,
            snapshot_id="a1" * 32,
            index=0,
            total=1,
            kind="acked",
            now="2026-10-12T19:00:00Z",
        )

    write_delivery(acked)
    status = read_status("personal-local", now=NOW + timedelta(hours=1))
    assert status.collection_age_seconds == 3600 and status.stale
    assert status.delivery_age_seconds == 7200 and status.delivery_lines[0].state == "acknowledged"
    assert status.collection_age_seconds != status.delivery_age_seconds
    from ssgrep.cli.metrics_json import status_document

    document = status_document(status)
    data = document["data"]
    assert isinstance(data, dict)
    freshness = data["freshness"]
    assert freshness["collection"]["age_seconds"] == 3600
    assert freshness["delivery"]["age_seconds"] == 7200
