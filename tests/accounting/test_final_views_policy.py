"""Policy refusal is atomic; delivery state retains scope and real event ages."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ssgrep.accounting import acks, policy, service, service_delivery, service_status
from ssgrep.accounting.ledger_contracts import DestinationPolicy, ProfilePolicy, Signal
from ssgrep.accounting.policy_rules import (
    MAX_DESTINATIONS,
    MAX_ROOTS,
    PolicyError,
    PolicyFile,
    check_endpoint,
    check_proxy,
    validate_profile,
)
from ssgrep.accounting.scope import build_scope
from tests.accounting.http_builders import tree_state, write_delivery

NOW = datetime(2026, 10, 12, 12, 0, 30, tzinfo=UTC)
DEST = DestinationPolicy("otel", "https://collector.example.test", (Signal.METRICS,))


@pytest.mark.parametrize(
    "raw",
    [
        [],
        {"extra": True},
        {"profile_id": "other"},
        {"evidence_retention_days": 3651},
        {"assignments": []},
    ],
)
def test_invalid_managed_input_cannot_replace_existing_policy(tmp_path: Path, raw):
    policy.enable_collection("personal-local")
    before = tree_state(tmp_path / "data")
    with pytest.raises(PolicyError) as caught:
        policy.enable_collection("personal-local", raw)
    assert caught.value.condition == "invalid_policy"
    assert policy.load_profile_policy().collection_enabled
    assert tree_state(tmp_path / "data") == before


@pytest.mark.parametrize("url", [None, "", "x" * 2049])
def test_endpoint_requires_bounded_url(url):
    with pytest.raises(PolicyError) as caught:
        check_endpoint(url)
    assert caught.value.condition == "invalid_policy"
    assert check_endpoint("https://collector.example.test") == "https://collector.example.test"


@pytest.mark.parametrize(
    "url",
    [
        None,
        "",
        "x" * 2049,
        "ftp://proxy.test",
        "https:///",
        "https://secret@proxy.test",
        "https://proxy.test/path",
        "https://proxy.test/?token=secret",
        "https://proxy.test/#fragment",
    ],
)
def test_proxy_refuses_credential_and_unbounded_routing(url):
    with pytest.raises(PolicyError) as caught:
        check_proxy(url)
    assert caught.value.condition == "invalid_policy"
    assert check_proxy("https://proxy.test:8443/") == "https://proxy.test:8443/"


@pytest.mark.parametrize(
    "change",
    [
        {"roots": tuple(f"/root/{i}" for i in range(MAX_ROOTS + 1))},
        {
            "destinations": tuple(
                replace(DEST, destination_id=f"d{i}") for i in range(MAX_DESTINATIONS + 1)
            )
        },
        {"destinations": (DEST, DEST)},
        {"destinations": (replace(DEST, destination_id="../escape"),)},
        {"destinations": (replace(DEST, signals=()),)},
        {"destinations": (replace(DEST, signals=("traces",)),)},
        {"destinations": (replace(DEST, signals=(Signal.LOGS,)),)},
        {"evidence_retention_days": True},
        {"receipt_retention_days": 0},
        {"enabled_exports": (("missing", "metrics"),)},
    ],
)
def test_invalid_profile_candidate_cannot_gain_authority(change):
    baseline = ProfilePolicy(destinations=(DEST,))
    file = PolicyFile(profiles={"personal-local": baseline})
    with pytest.raises(PolicyError) as caught:
        validate_profile(replace(baseline, **change), file)
    assert caught.value.condition == "invalid_policy"
    assert file.profiles["personal-local"].enabled_exports == ()
    accepted = validate_profile(replace(baseline, enabled_exports=(("otel", "metrics"),)), file)
    assert accepted.enabled_exports == (("otel", "metrics"),)


@pytest.mark.parametrize("field", ["session", "task", "project"])
@pytest.mark.parametrize("value", ["", "x\0y", "x" * 5000])
def test_invalid_scope_filter_is_rejected_before_any_read(field: str, value: str, tmp_path: Path):
    before = tree_state(tmp_path)
    with pytest.raises(PolicyError) as caught:
        build_scope(
            session=value if field == "session" else None,
            task=value if field == "task" else None,
            project=value if field == "project" else None,
        )
    assert caught.value.condition == "invalid_filter" and caught.value.exit_code == 2
    assert tree_state(tmp_path) == before


@pytest.mark.parametrize(
    ("fields", "expected", "suffix"),
    [
        (
            {"blocked_reason": "policy_blocked", "pending_batches": 2},
            "blocked",
            "--preview --profile acme",
        ),
        ({"unknown_batches": 2}, "unknown", "--check --profile acme"),
        ({"pending_batches": 2}, "pending", "metrics status --profile acme"),
        ({}, "waiting", "--preview --profile acme"),
    ],
)
def test_delivery_states_keep_profile_and_real_attempt_ages(fields, expected, suffix):
    state = acks.Delivery("otel", Signal.METRICS, last_attempt_at="2026-10-12T12:00:00Z", **fields)
    line = service_delivery._line("acme", "otel", "metrics", False, state, NOW)
    assert line.state == expected and not line.enabled
    assert line.attempt_age_seconds == 30 and line.acked_age_seconds is None
    assert line.action is not None and line.action.command.endswith(suffix)
    assert line.pending_batches == fields.get("pending_batches", 0)
    assert line.unknown_batches == fields.get("unknown_batches", 0)
    assert line.acked_generation is None


def test_absent_ack_age_is_unknown_and_future_attempt_is_clamped():
    state = acks.Delivery(
        "otel", Signal.METRICS, last_acked_generation=2, last_attempt_at="2026-10-13T00:00:00Z"
    )
    line = service_delivery._line("acme", "otel", "metrics", True, state, NOW)
    assert line.state == "acknowledged" and line.action is None
    assert line.acked_age_seconds is None and line.attempt_age_seconds == 0
    assert line.acked_generation == 2


def test_unreadable_delivery_is_not_waiting_and_is_never_repaired(tmp_path: Path):
    profile = ProfilePolicy(
        profile_id="acme", destinations=(DEST,), enabled_exports=(("otel", "metrics"),)
    )
    path = write_delivery("acme", "otel", "metrics", pending_batches=4)
    path.write_text("{broken")
    before = tree_state(tmp_path / "data")
    (line,) = service_delivery.read_delivery_lines("acme", profile, NOW)
    assert line.state == "unreadable" and line.enabled
    assert line.action is not None and line.action.command == "ssgrep metrics status --profile acme"
    assert tree_state(tmp_path / "data") == before
    path.unlink()
    acks.update_delivery("acme", "otel", Signal.METRICS, lambda d: replace(d, pending_batches=4))
    (recovered,) = service_delivery.read_delivery_lines("acme", profile, NOW)
    assert recovered.state == "pending" and recovered.pending_batches == 4


def test_sending_policy_failure_preserves_local_status_but_requires_policy_repair(tmp_path: Path):
    policy.enable_collection("personal-local")
    path = policy.policy_path()
    original = path.read_bytes()
    document = json.loads(original)
    document["sending_paused"] = {"personal-local": "scope_expansion"}
    path.write_text(json.dumps(document))
    before = tree_state(tmp_path / "data")
    status = service_status.read_status(now=NOW)
    assert status.sending_paused_reason == "scope_expansion"
    assert status.policy.collection_enabled and not status.policy.paused
    # Repair needs an operator-supplied path: keep its template on the condition,
    # not among directly runnable actions. Local collection remains actionable.
    (repair,) = (condition for condition in status.conditions if condition.code == "sending_paused")
    assert repair.action is not None
    assert repair.action.command == "ssgrep metrics setup --policy <file>"
    assert status.next_actions[0].command == "ssgrep metrics collect"
    assert all("<" not in action.command for action in status.next_actions)
    summary = service.read_summary(now=NOW)
    assert any(condition.code == "sending_paused" for condition in summary.conditions)
    assert tree_state(tmp_path / "data") == before
    path.write_text("{broken")
    broken = service.read_summary(now=NOW)
    assert broken.reason == "policy_corrupt" and not broken.has_values
    path.write_bytes(original)
    assert service_status.read_status(now=NOW).sending_paused_reason is None
