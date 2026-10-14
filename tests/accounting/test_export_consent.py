"""Consent logic for ``metrics export``: mode validation, enable/disable, preview (real files)."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from ssgrep.accounting import acks, export_consent as consent, export_spool, policy, snapshots
from ssgrep.accounting.export_consent_preview import build_preview
from ssgrep.accounting.export_model import Batch
from ssgrep.accounting.ledger_contracts import DestinationPolicy, ProfilePolicy, Signal
from ssgrep.accounting.policy_rules import PolicyError
from tests.accounting.http_builders import make_snapshot

PROFILE = "personal-local"
DEST = "org-primary"
OTHER = "org-second"
CANARY = "sk-CANARY-SECRET-0123456789"
ENDPOINT = "http://127.0.0.1:4318/otlp"
NOW = "2026-10-13T10:00:00Z"
SCOPE: dict[str, Any] = {
    "roots": [],
    "endpoint": ENDPOINT,
    "fields_version": "metrics-v1",
    "policy_version": 1,
    "logs_permitted": False,
    "identity": "i",
}


def data_root() -> Path:
    return Path(os.environ["SSGREP_DATA_DIR"])


def tree() -> dict[str, tuple[int, int, bytes | None]]:
    """Every path under the data dir: size, mtime_ns and bytes (names are the keys)."""
    root = data_root()
    if not root.exists():
        return {}
    found: dict[str, tuple[int, int, bytes | None]] = {}
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        body = path.read_bytes() if path.is_file() else None
        found[str(path.relative_to(root))] = (info.st_size, info.st_mtime_ns, body)
    return found


def configure(
    *,
    destinations: tuple[DestinationPolicy, ...] | None = None,
    enabled: tuple[tuple[str, str], ...] = (),
    **overrides: Any,
) -> ProfilePolicy:
    dests = destinations or (
        DestinationPolicy(
            DEST,
            ENDPOINT,
            (Signal.METRICS, Signal.LOGS),
            logs_permitted=True,
            credential_ref="env:SSGREP_ORG_TOKEN",
        ),
        DestinationPolicy(OTHER, ENDPOINT, (Signal.METRICS,)),
    )
    return policy.save_profile(
        ProfilePolicy(
            destinations=dests, enabled_exports=enabled, collection_enabled=True, **overrides
        )
    )


def enabled_pairs() -> tuple[tuple[str, str], ...]:
    return policy.policy_status(PROFILE).profile.enabled_exports


def raising(**kwargs: Any) -> PolicyError:
    arguments: dict[str, Any] = {"destination": DEST, "signal": "metrics", "profile": PROFILE}
    with pytest.raises(PolicyError) as caught:
        consent.validate_modes(**(arguments | kwargs))
    return caught.value


# ---- mode validation happens before anything is read or written --------------------------------


@pytest.mark.parametrize(
    "modes",
    [
        {"preview": True, "check": True},
        {"preview": True, "enable": True},
        {"check": True, "disable": True},
        {"enable": True, "disable": True},
        {"preview": True, "check": True, "enable": True, "disable": True},
    ],
)
def test_more_than_one_mode_is_invalid_and_touches_nothing(modes: dict[str, bool]) -> None:
    error = raising(**modes)
    assert (error.condition, error.exit_code) == ("invalid_mode", 2)
    assert not data_root().exists()


@pytest.mark.parametrize(
    ("kwargs", "condition"),
    [
        ({"destination": None}, "destination_required"),
        ({"destination": ""}, "destination_required"),
        ({"destination": "../etc"}, "invalid_destination"),
        ({"destination": "A B"}, "invalid_destination"),
        ({"signal": "traces"}, "invalid_signal"),
        ({"signal": ""}, "invalid_signal"),
        ({"profile": "../x"}, "invalid_profile"),
    ],
)
def test_bad_arguments_exit_2_before_any_file_is_touched(
    kwargs: dict[str, Any], condition: str
) -> None:
    error = raising(**kwargs)
    assert (error.condition, error.exit_code) == (condition, 2)
    assert not data_root().exists()


def test_valid_arguments_pick_the_one_mode_and_default_to_a_manual_send() -> None:
    sent = consent.validate_modes(destination=DEST, signal="metrics", profile=PROFILE)
    assert (sent.mode, sent.signal) == (consent.SEND, Signal.METRICS)
    for mode in ("preview", "check", "enable", "disable"):
        picked = consent.validate_modes(
            destination=DEST, signal="logs", profile="acme", **{mode: True}
        )
        assert (picked.mode, picked.signal, picked.profile) == (mode, Signal.LOGS, "acme")
    assert not data_root().exists()


# ---- enable plan -------------------------------------------------------------------------------


def test_unknown_destination_and_unpermitted_signal_exit_2_and_change_nothing() -> None:
    configure()
    before = tree()
    for destination, signal, condition in (
        ("nope", Signal.METRICS, "unknown_destination"),
        (OTHER, Signal.LOGS, "signal_not_permitted"),
    ):
        with pytest.raises(PolicyError) as caught:
            consent.plan_enable(PROFILE, destination, signal)
        assert (caught.value.condition, caught.value.exit_code) == (condition, 2)
        with pytest.raises(PolicyError):
            consent.apply_enable(PROFILE, destination, signal)
    assert tree() == before and enabled_pairs() == ()


def test_logs_without_policy_permission_are_refused_before_anything_changes() -> None:
    configure()
    path = policy.policy_path()
    document = json.loads(path.read_text())
    for dest in document["profiles"][PROFILE]["destinations"]:
        if dest["destination_id"] == DEST:
            dest["logs_permitted"] = False
    path.write_text(json.dumps(document))
    before = tree()
    with pytest.raises(PolicyError) as caught:
        consent.plan_enable(PROFILE, DEST, Signal.LOGS)
    assert (caught.value.condition, caught.value.exit_code) == ("logs_not_permitted", 2)
    with pytest.raises(PolicyError):
        consent.apply_enable(PROFILE, DEST, Signal.LOGS)
    assert tree() == before


def test_plan_shows_references_and_numbers_but_never_secret_values(tmp_path: Path) -> None:
    secret_file = tmp_path / "tokens" / "ingest.token"
    secret_file.parent.mkdir()
    secret_file.write_text(CANARY + "\n")
    secret_file.chmod(0o600)
    os.environ["SSGREP_ORG_TOKEN"] = CANARY
    try:
        configure(
            destinations=(
                DestinationPolicy(
                    DEST,
                    "https://collector.example.org:4318/otlp",
                    (Signal.METRICS, Signal.LOGS),
                    logs_permitted=True,
                    credential_ref=f"file:{secret_file}",
                    proxy="http://proxy.example.org:3128",
                ),
                DestinationPolicy(OTHER, ENDPOINT, credential_ref="env:SSGREP_ORG_TOKEN"),
            ),
            evidence_retention_days=14,
        )
        files = consent.plan_enable(PROFILE, DEST, Signal.LOGS)
        env = consent.plan_enable(PROFILE, OTHER, Signal.METRICS)
    finally:
        del os.environ["SSGREP_ORG_TOKEN"]
    assert files.credential_ref == "file:ingest.token"  # basename only, never the directory
    assert env.credential_ref == "env:SSGREP_ORG_TOKEN"
    assert files.endpoint == "https://collector.example.org:4318/otlp"
    assert files.proxy == "proxy.example.org:3128"
    assert files.fields_version == "logs-v1" and env.fields_version == "metrics-v1"
    assert files.retention["evidence_days"] == 14 and files.retention["spool_days"] == 7
    assert files.policy_version == 1 and files.signal is Signal.LOGS
    for plan in (files, env):
        assert CANARY not in repr(plan) and str(secret_file.parent) not in repr(plan)
    joined = " ".join(files.notices).lower()
    for must in ("not a connectivity", "collector accepted", "non-reversible", "tombstones"):
        assert must in joined, must
    assert "tombstones" not in " ".join(env.notices).lower()
    assert any("counters" in f for f in env.fields) and any(
        "ledger records" in f for f in files.fields
    )


def test_planning_is_read_only() -> None:
    configure()
    before = tree()
    plan = consent.plan_enable(PROFILE, DEST, Signal.METRICS)
    assert plan.already_enabled is False and tree() == before


# ---- apply enable / disable --------------------------------------------------------------------


def test_enable_adds_exactly_one_pair_and_is_idempotent() -> None:
    configure(enabled=((DEST, "logs"),))
    assert consent.apply_enable(PROFILE, DEST, Signal.METRICS) is True
    assert enabled_pairs() == ((DEST, "logs"), (DEST, "metrics"))
    stored = policy.policy_path().read_bytes()
    stamp = policy.policy_path().stat().st_mtime_ns
    assert consent.apply_enable(PROFILE, DEST, Signal.METRICS) is False
    assert policy.policy_path().read_bytes() == stored
    assert policy.policy_path().stat().st_mtime_ns == stamp
    assert consent.plan_enable(PROFILE, DEST, Signal.METRICS).already_enabled is True


def test_enable_sends_nothing_and_creates_no_export_state() -> None:
    configure()
    consent.apply_enable(PROFILE, DEST, Signal.METRICS)
    names = {p for p in tree()}
    assert not any(
        n.startswith(("accounting/personal-local/spool", "accounting/personal-local/delivery"))
        for n in names
    )
    assert not any("export" in n for n in names)


def test_disable_affects_only_the_chosen_signal() -> None:
    configure(enabled=((DEST, "logs"), (DEST, "metrics"), (OTHER, "metrics")))
    report = consent.apply_disable(PROFILE, DEST, Signal.METRICS)
    assert report.was_enabled is True
    assert enabled_pairs() == ((DEST, "logs"), (OTHER, "metrics"))
    again = consent.apply_disable(PROFILE, DEST, Signal.METRICS)
    assert again.was_enabled is False and enabled_pairs() == ((DEST, "logs"), (OTHER, "metrics"))


def spool_one(signal: Signal = Signal.METRICS) -> Batch:
    batch = Batch(
        "g0000000001-" + "a" * 16 + "-0000",
        signal,
        1,
        "a" * 32,
        0,
        1,
        "data",
        b'{"resourceMetrics":[]}' if signal is Signal.METRICS else b'{"resourceLogs":[]}',
        0,
    )
    export_spool.write_batch(PROFILE, DEST, signal, batch, created_utc=NOW, scope=SCOPE)
    return batch


def test_disable_preserves_history_spool_and_registry_and_reports_what_is_retained() -> None:
    configure(enabled=((DEST, "metrics"),))
    batch = spool_one()
    acks.update_delivery(
        PROFILE,
        DEST,
        Signal.METRICS,
        lambda d: acks.start_in_flight(acks.collected(d, 1), batch.batch_id, 1, NOW),
    )
    registry = data_root() / "accounting" / PROFILE / "export" / DEST / "records.json"
    registry.parent.mkdir(parents=True, mode=0o700)
    registry.write_text('{"schema":1,"records":{}}')
    owned = ("accounting", "accounting/policy.json", "accounting/.lock")  # written by policy
    before = {k: v for k, v in tree().items() if k not in owned}
    report = consent.apply_disable(PROFILE, DEST, Signal.METRICS)
    after = {k: v for k, v in tree().items() if k not in owned}
    assert after == before, "disable must not touch spool, delivery, registry or tombstones"
    assert report.was_enabled and report.pending_batches == 1
    assert report.in_flight == batch.batch_id
    assert enabled_pairs() == ()


def test_resume_never_reenables_a_disabled_signal_and_keeps_enabled_ones() -> None:
    configure(enabled=((DEST, "logs"), (DEST, "metrics")))
    consent.apply_disable(PROFILE, DEST, Signal.LOGS)
    policy.pause(PROFILE)
    assert consent.consent_state(PROFILE, DEST, Signal.METRICS).state == "paused"
    policy.resume(PROFILE)
    assert enabled_pairs() == ((DEST, "metrics"),)
    assert consent.consent_state(PROFILE, DEST, Signal.LOGS).state == "disabled"
    assert consent.consent_state(PROFILE, DEST, Signal.METRICS).state == "enabled"


# ---- policy shrink / corruption ----------------------------------------------------------------


def test_policy_shrink_is_reported_as_a_usage_refusal_not_a_permissive_default() -> None:
    configure(enabled=((DEST, "metrics"),))
    current = policy.policy_status(PROFILE).profile
    policy.save_profile(replace(current, destinations=(), enabled_exports=()))
    state = consent.consent_state(PROFILE, DEST, Signal.METRICS)
    assert (state.state, state.enabled) == ("unknown_destination", False)
    with pytest.raises(PolicyError) as caught:
        consent.require_target(PROFILE, DEST, Signal.METRICS)
    assert (caught.value.condition, caught.value.exit_code) == ("unknown_destination", 2)


@pytest.mark.parametrize("junk", [b"{nope", b'{"schema": 99}', b"[]"])
def test_corrupt_policy_is_policy_corrupt_exit_1_and_nothing_is_rewritten(junk: bytes) -> None:
    configure(enabled=((DEST, "metrics"),))
    path = policy.policy_path()
    path.write_bytes(junk)
    before = tree()
    state = consent.consent_state(PROFILE, DEST, Signal.METRICS)
    assert state.state == "policy_corrupt" and state.enabled is False
    for action in (
        lambda: consent.require_target(PROFILE, DEST, Signal.METRICS),
        lambda: consent.plan_enable(PROFILE, DEST, Signal.METRICS),
        lambda: consent.apply_enable(PROFILE, DEST, Signal.METRICS),
        lambda: consent.apply_disable(PROFILE, DEST, Signal.METRICS),
    ):
        with pytest.raises(PolicyError) as caught:
            action()
        assert (caught.value.condition, caught.value.exit_code) == ("policy_corrupt", 1)
    assert tree() == before


# ---- preview -----------------------------------------------------------------------------------


def publish_snapshot(generation: int = 1) -> None:
    snapshots.publish(PROFILE, make_snapshot(generation))


def test_preview_without_snapshot_reports_state_and_creates_nothing() -> None:
    configure()
    before = tree()
    view = build_preview(PROFILE, DEST, Signal.METRICS)
    assert (view.state, view.batches) == ("no_snapshot", ())
    assert view.endpoint_host == "127.0.0.1:4318" and view.policy_version == 1
    assert tree() == before


def test_preview_is_exact_sanitized_bytes_and_has_zero_side_effects() -> None:
    configure(enabled=())  # not enabled: preview is still allowed
    publish_snapshot()
    before = tree()
    first = build_preview(PROFILE, DEST, Signal.METRICS)
    second = build_preview(PROFILE, DEST, Signal.METRICS)
    assert tree() == before, "names, sizes, mtimes and bytes of the whole data dir are unchanged"
    assert first.state == "ok" and first.batches and first.ephemeral_key is True
    assert first.fields_version == "metrics-v1" and first.policy_version == 1
    for batch in first.batches:
        assert batch.bytes == len(batch.body)
        assert batch.sha256 == hashlib.sha256(batch.body).hexdigest()
        assert batch.payload()["resourceMetrics"], "parsed payload is the request itself"
    assert first.total_bytes == sum(b.bytes for b in first.batches)
    assert {b.batch_id for b in first.batches} == {b.batch_id for b in second.batches}


def test_preview_is_allowed_while_paused_and_leaves_pause_state_alone() -> None:
    configure(enabled=((DEST, "metrics"),))
    publish_snapshot()
    policy.pause(PROFILE)
    before = tree()
    view = build_preview(PROFILE, DEST, Signal.METRICS)
    assert view.state == "ok" and tree() == before
    assert policy.policy_status(PROFILE).profile.paused is True


def test_preview_contains_no_credential_header_or_key_material(tmp_path: Path) -> None:
    os.environ["SSGREP_ORG_TOKEN"] = CANARY
    try:
        configure()
        publish_snapshot()
        view = build_preview(PROFILE, DEST, Signal.METRICS)
    finally:
        del os.environ["SSGREP_ORG_TOKEN"]
    blob = b"".join(b.body for b in view.batches) + repr(view).encode()
    assert CANARY.encode() not in blob and b"Authorization" not in blob and b"Bearer" not in blob
    assert view.ephemeral_key is True
    assert not (data_root() / "accounting" / PROFILE / "export" / "pseudonym.key").exists()


def test_preview_of_logs_without_ledger_reports_a_state_not_an_error() -> None:
    configure()
    publish_snapshot()
    before = tree()
    view = build_preview(PROFILE, DEST, Signal.LOGS)
    assert view.state in {"no_records", "no_changes", "ok"} and tree() == before


def test_preview_of_a_refused_target_is_a_usage_error() -> None:
    configure()
    with pytest.raises(PolicyError) as caught:
        build_preview(PROFILE, "ghost", Signal.METRICS)
    assert (caught.value.condition, caught.value.exit_code) == ("unknown_destination", 2)
