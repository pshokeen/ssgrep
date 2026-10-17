"""Consent logic for ``metrics export``: mode validation, enable/disable, preview (real files)."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from ssgrep.accounting.collect import store as collector_store
from ssgrep.accounting.export import acks, consent, spool as export_spool
from ssgrep.accounting.export.consent import build_preview
from ssgrep.accounting.export.otlp import Batch
from ssgrep.accounting.export.projection import REASON_CODES, safe_reason
from ssgrep.accounting.ledger.ledger_contracts import DestinationPolicy, ProfilePolicy, Signal
from ssgrep.accounting.privacy import policy
from ssgrep.accounting.privacy.policy import PolicyError
from ssgrep.accounting.views import snapshots
from tests.accounting.export.http_builders import make_snapshot, synthetic_current_checkpoint

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
    assert files.fields_version == "logs-v2" and env.fields_version == "metrics-v2"
    assert files.retention["evidence_days"] == 14 and files.retention["spool_days"] == 7
    assert files.policy_version == 1 and files.signal is Signal.LOGS
    for plan in (files, env):
        assert CANARY not in repr(plan) and str(secret_file.parent) not in repr(plan)
    joined = " ".join(files.notices).lower()
    for must in ("not a connectivity", "collector accepted", "non-reversible", "tombstones"):
        assert must in joined, must
    assert "tombstones" not in " ".join(env.notices).lower()
    assert f"`ssgrep savings export check {OTHER}`" in " ".join(env.notices)
    assert f"`ssgrep savings export check {DEST} --logs`" in " ".join(files.notices)
    assert "--check" not in " ".join((*files.notices, *env.notices))
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


def test_enable_sends_nothing_and_creates_only_consent_state() -> None:
    configure()
    consent.apply_enable(PROFILE, DEST, Signal.METRICS)
    names = set(tree())
    assert "accounting/personal-local/export/org-primary/consent-metrics.json" in names
    assert not any(
        n.startswith(("accounting/personal-local/spool", "accounting/personal-local/delivery"))
        for n in names
    )


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


def test_disable_counts_only_unacknowledged_spool_batches() -> None:
    configure(enabled=((DEST, "metrics"),))
    queued = spool_one()
    settled = Batch(
        "g0000000001-" + "b" * 16 + "-0001",
        Signal.METRICS,
        1,
        "a" * 32,
        1,
        2,
        "data",
        b'{"resourceMetrics":[]}',
        0,
    )
    export_spool.write_batch(PROFILE, DEST, Signal.METRICS, settled, created_utc=NOW, scope=SCOPE)
    export_spool.update_meta(
        PROFILE, DEST, Signal.METRICS, settled.batch_id, lambda meta: replace(meta, state="acked")
    )
    assert len(export_spool.list_batches(PROFILE, DEST, Signal.METRICS)) == 2
    report = consent.apply_disable(PROFILE, DEST, Signal.METRICS)
    assert report.pending_batches == 1 and queued.batch_id != settled.batch_id
    export_spool.update_meta(
        PROFILE, DEST, Signal.METRICS, queued.batch_id, lambda meta: replace(meta, state="acked")
    )
    assert consent.apply_disable(PROFILE, DEST, Signal.METRICS).pending_batches == 0


def test_resume_never_reenables_a_disabled_signal_and_keeps_enabled_ones() -> None:
    configure(enabled=((DEST, "logs"), (DEST, "metrics")))
    consent.apply_enable(PROFILE, DEST, Signal.LOGS)
    consent.apply_enable(PROFILE, DEST, Signal.METRICS)
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
    current = policy.policy_status(PROFILE).profile
    collector_store.commit_checkpoint(PROFILE, synthetic_current_checkpoint(PROFILE, current))
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
    assert first.fields_version == "metrics-v2" and first.policy_version == 1
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


def test_preview_reports_policy_block_reason_for_disallowed_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import datetime

    from ssgrep.accounting.export import projection

    configure()
    publish_snapshot()
    preview_candidate = projection.preview_candidate

    def revoke_policy_then_preview(
        profile: str,
        destination: str,
        signal: Signal,
        *,
        now: datetime | None = None,
    ) -> projection.PreviewCandidate:
        configure(destinations=(DestinationPolicy(DEST, ENDPOINT, (Signal.METRICS,)),))
        return preview_candidate(profile, destination, signal, now=now)

    monkeypatch.setattr(projection, "preview_candidate", revoke_policy_then_preview)
    view = build_preview(PROFILE, DEST, Signal.LOGS)

    assert view.state == "policy_blocked"
    assert view.reason == "signal_not_permitted"


@pytest.mark.parametrize("code", sorted(REASON_CODES))
def test_every_known_code_passes_unchanged(code: str) -> None:
    assert safe_reason(code) == code
    assert safe_reason(f"incomplete:{code}") == f"incomplete:{code}"


@pytest.mark.parametrize(
    "shipped",
    [
        "downstream_exposure_unobserved",
        "mixed_shell_output_unattributable",
        "lookup_work_seconds_not_modeled",
        "output_shared_by_several_executions",
        "no_runtime_exposure_joined",
        "no_render_event",
        "exact_tokens_unavailable",
        "source_seconds_unavailable",
        "debit_chars_unavailable",
        "conflicting_duplicate_debit",
        "unit_not_in_period",
        "no_closed_periods",
        "history_unreadable",
        "malformed_quantity",
        "no_tariff",
        "provider_cost_missing",
        "monotonic_clock_reversed",
        "no_dependency_evidence",
        "invalid_timing_in_gaps",
        "tokenizer_file_missing",
        "tokenizer_file_digest_mismatch",
        "visible_bytes_missing_for_some_executions",
        "rendered_bytes_missing_for_some_executions",
        "not_estimated",
        "scenario_missing",
    ],
)
def test_reasons_the_reducers_actually_produce_are_in_the_vocabulary(shipped: str) -> None:
    assert safe_reason(shipped) == shipped


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("no_applicable_tariff:input_uncached", "no_applicable_tariff"),
        ("tokenizer_load_failed:ValueError", "tokenizer_load_failed"),
        ("tokenizer_library_not_installed:tokenizers", "tokenizer_library_not_installed"),
        ("tokenizer_version_mismatch:0.1!=0.2", "tokenizer_version_mismatch"),
        ("unsupported_tokenizer_library:mylib", "unsupported_tokenizer_library"),
    ],
)
def test_known_codes_with_free_value_suffixes_drop_the_suffix(raw: str, expected: str) -> None:
    assert safe_reason(raw) == expected


@pytest.mark.parametrize(
    "free",
    [
        "sk-canary-7f3a91c2",
        "deadbeef" * 5,
        "Secret Reason! /home/u",
        "incomplete:sk-canary-7f3a91c2",
        "no_tariff:sk-canary",  # only suffix-family codes may carry a suffix
        "",
        None,
        7,
    ],
)
def test_anything_outside_the_vocabulary_collapses_to_unspecified(free: object) -> None:
    out = safe_reason(free)
    assert "canary" not in out and "deadbeef" not in out and "home" not in out
    assert out == "unspecified" or out in REASON_CODES or out.startswith("incomplete:")


def test_incomplete_members_are_normalized_sorted_and_deduplicated() -> None:
    known = (
        "incomplete:source_seconds_unavailable,exact_tokens_unavailable,source_seconds_unavailable"
    )
    assert safe_reason(known) == "incomplete:exact_tokens_unavailable,source_seconds_unavailable"
    suffixed = "incomplete:no_applicable_tariff:output,no_applicable_tariff:input_uncached"
    assert safe_reason(suffixed) == "incomplete:no_applicable_tariff"


def test_one_unknown_member_collapses_the_whole_incomplete_reason() -> None:
    mixed = "incomplete:source_seconds_unavailable,sk-canary-1,exact_tokens_unavailable"
    assert safe_reason(mixed) == "unspecified"
    assert safe_reason("incomplete:") == "unspecified"


@pytest.mark.parametrize("bad", ["sk-CANARY-1234", "2026-13-40T00:00:00Z", "x", "2026-10-12"])
def test_unix_nano_errors_never_quote_the_value(bad: str) -> None:

    from ssgrep.accounting.export.projection import unix_nano

    with pytest.raises(ValueError, match="invalid_timestamp") as caught:
        unix_nano(bad)
    assert bad not in str(caught.value) and caught.value.__suppress_context__


def test_consent_corruption_blocks_enable_without_replacing_evidence() -> None:
    configure()
    assert consent.apply_enable(PROFILE, DEST, Signal.METRICS) is True
    path = data_root() / "accounting" / PROFILE / "export" / DEST / "consent-metrics.json"
    path.write_bytes(b'{"schema":')
    before = tree()

    assert consent.approval_reason(PROFILE, DEST, Signal.METRICS) == "consent_state_unavailable"
    with pytest.raises(PolicyError) as caught:
        consent.apply_enable(PROFILE, DEST, Signal.METRICS)

    assert caught.value.condition == "consent_state_unavailable"
    assert tree() == before


@pytest.mark.parametrize(
    ("checkpoint", "expected"),
    [
        (None, consent.MigrationView("legacy", "identity_migration_required")),
        (
            collector_store.Checkpoint(PROFILE),
            consent.MigrationView("legacy", "identity_migration_required"),
        ),
        (
            collector_store.Checkpoint(
                PROFILE,
                identity_migration={"target": consent.NATIVE_IDENTITY, "status": "pending"},
            ),
            consent.MigrationView("pending", "identity_migration_pending"),
        ),
        (
            collector_store.Checkpoint(
                PROFILE,
                identity_method="old",
                identity_migration={"target": consent.NATIVE_IDENTITY, "status": "complete"},
            ),
            consent.MigrationView("unavailable", "identity_migration_unavailable"),
        ),
        (
            collector_store.Checkpoint(
                PROFILE,
                identity_method=consent.NATIVE_IDENTITY,
                identity_migration={"target": consent.NATIVE_IDENTITY, "status": "complete"},
            ),
            consent.MigrationView("current"),
        ),
        (
            collector_store.Checkpoint(PROFILE, identity_migration=cast(Any, None)),
            consent.MigrationView("unavailable", "identity_migration_unavailable"),
        ),
        (
            collector_store.Checkpoint(
                PROFILE, identity_migration={"target": consent.NATIVE_IDENTITY, "status": "future"}
            ),
            consent.MigrationView("unavailable", "identity_migration_unavailable"),
        ),
    ],
)
def test_migration_state_requires_a_well_formed_current_identity_marker(checkpoint, expected):
    assert consent.migration_of(checkpoint) == expected


def test_migration_view_fails_closed_when_checkpoint_cannot_be_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unreadable(_profile: str):
        raise ValueError("malformed checkpoint")

    monkeypatch.setattr(collector_store, "load_checkpoint", unreadable)

    assert consent.migration_view(PROFILE) == consent.MigrationView(
        "unavailable", "identity_migration_unavailable"
    )


def test_consent_target_validation_rejects_invalid_storage_names() -> None:
    with pytest.raises(ValueError, match="invalid_consent_target"):
        consent.approval_reason("../outside", DEST, Signal.METRICS)


@pytest.mark.parametrize("state", ["empty", "stray", "corrupt-pointer", "current"])
def test_empty_snapshot_store_proves_only_a_genuinely_empty_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    directory = tmp_path / "snapshots"
    directory.mkdir()
    if state == "stray":
        (directory / "unrelated.json").write_text("{}")
    elif state == "corrupt-pointer":
        pointer = directory / snapshots.CURRENT_NAME
        pointer.write_bytes(b"{invalid")
        pointer.chmod(0o600)
    elif state == "current":
        pointer = directory / snapshots.CURRENT_NAME
        pointer.write_bytes(snapshots.encode(make_snapshot(1)))
        pointer.chmod(0o600)

    monkeypatch.setattr(
        consent.storage,
        "open_profile_dir",
        lambda *_args, **_kwargs: os.open(directory, os.O_RDONLY),
    )
    if state == "corrupt-pointer":
        with pytest.raises(ValueError):
            consent._empty_snapshot_store(PROFILE)
    else:
        assert consent._empty_snapshot_store(PROFILE) is (state == "empty")


def test_preview_reports_identity_migration_instead_of_claiming_no_snapshot() -> None:
    configure()
    collector_store.commit_checkpoint(PROFILE, collector_store.Checkpoint(PROFILE))

    view = build_preview(PROFILE, DEST, Signal.METRICS)

    assert view.state == view.reason == "identity_migration_required"
    assert view.batches == () and view.generation is None


def test_approval_fails_closed_when_locked_directory_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import contextmanager

    configure()
    before = tree()

    @contextmanager
    def no_directory(*_args, **_kwargs):
        yield None

    monkeypatch.setattr(export_spool, "locked", no_directory)

    with pytest.raises(ValueError, match="consent_state_unavailable"):
        consent.approve(PROFILE, DEST, Signal.METRICS)

    assert tree() == before


@pytest.mark.parametrize(
    "result, condition, command",
    [
        (
            ("failed", "identity_migration_required"),
            "identity_migration_required",
            "savings collect",
        ),
        (("revoked", "policy_revoked"), "policy_revoked", "savings status"),
    ],
)
def test_cycle_verdict_explains_migration_and_revocation_holds(result, condition, command):
    from ssgrep.accounting.export.delivery import CycleResult

    state, reason = result
    verdict = consent.classify_cycle(CycleResult(DEST, Signal.METRICS, state, reason), PROFILE)

    assert not verdict.ok
    assert verdict.condition == condition
    assert verdict.next_command is not None
    assert command in verdict.next_command
    assert "nothing was sent" in verdict.message.lower()


@pytest.mark.parametrize("reason", ["unauthorized", "forbidden", "credential_invalid"])
def test_credential_block_guidance_says_only_a_changed_credential_clears_it(reason: str) -> None:
    from ssgrep.accounting.export.delivery import CycleResult

    verdict = consent.classify_cycle(
        CycleResult(DEST, Signal.LOGS, "blocked", f"blocked_{reason}", blocked_reason=reason),
        PROFILE,
    )
    assert verdict.condition == "export_blocked" and not verdict.ok
    assert f"Sending is blocked ({reason})." in verdict.message
    assert "check does not clear it" in verdict.message
    assert "credential value, endpoint or proxy changes" in verdict.message
    assert "--check" not in verdict.message
    assert verdict.next_command == f"ssgrep savings export check {DEST} --logs"
    held = consent.classify_cycle(
        CycleResult(
            DEST,
            Signal.METRICS,
            "blocked",
            "blocked_policy_blocked",
            blocked_reason="policy_blocked",
        ),
        PROFILE,
    )
    assert held.message == "Sending is blocked (policy_blocked)."
