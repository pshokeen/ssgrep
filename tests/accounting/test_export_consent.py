"""Consent logic for ``metrics export``: mode validation, enable/disable, preview (real files)."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from ssgrep.accounting import (
    acks,
    collector_state,
    collector_store,
    export_authority,
    export_consent as consent,
    export_consent_fields,
    export_consent_preview,
    export_creds,
    export_spool,
    plainfile,
    policy,
    snapshots,
    storage,
)
from ssgrep.accounting.export_consent_preview import BatchView, PreviewView, build_preview
from ssgrep.accounting.export_model import Batch
from ssgrep.accounting.ledger_contracts import DestinationPolicy, ProfilePolicy, Signal
from ssgrep.accounting.policy_rules import PolicyError
from ssgrep.cli import export_render
from tests.accounting import delivery_fixtures as fx
from tests.accounting.delivery_fixtures import M
from tests.accounting.http_builders import make_snapshot, synthetic_current_checkpoint
from tests.accounting.lab_receiver import LabReceiver
from tests.accounting.test_savings_cli import Run, document

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
    current = policy.save_profile(
        ProfilePolicy(
            destinations=dests, enabled_exports=enabled, collection_enabled=True, **overrides
        )
    )
    for destination, signal in enabled:
        consent.apply_enable(PROFILE, destination, Signal(signal))
    return current


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


def test_enable_persists_only_consent_without_wire_or_delivery_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_credential_read(*args: Any, **kwargs: Any) -> None:
        pytest.fail("enabling export must not resolve ingest credentials")

    with LabReceiver.start() as lab:
        configure(
            destinations=(DestinationPolicy(DEST, lab.url, credential_ref="env:SSGREP_ORG_TOKEN"),)
        )
        monkeypatch.setattr(export_creds, "resolve", unexpected_credential_read)
        assert consent.apply_enable(PROFILE, DEST, Signal.METRICS) is True
        assert lab.requests == []
    assert export_consent_fields.approval_reason(PROFILE, DEST, Signal.METRICS) is None
    assert export_consent_fields.approval_reason(PROFILE, DEST, Signal.LOGS) == (
        "consent_reapproval_required"
    )
    export_root = data_root() / "accounting" / PROFILE
    assert not (export_root / "spool").exists()
    assert not (export_root / "delivery").exists()
    assert not (export_root / "export" / "pseudonym.key").exists()
    assert not list(export_root.rglob("records.json"))
    assert not list(export_root.rglob("*.batch"))


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
    registry.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
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
    current = policy.load_profile_policy(PROFILE)
    checkpoint = synthetic_current_checkpoint(PROFILE, current)
    collector_store.commit_checkpoint(PROFILE, checkpoint)
    snapshots.publish(PROFILE, make_snapshot(generation))


@pytest.mark.parametrize("signal", [Signal.METRICS, Signal.LOGS])
@pytest.mark.parametrize("snapshot_directory", ["absent", "empty"])
def test_preview_without_snapshot_reports_state_and_creates_nothing(
    signal: Signal,
    snapshot_directory: str,
) -> None:
    configure()
    if snapshot_directory == "empty":
        os.close(storage.open_profile_dir(PROFILE, "snapshots", create=True))
    before = tree()
    view = build_preview(PROFILE, DEST, signal)
    assert (view.state, view.batches) == ("no_snapshot", ())
    assert view.endpoint_host == "127.0.0.1:4318" and view.policy_version == 1
    assert tree() == before
    assert export_authority.migration_view(PROFILE).reason == "identity_migration_required"
    assert view.ephemeral_key is False


@pytest.mark.parametrize("with_snapshot", [False, True])
def test_preview_of_present_legacy_state_keeps_identity_hold_without_repair(
    with_snapshot: bool,
) -> None:
    configure()
    collector_store.commit_checkpoint(PROFILE, collector_store.Checkpoint(PROFILE))
    if with_snapshot:
        snapshots.publish(PROFILE, make_snapshot(1))
    before = tree()
    view = build_preview(PROFILE, DEST, Signal.METRICS)
    assert (view.state, view.reason, view.batches) == (
        "identity_migration_required",
        "identity_migration_required",
        (),
    )
    assert view.ephemeral_key is False and tree() == before


def test_snapshot_without_checkpoint_is_not_current_export_authority() -> None:
    configure()
    snapshots.publish(PROFILE, make_snapshot(1))
    before = tree()
    view = build_preview(PROFILE, DEST, Signal.METRICS)
    assert view.state == "identity_migration_required" and view.batches == ()
    assert tree() == before


def test_current_identity_without_snapshot_is_no_snapshot_not_migration_required() -> None:
    current = configure()
    collector_store.commit_checkpoint(PROFILE, synthetic_current_checkpoint(PROFILE, current))
    before = tree()
    view = build_preview(PROFILE, DEST, Signal.METRICS)
    assert view.state == "no_snapshot" and view.batches == ()
    assert export_authority.migration_view(PROFILE).state == "current"
    assert tree() == before


@pytest.mark.parametrize(
    ("kind", "condition"),
    [
        ("checkpoint_corrupt", "state_corrupt"),
        ("checkpoint_unsafe", "state_corrupt"),
        ("checkpoint_directory_unsafe", "storage_unsafe"),
        ("snapshot_corrupt", "preview_unavailable"),
        ("snapshot_unsafe", "preview_unavailable"),
        ("snapshot_directory_unsafe", "preview_unavailable"),
    ],
)
def test_unreadable_preview_state_is_not_fresh_empty_setup(
    run: Run,
    kind: str,
    condition: str,
) -> None:
    configure()
    if kind.startswith("checkpoint"):
        parts = ("collector", "checkpoint.json")
    else:
        parts = ("snapshots", "current.json")
    plainfile.write_private(run.data, "accounting", PROFILE, *parts, data=b"{")
    target = run.data.joinpath("accounting", PROFILE, *parts)
    if kind.endswith("directory_unsafe"):
        target.parent.chmod(0o755)
    elif kind.endswith("unsafe"):
        target.chmod(0o644)
    before = tree()
    modes = (target.stat().st_mode, target.parent.stat().st_mode)
    result = run("metrics", "export", "--preview", "--destination", DEST, "--json")
    assert result.returncode == 1
    assert document(result)["condition"] == condition
    assert tree() == before
    assert (target.stat().st_mode, target.parent.stat().st_mode) == modes


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


# ---- preview and consent control under damaged or racing state ---------------------------------


@pytest.fixture
def run(tmp_path: Path) -> Run:
    home = tmp_path / "home"
    home.mkdir()
    return Run(home, tmp_path / "data")


def flat(text: str) -> str:
    """Collapse wrapping (and shell ``\\`` continuations) so phrase assertions ignore breaks."""
    return " ".join(text.replace(" \\\n", " ").split())


def preview(bodies: tuple[bytes, ...]) -> PreviewView:
    batches = tuple(
        BatchView(f"b-{i}", len(body), i + 1, hashlib.sha256(body).hexdigest(), body)
        for i, body in enumerate(bodies)
    )
    return PreviewView(
        "personal-local",
        DEST,
        Signal.METRICS,
        "ok",
        None,
        1,
        "localhost",
        "metrics-v1",
        False,
        7,
        "a" * 16,
        0,
        (),
        batches,
    )


@pytest.mark.parametrize(
    ("kind", "condition"),
    [("key", "preview_unavailable"), ("ledger", "state_corrupt"), ("registry", "registry_invalid")],
)
def test_export_preview_refuses_damaged_private_state_without_repairing_it(
    run: Run,
    kind: str,
    condition: str,
):
    configure()
    publish_snapshot()
    fx.commit_ledger(1)
    signal = Signal.METRICS if kind == "key" else Signal.LOGS
    assert build_preview(PROFILE, DEST, signal).batches  # reach the damaged-state read path
    checkpoint = collector_store.load_checkpoint(PROFILE)
    assert checkpoint is not None
    ledger_blob = checkpoint.blobs[collector_state.ledger_blob_name(checkpoint.methodology)]
    parts = {
        "key": ("export", "pseudonym.key"),
        "ledger": ("collector", "state", ledger_blob),
        "registry": ("export", DEST, "records.json"),
    }[kind]
    plainfile.write_private(run.data, "accounting", PROFILE, *parts, data=b"{")
    target = run.data.joinpath("accounting", PROFILE, *parts)
    before = target.stat().st_mtime_ns
    result = run(
        "metrics",
        "export",
        "--preview",
        "--destination",
        DEST,
        "--signal",
        "metrics" if kind == "key" else "logs",
        "--json",
    )
    assert result.returncode == 1
    error = document(result)
    assert error["condition"] == condition
    assert error["command"] == "ssgrep metrics status"
    assert target.read_bytes() == b"{" and target.stat().st_mtime_ns == before
    assert not (run.data / "accounting" / PROFILE / "spool").exists()


@pytest.mark.parametrize("cap", [0, 12, 24])
def test_export_preview_print_cap_preserves_whole_first_request_and_summarizes_rest(cap: int):
    bodies = tuple(
        json.dumps({"value": n}, separators=(",", ":")).encode("ascii") for n in (1, 2, 3)
    )
    view = preview(bodies)
    lines = export_render.preview_payload_lines(view, cap)
    shown = [line.encode("ascii") for line in lines if line.startswith("{")]
    count = max(1, min(len(bodies), cap // len(bodies[0])))
    assert shown == list(bodies[:count])
    for batch in view.batches[count:]:
        assert any(
            batch.batch_id in line and batch.sha256 in line and f"{batch.bytes} bytes" in line
            for line in lines
        )
    if count < len(bodies):
        assert any("--json" in line for line in lines)


def test_preview_rechecks_policy_after_target_validation_and_exposes_no_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with LabReceiver.start() as lab:
        configured = fx.configure(lab.url)
        fx.publish(1)
        real_require = export_consent_preview.require_target

        def revoke_after_validation(*args, **kwargs):
            result = real_require(*args, **kwargs)
            fx.amend(destinations=(), enabled_exports=())
            return result

        with monkeypatch.context() as race:
            race.setattr(export_consent_preview, "require_target", revoke_after_validation)
            preview = export_consent_preview.build_preview(PROFILE, DEST, M)
        assert preview.state == "policy_blocked" and preview.reason is not None
        assert preview.batches == () and preview.total_bytes == 0 and lab.requests == []
        policy.save_profile(configured)
        control = export_consent_preview.build_preview(PROFILE, DEST, M)
        assert control.state == "ok" and control.total_bytes > 0
        assert fx.cycle().acked == 1


def test_revoked_cycle_verdict_retains_actionable_failure_and_reenable_control() -> None:
    with LabReceiver.start() as lab:
        configured = fx.configure(lab.url)
        fx.publish(1)
        fx.cycle(deps=fx.deps(max_batches=0))
        fx.amend(destinations=(), enabled_exports=())
        result = fx.cycle(seconds=1)
        verdict = consent.classify_cycle(result, PROFILE)
        assert result.state == "revoked" and result.invalidated_batches == 1
        assert not verdict.ok and verdict.code != 0
        assert verdict.condition == result.reason and verdict.next_command is not None
        assert fx.state().last_acked_generation is None and lab.requests == []
        policy.save_profile(configured)
        fx.publish(2)
        assert fx.cycle(seconds=2).acked == 1 and len(lab.requests) == 1
