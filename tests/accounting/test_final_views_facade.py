"""Committed read views reject mismatched generations and preserve unavailable units."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from ssgrep.accounting import (
    collector_state,
    collector_store,
    metrics,
    policy,
    service,
    service_ledger,
    snapshots,
)
from ssgrep.accounting.metrics_render import Sample, ValidationError, render_openmetrics
from ssgrep.accounting.policy_rules import PolicyError
from ssgrep.accounting.scope import build_scope
from tests.accounting.http_builders import make_snapshot, tree_state
from tests.accounting.snapshot_builders import RECEIPT_ID, write_receipt
from tests.accounting.test_ux_support import NOW, collected_scene, fingerprint


def test_snapshot_ahead_of_ledger_never_exposes_unfiltered_totals(tmp_path: Path):
    collected_scene(tmp_path, pairs=(0,))
    scope = build_scope(runtime="claude")
    before = service.read_summary(scope=scope, now=NOW)
    assert dict(before.measured)["searches"].value == 1
    assert before.generation is not None
    envelope = snapshots.read_generation("personal-local", before.generation)
    assert envelope is not None
    newer = snapshots.build_envelope(
        generation=envelope.generation + 1,
        counter_epoch=envelope.counter_epoch,
        methodology_id=envelope.methodology_id,
        policy_version=envelope.policy_version,
        collected_at=envelope.collected_at,
        coverage_through=envelope.coverage_through,
        source_manifest_digest=envelope.source_manifest_digest,
        windows=envelope.windows,
        coverage=envelope.coverage,
        observations=envelope.observations,
        estimates=envelope.estimates,
        warnings=envelope.warnings,
    )
    snapshots.publish("personal-local", newer)
    state = fingerprint(tmp_path / "data")
    selected = service.read_summary(scope=scope, now=NOW)
    assert not selected.has_values
    assert selected.generation == envelope.generation + 1
    assert any(warning.startswith("filters_unavailable:") for warning in selected.warnings)
    assert service.read_summary(now=NOW).has_values
    assert fingerprint(tmp_path / "data") == state


def test_absent_checkpoint_and_absent_reduction_are_distinct_without_writes(tmp_path: Path):
    assert service_ledger.load_reduction("personal-local") == (None, "no_checkpoint")
    assert not (tmp_path / "data" / "accounting").exists()
    collected_scene(tmp_path, pairs=(0,))
    checkpoint_path = (
        tmp_path / "data" / "accounting" / "personal-local" / "collector" / "checkpoint.json"
    )
    original = checkpoint_path.read_bytes()
    document = json.loads(original)
    document["blobs"].pop(collector_state.REDUCTION)
    checkpoint_path.write_text(json.dumps(document))
    before = fingerprint(tmp_path / "data")
    assert service_ledger.load_reduction("personal-local") == (None, "no_reduction")
    assert fingerprint(tmp_path / "data") == before
    checkpoint_path.write_bytes(original)
    doc, error = service_ledger.load_reduction("personal-local")
    assert doc is not None and doc.executions and error is None


def test_missing_ledger_does_not_erase_readable_execution_state(tmp_path: Path):
    collected_scene(tmp_path, pairs=(0,))
    checkpoint_path = (
        tmp_path / "data" / "accounting" / "personal-local" / "collector" / "checkpoint.json"
    )
    checkpoint = collector_store.load_checkpoint("personal-local")
    assert checkpoint is not None
    original = checkpoint_path.read_bytes()
    document = json.loads(original)
    document["blobs"].pop(collector_state.ledger_blob_name(checkpoint.methodology))
    checkpoint_path.write_text(json.dumps(document))
    before = fingerprint(tmp_path / "data")
    assert service_ledger.load_ledger("personal-local") == (None, "no_ledger")
    reduction, error = service_ledger.load_reduction("personal-local")
    assert reduction is not None and reduction.executions and error is None
    assert fingerprint(tmp_path / "data") == before
    checkpoint_path.write_bytes(original)
    ledger, error = service_ledger.load_ledger("personal-local")
    assert ledger is not None and ledger.rows.periods and error is None


def test_receipt_identity_mismatch_is_unreadable_not_another_receipt(tmp_path: Path):
    root = tmp_path / "data"
    write_receipt(root, "acme")
    path = root / "accounting" / "acme" / "receipts" / f"{RECEIPT_ID}.json"
    original = path.read_bytes()
    raw = json.loads(original)
    raw["receipt_id"] = "rcpt-other"
    path.write_text(json.dumps(raw))
    before = fingerprint(root)
    with pytest.raises(PolicyError) as caught:
        service.read_receipt("acme", RECEIPT_ID)
    assert caught.value.condition == "receipt_unreadable" and caught.value.exit_code == 1
    assert service.read_receipt("personal-local", RECEIPT_ID) is None
    assert fingerprint(root) == before
    path.write_bytes(original)
    assert service.read_receipt("acme", RECEIPT_ID) is not None


def test_nonobject_profile_revokes_collection_authority_without_repair(tmp_path: Path):
    policy.enable_collection("personal-local")
    path = policy.policy_path()
    original = path.read_bytes()
    raw = json.loads(original)
    raw["profiles"]["personal-local"] = []
    path.write_text(json.dumps(raw))
    before = tree_state(tmp_path / "data")
    status = policy.policy_status()
    assert status.error is not None and status.error.condition == "policy_corrupt"
    assert not status.profile.collection_enabled
    assert tree_state(tmp_path / "data") == before
    path.write_bytes(original)
    assert policy.load_profile_policy().collection_enabled


def test_invalid_collection_timestamp_omits_only_timestamp_not_snapshot_metrics():
    envelope = replace(make_snapshot(1), collected_at="2026-02-30T00:00:00Z")
    samples = metrics.snapshot_samples(envelope)
    by_name = {sample.family: sample for sample in samples}
    assert "ssgrep_snapshot_collected_timestamp_seconds" not in by_name
    assert by_name["ssgrep_snapshot_available"].value == 1
    assert by_name["ssgrep_accounting_info"].value == 1
    valid = metrics.snapshot_samples(make_snapshot(1))
    timestamp = next(s for s in valid if s.family == "ssgrep_snapshot_collected_timestamp_seconds")
    assert timestamp.value == 1791806400


@pytest.mark.parametrize(
    "kind", ["counter", "histogram_bucket", "histogram_count", "histogram_sum"]
)
def test_renderer_rejects_wrong_sample_suffix_before_emitting_text(kind: str):
    with pytest.raises(ValidationError) as caught:
        render_openmetrics([Sample("ssgrep_invalid", kind, (), 1)])
    assert caught.value.condition == "sample_name"


def test_checkpoint_cannot_relabel_ledger_as_another_methodology(tmp_path: Path):
    collected_scene(tmp_path, pairs=(0,))
    checkpoint = collector_store.load_checkpoint("personal-local")
    assert checkpoint is not None
    path = tmp_path / "data" / "accounting" / "personal-local" / "collector" / "checkpoint.json"
    original = path.read_bytes()
    raw = json.loads(original)
    wrong = "f" * len(checkpoint.methodology)
    original_name = collector_state.ledger_blob_name(checkpoint.methodology)
    raw["blobs"][collector_state.ledger_blob_name(wrong)] = raw["blobs"].pop(original_name)
    raw["methodology"] = wrong
    path.write_text(json.dumps(raw))
    before = fingerprint(tmp_path / "data")
    assert service_ledger.load_ledger("personal-local") == (None, "state_corrupt")
    assert fingerprint(tmp_path / "data") == before
    path.write_bytes(original)
    ledger, error = service_ledger.load_ledger("personal-local")
    assert ledger is not None and error is None
    assert ledger.methodology == checkpoint.methodology


def test_unsafe_committed_blob_directory_is_not_misclassified_as_corruption(tmp_path: Path):
    collected_scene(tmp_path, pairs=(0,))
    directory = tmp_path / "data" / "accounting" / "personal-local" / "collector" / "state"
    directory.chmod(0o777)
    before = fingerprint(tmp_path / "data")
    assert service_ledger.load_reduction("personal-local") == (None, "storage_unsafe")
    assert service_ledger.load_ledger("personal-local") == (None, "storage_unsafe")
    assert fingerprint(tmp_path / "data") == before
    assert directory.stat().st_mode & 0o777 == 0o777
    directory.chmod(0o700)
    doc, error = service_ledger.load_reduction("personal-local")
    assert error is None and doc is not None and doc.executions
    ledger, error = service_ledger.load_ledger("personal-local")
    assert error is None and ledger is not None and ledger.rows.periods


@pytest.mark.parametrize(
    ("seconds", "quantity"),
    [
        (180, None),
        (181, "3 min"),
        (7199, "119 min"),
        (7200, "2 h"),
        (172799, "47 h"),
        (172800, "2 d"),
    ],
)
def test_stale_age_boundary_preserves_elapsed_quantity_and_refresh_action(
    tmp_path: Path, seconds, quantity
):
    collected_scene(tmp_path, pairs=(0,))
    initial = service.read_summary(now=NOW)
    assert initial.collected_at is not None
    collected = datetime.fromisoformat(initial.collected_at)
    before = fingerprint(tmp_path / "data")
    view = service.read_summary(now=collected + timedelta(seconds=seconds))
    assert view.age_seconds == seconds
    assert view.stale is (seconds > 180)
    stale = [condition for condition in view.conditions if condition.code == "stale"]
    if quantity is None:
        assert stale == []
    else:
        assert len(stale) == 1
        assert quantity in stale[0].cause
        assert stale[0].action is not None
        assert stale[0].action.command == "ssgrep metrics collect"
    assert view.measured == initial.measured and view.scenarios == initial.scenarios
    assert fingerprint(tmp_path / "data") == before


def test_unparseable_snapshot_age_stays_unknown_without_fabricating_staleness(tmp_path: Path):
    collected_scene(tmp_path, pairs=(0,))
    path = tmp_path / "data" / "accounting" / "personal-local" / "snapshots" / "current.json"
    original = path.read_bytes()
    raw = json.loads(original)
    raw["collected_at"] = "2026-02-30T12:00:00Z"
    path.write_text(json.dumps(raw))
    before = fingerprint(tmp_path / "data")
    view = service.read_summary(now=NOW)
    assert view.age_seconds is None and not view.stale
    assert all(condition.code != "stale" for condition in view.conditions)
    assert view.has_values
    assert fingerprint(tmp_path / "data") == before
    path.write_bytes(original)
    recovered = service.read_summary(now=NOW)
    assert recovered.age_seconds is not None
    assert recovered.measured == view.measured and recovered.scenarios == view.scenarios
