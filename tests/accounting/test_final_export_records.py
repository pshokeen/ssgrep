"""Record provenance, registry limits, and receiver state remain truthful under damage."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from ssgrep.accounting import (
    collector_state,
    collector_store,
    delivery_registry,
    export_records,
    export_registry,
    export_spool,
    otlp,
    policy,
    receiver_contract,
)
from tests.accounting import delivery_fixtures as fx
from tests.accounting.delivery_fixtures import DEST, PROFILE, L
from tests.accounting.lab_receiver import LabReceiver
from tests.accounting.test_export_records import KEY, project, rows_a, rows_b


@pytest.mark.parametrize("damage", ["checkpoint-scope", "methodology", "state-symlink"])
def test_ledger_provenance_damage_refuses_export_without_mutating_ledger_and_recovers(
    damage: str,
    tmp_path: Path,
) -> None:
    with LabReceiver.start() as lab:
        configured = fx.configure(lab.url, signals=(L,), logs_permitted=True, enabled=(L,))
        fx.commit_ledger(1)
        checkpoint = collector_store.load_checkpoint(PROFILE)
        assert checkpoint is not None
        original_checkpoint = checkpoint.to_dict()
        expected = export_records.ledger_rows(PROFILE)
        scope = policy.collection_scope(configured)
        state_dir = fx.profile_root() / "collector" / "state"
        saved = tmp_path / "saved-state"
        if damage == "checkpoint-scope":
            checkpoint.collection_scope = {**scope, "digest": "f" * 64}
            collector_store.commit_checkpoint(PROFILE, checkpoint)
        elif damage == "methodology":
            state = collector_state.load_state(PROFILE, checkpoint, checkpoint.methodology)
            checkpoint.methodology = "f" * len(checkpoint.methodology)
            collector_state.save_state(PROFILE, state, checkpoint.methodology, checkpoint.blobs)
            collector_store.commit_checkpoint(PROFILE, checkpoint)
        else:
            state_dir.rename(saved)
            state_dir.symlink_to(saved, target_is_directory=True)
        before = fx.tree()
        with pytest.raises(export_records.RecordsUnavailable) as error:
            export_records.ledger_rows(PROFILE, collection_scope=scope)
        condition = {
            "checkpoint-scope": "ineligible_scope",
            "methodology": "state_corrupt",
            "state-symlink": "storage_unsafe",
        }[damage]
        assert error.value.condition == condition
        assert fx.tree() == before and lab.requests == []
        if damage == "state-symlink":
            state_dir.unlink()
            saved.rename(state_dir)
        collector_store.commit_checkpoint(
            PROFILE, collector_store.Checkpoint.from_dict(original_checkpoint)
        )
        assert export_records.ledger_rows(PROFILE, collection_scope=scope) == expected
        fx.publish(1)
        assert fx.cycle(L).acked == 1 and len(lab.requests) == 1


@pytest.mark.parametrize("damage", ["period-format", "period-date", "count-range", "conflict"])
def test_projection_omits_bad_leaf_without_inventing_credit_and_valid_neighbor_survives(
    damage: str,
) -> None:
    rows = rows_a()
    (row,) = rows.periods
    (leaf,) = row.leaves
    if damage == "period-format":
        changed = replace(row, period="not-a-day")
    elif damage == "period-date":
        changed = replace(row, period="2026-02-30")
    elif damage == "count-range":
        changed = replace(row, leaves=(replace(leaf, debit_raw=1 << 63),))
    else:
        changed = replace(row, leaves=(leaf, replace(leaf, debit_raw=leaf.debit_raw + 1)))
    candidates, omitted = export_records.project_candidates(
        replace(rows, periods=(changed,)), key=KEY, destination=DEST
    )
    assert candidates == [] and omitted == (2 if damage == "conflict" else 1)
    valid, omitted = export_records.project_candidates(rows, key=KEY, destination=DEST)
    assert omitted == 0 and len(valid) == 1
    planned = export_records.plan_revisions(export_records.Registry(), valid)
    key = receiver_contract.Key("tenant", "device", planned[0].record_id)
    received, outcome = receiver_contract.apply(
        receiver_contract.ReceiverState(), key, planned[0].body
    )
    assert outcome is receiver_contract.Outcome.APPLIED_NEW
    assert received.current[key].body["scenarios"]["central"]["net"]["value"] == -60


@pytest.mark.parametrize("bound", ["decoded-bytes", "encoded-bytes", "queued-revisions"])
def test_registry_size_limit_cannot_replace_previously_committed_record(
    bound: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planned = export_records.plan_revisions(export_records.Registry(), project(rows_a()))
    export_records.mark_queued(PROFILE, DEST, planned, "batch-first")
    path = fx.profile_root() / "export" / DEST / "records.json"
    original = path.read_bytes()
    registry = fx.registry()
    entry = registry.entries[planned[0].record_id]
    if bound == "decoded-bytes":
        with monkeypatch.context() as limit:
            limit.setattr(export_registry, "MAX_REGISTRY_BYTES", len(original) - 1)
            with pytest.raises(export_records.RegistryCorrupt) as error:
                export_registry.decode(original)
    elif bound == "encoded-bytes":
        with monkeypatch.context() as limit:
            limit.setattr(export_registry, "MAX_REGISTRY_BYTES", len(original) - 1)
            with pytest.raises(export_records.RegistryCorrupt) as error:
                export_registry.encode(registry)
    else:
        states = tuple(
            replace(entry.queued[0], revision=i + 1) for i in range(export_registry.MAX_QUEUED + 1)
        )
        enlarged = replace(registry, entries={planned[0].record_id: replace(entry, queued=states)})
        with pytest.raises(export_records.RegistryCorrupt) as error:
            export_registry.update_registry(PROFILE, DEST, lambda current: enlarged)
    assert error.value.condition == "registry_oversize"
    assert path.read_bytes() == original
    export_records.mark_acked(PROFILE, DEST, "batch-first")
    settled = fx.registry().entries[planned[0].record_id]
    assert settled.acked and settled.acked.revision == 1 and not settled.queued


@pytest.mark.parametrize("damage", ["invalid-id", "directory-symlink", "lock-symlink"])
def test_registry_unsafe_path_refuses_updates_and_preserves_existing_record(
    damage: str,
    tmp_path: Path,
) -> None:
    planned = export_records.plan_revisions(export_records.Registry(), project(rows_a()))
    export_records.mark_queued(PROFILE, DEST, planned, "batch-first")
    directory = fx.profile_root() / "export" / DEST
    original = (directory / "records.json").read_bytes()
    saved = tmp_path / "saved-registry"
    lock = directory / ".records.lock"
    if damage == "directory-symlink":
        directory.rename(saved)
        directory.symlink_to(saved, target_is_directory=True)
    elif damage == "lock-symlink":
        lock.unlink()
        lock.symlink_to(directory / "records.json")
    with pytest.raises(export_records.RegistryCorrupt) as error:
        export_records.mark_acked(
            PROFILE, "../escape" if damage == "invalid-id" else DEST, "batch-first"
        )
    assert error.value.condition == "registry_unsafe"
    assert (directory / "records.json").read_bytes() == original
    if damage == "directory-symlink":
        directory.unlink()
        saved.rename(directory)
    elif damage == "lock-symlink":
        lock.unlink()
    export_records.mark_acked(PROFILE, DEST, "batch-first")
    assert fx.registry().entries[planned[0].record_id].acked.revision == 1


def test_late_unknown_outcome_never_regresses_newer_unknown_revision() -> None:
    record_id = project(rows_a())[0].record_id
    for revision, rows in enumerate((rows_a(), rows_b(), rows_a()), 1):
        planned = export_records.plan_revisions(fx.registry(), project(rows))
        assert planned[0].revision == revision
        export_records.mark_queued(PROFILE, DEST, planned, f"batch-{revision}")
    export_records.mark_unknown(PROFILE, DEST, "batch-3")
    export_records.mark_unknown(PROFILE, DEST, "batch-2")
    entry = fx.registry().entries[record_id]
    assert entry.unknown is not None and entry.unknown.revision == 3
    assert [s.revision for s in entry.queued] == [1] and entry.acked is None
    export_records.mark_acked(PROFILE, DEST, "batch-1")
    delivery_registry.promote_unknown(PROFILE, DEST, [(record_id, 3)])
    settled = fx.registry().entries[record_id]
    assert settled.acked and settled.acked.revision == 3
    assert settled.unknown is None and not settled.queued


def test_body_pairs_discards_partial_parse_instead_of_promoting_only_a_prefix() -> None:
    with LabReceiver.start() as lab:
        fx.configure(lab.url, signals=(L,), logs_permitted=True, enabled=(L,))
        fx.commit_ledger(2)
        fx.publish(1)
        fx.cycle(L, deps=fx.deps(max_batches=0))
        (meta,) = fx.metas(L)
        original = fx.body_of(meta)
        expected = [(record_id, 1) for record_id in meta.record_ids]
        assert delivery_registry.body_pairs(original) == expected
        damaged = json.loads(original)
        records = damaged["resourceLogs"][0]["scopeLogs"][0]["logRecords"]
        records[-1]["body"]["stringValue"] = "not JSON"
        assert delivery_registry.body_pairs(json.dumps(damaged).encode()) == []
        assert export_spool.read_body(PROFILE, DEST, L, meta.batch_id) == original
        assert fx.cycle(L, seconds=1).acked == 1
        assert [r.body for r in lab.requests] == [original]
        assert all(e.acked and not e.queued for e in fx.registry().entries.values())


def test_receiver_requires_authenticated_key_type_before_mutating_state() -> None:
    (planned,) = export_records.plan_revisions(export_records.Registry(), project(rows_a()))
    state = receiver_contract.ReceiverState()
    for bad_key in (
        None,
        ("tenant", "device", planned.record_id),
        {"record_id": planned.record_id},
    ):
        rejected, outcome = receiver_contract.apply(
            state, cast(receiver_contract.Key, bad_key), planned.body
        )
        assert rejected is state and outcome is receiver_contract.Outcome.REJECTED_INVALID
    key = receiver_contract.Key("tenant", "device", planned.record_id)
    accepted, outcome = receiver_contract.apply(state, key, planned.body)
    assert outcome is receiver_contract.Outcome.APPLIED_NEW
    assert accepted.current[key].revision == 1 and state.current == {}


def test_otlp_json_preserves_null_and_decimal_tokens_but_refuses_python_float() -> None:
    value = {"missing": None, "duration": otlp.Raw("0.001"), "flags": [True, False]}
    assert otlp.dumps(value) == b'{"missing":null,"duration":0.001,"flags":[true,false]}'
    with pytest.raises(TypeError):
        otlp.dumps({**value, "duration": 0.001})
    assert json.loads(otlp.dumps(value))["missing"] is None
