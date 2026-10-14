"""Collector recovery refuses unsafe or mismatched state without publishing new totals."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ssgrep.accounting import collector_progress, collector_state, collector_store as store
from ssgrep.accounting.collector_ledger import VersionedLedger
from ssgrep.accounting.collector_types import CollectError
from ssgrep.accounting.contracts import SCHEMA_VERSION
from ssgrep.accounting.ledger_codec import ledger_state_to_dict
from tests.accounting import collector_scene as scene, verify_scenes as vs
from tests.accounting.reduce_builders import PROFILE
from tests.accounting.test_guard_collector_recovery import NOW, seeded
from tests.accounting.valuation_builders import METHOD


@pytest.mark.parametrize("component", ["worker", "snapshots"])
def test_unsafe_worker_or_snapshot_path_is_refused_without_following_it(
    tmp_path: Path, component: str
) -> None:
    seeded(tmp_path)
    directory = vs.accounting_dir() / component
    saved = tmp_path / f"saved-{component}"
    current = vs.accounting_dir() / "snapshots" / "current.json"
    before = current.read_bytes()
    directory.rename(saved)
    directory.symlink_to(saved, target_is_directory=True)
    try:
        with pytest.raises(CollectError) as failed:
            scene.collect(NOW)
        assert failed.value.condition == "storage_unsafe"
        assert current.read_bytes() == before
        assert scene.committed().generation == 1
    finally:
        directory.unlink()
        saved.rename(directory)
    assert scene.report(scene.collect(NOW)).generation == 2
    assert scene.counter_total(scene.current()) == 2


def test_newer_snapshot_schema_is_not_repaired_or_overwritten(tmp_path: Path) -> None:
    seeded(tmp_path)
    path = vs.accounting_dir() / "snapshots" / "current.json"
    original = path.read_bytes()
    newer = json.loads(original)
    newer["schema_version"] = SCHEMA_VERSION + 1
    path.write_text(json.dumps(newer))
    before = path.read_bytes()
    with pytest.raises(CollectError) as failed:
        scene.collect(NOW)
    assert failed.value.condition == "unsupported_schema"
    assert path.read_bytes() == before and scene.committed().generation == 1
    path.write_bytes(original)
    assert scene.report(scene.collect(NOW)).generation == 2


def test_malformed_current_pointer_is_repaired_from_the_committed_generation(
    tmp_path: Path,
) -> None:
    seeded(tmp_path)
    path = vs.accounting_dir() / "snapshots" / "current.json"
    path.write_bytes(b"{")
    result = scene.collect(NOW)
    assert scene.report(result).generation == 2
    assert scene.counter_total(scene.current()) == 2
    assert scene.current().generation == 2


def test_missing_committed_snapshot_is_corruption_not_permission_to_reset(tmp_path: Path) -> None:
    seeded(tmp_path)
    directory = vs.accounting_dir() / "snapshots"
    generation = next(directory.glob("gen-*.json"))
    original = generation.read_bytes()
    current = (directory / "current.json").read_bytes()
    generation.unlink()
    with pytest.raises(CollectError) as failed:
        scene.collect(NOW)
    assert failed.value.condition == "state_corrupt"
    assert (directory / "current.json").read_bytes() == current
    generation.write_bytes(original)
    generation.chmod(0o600)
    assert scene.report(scene.collect(NOW)).generation == 2


@pytest.mark.parametrize("component", ["state", "evidence"])
def test_missing_committed_input_directory_is_not_an_empty_dataset(
    tmp_path: Path, component: str
) -> None:
    seeded(tmp_path)
    directory = vs.accounting_dir() / "collector" / component
    saved = tmp_path / f"saved-{component}"
    before = scene.current()
    directory.rename(saved)
    vs.filler(vs.projects(tmp_path), "new-input", NOW, 1)
    try:
        with pytest.raises(CollectError) as failed:
            scene.collect(NOW)
        assert failed.value.condition == "state_corrupt"
        assert scene.current() == before
    finally:
        if directory.exists():
            directory.rmdir()  # an empty staging directory may have been opened before reading
        saved.rename(directory)
    assert scene.report(scene.collect(NOW)).generation == 2
    assert scene.counter_total(scene.current()) == 2


def test_cross_namespace_ledger_reference_is_not_published(tmp_path: Path) -> None:
    seeded(tmp_path)
    checkpoint = scene.committed()
    original = checkpoint.to_dict()
    key = collector_state.ledger_blob_name(checkpoint.methodology)
    original_path = vs.accounting_dir() / "collector" / "state" / checkpoint.blobs[key]
    original_bytes = original_path.read_bytes()
    assert METHOD.hash != checkpoint.methodology
    checkpoint.blobs[key] = store.write_blob(
        PROFILE,
        key,
        {
            "state": ledger_state_to_dict(VersionedLedger(METHOD).state()),
            "superseded": {},
            "collection_scope": checkpoint.collection_scope,
        },
    )
    store.commit_checkpoint(PROFILE, checkpoint)
    before = scene.current()
    with pytest.raises(CollectError) as failed:
        scene.collect(NOW)
    assert failed.value.condition == "state_corrupt"
    assert scene.current() == before
    original_path.write_bytes(original_bytes)
    original_path.chmod(0o600)
    store.commit_checkpoint(PROFILE, store.Checkpoint.from_dict(original))
    assert scene.report(scene.collect(NOW)).generation == 2


@pytest.mark.parametrize("document", [b"{", b'{"cycles":{}}'])
def test_bad_advisory_progress_does_not_reset_committed_counters(
    tmp_path: Path, document: bytes
) -> None:
    seeded(tmp_path)
    path = vs.accounting_dir() / "worker" / collector_progress.PROGRESS_FILE
    path.write_bytes(document)
    assert collector_progress.load_previous(PROFILE) is None
    result = scene.collect(NOW)
    assert result.epoch == 1 and scene.counter_total(scene.current()) == 2
    restored = collector_progress.read_progress(PROFILE)
    assert restored is not None and restored["last_cycle"]["generation"] == 2
