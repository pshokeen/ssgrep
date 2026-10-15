"""Collection rarity uses compatible index inputs, never retained-history filler.

The obsolete 199/200 floor-only test is superseded by
test_frequency_generations.test_singleton_feasibility_independent_of_former_200_floor
at documents=199/200/511. Crash and private-file security controls remain here.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from ssgrep.accounting import collector_frequency as frequency, fingerprint, index_frequency
from ssgrep.accounting.collector_ledger import VersionedLedger
from ssgrep.accounting.collector_state import load_state
from ssgrep.accounting.collector_types import CollectError, CollectOptions
from ssgrep.accounting.frequency import FrequencyView
from ssgrep.accounting.plain_codec import to_plain
from tests.accounting import collector_scene as scene, verify_scenes as vs
from tests.accounting.reduce_builders import PROFILE, claude_prompt, ts
from tests.accounting.test_frequency_generations import MetadataRepository, database

CLOCK = scene.at(hour=9)
NOW = scene.at(hour=20)


class CompatibleIndex:
    """Independent read-only repository fixture with explicitly published generations."""

    def __init__(self, path: Path) -> None:
        self.path = path
        database(path, 511)
        self.repository = MetadataRepository(path)
        self.advance("generation-1")

    def advance(self, generation: str) -> None:
        writer = sqlite3.connect(self.path)
        try:
            writer.execute("UPDATE metadata SET value=? WHERE key='last_index_time'", (generation,))
            writer.execute(
                "UPDATE metadata SET value=? WHERE key='lexical_stats'",
                (
                    json.dumps(
                        {"n_docs": 511, "avgdl": 3.0, "df": {"alpha": 1, "err_quota_alpha": 1}}
                    ),
                ),
            )
            writer.commit()
        finally:
            writer.close()

    def remove_stats(self) -> None:
        writer = sqlite3.connect(self.path)
        try:
            writer.execute("DELETE FROM metadata WHERE key='lexical_stats'")
            writer.execute(
                "UPDATE metadata SET value='generation-unavailable' WHERE key='last_index_time'"
            )
            writer.commit()
        finally:
            writer.close()


@pytest.fixture
def compatible_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    index = CompatibleIndex(tmp_path / "frequency-index.sqlite")
    original = index_frequency.read_index_view

    def current(profile: str, **kwargs):
        return original(
            profile,
            repository=index.repository,
            authority=kwargs.get("authority") or (lambda cwd: cwd == "/approved"),
            authority_id=kwargs.get("authority_id") or "approved-v1",
            cached=kwargs.get("cached"),
        )

    monkeypatch.setattr(index_frequency, "read_index_view", current)
    try:
        yield index
    finally:
        index.repository.close()


def lifetime_credit(envelope: Any) -> dict[str, dict[str, Any]]:
    return {
        unit: scenarios["central"]["credit"]
        for unit, scenarios in scene.window(envelope, "lifetime")["estimates"].items()
    }


def committed_state():
    checkpoint = vs.checkpoint()
    assert checkpoint is not None
    return load_state(PROFILE, checkpoint, checkpoint.methodology)


def corpus(tmp_path: Path) -> None:
    """Actual source, native exposure/action and journal execution; no rarity filler."""
    vs.pair(vs.projects(tmp_path), 0, CLOCK)


def pinned_view() -> FrequencyView:
    view = frequency.load_pinned(PROFILE)
    assert view is not None, "no frequency view is pinned"
    return view


def pinned_files() -> list[str]:
    return sorted(path.name for path in (vs.accounting_dir() / "collector").glob("frequency*.json"))


def test_absent_index_is_unknown_not_zero_despite_retained_evidence(tmp_path: Path) -> None:
    vs.enable()
    vs.start_capture()
    corpus(tmp_path)
    result = scene.collect(NOW)
    assert result.state == "published"
    envelope = scene.current()
    block = scene.coverage(envelope)["frequency"]
    assert block["state"] == "unavailable" and block["reason"] == "frequency_index_absent"
    assert "view_id" not in block and pinned_files() == []
    estimates = scene.window(envelope, "lifetime")["estimates"]
    bands = [band for scenarios in estimates.values() for band in scenarios.values()]
    assert all(not band["complete"] and band["net"]["value"] is None for band in bands)
    assert all(band["credit_missing"] for band in bands)
    assert scene.coverage(envelope)["exclusions"]["rarity_unknown"] >= 1
    assert scene.coverage(envelope)["partial"] is True


def test_compatible_generation_refresh_does_not_rerate_existing_lookups(
    tmp_path: Path, compatible_index: CompatibleIndex
) -> None:
    vs.enable()
    vs.start_capture()
    corpus(tmp_path)
    scene.collect(NOW)
    original = pinned_view()
    old_file = vs.accounting_dir() / "collector" / frequency.view_file_name(original.version_id)
    old_bytes = old_file.read_bytes()
    before = committed_state()
    assignments = tuple(to_plain(row) for row in before.frequency_assignments)
    ledger = vs.ledger_digest()
    method = scene.current().methodology_id
    compatible_index.advance("generation-2")
    scene.collect(NOW)
    assert pinned_view().version_id != original.version_id
    assert scene.current().methodology_id == method
    assert tuple(to_plain(row) for row in committed_state().frequency_assignments) == assignments
    assert old_file.read_bytes() == old_bytes
    assert vs.ledger_digest() == ledger


def test_restart_and_original_source_disappearance_preserve_frozen_inputs(
    tmp_path: Path, compatible_index: CompatibleIndex
) -> None:
    vs.enable()
    vs.start_capture()
    corpus(tmp_path)
    scene.collect(NOW)
    original = pinned_view()
    assignments = committed_state().frequency_assignments
    ledger = vs.ledger_digest()
    fingerprint.forget_keys()  # reload durable inputs rather than carrying process key state
    (vs.projects(tmp_path) / "-work" / "src0.jsonl").unlink()
    scene.append_transcript(
        vs.projects(tmp_path) / "-work" / "cons0.jsonl",
        [
            claude_prompt(
                "cons0", "Continue diagnostics", uuid="after-source-removal", ts=ts(12, 19)
            )
        ],
        NOW,
    )
    scene.collect(NOW)
    assert pinned_view() == original
    assert committed_state().frequency_assignments == assignments
    assert vs.ledger_digest() == ledger


def test_explicit_recompute_preserves_named_old_rows_without_adding_their_totals(
    tmp_path: Path, compatible_index: CompatibleIndex
) -> None:
    vs.enable()
    vs.start_capture()
    corpus(tmp_path)
    scene.collect(NOW)
    before = scene.current()
    assert any(
        amount["value"] is not None and amount["value"] > 0
        for amount in lifetime_credit(before).values()
    )
    old_state = committed_state()
    old_assignments = old_state.frequency_assignments
    previous_rows = VersionedLedger.restore_versioned(
        old_state.ledger, old_state.superseded, old_state.archives
    ).rows()
    compatible_index.advance("generation-2")
    scene.collect(NOW)
    scene.collect(NOW, CollectOptions(recompute=True))
    current = committed_state()
    latest = pinned_view()
    assert scene.current().methodology_id == before.methodology_id
    assert current.archives
    assert any(
        archive.as_rows().canonical_bytes() == previous_rows.canonical_bytes()
        for archive in current.archives.values()
    )
    assert all(row.view_id == latest.version_id for row in current.frequency_assignments)
    assert all(
        row.revision_name == f"frequency-recompute:{latest.version_id}"
        for row in current.frequency_assignments
    )
    assert any(row.view_id != latest.version_id for row in old_assignments)
    assert lifetime_credit(scene.current()) == lifetime_credit(before)


def test_missing_current_stats_never_authorize_stale_pin_for_new_work(
    tmp_path: Path, compatible_index: CompatibleIndex
) -> None:
    vs.enable()
    vs.start_capture()
    corpus(tmp_path)
    scene.collect(NOW)
    original = pinned_view()
    compatible_index.remove_stats()
    result = scene.collect(NOW, CollectOptions(repin_frequency=True))
    assert result.state == "published"
    assert pinned_view() == original
    block = scene.coverage(scene.current())["frequency"]
    assert block["state"] == "unavailable" and block["reason"] == "frequency_stats_absent"
    assert any("frequency_stats_absent" in warning for warning in result.warnings)


def test_explicit_repin_without_prior_pin_uses_compatible_index(
    tmp_path: Path, compatible_index: CompatibleIndex
) -> None:
    vs.enable()
    vs.start_capture()
    corpus(tmp_path)
    result = scene.collect(NOW, CollectOptions(repin_frequency=True))
    assert result.state == "published"
    assert pinned_view().doc_count == 511
    assert scene.coverage(scene.current())["frequency"]["state"] == "pinned"


@pytest.mark.parametrize("step", ["frequency_view_written", "frequency_pinned"])
def test_interrupted_first_pin_recovers_to_one_published_namespace(
    tmp_path: Path, compatible_index: CompatibleIndex, step: str
) -> None:
    vs.enable()
    vs.start_capture()
    corpus(tmp_path)

    def interrupt(current: str) -> None:
        if current == step:
            raise RuntimeError("interrupted frequency publication")

    with pytest.raises(CollectError) as failed:
        scene.collect(NOW, on_step=interrupt)
    assert failed.value.condition == "collection_failed"
    assert vs.checkpoint() is None
    pin = frequency.load_pinned(PROFILE)
    assert (pin is not None) == (step == "frequency_pinned")
    result = scene.collect(NOW)
    assert scene.report(result).generation == 1
    assert scene.counter_total(scene.current()) == 2
    current = frequency.load_pinned(PROFILE)
    assert current is not None and current.doc_count == 511
    assert len([key for key in scene.committed().blobs if key.startswith("ledger-")]) == 1
    scene.collect(NOW)
    assert frequency.load_pinned(PROFILE) == current
    assert scene.counter_total(scene.current()) == 2


def test_unchanged_stats_and_prospective_refresh_keep_one_model_and_explicit_recompute_archives(
    tmp_path: Path, compatible_index: CompatibleIndex
) -> None:
    vs.enable()
    vs.start_capture()
    corpus(tmp_path)
    scene.collect(NOW)
    before = scene.current()
    current = pinned_view()
    files = {
        path.name: path.read_bytes()
        for path in (vs.accounting_dir() / "collector").glob("frequency*.json")
    }
    scene.collect(NOW)
    assert pinned_view() == current
    assert scene.current().methodology_id == before.methodology_id
    assert {
        path.name: path.read_bytes()
        for path in (vs.accounting_dir() / "collector").glob("frequency*.json")
    } == files
    compatible_index.advance("generation-2")
    scene.collect(NOW)
    refreshed = pinned_view()
    assert refreshed.version_id != current.version_id
    assert scene.current().methodology_id == before.methodology_id
    scene.collect(NOW, CollectOptions(recompute=True))
    assert committed_state().archives
    assert all(
        row.view_id == refreshed.version_id for row in committed_state().frequency_assignments
    )
    assert len([key for key in scene.committed().blobs if key.startswith("ledger-")]) == 1


@pytest.mark.parametrize(
    "damage",
    [
        "pointer-json",
        "pointer-key",
        "pointer-id",
        "missing-view",
        "view-json",
        "view-name",
        "view-digest",
    ],
)
def test_corrupt_pin_keeps_the_last_generation_and_namespace(
    tmp_path: Path, compatible_index: CompatibleIndex, damage: str
) -> None:
    vs.enable()
    vs.start_capture()
    corpus(tmp_path)
    scene.collect(NOW)
    old = scene.current()
    directory = vs.accounting_dir() / "collector"
    pointer = directory / frequency.POINTER
    view = frequency.load_pinned(PROFILE)
    assert view is not None
    path = directory / frequency.view_file_name(view.version_id)
    original = {pointer: pointer.read_bytes(), path: path.read_bytes()}
    if damage.startswith("pointer-"):
        broken = {
            "pointer-json": b"{",
            "pointer-key": b"{}",
            "pointer-id": b'{"view_id":"../other"}',
        }
        pointer.write_bytes(broken[damage])
    elif damage == "missing-view":
        path.unlink()
    elif damage == "view-json":
        path.write_bytes(b"[]")
    else:
        raw = json.loads(path.read_bytes())
        if damage == "view-name":
            raw["version_id"] = "freq-" + "0" * 24
        else:
            raw["doc_count"] += 1
        path.write_text(json.dumps(raw))
    corrupted = {p.name: p.read_bytes() for p in directory.glob("frequency*.json")}
    with pytest.raises(CollectError) as failed:
        scene.collect(NOW, CollectOptions(repin_frequency=True))
    assert failed.value.condition == "state_corrupt"
    assert scene.current() == old
    assert scene.committed().methodology == old.methodology_id
    assert {p.name: p.read_bytes() for p in directory.glob("frequency*.json")} == corrupted
    for file, content in original.items():
        file.write_bytes(content)
        file.chmod(0o600)
    scene.collect(NOW)
    assert scene.current().methodology_id == old.methodology_id
    assert scene.counter_total(scene.current()) == scene.counter_total(old)


def test_frequency_size_limits_fail_closed_without_losing_the_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vs.enable(pin=True)
    view = frequency.load_pinned(PROFILE)
    assert view is not None
    directory = vs.accounting_dir() / "collector"
    for name, path in (
        ("MAX_POINTER_BYTES", directory / frequency.POINTER),
        ("MAX_VIEW_BYTES", directory / frequency.view_file_name(view.version_id)),
    ):
        with monkeypatch.context() as patch:
            patch.setattr(frequency, name, path.stat().st_size - 1)
            with pytest.raises(CollectError) as failed:
                frequency.load_pinned(PROFILE)
            assert failed.value.condition == "state_corrupt"
        assert frequency.load_pinned(PROFILE) == view


def test_repin_cannot_overwrite_a_damaged_content_addressed_view() -> None:
    first = vs.pinned_view(1)
    second = vs.pinned_view(2)
    frequency.pin(PROFILE, first)
    directory = vs.accounting_dir() / "collector"
    pointer = (directory / frequency.POINTER).read_bytes()
    target = directory / frequency.view_file_name(second.version_id)
    target.write_bytes(b"damaged but already named")
    target.chmod(0o600)
    with pytest.raises(CollectError) as failed:
        frequency.pin(PROFILE, second)
    assert failed.value.condition == "state_corrupt"
    assert (directory / frequency.POINTER).read_bytes() == pointer
    assert target.read_bytes() == b"damaged but already named"
    target.unlink()
    frequency.pin(PROFILE, second)
    assert frequency.load_pinned(PROFILE) == second


@pytest.mark.parametrize("damage", ["basis", "empty", "identity"])
def test_invalid_candidate_view_does_not_replace_a_valid_pin(damage: str) -> None:
    valid = vs.pinned_view()
    frequency.pin(PROFILE, valid)
    invalid = {
        "basis": replace(valid, basis="words"),
        "empty": replace(valid, doc_count=0),
        "identity": replace(valid, version_id="freq-" + "0" * 24),
    }[damage]
    with pytest.raises(ValueError):
        frequency.pin(PROFILE, invalid)
    assert frequency.load_pinned(PROFILE) == valid


@pytest.mark.parametrize("artifact", ["pointer", "view"])
@pytest.mark.parametrize("boundary", ["mode", "symlink"])
def test_unsafe_frequency_artifacts_preserve_committed_generation(
    tmp_path: Path, compatible_index: CompatibleIndex, artifact: str, boundary: str
) -> None:
    vs.enable()
    vs.start_capture()
    corpus(tmp_path)
    scene.collect(NOW)
    old = scene.current()
    view = pinned_view()
    directory = vs.accounting_dir() / "collector"
    name = frequency.POINTER if artifact == "pointer" else frequency.view_file_name(view.version_id)
    path = directory / name
    original = path.read_bytes()
    if boundary == "mode":
        path.chmod(0o644)
    else:
        outside = tmp_path / f"outside-{artifact}.json"
        outside.write_bytes(original)
        outside.chmod(0o600)
        path.unlink()
        path.symlink_to(outside)
    with pytest.raises(CollectError):
        scene.collect(NOW)
    assert scene.current() == old
    if boundary == "symlink":
        assert outside.read_bytes() == original
        assert path.is_symlink()
        path.unlink()
        path.write_bytes(original)
    path.chmod(0o600)
    scene.collect(NOW)
    assert pinned_view() == view
    assert scene.counter_total(scene.current()) == scene.counter_total(old)
