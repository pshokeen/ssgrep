"""Large authorized metadata progresses without reading a plaintext corpus or moving pins."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest

from ssgrep.accounting import fingerprint, storage
from ssgrep.accounting.collector_store import dumps
from ssgrep.accounting.evidence_tokens import Token
from ssgrep.accounting.frequency_assignment import FrequencyAssignments
from ssgrep.accounting.index_frequency import read_index_view
from tests.accounting.test_frequency_generations import PROFILE, RARE, MetadataRepository, database


class SnapshotRepository(MetadataRepository):
    """Versioned SQL fixtures are real separate immutable database files."""

    def __init__(self, path: Path, snapshots: dict[str, Path]) -> None:
        super().__init__(path)
        self.snapshots = snapshots

    def snapshot(self, generation: str):
        return MetadataRepository(self.snapshots[generation])


def current(repository, authority_id="approved-v1", authority=None, cached=None):
    return read_index_view(
        PROFILE,
        repository=repository,
        authority=authority or (lambda cwd: cwd == "/approved"),
        authority_id=authority_id,
        cached=cached,
    )


@pytest.fixture
def large_index(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    fingerprint.load_key(PROFILE, create=True)
    path = tmp_path / "large.sqlite"
    database(path, 253_526)
    writer = sqlite3.connect(path)
    terms = {f"term_{i:08d}": 2 for i in range(112_168)} | {RARE.text: 1}
    raw = json.dumps({"n_docs": 253_526, "df": terms, "avgdl": 3.0})
    assert len(raw.encode()) > 2_068_504
    writer.execute("UPDATE metadata SET value=? WHERE key='lexical_stats'", (raw,))
    writer.commit()
    writer.close()
    return path


def test_real_scale_finishes_after_restart_despite_new_live_generations(large_index, tmp_path):
    initial = MetadataRepository(large_index)
    first = current(initial)
    initial.close()
    assert not first.known and first.unavailable_reason == "frequency_build_pending"
    pending = FrequencyAssignments(first)
    assert pending.view_for("journal:pending").rarity_milli(RARE) is None
    records = pending.records()
    snapshot = tmp_path / "generation-1.sqlite"
    source = sqlite3.connect(large_index)
    target = sqlite3.connect(snapshot)
    source.backup(target)
    target.close()
    source.close()
    calls = 0
    for cycle in range(256):
        writer = sqlite3.connect(large_index)
        writer.execute(
            "UPDATE metadata SET value=? WHERE key='last_index_time'", (f"generation-{cycle + 2}",)
        )
        writer.commit()
        writer.close()
        repository = SnapshotRepository(large_index, {"generation-1": snapshot})
        authorized = [0]

        def authorize(cwd, count=authorized):
            count[0] += 1
            return cwd == "/approved"

        result = current(repository, authority=authorize, cached=first)
        assert authorized[0] <= 4096
        repository.close()
        calls += 1
        if result.known:
            break
        assert result.unavailable_reason == "frequency_build_pending"
    else:
        pytest.fail("bounded frequency build did not finish")
    assert calls > 1 and result.doc_count == 253_526
    assert result.generation == "generation-1"
    assert len(result.df) == 112_169
    rarity = result.rarity_milli(RARE)
    assert rarity is not None and rarity >= 8000
    assert result.rarity_milli(Token("identifier", "not_in_stats")) is None
    completed = FrequencyAssignments(result, assignments=records)
    assert completed.completed_pending
    assert completed.view_for("journal:pending") == result
    assert completed.records()[0].revision_name == f"frequency-build-complete:{result.version_id}"
    directory = storage.data_root() / "accounting" / PROFILE / "collector"
    for path in directory.glob("frequency-build*.json"):
        body = path.read_bytes()
        assert RARE.text.encode() not in body and b"/approved" not in body

    class IdleRepository(MetadataRepository):
        def get_meta(self, key):
            assert key != "lexical_stats", "idle generation must not reread its statistics"
            return super().get_meta(key)

        def snapshot(self, generation):
            pytest.fail("idle generation must not open build pages or corpus projections")

    idle = IdleRepository(snapshot)
    assert current(idle, cached=result) is result
    idle.close()


@pytest.mark.parametrize("workspace", [None, "/approved"])
def test_erasure_deletes_pending_head_pages_and_uncommitted_slots(tmp_path, monkeypatch, workspace):
    from ssgrep.accounting import index_frequency_build
    from ssgrep.accounting.frequency_authority import erased_view
    from ssgrep.accounting.purge_derived import purge_derived
    from ssgrep.accounting.purge_select import Selection

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(index_frequency_build, "VISITS", 16)
    fingerprint.load_key(PROFILE, create=True)
    path = tmp_path / "pending.sqlite"
    database(path, 511)
    repository = MetadataRepository(path)
    pending = current(repository)
    assert pending.unavailable_reason == "frequency_build_pending"
    directory = storage.data_root() / "accounting" / PROFILE / "collector"
    assert len(list(directory.glob("frequency-build-*.json"))) >= 2
    descriptor = storage.open_profile_dir(PROFILE, "collector")
    try:
        storage.atomic_write(descriptor, "frequency-build-seen-00000001.json", b"[]")
    finally:
        os.close(descriptor)
    purge_derived(Selection(PROFILE, "2026-12-01T00:00:00Z", workspace), invalidate_frequency=True)
    assert not list(directory.glob("frequency-build-*.json"))
    erased = erased_view(PROFILE, scope_epoch=1, reset_authority=False, repin=False)
    assert erased is not None and erased.unavailable_reason == "frequency_scope_erased"
    assert erased.rarity_milli(RARE) is None
    repository.close()


def test_late_foreign_chunk_prevents_global_df_publication(large_index):
    writer = sqlite3.connect(large_index)
    writer.execute("UPDATE chunks SET cwd='/foreign' WHERE chunk_id='253525'")
    writer.commit()
    writer.close()
    repository = MetadataRepository(large_index)
    for _ in range(256):
        result = current(repository)
        if result.unavailable_reason != "frequency_build_pending":
            break
    assert not result.known and result.unavailable_reason == "frequency_scope_unapproved"
    assert result.rarity_milli(RARE) is None
    repository.close()


def test_pending_authority_change_cannot_adopt_previous_assignment(large_index):
    repository = MetadataRepository(large_index)
    first = current(repository)
    pending = FrequencyAssignments(first)
    pending.view_for("journal:pending")
    records = pending.records()
    for _ in range(256):
        result = current(repository, authority_id="approved-v2")
        if result.known:
            break
    assert result.known
    assignments = FrequencyAssignments(result, assignments=records)
    assert not assignments.completed_pending
    assert assignments.view_for("journal:pending").rarity_milli(RARE) is None
    repository.close()


def test_duplicate_identity_in_a_later_page_cannot_complete(tmp_path, monkeypatch):
    from ssgrep.accounting import index_frequency_build

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(index_frequency_build, "VISITS", 128)
    fingerprint.load_key(PROFILE, create=True)
    path = tmp_path / "duplicate.sqlite"
    database(path, 511)

    class DuplicateRepository(MetadataRepository):
        def rows_page(self, *, start, limit):
            rows = super().rows_page(start=start, limit=limit)
            if start >= 128:
                rows[0]["chunk_id"] = "0"
            return rows

    repository = DuplicateRepository(path)
    assert current(repository).unavailable_reason == "frequency_build_pending"
    duplicate = current(repository)
    assert duplicate.unavailable_reason == "frequency_stats_incompatible"
    assert duplicate.rarity_milli(RARE) is None
    repository.close()


def test_statistics_change_after_pending_never_publishes_mixed_df(tmp_path, monkeypatch):
    from ssgrep.accounting import index_frequency_build

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(index_frequency_build, "VISITS", 128)
    fingerprint.load_key(PROFILE, create=True)
    path = tmp_path / "raced.sqlite"
    database(path, 511)
    repository = MetadataRepository(path)
    pending = current(repository)
    assert pending.unavailable_reason == "frequency_build_pending"
    writer = sqlite3.connect(path)
    writer.execute(
        "UPDATE metadata SET value=? WHERE key='lexical_stats'",
        (json.dumps({"n_docs": 511, "avgdl": 3.0, "df": {RARE.text: 511}}),),
    )
    writer.commit()
    writer.close()
    raced = current(repository, cached=pending)
    assert raced.unavailable_reason == "frequency_generation_changed"
    assert raced.rarity_milli(RARE) is None
    repository.close()


def write_build_file(name, body):
    descriptor = storage.open_profile_dir(PROFILE, "collector")
    try:
        storage.atomic_write(descriptor, name, body)
    finally:
        os.close(descriptor)


def finish_build(repository, **kwargs):
    for _ in range(8):
        result = current(repository, **kwargs)
        if result.unavailable_reason != "frequency_build_pending":
            return result
    pytest.fail("small bounded build did not finish")


@pytest.fixture
def pending_build(tmp_path, monkeypatch):
    from ssgrep.accounting import index_frequency_build

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(index_frequency_build, "VISITS", 128)
    fingerprint.load_key(PROFILE, create=True)
    path = tmp_path / "bounded.sqlite"
    database(path, 511)
    repository = MetadataRepository(path)
    try:
        pending = current(repository)
        assert not pending.known and pending.unavailable_reason == "frequency_build_pending"
        directory = storage.data_root() / "accounting" / PROFILE / "collector"
        head = json.loads((directory / "frequency-build-head.json").read_bytes())["job"]
        assert head["term_cursor"] == 2 and head["chunk_cursor"] == 128
        yield repository, directory
    finally:
        repository.close()


@pytest.mark.parametrize("interruption", ["before-page", "before-head", "after-head"])
def test_interrupted_public_build_resumes_from_atomic_head(
    pending_build, monkeypatch, interruption
):
    repository, directory = pending_build
    head_path = directory / "frequency-build-head.json"
    before_head = head_path.read_bytes()
    before = json.loads(before_head)["job"]
    committed = {
        reference["name"]: (directory / reference["name"]).read_bytes()
        for reference in before["terms"] + before["seen"]
    }
    next_page = "frequency-build-seen-00000001.json"
    atomic_write = storage.atomic_write

    def interrupt(descriptor, name, body):
        if interruption == "before-page" and name == next_page:
            raise OSError("interrupted immutable page publication")
        if name == head_path.name:
            if interruption == "after-head":
                atomic_write(descriptor, name, body)
            raise OSError("interrupted head publication")
        atomic_write(descriptor, name, body)

    with monkeypatch.context() as patch:
        patch.setattr(storage, "atomic_write", interrupt)
        interrupted = current(repository)
    assert not interrupted.known
    assert interrupted.unavailable_reason == "frequency_index_unreadable"
    assert interrupted.rarity_milli(RARE) is None
    assert (head_path.read_bytes() == before_head) == (interruption != "after-head")
    assert (directory / next_page).exists() == (interruption != "before-page")
    for name, body in committed.items():
        assert (directory / name).read_bytes() == body
    recovered = finish_build(repository)
    assert recovered.known and recovered.doc_count == 511
    assert len(recovered.df) == 2 and recovered.rarity_milli(RARE) == 8000
    accepted = json.loads(head_path.read_bytes())["job"]
    identities = [
        identity
        for reference in accepted["seen"]
        for identity in json.loads((directory / reference["name"]).read_bytes())
    ]
    assert accepted["chunk_cursor"] == len(identities) == len(set(identities)) == 511
    for name, body in committed.items():
        assert (directory / name).read_bytes() == body


@pytest.mark.parametrize("damage", ["malformed", "digest", "oversized"])
def test_damaged_head_never_authorizes_rarity_and_valid_head_recovers(pending_build, damage):
    from ssgrep.accounting import index_frequency_build

    repository, directory = pending_build
    head_path = directory / "frequency-build-head.json"
    original = head_path.read_bytes()
    if damage == "malformed":
        damaged = b"{"
    elif damage == "digest":
        document = json.loads(original)
        document["digest"] = "0" * 64
        damaged = dumps(document)
    else:
        damaged = b" " * (index_frequency_build.HEAD_BYTES + 1)
    write_build_file(head_path.name, damaged)
    rejected = current(repository)
    assert not rejected.known and rejected.unavailable_reason == "frequency_build_damaged"
    assert rejected.rarity_milli(RARE) is None
    assert head_path.read_bytes() == damaged
    write_build_file(head_path.name, original)
    recovered = finish_build(repository)
    assert recovered.known and recovered.doc_count == 511
    assert recovered.rarity_milli(RARE) == 8000


@pytest.mark.parametrize(
    "kind,damage", [("seen", "missing"), ("terms", "digest"), ("seen", "oversized")]
)
def test_untrusted_referenced_page_cannot_publish_and_restored_page_recovers(
    pending_build, kind, damage
):
    from ssgrep.accounting import index_frequency_build

    repository, directory = pending_build
    head = json.loads((directory / "frequency-build-head.json").read_bytes())["job"]
    page_path = directory / head[kind][0]["name"]
    original = page_path.read_bytes()
    if damage == "missing":
        page_path.unlink()
    else:
        damaged = (
            b" " * (index_frequency_build.PAGE_BYTES + 1)
            if damage == "oversized"
            else original + b" "
        )
        write_build_file(page_path.name, damaged)
    rejected = finish_build(repository)
    assert not rejected.known and rejected.unavailable_reason == "frequency_index_unreadable"
    assert rejected.rarity_milli(RARE) is None
    assert page_path.exists() == (damage != "missing")
    write_build_file(page_path.name, original)
    recovered = finish_build(repository)
    assert recovered.known and recovered.doc_count == 511
    assert len(recovered.df) == 2 and recovered.rarity_milli(RARE) == 8000


def test_missing_head_rebuild_revokes_orphans_from_larger_generation(pending_build, tmp_path):
    repository, directory = pending_build
    assert current(repository).unavailable_reason == "frequency_build_pending"
    assert current(repository).unavailable_reason == "frequency_build_pending"
    assert (directory / "frequency-build-seen-00000002.json").exists()
    (directory / "frequency-build-head.json").unlink()
    path = tmp_path / "smaller.sqlite"
    database(path, 127)
    writer = sqlite3.connect(path)
    writer.execute("UPDATE metadata SET value='generation-2' WHERE key='last_index_time'")
    writer.commit()
    writer.close()
    smaller = MetadataRepository(path)
    try:
        recovered = finish_build(smaller)
        assert recovered.known and recovered.doc_count == 127
        assert recovered.generation == "generation-2"
        assert len(recovered.df) == 2 and recovered.rarity_milli(RARE) == 6000
        head_path = directory / "frequency-build-head.json"
        head = json.loads(head_path.read_bytes())["job"]
        referenced = {ref["name"] for ref in head["terms"] + head["seen"]}
        retained = {page.name for page in directory.glob("frequency-build-*.json")}
        assert retained == referenced | {head_path.name}
    finally:
        smaller.close()


@pytest.mark.parametrize("revision", ["authority", "key"])
def test_invalidated_pending_build_revokes_prior_pages_and_next_slots(pending_build, revision):
    repository, directory = pending_build
    assert current(repository).unavailable_reason == "frequency_build_pending"
    head_path = directory / "frequency-build-head.json"
    previous = json.loads(head_path.read_bytes())["job"]
    previous_terms = set(json.loads((directory / previous["terms"][0]["name"]).read_bytes()))
    orphan_names = ["frequency-build-terms-00000001.json", "frequency-build-seen-00000002.json"]
    for name in orphan_names:
        write_build_file(name, b'["uncommitted-keyed-slot"]')
    if revision == "key":
        descriptor = storage.open_profile_dir(PROFILE)
        try:
            storage.atomic_write(descriptor, fingerprint.KEY_FILE, b"r" * fingerprint.KEY_BYTES)
        finally:
            os.close(descriptor)
        revised = current(repository)
        assert not revised.known and revised.unavailable_reason == "frequency_build_pending"
    else:
        revised = current(repository, authority_id="approved-v2", authority=lambda cwd: False)
        assert not revised.known and revised.unavailable_reason == "frequency_scope_unapproved"
        assert not (directory / "frequency-build-seen-00000000.json").exists()
    assert revised.rarity_milli(RARE) is None
    assert not (directory / previous["seen"][1]["name"]).exists()
    assert all(not (directory / name).exists() for name in orphan_names)
    replacement = json.loads(head_path.read_bytes())["job"]
    if revision == "key":
        assert replacement["key_id"] != previous["key_id"]
        replacement_terms = set(
            json.loads((directory / replacement["terms"][0]["name"]).read_bytes())
        )
        assert replacement_terms.isdisjoint(previous_terms)
        recovered = finish_build(repository)
    else:
        assert replacement["authority_id"] == "approved-v2"
        recovered = finish_build(repository, authority_id="approved-v3")
    assert recovered.known and recovered.doc_count == 511
    assert recovered.rarity_milli(RARE) == 8000


@pytest.mark.parametrize("remaining", [0, -1], ids=["exact-cap", "one-byte-over"])
def test_aggregate_page_budget_is_inclusive_and_failure_survives_resume(
    pending_build, monkeypatch, remaining
):
    from ssgrep.accounting import index_frequency_build

    repository, directory = pending_build
    control = finish_build(repository)
    assert control.known and control.rarity_milli(RARE) == 8000
    control_head = json.loads((directory / "frequency-build-head.json").read_bytes())["job"]
    # Scale the aggregate budget to these real keyed pages, not synthetic references.
    budget = control_head["bytes"] + remaining
    monkeypatch.setattr(index_frequency_build, "MAX_BUILD_BYTES", budget)
    result = finish_build(repository, authority_id="approved-v2")
    head = json.loads((directory / "frequency-build-head.json").read_bytes())["job"]
    retained = sum((directory / ref["name"]).stat().st_size for ref in head["terms"] + head["seen"])
    assert retained == head["bytes"] <= budget
    if remaining == 0:
        assert result.known and result.doc_count == 511
        assert retained == budget and result.rarity_milli(RARE) == 8000
    else:
        assert not result.known and result.unavailable_reason == "frequency_build_budget_exceeded"
        assert result.rarity_milli(RARE) is None
        saved = (directory / "frequency-build-head.json").read_bytes()
        repeated = current(repository, authority_id="approved-v2")
        assert repeated.unavailable_reason == "frequency_build_budget_exceeded"
        assert repeated.rarity_milli(RARE) is None
        assert (directory / "frequency-build-head.json").read_bytes() == saved


@pytest.mark.parametrize("remaining", [0, -1], ids=["exact-cap", "one-byte-over"])
def test_head_read_budget_preserves_prior_commit_and_resumes(pending_build, monkeypatch, remaining):
    from ssgrep.accounting import index_frequency_build

    repository, directory = pending_build
    control = finish_build(repository)
    assert control.known and control.rarity_milli(RARE) == 8000
    head_path = directory / "frequency-build-head.json"
    accepted_bytes = head_path.read_bytes()
    budget = len(accepted_bytes) + remaining
    with monkeypatch.context() as patch:
        patch.setattr(index_frequency_build, "HEAD_BYTES", budget)
        result = finish_build(repository, authority_id="approved-v2")
    if remaining == 0:
        assert result.known and result.doc_count == 511
        assert head_path.stat().st_size == budget and result.rarity_milli(RARE) == 8000
    else:
        assert not result.known and result.unavailable_reason == "frequency_build_damaged"
        assert result.rarity_milli(RARE) is None
        assert head_path.read_bytes() == accepted_bytes
        recovered = finish_build(repository, authority_id="approved-v2")
        assert recovered.known and recovered.doc_count == 511
        assert recovered.rarity_milli(RARE) == 8000


@pytest.mark.parametrize("extra_bytes", [0, 1], ids=["exact-page", "one-byte-over"])
def test_scope_projection_byte_boundary_preserves_unknown_not_zero(
    tmp_path, monkeypatch, extra_bytes
):
    from ssgrep.accounting import index_frequency_build

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(index_frequency_build, "VISITS", 1024)
    fingerprint.load_key(PROFILE, create=True)
    path = tmp_path / "projection.sqlite"
    database(path, 511)

    class WideIdentityRepository(MetadataRepository):
        def rows_page(self, *, start, limit):
            rows = super().rows_page(start=start, limit=limit)
            padding = index_frequency_build.PAGE_BYTES - len(dumps(rows)) + extra_bytes
            rows[0]["chunk_id"] += "x" * padding
            assert len(dumps(rows)) == index_frequency_build.PAGE_BYTES + extra_bytes
            return rows

    repository = WideIdentityRepository(path)
    try:
        result = current(repository)
        if extra_bytes == 0:
            assert result.known and result.doc_count == 511
            assert len(result.df) == 2 and result.rarity_milli(RARE) == 8000
        else:
            assert not result.known
            assert result.unavailable_reason == "frequency_scope_page_budget_exceeded"
            assert result.rarity_milli(RARE) is None
            directory = storage.data_root() / "accounting" / PROFILE / "collector"
            head = json.loads((directory / "frequency-build-head.json").read_bytes())["job"]
            assert head["chunk_cursor"] == 0 and head["seen"] == []
    finally:
        repository.close()


def test_orphan_cleanup_drains_across_calls_without_erasing_accepted_view(pending_build):
    from ssgrep.accounting import collector_frequency

    repository, directory = pending_build
    accepted = finish_build(repository)
    assert accepted.known and accepted.rarity_milli(RARE) == 8000
    collector_frequency.pin(PROFILE, accepted)
    head_path = directory / "frequency-build-head.json"
    head = json.loads(head_path.read_bytes())["job"]
    orphan_names = {ref["name"] for ref in head["terms"] + head["seen"]}
    opaque_page = (directory / head["seen"][0]["name"]).read_bytes()
    for number in range(65 - len(orphan_names)):
        name = f"frequency-build-seen-{100 + number:08d}.json"
        write_build_file(name, opaque_page)
        orphan_names.add(name)
    for name in (
        "frequency-build-seen-00000000.json.backup",
        "frequency-build-terms-0000000.json",
        "frequency-build-other-00000000.json",
    ):
        write_build_file(name, b'{"unrelated":"preserve"}')
    preserved = {
        path.name: path.read_bytes()
        for path in directory.iterdir()
        if path.name not in orphan_names and path.name != head_path.name
    }
    head_path.unlink()

    draining = current(repository)
    assert not draining.known
    assert draining.unavailable_reason == "frequency_build_orphan_cleanup_pending"
    assert draining.rarity_milli(RARE) is None
    assert not head_path.exists()
    remaining = {name for name in orphan_names if (directory / name).exists()}
    assert len(remaining) == 1
    for name, body in preserved.items():
        assert (directory / name).read_bytes() == body
    historical = collector_frequency.load_view(PROFILE, accepted.version_id)
    assert historical.known and historical.doc_count == 511
    assert historical.rarity_milli(RARE) == 8000

    rebuilding = current(repository)
    assert not rebuilding.known
    assert rebuilding.unavailable_reason == "frequency_build_pending"
    assert rebuilding.rarity_milli(RARE) is None
    assert head_path.exists()
    recovered = finish_build(repository)
    assert recovered.known and recovered.doc_count == 511
    assert recovered.rarity_milli(RARE) == 8000
    replacement = json.loads(head_path.read_bytes())["job"]
    referenced = {ref["name"] for ref in replacement["terms"] + replacement["seen"]}
    assert {name for name in orphan_names if (directory / name).exists()} == referenced
    for name, body in preserved.items():
        assert (directory / name).read_bytes() == body


def test_orphan_inventory_overflow_preserves_pages_until_inventory_fits(pending_build, monkeypatch):
    from ssgrep.accounting import index_frequency_build

    repository, directory = pending_build
    head_path = directory / "frequency-build-head.json"
    head_path.unlink()
    pages = {path.name: path.read_bytes() for path in directory.glob("frequency-build-*.json")}
    unrelated = {f"unrelated-{number}.json": dumps({"preserve": number}) for number in range(3)}
    for name, body in unrelated.items():
        write_build_file(name, body)
    # Scale the production inventory ceiling: four entries fit, the fifth must
    # refuse the entire cleanup before deleting even a valid orphan page.
    monkeypatch.setattr(index_frequency_build, "ORPHAN_ENTRIES", 4)
    assert len(pages) + len(unrelated) == 5
    rejected = current(repository)
    assert not rejected.known
    assert rejected.unavailable_reason == "frequency_build_orphan_inventory_budget_exceeded"
    assert rejected.rarity_milli(RARE) is None
    assert not head_path.exists()
    for name, body in pages.items() | unrelated.items():
        assert (directory / name).read_bytes() == body

    removed = next(iter(unrelated))
    (directory / removed).unlink()
    del unrelated[removed]
    rebuilding = current(repository)
    assert not rebuilding.known
    assert rebuilding.unavailable_reason == "frequency_build_pending"
    assert rebuilding.rarity_milli(RARE) is None
    assert head_path.exists()
    recovered = finish_build(repository)
    assert recovered.known and recovered.doc_count == 511
    assert len(recovered.df) == 2 and recovered.rarity_milli(RARE) == 8000
    for name, body in unrelated.items():
        assert (directory / name).read_bytes() == body
