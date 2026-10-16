# ---- Survivor source: test_generations.py ----
"""Independent lexical generations, conservative mapping and immutable revision evidence."""

import hashlib
import json
import math
import os
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import lancedb
import pytest

from ssgrep.accounting.capture import fingerprint
from ssgrep.accounting.collect import (
    frequency as collector_frequency,
    frequency_index as index_frequency,
    frequency_index as index_frequency_repository,
    state as collector_state,
)
from ssgrep.accounting.collect.collector import CollectOptions
from ssgrep.accounting.collect.frequency import (
    THRESHOLD_MILLI,
    FrequencyAssignments,
    eligible_scenarios,
    prune_views,
)
from ssgrep.accounting.collect.frequency_index import _generation, read_index_view
from ssgrep.accounting.collect.replay_evidence import Token
from ssgrep.accounting.collect.store import dumps
from ssgrep.accounting.ledger.contracts import Scenario, canonical_bytes
from ssgrep.accounting.ledger.reduce import load_evidence, reduce_profile
from ssgrep.accounting.privacy import storage
from ssgrep.accounting.privacy.policy import enable_collection, load_profile_policy
from ssgrep.accounting.privacy.storage import to_plain
from ssgrep.store import LanceStore
from tests.accounting.collect import collector_scene as scene, verify_scenes as vs
from tests.accounting.ledger.reduce_builders import (
    MCP,
    card,
    claude_prompt,
    claude_tool_result,
    claude_tool_use,
    results_json,
    source_session,
    ts,
)

PROFILE = "personal-local"
RARE = Token("identifier", "review_rare_identifier")
LATER = Token("identifier", "review_later_identifier")
COMMON = Token("identifier", "review_common_identifier")


class MetadataRepository:
    """Real read-only SQL metadata, implementing the existing repository read seam."""

    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        self.connection.execute("PRAGMA query_only=ON")

    def exists(self) -> bool:
        return True

    def get_meta(self, key: str) -> str | None:
        row = self.connection.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def count(self, name: str) -> int:
        assert name == "chunks"
        return self.connection.execute("SELECT count(*) FROM chunks").fetchone()[0]

    def snapshot(self, generation: str):
        # This SQL fixture is immutable during a read. Racing subclasses intentionally
        # keep their dynamic metadata reads so the before/after guard is exercised.
        return self

    def rows_page(self, *, start: int, limit: int) -> list[dict]:
        rows = self.connection.execute(
            "SELECT chunk_id,cwd FROM chunks WHERE rowid>? ORDER BY rowid LIMIT ?", (start, limit)
        )
        return [dict(zip(["chunk_id", "cwd"], row, strict=True)) for row in rows]

    def close(self) -> None:
        self.connection.close()


def database(path: Path, documents: int, *, later: bool = False, mixed: bool = False) -> None:
    writer = sqlite3.connect(path)
    writer.executescript(
        "CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);"
        "CREATE TABLE chunks(chunk_id TEXT PRIMARY KEY,cwd TEXT);"
    )
    writer.executemany(
        "INSERT INTO chunks VALUES (?,?)",
        [(str(i), "/private" if mixed and i == 0 else "/approved") for i in range(documents)],
    )
    terms = {RARE.text: 1, COMMON.text: documents}
    if later:
        terms[LATER.text] = 1
    writer.executemany(
        "INSERT INTO metadata VALUES (?,?)",
        [
            ("index_state", "ready"),
            ("last_index_time", "generation-1"),
            ("lexical_stats", json.dumps({"n_docs": documents, "df": terms, "avgdl": 3.0})),
        ],
    )
    writer.commit()
    writer.close()


def view(repository: MetadataRepository, profile: str = PROFILE):
    return read_index_view(
        profile,
        repository=repository,
        authority=lambda cwd: cwd == "/approved",
        authority_id="approved-v1",
    )


def test_generation_includes_table_versions_and_rejects_invalid_version(tmp_path: Path) -> None:
    path = tmp_path / "index.sqlite"
    database(path, 1)

    class Table:
        def __init__(self, version):
            self.version = version

    class VersionedRepository(MetadataRepository):
        versions = {"metadata": 7, "chunks": 12}

        def table(self, name):
            return Table(self.versions[name])

    repository = VersionedRepository(path)
    try:
        assert _generation(repository) == "generation-1@metadata:7,chunks:12"
        repository.versions["metadata"] = -1
        assert _generation(repository) is None
    finally:
        repository.close()


def test_frequency_logarithm_rejects_invalid_ratio_and_matches_exact_power() -> None:
    assert collector_frequency.milli_log2(2, 1) == 1000
    with pytest.raises(ValueError, match="numerator"):
        collector_frequency.milli_log2(1, 2)


def test_index_rarity_fails_closed_for_unmapped_absent_and_invalid_df_tokens(
    tmp_path: Path, keyed_profile: bytes
) -> None:
    path = tmp_path / "index.sqlite"
    database(path, 4)
    repository = MetadataRepository(path)
    try:
        frozen = view(repository)
        assert frozen.known
        assert frozen.rarity_evidence(Token("sentence", RARE.text)).reason == (
            "frequency_mapping_unavailable"
        )
        assert frozen.rarity_evidence(Token("identifier", "x")).reason == (
            "frequency_candidate_absent"
        )

        from dataclasses import replace

        invalid = replace(
            frozen,
            df={
                collector_frequency.lexical_key(keyed_profile, RARE.text): 0,
            },
        )
        evidence = invalid.rarity_evidence(RARE)
        assert evidence.value is None
        assert evidence.reason == "frequency_df_invalid"
        assert invalid.rarity_milli(RARE) is None
    finally:
        repository.close()


def test_frequency_assignments_keep_historical_view_and_reject_changed_identity() -> None:
    from dataclasses import replace

    from ssgrep.accounting.collect.frequency import FrequencyView
    from ssgrep.accounting.ledger.ledger_contracts import FrequencyAssignment

    old = FrequencyView(
        "freq-old",
        basis=collector_frequency.BASIS_CANDIDATES,
        doc_count=2,
        df={RARE.key: 1},
    )
    current = replace(old, version_id="freq-current", df={COMMON.key: 2})
    assignments = FrequencyAssignments(
        current,
        assignments=(
            FrequencyAssignment("accepted-old", "freq-old"),
            FrequencyAssignment("missing-old", "freq-missing"),
        ),
        loader=lambda _: old,
    )

    assert assignments.view_for("accepted-old") == old
    assert assignments.view_for("accepted-old").rarity_milli(RARE) is not None
    unavailable = assignments.view_for("missing-old")
    assert unavailable.unavailable_reason == "frequency_revision_unavailable"
    assert assignments.unavailable["missing-old"] == "frequency_revision_unavailable"
    assert assignments.view_for("missing-old") != current


def test_orphan_build_pages_are_removed_in_bounded_batches_before_index_use(
    tmp_path: Path, keyed_profile: bytes
) -> None:
    path = tmp_path / "index.sqlite"
    database(path, 1)
    directory = storage.open_profile_dir(PROFILE, "collector")
    try:
        for number in range(index_frequency.ORPHAN_DELETIONS + 1):
            storage.atomic_write(
                directory,
                index_frequency._page_name("terms", number),
                b'{"orphan":1}',
            )
    finally:
        os.close(directory)

    repository = MetadataRepository(path)
    try:
        first = view(repository)
        assert not first.known
        assert first.unavailable_reason == "frequency_build_orphan_cleanup_pending"
        directory = storage.open_profile_dir(PROFILE, "collector")
        try:
            leftovers = [
                name
                for name in storage.list_names(directory)
                if name.startswith("frequency-build-terms-")
            ]
        finally:
            os.close(directory)
        assert len(leftovers) == 1

        resumed = view(repository)
        assert resumed.known and resumed.doc_count == 1
        assert resumed.coverage == "complete_indexed_chunks"
    finally:
        repository.close()


def test_short_frequency_page_fails_closed_against_a_complete_page_control(tmp_path: Path) -> None:
    path = tmp_path / "index.sqlite"
    database(path, 1)
    fingerprint.load_key("frequency-control", create=True)
    fingerprint.load_key("frequency-short-page", create=True)
    control = MetadataRepository(path)
    try:
        complete = view(control, "frequency-control")
        assert complete.known and complete.doc_count == 1
    finally:
        control.close()

    class ShortPageRepository(MetadataRepository):
        def rows_page(self, *, start: int, limit: int) -> list[dict]:
            return []

    damaged = ShortPageRepository(path)
    try:
        rejected = view(damaged, "frequency-short-page")
        assert not rejected.known
        assert rejected.unavailable_reason == "frequency_stats_incompatible"
        assert rejected.rarity_milli(RARE) is None
    finally:
        damaged.close()


def test_index_view_requires_stable_authority_identity_even_for_cached_view(
    tmp_path: Path, keyed_profile: bytes
) -> None:
    path = tmp_path / "index.sqlite"
    database(path, 511)
    repository = MetadataRepository(path)
    try:
        approved = view(repository)
        assert approved.known and approved.rarity_milli(RARE) == 8000

        missing_identity = read_index_view(
            PROFILE,
            repository=repository,
            authority=lambda cwd: cwd == "/approved",
            cached=approved,
        )
        assert missing_identity.unavailable_reason == "frequency_scope_unapproved"
        assert missing_identity.rarity_milli(RARE) is None

        reused = read_index_view(
            PROFILE,
            repository=repository,
            authority=lambda cwd: cwd == "/approved",
            authority_id="approved-v1",
            cached=approved,
        )
        assert reused is approved
    finally:
        repository.close()


@pytest.mark.parametrize(
    ("transition", "expected"),
    [
        ("generation_removed", "frequency_generation_unknown"),
        ("index_rebuilding", "frequency_index_not_ready"),
    ],
)
def test_index_view_fails_closed_after_ready_metadata_is_withdrawn(
    tmp_path: Path, keyed_profile: bytes, transition: str, expected: str
) -> None:
    path = tmp_path / "index.sqlite"
    database(path, 511)
    repository = MetadataRepository(path)
    try:
        ready = view(repository)
        assert ready.known and ready.rarity_milli(RARE) == 8000

        writer = sqlite3.connect(path)
        if transition == "generation_removed":
            writer.execute("UPDATE metadata SET value='' WHERE key='last_index_time'")
        else:
            writer.execute("UPDATE metadata SET value='building' WHERE key='index_state'")
            writer.execute("UPDATE metadata SET value='generation-2' WHERE key='last_index_time'")
        writer.commit()
        writer.close()

        unavailable = view(repository)
        assert unavailable.unavailable_reason == expected
        assert unavailable.rarity_milli(RARE) is None
    finally:
        repository.close()


def test_index_view_closes_detached_metadata_snapshot_after_success(
    tmp_path: Path, keyed_profile: bytes
) -> None:
    path = tmp_path / "index.sqlite"
    database(path, 511)

    class TrackedSnapshot(MetadataRepository):
        closed = False

        def close(self) -> None:
            self.closed = True
            super().close()

    class SnapshottingRepository(MetadataRepository):
        detached: TrackedSnapshot | None = None

        def snapshot(self, generation: str) -> MetadataRepository:
            detached = TrackedSnapshot(path)
            self.detached = detached
            detached.connection.execute("BEGIN")
            detached.get_meta("last_index_time")
            return detached

    repository = SnapshottingRepository(path)
    try:
        frozen = view(repository)
        assert frozen.known and frozen.rarity_milli(RARE) == 8000
        assert repository.detached is not None and repository.detached.closed
    finally:
        repository.close()


@pytest.mark.parametrize("corruption", ["empty_chunk_id", "non_string_cwd", "duplicate_chunk_id"])
def test_index_view_rejects_incompatible_chunk_rows_without_accepting_rarity(
    tmp_path: Path, keyed_profile: bytes, corruption: str
) -> None:
    path = tmp_path / "index.sqlite"
    database(path, 2)
    fingerprint.load_key("frequency-control", create=True)
    control = MetadataRepository(path)
    try:
        complete = view(control, "frequency-control")
        assert complete.known and complete.doc_count == 2
    finally:
        control.close()

    class IncompatibleRowsRepository(MetadataRepository):
        def rows_page(self, *, start: int, limit: int) -> list[dict]:
            rows = super().rows_page(start=start, limit=limit)
            if corruption == "empty_chunk_id":
                rows[0]["chunk_id"] = ""
            elif corruption == "non_string_cwd":
                rows[0]["cwd"] = 7
            else:
                rows[1]["chunk_id"] = rows[0]["chunk_id"]
            return rows

    repository = IncompatibleRowsRepository(path)
    try:
        rejected = view(repository)
        assert rejected.unavailable_reason == "frequency_stats_incompatible"
        assert rejected.rarity_milli(RARE) is None
    finally:
        repository.close()


@pytest.mark.parametrize(
    ("stats", "expected"),
    [
        ({"n_docs": True, "df": {}, "avgdl": 1}, "frequency_stats_damaged"),
        ({"n_docs": 1, "df": [], "avgdl": 1}, "frequency_stats_damaged"),
        ({"n_docs": 1, "df": {}, "avgdl": float("nan")}, "frequency_stats_damaged"),
        ({"n_docs": 1, "df": {}, "avgdl": -1}, "frequency_stats_damaged"),
        ({"n_docs": -1, "df": {}, "avgdl": 1}, "frequency_stats_damaged"),
        ({"n_docs": 0, "df": {}, "avgdl": 0}, "frequency_empty_corpus"),
        ([], "frequency_stats_damaged"),
    ],
)
def test_index_view_rejects_invalid_statistics_shapes_after_a_valid_control(
    tmp_path: Path,
    keyed_profile: bytes,
    stats: dict,
    expected: str,
) -> None:
    path = tmp_path / "index.sqlite"
    database(path, 1)
    fingerprint.load_key("frequency-control", create=True)
    repository = MetadataRepository(path)
    try:
        assert view(repository, "frequency-control").known
        writer = sqlite3.connect(path)
        writer.execute(
            "UPDATE metadata SET value=? WHERE key='lexical_stats'",
            (json.dumps(stats),),
        )
        writer.commit()
        writer.close()

        rejected = view(repository)
        assert rejected.unavailable_reason == expected
        assert rejected.rarity_milli(RARE) is None
    finally:
        repository.close()


def test_index_view_does_not_authorize_rarity_when_snapshot_is_unavailable(
    tmp_path: Path, keyed_profile: bytes
) -> None:
    path = tmp_path / "index.sqlite"
    database(path, 511)

    class SnapshotUnavailableRepository(MetadataRepository):
        def snapshot(self, generation: str) -> MetadataRepository:
            raise OSError("snapshot is unavailable")

    fingerprint.load_key("frequency-control", create=True)
    control = MetadataRepository(path)
    try:
        ready = view(control, "frequency-control")
        assert ready.known and ready.rarity_milli(RARE) == 8000
    finally:
        control.close()

    repository = SnapshotUnavailableRepository(path)
    try:
        unavailable = view(repository)
        assert unavailable.unavailable_reason == "frequency_snapshot_unavailable"
        assert unavailable.rarity_milli(RARE) is None
    finally:
        repository.close()


def test_index_view_reports_absent_repository_without_reusing_a_known_view(
    tmp_path: Path, keyed_profile: bytes
) -> None:
    path = tmp_path / "index.sqlite"
    database(path, 511)

    class AbsentRepository(MetadataRepository):
        def exists(self) -> bool:
            return False

    control = MetadataRepository(path)
    try:
        ready = view(control)
        assert ready.known and ready.rarity_milli(RARE) == 8000
    finally:
        control.close()

    repository = AbsentRepository(path)
    try:
        unavailable = read_index_view(
            PROFILE,
            repository=repository,
            authority=lambda cwd: cwd == "/approved",
            authority_id="approved-v1",
            cached=ready,
        )
        assert unavailable.unavailable_reason == "frequency_index_absent"
        assert unavailable.rarity_milli(RARE) is None
    finally:
        repository.close()


def test_index_view_does_not_reuse_cached_rarity_when_generation_read_fails(
    tmp_path: Path, keyed_profile: bytes
) -> None:
    path = tmp_path / "index.sqlite"
    database(path, 511)
    repository = MetadataRepository(path)
    try:
        cached = view(repository)
        assert cached.known and cached.rarity_milli(RARE) == 8000

        class UnreadableRepository(MetadataRepository):
            def get_meta(self, key: str) -> str | None:
                if key == "last_index_time":
                    raise OSError("metadata became unreadable")
                return super().get_meta(key)

        unreadable = UnreadableRepository(path)
        try:
            rejected = read_index_view(
                PROFILE,
                repository=unreadable,
                authority=lambda cwd: cwd == "/approved",
                authority_id="approved-v1",
                cached=cached,
            )
            assert not rejected.known
            assert rejected.unavailable_reason == "frequency_index_unreadable"
            assert rejected.rarity_milli(RARE) is None
            assert cached.rarity_milli(RARE) == 8000
        finally:
            unreadable.close()
    finally:
        repository.close()


@pytest.fixture
def keyed_profile() -> bytes:
    return fingerprint.load_key(PROFILE, create=True)


@pytest.fixture(autouse=True)
def private_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


@pytest.mark.parametrize(
    "documents,expected",
    [
        (199, (Scenario.CENTRAL, Scenario.OPTIMISTIC)),
        (200, (Scenario.CENTRAL, Scenario.OPTIMISTIC)),
        (511, (Scenario.CONSERVATIVE, Scenario.CENTRAL, Scenario.OPTIMISTIC)),
    ],
)
def test_singleton_feasibility_independent_of_former_200_floor(
    tmp_path: Path, keyed_profile: bytes, documents: int, expected: tuple[Scenario, ...]
) -> None:
    path = tmp_path / "index.sqlite"
    database(path, documents)
    repository = MetadataRepository(path)
    frozen = view(repository)
    assert frozen.known
    rarity = frozen.rarity_milli(RARE)
    assert rarity is not None and abs(rarity - math.log2((documents + 1) / 2) * 1000) < 2
    assert eligible_scenarios(rarity) == expected
    assert frozen.rarity_milli(COMMON) is None
    assert frozen.rarity_evidence(COMMON).value == 0
    assert frozen.rarity_evidence(COMMON).reason == "rarity_bound_inconclusive"
    assert THRESHOLD_MILLI == {
        Scenario.CONSERVATIVE: 8000,
        Scenario.CENTRAL: 6000,
        Scenario.OPTIMISTIC: 4000,
    }
    assert not frozen.calibrated
    repository.close()


def test_mapping_provenance_and_keyed_storage(tmp_path: Path, keyed_profile: bytes) -> None:
    path = tmp_path / "index.sqlite"
    database(path, 511)
    repository = MetadataRepository(path)
    before = path.read_bytes()
    frozen = view(repository)
    assert frozen.document_unit == "indexed_chunk"
    assert frozen.normalization == "lexical_ascii_lower_v1"
    assert frozen.profile_scope == PROFILE and frozen.authority_id == "approved-v1"
    assert frozen.generation == "generation-1" and len(frozen.corpus_digest) == 64
    assert frozen.coverage == "complete_indexed_chunks"
    assert frozen.rarity_milli(LATER) is None
    for token in [
        Token("path", "src/review_rare_identifier.py"),
        Token("identifier", "Review_Rare_Identifier"),
    ]:
        bound = frozen.rarity_evidence(token)
        assert bound.kind == "lower_bound" and bound.value == 8000 and bound.upper_df == 1
    for token in [
        Token("flag", "--review-rare-identifier"),
        Token("identifier", "review.rare_identifier"),
        Token("error_code", "ERR_REVIEW"),
        Token("identifier", "réview_identifier"),
    ]:
        assert frozen.rarity_milli(token) is None
    stored = json.dumps(to_plain(frozen), sort_keys=True)
    assert RARE.text not in stored and COMMON.text not in stored
    with pytest.raises(TypeError):
        frozen.df["changed"] = 1
    repository.close()
    assert path.read_bytes() == before


def test_mixed_organization_corpus_is_not_authority(tmp_path: Path) -> None:
    fingerprint.load_key("employer", create=True)
    path = tmp_path / "index.sqlite"
    database(path, 511, mixed=True)
    repository = MetadataRepository(path)
    rejected = view(repository, "employer")
    assert not rejected.known and rejected.unavailable_reason == "frequency_scope_unapproved"
    assert rejected.rarity_milli(RARE) is None
    repository.close()


@pytest.mark.parametrize(
    "raw,reason",
    [
        (None, "frequency_stats_absent"),
        ("{", "frequency_stats_damaged"),
        (json.dumps({"n_docs": 511, "avgdl": 1, "df": {RARE.text: 0}}), "frequency_stats_damaged"),
        (
            json.dumps({"n_docs": 511, "avgdl": 1, "df": {RARE.text: 512}}),
            "frequency_stats_damaged",
        ),
        (
            json.dumps({"n_docs": 510, "avgdl": 1, "df": {RARE.text: 1}}),
            "frequency_stats_incompatible",
        ),
    ],
)
def test_missing_damaged_incompatible_never_rate_absence(
    tmp_path: Path, keyed_profile: bytes, raw: str | None, reason: str
) -> None:
    path = tmp_path / "index.sqlite"
    database(path, 511)
    writer = sqlite3.connect(path)
    if raw is None:
        writer.execute("DELETE FROM metadata WHERE key='lexical_stats'")
    else:
        writer.execute("UPDATE metadata SET value=? WHERE key='lexical_stats'", (raw,))
    writer.commit()
    writer.close()
    repository = MetadataRepository(path)
    frozen = view(repository)
    assert not frozen.known and frozen.unavailable_reason == reason
    assert frozen.rarity_milli(RARE) is None
    repository.close()


def test_refresh_without_manual_repin_preserves_old_revision_bytes(
    tmp_path: Path, keyed_profile: bytes
) -> None:
    path = tmp_path / "index.sqlite"
    database(path, 511)
    repository = MetadataRepository(path)

    def resolve(pinned=None, *, repin=False):
        return collector_frequency.resolve(
            PROFILE,
            pinned=pinned,
            last_documents=None,
            evidence=lambda: pytest.fail("must not scan/copy retained plaintext corpus"),
            scan=False,
            repin=repin,
            index_view=lambda: view(repository),
        )

    first = resolve()
    old = first.view
    assert old.rarity_milli(LATER) is None
    directory = collector_frequency.storage.data_root() / "accounting" / PROFILE / "collector"
    old_file = directory / collector_frequency.view_file_name(old.version_id)
    old_bytes = old_file.read_bytes()
    assert resolve(old).view.version_id == old.version_id
    writer = sqlite3.connect(path)
    writer.execute(
        "UPDATE metadata SET value=? WHERE key='lexical_stats'",
        (
            json.dumps(
                {"n_docs": 511, "avgdl": 3.0, "df": {RARE.text: 1, COMMON.text: 511, LATER.text: 1}}
            ),
        ),
    )
    writer.execute("UPDATE metadata SET value='generation-2' WHERE key='last_index_time'")
    writer.commit()
    writer.close()
    refreshed = resolve(old)
    assert refreshed.transition == "prospective"
    assert refreshed.view.version_id != old.version_id
    assert refreshed.previous_view_id == old.version_id
    assert refreshed.view.rarity_milli(LATER) == 8000
    assert old.rarity_milli(LATER) is None and old_file.read_bytes() == old_bytes
    assert collector_frequency.load_pinned(PROFILE) == refreshed.view
    assert collector_frequency.load_view(PROFILE, old.version_id) == old
    assert collector_frequency.load_view(PROFILE, refreshed.view.version_id) == refreshed.view
    replacement = resolve(old, repin=True)
    assert replacement.transition == "replacement"
    assert replacement.revision_name == f"frequency-recompute:{refreshed.view.version_id}"
    repository.close()


def test_missing_key_and_inactive_profile_are_unavailable_without_creating_key(
    tmp_path: Path,
) -> None:
    path = tmp_path / "index.sqlite"
    database(path, 511)
    repository = MetadataRepository(path)
    frozen = view(repository)
    assert frozen.unavailable_reason == "frequency_key_unavailable"
    assert fingerprint.load_key(PROFILE, create=False) is None
    assert (
        read_index_view("employer", repository=repository).unavailable_reason
        == "frequency_scope_unapproved"
    )
    assert fingerprint.load_key("employer", create=False) is None
    repository.close()


def test_unavailable_current_generation_does_not_authorize_stale_pin(
    tmp_path: Path, keyed_profile: bytes
) -> None:
    path = tmp_path / "index.sqlite"
    database(path, 511)
    repository = MetadataRepository(path)
    old = view(repository)
    collector_frequency.pin(PROFILE, old)
    directory = collector_frequency.storage.data_root() / "accounting" / PROFILE / "collector"
    pointer = directory / collector_frequency.POINTER
    before = pointer.read_bytes()
    writer = sqlite3.connect(path)
    writer.execute("DELETE FROM metadata WHERE key='lexical_stats'")
    writer.commit()
    writer.close()
    unavailable = collector_frequency.resolve(
        PROFILE,
        pinned=old,
        last_documents=511,
        evidence=lambda: (),
        scan=False,
        repin=False,
        index_view=lambda: view(repository),
    )
    assert unavailable.transition == "unavailable"
    assert unavailable.view.rarity_milli(RARE) is None
    assert unavailable.coverage()["reason"] == "frequency_stats_absent"
    assert pointer.read_bytes() == before
    assert collector_frequency.load_pinned(PROFILE) == old
    repository.close()


def test_generation_change_during_read_cannot_publish_mixed_statistics(
    tmp_path: Path, keyed_profile: bytes
) -> None:
    path = tmp_path / "index.sqlite"
    database(path, 511)

    class RacingRepository(MetadataRepository):
        reads = 0

        def get_meta(self, key: str) -> str | None:
            value = super().get_meta(key)
            if key == "last_index_time":
                self.reads += 1
                if self.reads == 2:
                    return "newer-generation"
            return value

    repository = RacingRepository(path)
    frozen = view(repository)
    assert not frozen.known and frozen.unavailable_reason == "frequency_generation_changed"
    assert frozen.rarity_milli(RARE) is None
    repository.close()


def test_scope_identity_changes_revision_even_when_df_and_generation_are_equal(
    tmp_path: Path, keyed_profile: bytes
) -> None:
    path = tmp_path / "index.sqlite"
    database(path, 511)
    repository = MetadataRepository(path)
    first = view(repository)
    same = view(repository)
    authorized_revision = read_index_view(
        PROFILE,
        repository=repository,
        authority=lambda cwd: cwd == "/approved",
        authority_id="approved-v2",
    )
    assert same.version_id == first.version_id
    assert authorized_revision.generation == first.generation
    assert authorized_revision.version_id != first.version_id
    assert authorized_revision.rarity_milli(RARE) == first.rarity_milli(RARE) == 8000
    repository.close()


def test_cross_profile_frozen_index_view_cannot_be_pinned(
    tmp_path: Path, keyed_profile: bytes
) -> None:
    path = tmp_path / "index.sqlite"
    database(path, 511)
    repository = MetadataRepository(path)
    frozen = view(repository)
    with pytest.raises(ValueError, match="another profile"):
        collector_frequency.pin("employer", frozen)
    assert collector_frequency.load_pinned("employer") is None
    repository.close()


def test_unchanged_generation_reuses_frozen_input_without_df_or_corpus_projection(
    tmp_path: Path, keyed_profile: bytes
) -> None:
    path = tmp_path / "index.sqlite"
    database(path, 511)
    repository = MetadataRepository(path)
    frozen = view(repository)
    repository.close()

    class IdleRepository(MetadataRepository):
        def get_meta(self, key: str) -> str | None:
            assert key != "lexical_stats", "unchanged generation must not reread full stats"
            return super().get_meta(key)

        def rows_page(self, *args, **kwargs):
            pytest.fail("unchanged generation must not project the corpus again")

        def count(self, *args, **kwargs):
            pytest.fail("unchanged generation must not count the corpus again")

    idle = IdleRepository(path)
    reused = read_index_view(
        PROFILE,
        repository=idle,
        authority=lambda cwd: True,
        authority_id="approved-v1",
        cached=frozen,
    )
    assert reused is frozen
    idle.close()


def test_erased_scope_tombstone_cannot_be_reauthorized_by_automatic_refresh_or_repin(
    tmp_path: Path, keyed_profile: bytes
) -> None:
    import os

    directory = collector_frequency.storage.open_profile_dir(PROFILE, "collector")
    try:
        collector_frequency.storage.atomic_write(directory, "frequency-purged.json", b"{}")
    finally:
        os.close(directory)
    outcome = collector_frequency.resolve(
        PROFILE,
        pinned=None,
        last_documents=None,
        evidence=lambda: (),
        scan=True,
        repin=True,
        index_view=lambda: pytest.fail("purged scope cannot regain authority from same index"),
    )
    assert outcome.view.unavailable_reason == "frequency_scope_erased"
    assert not outcome.view.known


def test_late_native_join_inherits_journal_generation_not_new_statistics(
    tmp_path: Path, keyed_profile: bytes
) -> None:
    from ssgrep.accounting.collect.frequency import FrequencyAssignments

    path = tmp_path / "index.sqlite"
    database(path, 511)
    repository = MetadataRepository(path)
    old = view(repository)
    frozen = FrequencyAssignments(old)
    frozen.view_for("journal:accepted-before-native")
    writer = sqlite3.connect(path)
    writer.execute(
        "UPDATE metadata SET value=? WHERE key='lexical_stats'",
        (json.dumps({"n_docs": 511, "avgdl": 1.0, "df": {LATER.text: 1}}),),
    )
    writer.execute("UPDATE metadata SET value='generation-2' WHERE key='last_index_time'")
    writer.commit()
    writer.close()
    current = view(repository)
    resumed = FrequencyAssignments(current, assignments=frozen.records(), loader=lambda _: old)
    resumed.bind("late-native", ["accepted-before-native"])
    assert resumed.view_for("late-native").version_id == old.version_id
    assert resumed.view_for("late-native").rarity_milli(LATER) is None
    resumed.bind("genuine-new-lookup", [])
    assert resumed.view_for("genuine-new-lookup").rarity_milli(LATER) == 8000
    repository.close()


def test_native_join_binds_only_unambiguous_frozen_frequency_aliases():
    from ssgrep.accounting.collect.frequency import FrequencyView
    from ssgrep.accounting.ledger.ledger_contracts import FrequencyAssignment

    current = FrequencyView.unknown("frequency_index_absent", profile=PROFILE)
    same = FrequencyAssignments(
        current,
        assignments=(
            FrequencyAssignment("journal:first", "freq-old", profile_scope=PROFILE),
            FrequencyAssignment("journal:second", "freq-old", profile_scope=PROFILE),
        ),
    )
    same.bind("native-same", ["first", "second"])
    assert same.records(("native-same",))[0].view_id == "freq-old"

    conflicting = FrequencyAssignments(
        current,
        assignments=(
            FrequencyAssignment("journal:first", "freq-old", profile_scope=PROFILE),
            FrequencyAssignment("journal:second", "freq-new", profile_scope=PROFILE),
        ),
    )
    conflicting.bind("native-conflict", ["first", "second"])
    assert conflicting.view_for("native-conflict").unavailable_reason == (
        "frequency_assignment_ambiguous"
    )
    assert conflicting.records(("native-conflict",))[0].view_id == "frequency-unknown"


def test_whole_path_flag_and_casefold_component_proofs_never_claim_exact_df(
    tmp_path: Path, keyed_profile: bytes
) -> None:
    path = tmp_path / "index.sqlite"
    database(path, 511)
    writer = sqlite3.connect(path)
    writer.execute(
        "UPDATE metadata SET value=? WHERE key='lexical_stats'",
        (
            json.dumps(
                {
                    "n_docs": 511,
                    "avgdl": 1.0,
                    "df": {"danger": 1, "src": 511, "py": 511, "common": 511},
                }
            ),
        ),
    )
    writer.commit()
    writer.close()
    repository = MetadataRepository(path)
    frozen = view(repository)
    for token in (
        Token("path", "src/danger.py"),
        Token("flag", "--danger-missing"),
        Token("identifier", "DANGER"),
        Token("error_code", "DANGER"),
    ):
        proof = frozen.rarity_evidence(token)
        assert proof.kind == "lower_bound" and proof.upper_df == 1 and proof.value == 8000
    assert frozen.rarity_milli(Token("identifier", "DangerFlag")) is None
    common = frozen.rarity_evidence(Token("flag", "--common-absent"))
    assert common.kind == "lower_bound" and common.value == 0
    assert common.reason == "rarity_bound_inconclusive"
    assert frozen.rarity_milli(Token("flag", "--common-absent")) is None
    repository.close()


# ---- Survivor source: test_collection.py ----
"""Frozen rarity inputs survive new lookups in the same root and collector restart."""


OLD_TERM = "review_initial_identifier"
LATER_TERM = "review_later_identifier"


def body(term: str) -> str:
    return results_json(card("claude:historical:ep:0", term))


def call(call_id: str, term: str, second: int) -> list[dict]:
    return [
        claude_tool_use(
            "same-root",
            call_id,
            MCP,
            {"query": "investigate deployment"},
            ts=ts(12, 9, 0, second),
            message_id=call_id + "-message",
        ),
        claude_tool_result("same-root", call_id, body(term), ts=ts(12, 9, 0, second + 2)),
    ]


def action(call_id: str, term: str, minute: int) -> list[dict]:
    return [
        claude_tool_use(
            "same-root",
            call_id,
            "Bash",
            {"command": f"grep {term} deploy.log"},
            ts=ts(12, 9, minute),
            message_id=call_id + "-message",
        ),
        claude_tool_result("same-root", call_id, "ok", ts=ts(12, 9, minute, 2)),
    ]


def state():
    checkpoint = vs.checkpoint()
    assert checkpoint is not None
    return collector_state.load_state(PROFILE, checkpoint, checkpoint.methodology)


def test_same_root_new_lookup_uses_new_generation_old_no_reuse_lookup_stays_frozen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vs.enable()
    vs.start_capture()
    path = tmp_path / "index.sqlite"
    database(path, 511)
    repository = MetadataRepository(path)
    original = index_frequency.read_index_view

    def current(profile, **kwargs):
        return original(
            profile,
            repository=repository,
            authority=lambda cwd: cwd == "/approved",
            authority_id="approved-v1",
            cached=kwargs.get("cached"),
        )

    monkeypatch.setattr(index_frequency, "read_index_view", current)
    scene.write_transcript(
        tmp_path,
        "historical",
        source_session("historical", f"{OLD_TERM} {LATER_TERM}", when_day=11),
        scene.at(hour=9),
    )
    transcript = scene.write_transcript(
        tmp_path,
        "same-root",
        [
            claude_prompt("same-root", "fix deployment", uuid="original-user", ts=ts(12, 9)),
            *call("initial-lookup", OLD_TERM, 10),
        ],
        scene.at(hour=9),
    )
    scene.journal_events(
        scene.execution(
            "n-initial", ts(12, 9, 0, 11), text="investigate deployment", transport="mcp"
        )
    )
    scene.collect(scene.at(hour=10))
    first = state()
    native_assignments = [
        row for row in first.frequency_assignments if not row.invocation_id.startswith("journal:")
    ]
    assert len(native_assignments) == 1  # frozen even without candidate or reuse
    old_assignment = native_assignments[0]
    old_bytes = canonical_bytes(old_assignment)
    old_view = collector_frequency.load_view(PROFILE, old_assignment.view_id)
    method = scene.current().methodology_id
    writer = sqlite3.connect(path)
    writer.execute(
        "UPDATE metadata SET value=? WHERE key='lexical_stats'",
        (
            json.dumps(
                {
                    "n_docs": 511,
                    "avgdl": 3.0,
                    "df": {OLD_TERM: 1, LATER_TERM: 1},
                }
            ),
        ),
    )
    writer.execute("UPDATE metadata SET value='generation-2' WHERE key='last_index_time'")
    writer.commit()
    writer.close()
    scene.append_transcript(
        transcript,
        [
            *action("old-action", OLD_TERM, 1),
            *call("later-lookup", LATER_TERM, 20),
            *action("later-action", LATER_TERM, 2),
        ],
        scene.at(hour=11),
    )
    scene.journal_events(
        scene.execution("n-later", ts(12, 9, 0, 21), text="investigate deployment", transport="mcp")
    )
    scene.collect(scene.at(hour=12))
    second = state()
    assert scene.current().methodology_id == method  # stats are inputs, not model namespaces
    assignments = {
        row.invocation_id: row
        for row in second.frequency_assignments
        if not row.invocation_id.startswith("journal:")
    }
    assert canonical_bytes(assignments[old_assignment.invocation_id]) == old_bytes
    assert len(assignments) == 2
    newer = next(
        row for row in assignments.values() if row.invocation_id != old_assignment.invocation_id
    )
    assert newer.view_id != old_assignment.view_id
    assert collector_frequency.load_view(PROFILE, old_assignment.view_id) == old_view
    assert any(
        dict(candidate.provenance)["frequency"] == newer.view_id
        and candidate.eligibility.value == "supported"
        for candidate in second.ledger.candidates
    )
    assert any(
        dict(candidate.provenance)["frequency"] == old_assignment.view_id
        and "rarity_unknown" in candidate.eligibility_reasons
        for candidate in second.ledger.candidates
    )
    before = vs.ledger_digest()
    fingerprint.forget_keys()  # fresh process-equivalent key/cache and persisted state reload
    scene.collect(scene.at(hour=12))
    assert vs.ledger_digest() == before
    replay = FrequencyAssignments(
        current(PROFILE),
        assignments=second.frequency_assignments,
        loader=lambda view_id: collector_frequency.load_view(PROFILE, view_id),
    )
    loaded = load_evidence()
    from ssgrep.accounting.capture.journal_reader import read_journal

    full = reduce_profile(
        PROFILE,
        loaded.evidence,
        read_journal(PROFILE).events,
        policy=load_profile_policy(PROFILE),
        frequency=current(PROFILE),
        frequency_assignments=replay,
        fingerprint_key=fingerprint.load_key(PROFILE, create=False),
    )
    assert full.frequency_assignments == second.frequency_assignments
    assert {record.frequency_view_id for record in full.candidates} == {
        old_assignment.view_id,
        newer.view_id,
    }
    scene.collect(scene.at(hour=13), CollectOptions(recompute=True))
    replaced = state()
    assert all(row.view_id == newer.view_id for row in replaced.frequency_assignments)
    assert all(
        row.revision_name == f"frequency-recompute:{newer.view_id}"
        for row in replaced.frequency_assignments
    )
    assert scene.current().methodology_id == method
    repository.close()


def test_pending_frequency_completion_refines_receipts_without_counting_usage_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ssgrep.accounting.collect import frequency_index as index_frequency_build

    vs.enable()
    vs.start_capture()
    path = tmp_path / "pending-index.sqlite"
    database(path, 511)
    writer = sqlite3.connect(path)
    writer.execute(
        "UPDATE metadata SET value=? WHERE key='lexical_stats'",
        (json.dumps({"n_docs": 511, "avgdl": 3.0, "df": {OLD_TERM: 1}}),),
    )
    writer.commit()
    writer.close()
    repository = MetadataRepository(path)
    original = index_frequency.read_index_view
    monkeypatch.setattr(index_frequency_build, "VISITS", 128)

    def current(profile, **kwargs):
        return original(
            profile,
            repository=repository,
            authority=lambda cwd: cwd == "/approved",
            authority_id="approved-v1",
            cached=kwargs.get("cached"),
        )

    monkeypatch.setattr(index_frequency, "read_index_view", current)
    scene.write_transcript(
        tmp_path,
        "historical",
        source_session("historical", OLD_TERM, when_day=11),
        scene.at(hour=9),
    )
    scene.write_transcript(
        tmp_path,
        "same-root",
        [
            claude_prompt("same-root", "fix deployment", uuid="pending-user", ts=ts(12, 9)),
            *call("pending-lookup", OLD_TERM, 10),
            *action("pending-action", OLD_TERM, 1),
        ],
        scene.at(hour=9),
    )
    scene.journal_events(
        scene.execution(
            "n-pending", ts(12, 9, 0, 11), text="investigate deployment", transport="mcp"
        )
    )
    scene.collect(scene.at(hour=10))
    pending = state()
    assert pending.frequency_assignments
    assert all(
        row.unavailable_reason == "frequency_build_pending" for row in pending.frequency_assignments
    )
    usage = scene.counter_total(scene.current())
    assert usage > 0
    assert not any(
        candidate.eligibility.value == "supported" for candidate in pending.ledger.candidates
    )
    for _ in range(16):
        fingerprint.forget_keys()
        scene.collect(scene.at(hour=11))
        completed = state()
        if completed.frequency_assignments and all(
            not row.unavailable_reason for row in completed.frequency_assignments
        ):
            break
    else:
        pytest.fail("pending collection inputs never acquired their completed generation")
    assert scene.counter_total(scene.current()) == usage
    assert {row.invocation_id for row in completed.frequency_assignments} == {
        row.invocation_id for row in pending.frequency_assignments
    }
    assert all(
        row.revision_name == f"frequency-build-complete:{row.view_id}"
        for row in completed.frequency_assignments
    )
    assert any(
        candidate.eligibility.value == "supported" for candidate in completed.ledger.candidates
    )
    digest = vs.ledger_digest()
    scene.collect(scene.at(hour=11))
    assert vs.ledger_digest() == digest
    assert scene.counter_total(scene.current()) == usage
    repository.close()


# ---- Survivor source: test_generation_cache.py ----
"""Manifest-version changes invalidate cached rarity even without timestamp changes."""


class VersionedRepository(MetadataRepository):
    """SQLite's real data-version mirrors the repository table-version read interface."""

    def table(self, name: str):
        assert name in ("metadata", "chunks")
        return self

    @property
    def version(self) -> int:
        return self.connection.execute("PRAGMA data_version").fetchone()[0]


def test_damaged_stats_cannot_hide_behind_unchanged_last_index_time(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    fingerprint.load_key(PROFILE, create=True)
    path = tmp_path / "index.sqlite"
    database(path, 511)
    repository = VersionedRepository(path)

    def current(cached=None):
        return read_index_view(
            PROFILE,
            repository=repository,
            authority=lambda cwd: True,
            authority_id="approved-v1",
            cached=cached,
        )

    first = current()
    assert first.rarity_milli(RARE) == 8000
    assert "@metadata:" in first.generation and "chunks:" in first.generation
    writer = sqlite3.connect(path)
    writer.execute("UPDATE metadata SET value='{' WHERE key='lexical_stats'")
    writer.commit()
    writer.close()
    damaged = current(first)
    assert not damaged.known and damaged.unavailable_reason == "frequency_stats_damaged"
    assert damaged.rarity_milli(RARE) is None
    assert first.rarity_milli(RARE) == 8000  # frozen prior input remains intact
    repository.close()


# ---- Survivor source: test_key_revision.py ----
"""Key replacement is an input revision, never automatic rerating of old lookups."""


def test_warmed_key_rotation_preserves_old_lookup_assignment_and_rates_new_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    first_key = fingerprint.load_key(PROFILE, create=True)
    path = tmp_path / "index.sqlite"
    database(path, 511)
    repository = MetadataRepository(path)

    def current(cached=None):
        return read_index_view(
            PROFILE,
            repository=repository,
            authority=lambda cwd: True,
            authority_id="approved-v1",
            cached=cached,
        )

    old = current()
    original = FrequencyAssignments(old, fingerprint_key_id=fingerprint.key_id(first_key))
    original.bind("old-native-lookup", [])
    assignment = original.records()[0]
    directory = storage.open_profile_dir(PROFILE)
    try:
        storage.atomic_write(directory, fingerprint.KEY_FILE, b"r" * fingerprint.KEY_BYTES)
    finally:
        os.close(directory)
    rotated_key = fingerprint.load_key(PROFILE, create=False)
    assert rotated_key != first_key
    refreshed = current(old)
    assert refreshed.version_id != old.version_id
    resumed = FrequencyAssignments(
        refreshed,
        assignments=original.records(),
        loader=lambda _: old,
        fingerprint_key_id=fingerprint.key_id(rotated_key),
    )
    unavailable = resumed.view_for("old-native-lookup")
    assert unavailable.unavailable_reason == "fingerprint_key_revision_unavailable"
    assert resumed.unavailable["old-native-lookup"] == "fingerprint_key_revision_unavailable"
    assert resumed.records()[0] == assignment
    resumed.bind("new-native-lookup", [])
    assert resumed.view_for("new-native-lookup").rarity_milli(RARE) == 8000
    repository.close()


# ---- Survivor source: test_repository.py ----
"""Real Lance checkout/direct-take semantics, isolated from the application database."""


def test_checkout_keeps_metadata_and_direct_row_offsets_after_live_append(tmp_path, monkeypatch):
    path = tmp_path / "isolated.lance"
    database = lancedb.connect(str(path), read_consistency_interval=timedelta(seconds=0))
    metadata = database.create_table(
        "metadata",
        [
            {"key": "last_index_time", "value": "epoch-1"},
            {"key": "index_state", "value": "ready"},
        ],
    )
    chunks = database.create_table(
        "chunks", [{"chunk_id": f"chunk-{i}", "cwd": "/approved"} for i in range(20)]
    )

    def private_store():
        store = LanceStore()
        store.path = path
        return store

    monkeypatch.setattr(index_frequency_repository, "LanceStore", private_store)
    live = index_frequency_repository.IndexRepository()
    generation = _generation(live)
    assert generation is not None
    frozen = live.snapshot(generation)
    try:
        before = frozen.rows_page(start=7, limit=5)
        assert [row["chunk_id"] for row in before] == [f"chunk-{i}" for i in range(7, 12)]
        chunks.add([{"chunk_id": "new-chunk", "cwd": "/foreign"}])
        metadata.update(where="key = 'last_index_time'", values={"value": "epoch-2"})
        assert _generation(live) != generation and live.count("chunks") == 21
        assert _generation(frozen) == generation and frozen.count("chunks") == 20
        assert frozen.rows_page(start=7, limit=5) == before
        assert frozen.rows_page(start=19, limit=1) == [{"chunk_id": "chunk-19", "cwd": "/approved"}]
        assert frozen.get_meta("last_index_time") == "epoch-1"
    finally:
        frozen.close()
        live.close()


@pytest.mark.parametrize("duplicate", [False, True], ids=["distinct", "exact-duplicate"])
def test_public_frequency_requires_unique_real_lance_chunk_identities(
    tmp_path, monkeypatch, duplicate
):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    fingerprint.load_key(PROFILE, create=True)
    path = tmp_path / "identity.lance"
    database = lancedb.connect(str(path), read_consistency_interval=timedelta(seconds=0))
    database.create_table(
        "metadata",
        [
            {"key": "last_index_time", "value": "identity-epoch"},
            {"key": "index_state", "value": "ready"},
            {
                "key": "lexical_stats",
                "value": json.dumps({"n_docs": 511, "avgdl": 1, "df": {RARE.text: 1}}),
            },
        ],
    )
    rows = [{"chunk_id": f"chunk-{i}", "cwd": "/approved"} for i in range(511)]
    if duplicate:
        rows[1]["chunk_id"] = rows[0]["chunk_id"]
    database.create_table("chunks", rows)

    def private_store():
        store = LanceStore()
        store.path = path
        return store

    monkeypatch.setattr(index_frequency_repository, "LanceStore", private_store)
    repository = index_frequency_repository.IndexRepository()
    try:
        result = read_index_view(
            PROFILE,
            repository=repository,
            authority=lambda cwd: cwd == "/approved",
            authority_id="approved-v1",
        )
        if duplicate:
            assert not result.known and result.unavailable_reason == "frequency_stats_incompatible"
            assert result.rarity_milli(RARE) is None
            descriptor = storage.open_profile_dir(PROFILE, "collector", create=False)
            try:
                encoded = storage.read_file(descriptor, "frequency-build-head.json", 1024 * 1024)
                assert encoded is not None
                head = json.loads(encoded)
            finally:
                os.close(descriptor)
            assert head["job"]["chunk_cursor"] == 1
            assert head["job"]["seen"][0]["records"] == 1
        else:
            assert result.known and result.doc_count == 511
            assert len(result.df) == 1 and result.rarity_milli(RARE) == 8000
    finally:
        repository.close()


# ---- Survivor source: test_retention.py ----
"""Expired rarity inputs remain while exact canonical replay still references them."""


def test_bounded_gc_protects_lookup_archive_references_and_current_pointer(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    fingerprint.load_key(PROFILE, create=True)
    path = tmp_path / "index.sqlite"
    database(path, 511)
    repository = MetadataRepository(path)
    views = []
    for generation in ("generation-1", "generation-2", "generation-3"):
        writer = sqlite3.connect(path)
        writer.execute("UPDATE metadata SET value=? WHERE key='last_index_time'", (generation,))
        writer.commit()
        writer.close()
        view = read_index_view(
            PROFILE, repository=repository, authority=lambda cwd: True, authority_id="approved-v1"
        )
        collector_frequency.pin(PROFILE, view)
        views.append(view)
    directory = storage.data_root() / "accounting" / PROFILE / "collector"
    names = [collector_frequency.view_file_name(view.version_id) for view in views]
    age = datetime(2025, 1, 1, tzinfo=UTC).timestamp()
    for name in names:
        os.utime(directory / name, (age, age))
    result = prune_views(
        PROFILE,
        candidates=iter(names),
        referenced=[views[0].version_id],
        before_day="2026-01-01",
        limit=2,
    )
    assert (result.scanned, result.removed, result.retained) == (2, 1, 1)
    assert (directory / names[0]).is_file()  # live assignment or inactive canonical archive
    assert not (directory / names[1]).exists()  # expired, orphan, not current
    protected = prune_views(PROFILE, candidates=[names[2]], referenced=[], before_day="2026-01-01")
    assert protected.removed == 0 and protected.retained == 1
    assert collector_frequency.load_view(PROFILE, views[0].version_id) == views[0]
    assert collector_frequency.load_pinned(PROFILE) == views[2]
    repository.close()


def test_index_rarity_uses_the_rarest_valid_component_and_filters_short_components(
    tmp_path: Path, keyed_profile: bytes
) -> None:
    path = tmp_path / "index.sqlite"
    database(path, 511)
    repository = MetadataRepository(path)
    try:
        from dataclasses import replace

        frozen = replace(
            view(repository),
            df={
                collector_frequency.lexical_key(keyed_profile, "review"): 4,
                collector_frequency.lexical_key(keyed_profile, "rare"): 1,
                collector_frequency.lexical_key(keyed_profile, "identifier"): 3,
            },
        )
        evidence = frozen.rarity_evidence(Token("identifier", "x.review-rare-identifier"))
        assert evidence.kind == "lower_bound"
        assert evidence.upper_df == 1
        assert evidence.value is not None
        assert evidence.reason == ""
        assert frozen.rarity_evidence(Token("identifier", "éclair")).reason == (
            "frequency_mapping_unavailable"
        )
    finally:
        repository.close()


@pytest.mark.parametrize(
    ("numerator", "denominator", "expected"),
    [(3, 2, 584), (5, 4, 321), (9, 8, 169), (4, 3, 414), (7, 4, 806)],
)
def test_integer_logarithm_floor_is_stable_for_fractional_ratios(numerator, denominator, expected):
    assert collector_frequency.milli_log2(numerator, denominator) == expected
    exact = 1000 * math.log2(numerator / denominator)
    assert exact - 2 <= expected <= exact


def test_generation_without_table_versions_uses_metadata_and_absent_generation_is_unknown():
    from typing import Any

    class MetadataOnly:
        value = "immutable-generation"

        def get_meta(self, key: str) -> str | None:
            return self.value if key == "last_index_time" else None

        def exists(self) -> bool:
            return True

        def count(self, name: str) -> int:
            return 0

        def snapshot(self, generation: str):
            return self

        def rows_page(self, *, start: int, limit: int) -> list[dict[str, Any]]:
            return []

        def close(self) -> None:
            pass

    repository = MetadataOnly()
    assert _generation(repository) == "immutable-generation"
    repository.value = ""
    assert _generation(repository) is None


def collector_directory() -> int:
    return storage.open_profile_dir(PROFILE, "collector")


def write_collector_file(name: str, content: bytes, profile: str = PROFILE) -> None:
    directory = storage.open_profile_dir(profile, "collector")
    try:
        storage.atomic_write(directory, name, content)
    finally:
        os.close(directory)


def read_collector_file(name: str) -> bytes | None:
    directory = collector_directory()
    try:
        return storage.read_file(directory, name, 1024 * 1024)
    finally:
        os.close(directory)


def key_path() -> Path:
    return storage.data_root() / "accounting" / PROFILE / fingerprint.KEY_FILE


def test_exact_view_reports_absent_candidate_and_lexical_view_rates_nothing() -> None:
    exact = collector_frequency.build_view(["alpha_token one", "alpha_token two"])
    present = next(iter(exact.df))
    assert exact.rarity_evidence(Token("identifier", "never_seen_here")).reason == (
        "frequency_candidate_absent"
    )
    assert exact.df[present] >= 1 and exact.known

    lexical = collector_frequency.lexical_view({"b": 1, "a": 2}, 3)
    assert lexical.basis == collector_frequency.BASIS_LEXICAL
    assert list(lexical.df) == ["a", "b"]
    assert lexical.version_id.startswith("freq-lex-")
    assert collector_frequency.lexical_view({"a": 2, "b": 1}, 3).version_id == lexical.version_id
    assert collector_frequency.lexical_view({"a": 2, "b": 1}, 4).version_id != lexical.version_id
    assert not lexical.known
    assert lexical.rarity_evidence(RARE).reason == "frequency_unavailable"


def test_index_rarity_requires_the_exact_profile_key_used_to_freeze_the_view(
    tmp_path: Path, keyed_profile: bytes
) -> None:
    path = tmp_path / "index.sqlite"
    database(path, 511)
    repository = MetadataRepository(path)
    try:
        frozen = view(repository)
        assert frozen.rarity_evidence(RARE).reason == ""
        original = key_path().read_bytes()

        key_path().write_bytes(os.urandom(len(original)))
        assert frozen.rarity_evidence(RARE).reason == "frequency_key_unavailable"

        key_path().write_bytes(original[:-1])
        assert frozen.rarity_evidence(RARE).reason == "frequency_key_unavailable"

        key_path().unlink()
        assert frozen.rarity_evidence(RARE).reason == "frequency_key_unavailable"

        key_path().write_bytes(original)
        key_path().chmod(0o600)
        assert frozen.rarity_evidence(RARE).reason == ""
    finally:
        repository.close()


def test_load_view_rejects_invalid_missing_and_cross_profile_revisions(
    tmp_path: Path, keyed_profile: bytes
) -> None:
    path = tmp_path / "index.sqlite"
    database(path, 511)
    repository = MetadataRepository(path)
    try:
        frozen = view(repository)
    finally:
        repository.close()
    collector_frequency.pin(PROFILE, frozen)
    name = collector_frequency.view_file_name(frozen.version_id)
    content = read_collector_file(name)
    assert content is not None
    assert collector_frequency.load_view(PROFILE, frozen.version_id) == frozen

    with pytest.raises(collector_state.CollectError, match="no valid view"):
        collector_frequency.load_view(PROFILE, "../freq-escape")
    with pytest.raises(collector_state.CollectError, match="view is missing"):
        collector_frequency.load_view("never-created", frozen.version_id)

    write_collector_file(name, content, "employer")
    with pytest.raises(collector_state.CollectError, match="another profile"):
        collector_frequency.load_view("employer", frozen.version_id)
    assert collector_frequency.load_view(PROFILE, frozen.version_id) == frozen


def test_resolve_refuses_an_index_view_scoped_to_another_profile(
    tmp_path: Path, keyed_profile: bytes
) -> None:
    path = tmp_path / "index.sqlite"
    database(path, 511)
    repository = MetadataRepository(path)
    try:
        foreign = view(repository)
    finally:
        repository.close()
    assert foreign.known and foreign.profile_scope == PROFILE
    outcome = collector_frequency.resolve(
        "employer",
        pinned=None,
        last_documents=None,
        evidence=lambda: (),
        scan=False,
        repin=False,
        index_view=lambda: foreign,
    )
    assert not outcome.view.known
    assert outcome.view.unavailable_reason == "frequency_scope_unapproved"
    assert outcome.view.profile_scope == "employer"
    assert outcome.transition == "unavailable"
    assert collector_frequency.load_pinned("employer") is None


def test_assignments_reject_duplicates_and_missing_loader_and_model_drift_and_blank_recompute(
    tmp_path: Path, keyed_profile: bytes
) -> None:
    from dataclasses import replace

    from ssgrep.accounting.collect.frequency import FrequencyView
    from ssgrep.accounting.ledger.ledger_contracts import FrequencyAssignment

    old = FrequencyView("freq-old", collector_frequency.BASIS_CANDIDATES, doc_count=2, df={})
    current = replace(old, version_id="freq-current")
    with pytest.raises(ValueError, match="duplicate lookup frequency assignment"):
        FrequencyAssignments(
            current,
            assignments=(FrequencyAssignment("same", "freq-old"), FrequencyAssignment("same", "x")),
        )

    no_loader = FrequencyAssignments(current, assignments=(FrequencyAssignment("a", "freq-old"),))
    assert no_loader.view_for("a").unavailable_reason == "frequency_revision_unavailable"
    assert no_loader.unavailable["a"] == "frequency_revision_unavailable"

    other_model = replace(old, basis=collector_frequency.BASIS_INDEX)
    drifted = FrequencyAssignments(
        current,
        assignments=(FrequencyAssignment("a", "freq-old"),),
        loader=lambda _: other_model,
    )
    assert drifted.view_for("a").unavailable_reason == "frequency_model_incompatible"
    assert drifted.unavailable["a"] == "frequency_model_incompatible"
    same_model = FrequencyAssignments(
        current, assignments=(FrequencyAssignment("a", "freq-old"),), loader=lambda _: old
    )
    assert same_model.view_for("a") == old
    assert not same_model.unavailable

    before = same_model.records()
    with pytest.raises(ValueError, match="explicit revision name"):
        same_model.recompute("")
    assert same_model.records() == before


def test_assignments_without_a_readable_key_have_no_fingerprint_key_identity(
    tmp_path: Path, keyed_profile: bytes
) -> None:
    path = tmp_path / "index.sqlite"
    database(path, 511)
    repository = MetadataRepository(path)
    try:
        frozen = view(repository)
    finally:
        repository.close()
    assert FrequencyAssignments(frozen).fingerprint_key_id == fingerprint.key_id(keyed_profile)
    key_path().write_bytes(keyed_profile[:-1])
    assert FrequencyAssignments(frozen).fingerprint_key_id is None


class ErasedFixture:
    def __init__(self, tmp_path: Path) -> None:
        self.path = tmp_path / "erased.sqlite"
        database(self.path, 511)
        self.repository = MetadataRepository(self.path)
        self.generation = view(self.repository).generation

    def read(
        self, *, reset: bool = False, repin: bool = False, epoch: int = 1, authority="default"
    ):
        approved = (lambda cwd: cwd == "/approved") if authority == "default" else authority
        return collector_frequency.erased_view(
            PROFILE,
            scope_epoch=epoch,
            reset_authority=reset,
            repin=repin,
            repository=self.repository,
            authority=approved,
            authority_id="approved-v1",
        )


@pytest.fixture
def erased(tmp_path: Path, keyed_profile: bytes):
    fixture = ErasedFixture(tmp_path)
    try:
        yield fixture
    finally:
        fixture.repository.close()


def authorized(erased: ErasedFixture, **extra) -> dict:
    return {
        "schema": 1,
        "authorized_scope_epoch": 1,
        "authorized_generation": erased.generation,
        **extra,
    }


@pytest.mark.parametrize(
    ("marker", "reason"),
    [
        (b"{", "frequency_scope_erased"),
        (b"[]", "frequency_scope_erased"),
        ({"schema": 2}, "frequency_scope_erased"),
        ({"schema": 1, "pending_purge": True}, "frequency_scope_erased"),
        ({"schema": 1, "erased_workspaces": "/a"}, "frequency_scope_erased"),
        ({"schema": 1, "erased_workspaces": ["relative"]}, "frequency_scope_erased"),
        (
            {"schema": 1, "erased_workspaces": [f"/w{n}" for n in range(65)]},
            "frequency_scope_erased",
        ),
        ({"schema": 1, "erased_scope_unknown": True}, "frequency_erased_scope_unknown"),
        ({"schema": 1, "scoped": True}, "frequency_erased_scope_unknown"),
        (
            {"schema": 1, "erased_scope_unknown": True, "whole_profile": True},
            "frequency_scope_erased",
        ),
        (
            {"schema": 1, "authorized_scope_epoch": 2, "authorized_generation": "g"},
            "frequency_scope_erased",
        ),
        (
            {"schema": 1, "authorized_scope_epoch": 1, "authorized_generation": ""},
            "frequency_scope_erased",
        ),
    ],
)
def test_erased_scope_marker_variants_stay_unavailable_without_a_new_authorization(
    erased: ErasedFixture, marker, reason: str
) -> None:
    write_collector_file(
        "frequency-purged.json",
        marker if isinstance(marker, bytes) else json.dumps(marker).encode(),
    )
    before = read_collector_file("frequency-purged.json")
    result = erased.read()
    assert result is not None and not result.known
    assert result.unavailable_reason == reason
    assert result.profile_scope == PROFILE
    assert read_collector_file("frequency-purged.json") == before


def test_erased_scope_absent_marker_and_valid_authorization_are_distinguished(
    erased: ErasedFixture,
) -> None:
    assert erased.read() is None
    write_collector_file("frequency-purged.json", json.dumps(authorized(erased)).encode())
    result = erased.read()
    assert result is not None and result.known
    assert result.doc_count == 511 and result.generation == erased.generation
    assert result.authority_id != "approved-v1"


def test_erased_scope_authority_needs_identity_and_default_policy(erased: ErasedFixture) -> None:
    write_collector_file("frequency-purged.json", json.dumps(authorized(erased)).encode())
    unidentified = collector_frequency.erased_view(
        PROFILE,
        scope_epoch=1,
        reset_authority=False,
        repin=False,
        repository=erased.repository,
        authority=lambda cwd: True,
        authority_id="",
    )
    assert unidentified is not None
    assert unidentified.unavailable_reason == "frequency_scope_unapproved"
    default = collector_frequency.erased_view(
        PROFILE,
        scope_epoch=1,
        reset_authority=False,
        repin=False,
        repository=erased.repository,
    )
    assert default is not None and not default.known
    assert default.unavailable_reason == "frequency_scope_erased"


def test_erased_roots_remove_authority_for_contained_and_unknown_working_directories(
    erased: ErasedFixture,
) -> None:
    write_collector_file(
        "frequency-purged.json",
        json.dumps(authorized(erased, erased_workspaces=["/elsewhere"])).encode(),
    )
    outside = erased.read()
    assert outside is not None and outside.known

    write_collector_file(
        "frequency-purged.json",
        json.dumps(authorized(erased, erased_workspaces=["/approved"])).encode(),
    )
    inside = erased.read()
    assert inside is not None and not inside.known

    sqlite = sqlite3.connect(erased.path)
    sqlite.execute("UPDATE chunks SET cwd=NULL WHERE chunk_id='0'")
    sqlite.commit()
    sqlite.close()
    anywhere = lambda cwd: True  # noqa: E731
    write_collector_file("frequency-purged.json", json.dumps(authorized(erased)).encode())
    no_roots = erased.read(authority=anywhere)
    assert no_roots is not None and no_roots.known
    write_collector_file(
        "frequency-purged.json",
        json.dumps(authorized(erased, erased_workspaces=["/elsewhere"])).encode(),
    )
    unknown_cwd = erased.read(authority=anywhere)
    assert unknown_cwd is not None and not unknown_cwd.known


def test_erased_authority_reset_requires_a_newer_generation_then_persists(
    erased: ErasedFixture,
) -> None:
    stale = {"schema": 1, "whole_profile": True, "erased_generation": erased.generation}
    write_collector_file("frequency-purged.json", json.dumps(stale).encode())
    refused = erased.read(reset=True, repin=True)
    assert refused is not None
    assert refused.unavailable_reason == "frequency_authority_reset_requires_new_generation"
    assert json.loads(read_collector_file("frequency-purged.json") or b"") == stale
    only_reset = erased.read(reset=True)
    assert only_reset is not None and only_reset.unavailable_reason == "frequency_scope_erased"

    older = {**stale, "erased_generation": "older-generation"}
    write_collector_file("frequency-purged.json", json.dumps(older).encode())
    granted = erased.read(reset=True, repin=True, epoch=7)
    assert granted is not None and granted.known
    persisted = json.loads(read_collector_file("frequency-purged.json") or b"")
    assert persisted == {
        **older,
        "authorized_scope_epoch": 7,
        "authorized_generation": erased.generation,
    }
    assert erased.read(epoch=7) is not None and erased.read(epoch=7).known  # type: ignore[union-attr]
    again = erased.read(epoch=8)
    assert again is not None and again.unavailable_reason == "frequency_scope_erased"


def test_prune_views_validates_batch_and_reports_missing_profile_without_creating_it() -> None:
    for limit in (0, 129):
        with pytest.raises(ValueError, match="1..128"):
            prune_views(PROFILE, candidates=(), referenced=(), before_day="2026-01-01", limit=limit)
    outcome = prune_views(
        "never-created", candidates=("anything",), referenced=(), before_day="2026-01-01"
    )
    assert (outcome.scanned, outcome.removed, outcome.retained) == (0, 0, 0)
    with pytest.raises(FileNotFoundError):
        storage.open_profile_dir("never-created", "collector", create=False)


@pytest.mark.parametrize(
    ("prior", "expected"),
    [
        (
            b"{",
            {"whole_profile": True, "erased_scope_unknown": True, "erased_workspaces": ["/new"]},
        ),
        (
            b"[]",
            {"whole_profile": True, "erased_scope_unknown": True, "erased_workspaces": ["/new"]},
        ),
        (
            {"schema": 1, "erased_workspaces": ["relative"]},
            {"whole_profile": False, "erased_scope_unknown": True, "erased_workspaces": ["/new"]},
        ),
        (
            {"schema": 1, "scoped": True},
            {"whole_profile": False, "erased_scope_unknown": True, "erased_workspaces": ["/new"]},
        ),
        (
            {"schema": 1, "erased_workspaces": ["/old"]},
            {
                "whole_profile": False,
                "erased_scope_unknown": False,
                "erased_workspaces": ["/new", "/old"],
            },
        ),
    ],
)
def test_erasure_document_merges_prior_scope_and_fails_closed_on_damage(prior, expected) -> None:
    from types import SimpleNamespace

    write_collector_file(
        "frequency-purged.json", prior if isinstance(prior, bytes) else json.dumps(prior).encode()
    )
    directory = collector_directory()
    try:
        document = collector_frequency.erasure_document(
            directory, SimpleNamespace(workspace="/new", cutoff_day="2026-01-01")
        )
    finally:
        os.close(directory)
    assert {key: document[key] for key in expected} == expected
    assert document["schema"] == 1 and document["scoped"] is True
    assert document["before"] == "2026-01-01"
    assert document["authorized_scope_epoch"] is None and document["erased_generation"] is None


def test_erasure_document_records_the_pinned_generation_and_fails_closed_on_damaged_pins(
    tmp_path: Path, keyed_profile: bytes
) -> None:
    from types import SimpleNamespace

    path = tmp_path / "index.sqlite"
    database(path, 511)
    repository = MetadataRepository(path)
    try:
        frozen = view(repository)
    finally:
        repository.close()
    collector_frequency.pin(PROFILE, frozen)
    selection = SimpleNamespace(workspace=None, cutoff_day="2026-02-02")

    def document() -> dict:
        directory = collector_directory()
        try:
            return collector_frequency.erasure_document(directory, selection)
        finally:
            os.close(directory)

    pinned = document()
    assert pinned["erased_generation"] == frozen.generation
    assert pinned["whole_profile"] is True and pinned["scoped"] is False

    write_collector_file(collector_frequency.POINTER, b'{"view_id": "not-a-view"}')
    assert document()["erased_generation"] is None
    write_collector_file(
        "frequency-purged.json",
        json.dumps({"schema": 1, "erased_generation": "prior-generation"}).encode(),
    )
    assert document()["erased_generation"] == "prior-generation"
    write_collector_file(collector_frequency.POINTER, b"{")
    assert document()["erased_generation"] is None


# ---- Survivor source: frequency index build, budgets and repository seams ----
"""Index builds fail closed on damaged or raced inputs and resume within bounded budgets."""

FI_AUTHORITY = "approved-v1"


def fi_approve(cwd: str | None) -> bool:
    return cwd == "/approved"


def fi_collector_directory() -> int:
    return storage.open_profile_dir(PROFILE, "collector")


def fi_collector_names(prefix: str) -> list[str]:
    directory = fi_collector_directory()
    try:
        return sorted(name for name in storage.list_names(directory) if name.startswith(prefix))
    finally:
        os.close(directory)


def fi_write_collector_file(name: str, body: bytes) -> None:
    directory = fi_collector_directory()
    try:
        storage.atomic_write(directory, name, body)
    finally:
        os.close(directory)


def fi_plant_head() -> None:
    fi_write_collector_file(index_frequency.CHECKPOINT, b"{}")


def fi_put_page(kind: str, number: int, document: Any) -> dict:
    body = dumps(document)
    fi_write_collector_file(index_frequency._page_name(kind, number), body)
    return {
        "name": index_frequency._page_name(kind, number),
        "digest": hashlib.sha256(body).hexdigest(),
        "records": len(document),
        "bytes": len(body),
    }


def fi_tamper(mutator) -> None:
    job = index_frequency._load(PROFILE)
    assert job is not None
    mutator(job)
    index_frequency._save(PROFILE, job)


def fi_retarget_terms(job: dict, documents: list) -> None:
    job["terms"] = [fi_put_page("terms", number, page) for number, page in enumerate(documents)]
    job["term_cursor"] = sum(page["records"] for page in job["terms"])
    job["bytes"] = sum(page["bytes"] for page in job["terms"] + job["seen"])


def fi_built(path: Path, documents: int = 3) -> MetadataRepository:
    database(path, documents)
    repository = MetadataRepository(path)
    control = view(repository)
    assert control.known and control.doc_count == documents
    return repository


def fi_lance_index(
    tmp_path: Path, monkeypatch, *, documents: int = 3, metadata: bool = True
) -> None:
    path = tmp_path / "index.lance"
    connection = lancedb.connect(str(path), read_consistency_interval=timedelta(seconds=0))
    if metadata:
        connection.create_table(
            "metadata",
            [
                {"key": "last_index_time", "value": "epoch-1"},
                {"key": "index_state", "value": "ready"},
                {
                    "key": "lexical_stats",
                    "value": json.dumps(
                        {"n_docs": documents, "avgdl": 1, "df": {RARE.text: 1}},
                    ),
                },
            ],
        )
    connection.create_table(
        "chunks", [{"chunk_id": f"chunk-{i}", "cwd": "/approved"} for i in range(documents)]
    )

    def private_store():
        store = LanceStore()
        store.path = path
        return store

    monkeypatch.setattr(index_frequency_repository, "LanceStore", private_store)


def test_default_authority_follows_the_active_local_profile_policy(
    tmp_path: Path, keyed_profile: bytes
) -> None:
    path = tmp_path / "index.sqlite"
    database(path, 3)
    repository = MetadataRepository(path)
    try:
        inactive = read_index_view(PROFILE, repository=repository)
        assert inactive.unavailable_reason == "frequency_scope_unapproved"

        enable_collection(PROFILE)
        active = read_index_view(PROFILE, repository=repository)
        assert active.known and active.doc_count == 3
        assert active.authority_id == index_frequency._authority(PROFILE)[1]
    finally:
        repository.close()


def test_read_reports_a_key_that_becomes_unusable_as_unavailable(
    tmp_path: Path, keyed_profile: bytes
) -> None:
    path = tmp_path / "index.sqlite"
    database(path, 3)
    repository = MetadataRepository(path)
    try:
        assert index_frequency._read(PROFILE, repository, fi_approve, FI_AUTHORITY).known

        fingerprint.forget_keys()
        directory = storage.open_profile_dir(PROFILE, create=False)
        try:
            storage.atomic_write(directory, fingerprint.KEY_FILE, b"short")
        finally:
            os.close(directory)

        rejected = index_frequency._read(PROFILE, repository, fi_approve, FI_AUTHORITY)
        assert not rejected.known
        assert rejected.unavailable_reason == "frequency_key_unavailable"
        assert view(repository).unavailable_reason == "frequency_index_unreadable"
    finally:
        repository.close()


def test_owned_repository_close_failure_withdraws_an_otherwise_complete_view(
    tmp_path: Path, monkeypatch, keyed_profile: bytes
) -> None:
    fi_lance_index(tmp_path, monkeypatch)

    def read():
        return read_index_view(PROFILE, authority=fi_approve, authority_id=FI_AUTHORITY)

    ok = read()
    assert ok.known and ok.doc_count == 3

    def failing_close(self):
        self.store.close()
        raise OSError("close failed")

    monkeypatch.setattr(index_frequency_repository.IndexRepository, "close", failing_close)
    rejected = read()
    assert not rejected.known
    assert rejected.unavailable_reason == "frequency_index_unreadable"


def test_repository_snapshot_rejects_bad_generation_and_cleans_up_failed_checkouts(
    tmp_path: Path, monkeypatch
) -> None:
    fi_lance_index(tmp_path, monkeypatch)
    closed = []
    real_close = index_frequency_repository.IndexRepository.close

    def counting_close(self):
        closed.append(self)
        real_close(self)

    monkeypatch.setattr(index_frequency_repository.IndexRepository, "close", counting_close)
    live = index_frequency_repository.IndexRepository()
    try:
        generation = _generation(live)
        assert generation is not None
        live.snapshot(generation).close()
        closed.clear()

        with pytest.raises(ValueError, match="immutable table versions"):
            live.snapshot("epoch-1")
        assert closed == []  # rejected before any handle was opened

        with pytest.raises(Exception, match=r"."):
            live.snapshot("epoch-1@metadata:999,chunks:999")
        assert len(closed) == 1 and closed[0] is not live
    finally:
        live.close()


def test_repository_snapshot_rejects_an_index_without_a_metadata_table(
    tmp_path: Path, monkeypatch
) -> None:
    fi_lance_index(tmp_path, monkeypatch, metadata=False)
    live = index_frequency_repository.IndexRepository()
    try:
        with pytest.raises(ValueError, match="table is absent"):
            live.snapshot("epoch-1@metadata:1,chunks:1")
    finally:
        live.close()


def test_rows_page_rejects_an_offset_range_beyond_the_immutable_table(
    tmp_path: Path, monkeypatch
) -> None:
    fi_lance_index(tmp_path, monkeypatch, documents=3)
    live = index_frequency_repository.IndexRepository()
    try:
        generation = _generation(live)
        assert generation is not None
        frozen = live.snapshot(generation)
        try:
            assert [row["chunk_id"] for row in frozen.rows_page(start=1, limit=2)] == [
                "chunk-1",
                "chunk-2",
            ]
            with pytest.raises(Exception, match=r"."):
                frozen.rows_page(start=2, limit=3)
        finally:
            frozen.close()
    finally:
        live.close()


def test_orphan_inventory_beyond_its_entry_budget_blocks_index_use(
    tmp_path: Path, monkeypatch, keyed_profile: bytes
) -> None:
    path = tmp_path / "index.sqlite"
    database(path, 3)
    for number in range(3):
        fi_write_collector_file(index_frequency._page_name("terms", number), b'{"orphan":1}')
    repository = MetadataRepository(path)
    try:
        monkeypatch.setattr(index_frequency, "ORPHAN_ENTRIES", 2)
        blocked = view(repository)
        assert blocked.unavailable_reason == "frequency_build_orphan_inventory_budget_exceeded"
        assert len(fi_collector_names("frequency-build-terms-")) == 3

        monkeypatch.setattr(index_frequency, "ORPHAN_ENTRIES", 4096)
        assert view(repository).known
    finally:
        repository.close()


def test_orphan_cleanup_stops_when_a_build_head_appears_after_the_inventory(
    tmp_path: Path, monkeypatch, keyed_profile: bytes
) -> None:
    path = tmp_path / "index.sqlite"
    database(path, 3)
    for number in range(2):
        fi_write_collector_file(index_frequency._page_name("terms", number), b'{"orphan":1}')
    real_scandir = os.scandir

    class PlantingScan:
        def __init__(self, directory):
            self.inner = real_scandir(directory)

        def __enter__(self):
            return self.inner.__enter__()

        def __exit__(self, *error):
            outcome = self.inner.__exit__(*error)
            fi_plant_head()
            return outcome

    repository = MetadataRepository(path)
    try:
        monkeypatch.setattr(index_frequency.os, "scandir", PlantingScan)
        blocked = view(repository)
        assert blocked.unavailable_reason == "frequency_build_orphan_cleanup_pending"
        assert len(fi_collector_names("frequency-build-terms-")) == 2
        assert fi_collector_names(index_frequency.CHECKPOINT)
    finally:
        repository.close()


def test_orphan_cleanup_stops_when_a_build_head_appears_between_deletions(
    tmp_path: Path, monkeypatch, keyed_profile: bytes
) -> None:
    path = tmp_path / "index.sqlite"
    database(path, 3)
    for number in range(3):
        fi_write_collector_file(index_frequency._page_name("terms", number), b'{"orphan":1}')
    real_delete = storage.delete_file
    planted = []

    def deleting_then_planting(directory, name):
        removed = real_delete(directory, name)
        if not planted:
            planted.append(name)
            fi_plant_head()
        return removed

    repository = MetadataRepository(path)
    try:
        monkeypatch.setattr(storage, "delete_file", deleting_then_planting)
        blocked = view(repository)
        assert blocked.unavailable_reason == "frequency_build_orphan_cleanup_pending"
        assert len(planted) == 1
        assert len(fi_collector_names("frequency-build-terms-")) == 2
    finally:
        repository.close()


def test_build_head_exceeding_its_byte_budget_is_not_saved(
    tmp_path: Path, monkeypatch, keyed_profile: bytes
) -> None:
    path = tmp_path / "index.sqlite"
    database(path, 3)
    repository = MetadataRepository(path)
    try:
        monkeypatch.setattr(index_frequency, "HEAD_BYTES", 10)
        rejected = view(repository)
        assert rejected.unavailable_reason == "frequency_index_unreadable"
        assert fi_collector_names(index_frequency.CHECKPOINT) == []

        monkeypatch.setattr(index_frequency, "HEAD_BYTES", 1024 * 1024)
        assert view(repository).known
    finally:
        repository.close()


def test_stats_larger_than_their_budget_are_rejected(
    tmp_path: Path, monkeypatch, keyed_profile: bytes
) -> None:
    path = tmp_path / "index.sqlite"
    database(path, 3)
    repository = MetadataRepository(path)
    try:
        monkeypatch.setattr(index_frequency, "MAX_STATS_BYTES", 10)
        assert view(repository).unavailable_reason == "frequency_stats_budget_exceeded"
        monkeypatch.setattr(index_frequency, "MAX_STATS_BYTES", 16 * 1024 * 1024)
        assert view(repository).known
    finally:
        repository.close()


def test_build_budget_failure_is_persisted_and_not_retried_on_the_same_generation(
    tmp_path: Path, monkeypatch, keyed_profile: bytes
) -> None:
    path = tmp_path / "index.sqlite"
    database(path, 3)
    repository = MetadataRepository(path)
    try:
        monkeypatch.setattr(index_frequency, "MAX_BUILD_BYTES", 10)
        first = view(repository)
        assert first.unavailable_reason == "frequency_build_budget_exceeded"
        job = index_frequency._load(PROFILE)
        assert job is not None
        assert job["failure"] == "frequency_build_budget_exceeded" and job["terms"] == []

        monkeypatch.setattr(index_frequency, "MAX_BUILD_BYTES", 64 * 1024 * 1024)
        again = view(repository)
        assert again.unavailable_reason == "frequency_build_budget_exceeded"

        writer = sqlite3.connect(path)
        writer.execute("UPDATE metadata SET value='generation-2' WHERE key='last_index_time'")
        writer.commit()
        writer.close()
        recovered = view(repository)
        assert recovered.known and recovered.generation == "generation-2"
    finally:
        repository.close()


@pytest.mark.parametrize("budget", [("VISITS", 1), ("SECONDS", 0)], ids=["visits", "deadline"])
def test_bounded_build_resumes_to_the_same_statistics_as_an_unbounded_build(
    tmp_path: Path, monkeypatch, keyed_profile: bytes, budget: tuple[str, float]
) -> None:
    path = tmp_path / "index.sqlite"
    database(path, 3)
    fingerprint.load_key("frequency-control", create=True)
    control_repository = MetadataRepository(path)
    try:
        control = view(control_repository, "frequency-control")
        assert control.known
    finally:
        control_repository.close()

    monkeypatch.setattr(index_frequency, *budget)
    repository = MetadataRepository(path)
    try:
        reasons = []
        for _ in range(12):
            result = view(repository)
            if result.known:
                break
            reasons.append(result.unavailable_reason)
        assert reasons == ["frequency_build_pending"] * 3
        assert result.known and result.doc_count == 3
        assert sorted(result.df.values()) == sorted(control.df.values()) == [1, 3]
    finally:
        repository.close()


def test_pending_build_is_discarded_when_its_snapshot_is_no_longer_available(
    tmp_path: Path, monkeypatch, keyed_profile: bytes
) -> None:
    path = tmp_path / "index.sqlite"
    database(path, 3)
    monkeypatch.setattr(index_frequency, "VISITS", 1)
    repository = MetadataRepository(path)
    try:
        assert view(repository).unavailable_reason == "frequency_build_pending"
        assert index_frequency._load(PROFILE) is not None

        class SnapshotGoneRepository(MetadataRepository):
            def snapshot(self, generation: str):
                raise OSError("version expired")

        gone = SnapshotGoneRepository(path)
        try:
            rejected = view(gone)
        finally:
            gone.close()
        assert rejected.unavailable_reason == "frequency_snapshot_unavailable"
        assert index_frequency._load(PROFILE) is None
        assert fi_collector_names("frequency-build-") == []
    finally:
        repository.close()


def test_pending_build_rejects_statistics_that_changed_under_the_same_generation(
    tmp_path: Path, monkeypatch, keyed_profile: bytes
) -> None:
    path = tmp_path / "index.sqlite"
    database(path, 3)
    monkeypatch.setattr(index_frequency, "VISITS", 1)
    repository = MetadataRepository(path)
    try:
        assert view(repository).unavailable_reason == "frequency_build_pending"
        writer = sqlite3.connect(path)
        writer.execute(
            "UPDATE metadata SET value=? WHERE key='lexical_stats'",
            (json.dumps({"n_docs": 3, "df": {RARE.text: 1, COMMON.text: 3}, "avgdl": 9.0}),),
        )
        writer.commit()
        writer.close()
        changed = view(repository)
        assert changed.unavailable_reason == "frequency_generation_changed"
    finally:
        repository.close()


@pytest.mark.parametrize("drift", ["index_state", "last_index_time"])
def test_index_drift_after_the_snapshot_is_taken_cannot_publish_statistics(
    tmp_path: Path, keyed_profile: bytes, drift: str
) -> None:
    path = tmp_path / "index.sqlite"
    database(path, 3)

    class Frozen(MetadataRepository):
        meta: dict[str, str | None]

        def get_meta(self, key: str) -> str | None:
            return self.meta[key]

    class DriftingRepository(MetadataRepository):
        def snapshot(self, generation: str):
            frozen = Frozen(path)
            frozen.meta = {
                key: super(Frozen, frozen).get_meta(key)
                for key in ("index_state", "last_index_time", "lexical_stats")
            }
            writer = sqlite3.connect(path)
            value = "building" if drift == "index_state" else "generation-2"
            writer.execute("UPDATE metadata SET value=? WHERE key=?", (value, drift))
            writer.commit()
            writer.close()
            return frozen

    repository = DriftingRepository(path)
    try:
        rejected = view(repository)
        assert rejected.unavailable_reason == "frequency_generation_changed"
        assert index_frequency._load(PROFILE) is None
    finally:
        repository.close()


def test_statistics_changed_while_chunk_scope_was_read_cannot_be_published(
    tmp_path: Path, keyed_profile: bytes
) -> None:
    path = tmp_path / "index.sqlite"
    database(path, 3)

    class RacingRepository(MetadataRepository):
        def rows_page(self, *, start: int, limit: int) -> list[dict]:
            rows = super().rows_page(start=start, limit=limit)
            writer = sqlite3.connect(path)
            writer.execute("UPDATE metadata SET value='{\"n_docs\":3}' WHERE key='lexical_stats'")
            writer.commit()
            writer.close()
            return rows

    repository = RacingRepository(path)
    try:
        rejected = view(repository)
        assert rejected.unavailable_reason == "frequency_generation_changed"
    finally:
        repository.close()


def test_chunk_scope_page_over_its_byte_budget_fails_closed_against_a_fitting_control(
    tmp_path: Path, monkeypatch, keyed_profile: bytes
) -> None:
    path = tmp_path / "index.sqlite"
    database(path, 2)
    fingerprint.load_key("frequency-control", create=True)
    control_repository = MetadataRepository(path)
    try:
        assert view(control_repository, "frequency-control").known
    finally:
        control_repository.close()
    descriptor = storage.open_profile_dir("frequency-control", "collector", create=False)
    try:
        encoded = storage.read_file(descriptor, index_frequency.CHECKPOINT, 1024 * 1024)
        assert encoded is not None
        pages = json.loads(encoded)["job"]
    finally:
        os.close(descriptor)
    page_bytes = max(page["bytes"] for page in pages["terms"] + pages["seen"])
    monkeypatch.setattr(index_frequency, "PAGE_BYTES", page_bytes)

    class WideRepository(MetadataRepository):
        def rows_page(self, *, start: int, limit: int) -> list[dict]:
            rows = super().rows_page(start=start, limit=limit)
            rows[0]["chunk_id"] = "x" * (page_bytes + 1)
            return rows

    wide = WideRepository(path)
    try:
        rejected = view(wide)
        assert rejected.unavailable_reason == "frequency_scope_page_budget_exceeded"
    finally:
        wide.close()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda job: job.update(schema=2),
        lambda job: job.pop("profile"),
        lambda job: job.update(term_cursor=99),
        lambda job: job.update(chunk_cursor=-1),
        lambda job: job.update(bytes=job["bytes"] + 1),
        lambda job: job.update(bytes=index_frequency.MAX_BUILD_BYTES + 1),
    ],
    ids=["schema", "missing-key", "term-cursor", "chunk-cursor", "byte-sum", "byte-limit"],
)
def test_damaged_build_head_is_never_trusted_after_a_complete_build_control(
    tmp_path: Path, keyed_profile: bytes, mutate
) -> None:
    repository = fi_built(tmp_path / "index.sqlite")
    try:
        fi_tamper(mutate)
        damaged = view(repository)
        assert not damaged.known
        assert damaged.unavailable_reason == "frequency_build_damaged"
    finally:
        repository.close()


def test_non_object_build_head_and_digest_mismatch_are_damaged(
    tmp_path: Path, keyed_profile: bytes
) -> None:
    repository = fi_built(tmp_path / "index.sqlite")
    try:
        index_frequency._save(PROFILE, ["not", "a", "job"])
        assert view(repository).unavailable_reason == "frequency_build_damaged"

        fi_write_collector_file(
            index_frequency.CHECKPOINT,
            dumps({"job": {"schema": 1}, "digest": "0" * 64}),
        )
        assert view(repository).unavailable_reason == "frequency_build_damaged"
    finally:
        repository.close()


def test_seen_pages_that_do_not_cover_their_chunk_cursor_are_damaged(
    tmp_path: Path, monkeypatch, keyed_profile: bytes
) -> None:
    path = tmp_path / "index.sqlite"
    database(path, 3)
    monkeypatch.setattr(index_frequency, "VISITS", 1)
    repository = MetadataRepository(path)
    try:
        assert view(repository).unavailable_reason == "frequency_build_pending"
        assert view(repository).unavailable_reason == "frequency_build_pending"
        job = index_frequency._load(PROFILE)
        assert job is not None and job["term_cursor"] == 2 and job["chunk_cursor"] == 1

        def duplicate_seen(job: dict) -> None:
            job["seen"] = [fi_put_page("seen", 0, ["a" * 64, "a" * 64])]
            job["chunk_cursor"] = 2
            job["bytes"] = sum(page["bytes"] for page in job["terms"] + job["seen"])

        fi_tamper(duplicate_seen)
        assert view(repository).unavailable_reason == "frequency_build_damaged"
    finally:
        repository.close()


def retarget_name(job: dict) -> None:
    job["terms"][0]["name"] = "frequency-build-terms-1.json"


def retarget_digest(job: dict) -> None:
    job["terms"][0]["digest"] = "0" * 64


def retarget_missing(job: dict) -> None:
    directory = fi_collector_directory()
    try:
        storage.delete_file(directory, job["terms"][0]["name"])
    finally:
        os.close(directory)


def retarget_record_count(job: dict) -> None:
    job["terms"][0]["records"] = 1
    job["term_cursor"] = 1


def retarget_list_page(job: dict) -> None:
    fi_retarget_terms(job, [["only-a-list"]])


def retarget_overlap(job: dict) -> None:
    fi_retarget_terms(job, [{"a" * 64: 1}, {"a" * 64: 1}])


@pytest.mark.parametrize(
    "mutate",
    [
        retarget_name,
        retarget_digest,
        retarget_missing,
        retarget_record_count,
        retarget_list_page,
        retarget_overlap,
    ],
    ids=["page-name", "digest", "missing-page", "record-count", "list-page", "overlap"],
)
def test_complete_head_with_damaged_term_pages_never_yields_a_frequency_view(
    tmp_path: Path, keyed_profile: bytes, mutate
) -> None:
    repository = fi_built(tmp_path / "index.sqlite")
    try:
        fi_tamper(mutate)
        damaged = view(repository)
        assert not damaged.known
        assert damaged.unavailable_reason == "frequency_index_unreadable"
    finally:
        repository.close()
