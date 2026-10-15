"""Independent lexical generations, conservative mapping and immutable revision evidence."""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import pytest

from ssgrep.accounting import collector_frequency, fingerprint
from ssgrep.accounting.contracts import Scenario
from ssgrep.accounting.evidence_tokens import Token
from ssgrep.accounting.frequency import THRESHOLD_MILLI, eligible_scenarios
from ssgrep.accounting.index_frequency import read_index_view
from ssgrep.accounting.plain_codec import to_plain

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
    from ssgrep.accounting.frequency_assignment import FrequencyAssignments

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
