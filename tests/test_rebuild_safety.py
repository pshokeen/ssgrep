"""Rebuild safety: a rebuild must never silently destroy a populated index,
and must never be permanently blocked by its own abandoned residue.

Both defects below were demonstrated end to end before these tests existed,
and both lose a buyer's entire indexed history with no warning:

C2. indexer.index() committed the rebuilt generation unconditionally. A scope
    mismatch makes discovery return zero transcripts, so the empty generation
    was swapped over a healthy one -- "Indexed 0 sessions, 0 episodes, 0
    chunks." and exit 0, with the prior generation's files unlinked by
    commit_generation()'s cleanup. Not a --rebuild-only hazard: index()
    computes `rebuild or needs_rebuild(...)`, so a plain `ssgrep index` after
    any release that bumps SCHEMA_VERSION or the embedding model takes the
    identical path. Both routes are covered here.

C3. stage_generation() opened generation N's files without clearing residue
    from an earlier attempt at N that died partway. The staged vectors.f32
    then carried rows from both attempts while the staged chunks table
    referenced only the second's, so validate_generations() rejected it and
    commit_generation() raised "not mutually consistent" -- forever, for
    every later rebuild, with `rm -rf .ssgrep` the only escape.

embed.encode() is monkeypatched to a deterministic stand-in throughout: these
tests exercise commit/staging control flow, not embedding quality.
"""

from __future__ import annotations

import fcntl
import gc
import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ssgrep import embed, indexer, rebuild_guard, store
from ssgrep.types import RebuildWouldShrinkError

DIM = 256


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tmp_path-rooted stand-in for ~, holding synthetic transcripts.

    discovery.discover_sessions() reads Path.home()/".claude"/"projects" with
    no override, so the only way to isolate it is to replace Path.home().
    """
    home = tmp_path / "home"
    (home / ".claude" / "projects" / "proj").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    return home


def _committed_manifest(index_dir: Path, generation: int = 0) -> None:
    """Give a store the manifest a real one always has.

    Every store that has ever indexed carries a manifest: the incremental
    path writes one through checkpoint(), and a rebuild writes one through
    commit_generation(). Its absence is not a neutral fixture detail --
    without it, an index.db.N on disk is indistinguishable from a live
    generation whose manifest was lost, which discard_generation() now
    refuses to unlink. Fixtures that mean "residue above a live generation"
    must therefore say which generation is live, exactly as production does.
    """
    (index_dir / ".manifest").write_text(
        json.dumps({"generation": generation, "vector_row_count": 0})
    )


def _fake_vectors(n: int) -> np.ndarray:
    return np.full((n, DIM), 1.0 / (DIM**0.5), dtype=np.float32)


def _install_fake_encode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embed, "encode", lambda texts: _fake_vectors(len(texts)))


def _write_transcript(home: Path, project_dir: Path, session_id: str) -> Path:
    """Write one two-turn transcript whose records record `project_dir` as cwd.

    Scope matching is cwd-based (D11), so this `cwd` value -- not the file's
    location -- is what decides whether discovery finds this session.
    """
    path = home / ".claude" / "projects" / "proj" / f"{session_id}.jsonl"
    records: list[dict[str, Any]] = [
        {
            "parentUuid": None,
            "isSidechain": False,
            "type": "user",
            "message": {"role": "user", "content": f"how do I fix the {session_id} problem"},
            "uuid": f"{session_id}-u1",
            "timestamp": "2026-07-01T10:00:00.000Z",
            "cwd": str(project_dir),
            "sessionId": session_id,
            "gitBranch": "main",
        },
        {
            "parentUuid": f"{session_id}-u1",
            "isSidechain": False,
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": f"you fix the {session_id} problem by rebuilding the widget "
                        "and then checking that the counts still line up afterwards",
                    }
                ],
            },
            "uuid": f"{session_id}-a1",
            "timestamp": "2026-07-01T10:01:00.000Z",
            "cwd": str(project_dir),
            "sessionId": session_id,
            "gitBranch": "main",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return path


def _write_corpus(home: Path, project_dir: Path, count: int) -> list[Path]:
    return [
        _write_transcript(home, project_dir, f"aaaaaaaa-0000-4000-8000-00000000000{i}")
        for i in range(count)
    ]


def _write_subagent_corpus(home: Path, project_dir: Path, count: int) -> list[Path]:
    """Transcripts discovery classifies as SUBAGENT sessions.

    Classification is by path position, not by content
    (discovery.classify_session): a subagent lives at
    ``<projectDir>/<sessionId>/subagents/**/*.jsonl``. Writing them anywhere
    else would produce main sessions and quietly make every --no-subagents
    assertion below vacuous, so the shape here is load-bearing.
    """
    written = []
    parent = "cccccccc-0000-4000-8000-000000000000"
    for i in range(count):
        session_id = f"bbbbbbbb-0000-4000-8000-00000000000{i}"
        path = home / ".claude" / "projects" / "proj" / parent / "subagents" / f"{session_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        source = _write_transcript(home, project_dir, session_id)
        path.write_text(source.read_text())
        source.unlink()
        written.append(path)
    return written


def _live_counts(index_dir: Path) -> tuple[int, int, int]:
    """(sessions, episodes, chunks) of whatever the manifest currently points at.

    Deliberately re-reads the manifest through a FRESH GenerationalStore, so a
    committed generation swap is reflected: this is what a later `ssgrep
    search` would actually open.
    """
    gen_store = store.GenerationalStore(index_dir)
    conn = sqlite3.connect(str(gen_store.get_index_path()))
    try:
        return (
            conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0],
            conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0],
            conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
        )
    finally:
        conn.close()


def _strip_persisted_scope(index_dir: Path) -> None:
    """Turn a freshly-built index into a LEGACY index (no persisted scope).

    Every test in this module simulates the pre-scope-persistence trap: a
    project moves, its index travels with it, and a rebuild's empty
    discovery must be refused rather than committed. With scope persistence
    (2026-08-07), an index built by current code REMEMBERS its original
    scope and keeps discovering the old corpus -- the trap these tests
    exist to cover can no longer be reproduced with a current-format index.
    It remains fully live for every index built before the feature (no meta
    key -> discovery falls back to project_dir), which is exactly the
    artifact this helper manufactures. Scoped-index behavior has its own
    coverage in tests/test_scope_flag.py.
    """
    import sqlite3

    from ssgrep import store as store_mod

    gen = store_mod.GenerationalStore(index_dir)
    conn = sqlite3.connect(str(gen.get_index_path()))
    try:
        conn.execute("DELETE FROM meta WHERE key = 'scope'")
        conn.commit()
    finally:
        conn.close()


def _build_populated_index(
    home: Path, project_dir: Path, index_dir: Path, count: int = 4
) -> tuple[int, int, int]:
    _write_corpus(home, project_dir, count)
    stats = indexer.index(project_dir, index_dir=index_dir, quiet=True)
    assert stats.session_count == count, "fixture must actually index the whole corpus"
    assert stats.episode_count > 0 and stats.chunk_count > 0
    _strip_persisted_scope(index_dir)
    return stats.session_count, stats.episode_count, stats.chunk_count


# ---------------------------------------------------------------------------
# C2 -- a shrinking commit must be refused, on BOTH routes into the rebuild.
# ---------------------------------------------------------------------------


def test_explicit_rebuild_refuses_to_replace_a_populated_index_with_an_empty_one(
    fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`ssgrep index --rebuild` run with a mismatched scope must abort with a
    non-zero outcome and leave every indexed episode and chunk in place.

    The scope mismatch is the real one: the transcripts record their cwd as
    the ORIGINAL project path, the index directory travels with the project,
    and the rebuild runs against the project's new path. Discovery is
    cwd-matched, so it finds nothing -- which is exactly the state that used
    to be committed straight over the good generation.
    """
    _install_fake_encode(monkeypatch)
    original_project = tmp_path / "original-project"
    index_dir = tmp_path / "idx"
    before = _build_populated_index(fake_home, original_project, index_dir)

    moved_project = tmp_path / "moved-project"
    with pytest.raises(RebuildWouldShrinkError) as excinfo:
        indexer.index(moved_project, index_dir=index_dir, rebuild=True, quiet=True)

    message = str(excinfo.value)
    assert f"{before[0]} sessions" in message, f"must name the old session count: {message}"
    assert f"{before[1]} episodes" in message, f"must name the old episode count: {message}"
    assert "0 sessions" in message, f"must name the new (empty) counts: {message}"
    assert "scope mismatch" in message, f"must name the likely cause: {message}"
    assert excinfo.value.old_counts == before
    assert excinfo.value.new_counts == (0, 0, 0)

    assert _live_counts(index_dir) == before, "the buyer's indexed history must survive intact"
    assert (
        store.GenerationalStore(index_dir).current_generation == 0
    ), "the manifest must still point at the generation that holds the data"
    assert not (index_dir / "index.db.1").exists(), "the refused staged generation must be dropped"
    assert not (index_dir / "vectors.f32.1").exists()


def test_flagless_index_after_a_model_change_refuses_to_shrink(
    fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The aggravator: no --rebuild flag anywhere.

    index() computes `rebuild or needs_rebuild(...)`, and needs_rebuild()
    returns True on any stored-model mismatch -- the exact state every buyer's
    index is in the first time they run a release that changes the embedding
    model. Mutating the stored model_id reproduces that upgrade faithfully:
    from here on a plain `ssgrep index` takes the destructive rebuild path,
    and with a mismatched scope it used to commit an empty generation over a
    healthy one without the user ever asking for a rebuild.
    """
    _install_fake_encode(monkeypatch)
    original_project = tmp_path / "original-project"
    index_dir = tmp_path / "idx"
    before = _build_populated_index(fake_home, original_project, index_dir)

    live_db = store.GenerationalStore(index_dir).get_index_path()
    conn = sqlite3.connect(str(live_db))
    try:
        conn.execute("UPDATE meta SET value = ? WHERE key = 'model_id'", ("some-other/model-v2",))
        conn.commit()
    finally:
        conn.close()
    model_id, dimension = embed.get_model_info()
    assert rebuild_guard.needs_rebuild(
        live_db, store.GenerationalStore(index_dir).get_vector_path(), model_id, dimension
    ), "precondition: the mutated model binding must be what forces the rebuild path"

    moved_project = tmp_path / "moved-project"
    with pytest.raises(RebuildWouldShrinkError):
        indexer.index(moved_project, index_dir=index_dir, quiet=True)  # no rebuild flag

    assert _live_counts(index_dir) == before, "an upgrade must not silently empty the index"
    assert store.GenerationalStore(index_dir).current_generation == 0


def test_allow_shrink_lets_the_rebuild_through(
    fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The override has to actually override: with allow_shrink the same
    rebuild that was refused above commits, advances the generation, and
    leaves the smaller index in place as the live one.
    """
    _install_fake_encode(monkeypatch)
    original_project = tmp_path / "original-project"
    index_dir = tmp_path / "idx"
    _build_populated_index(fake_home, original_project, index_dir)

    moved_project = tmp_path / "moved-project"
    stats = indexer.index(
        moved_project, index_dir=index_dir, rebuild=True, allow_shrink=True, quiet=True
    )

    assert stats.session_count == 0
    assert (
        store.GenerationalStore(index_dir).current_generation == 1
    ), "allow_shrink must commit the staged generation, not merely suppress the message"
    assert _live_counts(index_dir) == (0, 0, 0)


def test_guard_does_not_block_a_legitimate_rebuild_of_the_same_corpus(
    fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard must not make ordinary rebuilds unusable: rebuilding the same
    corpus in the same scope commits and preserves every count.
    """
    _install_fake_encode(monkeypatch)
    project = tmp_path / "project"
    index_dir = tmp_path / "idx"
    before = _build_populated_index(fake_home, project, index_dir)

    stats = indexer.index(project, index_dir=index_dir, rebuild=True, quiet=True)

    assert (stats.session_count, stats.episode_count, stats.chunk_count) == before
    assert store.GenerationalStore(index_dir).current_generation == 1
    assert _live_counts(index_dir) == before


def test_guard_permits_a_first_ever_rebuild_of_an_empty_index(
    fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing indexed yet means nothing to lose: an empty-in, empty-out
    rebuild must still commit, or a brand new install with no transcripts in
    scope could never build an index at all.
    """
    _install_fake_encode(monkeypatch)
    project = tmp_path / "project"
    index_dir = tmp_path / "idx"
    indexer.index(project, index_dir=index_dir, quiet=True)

    stats = indexer.index(project, index_dir=index_dir, rebuild=True, quiet=True)

    assert stats.session_count == 0
    assert store.GenerationalStore(index_dir).current_generation == 1


def test_rebuild_is_refused_when_every_chunk_would_be_lost_but_sessions_survive(
    fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rebuild that destroys 100% of searchable content must not commit.

    Session rows are not a proxy for indexed content: _index_file() calls
    store.insert_session() unconditionally for every discovered transcript,
    before and independently of the block that builds episodes and chunks. So
    any regression downstream of discovery -- an upstream transcript-format
    change, an episode-builder or chunker bug -- yields the identical session
    count with an empty corpus. Gating on sessions alone let that commit,
    unlink the previous generation, and exit 0.

    Modelled here by rewriting each transcript so its records still parse and
    still carry the matching cwd (discovery and session insertion are
    untouched, and asserted below) but yield no episode.
    """
    _install_fake_encode(monkeypatch)
    project = tmp_path / "project"
    index_dir = tmp_path / "idx"
    before = _build_populated_index(fake_home, project, index_dir)
    assert before[2] > 0, "precondition: the retiring index must hold chunks"

    for path in (fake_home / ".claude" / "projects" / "proj").glob("*.jsonl"):
        original = [json.loads(line) for line in path.read_text().splitlines()]
        # Same cwd, same sessionId, still valid JSONL -- but no user/assistant
        # turn, so segmentation produces nothing to chunk.
        gutted = {
            "type": "system",
            "subtype": "compact_boundary",
            "cwd": original[0]["cwd"],
            "sessionId": original[0]["sessionId"],
            "uuid": original[0]["uuid"],
            "timestamp": original[0]["timestamp"],
        }
        path.write_text(json.dumps(gutted) + "\n")

    with pytest.raises(RebuildWouldShrinkError) as excinfo:
        indexer.index(project, index_dir=index_dir, rebuild=True, quiet=True)

    assert excinfo.value.new_counts[0] == before[0], (
        "precondition: this must be the sessions-survive case -- if the session "
        f"count also dropped, the old sessions-only gate would have caught it: "
        f"{excinfo.value.new_counts}"
    )
    assert excinfo.value.new_counts[2] == 0, f"precondition: {excinfo.value.new_counts}"
    assert _live_counts(index_dir) == before, "every chunk must survive the refusal"
    assert store.GenerationalStore(index_dir).current_generation == 0
    assert not (index_dir / "index.db.1").exists()


def test_cli_rebuild_that_empties_the_corpus_does_not_exit_zero(
    fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """The same case through the real entry point, because "exit 0 with a
    cheerful count line" is the whole defect.

    Before the chunk floor this returned None (exit 0), printed "Indexed 4
    sessions, 0 episodes, 0 chunks.", printed NOTHING on stderr (the
    zero-discovery diagnostic is gated on session_count == 0, which is false
    here), advanced the generation, and unlinked the previous one. Every
    subsequent search returned nothing, permanently.
    """
    from unittest.mock import MagicMock

    from ssgrep.cli import exit_codes
    from ssgrep.cli.commands.index import IndexCommand

    _install_fake_encode(monkeypatch)
    project = tmp_path / "project"
    project.mkdir()
    before = _build_populated_index(fake_home, project, project / ".ssgrep")

    for path in (fake_home / ".claude" / "projects" / "proj").glob("*.jsonl"):
        first = json.loads(path.read_text().splitlines()[0])
        path.write_text(
            json.dumps(
                {
                    "type": "system",
                    "subtype": "compact_boundary",
                    "cwd": first["cwd"],
                    "sessionId": first["sessionId"],
                    "uuid": first["uuid"],
                    "timestamp": first["timestamp"],
                }
            )
            + "\n"
        )

    with pytest.raises(SystemExit) as excinfo:
        IndexCommand(MagicMock()).handle(project_dir=str(project), rebuild=True)

    assert excinfo.value.code == exit_codes.USAGE_ERROR
    assert f"{before[2]} chunks" in capsys.readouterr().err
    assert _live_counts(project / ".ssgrep") == before


def test_deleted_transcripts_do_not_block_a_flagless_rebuild(
    fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tombstoned sessions must not count as "existing" against the rebuild.

    A rebuild structurally cannot reproduce a tombstone -- indexer.py says so
    explicitly -- so counting them on the retiring side compares a number that
    includes them against one that cannot. A buyer who has deleted more than
    half their project's transcripts was hard-blocked on the FLAGLESS route:
    the first `ssgrep index` after any release that bumps SCHEMA_VERSION or
    the embedding model exited 2 blaming a scope mismatch that did not occur,
    and stayed wedged there -- new sessions written afterwards could not be
    indexed either.
    """
    _install_fake_encode(monkeypatch)
    project = tmp_path / "project"
    index_dir = tmp_path / "idx"
    written = _write_corpus(fake_home, project, 4)
    indexer.index(project, index_dir=index_dir, quiet=True)

    for path in written[1:]:  # 3 of 4 transcripts age off the disk
        path.unlink()
    tombstoning = indexer.index(project, index_dir=index_dir, quiet=True)
    assert tombstoning.tombstoned_source_count == 3, "precondition: the rows must be tombstoned"
    assert _live_counts(index_dir)[0] == 4, "precondition: tombstones stay in the sessions table"

    # The upgrade every buyer takes: a shipped release changes the model, so a
    # plain `ssgrep index` with no flag becomes a rebuild.
    live_db = store.GenerationalStore(index_dir).get_index_path()
    conn = sqlite3.connect(str(live_db))
    try:
        conn.execute("UPDATE meta SET value = ? WHERE key = 'model_id'", ("some-other/model-v2",))
        conn.commit()
    finally:
        conn.close()

    stats = indexer.index(project, index_dir=index_dir, quiet=True)  # must not raise

    assert stats.session_count == 1, "the one surviving transcript must be indexed"
    assert store.GenerationalStore(index_dir).current_generation == 1, "the rebuild must commit"


def test_a_real_scope_mismatch_is_still_refused_when_tombstones_exist(
    fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Excluding tombstones must not disarm the guard.

    The two cases are cleanly separable and this pins that they stay so: with
    the SAME tombstoned index, a rebuild whose scope matches nothing still
    finds zero present sessions against one, and must still be refused.
    """
    _install_fake_encode(monkeypatch)
    project = tmp_path / "project"
    index_dir = tmp_path / "idx"
    written = _write_corpus(fake_home, project, 4)
    indexer.index(project, index_dir=index_dir, quiet=True)
    for path in written[1:]:
        path.unlink()
    indexer.index(project, index_dir=index_dir, quiet=True)
    _strip_persisted_scope(index_dir)  # legacy-index trap; see helper docstring

    moved_project = tmp_path / "moved-project"
    with pytest.raises(RebuildWouldShrinkError):
        indexer.index(moved_project, index_dir=index_dir, rebuild=True, quiet=True)

    assert store.GenerationalStore(index_dir).current_generation == 0


def test_rebuild_with_no_subagents_is_not_refused_for_excluding_subagents(
    fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--rebuild --no-subagents` is a documented, supported combination.

    The flag drops whole sessions on purpose (discovery: `if no_subagents and
    not is_main: continue`), and README puts 82% of searchable content in
    subagents -- so for any subagent-heavy corpus the shrink is guaranteed and
    the rebuild was unconditionally refused, with a message blaming a scope
    mismatch the user could verify had not happened.
    """
    _install_fake_encode(monkeypatch)
    project = tmp_path / "project"
    index_dir = tmp_path / "idx"
    _write_corpus(fake_home, project, 2)
    _write_subagent_corpus(fake_home, project, 6)
    full = indexer.index(project, index_dir=index_dir, quiet=True)
    assert full.session_count == 8, "precondition: subagents must dominate the corpus"

    stats = indexer.index(project, index_dir=index_dir, rebuild=True, no_subagents=True, quiet=True)

    assert stats.session_count == 2, "only the main sessions survive, as requested"
    assert store.GenerationalStore(index_dir).current_generation == 1, "the rebuild must commit"
    assert _live_counts(index_dir)[0] == 2


def test_no_subagents_rebuild_is_still_refused_on_a_real_scope_mismatch(
    fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Making the guard flag-aware must not make it flag-blind.

    --no-subagents compares main-session counts on BOTH sides, so a scope
    mismatch that loses the main sessions too is still caught. Skipping the
    gate outright whenever the flag is passed would have lost this.
    """
    _install_fake_encode(monkeypatch)
    project = tmp_path / "project"
    index_dir = tmp_path / "idx"
    _write_corpus(fake_home, project, 2)
    _write_subagent_corpus(fake_home, project, 6)
    indexer.index(project, index_dir=index_dir, quiet=True)
    _strip_persisted_scope(index_dir)  # legacy-index trap; see helper docstring

    moved_project = tmp_path / "moved-project"
    with pytest.raises(RebuildWouldShrinkError) as excinfo:
        indexer.index(
            moved_project, index_dir=index_dir, rebuild=True, no_subagents=True, quiet=True
        )

    assert excinfo.value.old_counts[0] == 2, (
        "the retiring side must be counted main-only too, or the comparison is "
        f"still asymmetric: {excinfo.value.old_counts}"
    )
    assert "--no-subagents alone does not explain this" in str(excinfo.value), (
        "the message must not leave the user hunting for a cause the counts "
        f"already account for: {excinfo.value}"
    )
    assert store.GenerationalStore(index_dir).current_generation == 0


def test_refusal_names_causes_it_cannot_distinguish(
    fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The message must hedge where the counts cannot decide, and must not
    name a cause it has already ruled out.

    It used to assert one cause as fact -- "The most likely cause is a scope
    mismatch" -- which sends a user whose real cause was something else
    hunting for a scoping problem that does not exist. It also offered exactly
    one escape, --allow-shrink, which is the destructive one.

    It then over-corrected: it named deleted transcripts as a cause and
    `ssgrep prune` as the escape, when read_counts() excludes every session
    whose transcript is gone from BOTH sides. Deleted transcripts therefore
    cannot produce this refusal, and prune -- which only touches rows already
    tombstoned -- reports "Nothing to prune" for anyone who follows it. A
    remedy that cannot work is worse than no remedy.
    """
    _install_fake_encode(monkeypatch)
    project = tmp_path / "project"
    index_dir = tmp_path / "idx"
    _build_populated_index(fake_home, project, index_dir)

    with pytest.raises(RebuildWouldShrinkError) as excinfo:
        indexer.index(tmp_path / "moved", index_dir=index_dir, rebuild=True, quiet=True)

    message = str(excinfo.value)
    assert "scope mismatch" in message
    assert "--allow-shrink" in message
    assert "`ssgrep prune`" not in message, (
        "prune cannot clear this refusal -- it only deletes already-tombstoned "
        f"rows, and none exist on this path:\n{message}"
    )


def test_init_refuses_a_shrinking_rebuild_the_same_way_index_does(
    fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """`ssgrep init` takes the rebuild path too, and must not misreport it.

    init calls api.index(rebuild=False), but indexer computes `rebuild or
    needs_rebuild(...)`, so after any release that bumps SCHEMA_VERSION or the
    embedding model, init is gated as well. It used to fall into its generic
    `except Exception` handler: exit INTERNAL_FAILURE, and a remedy advising
    `ssgrep init` again or `ssgrep index --rebuild` -- both of which hit this
    same guard, and neither of which init has an --allow-shrink for.
    """
    from unittest.mock import MagicMock

    from ssgrep.cli import exit_codes
    from ssgrep.cli.commands.init import InitCommand

    _install_fake_encode(monkeypatch)
    project = tmp_path / "project"
    project.mkdir()
    before = _build_populated_index(fake_home, project, project / ".ssgrep")

    live_db = store.GenerationalStore(project / ".ssgrep").get_index_path()
    conn = sqlite3.connect(str(live_db))
    try:
        conn.execute("UPDATE meta SET value = ? WHERE key = 'model_id'", ("some-other/model-v2",))
        conn.commit()
    finally:
        conn.close()
    moved = tmp_path / "project-renamed"
    project.rename(moved)

    with pytest.raises(SystemExit) as excinfo:
        InitCommand(MagicMock()).handle(project_dir=str(moved))

    assert excinfo.value.code == exit_codes.USAGE_ERROR, (
        "nothing broke and nothing was written -- this is the same 'required "
        "confirmation' condition `ssgrep index` classifies as USAGE_ERROR"
    )
    stderr = capsys.readouterr().err
    assert f"{before[0]} sessions" in stderr, f"the counts must reach the user: {stderr}"
    assert "ssgrep index --rebuild` for full rebuild" not in stderr, (
        "must not advise a command that hits this same guard: " f"{stderr}"
    )
    assert _live_counts(moved / ".ssgrep") == before


def test_shrink_floor_is_the_boundary_it_claims_to_be() -> None:
    """check_shrink()'s threshold, exercised directly at the boundary.

    Halving exactly is allowed; one session below the floor is not; and a
    growing or steady rebuild is never touched.
    """
    assert rebuild_guard.SHRINK_FLOOR == 0.5

    rebuild_guard.check_shrink((10, 100, 500), (5, 50, 250))  # exactly at the floor
    rebuild_guard.check_shrink((10, 100, 500), (10, 100, 500))
    rebuild_guard.check_shrink((10, 100, 500), (40, 400, 2000))
    rebuild_guard.check_shrink((0, 0, 0), (0, 0, 0))

    with pytest.raises(RebuildWouldShrinkError):
        rebuild_guard.check_shrink((10, 100, 500), (4, 40, 200))
    with pytest.raises(RebuildWouldShrinkError):
        rebuild_guard.check_shrink((1, 5, 20), (0, 0, 0))

    # The override is what makes the floor a policy rather than a wall.
    rebuild_guard.check_shrink((10, 100, 500), (0, 0, 0), allow_shrink=True)


# ---------------------------------------------------------------------------
# C3 -- staging must be idempotent.
# ---------------------------------------------------------------------------


def test_rebuild_succeeds_after_an_earlier_rebuild_was_aborted_midway(
    fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Abort a rebuild after it has durably appended vectors to the staged
    generation, then rebuild again: the second rebuild must succeed.

    Before the fix this was permanently unrecoverable. The abandoned
    vectors.f32.1 rows stayed on disk, the retry appended its own rows after
    them, and the retry's chunks referenced only its own -- leaving the first
    attempt's rows orphaned, which validate_generations() rejects, so
    commit_generation() raised "not mutually consistent" on that rebuild and
    on every rebuild after it.

    The abort mechanism is verified, not assumed: the assertions below check
    that staged artifacts really are left behind before the retry runs.
    """
    _install_fake_encode(monkeypatch)
    project = tmp_path / "project"
    index_dir = tmp_path / "idx"
    before = _build_populated_index(fake_home, project, index_dir, count=3)

    # Ctrl-C mid-rebuild, modelled where a real one lands: after at least one
    # session's vectors are already appended to the staged generation and long
    # before the manifest is touched.
    calls = {"n": 0}

    def interrupting_encode(texts: list[str]) -> np.ndarray:
        calls["n"] += 1
        if calls["n"] >= 2:
            raise KeyboardInterrupt("simulated Ctrl-C mid-rebuild")
        return _fake_vectors(len(texts))

    monkeypatch.setattr(embed, "encode", interrupting_encode)

    aborted = False
    try:
        indexer.index(project, index_dir=index_dir, rebuild=True, quiet=True)
    except KeyboardInterrupt:
        aborted = True
    # A real Ctrl-C ends the process, which releases the aborted run's sqlite
    # connection and mmap. This test keeps running in-process, so it has to
    # drop them explicitly or the retry contends with its own leftovers.
    gc.collect()

    assert aborted, "the rebuild must actually have been interrupted"
    assert calls["n"] >= 2, "at least one session must have been embedded before the abort"
    staged_db = index_dir / "index.db.1"
    staged_vec = index_dir / "vectors.f32.1"
    assert staged_db.exists(), "the abort must genuinely leave a staged index.db behind"
    assert staged_vec.stat().st_size > 0, (
        "the abort must genuinely leave staged vector rows behind -- with an empty "
        "vectors.f32.1 this test would prove nothing about the retry"
    )
    assert (
        store.GenerationalStore(index_dir).current_generation == 0
    ), "the aborted rebuild must not have been committed"
    assert _live_counts(index_dir) == before, "the live generation must be untouched by the abort"

    _install_fake_encode(monkeypatch)
    recovered = indexer.index(project, index_dir=index_dir, rebuild=True, quiet=True)

    assert (recovered.session_count, recovered.episode_count, recovered.chunk_count) == before
    assert store.GenerationalStore(index_dir).current_generation == 1
    assert _live_counts(index_dir) == before


def test_stage_generation_clears_residue_from_an_abandoned_attempt(tmp_path: Path) -> None:
    """stage_generation(N) must hand back paths cleared of everything a
    previous attempt at N left, SQLite's -wal/-shm sidecars included.
    """
    index_dir = tmp_path / ".ssgrep"
    index_dir.mkdir()
    live_db = index_dir / "index.db"
    live_db.write_bytes(b"the live index")
    _committed_manifest(index_dir, generation=0)
    gen_store = store.GenerationalStore(index_dir)

    residue = {
        index_dir / "index.db.1": b"abandoned database",
        index_dir / "index.db.1-wal": b"abandoned wal",
        index_dir / "index.db.1-shm": b"abandoned shm",
        index_dir / "vectors.f32.1": b"\x00" * (DIM * 4 * 3),
    }
    for path, payload in residue.items():
        path.write_bytes(payload)

    staged_db, staged_vec = gen_store.stage_generation(1)

    assert staged_db == index_dir / "index.db.1"
    assert staged_vec == index_dir / "vectors.f32.1"
    for path in residue:
        assert not path.exists(), f"{path.name} must be unlinked before the generation is reused"
    assert live_db.read_bytes() == b"the live index", "the live generation must survive staging"


def test_staging_never_deletes_the_live_generation(tmp_path: Path) -> None:
    """The cleanup must not become the very data loss it exists to prevent:
    the current generation's files are the live index, not staging residue.
    """
    index_dir = tmp_path / ".ssgrep"
    index_dir.mkdir()
    gen_store = store.GenerationalStore(index_dir)
    live_db = index_dir / "index.db"
    live_vec = index_dir / "vectors.f32"
    live_db.write_bytes(b"the live index")
    live_vec.write_bytes(b"the live vectors")

    gen_store.stage_generation(gen_store.current_generation)

    assert live_db.read_bytes() == b"the live index"
    assert live_vec.read_bytes() == b"the live vectors"

    with pytest.raises(ValueError, match="not newer than the live generation"):
        gen_store.discard_generation(gen_store.current_generation)


def test_staging_cannot_delete_a_generation_another_run_committed_meanwhile(
    tmp_path: Path,
) -> None:
    """The refusal must be decided against the manifest ON DISK, not a cached one.

    stage_generation() now unlinks, so its guard is a destructive operation's
    only safety check -- and it read ``self.current_generation``, the value
    cached when __init__ read the manifest. indexer.index() constructs its
    store at the top of a run and reaches staging much later, so this window
    is entirely ordinary: two runs can be on the rebuild path at once with no
    --rebuild flag anywhere, since `force_rebuild = rebuild or
    needs_rebuild(...)` sends every buyer down it after a release that bumps
    SCHEMA_VERSION or the embedding model.

    Against a stale cached 0, `next_gen=1 > 0` was still true for the
    generation the manifest had since been moved to -- so the unlink destroyed
    the LIVE index while the manifest went on naming it. Nothing here pokes at
    private state: store A is simply constructed before B commits.
    """
    index_dir = tmp_path / ".ssgrep"
    index_dir.mkdir()

    a = store.GenerationalStore(index_dir)  # constructed while the manifest says 0
    b = store.GenerationalStore(index_dir)

    db1, vec1 = b.stage_generation(1)
    store.init_db(db1).close()
    vec1.write_bytes(b"")
    b.commit_generation(1, vector_row_count=0)  # generation 1 is now LIVE

    assert a.current_generation == 0, "precondition: A's cached view must be stale"
    assert store.GenerationalStore(index_dir).current_generation == 1

    with pytest.raises(ValueError, match="not newer than the live generation"):
        a.stage_generation(a.current_generation + 1)

    assert db1.exists(), "the live index.db.1 must survive a stale run's staging"
    assert vec1.exists(), "the live vectors.f32.1 must survive a stale run's staging"
    assert store.GenerationalStore(index_dir).current_generation == 1


def test_discard_refuses_a_generation_another_run_holds_open(tmp_path: Path) -> None:
    """discard_generation() must take the lock the rest of the module uses.

    The manifest re-read closes the case where a commit has already landed;
    the flock closes the one where it is landing right now. commit_generation()
    holds this same lock across its validate-then-write-manifest sequence, so
    an interleaving that would leave the manifest pointing at unlinked files
    cannot occur -- whichever side gets the lock first wins outright, and the
    loser refuses without destroying anything.
    """
    index_dir = tmp_path / ".ssgrep"
    index_dir.mkdir()
    _committed_manifest(index_dir, generation=0)
    gen_store = store.GenerationalStore(index_dir)
    staged_db, staged_vec = gen_store.stage_generation(1)
    staged_db.write_bytes(b"a generation someone else is committing")
    staged_vec.write_bytes(b"its vectors")

    # A second opener holding generation 1 exclusively, exactly as
    # commit_generation() does while it writes the manifest.
    lock_fd = os.open(gen_store._generation_lock_path(1), os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    try:
        with pytest.raises(ValueError, match="holds it open"):
            gen_store.discard_generation(1)
        assert staged_db.read_bytes() == b"a generation someone else is committing"
        assert staged_vec.exists()
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)

    # Once released, the same call proceeds normally.
    gen_store.discard_generation(1)
    assert not staged_db.exists()


def test_commit_generation_holds_the_generation_lock_while_it_commits(tmp_path: Path) -> None:
    """The other half of the protocol, observed rather than assumed.

    If commit_generation() did not hold generation N's lock, a concurrent
    discard_generation(N) could unlink N's files in the window between
    validate_generations() passing and _write_manifest() landing -- producing
    the exact end state both guards exist to prevent: a manifest naming a
    generation whose files are gone. Here the lock is held by the test, and
    the commit must be observably blocked until it is released.
    """
    index_dir = tmp_path / ".ssgrep"
    index_dir.mkdir()
    gen_store = store.GenerationalStore(index_dir)
    staged_db, staged_vec = gen_store.stage_generation(1)
    store.init_db(staged_db).close()
    staged_vec.write_bytes(b"")

    lock_fd = os.open(gen_store._generation_lock_path(1), os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(lock_fd, fcntl.LOCK_EX)

    committed = threading.Event()
    failure: list[BaseException] = []

    def commit() -> None:
        try:
            gen_store.commit_generation(1, vector_row_count=0)
        except BaseException as error:  # pragma: no cover - surfaced via `failure`
            failure.append(error)
        finally:
            committed.set()

    worker = threading.Thread(target=commit, daemon=True)
    worker.start()
    try:
        assert not committed.wait(0.5), (
            "commit_generation() reached the manifest while another opener held "
            "generation 1's lock -- it is not participating in the lock protocol "
            "discard_generation() relies on"
        )
        assert store.GenerationalStore(index_dir).current_generation == 0
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)

    assert committed.wait(10), "commit_generation() never completed after the lock was released"
    worker.join(timeout=10)
    assert not failure, f"commit_generation() raised once unblocked: {failure}"
    assert store.GenerationalStore(index_dir).current_generation == 1


# ---------------------------------------------------------------------------
# The CLI contract: a refused rebuild must be visibly, machine-detectably
# refused. Exiting 0 with a cheerful "Indexed 0 sessions" line is what made
# the original defect silent.
# ---------------------------------------------------------------------------


def test_cli_index_exits_nonzero_and_reports_counts_when_a_rebuild_would_shrink(
    fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Driven through IndexCommand.handle(), which is what the `ssgrep index`
    entry point actually runs -- only embed.encode() is stubbed.

    The exit code is USAGE_ERROR (2), the "required confirmations" arm of the
    exit-code contract: nothing broke and nothing was written, the operation is
    refused pending --allow-shrink. It must NOT be 0, which is what an agent or
    script would read as "your history was reindexed fine".
    """
    from unittest.mock import MagicMock

    from ssgrep.cli import exit_codes
    from ssgrep.cli.commands.index import IndexCommand

    _install_fake_encode(monkeypatch)
    project = tmp_path / "project"
    project.mkdir()
    before = _build_populated_index(fake_home, project, project / ".ssgrep")

    # The scope mismatch as a buyer hits it: the project directory is renamed,
    # so .ssgrep/ travels with it while every transcript still records the OLD
    # path as its cwd. Discovery is cwd-matched (D11), so the rebuild that runs
    # here discovers nothing at all.
    moved = tmp_path / "project-renamed"
    project.rename(moved)

    with pytest.raises(SystemExit) as excinfo:
        IndexCommand(MagicMock()).handle(project_dir=str(moved), rebuild=True)

    assert excinfo.value.code == exit_codes.USAGE_ERROR
    stderr = capsys.readouterr().err
    assert f"{before[0]} sessions" in stderr, f"old counts must reach the user: {stderr}"
    assert "0 sessions" in stderr, f"new counts must reach the user: {stderr}"
    assert "scope mismatch" in stderr
    assert "--allow-shrink" in stderr, "the user must be told how to proceed deliberately"
    assert _live_counts(moved / ".ssgrep") == before

    # And the documented escape hatch works from the same entry point.
    assert (
        IndexCommand(MagicMock()).handle(project_dir=str(moved), rebuild=True, allow_shrink=True)
        is None
    )
    assert _live_counts(moved / ".ssgrep") == (0, 0, 0)


def test_cli_refusal_carries_the_census_and_a_non_destructive_command(
    fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """The one path that has DEFINITIVELY detected impending data loss must not
    also be the least informative one.

    check_shrink() raises before handle() ever reaches its session_count == 0
    branch, so the zero-discovery census never ran on the refusal path: the
    message asserted "scope mismatch" without naming the rejected cwds, the
    transcript root, or the canonical scope -- all of which build_scope_report
    already computes. And the only command offered, in both the text and the
    machine-readable ``command`` field, was
    ``ssgrep index --rebuild --allow-shrink``: the action that discards the
    buyer's index. An agent that auto-executes ``command`` on failure would
    destroy the index this guard just saved.
    """
    from unittest.mock import MagicMock

    from ssgrep.cli import exit_codes
    from ssgrep.cli.commands.index import IndexCommand, _attach_census

    _install_fake_encode(monkeypatch)
    original_project = tmp_path / "original-project"
    index_dir = tmp_path / "idx"
    _build_populated_index(fake_home, original_project, index_dir)

    moved_project = tmp_path / "moved-project"
    moved_project.mkdir()
    (moved_project / ".ssgrep").mkdir()
    for artifact in index_dir.iterdir():
        (moved_project / ".ssgrep" / artifact.name).write_bytes(artifact.read_bytes())

    with pytest.raises(SystemExit) as excinfo:
        IndexCommand(MagicMock()).handle(project_dir=str(moved_project), rebuild=True)

    assert excinfo.value.code == exit_codes.USAGE_ERROR
    stderr = capsys.readouterr().err
    assert str(original_project) in stderr, (
        "the refusal must name the cwd the transcripts actually record -- it is "
        f"the evidence for the 'scope mismatch' it claims:\n{stderr}"
    )
    assert "transcripts recorded cwd=" in stderr, f"the census must be shown:\n{stderr}"
    assert (
        str(fake_home / ".claude" / "projects") in stderr
    ), f"the transcript root scanned must be named:\n{stderr}"

    # The machine-readable remedy: corrective, or absent. Never destructive.
    error = RebuildWouldShrinkError("refused", old_counts=(4, 4, 8), new_counts=(0, 0, 0))
    assert error.command is None, (
        "the default must not be --allow-shrink: a multi-project buyer gets a flat "
        "rejected-cwd histogram, no remedy is found, and the default is what a "
        "`jq -r .command | sh` wrapper would then execute"
    )
    _attach_census(error, moved_project)
    assert error.command == f"ssgrep index --scope {original_project}", (
        "a caller that follows `command` on failure must be steered to the fix, "
        f"not to the command that discards the index: {error.command!r} -- and "
        "since the --scope flag exists, the fix is indexing the recorded cwd's "
        "history into THIS project's index, not a second stranded index at the "
        "old path"
    )


def test_cli_accepts_the_allow_shrink_flag(tmp_path: Path) -> None:
    """`--allow-shrink` must exist as a real CLI flag, not just a Python
    keyword argument: usecli derives flags from handle()'s signature, so a
    renamed or missing parameter leaves the override unreachable from the
    command line and turns the guard into a dead end.

    Run against an empty isolated HOME, so no embedding model is ever loaded.
    """
    import os
    import subprocess

    from tests.conftest import get_ssgrep_binary

    home = tmp_path / "home"
    (home / ".claude" / "projects").mkdir(parents=True)
    project = tmp_path / "project"
    project.mkdir()

    env = os.environ.copy()
    env["HOME"] = str(home)
    # resolve_claude_dir() reads CLAUDE_CONFIG_DIR BEFORE HOME, so leaving an
    # inherited value in place would run this subprocess against the
    # developer's real corpus no matter what HOME says. conftest's autouse
    # fixture already removes it from os.environ; this is belt-and-braces for
    # the copy, since the cost of being wrong here is touching real data.
    env.pop("CLAUDE_CONFIG_DIR", None)
    result = subprocess.run(
        [str(get_ssgrep_binary()), "index", "--project-dir", str(project), "--allow-shrink"],
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, f"--allow-shrink must be a known flag: {result.stderr}"


# ---------------------------------------------------------------------------
# The wedge: a rebuild refused for deletions nothing ever tombstoned.
# ---------------------------------------------------------------------------


def _bump_schema_version(index_dir: Path) -> None:
    """Age the stored schema so needs_rebuild() is true, as a shipped release
    that bumps SCHEMA_VERSION does for every buyer at once.
    """
    live_db = store.GenerationalStore(index_dir).get_index_path()
    conn = sqlite3.connect(str(live_db))
    try:
        conn.execute("UPDATE meta SET value = '1' WHERE key = 'schema_version'")
        conn.commit()
    finally:
        conn.close()


def test_deletions_the_incremental_path_never_saw_do_not_wedge_the_rebuild(
    fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tombstone exclusion must hold even when nothing has tombstoned.

    test_deleted_transcripts_do_not_block_a_flagless_rebuild covers the case
    where an ordinary `ssgrep index` ran between the deletion and the
    upgrade, writing tombstones for read_counts() to exclude. That run is the
    ONLY writer of ``source_status = 'absent'``:
    indexer._reconcile_vanished_and_reappeared() is called under
    ``if not force_rebuild:`` and nowhere else. So when the deletion happens
    first and the schema bump second -- the ordinary order, since `ssgrep
    search` never indexes and the SessionEnd hook only enqueues -- there are
    no tombstones, the exclusion filters zero rows, and the refusal fires.

    It then fires forever: a refusal commits nothing, so the live
    schema_version never advances, needs_rebuild() stays true, and every
    future `ssgrep index` (flag or no flag) takes the gated path. The index
    stops updating and, because search checks schema_version too, stops being
    readable -- with the only escape being the --allow-shrink flag the same
    message calls irreversible, against transcripts that no longer exist
    anywhere else.
    """
    _install_fake_encode(monkeypatch)
    project = tmp_path / "project"
    index_dir = tmp_path / "idx"
    written = _write_corpus(fake_home, project, 8)
    indexer.index(project, index_dir=index_dir, quiet=True)

    for path in written[2:]:  # 6 of 8 archived off to cold storage
        path.unlink()

    # The precondition that separates this from the tombstoned case: no
    # `ssgrep index` ran in between, so nothing is marked absent.
    live_db = store.GenerationalStore(index_dir).get_index_path()
    conn = sqlite3.connect(str(live_db))
    try:
        tombstoned = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE source_status = 'absent'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert tombstoned == 0, "precondition: the deletions must be unobserved"

    _bump_schema_version(index_dir)

    stats = indexer.index(project, index_dir=index_dir, quiet=True)  # must not raise

    assert stats.session_count == 2, "the two surviving transcripts must be indexed"
    assert _live_counts(index_dir)[0] == 2, "the rebuild must actually be committed"

    # And the index must be usable again afterwards, which is the property the
    # wedge destroyed: a fresh transcript indexes normally.
    _write_transcript(fake_home, project, "dddddddd-0000-4000-8000-000000000009")
    assert indexer.index(project, index_dir=index_dir, quiet=True).session_count == 3


def test_a_scope_mismatch_is_still_refused_when_transcripts_were_also_deleted(
    fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Excluding vanished sources must not disarm the guard.

    The exclusion is what unwedges the deletion case; it must not also let a
    real scope mismatch through. Here the surviving transcripts are perfectly
    present -- they are simply not found, because the scope moved.
    """
    _install_fake_encode(monkeypatch)
    project = tmp_path / "project"
    index_dir = tmp_path / "idx"
    written = _write_corpus(fake_home, project, 8)
    indexer.index(project, index_dir=index_dir, quiet=True)
    _strip_persisted_scope(index_dir)  # legacy-index trap; see helper docstring
    for path in written[6:]:  # a couple genuinely deleted
        path.unlink()

    with pytest.raises(RebuildWouldShrinkError):
        indexer.index(tmp_path / "moved", index_dir=index_dir, rebuild=True, quiet=True)

    assert _live_counts(index_dir)[0] == 8, "the refusal must leave the index untouched"


def test_the_refusal_does_not_claim_zero_discovery_when_it_found_transcripts(
    fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The census attached to a partial-shrink refusal must not contradict the
    counts printed three lines above it.

    _attach_census rendered the census with its zero-discovery headline
    unconditionally, so a buyer read "rebuilt: N sessions" and then "ssgrep
    found no session transcripts for this project", next to "rejected by
    scope: 0". The one message that has definitively detected impending data
    loss was also the one asserting something demonstrably false, and the
    false claim named a cause -- a scope bug -- the buyer did not have.
    """
    from ssgrep.cli.commands.index import _attach_census

    _install_fake_encode(monkeypatch)
    project = tmp_path / "project"
    index_dir = tmp_path / "idx"
    _write_corpus(fake_home, project, 8)
    indexer.index(project, index_dir=index_dir, quiet=True)

    error = RebuildWouldShrinkError("refused", old_counts=(8, 8, 16), new_counts=(2, 2, 4))
    _attach_census(error, project)
    message = str(error)

    assert "found no session transcripts" not in message, (
        "discovery found this project's transcripts -- claiming otherwise invents a "
        f"scope bug the buyer does not have:\n{message}"
    )
    assert (
        "transcripts there: 8" in message
    ), f"the census itself must still be attached as evidence:\n{message}"


def test_the_refusal_never_points_a_caller_at_an_unrelated_project(
    fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`command` must not be repointed at another project when scope matched.

    Any buyer works in more than one project, so ~/.claude/projects holds
    every project's transcripts. On a refusal for project A, the census
    counted project B's transcripts as "rejected by scope", found B dominant,
    and overwrote ``error.command`` with `ssgrep index --project-dir <B>`.
    A caller that follows the field indexes an unrelated directory and leaves
    a stray .ssgrep in it, while A's actual problem goes untouched -- and the
    accompanying prose asserts A "moved or was renamed" when it did not.
    """
    from ssgrep.cli.commands.index import _attach_census

    _install_fake_encode(monkeypatch)
    project = tmp_path / "project"
    other = tmp_path / "other-project"
    index_dir = tmp_path / "idx"
    _write_corpus(fake_home, project, 2)
    for i in range(6):  # the buyer's busier other project
        _write_transcript(fake_home, other, f"eeeeeeee-0000-4000-8000-00000000000{i}")
    indexer.index(project, index_dir=index_dir, quiet=True)

    error = RebuildWouldShrinkError("refused", old_counts=(2, 2, 4), new_counts=(1, 1, 1))
    _attach_census(error, project)

    assert error.command is None or str(other) not in error.command, (
        "scope matched this project's own transcripts, so the buyer's other "
        f"project is not a remedy: {error.command!r}"
    )
    assert "moved or was renamed" not in str(error), f"nothing moved -- scope matched:\n{error}"
