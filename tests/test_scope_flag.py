"""Tests for the --scope flag: discovery scope decoupled from --project-dir.

The design's load-bearing property is STALENESS COHERENCE: an index built
with a foreign scope must not look vanished/stale when searched from the
project directory. Dropping search's persisted-scope read (the mutation this
file exists to catch) makes test_staleness_coherent_for_foreign_scope_index
go red.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ssgrep import api, indexer, indexer_support, staleness
from ssgrep import search as search_module

OLD_SCOPE = "/Users/someone/code/oldorg/moved-project"


def _write_transcript(path: Path, cwd: str, session_id: str) -> None:
    records = [
        {
            "parentUuid": None,
            "isSidechain": False,
            "type": "user",
            "message": {"role": "user", "content": "How do I fix the widget cache TTL bug?"},
            "uuid": f"{session_id}-u1",
            "timestamp": "2026-07-01T10:00:00.000Z",
            "cwd": cwd,
            "sessionId": session_id,
            "gitBranch": "main",
        },
        {
            "parentUuid": f"{session_id}-u1",
            "isSidechain": False,
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Set the widget cache TTL in settings."}],
            },
            "uuid": f"{session_id}-a1",
            "timestamp": "2026-07-01T10:01:00.000Z",
            "cwd": cwd,
            "sessionId": session_id,
            "gitBranch": "main",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


@pytest.fixture
def moved_corpus(tmp_path, monkeypatch):
    """A corpus recorded entirely under OLD_SCOPE, plus a project at a new
    path that matches none of it -- the org-rename scenario."""
    home = tmp_path / "home"
    projects = home / ".claude" / "projects"
    encoded = OLD_SCOPE.replace("/", "-").replace(".", "-")
    for i in range(3):
        _write_transcript(projects / encoded / f"moved-{i:04d}.jsonl", OLD_SCOPE, f"moved-{i:04d}")
    project = tmp_path / "code" / "neworg" / "moved-project"
    project.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    return project


def test_scope_indexes_foreign_history_into_local_index(moved_corpus):
    """The field scenario: history under the old org path, indexed from the
    new checkout via scope -- discovery must find all of it."""
    project = moved_corpus
    stats = indexer.index(project, quiet=True, index_dir=project / ".ssgrep", scope=OLD_SCOPE)
    assert stats.session_count == 3, (
        f"scope={OLD_SCOPE} must discover the 3 old-path sessions; got " f"{stats.session_count}"
    )


def test_scope_is_persisted_in_index_meta(moved_corpus):
    project = moved_corpus
    indexer.index(project, quiet=True, index_dir=project / ".ssgrep", scope=OLD_SCOPE)
    from ssgrep import store

    gen = store.GenerationalStore(project / ".ssgrep")
    persisted = indexer_support.read_persisted_scope(gen.get_index_path())
    assert persisted == OLD_SCOPE


def test_staleness_coherent_for_foreign_scope_index(moved_corpus):
    """THE load-bearing property: search-time staleness must compare against
    the persisted scope, not re-derive from project_dir. Without the
    persisted-scope read, every cursor file reads as VANISHED and the index
    reports total staleness -- the destructive-rebuild trap."""
    project = moved_corpus
    indexer.index(project, quiet=True, index_dir=project / ".ssgrep", scope=OLD_SCOPE)

    report = search_module.staleness_summary(project)
    vanished = [f for f in report.stale_files if f.status == staleness.FileStatus.VANISHED]
    assert vanished == [], (
        "a foreign-scope index must not report its own files as vanished at "
        f"search time; got {len(vanished)} vanished of {len(report.stale_files)} stale"
    )
    assert not staleness.is_index_stale(
        report
    ), "freshly built foreign-scope index must not be stale"


def test_search_finds_foreign_scope_content_end_to_end(moved_corpus):
    project = moved_corpus
    indexer.index(project, quiet=True, index_dir=project / ".ssgrep", scope=OLD_SCOPE)
    # search() resolves index at project_dir/.ssgrep -- same layout we built
    response = api.search(project, "widget cache TTL")
    assert response.results, "old-scope content must be searchable from the new project"
    assert not response.stale


def test_omitted_scope_preserves_the_persisted_scope(moved_corpus):
    """THE blind-review blocker's regression test: scope is an index
    property. A plain re-index (no --scope) after a scoped build -- the
    normal way every internal caller reindexes, including MCP startup
    reconciliation and `ssgrep init` -- must KEEP the persisted scope: no
    forced rebuild, no shrink refusal, no silently discarded history."""
    project = moved_corpus
    indexer.index(project, quiet=True, index_dir=project / ".ssgrep", scope=OLD_SCOPE)
    from ssgrep import store

    gen = store.GenerationalStore(project / ".ssgrep")
    generation_before = gen.current_generation

    # The exact shape of mcp_server._drain_startup_workqueue's call.
    stats = indexer.index(project, quiet=True, index_dir=project / ".ssgrep")

    assert stats.session_count == 3, (
        "a plain re-index must keep the scoped history; losing sessions here "
        "is the silent-data-loss failure the blind review reproduced"
    )
    assert (
        indexer_support.read_persisted_scope(gen.get_index_path()) == OLD_SCOPE
    ), "the persisted scope must survive an omitted-flag call"
    gen_after = store.GenerationalStore(project / ".ssgrep")
    assert (
        gen_after.current_generation == generation_before
    ), "an omitted-flag re-index must be incremental, not a forced rebuild"


def test_explicit_scope_change_forces_full_rebuild(moved_corpus):
    """Only an EXPLICIT, different --scope changes scope -- and that change
    rebuilds (like a model change), with the shrink guard still protecting
    the content it would discard."""
    project = moved_corpus
    indexer.index(project, quiet=True, index_dir=project / ".ssgrep", scope=OLD_SCOPE)
    from ssgrep.types import RebuildWouldShrinkError

    # Explicitly changing scope to the (empty) project path: forced rebuild;
    # the shrink guard refuses the 3 -> 0 collapse without consent.
    with pytest.raises(RebuildWouldShrinkError):
        indexer.index(project, quiet=True, index_dir=project / ".ssgrep", scope=str(project))
    # With explicit consent the scope change lands and is persisted.
    stats = indexer.index(
        project,
        quiet=True,
        index_dir=project / ".ssgrep",
        scope=str(project),
        allow_shrink=True,
    )
    assert stats.session_count == 0
    from ssgrep import store

    gen = store.GenerationalStore(project / ".ssgrep")
    assert indexer_support.read_persisted_scope(gen.get_index_path()) == str(project)


def test_default_scope_unchanged(moved_corpus):
    """No scope argument == exact legacy behavior: project_dir scope,
    persisted as such."""
    project = moved_corpus
    stats = indexer.index(project, quiet=True, index_dir=project / ".ssgrep")
    assert stats.session_count == 0  # nothing recorded under the new path
    from ssgrep import store

    gen = store.GenerationalStore(project / ".ssgrep")
    assert indexer_support.read_persisted_scope(gen.get_index_path()) == str(project)


def test_sessionend_enqueue_uses_the_persisted_scope(moved_corpus):
    """The scope-re-derivation consumer the blind review told us to hunt in
    hooks paths: the SessionEnd enqueue must discover at the index's
    persisted scope, not project_dir -- otherwise a scoped index's work
    queue gets out-of-scope transcripts injected through the drain."""
    from ssgrep.cli.commands.hooks import _enqueue_sessions
    from ssgrep.workqueue import WorkQueue

    project = moved_corpus
    indexer.index(project, quiet=True, index_dir=project / ".ssgrep", scope=OLD_SCOPE)

    result = _enqueue_sessions(project)
    assert result["error"] is None
    assert result["enqueued"] == 3, (
        "the hook must enqueue the SCOPED sessions (3 under the old path), "
        f"not the 0 sessions recorded under project_dir; got {result}"
    )
    queue = WorkQueue(project / ".ssgrep")
    queue.open()
    try:
        pending = queue.pending()
        assert len(pending) == 3
        assert all("moved-" in item.session_id for item in pending)
    finally:
        queue.close()


def test_notes_ride_a_scoped_index(moved_corpus):
    """Design trap #3 pinned: notes follow the INDEX they were written into,
    independent of --scope. A scoped index (foreign history) plus a local
    note must index both, keep staleness coherent, and retrieve the note."""
    from ssgrep import notes as notes_mod

    project = moved_corpus
    notes_mod.write_note(
        project,
        "what is the frobnicator retry policy",
        "Exponential backoff with jitter, capped at 30s.",
    )
    stats = indexer.index(project, quiet=True, index_dir=project / ".ssgrep", scope=OLD_SCOPE)
    assert (
        stats.session_count == 4
    ), f"3 scoped sessions + 1 note shard expected; got {stats.session_count}"

    report = search_module.staleness_summary(project)
    assert not staleness.is_index_stale(report), (
        "scoped index + note must be staleness-coherent (neither the scoped "
        "files nor the note shard may read as vanished)"
    )

    response = api.search(project, "frobnicator retry policy")
    assert response.results and response.results[0].ref.startswith(
        "notes-"
    ), "the note must be retrievable from the scoped index"


def test_drain_drops_out_of_scope_queue_items(moved_corpus, tmp_path):
    """The re-review's unanimous blocker, drain side: the queue must not be
    trusted. An out-of-scope item (enqueued by an older binary, a stale
    process, or a foreign hook -- here injected directly, bypassing the
    fixed enqueue) must be DROPPED by the drain: not indexed into the
    scoped index, and not left wedging the queue. Reproduces the reviewers'
    exact scenario: scoped index (3 sessions), out-of-scope session queued,
    plain omitted-scope reindex (the MCP-startup shape) -- session_count
    must STAY 3."""
    from ssgrep.workqueue import WorkQueue

    project = moved_corpus
    indexer.index(project, quiet=True, index_dir=project / ".ssgrep", scope=OLD_SCOPE)

    # A real transcript recorded at the CURRENT project_dir (out of scope).
    home = Path(os.environ["HOME"])
    encoded = str(project).replace("/", "-").replace(".", "-")
    foreign = home / ".claude" / "projects" / encoded / "local-0001.jsonl"
    _write_transcript(foreign, str(project), "local-0001")

    # Inject the hint directly -- the enqueue-side filter is bypassed on
    # purpose (that is the drain's threat model).
    queue = WorkQueue(project / ".ssgrep")
    queue.open()
    try:
        queue.enqueue(
            transcript_path=str(foreign),
            session_id="local-0001",
            reason="stale/foreign hint",
        )
    finally:
        queue.close()

    stats = indexer.index(project, quiet=True, index_dir=project / ".ssgrep")

    assert stats.session_count == 3, (
        "an out-of-scope queue item must not be absorbed into a scoped "
        f"index (reviewers reproduced 3->4 here); got {stats.session_count}"
    )
    # Stronger than a counter: the foreign session must be absent from the
    # sessions table itself.
    import sqlite3

    from ssgrep import store

    gen = store.GenerationalStore(project / ".ssgrep")
    conn = sqlite3.connect(str(gen.get_index_path()))
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE session_id = ?", ("local-0001",)
        ).fetchone()
    finally:
        conn.close()
    assert row[0] == 0, "the out-of-scope session must not exist in the index"
    # Observability chain (blind-review nit): the drop is surfaced in the
    # returned stats AND persisted so `ssgrep status` reports it.
    assert (
        stats.queue_items_out_of_scope == 1
    ), "the drop must be surfaced in IndexStats, not just counted internally"
    from ssgrep import api

    status_stats = api.status(project)
    assert (
        status_stats.queue_items_out_of_scope == 1
    ), "status must report the last run's dropped-hint count (persisted meta)"
    queue = WorkQueue(project / ".ssgrep")
    queue.open()
    try:
        assert (
            queue.pending() == [] and queue.abandoned() == []
        ), "the dropped item must be completed, not left wedging the queue"
    finally:
        queue.close()


def test_mcp_startup_drain_preserves_a_scoped_index(moved_corpus):
    """The re-review's named coverage gap: the MCP startup reconciliation
    path (mcp_server._drain_startup_workqueue -> indexer.index with scope
    omitted, exceptions swallowed) against a --scope-built index. It must
    preserve the scoped history and the persisted scope -- this exact path
    is where both reviewers traced the original blocker's silent breakage."""
    import ssgrep.mcp_server as mcp_server

    project = moved_corpus
    # The MCP server resolves <project>/.ssgrep, so build in the real layout.
    indexer.index(project, quiet=True, index_dir=project / ".ssgrep", scope=OLD_SCOPE)

    mcp_server.set_project_dir(project)
    try:
        mcp_server._drain_startup_workqueue()
    finally:
        mcp_server.set_project_dir(None)

    from ssgrep import store

    gen = store.GenerationalStore(project / ".ssgrep")
    assert (
        indexer_support.read_persisted_scope(gen.get_index_path()) == OLD_SCOPE
    ), "MCP startup reconciliation must not revert the persisted scope"
    import sqlite3

    conn = sqlite3.connect(str(gen.get_index_path()))
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE source_status != 'absent'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert count == 3, f"MCP startup must keep the 3 scoped sessions intact; got {count}"


def test_post_run_census_uses_the_persisted_scope_when_flag_omitted(moved_corpus, capsys):
    """The notes-review blocker: post-run diagnostics must census the
    EFFECTIVE scope (persisted when --scope is omitted), not raw
    project_dir -- otherwise the suspicious-discovery remedy can point an
    auto-executing agent at an unrelated same-basename directory."""
    from unittest.mock import MagicMock

    from ssgrep.cli.commands.index import IndexCommand

    project = moved_corpus
    # Scoped index with exactly ONE session so the ==1 path fires on re-run.

    home = Path(os.environ["HOME"])
    projects = home / ".claude" / "projects"
    encoded_old = OLD_SCOPE.replace("/", "-").replace(".", "-")
    for extra in sorted((projects / encoded_old).glob("moved-*.jsonl"))[1:]:
        extra.unlink()
    indexer.index(project, quiet=True, index_dir=project / ".ssgrep", scope=OLD_SCOPE)

    # A same-basename sibling that must NOT be recommended: it is rejected
    # only against a project_dir census, not against the persisted scope.
    sibling = f"/Users/elsewhere/{Path(str(project)).name}"
    encoded_sib = sibling.replace("/", "-").replace(".", "-")
    _write_transcript(projects / encoded_sib / "sib-0001.jsonl", sibling, "sib-0001")

    # Plain re-run (no --scope): ==1 fires; census must run at OLD_SCOPE,
    # where the sibling is rejected -- but the warning must not recommend
    # re-scoping to the already-active OLD_SCOPE either.
    IndexCommand(MagicMock()).handle(project_dir=str(project))
    err = capsys.readouterr().err
    assert sibling not in err, (
        "an unrelated same-basename directory must not be recommended when "
        f"the persisted scope is active:\n{err}"
    )


def test_search_zero_discovery_censuses_the_persisted_scope(moved_corpus):
    """Blind-review minor: `ssgrep search`'s zero-discovery census (and the
    MCP tool's -- same builder) must name the EFFECTIVE persisted scope,
    not raw project_dir, when a scoped index comes up empty."""
    from ssgrep import scope_report
    from ssgrep.indexer_support import effective_scope

    project = moved_corpus
    bogus = "/nonexistent/typo/path"
    indexer.index(project, quiet=True, index_dir=project / ".ssgrep", scope=bogus)
    # The exact expression the three fixed call sites now use:
    census_scope = effective_scope(project)
    assert census_scope == bogus, (
        "the census scope must be the persisted scope responsible for the "
        f"empty index; got {census_scope}"
    )
    report = scope_report.build_scope_report(census_scope)
    assert report.scope == bogus
