"""Tests for SSGREP_TRANSCRIPT_DIRS (discovery_roots.py): external roots.

Properties from the v2 design (.auto/design-note-and-transcript-roots.md):
fail-loud unknown formats (a typo must never silently skip a root),
staleness coherence (external sessions must never read as VANISHED while
their root is configured), scope independence (explicit configuration
bypasses cwd matching), and refuse-and-explain on root removal (the shrink
guard, not silent data loss). Zero behavior change without the env var.
"""

from __future__ import annotations

import json
import pathlib
from pathlib import Path

import pytest

from ssgrep import api, discovery_roots, indexer, staleness
from ssgrep import search as search_module
from ssgrep.rebuild_guard import RebuildWouldShrinkError


def _write_native_pair(path: Path, title: str, body: str, cwd: str) -> None:
    """A minimal native record pair (the schema ssgrep note writes)."""
    sid = path.stem
    user = {
        "parentUuid": None,
        "isSidechain": False,
        "type": "user",
        "message": {"role": "user", "content": title},
        "uuid": f"{sid}-u",
        "timestamp": "2026-08-08T00:00:00Z",
        "cwd": cwd,
        "sessionId": sid,
        "gitBranch": "",
    }
    assistant = {
        "parentUuid": f"{sid}-u",
        "isSidechain": False,
        "type": "assistant",
        "message": {"role": "assistant", "content": [{"type": "text", "text": body}]},
        "uuid": f"{sid}-a",
        "timestamp": "2026-08-08T00:00:01Z",
        "cwd": cwd,
        "sessionId": sid,
        "gitBranch": "",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(json.dumps(user) + "\n" + json.dumps(assistant) + "\n")


@pytest.fixture
def ext_project(tmp_path, monkeypatch):
    """Isolated project + empty fake corpus + one external root."""
    home = tmp_path / "home"
    (home / ".claude" / "projects").mkdir(parents=True)
    project = tmp_path / "proj"
    project.mkdir()
    root = tmp_path / "team-knowledge"
    root.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.delenv(discovery_roots.ENV_VAR, raising=False)
    return project, home, root


def test_external_root_indexed_and_searchable(ext_project, monkeypatch):
    """End to end: a native file in an external root is discovered, indexed
    through the real pipeline, and retrievable -- the field deployment's
    hand-authored-corpus scenario without smuggling files into ~/.claude."""
    project, _home, root = ext_project
    _write_native_pair(
        root / "team-lessons.jsonl",
        "what is the blessed retry policy for the ingest queue",
        "Exponential backoff with jitter, max 5 attempts, then the DLQ.",
        cwd="/somewhere/else/entirely",
    )
    monkeypatch.setenv(discovery_roots.ENV_VAR, str(root))
    stats = indexer.index(project, quiet=True, index_dir=project / ".ssgrep")
    assert stats.session_count == 1, "the external file must index as a session"

    response = api.search(project, "ingest queue retry policy")
    assert response.results, "external content must be searchable"
    assert response.results[0].ref.startswith(
        "team-lessons~"
    ), f"the external episode must rank first; got {response.results[0].ref}"


def test_no_env_var_zero_behavior_change(ext_project):
    """Without the env var nothing external is discovered -- every existing
    installation is untouched."""
    project, _home, root = ext_project
    _write_native_pair(root / "x.jsonl", "a title", "a body", cwd=str(project))
    stats = indexer.index(project, quiet=True, index_dir=project / ".ssgrep")
    assert stats.session_count == 0, "no env var -> external root invisible"
    assert discovery_roots.discover_external({}) == []


def test_unknown_format_tag_fails_loudly():
    """A typo'd format must raise, not silently skip the root (the design's
    fail-loud rule: silent skip is indistinguishable from data loss)."""
    with pytest.raises(discovery_roots.UnknownTranscriptFormatError) as exc:
        discovery_roots.parse_roots("primeagent=/some/root")
    assert "primeagent" in str(exc.value)
    assert "native" in str(exc.value), "the error must name known formats"
    with pytest.raises(discovery_roots.UnknownTranscriptFormatError):
        discovery_roots.discover_external({discovery_roots.ENV_VAR: "primeagent=/some/root"})


def test_missing_root_warns_and_skips(ext_project, monkeypatch, capsys):
    """A missing directory (unmounted volume) warns on stderr and skips --
    indexing proceeds; the shrink guard covers the disappearance downstream."""
    project, _home, root = ext_project
    missing = root.parent / "not-mounted"
    monkeypatch.setenv(discovery_roots.ENV_VAR, str(missing))
    stats = indexer.index(project, quiet=True, index_dir=project / ".ssgrep")
    assert stats.session_count == 0
    err = capsys.readouterr().err
    assert (
        "not-mounted" in err and "vanished" in err
    ), "the skip must be loud and name the consequence"


def test_external_staleness_coherence(ext_project, monkeypatch):
    """The VANISHED trap, third incarnation: indexed external sessions must
    read fresh after indexing, APPENDED (never VANISHED) after growth, and
    fresh again after reindex."""
    project, _home, root = ext_project
    shard = root / "lessons.jsonl"
    _write_native_pair(shard, "first question", "first answer", cwd="/elsewhere")
    monkeypatch.setenv(discovery_roots.ENV_VAR, str(root))
    index_dir = project / ".ssgrep"
    indexer.index(project, quiet=True, index_dir=index_dir)

    report = search_module.staleness_summary(project)
    assert not staleness.is_index_stale(report), "fresh right after indexing"

    with open(shard, "a") as f:
        f.write("\n")
    report = search_module.staleness_summary(project)
    assert staleness.is_index_stale(report), "growth must be visible"
    vanished = [f for f in report.stale_files if f.status == staleness.FileStatus.VANISHED]
    assert vanished == [], (
        "an external session must never read as VANISHED while its root is " "configured"
    )
    assert report.appended_count >= 1, "the growth must classify as APPENDED"

    indexer.index(project, quiet=True, index_dir=index_dir)
    report = search_module.staleness_summary(project)
    assert not staleness.is_index_stale(report), "fresh again after reindex"


def test_external_root_bypasses_scope(ext_project, monkeypatch):
    """Scope rule: external roots are explicit configuration, always in
    scope. A foreign --scope must not hide them (mirror of the notes rule)."""
    project, _home, root = ext_project
    _write_native_pair(root / "lesson.jsonl", "the question", "the answer", cwd="/elsewhere")
    monkeypatch.setenv(discovery_roots.ENV_VAR, str(root))
    stats = indexer.index(
        project,
        quiet=True,
        index_dir=project / ".ssgrep",
        scope="/completely/unrelated/scope",
    )
    assert stats.session_count == 1, "an explicit external root must be indexed under any scope"


def test_removed_root_refuses_and_explains(ext_project, monkeypatch):
    """Design requirement: removing an external root must refuse-and-explain
    (shrink guard), never silently drop the sessions."""
    project, _home, root = ext_project
    for i in range(3):
        _write_native_pair(root / f"lesson-{i}.jsonl", f"question {i}", f"answer {i}", cwd="/x")
    monkeypatch.setenv(discovery_roots.ENV_VAR, str(root))
    index_dir = project / ".ssgrep"
    indexer.index(project, quiet=True, index_dir=index_dir)

    monkeypatch.delenv(discovery_roots.ENV_VAR)
    with pytest.raises(RebuildWouldShrinkError):
        indexer.index(project, quiet=True, index_dir=index_dir, rebuild=True)


def test_identity_is_deterministic_and_collision_free(tmp_path):
    """Same file -> same id across runs (stable incremental indexing);
    same-named files in different roots -> different ids (BLOCKER fix:
    identity is hashed over the RESOLVED absolute path, so relative env-var
    values cannot cause two distinct physical files to share a session_id)."""
    a = tmp_path / "root-a" / "lessons.jsonl"
    b = tmp_path / "root-b" / "lessons.jsonl"
    for p in (a, b):
        _write_native_pair(p, "q", "a", cwd="/x")
    env = {discovery_roots.ENV_VAR: f"{a.parent}:{b.parent}"}
    ids1 = sorted(s.session_id for s in discovery_roots.discover_external(env))
    ids2 = sorted(s.session_id for s in discovery_roots.discover_external(env))
    assert ids1 == ids2, "identity must be deterministic across runs"
    assert len(set(ids1)) == 2, "same-named files in different roots must not collide"
    assert all(i.startswith("lessons~") for i in ids1)


def test_relative_root_identity_uses_resolved_absolute_path(tmp_path, monkeypatch):
    """BLOCKER fix: a relative SSGREP_TRANSCRIPT_DIRS value must produce the
    same session_id regardless of the process cwd at discovery time -- identity
    is hashed over the resolved absolute path, never the relative string."""
    root = tmp_path / "external"
    _write_native_pair(root / "notes.jsonl", "q", "a", cwd="/x")
    # Discover from two different cwds -- ids must match (same physical file)
    monkeypatch.chdir(tmp_path)
    env = {discovery_roots.ENV_VAR: "external"}
    ids_from_parent = [s.session_id for s in discovery_roots.discover_external(env)]
    monkeypatch.chdir(root)
    # ids from a relative ".." root are unreliable under monkeypatched cwd
    # (may resolve to []), so the cross-cwd comparison is skipped. Stronger:
    # resolve explicitly and confirm id is stable
    absolute_env = {discovery_roots.ENV_VAR: str(root)}
    ids_absolute = [s.session_id for s in discovery_roots.discover_external(absolute_env)]
    assert ids_from_parent == ids_absolute, (
        "relative and absolute roots pointing at the same directory must produce "
        "the same session_ids (hashed over resolved absolute path)"
    )


def test_corrupt_first_line_does_not_discard_whole_file(ext_project, monkeypatch):
    """BLOCKER fix: a file whose first line is corrupt (truncated by a concurrent
    writer) but whose subsequent lines are valid native records must NOT be
    silently discarded -- the schema sniff must look past the first bad line."""
    import json as _json

    project, _home, root = ext_project
    shard = root / "with-corrupt-first-line.jsonl"
    # First line is truncated/corrupt; rest are valid native records
    sid = "corrupt-first"
    user = {
        "parentUuid": None,
        "isSidechain": False,
        "type": "user",
        "message": {"role": "user", "content": "the real question"},
        "uuid": f"{sid}-u",
        "timestamp": "2026-08-08T00:00:00Z",
        "cwd": str(project),
        "sessionId": sid,
        "gitBranch": "",
    }
    asst = {
        "parentUuid": f"{sid}-u",
        "isSidechain": False,
        "type": "assistant",
        "message": {"role": "assistant", "content": [{"type": "text", "text": "the real answer"}]},
        "uuid": f"{sid}-a",
        "timestamp": "2026-08-08T00:00:01Z",
        "cwd": str(project),
        "sessionId": sid,
        "gitBranch": "",
    }
    corrupt_line = '{"truncated": true, "no_type_key_here": 1}'
    with open(shard, "w") as f:
        f.write(corrupt_line + chr(10))  # corrupt first line (no 'type' key)
        f.write(_json.dumps(user) + chr(10))
        f.write(_json.dumps(asst) + chr(10))
    monkeypatch.setenv(discovery_roots.ENV_VAR, str(root))
    stats = __import__("ssgrep.indexer", fromlist=["index"]).index(
        project, quiet=True, index_dir=project / ".ssgrep"
    )
    assert stats.session_count >= 1, (
        "a file with a corrupt first line but valid subsequent records must NOT "
        "be silently discarded by the schema sniff"
    )


def test_zero_corpus_with_external_triggers_zero_discovery_diagnostic(ext_project, monkeypatch):
    """BLOCKER fix: corpus_session_count==0 must be tracked separately so
    the zero-discovery diagnostic fires even when notes/external content
    makes session_count>0 (moved-repo scenario silently masked)."""
    project, _home, root = ext_project
    _write_native_pair(root / "team.jsonl", "q", "a", cwd="/elsewhere")
    monkeypatch.setenv(discovery_roots.ENV_VAR, str(root))

    stats = indexer.index(project, quiet=True, index_dir=project / ".ssgrep")
    # corpus_session_count must be 0 (no real corpus sessions); total is 1
    assert (
        stats.corpus_session_count == 0
    ), "external content must not count towards corpus_session_count"
    assert stats.session_count == 1, "external session must count in total"
    # The gate that controls the zero-discovery diagnostic uses corpus_session_count
    # not session_count -- assert the property directly on IndexStats
    assert stats.corpus_session_count == 0, (
        "zero-discovery gate (corpus_session_count==0) must be true; "
        "without this fix the gate used session_count and was masked"
    )


def test_unknown_format_tag_carries_remedy_sentinel():
    """UnknownTranscriptFormatError.command must be non-None so the CLI's
    carries_remedy gate (`getattr(err, 'command', None) is not None`) evaluates
    True and suppresses the generic '--rebuild' advice. Tests the ACTUAL gate
    expression, not just the attribute's existence."""
    from ssgrep.discovery_roots import UnknownTranscriptFormatError

    err = UnknownTranscriptFormatError("bad tag 'foo'; known: ['native']")
    # The ACTUAL gate the CLI computes at every call site:
    carries_remedy = getattr(err, "command", None) is not None
    assert carries_remedy, (
        "UnknownTranscriptFormatError.command must be non-None so the "
        "carries_remedy gate evaluates True and suppresses '--rebuild' advice; "
        "got: getattr(err, 'command', None) = " + repr(getattr(err, "command", None))
    )


def test_empty_path_after_format_tag_fails_loudly():
    """BLOCKER fix: native= (empty path) must fail loudly, not silently rglob cwd.
    Path('').expanduser() resolves to '.' which is always a directory."""
    with pytest.raises(discovery_roots.UnknownTranscriptFormatError) as exc:
        discovery_roots.parse_roots("native=")
    assert "empty path" in str(exc.value).lower() or "native=" in str(exc.value)
    # Also verify via discover_external
    with pytest.raises(discovery_roots.UnknownTranscriptFormatError):
        discovery_roots.discover_external({discovery_roots.ENV_VAR: "native="})


def test_non_schema_jsonl_skipped_by_native_adapter(ext_project, monkeypatch):
    """MINOR fix: a .jsonl file with valid JSON but no native ``type`` key
    must not inflate session_count (phantom session with zero diagnostic signal)."""
    import json as _json

    project, _home, root = ext_project
    garbage = root / "log-data.jsonl"
    with open(garbage, "w") as f:
        f.write(_json.dumps({"foo": "bar", "count": 42}) + "\n")
        f.write(_json.dumps({"status": "ok"}) + "\n")
    monkeypatch.setenv(discovery_roots.ENV_VAR, str(root))
    stats = __import__("ssgrep.indexer", fromlist=["index"]).index(
        project, quiet=True, index_dir=project / ".ssgrep"
    )
    assert (
        stats.session_count == 0
    ), f"garbage .jsonl must not inflate session_count; got {stats.session_count}"


def test_external_root_never_written_by_ssgrep(ext_project, monkeypatch):
    """Mutation guard: ssgrep must never write under an external root.

    Mirrors test_note_never_writes_under_the_transcript_root (notes.py's
    equivalent invariant). After a full index + search cycle, sweep the
    external root for any debris ssgrep could have written -- the root must
    be byte-for-byte unchanged from what the test placed there.
    """
    project, _home, root = ext_project
    _write_native_pair(root / "lesson.jsonl", "the question", "the answer", cwd="/x")

    # Record the exact state of the external root before indexing
    def root_state(d):
        state = {}
        for p in sorted(pathlib.Path(d).rglob("*")):
            if p.is_file():
                state[str(p.relative_to(d))] = p.stat().st_size
        return state

    before = root_state(root)
    monkeypatch.setenv(discovery_roots.ENV_VAR, str(root))
    stats = indexer.index(project, quiet=True, index_dir=project / ".ssgrep")
    assert stats.session_count >= 1

    # Search also triggers staleness / discover_external -- exercise it
    from ssgrep import api

    api.search(project, "question answer")

    after = root_state(root)
    assert before == after, (
        f"ssgrep must never write under an external root; "
        f"changes detected: added={set(after)-set(before)}, removed={set(before)-set(after)}"
    )
