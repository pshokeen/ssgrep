"""Unit tests for transcript discovery and cwd indexing."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ssgrep.sessions import discovery


def test_extract_cwds_from_line_handles_all_input_shapes():
    assert discovery._extract_cwds_from_line("  ") == set()
    assert discovery._extract_cwds_from_line("not json") == set()
    assert discovery._extract_cwds_from_line("[]") == set()
    assert discovery._extract_cwds_from_line('{"cwd": 3}') == set()
    assert discovery._extract_cwds_from_line('{"cwd": "bad\\npath"}') == set()
    assert discovery._extract_cwds_from_line('{"cwd": "/work/tree"}') == {"/work/tree"}


def test_exclusion_is_position_aware_and_outside_paths_are_kept(tmp_path: Path):
    base = tmp_path / "projects"
    project = base / "encoded-project"
    session = project / "session"
    assert discovery._should_exclude_path(project / "memory" / "m.jsonl", base)
    assert discovery._should_exclude_path(session / "tool-results" / "t.jsonl", base)
    assert discovery._should_exclude_path(session / "workflows" / "w.jsonl", base)
    assert not discovery._should_exclude_path(project / "main.jsonl", base)
    assert not discovery._should_exclude_path(
        session / "subagents" / "workflows" / "agent.jsonl", base
    )
    assert not discovery._should_exclude_path(tmp_path / "elsewhere.jsonl", base)


def test_iter_transcript_files_filters_sidecars_and_non_jsonl(tmp_path: Path):
    base = tmp_path / "projects"
    keep = base / "project" / "main.jsonl"
    nested = base / "project" / "session" / "subagents" / "agent.jsonl"
    excluded = base / "project" / "memory" / "cache.jsonl"
    for path in (keep, nested, excluded):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    assert set(discovery.iter_transcript_files(base)) == {keep, nested}

    # rglob normally guarantees this cannot happen; a synthetic iterator pins
    # the defensive suffix guard without relying on filesystem glob behavior.
    fake_root = MagicMock()
    fake_root.rglob.return_value = [Path("odd.json"), Path("fine.jsonl")]
    assert list(discovery.iter_transcript_files(fake_root)) == [Path("fine.jsonl")]


def test_scan_cwds_reads_all_valid_lines(tmp_path: Path):
    shard = tmp_path / "shard.jsonl"
    shard.write_text("\n".join(['{"cwd": "/a"}', "broken", '{"cwd": "/b"}', '{"cwd": "/a"}']))
    assert discovery._scan_cwds(shard) == {"/a", "/b"}


class _Repository:
    def __init__(self, *, exists: bool, rows: list[dict] | None = None):
        self._exists = exists
        self._rows = rows or []
        self.rows_args = None

    def exists(self):
        return self._exists

    def rows(self, *args, **kwargs):
        self.rows_args = (args, kwargs)
        return self._rows


def test_build_cwd_index_reuses_cache_and_rescans_changed_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    unchanged = tmp_path / "same.jsonl"
    changed = tmp_path / "changed.jsonl"
    empty_cached = tmp_path / "empty.jsonl"
    for path, text in ((unchanged, "x"), (changed, "xx"), (empty_cached, "y")):
        path.write_text(text)
    same_stat = unchanged.stat()
    empty_stat = empty_cached.stat()
    repository = _Repository(
        exists=True,
        rows=[
            {
                "path": str(unchanged),
                "size": same_stat.st_size,
                "mtime": same_stat.st_mtime,
                "cwds": "/cached/a\n/cached/b",
            },
            {
                "path": str(changed),
                "size": 999,
                "mtime": changed.stat().st_mtime,
                "cwds": "/stale",
            },
            {
                "path": str(empty_cached),
                "size": empty_stat.st_size,
                "mtime": empty_stat.st_mtime,
                "cwds": None,
            },
        ],
    )
    monkeypatch.setattr(discovery, "LanceStore", lambda: repository)
    monkeypatch.setattr(
        discovery, "iter_transcript_files", lambda _root: iter([unchanged, changed, empty_cached])
    )
    scans: list[Path] = []

    def scan(path: Path):
        scans.append(path)
        return {"/fresh"}

    monkeypatch.setattr(discovery, "_scan_cwds", scan)
    result = discovery._build_cwd_index(tmp_path)
    assert result == {
        str(unchanged): {"/cached/a", "/cached/b"},
        str(changed): {"/fresh"},
        str(empty_cached): set(),
    }
    assert scans == [changed]
    assert repository.rows_args == ((discovery.CWD_CACHE_TABLE,), {"limit": 1_000_000})


def test_build_cwd_index_tolerates_stat_and_scan_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    stat_error = MagicMock()
    stat_error.stat.side_effect = OSError("gone")
    stat_error.__str__.return_value = "gone"
    scan_error = tmp_path / "unreadable.jsonl"
    scan_error.write_text("x")
    good = tmp_path / "good.jsonl"
    good.write_text("x")
    repository = _Repository(exists=False)
    monkeypatch.setattr(discovery, "LanceStore", lambda: repository)
    monkeypatch.setattr(
        discovery, "iter_transcript_files", lambda _root: iter([stat_error, scan_error, good])
    )

    def scan(path: Path):
        if path == scan_error:
            raise OSError("unreadable")
        return {"/good"}

    monkeypatch.setattr(discovery, "_scan_cwds", scan)
    assert discovery._build_cwd_index(tmp_path) == {str(good): {"/good"}}


def test_cwd_index_for_builds_once_per_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    built: list[Path] = []

    def build(path: Path):
        built.append(path)
        return {"file": {"cwd"}}

    monkeypatch.setattr(discovery, "_build_cwd_index", build)
    first = discovery.cwd_index_for(tmp_path)
    second = discovery.cwd_index_for(tmp_path)
    assert first is second
    assert built == [tmp_path]


@pytest.mark.parametrize(
    ("cwd", "scope", "expected"),
    [
        ("/work/project", "/work/project", True),
        ("/work/project/src", "/work/project", True),
        ("/work/project-other", "/work/project", False),
    ],
)
def test_scope_matches_cwd(cwd: str, scope: str, expected: bool):
    assert discovery._scope_matches_cwd(cwd, scope) is expected


def test_file_scope_matching_caches_fallback_and_handles_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    shard = tmp_path / "s.jsonl"
    shard.write_text("x")
    index: dict[str, set[str]] = {}
    monkeypatch.setattr(discovery, "_scan_cwds", lambda _p: {"/work/project"})
    assert discovery._file_matches_scope(shard, "/work", index)
    assert index[str(shard)] == {"/work/project"}

    # A populated cache can miss without rescanning.
    assert not discovery._file_matches_scope(shard, "/elsewhere", index)

    empty = tmp_path / "empty.jsonl"
    monkeypatch.setattr(discovery, "_scan_cwds", lambda _p: set())
    assert not discovery._file_matches_scope(empty, "/work", index)

    failed = tmp_path / "failed.jsonl"
    monkeypatch.setattr(discovery, "_scan_cwds", MagicMock(side_effect=OSError("cannot read")))
    assert not discovery._file_matches_scope(failed, "/work", index)


def test_classify_session_path_shapes(tmp_path: Path):
    base = tmp_path / "projects"
    assert discovery.classify_session(tmp_path / "outside.jsonl", base) == (True, None, None)
    assert discovery.classify_session(base / "project" / "main.jsonl", base) == (
        True,
        None,
        None,
    )
    subagent = base / "project" / "parent" / "subagents" / "agent.jsonl"
    assert discovery.classify_session(subagent, base) == (
        False,
        "parent",
        "subagents-agent",
    )
    workflow = base / "project" / "parent" / "subagents" / "workflows" / "wf" / "agent.jsonl"
    assert discovery.classify_session(workflow, base) == (
        False,
        "parent",
        "subagents-workflows-wf-agent",
    )
    assert discovery.classify_session(base / "project" / "misc" / "other.jsonl", base) == (
        True,
        None,
        None,
    )


def test_discover_sessions_missing_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(discovery.paths, "resolve_claude_dir", lambda: tmp_path / "missing")
    assert discovery.discover_sessions() == []


def test_discover_sessions_builds_main_subagent_and_fallback_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config = tmp_path / "claude"
    base = config / "projects"
    base.mkdir(parents=True)
    main = base / "encoded" / "main.jsonl"
    agent = base / "encoded" / "parent" / "subagents" / "agent-x.jsonl"
    outside = tmp_path / "outside.jsonl"
    for path in (main, agent, outside):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"cwd": "/scope", "type": "user"}) + "\n")
    index = {
        str(main): {"/z", "/a"},
        str(agent): {"/scope"},
        str(outside): set(),
    }
    monkeypatch.setattr(discovery.paths, "resolve_claude_dir", lambda: config)
    monkeypatch.setattr(discovery, "cwd_index_for", lambda _root: index)
    monkeypatch.setattr(
        discovery, "iter_transcript_files", lambda _root: iter([main, agent, outside])
    )

    sessions = discovery.discover_sessions()
    assert [s.session_id for s in sessions] == [
        "main",
        "parent:agent:subagents-agent-x",
        "outside",
    ]
    assert sessions[0].project_paths == ("/a", "/z")
    assert sessions[0].source_project == "encoded"
    assert not sessions[1].is_main
    assert sessions[1].parent_session_id == "parent"
    assert sessions[2].source_project == tmp_path.name

    assert [s.session_id for s in discovery.discover_sessions(no_subagents=True)] == [
        "main",
        "outside",
    ]


def test_discover_sessions_applies_scope_filter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = tmp_path / "claude"
    base = config / "projects"
    keep = base / "project" / "keep.jsonl"
    skip = base / "project" / "skip.jsonl"
    for path in (keep, skip):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    monkeypatch.setattr(discovery.paths, "resolve_claude_dir", lambda: config)
    monkeypatch.setattr(discovery, "cwd_index_for", lambda _root: {})
    monkeypatch.setattr(discovery, "iter_transcript_files", lambda _root: iter([keep, skip]))
    monkeypatch.setattr(discovery, "_file_matches_scope", lambda path, _scope, _index: path == keep)
    assert [s.session_id for s in discovery.discover_sessions(scope="/wanted")] == ["keep"]
