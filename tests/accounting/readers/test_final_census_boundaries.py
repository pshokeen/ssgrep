"""Census reports real read failures and aggregates identities without exposing content."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ssgrep.accounting.census import render_census_text, run_census
from ssgrep.accounting.census_repos import inventory
from ssgrep.accounting.readers.claude import ClaudeReader
from tests.accounting.readers_fixtures import claude_tool_use, codex_call, write_records


def test_discovered_source_replaced_by_directory_is_partial_and_other_sources_survive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    records = [claude_tool_use("unreadable", "call", "search_sessions", {"query": "private"})]
    broken = write_records(home / ".claude" / "projects" / "broken.jsonl", records)
    good = write_records(
        home / ".claude" / "projects" / "good.jsonl",
        [claude_tool_use("good", "call", "search_sessions", {"query": "visible control"})],
    )
    discover = ClaudeReader.discover

    def race_after_discovery(reader):
        sources = discover(reader)
        assert {source.path for source in sources} == {broken, good}
        broken.unlink()
        broken.mkdir()
        return sources

    with monkeypatch.context() as patch:
        patch.setattr(ClaudeReader, "discover", race_after_discovery)
        partial = run_census(tmp_path, {"HOME": str(home)}).to_dict()
    row = partial["runtimes"]["claude"]
    assert (row["status"], row["sources"], row["unreadable_sources"]) == ("partial", 2, 1)
    assert row["invocations"]["recognized_calls"] == 1
    assert row["evidence_records"] == 1
    broken.rmdir()
    write_records(broken, records)
    recovered = run_census(tmp_path, {"HOME": str(home)}).to_dict()["runtimes"]["claude"]
    assert recovered["status"] == "covered"
    assert recovered["unreadable_sources"] == 0
    assert recovered["invocations"]["recognized_calls"] == 2


def test_census_streaming_usage_and_cwdless_sessions_keep_distinct_denominators(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    root = tmp_path / "repos"
    repo = root / "project"
    (repo / ".git").mkdir(parents=True)
    first = claude_tool_use(
        "streamed",
        "same-call",
        "search_sessions",
        {"query": "private-query-canary"},
        usage={"input_tokens": 10},
        cwd=str(repo),
    )
    second = claude_tool_use(
        "streamed",
        "same-call",
        "search_sessions",
        {"query": "private-query-canary"},
        usage={"input_tokens": 14},
        cwd=str(repo),
    )
    write_records(home / ".claude" / "projects" / "stream.jsonl", [first, second])
    write_records(
        home / ".codex" / "sessions" / "rollout-cwdless.jsonl",
        [codex_call("cwdless", "bash", {"command": "ssgrep show private-reference-canary"})],
    )
    report = run_census(root, {"HOME": str(home)})
    data = report.to_dict()
    claude = data["runtimes"]["claude"]
    assert claude["streamed_usage_messages"] == 1
    assert claude["invocations"]["recognized_calls"] == 1
    assert claude["duplicates"] == 1
    assert claude["conflicts"] == 0
    assert data["association"] == {
        "sessions": 2,
        "in_repository": 1,
        "nested_repository": 0,
        "unknown_root": 0,
        "no_cwd": 1,
        "sessions_by_repository_count": {"project": 1},
        "repositories_omitted": 0,
    }
    rendered = json.dumps(data) + render_census_text(report)
    for private in ("private-query-canary", "private-reference-canary", str(home), str(repo)):
        assert private not in rendered
    assert data["totals"]["recognized_calls"] == 2
    assert data["totals"]["executions_claimed"] is None


def test_repository_file_root_is_not_a_directory_and_does_not_open_content(tmp_path: Path) -> None:
    path = tmp_path / "private.jsonl"
    path.write_text("private content that is not metadata", encoding="utf-8")
    report = run_census(path, {"HOME": str(tmp_path / "empty-home")}).to_dict()
    assert report["repositories"]["root_status"] == "not_a_directory"
    assert report["repositories"]["found"] == 0
    assert report["repositories"]["directories_walked"] == 0
    assert "private content" not in json.dumps(report)
    direct, _ = inventory(tmp_path)
    assert direct.root_status == "ok"
    assert direct.jsonl_files == 1


def test_repository_file_symlinks_are_counted_but_not_inventoried_or_opened(tmp_path: Path) -> None:
    root = tmp_path / "repos"
    repo = root / "project"
    (repo / ".git").mkdir(parents=True)
    outside = tmp_path / "private-outside.jsonl"
    outside.write_text("private-transcript-canary", encoding="utf-8")
    (repo / "linked.jsonl").symlink_to(outside)
    (repo / "dangling.jsonl").symlink_to(tmp_path / "does-not-exist")
    (repo / "actual.jsonl").write_bytes(b"not parsed by inventory\n")
    summary, index = inventory(root)
    assert summary.found == 1
    assert summary.symlinks_skipped == 2
    assert summary.jsonl_files == 1
    matched = index.match(str(repo / "src"))
    assert matched is not None and matched.name == "project"
    assert index.match(str(tmp_path / "project-suffix")) is None
    direct, _ = inventory(tmp_path)
    assert direct.jsonl_files == 2  # the outside target only counts when explicitly in scope
    assert outside.read_text(encoding="utf-8") == "private-transcript-canary"
