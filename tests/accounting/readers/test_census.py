"""4.7/4.8 census over a synthetic multi-repository home with all six runtime formats."""

from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path

import pytest

from ssgrep.accounting.census import CensusReport, render_census_text, run_census
from ssgrep.accounting.census_repos import inventory
from ssgrep.accounting.readers.claude import ClaudeReader
from tests.accounting.readers_fixtures import (
    PWNED,
    SyntheticHome,
    build_home,
    claude_tool_use,
    codex_call,
    write_records,
)


@pytest.fixture
def home(tmp_path: Path) -> SyntheticHome:
    return build_home(tmp_path / "home")


def tree_digest(root: Path) -> dict[str, str]:
    """Hash every file (and symlink target) under root; used to prove read-only behavior."""
    seen: dict[str, str] = {}
    for base, dirs, files in os.walk(root, followlinks=False):
        for name in sorted(dirs + files):
            path = Path(base, name)
            if path.is_symlink():
                seen[str(path)] = "->" + os.readlink(path)
            elif path.is_file():
                seen[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            else:
                seen[str(path)] = "dir"
    return seen


def scan(home: SyntheticHome, root: Path | None = None) -> CensusReport:
    return run_census(root if root is not None else home.code, env=home.env)


def runtime(report: CensusReport, name: str) -> dict:
    return report.to_dict()["runtimes"][name]


def test_every_runtime_is_covered_with_counts(home: SyntheticHome) -> None:
    data = scan(home).to_dict()
    assert sorted(data["runtimes"]) == ["claude", "codex", "omp", "opencode", "pi", "prime-agent"]
    for name, row in data["runtimes"].items():
        assert row["sources"] >= 1, name
        assert row["evidence_records"] > 0, name
        assert row["status"] == "covered", name
    assert runtime(scan(home), "claude")["sources"] == 4  # 3 copies/forks + 1 child
    assert runtime(scan(home), "prime-agent")["sources"] == 3  # main, artifact, lineage
    assert runtime(scan(home), "omp")["sources"] == 2
    assert runtime(scan(home), "opencode")["sources"] == 1
    assert data["coverage"] == {"runtimes_total": 6, "runtimes_covered": 6}


def test_copies_dedupe_and_conflicts_are_reported_not_double_counted(home: SyntheticHome) -> None:
    claude = runtime(scan(home), "claude")
    assert claude["duplicates"] > 0 and claude["conflicts"] == 1  # the CHANGED fork
    assert claude["conflicts_by_kind"] == {"tool_call": 1}
    # the identical copies and the forked copy repeat the same call ids: each is counted once
    # (mcp search, loop search, hostile-text search) plus the subagent child's show
    assert claude["invocations"]["by_operation"] == {"search": 3, "show": 1}
    assert claude["invocations"]["recognized_calls"] == 4


def test_recognized_invocations_exclude_docs_mentions_and_hostile_text(home: SyntheticHome) -> None:
    report = scan(home).to_dict()
    omp = report["runtimes"]["omp"]["invocations"]
    # omp: write xd://mcp__renamed... recognized under an unfamiliar prefix; the read is a mention
    assert omp["by_route"] == {"omp_xd_write": 1, "shell_cli": 1}
    assert omp["by_class"]["mention"] == 1
    claude = report["runtimes"]["claude"]["invocations"]
    assert claude["by_class"]["mention"] == 1  # echo "ssgrep search x"
    assert claude["by_route"]["mcp_name"] == 2  # renamed prefixes: acme_recall, other
    codex = report["runtimes"]["codex"]["invocations"]
    assert codex["by_route"] == {"mcp_namespace": 1, "shell_cli": 1}
    opencode = report["runtimes"]["opencode"]["invocations"]
    assert opencode["by_route"] == {"shell_cli": 1, "tool_name": 1}  # event copies ignored


def test_shell_loop_is_one_invocation_with_unknown_cardinality(home: SyntheticHome) -> None:
    totals = scan(home).to_dict()["totals"]
    assert totals["loop_or_unknown_cardinality"] >= 1
    claude = runtime(scan(home), "claude")["invocations"]
    assert claude["unknown_cardinality"] == 1  # only the loop; plain text invocations are single
    assert totals["executions_claimed"] is None  # text alone never yields an execution count
    assert "transcript_only" in totals["by_class"]


def test_hostile_transcript_strings_are_never_executed_or_fetched(
    home: SyntheticHome, monkeypatch: pytest.MonkeyPatch
) -> None:
    import socket
    import subprocess

    def refuse(*args, **kwargs):
        raise AssertionError("census must never spawn processes or open sockets")

    for target in ("run", "Popen", "check_output", "call", "check_call"):
        monkeypatch.setattr(subprocess, target, refuse)
    monkeypatch.setattr(os, "system", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    marker = Path("/tmp") / PWNED
    marker.unlink(missing_ok=True)
    report = scan(home)
    assert not marker.exists()
    assert report.to_dict()["runtimes"]["claude"]["invocations"]["recognized_calls"] > 0


def test_repository_inventory_nested_worktree_and_symlinks(home: SyntheticHome) -> None:
    repos = scan(home).to_dict()["repositories"]
    assert repos["found"] == 4  # alpha, nested, beta, gamma (.git file); not node_modules
    assert repos["nested"] == 1 and repos["worktree_files"] == 1
    assert repos["symlinks_skipped"] == 1  # `linked` is counted, never followed
    assert repos["jsonl_files"] == 3  # fixtures/copy, snapshots/old, data.jsonl
    assert repos["jsonl_in_fixtures"] == 2  # fixtures + snapshots dirs
    assert repos["pruned_dirs"] >= 1


def test_symlinked_repo_is_not_scanned_through(home: SyntheticHome, tmp_path: Path) -> None:
    # make the outside repo hold a transcript: if the link were followed it would be counted
    outside = home.paths["outside"]
    (outside / "leak.jsonl").write_bytes(b'{"type":"x"}\n')
    before = scan(home).to_dict()["repositories"]
    assert before["jsonl_files"] == 3 and before["found"] == 4
    # scanning the link target directly (explicit root) does count it
    direct = scan(home, outside).to_dict()["repositories"]
    assert direct["found"] == 1 and direct["jsonl_files"] == 1


def test_sessions_associate_with_repositories_by_longest_path(home: SyntheticHome) -> None:
    report = scan(home).to_dict()
    assoc = report["association"]
    assert assoc["in_repository"] >= 1 and assoc["nested_repository"] >= 1
    assert assoc["unknown_root"] >= 1  # retired project and /somewhere/else are not guessed
    by_repo = assoc["sessions_by_repository_count"]
    assert by_repo["alpha/vendor/nested"] >= 1  # longest match beats the enclosing alpha repo
    assert "alpha" in by_repo
    assert all(isinstance(value, int) for value in by_repo.values())


def test_output_is_aggregate_only_and_integer_only(home: SyntheticHome) -> None:
    report = scan(home)
    blob = json.dumps(report.to_dict())
    text = render_census_text(report)
    forbidden = [
        "alpha deploy",
        "find the deploy notes",
        "evil.invalid",
        "claude:c1",
        "retired-q",
        "toolu",
        "synthetic",
        str(home.root),
    ]
    for needle in forbidden:
        assert needle not in blob, needle
        assert needle not in text, needle

    def check(value) -> None:
        assert not isinstance(value, float)
        if isinstance(value, dict):
            [check(v) for v in value.values()]
        elif isinstance(value, list):
            [check(v) for v in value]

    check(report.to_dict())
    assert report.to_dict()["scope"] == {"root_is_cwd": False, "content_included": False}


def test_census_is_read_only_for_sources_and_the_scanned_tree(home: SyntheticHome) -> None:
    before = tree_digest(home.root)
    # the WAL side files are created by SQLite for readers: compare everything but them
    scan(home)
    scan(home)
    after = tree_digest(home.root)
    created = set(after) - set(before)
    allowed = {f"{home.paths['opencode']}-shm", f"{home.paths['opencode']}-wal"}
    assert created <= allowed, created
    for path, digest in before.items():
        assert after[path] == digest, path  # no file was modified, removed, or retargeted
    db = home.paths["opencode"]
    assert db.stat().st_size > 0 and not Path(f"{db}-journal").exists()


def test_default_root_is_cwd_and_census_never_writes_accounting_state(
    home: SyntheticHome, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(home.code / "beta")
    data_dir = tmp_path / "data"
    report = run_census(None, env=home.env)
    assert report.to_dict()["scope"]["root_is_cwd"] is True
    assert report.to_dict()["repositories"]["found"] == 1
    assert not data_dir.exists()  # no accounting directory is created by a census
    assert not (home.code / "beta" / ".ssgrep").exists()


def test_empty_result_is_not_proof_of_no_usage(tmp_path: Path) -> None:
    empty = tmp_path / "empty-home"
    (empty / "code").mkdir(parents=True)
    env = {"HOME": str(empty)}
    report = run_census(empty / "code", env=env)
    data = report.to_dict()
    assert all(row["status"] == "no_source" for row in data["runtimes"].values())
    assert data["totals"]["recognized_calls"] == 0
    assert data["notes"]["empty_is_not_absence"] is True
    assert "not evidence" in render_census_text(report)


def test_missing_root_and_unreadable_dirs_are_reported_not_fatal(
    home: SyntheticHome, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = run_census(tmp_path / "does-not-exist", env=home.env)
    assert report.to_dict()["repositories"]["root_status"] == "missing"
    locked = home.code / "locked"
    (locked / ".git").mkdir(parents=True)
    (locked / "evidence.jsonl").write_bytes(b'{"plain":"data"}\n')
    readable = scan(home).to_dict()["repositories"]
    assert readable["found"] == 5 and readable["jsonl_files"] == 4
    assert readable["unreadable_dirs"] == 0

    # chmod(0) is still readable by root. Inject the actual OS failure at
    # scandir so the real os.walk error callback and recovery run on every host.
    scandir = os.scandir
    denied: list[Path] = []

    def deny_locked(path):
        if Path(path) == locked:
            denied.append(Path(path))
            raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), str(path))
        return scandir(path)

    monkeypatch.setattr(os, "scandir", deny_locked)
    repositories = scan(home).to_dict()["repositories"]
    assert denied == [locked]
    assert repositories["unreadable_dirs"] == 1
    assert repositories["found"] == 4 and repositories["jsonl_files"] == 3


def test_oversized_record_is_a_coverage_gap_in_the_census(home: SyntheticHome) -> None:
    big = home.paths["pi"] / "-work" / "big.jsonl"
    with big.open("wb") as handle:
        handle.write(b'{"type":"session","id":"big","cwd":"/x"}\n')
        handle.write(b'{"pad":"' + b"x" * (8 * 1024 * 1024 + 10) + b'"}\n')
        handle.write(
            b'{"type":"message","id":"e","message":{"role":"assistant","content":'
            b'[{"type":"toolCall","id":"c","name":"bash",'
            b'"arguments":{"command":"ssgrep show r"}}]}}\n'
        )
    pi = runtime(scan(home), "pi")
    assert pi["gaps"] == {"oversized_record": 1}
    assert pi["invocations"]["recognized_calls"] == 2  # p1's search + the call after the gap


def test_text_output_lists_each_runtime_and_totals(home: SyntheticHome) -> None:
    text = render_census_text(scan(home))
    for name in ("claude", "codex", "omp", "opencode", "pi", "prime-agent"):
        assert name in text
    assert "repositories" in text and "loop" in text.lower()


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
