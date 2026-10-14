"""Recognition requires structured routes or resolvable bounded shell command positions."""

from __future__ import annotations

import shlex
from pathlib import Path

import pytest

from ssgrep.accounting.contracts import InvocationClass
from ssgrep.accounting.readers.classify import UNKNOWN, classify_evidence
from ssgrep.accounting.readers.omp import OmpReader
from ssgrep.accounting.readers.shell_scan import MAX_DEPTH
from tests.accounting.readers_fixtures import pi_header, pi_tool_call, write_records


def classify_native(tmp_path: Path, name: str, arguments: object):
    record = pi_tool_call("entry", "call", name, {})
    record["message"]["content"][0]["arguments"] = arguments
    path = write_records(
        tmp_path / "session.jsonl",
        [pi_header("session"), record],
    )
    reader = OmpReader({"HOME": str(tmp_path)})
    call = next(
        row for row in reader.read(reader.source_for(path)).evidence if row.kind == "tool_call"
    )
    return classify_evidence(call)


@pytest.mark.parametrize("where", [None, "runtime = 'codex'"])
def test_optional_where_schema_accepts_null_and_predicate(tmp_path: Path, where: object) -> None:
    result = classify_native(
        tmp_path, "mcp__renamed__search_sessions", {"query": "q", "where": where}
    )
    assert result.invocation_class is InvocationClass.TRANSCRIPT_ONLY
    assert result.invocation is not None
    assert result.invocation.arguments == {"query": "q", "where": where}
    assert result.invocation.operations == ("search",)


@pytest.mark.parametrize("where", [False, 7, [], {"runtime": "codex"}])
def test_nonpredicate_where_never_becomes_a_recognized_mcp_call(
    tmp_path: Path, where: object
) -> None:
    result = classify_native(
        tmp_path, "mcp__renamed__search_sessions", {"query": "q", "where": where}
    )
    assert result.invocation is None
    assert result.reason == "schema_mismatch"
    control = classify_native(
        tmp_path, "mcp__renamed__search_sessions", {"query": "q", "where": None}
    )
    assert control.invocation is not None


@pytest.mark.parametrize("content", ["", " \n\t"])
def test_empty_xd_content_is_empty_status_request_not_a_search(
    tmp_path: Path, content: str
) -> None:
    status = classify_native(
        tmp_path, "write", {"path": "xd://mcp__renamed_index_status", "content": content}
    )
    assert status.invocation is not None
    assert status.invocation.operations == ("index_status",)
    assert status.invocation.arguments == {}
    missing_query = classify_native(
        tmp_path, "write", {"path": "xd://mcp__renamed_search_sessions", "content": content}
    )
    assert missing_query.invocation is None
    assert missing_query.reason == "schema_mismatch"


@pytest.mark.parametrize(
    "name,arguments,reason",
    [
        ("write", [], "unknown_tool"),
        (
            "write",
            {"path": "xd://mcp__renamed_unknown", "content": "ssgrep search text"},
            "unknown_tool",
        ),
        ("bash", [], "no_command"),
        ("bash", {"cwd": "/not-a-command"}, "no_command"),
    ],
)
def test_unresolved_tool_shapes_do_not_invent_calls(
    tmp_path: Path, name: str, arguments: object, reason: str
) -> None:
    result = classify_native(tmp_path, name, arguments)
    assert result.invocation_class is InvocationClass.UNCLASSIFIED
    assert result.invocation is None
    assert result.reason == reason
    control = classify_native(tmp_path, "bash", {"command": "ssgrep search positive"})
    assert control.invocation is not None
    assert control.invocation.argv == (("search", "positive"),)


def test_escaped_quotes_stay_query_arguments_not_additional_commands(tmp_path: Path) -> None:
    result = classify_native(
        tmp_path,
        "bash",
        {"command": 'ssgrep search "first \\"quoted\\" line"\nssgrep show ref'},
    )
    assert result.invocation is not None
    assert result.invocation.argv == (("search", 'first "quoted" line'), ("show", "ref"))
    assert result.invocation.cardinality == UNKNOWN
    assert result.invocation.text_occurrences == 2


def test_nested_shell_scripts_stop_at_depth_bound_without_guessing_inner_execution(
    tmp_path: Path,
) -> None:
    script = "ssgrep search inner"
    for _ in range(MAX_DEPTH):
        script = "bash -c " + shlex.quote(script)
    boundary = classify_native(tmp_path, "bash", {"command": script})
    assert boundary.invocation is not None
    assert boundary.invocation.argv == (("search", "inner"),)
    assert boundary.invocation.cardinality == UNKNOWN
    too_deep = classify_native(tmp_path, "bash", {"command": "bash -c " + shlex.quote(script)})
    assert too_deep.invocation is None
    assert too_deep.invocation_class is InvocationClass.MENTION
    unresolved = classify_native(tmp_path, "bash", {"command": 'bash -c "$UNRESOLVED_SCRIPT"'})
    assert unresolved.invocation_class is InvocationClass.UNCLASSIFIED
    assert unresolved.invocation is None


def test_argv_find_terminator_and_uv_subcommand_distinguish_execution_from_arguments(
    tmp_path: Path,
) -> None:
    found = classify_native(
        tmp_path,
        "shell",
        {"command": ["find", ".", "-exec", "ssgrep", "show", "{}", "+", "ignored"]},
    )
    assert found.invocation is not None
    assert found.invocation.argv == (("show", "{}"),)
    assert found.invocation.cardinality == UNKNOWN
    not_run = classify_native(tmp_path, "shell", {"command": "uv pip install ssgrep search"})
    assert not_run.invocation is None
    actual = classify_native(tmp_path, "shell", {"command": "uv run ssgrep search q"})
    assert actual.invocation is not None
    assert actual.invocation.argv == (("search", "q"),)
