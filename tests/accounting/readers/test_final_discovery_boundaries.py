"""Header probes and configured discovery remain bounded, isolated, and identity-safe."""

from __future__ import annotations

from pathlib import Path

import pytest

from ssgrep.accounting.contracts import MAX_RAW_RECORD_BYTES
from ssgrep.accounting.readers.codex import CodexReader
from ssgrep.accounting.readers.omp import OmpReader
from ssgrep.accounting.readers.pi import PiReader
from ssgrep.accounting.readers.pi_family import HEADER_LINES, header_session_id
from ssgrep.accounting.readers.prime import PrimeReader
from tests.accounting.readers_fixtures import (
    codex_call,
    codex_meta,
    jsonl,
    pi_header,
    pi_tool_call,
    write_records,
)


def test_header_probe_skips_malformed_and_oversized_lines_before_valid_header(
    tmp_path: Path,
) -> None:
    path = tmp_path / "parent.jsonl"
    path.write_bytes(
        b"not json\n" + b"x" * (MAX_RAW_RECORD_BYTES + 1) + b"\n" + jsonl([pi_header("parent")])
    )
    assert header_session_id(path) == "parent"
    assert path.stat().st_size > MAX_RAW_RECORD_BYTES


def test_header_probe_accepts_last_bounded_line_but_not_later_header(tmp_path: Path) -> None:
    ignored = {"type": "title", "title": "not a session identity"}
    inside = write_records(
        tmp_path / "inside.jsonl", [ignored] * (HEADER_LINES - 1) + [pi_header("inside")]
    )
    outside = write_records(
        tmp_path / "outside.jsonl", [ignored] * HEADER_LINES + [pi_header("outside")]
    )
    assert header_session_id(inside) == "inside"
    assert header_session_id(outside) is None
    empty = write_records(tmp_path / "empty.jsonl", [])
    assert header_session_id(empty) is None
    assert header_session_id(tmp_path / "missing.jsonl") is None
    assert header_session_id(tmp_path) is None


def test_parent_resolution_distinguishes_missing_unknown_and_explicit_native_identity(
    tmp_path: Path,
) -> None:
    root = tmp_path / "sessions"
    parent = write_records(root / "parent.jsonl", [pi_header("native-parent")])
    unreadable = write_records(root / "no-header.jsonl", [{"type": "title"}])
    reader = OmpReader({"OMP_SESSIONS_DIR": str(root)})

    def link_for(name: str, **parent_fields):
        child = write_records(root / f"{name}.jsonl", [pi_header(name, **parent_fields)])
        header = reader.read(reader.source_for(child)).evidence[0]
        return reader.parent_link(header)

    unknown = link_for("unknown", parentSession=str(unreadable))
    assert (unknown.state, unknown.parent_session, unknown.parent_path) == (
        "unreadable",
        None,
        unreadable,
    )
    missing = link_for("missing", parentSessionId="not-discovered")
    assert (missing.state, missing.parent_session, missing.parent_path) == (
        "missing",
        "not-discovered",
        None,
    )
    resolved = link_for("resolved", parentSessionId="native-parent")
    assert (resolved.state, resolved.parent_session, resolved.parent_path) == (
        "resolved",
        "native-parent",
        parent,
    )
    external = write_records(tmp_path / "external.jsonl", [pi_header("external-parent")])
    alias = root / "alias.jsonl"
    alias.symlink_to(external)
    outside = link_for("outside", parentSession=str(alias))
    assert outside.state == "outside_roots"
    assert outside.parent_session is None
    assert alias not in [source.path for source in reader.discover()]
    assert parent in [source.path for source in reader.discover()]


@pytest.mark.parametrize(
    "reader_type,agent_var,session_var",
    [
        (PiReader, "PI_CODING_AGENT_DIR", "PI_SESSION_DIR"),
        (OmpReader, "OMP_AGENT_DIR", "OMP_SESSIONS_DIR"),
        (PrimeReader, "PRIME_AGENT_CODING_AGENT_DIR", "PRIME_AGENT_SESSION_DIR"),
    ],
)
def test_agent_root_expansion_and_explicit_session_precedence_use_only_selected_store(
    tmp_path: Path, reader_type, agent_var: str, session_var: str
) -> None:
    agent = write_records(
        tmp_path / "agent" / "sessions" / "agent.jsonl",
        [pi_header("agent"), pi_tool_call("e", "agent-call", "bash", {})],
    )
    explicit = write_records(
        tmp_path / "explicit" / "selected.jsonl",
        [pi_header("selected"), pi_tool_call("e", "selected-call", "bash", {})],
    )
    env = {"HOME": str(tmp_path), agent_var: " ~/agent "}
    reader = reader_type(env)
    assert [source.path for source in reader.discover()] == [agent]
    assert reader.read(reader.source_for(agent)).evidence[-1].record_id == "call:agent-call"
    selected = reader_type({**env, session_var: " ~/explicit "})
    assert [source.path for source in selected.discover()] == [explicit]
    assert selected.read(selected.source_for(explicit)).evidence[-1].session_id == "selected"


def test_codex_session_override_wins_over_home_and_expands_isolated_tilde(tmp_path: Path) -> None:
    selected = write_records(
        tmp_path / "chosen" / "rollout.jsonl",
        [codex_meta("selected"), codex_call("selected", "bash", {})],
    )
    write_records(
        tmp_path / "codex-home" / "sessions" / "rollout.jsonl",
        [codex_meta("ignored"), codex_call("ignored", "bash", {})],
    )
    reader = CodexReader(
        {
            "HOME": str(tmp_path),
            "CODEX_HOME": str(tmp_path / "codex-home"),
            "CODEX_SESSIONS_DIR": "~/chosen",
        }
    )
    assert [source.path for source in reader.discover()] == [selected]
    assert reader.read(reader.source_for(selected)).evidence_ids() == ["call:selected"]


def test_bare_tilde_root_does_not_resolve_against_real_home(tmp_path: Path) -> None:
    path = write_records(tmp_path / "session.jsonl", [pi_header("isolated")])
    reader = OmpReader({"HOME": str(tmp_path), "OMP_SESSIONS_DIR": "~"})
    assert [source.path for source in reader.discover()] == [path]
    assert reader.read(reader.source_for(path)).evidence[0].session_id == "isolated"
