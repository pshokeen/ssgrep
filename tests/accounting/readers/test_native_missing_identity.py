"""Unsupported native IDs remain deterministic but cannot invent tasks or org scope."""

from __future__ import annotations

from pathlib import Path

import pytest

from ssgrep.accounting.collector_baseline import baseline_record
from ssgrep.accounting.evidence_codec import evidence_from_line, evidence_line
from ssgrep.accounting.identity_claims import native_in_scope
from ssgrep.accounting.lineage import build_lineage
from ssgrep.accounting.readers.base import SourceState
from ssgrep.accounting.readers.pi import PiReader
from ssgrep.accounting.scope_events import Scoper
from ssgrep.accounting.timeline import build_timelines
from tests.accounting.readers_fixtures import (
    append_bytes,
    jsonl,
    pi_header,
    pi_tool_call,
    pi_user,
    write_records,
)
from tests.accounting.reduce_builders import policy


@pytest.mark.parametrize("missing", [None, ""])
def test_missing_header_ids_do_not_merge_equal_filename_headers(tmp_path: Path, missing):
    reader = PiReader()
    header = pi_header("ignored")
    if missing is None:
        header.pop("id")
    else:
        header["id"] = missing
    paths = [write_records(tmp_path / name / "same.jsonl", [header]) for name in ("a", "b")]
    items = [reader.read(reader.source_for(path)).evidence[0] for path in paths]
    assert all(
        e.kind == "session_header" and e.origin_status == "missing_session_id" for e in items
    )
    assert len({e.native_key for e in items}) == 2
    assert all(e.origin_domain == e.source_id for e in items)
    assert build_timelines(items).conflicts == 0


@pytest.mark.parametrize("missing", ["session", "entry"])
def test_missing_native_ids_force_unknown_task_and_nonexportable_org_scope(
    tmp_path: Path, missing: str
):
    reader = PiReader()
    records = [
        pi_header("native"),
        pi_user("turn", "prompt"),
        pi_tool_call("call", "call_0", "bash", {"command": "ssgrep search x"}, cwd="/work/demo"),
    ]
    if missing == "session":
        records.pop(0)
    else:
        records[-1].pop("id")
    path = write_records(tmp_path / "native.jsonl", records)
    view = build_timelines(reader.read(reader.source_for(path)).evidence)
    line = next(iter(view.sessions.values()))
    call = next(r for r in line.records if r.kind == "tool_call")
    lineage = build_lineage(view)
    task = lineage.task_of(line.key, call.seq)
    assert task.turn is None and task.branch == "unknown"
    assert "native-origin-unknown" in task.root
    assert lineage.context(line.key, call.seq) == ([], ["native_identity_unavailable"])
    scoper = Scoper(policy("/work/demo", profile="org"))
    assert scoper.in_scope(call.ev.payload["cwd"])
    assert not native_in_scope(call, view, scoper)


def test_compact_restart_keeps_native_turn_context_and_missing_prior_prompt_coverage(
    tmp_path: Path,
):
    reader = PiReader()
    path = write_records(
        tmp_path / "native.jsonl", [pi_header("native"), pi_user("turn", "private-before-setup")]
    )
    source = reader.source_for(path)
    state = SourceState.from_dict(baseline_record(reader, source).state)
    call = pi_tool_call("call", "call_0", "bash", {"command": "ssgrep search x"})
    call["parentId"] = "turn"
    append_bytes(path, jsonl([call]))
    [live] = reader.read(source, state).evidence
    retained = evidence_from_line(evidence_line(live))
    assert retained.payload["originating_turn_record_id"] == "turn:turn"
    assert retained.payload["task_context_basis"] == "native_parent_chain"
    assert retained.payload["baseline_prior_context_unavailable"] is True
    assert "private-before-setup" not in str(retained.payload)
    view = build_timelines([retained])
    line = next(iter(view.sessions.values()))
    assert build_lineage(view).task_of(line.key, 0).turn == "turn:turn"
    assert "baseline_prior_context_unavailable" in build_lineage(view).context(line.key, 0)[1]
