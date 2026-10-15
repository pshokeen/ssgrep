"""Authorized old witnesses are source evidence, never historical lookup imports."""

from __future__ import annotations

import hashlib
import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ssgrep.accounting.evidence_tokens import Token
from ssgrep.accounting.native_source import (
    NativeSourceAdapter,
    NativeSourceReference,
    NativeSourceStatus,
)
from ssgrep.accounting.readers.claude import ClaudeReader
from ssgrep.accounting.reduce import reduce_profile
from ssgrep.accounting.reduce_io import Loaded, reduce_stored_profile
from ssgrep.accounting.scope_events import Scoper
from ssgrep.accounting.sources import Consumer, Resolver
from ssgrep.accounting.timeline import build_timelines
from tests.accounting.reduce_builders import (
    KEY,
    PROFILE,
    claude_evidence,
    claude_prompt,
    claude_tool_result,
    claude_tool_use,
    policy,
    ts,
)
from tests.accounting.test_reduce_evidence import FACT, PATH, VIEW, consumer

REF = "claude:src1:ep:0"
NOW = int(datetime(2026, 10, 12, tzinfo=UTC).timestamp() * 1000)


def old_source(tmp_path: Path):
    old = datetime(2026, 10, 12, tzinfo=UTC) - timedelta(days=180)
    stamps = [(old + timedelta(seconds=s)).isoformat() for s in (0, 1, 2, 3)]
    rows = [
        claude_prompt("src1", "investigate", uuid="old-turn", ts=stamps[0]),
        claude_tool_use("src1", "w1", "Bash", {"command": "cat deploy.log"}, ts=stamps[1]),
        claude_tool_result("src1", "w1", FACT, ts=stamps[2]),
        claude_tool_use(
            "src1", "old-lookup", "Bash", {"command": "ssgrep search ancient"}, ts=stamps[3]
        ),
    ]
    evidence = claude_evidence(tmp_path, {"-work/src1.jsonl": rows})
    path = tmp_path / ".claude/projects/-work/src1.jsonl"
    ref = NativeSourceReference(
        REF,
        "claude",
        "src1",
        path,
        hashlib.sha256(path.read_bytes()).hexdigest(),
        tuple((e.native_identity, e.digest) for e in evidence),
        "/work/proj",
    )
    adapter = NativeSourceAdapter(
        {REF: ref},
        {"claude": ClaudeReader()},
        (tmp_path,),
        Scoper(policy()),
        NOW,
    )
    return path, ref, adapter


def test_old_witness_resolves_without_importing_old_lookup(tmp_path: Path) -> None:
    path, _, adapter = old_source(tmp_path / "old")
    before = path.read_bytes()
    evidence = claude_evidence(tmp_path / "current", {"-work/cons.jsonl": consumer()})
    reduced = reduce_profile(
        PROFILE,
        evidence,
        [],
        policy=policy(),
        frequency=VIEW,
        fingerprint_key=KEY,
        native_sources=adapter,
    )
    assert {call.call_id for call in reduced.invocations} == {"t1"}
    assert reduced.candidates and reduced.candidates[0].candidate.immediate
    assert all(a.usable for a in reduced.candidates[0].candidate.immediate)
    assert path.read_bytes() == before
    assert {item.session_id for item in evidence} == {"cons"}


@pytest.mark.parametrize(
    "case,status",
    [
        ("missing", NativeSourceStatus.MISSING),
        ("expired", NativeSourceStatus.EXPIRED),
        ("changed", NativeSourceStatus.CHANGED),
        ("private", NativeSourceStatus.UNAPPROVED),
        ("identity", NativeSourceStatus.IDENTITY_MISMATCH),
        ("record_digest", NativeSourceStatus.CHANGED),
        ("scope", NativeSourceStatus.OUT_OF_SCOPE),
    ],
)
def test_unavailable_sources_remain_distinct(tmp_path: Path, case, status) -> None:
    path, ref, adapter = old_source(tmp_path)
    if case == "missing":
        path.unlink()
    elif case == "expired":
        ref = replace(ref, expires_at_ms=NOW)
    elif case == "changed":
        path.write_bytes(path.read_bytes().replace(b"deploy.log", b"changed.log"))
    elif case == "private":
        adapter.approved_roots = (tmp_path / "other",)
    elif case == "identity":
        ref = replace(ref, session_id="different")
    elif case == "record_digest":
        ref = replace(ref, record_digests=(("result:w1", "0" * 64),))
    elif case == "scope":
        adapter.scoper = Scoper(policy("/approved", profile="company"))
    adapter.references = {REF: ref}
    result = adapter.resolve(REF)
    assert result.status is status
    assert result.line is None
    resolver = Resolver(build_timelines([]), adapter.scoper, native_sources=adapter)
    found = resolver.resolve(
        Token("path", PATH), (REF,), Consumer("claude:cons", ("claude", "cons"), 1, NOW)
    )
    assert result.reason in {reason for reason, _ in found.reasons}
    assert found.immediate and not found.immediate[0].usable


def test_first_native_read_computes_provenance_without_impossible_prior_attestation(
    tmp_path: Path,
) -> None:
    _, ref, adapter = old_source(tmp_path)
    adapter.references = {REF: replace(ref, source_digest="", record_digests=())}
    result = adapter.resolve(REF)
    assert result.status is NativeSourceStatus.OK
    assert len(result.source_digest) == 64
    assert result.record_digests
    assert adapter.observed_references[REF].source_digest == result.source_digest
    assert result.observed_at_ms == NOW


def test_total_byte_budget_is_strict_and_reads_create_nothing(tmp_path: Path) -> None:
    path, _, adapter = old_source(tmp_path)
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    adapter.byte_budget = len(path.read_bytes()) - 1
    bounded = adapter.resolve(REF)
    assert bounded.status is NativeSourceStatus.BOUNDED
    assert bounded.bytes_read == 0
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before


@pytest.mark.parametrize("kind", ["fifo", "symlink", "ancestor_symlink"])
def test_special_or_redirected_sources_never_open_blocking(tmp_path: Path, kind: str) -> None:
    path, ref, adapter = old_source(tmp_path)
    if kind == "ancestor_symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(path.parent, target_is_directory=True)
        ref = replace(ref, path=alias / path.name)
    else:
        path.unlink()
        if kind == "fifo":
            os.mkfifo(path)
        else:
            path.symlink_to(tmp_path / "outside")
    adapter.references = {REF: ref}
    assert adapter.resolve(REF).status is NativeSourceStatus.UNAPPROVED


def test_actual_stored_reduction_accepts_old_source_adapter(tmp_path: Path) -> None:
    _, _, adapter = old_source(tmp_path / "old")
    evidence = claude_evidence(tmp_path / "current", {"-work/cons.jsonl": consumer()})
    reduced, loaded = reduce_stored_profile(
        PROFILE,
        frequency=VIEW,
        loaded=Loaded(evidence=evidence),
        native_sources=adapter,
    )
    assert reduced.candidates and all(a.usable for a in reduced.candidates[0].candidate.immediate)
    assert {item.session_id for item in loaded.evidence} == {"cons"}


def test_native_copy_cannot_disguise_current_consumer_work_as_old_source(tmp_path: Path) -> None:
    rows = consumer()
    rows[1]["uuid"] = "7c557264-cba9-4a47-88ab-e5bbf1aa9440"
    rows[2]["uuid"] = "00ad4ca6-8c79-4d6c-b295-5d9c413772a2"
    current = claude_evidence(tmp_path / "current", {"-work/cons.jsonl": rows})
    copied = [
        {**row, "sessionId": "src1", "timestamp": "2026-04-15T00:00:00.000Z"} for row in rows[:3]
    ]
    claude_evidence(tmp_path / "old", {"-work/src1.jsonl": copied})
    path = tmp_path / "old/.claude/projects/-work/src1.jsonl"
    adapter = NativeSourceAdapter(
        {REF: NativeSourceReference(REF, "claude", "src1", path)},
        {"claude": ClaudeReader()},
        (tmp_path / "old",),
        Scoper(policy()),
        NOW,
    )
    reduced = reduce_profile(
        PROFILE,
        current,
        [],
        policy=policy(),
        frequency=VIEW,
        fingerprint_key=KEY,
        native_sources=adapter,
    )
    assert "consumer_own_later_work" in {row.reason for row in reduced.exclusions}
    assert not any(a.usable for c in reduced.candidates for a in c.candidate.immediate)


def test_verified_source_prefix_survives_append_but_old_record_mutation_fails(
    tmp_path: Path,
) -> None:
    path, _, adapter = old_source(tmp_path)
    original = adapter.resolve(REF)
    assert original.status is NativeSourceStatus.OK
    prefix = adapter.observed_references[REF]
    assert prefix.source_bytes == len(path.read_bytes())
    from tests.accounting.readers_fixtures import jsonl

    with path.open("ab") as handle:
        handle.write(
            jsonl(
                [
                    claude_tool_use(
                        "src1", "future", "Bash", {"command": "ssgrep search future"}, ts=ts()
                    ),
                    claude_tool_result("src1", "future", "future text", ts=ts()),
                ]
            )
        )
    appended = adapter.resolve(REF)
    assert appended.status is NativeSourceStatus.OK
    assert [r.identity for r in appended.line.records] == [
        r.identity for r in original.line.records
    ]
    assert appended.source_digest == original.source_digest
    assert not any(r.ev.payload.get("call_id") == "future" for r in appended.line.records)
    path.write_bytes(path.read_bytes().replace(b"ERR_QUOTA_EXCEEDED", b"ERR_QUOTA_MUTATED!"))
    assert adapter.resolve(REF).status is NativeSourceStatus.CHANGED


def test_source_native_cwd_obeys_foreign_assignment_inside_broad_org_root(tmp_path: Path) -> None:
    from ssgrep.accounting.policy_rules import PolicyFile
    from ssgrep.accounting.reduce_io import native_source_adapter

    broad = policy("/work", profile="org-a")
    other = policy("/work/team", profile="org-b")
    ownership = PolicyFile(
        profiles={"org-a": broad, "org-b": other},
        assignments={"/work/team/project": "org-b"},
    )
    rows = [
        {**claude_prompt("src1", "inspect", ts=ts(1)), "cwd": "/work/team/project"},
        claude_tool_use(
            "src1",
            "w1",
            "Bash",
            {"command": "cat log", "cwd": "/work"},
            cwd="/work/team/project",
            ts=ts(1, s=1),
        ),
        {**claude_tool_result("src1", "w1", FACT, ts=ts(1, s=2)), "cwd": "/work/team/project"},
    ]
    claude_evidence(tmp_path, {"-work/src1.jsonl": rows})
    path = tmp_path / ".claude/projects/-work/src1.jsonl"
    adapter = native_source_adapter(
        broad,
        {"HOME": str(tmp_path), "CLAUDE_CONFIG_DIR": str(tmp_path / ".claude")},
        now_ms=NOW,
        ownership=ownership,
        references={REF: NativeSourceReference(REF, "claude", "src1", path, cwd="/work")},
    )
    assert adapter.scoper.ownership is not None
    assert adapter.scoper.ownership.assignments == ownership.assignments
    assert adapter.resolve(REF).status is NativeSourceStatus.OUT_OF_SCOPE


@pytest.mark.parametrize("case", ["cwd", "root", "writable"])
def test_rejected_source_authority_preserves_bytes_and_accounting(tmp_path, case):
    path, reference, adapter = old_source(tmp_path)
    reader = adapter.readers["claude"]
    witness = next(
        item for item in reader.read(reader.source_for(path)).evidence if item.kind == "tool_result"
    )
    control = adapter.read_record(
        reference, witness.native_identity, witness.digest, locator=witness.locator
    )
    assert control.status is NativeSourceStatus.OK and control.evidence == witness
    charged = adapter.bytes_read
    if case == "cwd":
        adapter.references = {REF: replace(reference, cwd="/private/foreign-project")}
        adapter.scoper = Scoper(policy("/work", profile="company"))
        status = NativeSourceStatus.OUT_OF_SCOPE
    elif case == "root":
        adapter.approved_roots = (tmp_path / "unapproved",)
        status = NativeSourceStatus.UNAPPROVED
    else:
        path.chmod(0o666)
        status = NativeSourceStatus.UNAPPROVED
    before = path.read_bytes()
    mode = path.stat().st_mode
    result = adapter.resolve(REF)
    assert result.status is status and result.line is None
    assert result.bytes_read == 0 and adapter.bytes_read == charged
    assert not adapter.observed_references
    assert path.read_bytes() == before and path.stat().st_mode == mode
    assert str(path) not in result.reason and FACT not in result.reason


@pytest.mark.parametrize("case", ["invalid", "past_end", "context", "mid_record", "unterminated"])
def test_direct_jsonl_boundaries_return_unknown_without_hydrating_wrong_record(tmp_path, case):
    from tests.accounting.test_native_source_review_edges import json_scene

    path, _, _, witness, reference, adapter = json_scene(tmp_path, "claude")
    control = adapter.read_record(
        reference, witness.native_identity, witness.digest, locator=witness.locator
    )
    assert control.status is NativeSourceStatus.OK and control.evidence == witness
    charged = adapter.bytes_read
    original = path.read_bytes()
    locator = witness.locator
    if case == "invalid":
        locator, status, expected_bytes = "offset:-1", NativeSourceStatus.UNRESOLVED, 0
    elif case == "past_end":
        locator = f"offset:{len(original)}"
        status, expected_bytes = NativeSourceStatus.MISSING, 0
    elif case == "context":
        from ssgrep.accounting.native_source_record import CONTEXT_BYTES

        with path.open("ab") as stream:
            stream.truncate(2 * CONTEXT_BYTES + 2)
        locator = f"offset:{2 * CONTEXT_BYTES + 1}"
        status, expected_bytes = NativeSourceStatus.INCOMPLETE, 0
    elif case == "mid_record":
        locator, status, expected_bytes = "offset:1", NativeSourceStatus.INCOMPLETE, 1
    else:
        assert original.endswith(b"\n")
        path.write_bytes(original[:-1])
        status, expected_bytes = NativeSourceStatus.INCOMPLETE, len(original) - 1
    before = path.read_bytes()
    result = adapter.read_record(
        reference, witness.native_identity, witness.digest, locator=locator
    )
    assert result.status is status and result.evidence is None
    assert result.bytes_read == expected_bytes
    assert adapter.bytes_read - charged == expected_bytes
    assert path.read_bytes() == before
    assert str(path) not in result.reason and "answer" not in result.reason


@pytest.mark.parametrize("case", ["zero_budget", "oversized_record"])
def test_direct_record_byte_caps_do_not_return_unverified_payload(tmp_path, case):
    from ssgrep.accounting.contracts import MAX_RAW_RECORD_BYTES
    from tests.accounting.test_native_source_review_edges import json_scene

    path, _, _, witness, reference, adapter = json_scene(tmp_path, "claude")
    control = adapter.read_record(
        reference, witness.native_identity, witness.digest, locator=witness.locator
    )
    assert control.status is NativeSourceStatus.OK and control.evidence == witness
    if case == "zero_budget":
        adapter = replace(adapter, bytes_read=0, byte_budget=0)
        locator, expected_bytes = witness.locator, 0
    else:
        path.write_bytes(b"x" * (MAX_RAW_RECORD_BYTES + 1) + b"\n")
        locator, expected_bytes = "offset:0", MAX_RAW_RECORD_BYTES + 1
    charged = adapter.bytes_read
    result = adapter.read_record(
        reference, witness.native_identity, witness.digest, locator=locator
    )
    assert result.status is NativeSourceStatus.BOUNDED and result.evidence is None
    assert result.bytes_read == expected_bytes
    assert adapter.bytes_read - charged == expected_bytes


@pytest.mark.parametrize(
    "case,status",
    [
        ("session", NativeSourceStatus.IDENTITY_MISMATCH),
        ("missing_header", NativeSourceStatus.IDENTITY_MISMATCH),
        ("unknown_cwd", NativeSourceStatus.SCOPE_UNKNOWN),
    ],
)
def test_direct_codex_header_authority_is_required_before_payload(tmp_path, case, status):
    from tests.accounting.readers_fixtures import codex_meta, codex_output, write_records
    from tests.accounting.test_native_source_review_edges import json_scene

    path, _, _, witness, reference, adapter = json_scene(tmp_path, "codex")
    control = adapter.read_record(
        reference, witness.native_identity, witness.digest, locator=witness.locator
    )
    assert control.status is NativeSourceStatus.OK and control.evidence == witness
    if case == "session":
        reference = replace(reference, session_id="different-native-owner")
    elif case == "missing_header":
        write_records(path, [codex_output("c", "answer")])
    else:
        header = codex_meta("s", "/work/approved/project")
        del header["payload"]["cwd"]
        write_records(path, [header, codex_output("c", "answer")])
    first_line_bytes = len(path.read_bytes().splitlines(keepends=True)[0])
    charged = adapter.bytes_read
    result = adapter.read_record(
        reference, witness.native_identity, witness.digest, locator="offset:0"
    )
    assert result.status is status and result.evidence is None
    assert result.bytes_read == first_line_bytes
    assert adapter.bytes_read - charged == first_line_bytes
    assert "answer" not in result.reason and str(path) not in result.reason


def test_direct_record_accepts_complete_context_at_exact_seek_window(tmp_path):
    from ssgrep.accounting.native_source_record import CONTEXT_BYTES
    from tests.accounting.test_native_source_review_edges import json_scene

    path, _, originals, _, reference, adapter = json_scene(tmp_path, "claude")
    witness = next(item for item in originals if item.kind == "tool_call")
    original_offset = int(witness.locator.partition(":")[2])
    with path.open("rb") as stream:
        stream.seek(original_offset)
        record = stream.readline()
    opening = b'{"type":"system","cwd":"/work/approved/project","text":"'
    closing = b'"}\n'
    offset = 2 * CONTEXT_BYTES
    prefix = opening + b"x" * (offset - len(opening) - len(closing)) + closing
    assert len(prefix) == offset
    path.write_bytes(prefix + record)
    result = adapter.read_record(
        reference, witness.native_identity, witness.digest, locator=f"byte:{offset}"
    )
    assert result.status is NativeSourceStatus.OK and result.evidence is not None
    assert result.evidence.native_identity == witness.native_identity
    assert result.evidence.digest == witness.digest and result.evidence.payload == witness.payload
    assert result.bytes_read == adapter.bytes_read == offset + len(record)
