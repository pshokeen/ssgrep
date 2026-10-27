"""Authorized old witnesses are source evidence, never historical lookup imports."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from ssgrep.accounting.collect.replay_evidence import Consumer, Resolver, Token
from ssgrep.accounting.ledger.reduce import Loaded, reduce_profile, reduce_stored_profile
from ssgrep.accounting.ledger.units import WORK_SECONDS, UnitSpec
from ssgrep.accounting.ledger.valuation_inputs import footprint, footprint_size
from ssgrep.accounting.readers.jsonl_runtimes import ClaudeReader
from ssgrep.accounting.readers.native_source import (
    NativeSourceAdapter,
    NativeSourceReference,
    NativeSourceStatus,
)
from ssgrep.accounting.readers.timeline import build_timelines
from ssgrep.accounting.views.filter import Scoper
from tests.accounting.ledger.reduce_builders import (
    KEY,
    PROFILE,
    claude_evidence,
    claude_prompt,
    claude_tool_result,
    claude_tool_use,
    policy,
    ts,
)
from tests.accounting.ledger.test_reduce_evidence import FACT, PATH, VIEW, consumer
from tests.accounting.readers.readers_fixtures import jsonl
from tests.accounting.readers.review_corpus import build_corpus

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


def test_native_resolution_requires_known_reference_and_supported_reader(tmp_path: Path) -> None:
    _, _, adapter = old_source(tmp_path)
    assert adapter.resolve(REF).status is NativeSourceStatus.OK
    assert adapter.resolve("claude:missing:ep:0").status is NativeSourceStatus.UNRESOLVED

    adapter.readers.pop("claude")
    unsupported = adapter.resolve(REF)
    assert unsupported.status is NativeSourceStatus.UNSUPPORTED
    assert unsupported.line is None and unsupported.bytes_read == 0


def test_malformed_native_jsonl_is_incomplete_and_charges_only_examined_prefix(
    tmp_path: Path,
) -> None:
    path, _, adapter = old_source(tmp_path)
    assert adapter.resolve(REF).status is NativeSourceStatus.OK
    malformed = b"{bad json\n"
    path.write_bytes(malformed + path.read_bytes())
    charged = adapter.bytes_read
    result = adapter.resolve(REF)
    assert result.status is NativeSourceStatus.INCOMPLETE
    assert result.line is None
    assert result.bytes_read == len(malformed)
    assert adapter.bytes_read - charged == len(malformed)


def test_native_read_proof_coalesces_exact_ranges_and_rehashes_on_metadata_change(tmp_path):
    from ssgrep.accounting.readers.native_source import ReadProof, SourceReadError

    path = tmp_path / "proof.jsonl"
    path.write_bytes(b"safe\n")
    descriptor = os.open(path, os.O_RDONLY)
    try:
        before = os.fstat(descriptor)
        proof = ReadProof()
        # Settled: stamped well before the proof (beyond even a 2 s whole-second granule).
        proof.observed_ns = max(before.st_mtime_ns, before.st_ctime_ns) + 10**10
        proof.add(0, b"sa")
        proof.add(2, b"fe\n")
        proof.add(0, b"safe\n")  # a reread fully inside the held range adds no duplicate
        charged: list[int] = []
        proof.verify(descriptor, before, charged.append)
        assert charged == []
        assert len(proof.ranges) == 1

        path.write_bytes(b"evil\n")
        # A whole-second change, visible even on filesystems with coarse timestamps.
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 5 * 10**9))
        with pytest.raises(SourceReadError) as failure:
            proof.verify(descriptor, before, charged.append)
        assert failure.value.status is NativeSourceStatus.CHANGED
        assert sum(charged) == len(b"safe\n")
    finally:
        os.close(descriptor)


def test_native_read_proof_rehashes_racy_source_and_rejects_same_size_rewrite(tmp_path):
    """A rewrite inside one timestamp granule may leave size and times unchanged;
    a proof whose source was stamped that recently must compare bytes instead."""
    from ssgrep.accounting.readers.native_source import ReadProof, SourceReadError

    path = tmp_path / "racy.jsonl"
    path.write_bytes(b"safe\n")
    descriptor = os.open(path, os.O_RDONLY)
    try:
        before = os.fstat(descriptor)
        proof = ReadProof()
        proof.observed_ns = max(before.st_mtime_ns, before.st_ctime_ns)  # stamped just now
        proof.add(0, b"safe\n")
        charged: list[int] = []
        proof.verify(descriptor, before, charged.append)
        assert charged == []  # proven by an uncharged re-hash, not by trusting metadata

        with open(path, "r+b") as handle:  # same size, no timestamp manipulation
            handle.write(b"evil\n")
        with pytest.raises(SourceReadError) as failure:
            proof.verify(descriptor, before, charged.append)
        assert failure.value.status is NativeSourceStatus.CHANGED
    finally:
        os.close(descriptor)


def test_native_read_proof_rejects_world_writable_source_with_positive_control(tmp_path):
    from ssgrep.accounting.readers.native_source import ReadProof, SourceReadError

    path = tmp_path / "permissions.jsonl"
    path.write_text("safe\n")
    secure = path.stat()
    ReadProof._identity(secure, secure)
    path.chmod(0o666)
    try:
        with pytest.raises(SourceReadError) as failure:
            ReadProof._identity(secure, path.stat())
        assert failure.value.status is NativeSourceStatus.UNAPPROVED
    finally:
        path.chmod(0o600)


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
    from tests.accounting.readers.readers_fixtures import jsonl

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
    from ssgrep.accounting.ledger.reduce import native_source_adapter
    from ssgrep.accounting.privacy.policy import PolicyFile

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


def test_unchanged_native_source_uses_verified_snapshot_cache_without_rereading(tmp_path: Path):
    _, _, adapter = old_source(tmp_path)
    first = adapter.resolve(REF)
    assert first.status is NativeSourceStatus.OK
    charged = adapter.bytes_read
    assert charged > 0

    cached = adapter.resolve(REF)

    assert cached.status is NativeSourceStatus.OK
    assert cached.bytes_read == 0
    assert adapter.bytes_read == charged
    assert [record.identity for record in cached.line.records] == [
        record.identity for record in first.line.records
    ]


def test_snapshot_digest_read_error_falls_back_to_source_read_and_closes_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real snapshot file whose proof read errors is reread, never served from cache."""
    from ssgrep.accounting.readers import native_source

    path, _, adapter = old_source(tmp_path)
    # Make the first snapshot racy so the adapter creates a verified byte snapshot.
    monkeypatch.setattr(native_source.time, "time_ns", lambda: 0)
    first = adapter.resolve(REF)
    assert first.status is NativeSourceStatus.OK and first.bytes_read > 0
    assert adapter._snapshots

    opened: list[int] = []
    closed: list[int] = []
    real_open = native_source.open_source
    real_close = native_source.os.close

    def track_open(source, roots):
        descriptor = real_open(source, roots)
        opened.append(descriptor)
        return descriptor

    def track_close(descriptor: int) -> None:
        if descriptor in opened:
            closed.append(descriptor)
        real_close(descriptor)

    def fail_snapshot_read(_descriptor: int, _size: int) -> bytes:
        raise OSError("injected snapshot cache read failure")

    monkeypatch.setattr(native_source, "open_source", track_open)
    monkeypatch.setattr(native_source.os, "close", track_close)
    monkeypatch.setattr(native_source.os, "read", fail_snapshot_read)

    fallback = adapter.resolve(REF)

    assert fallback.status is NativeSourceStatus.OK
    # A cached snapshot hit would report zero; the failed proof forced a real source read.
    assert fallback.bytes_read > 0
    assert opened and opened[0] in closed


def test_native_snapshot_cache_reloads_when_source_stamp_changes(tmp_path: Path) -> None:
    path, _, adapter = old_source(tmp_path)
    first = adapter.resolve(REF)
    assert first.status is NativeSourceStatus.OK
    charged = adapter.bytes_read

    # The stamp invalidates the cached snapshot, but the referenced record stays valid.
    with path.open("ab") as source:
        source.write(b"\n")

    reread = adapter.resolve(REF)
    assert reread.status is NativeSourceStatus.OK
    assert reread.bytes_read > 0
    assert adapter.bytes_read > charged
    assert reread.line is not None
    assert [record.identity for record in reread.line.records] == [
        record.identity for record in first.line.records
    ]


def test_native_snapshot_cache_does_not_bypass_exhausted_budget_on_changed_source(
    tmp_path: Path,
) -> None:
    path, _, adapter = old_source(tmp_path)
    first = adapter.resolve(REF)
    assert first.status is NativeSourceStatus.OK
    adapter.byte_budget = adapter.bytes_read
    with path.open("ab") as source:
        source.write(b"\n")
    charged = adapter.bytes_read

    bounded = adapter.resolve(REF)
    assert bounded.status is NativeSourceStatus.BOUNDED
    assert bounded.line is None
    assert bounded.bytes_read == 0
    assert adapter.bytes_read == charged


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
    from tests.accounting.readers.test_native_source_review_edges import json_scene

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
        from ssgrep.accounting.readers.native_source import CONTEXT_BYTES

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
    from ssgrep.accounting.ledger.contracts import MAX_RAW_RECORD_BYTES
    from tests.accounting.readers.test_native_source_review_edges import json_scene

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
    from tests.accounting.readers.readers_fixtures import codex_meta, codex_output, write_records
    from tests.accounting.readers.test_native_source_review_edges import json_scene

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
    from ssgrep.accounting.readers.native_source import CONTEXT_BYTES
    from tests.accounting.readers.test_native_source_review_edges import json_scene

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


"""Compact replay resolves known native records directly, never whole histories."""


def test_direct_record_seek_does_not_scan_large_history(tmp_path: Path) -> None:
    path, reference, adapter = old_source(tmp_path)
    source = ClaudeReader().source_for(path)
    result = ClaudeReader().read(source)
    witness = next(item for item in result.evidence if item.kind == "tool_result")
    with path.open("ab") as handle:
        handle.truncate(2 * 1024 * 1024 * 1024)
    found = adapter.read_record(
        reference, witness.native_identity, witness.digest, locator=witness.locator
    )
    assert found.status is NativeSourceStatus.OK
    assert found.evidence.native_identity == witness.native_identity
    assert found.evidence.digest == witness.digest
    assert found.bytes_read < 128 * 1024


def test_direct_record_changes_are_not_silently_hydrated(tmp_path: Path) -> None:
    path, reference, adapter = old_source(tmp_path)
    result = ClaudeReader().read(ClaudeReader().source_for(path))
    witness = next(item for item in result.evidence if item.kind == "tool_result")
    path.write_bytes(path.read_bytes().replace(b"ERR_QUOTA_EXCEEDED", b"ERR_QUOTA_MUTATED!"))
    found = adapter.read_record(
        reference, witness.native_identity, witness.digest, locator=witness.locator
    )
    assert found.status is NativeSourceStatus.CHANGED
    assert found.evidence is None


def test_legacy_unknown_locator_is_explicit_unavailable_not_full_scan(tmp_path: Path) -> None:
    _, reference, adapter = old_source(tmp_path)
    found = adapter.read_record(reference, "claude:src1:result:w1", "0" * 64, locator="unknown")
    assert found.status is NativeSourceStatus.UNRESOLVED
    assert found.evidence is None and found.bytes_read == 0


@pytest.mark.parametrize("original", ["native-a", "native-b"])
@pytest.mark.parametrize("retarget_ancestor", [False, True])
def test_approved_native_root_retarget_never_opens_another_tenant(
    tmp_path: Path, original: str, retarget_ancestor: bool
) -> None:
    from ssgrep.accounting.readers.native_source import NativeSourceAdapter

    other = "native-b" if original == "native-a" else "native-a"
    original_dir = tmp_path.resolve() / original
    other_dir = tmp_path.resolve() / other
    native_root = original_dir / "records" if retarget_ancestor else original_dir
    foreign_root = other_dir / "records" if retarget_ancestor else other_dir
    _, reference, first = old_source(native_root)
    old_source(foreign_root)
    first.approved_roots = (native_root,)
    accepted = first.resolve(reference.ref)
    assert accepted.status is NativeSourceStatus.OK
    observed = first.observed_references[reference.ref]

    def restarted():
        return NativeSourceAdapter(
            {observed.ref: observed},
            first.readers,
            (native_root,),
            first.scoper,
            first.now_ms,
        )

    # A new process can still read the unchanged explicitly approved source.
    assert restarted().resolve(reference.ref).status is NativeSourceStatus.OK
    original_dir.rename(tmp_path / (original + "-accepted"))
    original_dir.symlink_to(other_dir, target_is_directory=True)
    result = restarted().resolve(reference.ref)
    # The foreign transcript intentionally has identical bytes/digests: content
    # equality cannot substitute for the approved directory's authorization.
    assert result.status is NativeSourceStatus.UNAPPROVED
    assert result.line is None


CACHE_CONSUMER = Consumer("claude:consumer", ("claude", "consumer"), 1, NOW + 172_800_000)


def test_indexed_filename_alias_uses_native_witness_not_colliding_retained_session(
    tmp_path: Path,
) -> None:
    _, reference, adapter = old_source(tmp_path / "original")
    alias = "claude:indexed-filename:ep:0"
    adapter.references = {
        alias: replace(reference, ref=alias, indexed_session_id="indexed-filename")
    }
    rows = [claude_prompt("src1", "separate retained history", uuid="retained-turn", ts=ts(1))]
    for i in range(4):
        rows.extend(
            [
                claude_tool_use(
                    "src1", f"retained-{i}", "Bash", {"command": "cat separate.log"}, ts=ts(2)
                ),
                claude_tool_result(
                    "src1", f"retained-{i}", FACT if i == 3 else "unrelated", ts=ts(3)
                ),
            ]
        )
    retained = claude_evidence(tmp_path / "retained", {"-work/src1.jsonl": rows})
    resolver = Resolver(build_timelines(retained), adapter.scoper, native_sources=adapter)
    held = resolver.resolve(Token("path", PATH), (REF,), CACHE_CONSUMER)
    assert any(atom.atom.origin_event.endswith("result:retained-3") for atom in held.immediate)
    actual = resolver.resolve(Token("path", PATH), (alias,), CACHE_CONSUMER)
    assert any(atom.atom.origin_event.endswith("result:w1") for atom in actual.immediate)
    assert not any("retained-" in atom.atom.origin_event for atom in actual.immediate)


def test_native_drift_rejection_is_cached_without_starving_unrelated_source_budget(
    tmp_path: Path,
) -> None:
    path, reference, adapter = old_source(tmp_path)
    alias = "claude:second-indexed-filename:ep:0"
    adapter.references = {
        REF: reference,
        alias: replace(
            reference,
            ref=alias,
            indexed_session_id="second-indexed-filename",
            source_digest="",
            record_digests=(),
            source_bytes=None,
        ),
    }
    resolver = Resolver(build_timelines([]), adapter.scoper, native_sources=adapter)
    first = resolver.resolve(Token("path", PATH), (REF,), CACHE_CONSUMER)
    assert any(atom.usable for atom in first.immediate)
    before = adapter.bytes_read
    later = (
        datetime(2026, 10, 12, tzinfo=UTC) - timedelta(days=180) + timedelta(seconds=5)
    ).isoformat()
    with path.open("ab") as stream:
        stream.write(
            jsonl(
                [
                    claude_tool_use(
                        "src1", "new-record", "Bash", {"command": "cat another.log"}, ts=later
                    )
                ]
            )
        )
    drift = resolver.resolve(Token("path", PATH), (alias,), CACHE_CONSUMER)
    assert ("source_digest_drift", alias) in drift.reasons
    spent = adapter.bytes_read
    assert spent > before
    repeated = resolver.resolve(Token("path", PATH), (alias,), CACHE_CONSUMER)
    assert ("source_digest_drift", alias) in repeated.reasons
    assert adapter.bytes_read == spent


GAP_BASE = datetime(2026, 4, 15, tzinfo=UTC)
GAP_CONSUMER = Consumer("claude:consumer", ("claude", "consumer"), 0, 1_800_000_000_000)
GAP_WORK = UnitSpec(WORK_SECONDS, "capped_gap_v1")


def gap_resolved(
    tmp_path: Path, gap: float | None, *, terminal=False, next_turn=False, next_lookup=False
):
    rows = [
        claude_prompt("src", "investigate", uuid="u1", ts=GAP_BASE.isoformat()),
        claude_tool_use("src", "w1", "Bash", {"command": "cat log"}, ts=GAP_BASE.isoformat()),
        claude_tool_result("src", "w1", FACT, ts=GAP_BASE.isoformat()),
    ]
    if not terminal:
        end = (GAP_BASE + timedelta(seconds=gap)).isoformat() if gap is not None else None
        row = (
            claude_prompt("src", "unrelated next task", uuid="u2")
            if next_turn
            else claude_tool_use(
                "src",
                "w2",
                "Bash",
                {"command": "ssgrep search unrelated" if next_lookup else "continue"},
            )
        )
        row["timestamp"] = end
        rows.append(row)
    evidence = claude_evidence(tmp_path, {"-work/src.jsonl": rows})
    resolver = Resolver(build_timelines(evidence), Scoper(policy()))
    result = resolver.resolve(Token("path", PATH), ("claude:src:ep:0",), GAP_CONSUMER)
    return next(a for a in result.immediate if a.atom.origin_event.endswith("result:w1"))


@pytest.mark.parametrize("gap,expected", [(299, 299), (300, 300), (301, 300), (28801, 300), (0, 0)])
def test_actual_resolver_positive_gap_cap(tmp_path: Path, gap, expected) -> None:
    atom = gap_resolved(tmp_path, gap)
    assert atom.t_end - atom.t_start == expected
    assert atom.work_method == "capped_gap_v1"
    assert atom.actor == "claude:src"
    fp, missing = footprint((atom,), GAP_WORK)
    assert footprint_size(fp) == expected * 1_000_000_000
    assert missing == ()


@pytest.mark.parametrize(
    "gap,terminal,next_turn",
    [(-1, False, False), (None, False, False), (1, True, False), (301, False, True)],
)
def test_missing_reversed_terminal_and_task_boundary_gaps_are_unknown(
    tmp_path: Path, gap, terminal, next_turn
) -> None:
    atom = gap_resolved(tmp_path, gap, terminal=terminal, next_turn=next_turn)
    assert atom.usable  # still usable for chars, not fabricated time zero
    assert atom.t_start is None and atom.t_end is None
    fp, missing = footprint((atom,), GAP_WORK)
    assert footprint_size(fp) == 0 and missing


def test_new_lookup_boundary_is_not_original_witness_work(tmp_path: Path) -> None:
    atom = gap_resolved(tmp_path, 299, next_lookup=True)
    assert atom.t_start_ms is None and atom.t_end_ms is None
    assert atom.cost_reason == "source_lookup_boundary_gap"


def test_future_completion_after_exposure_is_not_original_saved_work(tmp_path: Path) -> None:
    atom = gap_resolved(tmp_path, 90_000_000)
    assert atom.t_start_ms is None and atom.t_end_ms is None
    assert atom.cost_reason == "source_gap_after_exposure"


def test_cap_precedes_overlap_union_and_distinct_actors_sum(tmp_path: Path) -> None:
    first = gap_resolved(tmp_path, 28801)
    second = replace(
        first,
        atom=replace(first.atom, atom_id="second", origin_event="second"),
        t_start=first.t_start + 200,
        t_end=first.t_start + 500,
        t_start_ms=first.t_start_ms + 200_000,
        t_end_ms=first.t_start_ms + 500_000,
    )
    fp, _ = footprint((first, second, first), GAP_WORK)
    assert footprint_size(fp) == 500_000_000_000  # capped overlap union; copies add none
    parallel = replace(first, actor="claude:other")
    fp, _ = footprint((first, parallel), GAP_WORK)
    assert footprint_size(fp) == 600_000_000_000


def test_fractional_gaps_preserve_milliseconds_until_union(tmp_path: Path) -> None:
    first = gap_resolved(tmp_path, 0.5)
    second = replace(
        first,
        atom=replace(first.atom, atom_id="second", origin_event="second"),
        t_start_ms=first.t_start_ms + 500,
        t_end_ms=first.t_start_ms + 1000,
    )
    fp, missing = footprint((first, second), GAP_WORK)
    assert footprint_size(fp) == 1_000_000_000
    assert missing == ()


def test_frozen_corpus_28801_gap_caps_on_actual_resolver(tmp_path: Path) -> None:
    from ssgrep.accounting.ledger.reduce import load_evidence

    corpus = build_corpus(tmp_path, setup_at=datetime(2026, 10, 12, tzinfo=UTC))
    evidence = load_evidence(corpus.home.env, only_under=[corpus.home.paths["omp"]]).evidence
    resolver = Resolver(build_timelines(evidence), Scoper(policy()))
    line = resolver.timelines.sessions[("omp", "o-src-gap")]
    call = next(r for r in line.records if r.kind == "tool_call")
    atom = resolver._atom(line, call, GAP_CONSUMER, set())
    assert atom is not None and atom.t_end is not None and atom.t_start is not None
    assert atom.t_end - atom.t_start == 300
    assert atom.work_method == "capped_gap_v1"


def test_native_source_descriptor_requires_approved_absolute_nontraversing_path(
    tmp_path: Path,
) -> None:
    from ssgrep.accounting.readers.native_source import SourceReadError, open_source

    path = tmp_path / "approved" / "records.jsonl"
    path.parent.mkdir()
    path.write_bytes(b'{"type":"message"}\n')
    descriptor = open_source(path, (Path("relative-root"), tmp_path / "approved"))
    try:
        assert os.read(descriptor, path.stat().st_size) == b'{"type":"message"}\n'
    finally:
        os.close(descriptor)

    with pytest.raises(SourceReadError) as failure:
        open_source(path.parent, (path.parent,))
    assert failure.value.status is NativeSourceStatus.UNAPPROVED

    for unapproved in (Path("relative.jsonl"), tmp_path / "approved" / ".." / "records.jsonl"):
        with pytest.raises(SourceReadError) as failure:
            open_source(unapproved, (tmp_path,))
        assert failure.value.status is NativeSourceStatus.UNAPPROVED


def test_indexed_native_reference_uses_stable_sqlite_generation_and_real_source(
    tmp_path: Path,
) -> None:
    import sqlite3

    from ssgrep.accounting.readers.native_source import index_reference

    path, _, adapter = old_source(tmp_path / "native")
    source_before = path.read_bytes()
    index = sqlite3.connect(tmp_path / "index.sqlite")
    index.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    index.executemany(
        "INSERT INTO metadata VALUES (?, ?)",
        [("index_state", "ready"), ("last_index_time", "generation-1")],
    )
    index.execute(
        "CREATE TABLE episodes (episode_id, session_id, runtime, source_path, cwd, source_status)"
    )
    index.execute(
        "INSERT INTO episodes VALUES (?, ?, ?, ?, ?, ?)",
        (REF, "src1", "claude", str(path), "/work/proj", "available"),
    )
    index.commit()

    class SQLiteIndex:
        def __init__(self, connection, advance_generation=False):
            self.connection = connection
            self.advance_generation = advance_generation
            self.advanced = False

        def exists(self):
            return True

        def get_meta(self, key):
            row = self.connection.execute(
                "SELECT value FROM metadata WHERE key = ?", (key,)
            ).fetchone()
            value = row[0] if row else None
            if key == "last_index_time" and self.advance_generation and not self.advanced:
                self.advanced = True
                self.connection.execute(
                    "UPDATE metadata SET value = 'generation-2' WHERE key = 'last_index_time'"
                )
                self.connection.commit()
            return value

        def rows(self, table, *, where, columns, limit):
            selected = ", ".join(columns)
            return [
                dict(zip(columns, row, strict=True))
                for row in self.connection.execute(
                    f"SELECT {selected} FROM {table} WHERE {where} LIMIT ?", (limit,)
                )
            ]

    repository = SQLiteIndex(index)
    reference, status = index_reference(REF, repository)
    assert status is NativeSourceStatus.OK
    assert reference.path == path and reference.session_id == "src1"

    adapter.references = {}
    adapter.repository = repository
    result = adapter.resolve(REF)
    assert result.status is NativeSourceStatus.OK
    assert result.line is not None and result.line.key == ("claude", "src1")
    assert path.read_bytes() == source_before
    assert adapter.bytes_read == len(source_before)

    stale_adapter = NativeSourceAdapter(
        {},
        adapter.readers,
        adapter.approved_roots,
        adapter.scoper,
        adapter.now_ms,
        repository=SQLiteIndex(index, advance_generation=True),
    )
    stale = stale_adapter.resolve(REF)
    assert stale.status is NativeSourceStatus.CHANGED and stale.line is None
    assert stale_adapter.bytes_read == 0
    assert path.read_bytes() == source_before
    index.close()


def test_native_resolution_rejects_conflicting_same_record_without_attestation(
    tmp_path: Path,
) -> None:
    import json

    path, reference, adapter = old_source(tmp_path)
    control = adapter.resolve(REF)
    assert control.status is NativeSourceStatus.OK and control.line is not None
    original = path.read_bytes()
    rows = [json.loads(line) for line in original.splitlines()]
    call = next(
        row
        for row in rows
        if any(
            block.get("id") == "w1"
            for block in row.get("message", {}).get("content", [])
            if isinstance(block, dict)
        )
    )
    variant = json.loads(json.dumps(call))
    block = next(block for block in variant["message"]["content"] if block.get("id") == "w1")
    block["input"]["command"] = "cat unrelated.log"
    path.write_bytes(original + json.dumps(variant, separators=(",", ":")).encode() + b"\n")
    unpinned = replace(reference, source_digest="", record_digests=(), source_bytes=None)
    changed_adapter = NativeSourceAdapter(
        {REF: unpinned},
        adapter.readers,
        adapter.approved_roots,
        adapter.scoper,
        adapter.now_ms,
    )
    before_read = path.read_bytes()
    changed = changed_adapter.resolve(REF)
    assert changed.status is NativeSourceStatus.CHANGED and changed.line is None
    assert path.read_bytes() == before_read


def test_native_resolution_rejects_tool_record_missing_session_authority(
    tmp_path: Path,
) -> None:
    import json

    path, reference, adapter = old_source(tmp_path)
    control = adapter.resolve(REF)
    assert control.status is NativeSourceStatus.OK and control.line is not None
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    call = next(
        row
        for row in rows
        if any(
            block.get("id") == "w1"
            for block in row.get("message", {}).get("content", [])
            if isinstance(block, dict)
        )
    )
    call.pop("sessionId")
    unpinned = replace(reference, source_digest="", record_digests=(), source_bytes=None)
    path.write_bytes(
        b"".join(json.dumps(row, separators=(",", ":")).encode() + b"\n" for row in rows)
    )
    changed_adapter = NativeSourceAdapter(
        {REF: unpinned},
        adapter.readers,
        adapter.approved_roots,
        adapter.scoper,
        adapter.now_ms,
    )
    before_read = path.read_bytes()
    result = changed_adapter.resolve(REF)
    assert result.status is NativeSourceStatus.IDENTITY_MISMATCH and result.line is None
    assert path.read_bytes() == before_read


def test_native_jsonl_record_cap_charges_examined_prefix_without_writing_source(
    tmp_path: Path,
) -> None:
    from ssgrep.accounting.ledger.contracts import MAX_RAW_RECORD_BYTES

    path, reference, adapter = old_source(tmp_path)
    control = adapter.resolve(REF)
    assert control.status is NativeSourceStatus.OK and control.line is not None
    original = path.read_bytes()
    bounded_adapter = NativeSourceAdapter(
        {REF: replace(reference, source_digest="", record_digests=(), source_bytes=None)},
        adapter.readers,
        adapter.approved_roots,
        adapter.scoper,
        adapter.now_ms,
        max_records=2,
    )
    bounded = bounded_adapter.resolve(REF)
    examined = sum(len(row) + 1 for row in original.splitlines()[:3])
    assert bounded.status is NativeSourceStatus.BOUNDED and bounded.line is None
    assert bounded.bytes_read == examined == bounded_adapter.bytes_read
    assert path.read_bytes() == original

    oversized = b"x" * (MAX_RAW_RECORD_BYTES + 1) + b"\n"
    path.write_bytes(oversized)
    oversized_adapter = NativeSourceAdapter(
        {REF: replace(reference, source_digest="", record_digests=(), source_bytes=None)},
        adapter.readers,
        adapter.approved_roots,
        adapter.scoper,
        adapter.now_ms,
    )
    before_read = path.read_bytes()
    capped = oversized_adapter.resolve(REF)
    assert capped.status is NativeSourceStatus.BOUNDED and capped.line is None
    assert capped.bytes_read == oversized_adapter.bytes_read == MAX_RAW_RECORD_BYTES + 1
    assert path.read_bytes() == before_read


def test_native_resolution_rejects_a_registered_reader_without_native_decode_support(
    tmp_path: Path,
) -> None:
    path, reference, adapter = old_source(tmp_path)
    control = adapter.resolve(REF)
    assert control.status is NativeSourceStatus.OK
    source_before = path.read_bytes()
    unsupported_reference = replace(reference, runtime="generic")
    unsupported_adapter = NativeSourceAdapter(
        {REF: unsupported_reference},
        {"generic": object()},
        adapter.approved_roots,
        adapter.scoper,
        adapter.now_ms,
    )
    result = unsupported_adapter.resolve(REF)
    assert result.status is NativeSourceStatus.UNSUPPORTED
    assert result.line is None and result.bytes_read == unsupported_adapter.bytes_read == 0
    assert path.read_bytes() == source_before


def test_verified_jsonl_prefix_cannot_be_truncated_and_still_resolve(tmp_path: Path) -> None:
    path, _, adapter = old_source(tmp_path)
    control = adapter.resolve(REF)
    assert control.status is NativeSourceStatus.OK
    charged = adapter.bytes_read
    path.write_bytes(path.read_bytes()[:1])
    truncated = path.read_bytes()
    result = adapter.resolve(REF)
    assert result.status is NativeSourceStatus.CHANGED and result.line is None
    assert result.bytes_read == 0 and adapter.bytes_read == charged
    assert path.read_bytes() == truncated


def test_renamed_native_snapshot_cannot_be_served_from_cached_evidence(tmp_path: Path) -> None:
    path, _, adapter = old_source(tmp_path)
    first = adapter.resolve(REF)
    assert first.status is NativeSourceStatus.OK
    assert first.line is not None
    assert any(record.kind == "tool_result" for record in first.line.records)

    path.rename(path.with_suffix(".renamed"))
    missing = adapter.resolve(REF)
    assert missing.status is NativeSourceStatus.MISSING
    assert missing.line is None


class _RacingReader(ClaudeReader):
    """A real Claude reader whose hooks mutate the native source mid-read."""

    def __init__(self, *, source_for=None, decode=None, finish=None):
        super().__init__()
        self._hooks = {"source_for": source_for, "decode": decode, "finish": finish}

    def source_for(self, path):
        if self._hooks["source_for"] is not None:
            self._hooks["source_for"]()
        return super().source_for(path)

    def decode(self, source, obj, ctx, locator):
        if self._hooks["decode"] is not None:
            self._hooks["decode"]()
        return super().decode(source, obj, ctx, locator)

    def finish_cycle(self, evidence):
        if self._hooks["finish"] is not None:
            self._hooks["finish"]()
        return super().finish_cycle(evidence)


def _race(tmp_path: Path, **hooks):
    """old_source adapter whose reader runs hooks; the control read is un-hooked and OK."""
    path, reference, adapter = old_source(tmp_path)
    adapter.readers = {"claude": _RacingReader(**{k: v(path) for k, v in hooks.items()})}
    return path, reference, adapter


def _touch(path: Path) -> None:
    info = path.stat()
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 5_000_000_000))


def test_source_whose_path_cannot_be_stamped_is_unreadable_and_charged_nothing(tmp_path: Path):
    path, reference, adapter = old_source(tmp_path)
    assert adapter.resolve(REF).status is NativeSourceStatus.OK
    # A regular file in the middle of the path is not a missing source: its stat fails.
    blocked = NativeSourceAdapter(
        {REF: replace(reference, path=path / "not-a-directory")},
        adapter.readers,
        adapter.approved_roots,
        adapter.scoper,
        NOW,
    )
    result = blocked.resolve(REF)
    assert result.status is NativeSourceStatus.UNREADABLE and result.line is None
    assert result.bytes_read == 0 and blocked.bytes_read == 0


def test_native_io_failure_while_decoding_is_unreadable_and_charges_examined_bytes(
    tmp_path: Path,
):
    path, reference, adapter = old_source(tmp_path)
    assert adapter.resolve(REF).status is NativeSourceStatus.OK

    def fail(_path):
        calls = []

        def hook():
            calls.append(1)
            if len(calls) == 2:
                raise OSError("native storage vanished")

        return hook

    adapter = NativeSourceAdapter(
        {REF: reference},
        {"claude": _RacingReader(decode=fail(path))},
        adapter.approved_roots,
        adapter.scoper,
        NOW,
    )
    result = adapter.resolve(REF)
    first_row = len(path.read_bytes().splitlines(keepends=True)[0])
    second_row = len(path.read_bytes().splitlines(keepends=True)[1])
    assert result.status is NativeSourceStatus.UNREADABLE and result.line is None
    assert result.bytes_read == adapter.bytes_read == first_row + second_row
    assert not adapter.observed_references


def test_ownerless_duplicate_native_record_is_identity_mismatch_and_not_attested(tmp_path: Path):
    from tests.accounting.readers.readers_fixtures import write_records

    uuid = "11111111-1111-4111-8111-111111111111"
    row = {**claude_tool_use("src1", "w1", "Bash", {"command": "cat deploy.log"}), "uuid": uuid}
    path = tmp_path / ".claude/projects/-work/src1.jsonl"
    write_records(path, [row])
    reference = NativeSourceReference(REF, "claude", "src1", path, cwd="/work/proj")
    adapter = NativeSourceAdapter(
        {REF: reference}, {"claude": ClaudeReader()}, (tmp_path,), Scoper(policy()), NOW
    )
    assert adapter.resolve(REF).status is NativeSourceStatus.OK  # one owner: attested

    # The same native record copied under another cwd has two owners; neither is guessed.
    write_records(path, [row, {**row, "cwd": "/work/elsewhere"}])
    contested = NativeSourceAdapter(
        {REF: reference}, {"claude": ClaudeReader()}, (tmp_path,), Scoper(policy()), NOW
    )
    result = contested.resolve(REF)
    assert result.status is NativeSourceStatus.IDENTITY_MISMATCH and result.line is None
    assert result.bytes_read == path.stat().st_size
    assert not contested.observed_references


def test_snapshot_is_not_cached_when_the_source_changes_during_the_read(tmp_path: Path):
    path, _, adapter = _race(tmp_path, finish=lambda p: lambda: _touch(p))
    first = adapter.resolve(REF)
    assert first.status is NativeSourceStatus.OK and first.bytes_read > 0
    charged = adapter.bytes_read
    # A cache entry would have served this at zero cost; the racing read left none.
    second = adapter.resolve(REF)
    assert second.status is NativeSourceStatus.OK and second.bytes_read > 0
    assert adapter.bytes_read == 2 * charged


def test_snapshot_is_not_cached_when_the_source_cannot_be_stamped_after_the_read(tmp_path: Path):
    def swap_parent_for_file(path: Path):
        def hook():
            parent = path.parent
            parent.rename(parent.with_name("moved-away"))
            parent.write_text("not a directory")

        return hook

    path, _, adapter = _race(tmp_path, finish=swap_parent_for_file)
    result = adapter.resolve(REF)
    assert result.status is NativeSourceStatus.OK and result.line is not None
    assert result.source_digest and adapter.observed_references[REF].observed_at_ms == NOW
    again = adapter.resolve(REF)  # nothing was cached for the unstampable path
    assert again.status is NativeSourceStatus.UNREADABLE


def test_source_truncated_while_the_read_starts_is_changed(tmp_path: Path):
    def truncate(path: Path):
        return lambda: path.write_bytes(b"")

    path, _, adapter = _race(tmp_path, source_for=truncate)
    size_before = path.stat().st_size
    result = adapter.resolve(REF)
    assert size_before > 0 and path.stat().st_size == 0
    assert result.status is NativeSourceStatus.CHANGED and result.line is None
    assert not adapter.observed_references


def test_reverification_charges_the_budget_and_bounds_a_source_that_keeps_changing(
    tmp_path: Path,
):
    def touch_on_first_decode(path: Path):
        fired = []

        def hook():
            if not fired:
                fired.append(1)
                _touch(path)

        return hook

    path, reference, adapter = _race(tmp_path, decode=touch_on_first_decode)
    size = path.stat().st_size
    adapter.byte_budget = 2 * size - 1  # enough to read once, not to read and re-verify
    bounded = adapter.resolve(REF)
    assert bounded.status is NativeSourceStatus.BOUNDED and bounded.line is None
    assert bounded.bytes_read == adapter.bytes_read == 2 * size

    # Positive control: the same race with room for one re-verification pass succeeds.
    path2, _, adapter2 = _race(tmp_path / "control", decode=touch_on_first_decode)
    adapter2.byte_budget = 2 * size
    assert adapter2.resolve(REF).status is NativeSourceStatus.OK


def test_unterminated_final_record_is_incomplete_and_never_attested(tmp_path: Path):
    path, _, adapter = old_source(tmp_path)
    assert adapter.resolve(REF).status is NativeSourceStatus.OK
    path.write_bytes(path.read_bytes().rstrip(b"\n"))
    size = path.stat().st_size
    adapter = NativeSourceAdapter(
        {REF: replace(adapter.references[REF], source_digest="", record_digests=())},
        adapter.readers,
        adapter.approved_roots,
        adapter.scoper,
        NOW,
    )
    result = adapter.resolve(REF)
    assert result.status is NativeSourceStatus.INCOMPLETE and result.line is None
    assert result.bytes_read == adapter.bytes_read == size
    assert not adapter.observed_references


def test_one_record_yielding_more_evidence_than_the_cap_is_bounded(tmp_path: Path):
    from tests.accounting.readers.readers_fixtures import write_records

    def two_calls(session: str) -> dict:
        row = claude_tool_use(session, "one", "Bash", {"command": "ls"})
        row["message"]["content"].append(
            {"type": "tool_use", "id": "two", "name": "Bash", "input": {"command": "pwd"}}
        )
        return row

    path = tmp_path / ".claude/projects/-work/src1.jsonl"
    write_records(path, [two_calls("src1")])
    reference = NativeSourceReference(REF, "claude", "src1", path, cwd="/work/proj")

    def adapter(limit: int) -> NativeSourceAdapter:
        return NativeSourceAdapter(
            {REF: reference},
            {"claude": ClaudeReader()},
            (tmp_path,),
            Scoper(policy()),
            NOW,
            max_records=limit,
        )

    assert adapter(2).resolve(REF).status is NativeSourceStatus.OK  # one row, two calls
    capped = adapter(1)
    result = capped.resolve(REF)
    assert result.status is NativeSourceStatus.BOUNDED and result.line is None
    assert result.bytes_read == capped.bytes_read == path.stat().st_size


def test_read_proof_rejects_ranges_whose_bytes_no_longer_exist_and_truncated_descriptors(
    tmp_path: Path,
):
    from ssgrep.accounting.readers.native_source import ReadProof, SourceReadError

    path = tmp_path / "proof.jsonl"
    path.write_bytes(b"safe\n")
    descriptor = os.open(path, os.O_RDONLY)
    try:
        before = os.fstat(descriptor)
        proof = ReadProof()
        proof.add(0, b"safe\nmore\n")  # claims ten bytes though only five were ever there
        # A whole-second change, visible even on filesystems with coarse timestamps.
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 5 * 10**9))
        charged: list[int] = []
        with pytest.raises(SourceReadError) as vanished:
            proof.verify(descriptor, before, charged.append)
        assert vanished.value.status is NativeSourceStatus.CHANGED
        assert sum(charged) == len(b"safe\n")  # the examined prefix was still charged

        shorter = ReadProof()
        shorter.add(0, b"safe\n")
        path.write_bytes(b"sa")
        charged.clear()
        with pytest.raises(SourceReadError) as truncated:
            shorter.verify(descriptor, before, charged.append)
        assert truncated.value.status is NativeSourceStatus.CHANGED
        assert charged == []  # rejected on identity before any re-read
    finally:
        os.close(descriptor)


def _claude_direct(tmp_path: Path, rows: list[dict], *, reader_cwd: str | None = None):
    """An org-authorized Claude source plus direct-record witnesses for each row's evidence."""
    from tests.accounting.readers.readers_fixtures import write_records
    from tests.accounting.readers.test_native_source_review_edges import CWD, authority

    path = write_records(tmp_path / "direct.jsonl", rows)
    reader = ClaudeReader()
    witnesses = [
        item for item in reader.read(reader.source_for(path)).evidence if item.kind == "tool_call"
    ]
    reference = NativeSourceReference("claude:s:ep:0", "claude", "s", path, cwd=reader_cwd or CWD)
    adapter = NativeSourceAdapter({}, {"claude": reader}, (tmp_path,), authority(), NOW)
    return path, witnesses, reference, adapter


@pytest.mark.parametrize(
    "case,status",
    [
        ("expired", NativeSourceStatus.EXPIRED),
        ("no_reader", NativeSourceStatus.UNSUPPORTED),
        ("reader_without_native_decode", NativeSourceStatus.UNSUPPORTED),
    ],
)
def test_direct_record_refuses_expired_or_unsupported_references_without_reading(
    tmp_path, case, status
):
    from tests.accounting.readers.test_native_source_review_edges import json_scene

    path, _, _, witness, reference, adapter = json_scene(tmp_path, "claude")
    control = adapter.read_record(
        reference, witness.native_identity, witness.digest, locator=witness.locator
    )
    assert control.status is NativeSourceStatus.OK
    charged = adapter.bytes_read
    if case == "expired":
        reference = replace(reference, expires_at_ms=adapter.now_ms)
    elif case == "no_reader":
        reference = replace(reference, runtime="unknown-runtime")
    else:
        reference = replace(reference, runtime="generic")
        adapter.readers = {**adapter.readers, "generic": object()}
    result = adapter.read_record(
        reference, witness.native_identity, witness.digest, locator=witness.locator
    )
    assert result.status is status and result.evidence is None
    assert result.bytes_read == 0 and adapter.bytes_read == charged


def test_direct_record_with_unparseable_target_row_is_unreadable_but_charges_the_scan(tmp_path):
    from tests.accounting.readers.test_native_source_review_edges import json_scene

    path, _, _, witness, reference, adapter = json_scene(tmp_path, "claude")
    offset = int(witness.locator.partition(":")[2])
    original = path.read_bytes()
    assert (
        adapter.read_record(
            reference, witness.native_identity, witness.digest, locator=witness.locator
        ).status
        is NativeSourceStatus.OK
    )
    corrupt = b"{not json\n"
    path.write_bytes(original[:offset] + corrupt + original[offset:])
    fresh = replace(adapter, bytes_read=0)
    result = fresh.read_record(
        reference, witness.native_identity, witness.digest, locator=witness.locator
    )
    assert result.status is NativeSourceStatus.UNREADABLE and result.evidence is None
    assert result.bytes_read == fresh.bytes_read == offset + len(corrupt)
    assert "not json" not in repr(result) and str(path) not in repr(result)


def test_direct_record_budget_stops_exactly_at_the_unaffordable_byte(tmp_path):
    from tests.accounting.readers.test_native_source_review_edges import json_scene

    path, _, originals, _, reference, adapter = json_scene(tmp_path, "claude")
    witness = next(item for item in originals if item.kind == "tool_result")
    offset = int(witness.locator.partition(":")[2])
    with path.open("rb") as stream:
        stream.seek(offset)
        record = stream.readline()
    total = offset + len(record)
    assert offset > 0

    exact = replace(adapter, bytes_read=0, byte_budget=total)
    ok = exact.read_record(
        reference, witness.native_identity, witness.digest, locator=witness.locator
    )
    assert ok.status is NativeSourceStatus.OK and ok.bytes_read == total

    in_record = replace(adapter, bytes_read=0, byte_budget=total - 1)
    capped = in_record.read_record(
        reference, witness.native_identity, witness.digest, locator=witness.locator
    )
    assert capped.status is NativeSourceStatus.BOUNDED and capped.evidence is None
    assert capped.bytes_read == in_record.bytes_read == total

    in_prefix = replace(adapter, bytes_read=0, byte_budget=offset - 1)
    short = in_prefix.read_record(
        reference, witness.native_identity, witness.digest, locator=witness.locator
    )
    assert short.status is NativeSourceStatus.BOUNDED and short.evidence is None
    assert short.bytes_read == in_prefix.bytes_read == offset - 1


def test_direct_record_owner_must_match_even_when_native_identity_is_session_independent(tmp_path):
    uuid = "11111111-1111-4111-8111-111111111111"
    row = claude_tool_use("foreign", "c", "Bash", {"command": "ls"}, cwd="/work/approved/project")
    rows = [{**row, "uuid": uuid}]
    path, (witness,), reference, adapter = _claude_direct(tmp_path, rows)
    assert witness.session_id == "foreign" and witness.origin_domain is not None
    owned = replace(reference, session_id="foreign")
    ok = adapter.read_record(
        owned, witness.native_identity, witness.digest, locator=witness.locator
    )
    assert ok.status is NativeSourceStatus.OK and ok.evidence == witness
    charged = adapter.bytes_read
    other = adapter.read_record(
        reference, witness.native_identity, witness.digest, locator=witness.locator
    )
    assert other.status is NativeSourceStatus.IDENTITY_MISMATCH and other.evidence is None
    assert (
        other.bytes_read == path.stat().st_size and adapter.bytes_read - charged == other.bytes_read
    )


def test_direct_record_whose_native_cwd_leaves_the_approved_root_is_out_of_scope(tmp_path):
    inside = claude_tool_use("s", "c", "Bash", {"command": "ls"}, cwd="/work/approved/project")
    outside = {**inside, "cwd": "/work/foreign"}
    path, (witness,), reference, adapter = _claude_direct(tmp_path, [inside])
    ok = adapter.read_record(
        reference, witness.native_identity, witness.digest, locator=witness.locator
    )
    assert ok.status is NativeSourceStatus.OK

    from tests.accounting.readers.readers_fixtures import write_records

    write_records(path, [outside])
    fresh = replace(adapter, bytes_read=0)
    result = fresh.read_record(
        reference, witness.native_identity, witness.digest, locator=witness.locator
    )
    assert result.status is NativeSourceStatus.OUT_OF_SCOPE and result.evidence is None
    assert result.bytes_read == fresh.bytes_read == path.stat().st_size


def test_direct_codex_embedded_parent_metadata_cannot_replace_the_rollout_owner(tmp_path):
    from tests.accounting.readers.readers_fixtures import codex_meta, write_records
    from tests.accounting.readers.test_native_source_review_edges import json_scene

    path, _, originals, _, reference, adapter = json_scene(tmp_path, "codex")
    witness = next(item for item in originals if item.kind == "tool_call")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    # A fork replays its parent's meta after its own; the parent's cwd is unauthorized.
    forked = [rows[0], codex_meta("parent-session", "/work/foreign"), *rows[1:]]
    write_records(path, forked)
    shifted = len(path.read_bytes().splitlines(keepends=True)[1])
    locator = f"offset:{int(witness.locator.partition(':')[2]) + shifted}"
    result = adapter.read_record(
        reference, witness.native_identity, witness.digest, locator=locator
    )
    assert result.status is NativeSourceStatus.OK and result.evidence is not None
    assert result.evidence.session_id == "s"
    assert result.evidence.payload["cwd"] == witness.payload["cwd"] == "/work/approved/project"


def test_direct_pi_record_without_a_session_header_is_not_attributed_to_the_session(tmp_path):
    from tests.accounting.readers.readers_fixtures import write_records
    from tests.accounting.readers.test_native_source_review_edges import json_scene

    path, _, originals, _, reference, adapter = json_scene(tmp_path, "pi")
    witness = next(item for item in originals if item.kind == "tool_result")
    control = adapter.read_record(
        reference, witness.native_identity, witness.digest, locator=witness.locator
    )
    assert control.status is NativeSourceStatus.OK and control.evidence == witness
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    write_records(path, rows[1:])  # same entries, owner header removed
    offset = len(path.read_bytes().splitlines(keepends=True)[0])
    fresh = replace(adapter, bytes_read=0)
    result = fresh.read_record(
        reference, witness.native_identity, witness.digest, locator=f"offset:{offset}"
    )
    assert result.status is NativeSourceStatus.IDENTITY_MISMATCH and result.evidence is None
    assert result.bytes_read == fresh.bytes_read == path.stat().st_size


def test_unindexable_episode_references_are_unresolved_without_opening_storage(tmp_path):
    from ssgrep.accounting.readers.native_source import index_reference

    class Forbidden:
        def __getattr__(self, name):
            raise AssertionError(f"storage touched: {name}")

    for ref in ("no-episode-marker", "x" * 600 + ":ep:0", 17):
        assert index_reference(cast(Any, ref), Forbidden()) == (
            None,
            NativeSourceStatus.UNRESOLVED,
        )


@pytest.mark.parametrize(
    "error", [OSError("disk"), ValueError("bad"), KeyError("k"), RuntimeError("r")]
)
def test_index_storage_errors_are_unresolved_not_exceptions(error):
    from ssgrep.accounting.readers.native_source import index_reference

    class Failing:
        closed = False

        def exists(self):
            return True

        def get_meta(self, name):
            return {"index_state": "ready", "last_index_time": "1"}[name]

        def rows(self, *args, **kwargs):
            raise error

        def close(self):
            self.closed = True

    repository = Failing()
    assert index_reference("claude:s:ep:0", repository) == (None, NativeSourceStatus.UNRESOLVED)
    assert not repository.closed  # a caller-owned repository is never closed here


def test_direct_codex_refuses_a_reader_that_reinterprets_the_header_owner(tmp_path):
    from ssgrep.accounting.readers.jsonl_runtimes import CodexReader
    from tests.accounting.readers.test_native_source_review_edges import json_scene

    class Reinterpreting(CodexReader):
        @staticmethod
        def _meta(ctx, payload):
            ctx["session"] = "reinterpreted-" + str(payload.get("id"))
            ctx["cwd"] = payload.get("cwd")

    path, reader, _, witness, reference, adapter = json_scene(tmp_path, "codex")
    control = adapter.read_record(
        reference, witness.native_identity, witness.digest, locator=witness.locator
    )
    assert control.status is NativeSourceStatus.OK and control.evidence == witness
    adapter.readers = {"codex": Reinterpreting(reader.env)}
    header_bytes = len(path.read_bytes().splitlines(keepends=True)[0])
    charged = adapter.bytes_read
    result = adapter.read_record(
        reference, witness.native_identity, witness.digest, locator="offset:0"
    )
    assert result.status is NativeSourceStatus.IDENTITY_MISMATCH and result.evidence is None
    assert result.bytes_read == header_bytes and adapter.bytes_read - charged == header_bytes


def test_direct_record_payload_cwd_must_independently_stay_inside_the_approved_root(tmp_path):
    uuid = "11111111-1111-4111-8111-111111111111"
    row = claude_tool_use("s", "c", "Bash", {"command": "ls"}, cwd="/work/approved/project")
    path, (witness,), reference, adapter = _claude_direct(tmp_path, [{**row, "uuid": uuid}])
    control = adapter.read_record(
        reference, witness.native_identity, witness.digest, locator=witness.locator
    )
    assert control.status is NativeSourceStatus.OK and control.evidence == witness

    class DriftingCwd(ClaudeReader):
        def decode(self, source, obj, ctx, locator):
            return [
                replace(item, payload={**item.payload, "cwd": "/work/foreign"})
                for item in super().decode(source, obj, ctx, locator) or []
            ]

    adapter.readers = {"claude": DriftingCwd()}
    charged = adapter.bytes_read
    result = adapter.read_record(
        reference, witness.native_identity, witness.digest, locator=witness.locator
    )
    assert result.status is NativeSourceStatus.OUT_OF_SCOPE and result.evidence is None
    assert result.bytes_read == path.stat().st_size == adapter.bytes_read - charged
