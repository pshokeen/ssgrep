"""Source witness positions belong to the exact verified timeline snapshot."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ssgrep.accounting.evidence_tokens import Token
from ssgrep.accounting.sources import Consumer, Resolver
from ssgrep.accounting.timeline import build_timelines
from tests.accounting.readers_fixtures import jsonl
from tests.accounting.reduce_builders import (
    claude_evidence,
    claude_prompt,
    claude_tool_result,
    claude_tool_use,
    ts,
)
from tests.accounting.test_native_source import NOW, REF, old_source
from tests.accounting.test_reduce_evidence import FACT, PATH

CONSUMER = Consumer("claude:consumer", ("claude", "consumer"), 1, NOW + 172_800_000)


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
    held = resolver.resolve(Token("path", PATH), (REF,), CONSUMER)
    assert any(atom.atom.origin_event.endswith("result:retained-3") for atom in held.immediate)
    actual = resolver.resolve(Token("path", PATH), (alias,), CONSUMER)
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
    first = resolver.resolve(Token("path", PATH), (REF,), CONSUMER)
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
    drift = resolver.resolve(Token("path", PATH), (alias,), CONSUMER)
    assert ("source_digest_drift", alias) in drift.reasons
    spent = adapter.bytes_read
    assert spent > before
    repeated = resolver.resolve(Token("path", PATH), (alias,), CONSUMER)
    assert ("source_digest_drift", alias) in repeated.reasons
    assert adapter.bytes_read == spent
