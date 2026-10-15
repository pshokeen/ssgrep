"""reduce_io: read-only load of real stores + journal, then one deterministic reduction."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ssgrep.accounting import fingerprint, journal
from ssgrep.accounting.contracts import Attribution
from ssgrep.accounting.frequency import FrequencyView
from ssgrep.accounting.readers.claude import ClaudeReader
from ssgrep.accounting.reduce_io import load_evidence, reduce_stored_profile
from tests.accounting.readers_fixtures import build_home, write_records
from tests.accounting.reduce_builders import (
    MCP,
    PROFILE,
    card,
    claude_prompt,
    claude_tool_result,
    claude_tool_use,
    results_json,
    ts,
)

PATH = "services/billing/limits.py"
FACT = f"deploy failed: ERR_QUOTA_EXCEEDED raised in {PATH}"


def tree_digest(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


@pytest.fixture(autouse=True)
def _fresh_state():
    journal.close_all()
    fingerprint.forget_keys()
    yield
    journal.close_all()
    fingerprint.forget_keys()


@pytest.fixture
def stores(tmp_path: Path) -> dict[str, str]:
    claude = tmp_path / ".claude"
    write_records(
        claude / "projects" / "-w" / "src1.jsonl",
        [
            claude_prompt("src1", "why does the deploy fail", uuid="u1", ts=ts(1, 10)),
            claude_tool_use(
                "src1", "w1", "Bash", {"command": "cat deploy.log"}, ts=ts(1, 10, 0, 5)
            ),
            claude_tool_result("src1", "w1", FACT, ts=ts(1, 10, 0, 6)),
        ],
    )
    write_records(
        claude / "projects" / "-w" / "cons.jsonl",
        [
            claude_prompt("cons", "fix the deploy", uuid="u1", ts=ts(12, 9)),
            claude_tool_use("cons", "t1", MCP, {"query": "why did the deploy fail"}, ts=ts(12, 10)),
            claude_tool_result(
                "cons", "t1", results_json(card("claude:src1:ep:0", FACT)), ts=ts(12, 10, 0, 2)
            ),
            claude_tool_use("cons", "t2", "Bash", {"command": f"vim {PATH}"}, ts=ts(12, 10, 1)),
            claude_tool_result("cons", "t2", "ok", ts=ts(12, 10, 1, 2)),
        ],
    )
    return {"HOME": str(tmp_path), "CLAUDE_CONFIG_DIR": str(claude)}


def test_reduce_stored_profile_reads_real_files_and_never_writes(
    stores: dict[str, str], tmp_path: Path
) -> None:
    before = tree_digest(tmp_path)
    reduction, loaded = reduce_stored_profile(PROFILE, stores, frequency=FrequencyView.unknown())
    assert tree_digest(tmp_path) == before  # no checkpoint, quarantine, or key was created
    assert loaded.sources_read["claude"] == 2
    assert loaded.events == []
    assert len(reduction.invocations) == 1
    assert reduction.invocations[0].cls.value == "transcript_only"  # no journal: no join, no guess
    assert reduction.executions == ()
    assert reduction.coverage["recorded_executions"] == 0
    # Retained transcripts never become an implicit rarity corpus.
    assert reduction.candidates
    assert reduction.coverage["exclusion.rarity_unknown"] >= 1


def test_journal_events_written_by_the_real_journal_join_the_transcript(
    stores: dict[str, str], tmp_path: Path
) -> None:
    from tests.accounting.reduce_builders import event

    key = fingerprint.load_key(PROFILE, create=True)
    text = "why did the deploy fail"
    for item in event("zz-1", ts(12, 10, 0, 1), text=text, cwd="/work/proj"):
        assert journal.append(PROFILE, item)
    # the reducer's fixtures key fingerprints with their own constant; re-key with the real key
    from dataclasses import replace

    from ssgrep.accounting.contracts import Operation

    real = fingerprint.request_fingerprint(key, Operation.SEARCH, query=text)
    journal.close_all()
    for item in event("zz-2", ts(12, 10, 0, 1), text=text, cwd="/work/proj"):
        assert journal.append(PROFILE, replace(item, fingerprint=real))
    reduction, loaded = reduce_stored_profile(PROFILE, stores)
    by_id = {e.execution_id: e for e in reduction.executions}
    assert loaded.journal_segments >= 1 and len(loaded.events) == 6
    assert by_id["zz-2"].attribution is Attribution.INFERRED_UNIQUE  # real key, real journal
    assert by_id["zz-1"].attribution is Attribution.UNATTRIBUTED  # foreign key never joins
    assert reduction.executions[0].source in ("both", "journal")


def test_missing_index_never_falls_back_to_retained_corpus_rarity(
    stores: dict[str, str],
) -> None:
    reduced, loaded = reduce_stored_profile(PROFILE, stores)
    assert loaded.evidence
    assert reduced.coverage["exclusion.rarity_unknown"] >= 1


def test_loader_restricts_canonical_subtree_and_reports_bad_record_coverage(tmp_path: Path) -> None:
    home = build_home(tmp_path)
    target = home.paths["claude"] / "final" / "selected.jsonl"
    write_records(target, [claude_prompt("selected", "visible_fact", ts=ts())])
    with target.open("ab") as stream:
        stream.write(b"not-json\n")
    alias = tmp_path / "selected-alias"
    alias.symlink_to(target)
    loaded = load_evidence(home.env, only_under=[alias])
    assert {item.session_id for item in loaded.evidence} == {"selected"}
    assert loaded.sources_read["claude"] == 1
    assert sum(loaded.sources_read.values()) == 1
    assert sum(loaded.gaps.values()) == 1
    rejected = load_evidence(home.env, only_under=[target], include=lambda source: False)
    assert rejected.evidence == [] and sum(rejected.sources_read.values()) == 0


def test_loader_counts_source_read_error_without_discarding_other_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = build_home(tmp_path)
    folder = home.paths["claude"] / "read-fault"
    missing = folder / "missing.jsonl"
    write_records(missing, [claude_prompt("missing", "first fact", ts=ts())])
    write_records(folder / "retained.jsonl", [claude_prompt("retained", "second fact", ts=ts())])
    before = load_evidence(home.env, only_under=[folder])
    assert {item.session_id for item in before.evidence} == {"missing", "retained"}
    assert before.unreadable_sources == 0
    original = ClaudeReader.read

    def lose_first_source(self, source, *args, **kwargs):
        if source.path == missing:
            raise OSError("source disappeared after discovery")
        return original(self, source, *args, **kwargs)

    monkeypatch.setattr(ClaudeReader, "read", lose_first_source)
    after = load_evidence(home.env, only_under=[folder])
    assert after.unreadable_sources == 1
    assert {item.session_id for item in after.evidence} == {"retained"}
    assert after.sources_read == before.sources_read
