"""9.6: collection control facade: validation first, recompute from retained evidence only.

Everything runs through the real ``ssgrep`` binary or the real collector. "Before any file is
created" is observed as an empty (non-existent) data directory; "never reads changed live
sources" is observed by mutating a transcript after retention and proving no generation, evidence
record or ledger content changed with it, and with a spy on the readers' ``read``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from ssgrep.accounting import collector_read
from ssgrep.accounting.collector_types import CollectOptions
from tests.accounting import collector_scene as scene, verify_scenes as vs
from tests.accounting.reduce_builders import claude_prompt, ts
from tests.accounting.test_savings_cli import BINARY, Run, document

DAY = 12
CLOCK = scene.at(hour=9)


@pytest.fixture
def run(tmp_path: Path) -> Run:
    home = tmp_path / "home"
    home.mkdir()
    return Run(home, tmp_path / "data")


def data_dir_state(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*")) if root.exists() else []


# ---- validation before I/O -------------------------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [
        ("--backfill",),
        ("--backfill", "--since", "2026-10-01"),
        ("--backfill", "--until", "2026-10-02"),
        ("--backfill", "--since", "2026-10-01", "--until", "not-a-date"),
        ("--backfill", "--since", "2026-10-02", "--until", "2026-10-01"),
    ],
)
@pytest.mark.parametrize("enabled", [False, True], ids=["never-set-up", "enabled"])
def test_backfill_without_both_bounds_exits_2_before_any_file_is_created(
    run: Run, args: tuple[str, ...], enabled: bool
) -> None:
    if enabled:
        assert run("metrics", "setup", "--yes").returncode == 0
    before = run.files()
    for flags in ((), ("--json",)):
        result = run("metrics", "collect", *args, *flags)
        assert result.returncode == 2, (result.stdout, result.stderr)
        assert "Traceback" not in result.stderr
    assert run.files() == before  # nothing created, nothing touched
    if not enabled:
        assert not run.data.exists()  # not even the data directory
    assert not (run.data / "accounting" / "personal-local" / "worker").exists()
    assert not (run.data / "accounting" / "personal-local" / "collector").exists()


def test_a_valid_backfill_is_the_positive_twin_it_proceeds_past_validation(run: Run) -> None:
    result = run(
        "metrics", "collect", "--backfill", "--since", "2026-10-01", "--until", "2026-10-02"
    )
    assert result.returncode == 1  # reaches the disabled check: validation itself passed
    assert "not enabled" in result.stderr + result.stdout


def test_the_library_also_refuses_a_backfill_without_an_interval_before_any_io(
    tmp_path: Path,
) -> None:
    vs.enable()
    vs.start_capture()
    before = vs.tree(tmp_path / "data")
    for options in (
        CollectOptions(backfill=True),
        CollectOptions(backfill=True, since="2026-10-01T00:00:00Z"),
        CollectOptions(backfill=True, until="2026-10-02T00:00:00Z"),
    ):
        with pytest.raises(Exception, match="requires_interval|needs both"):
            scene.collect(scene.at(hour=10), options)
    after = vs.tree(tmp_path / "data")
    assert {k: v for k, v in after.items() if "worker/" not in k} == {
        k: v for k, v in before.items() if "worker/" not in k
    }
    assert vs.checkpoint() is None  # no checkpoint, no snapshot: no counters changed
    assert not (vs.accounting_dir() / "snapshots").exists()


# ---- recompute -------------------------------------------------------------------------------


def credit_scene(root: Path) -> None:
    vs.enable(pin=True)
    vs.start_capture()
    for n in (0, 1):
        vs.pair(vs.projects(root), n, CLOCK)


def test_recompute_with_expired_evidence_reports_unavailable_periods_and_keeps_totals(
    tmp_path: Path,
) -> None:
    credit_scene(tmp_path)
    scene.collect(scene.at(hour=10))
    live_total = vs.facts()
    horizon_now = scene.at(DAY) + __import__("datetime").timedelta(days=45)
    scene.collect(horizon_now)  # retention freezes the day-12 conversations
    frozen = vs.facts()
    assert scene.committed().retention["frozen_roots"] >= 1
    assert not vs.evidence_identities() or len(vs.evidence_identities()) < len(live_total.evidence)

    # the live source CHANGES after retention: recompute must never look at it
    pool = vs.projects(tmp_path)
    marker = "MUTATED-AFTER-RETENTION"
    vs.put(pool, "intruder", [claude_prompt("intr", marker, uuid="ui", ts=ts(12, 9))], CLOCK)
    for path in pool.rglob("src0.jsonl"):
        path.write_text(path.read_text() + "\n")  # a changed live source
    reads: list[str] = []
    real = collector_read.ClaudeReader.read

    def spy(self, source, state, *args, **kwargs):  # type: ignore[no-untyped-def]
        reads.append(str(source.path))
        return real(self, source, state, *args, **kwargs)

    collector_read.ClaudeReader.read = spy  # type: ignore[method-assign]
    try:
        options = CollectOptions(
            recompute=True, since="2026-10-01T00:00:00Z", until="2026-10-20T00:00:00Z"
        )
        result = scene.collect(horizon_now, options)
    finally:
        collector_read.ClaudeReader.read = real  # type: ignore[method-assign]
    assert reads == []  # no live source was read at all
    assert result.mode == "recompute" and result.state == "published"
    periods = dict(entry.split("=", 1) for entry in result.unavailable_periods)
    assert periods and set(periods.values()) == {"evidence_expired"}
    assert f"2026-10-{DAY}" in periods  # the expired period is named, with its reason
    after = vs.facts()
    assert after.ledger == frozen.ledger  # closed totals preserved, not recomputed downward
    assert not any(marker in repr(e) for e in vs.evidence_identities())
    assert "intruder" not in {e[1] for e in vs.evidence_identities()}
    assert not (tmp_path / "data" / "lancedb").exists()  # never rebuilds the search index


def test_the_reader_spy_sees_a_normal_collection_read_the_sources(tmp_path: Path) -> None:
    """Positive twin: the spy used above really fires when a source is read."""
    credit_scene(tmp_path)
    reads: list[str] = []
    real = collector_read.ClaudeReader.read

    def spy(self, source, state, *args, **kwargs):  # type: ignore[no-untyped-def]
        reads.append(str(source.path))
        return real(self, source, state, *args, **kwargs)

    collector_read.ClaudeReader.read = spy  # type: ignore[method-assign]
    try:
        scene.collect(scene.at(hour=10))
    finally:
        collector_read.ClaudeReader.read = real  # type: ignore[method-assign]
    assert len(reads) >= 4 and all(path.endswith(".jsonl") for path in reads)


def test_recompute_over_retained_evidence_reads_no_source_and_changes_no_ledger_content(
    tmp_path: Path,
) -> None:
    credit_scene(tmp_path)
    scene.collect(scene.at(hour=10))
    base = vs.facts()
    vs.pair(vs.projects(tmp_path), 2, CLOCK)  # a NEW live record the recompute must not import
    result = scene.collect(
        scene.at(hour=11),
        CollectOptions(recompute=True, since="2026-10-12T00:00:00Z", until="2026-10-13T00:00:00Z"),
    )
    assert result.mode == "recompute" and result.unavailable_periods == ()  # nothing expired
    after = vs.facts()
    assert after.evidence == base.evidence  # the new transcript was not read
    assert after.ledger == base.ledger
    assert scene.report(result).new_evidence == 0 and scene.report(result).bytes_read == 0
    follow = scene.collect(scene.at(hour=12))  # a normal collection imports it afterwards
    assert scene.report(follow).new_evidence > 0 and vs.facts().evidence != base.evidence


def test_recompute_through_the_cli_never_creates_an_index_or_touches_sources(
    run: Run, tmp_path: Path
) -> None:
    assert run("metrics", "setup", "--yes").returncode == 0
    transcript = tmp_path / "claude" / "projects" / "-work" / "s.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_text("")
    stamp = os.stat(transcript)
    result = run("metrics", "collect", "--recompute", "--since", "2020-01-01", "--json")
    assert result.returncode == 0, result.stderr
    assert document(result)["data"]["data"]["mode"] == "recompute"
    assert not (run.data / "lancedb").exists()  # no index appeared
    assert os.stat(transcript).st_mtime_ns == stamp.st_mtime_ns and transcript.read_text() == ""


# ---- an unavailable index does not block usage collection --------------------------------------


def test_missing_index_does_not_block_collection_and_is_reported_not_hidden(tmp_path: Path) -> None:
    credit_scene(tmp_path)
    assert not (tmp_path / "data" / "lancedb").exists()  # there is no search index at all
    result = scene.collect(scene.at(hour=10))
    assert result.state == "published" and scene.report(result).new_evidence > 0
    envelope = scene.current()
    assert scene.coverage(envelope)["index"] == "unavailable"  # said out loud ...
    assert (
        scene.window(envelope, "lifetime")["observations"]["searches"]["value"] == 3
    )  # ... usage counted
    assert not (tmp_path / "data" / "lancedb").exists()  # and collection never created one


def test_present_index_directory_is_never_opened_or_modified_by_collection(
    tmp_path: Path,
) -> None:
    credit_scene(tmp_path)
    index = tmp_path / "data" / "lancedb"
    index.mkdir(parents=True)
    (index / "table.lance").write_bytes(b"opaque index bytes")
    stamps = {p: p.stat().st_mtime_ns for p in index.rglob("*")}
    scene.collect(scene.at(hour=10))
    scene.collect(scene.at(hour=11), CollectOptions(recompute=True))
    assert scene.coverage(scene.current())["index"] == "available"
    assert {p: p.stat().st_mtime_ns for p in index.rglob("*")} == stamps
    assert (index / "table.lance").read_bytes() == b"opaque index bytes"


def test_a_corrupt_index_directory_does_not_stop_usage_collection(tmp_path: Path) -> None:
    credit_scene(tmp_path)
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "lancedb").write_text("not a directory at all")  # unusable index
    result = scene.collect(scene.at(hour=10))
    assert result.state == "published" and scene.report(result).complete
    assert scene.coverage(scene.current())["index"] == "unavailable"
    assert scene.window(scene.current(), "lifetime")["observations"]["searches"]["value"] == 3


def test_cli_collect_succeeds_with_no_index(run: Run, tmp_path: Path) -> None:
    assert run("metrics", "setup", "--yes").returncode == 0
    result = run("metrics", "collect", "--json")
    assert result.returncode == 0, result.stderr
    assert document(result)["data"]["state"] == "published"
    assert not (run.data / "lancedb").exists()
    assert subprocess.run([str(BINARY), "--help"], capture_output=True, timeout=60).returncode == 0
