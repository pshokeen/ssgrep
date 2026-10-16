"""Privacy purge: real collector state in, real files out.

The scene is produced by the real ``collect_once`` (native Claude transcripts for two workspaces
dated before and after the cutoff, a real journal, a pinned frequency view); export bookkeeping
is real registry/spool/delivery files written through the production modules. Assertions are on
outcomes: which files exist, what a fresh ``ledger_rows``/snapshot read shows, byte-identical
trees, and what the report says in each of its four tiers.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from ssgrep.accounting.collect import frequency as collector_frequency, store as collector_store
from ssgrep.accounting.collect.state import REDUCTION, State, load_state
from ssgrep.accounting.export import (
    acks,
    ids as export_ids,
    projection as export_build,
    records as export_records,
    registry as export_registry,
    spool as export_spool,
)
from ssgrep.accounting.export.records import project_candidates, root_key_of
from ssgrep.accounting.export.registry import (
    Registry,
    load_registry,
    mark_acked,
    mark_queued,
    plan_revisions,
)
from ssgrep.accounting.ledger.ledger_contracts import DestinationPolicy, ProfilePolicy, Signal
from ssgrep.accounting.privacy import (
    policy,
    purge,
    purge as purge_select,
    purge as purge_snapshot,
    purge_files,
    purge_state,
    storage,
    storage as plainfile,
)
from ssgrep.accounting.privacy.purge import BACKEND_ERASED, PurgeError, parse_selection
from ssgrep.accounting.serve.lease import Lease
from ssgrep.accounting.views import service, snapshots
from ssgrep.accounting.views.snapshots import pinned_snapshot_ids
from tests.accounting.collect.collector_scene import (
    accounting_dir,
    at,
    collect,
    execution,
    install_key,
    journal_events,
    write_transcript,
)
from tests.accounting.conftest import build_evidence
from tests.accounting.export.snapshots_driver import make_envelope
from tests.accounting.ledger.reduce_builders import (
    MCP,
    PROFILE,
    card,
    claude_prompt,
    claude_tool_result,
    claude_tool_use,
    rare_view,
    results_json,
    source_session,
    ts,
)
from tests.accounting.views.snapshot_builders import enable_policy

DEST = "acme-org"
OTHER = "other-org"
ENDPOINT = "https://collector.example.test"
CUTOFF = "2026-10-10"
ALPHA, BETA = "/work/alpha", "/work/beta"
# (session, workspace, UTC day, path token, error-code token)
SCENES = [
    ("a1", ALPHA, 5, "services/alpha/limits.py", "ERR_ALPHA_ONE"),
    ("b1", BETA, 6, "services/beta/limits.py", "ERR_BETA_ONE"),
    ("a2", ALPHA, 15, "services/alpha/other.py", "ERR_ALPHA_TWO"),
]
COLLECT_AT = at(21, 11)
PURGE_AT = datetime(2026, 10, 22, 9, 0, 0, tzinfo=UTC)


# ---- scene ------------------------------------------------------------------------------------


def _with_cwd(records: list[dict[str, Any]], cwd: str) -> list[dict[str, Any]]:
    return [{**r, "cwd": cwd} for r in records]


def _consumer(session: str, fact: str, day: int, cwd: str, path: str) -> list[dict[str, Any]]:
    return _with_cwd(
        [
            claude_prompt(session, "fix the deploy", uuid=f"u-{session}", ts=ts(day, 9)),
            claude_tool_use(
                session,
                "t1",
                MCP,
                {"query": "why did the deploy fail"},
                ts=ts(day, 10),
                message_id=f"m-{session}",
                cwd=cwd,
            ),  # fmt: skip
            claude_tool_result(
                session, "t1", results_json(card("claude:src:ep:0", fact)), ts=ts(day, 10, 0, 2)
            ),
            claude_tool_use(
                session,
                "t2",
                "Bash",
                {"command": f"vim {path}"},
                ts=ts(day, 10, 1),
                message_id=f"n-{session}",
                cwd=cwd,
            ),  # fmt: skip
            claude_tool_result(session, "t2", "ok", ts=ts(day, 10, 1, 2)),
        ],
        cwd,
    )


def build_scene(tmp_path: Path, *, orphan: bool = False) -> None:
    """Collect two workspaces (alpha: day 5 + day 15, beta: day 6) with a real collector."""
    enable_policy()
    install_key()
    rare = [token for scene in SCENES for token in scene[3:]]
    collector_frequency.pin(PROFILE, rare_view(*rare))
    journal_events(execution("n-start", ts(1, 8), text="capture begins", transport="cli"))
    for session, cwd, day, path, code in SCENES:
        fact = f"deploy failed: {code} raised in {path}"
        source = _with_cwd(source_session(f"src-{session}", fact, when_day=day - 1), cwd)
        write_transcript(tmp_path, f"src-{session}", source, at(20))
        write_transcript(tmp_path, session, _consumer(session, fact, day, cwd, path), at(20))
        journal_events(
            execution(f"n-{session}", ts(day, 10, 0, 1), text="why did the deploy fail", cwd=cwd)
        )
    if orphan:  # a conversation whose evidence never names a working directory
        bare = [
            {k: v for k, v in claude_prompt("orph", "hello", ts=ts(4, 9)).items() if k != "cwd"}
        ]
        write_transcript(tmp_path, "orph", bare, at(20))
    result = collect(COLLECT_AT)
    assert result.state == "published", result
    _age_old_journal_segments()


def _age_old_journal_segments() -> None:
    """Quiet files: every segment whose events predate the cutoff was last written long ago."""
    journal_dir = accounting_dir() / "journal"
    old = datetime(2026, 10, 7, tzinfo=UTC).timestamp()
    for path in journal_dir.iterdir():
        if CUTOFF > max(_event_days(path)):
            os.utime(path, (old, old))


def _event_days(path: Path) -> list[str]:
    import json

    return [json.loads(line)["ts_utc"][:10] for line in path.read_text().splitlines() if line]


def tree(root: Path) -> dict[str, tuple[Any, ...]]:
    """Every path under ``root``: mode, size, mtime and content digest (dirs: mode, mtime)."""
    out: dict[str, tuple[Any, ...]] = {}
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        key = str(path.relative_to(root))
        if path.is_file() and not path.is_symlink():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            out[key] = (info.st_mode, info.st_size, info.st_mtime_ns, digest)
        else:
            out[key] = (info.st_mode, info.st_mtime_ns)
    return out


def data_root(tmp_path: Path) -> Path:
    return tmp_path / "data"


def select(before: str = CUTOFF, workspace: str | None = None, profile: str = PROFILE):
    return parse_selection(
        profile=profile,
        before=before,
        workspace=workspace,
        policy=policy.load_profile_policy(profile),
    )


def periods() -> set[str]:
    rows = export_records.ledger_rows(PROFILE)
    assert rows is not None
    return {p.period for p in rows.periods}


def evidence_by_session() -> dict[str, int]:
    checkpoint = collector_store.load_checkpoint(PROFILE)
    assert checkpoint is not None
    counts: dict[str, int] = {}
    for item in collector_store.iter_evidence(PROFILE, checkpoint.segments):
        counts[item.session_id] = counts.get(item.session_id, 0) + 1
    return counts


def current_envelope():
    envelope, state, reason = service.read_snapshot(PROFILE)
    assert envelope is not None, (state, reason)
    return envelope


def generation_names() -> list[str]:
    return sorted(p.name for p in (accounting_dir() / "snapshots").glob("gen-*.json"))


# ---- selection and validation -----------------------------------------------------------------


def test_missing_before_is_a_usage_error_before_anything_is_read(tmp_path: Path) -> None:
    with pytest.raises(PurgeError) as raised:
        parse_selection(profile=PROFILE, before=None, workspace=None, policy=ProfilePolicy())
    assert (raised.value.condition, raised.value.exit_code) == ("invalid_interval", 2)
    for bad in ("yesterday", "2026-13-01", "10/10/2026", ""):
        with pytest.raises(PurgeError) as raised:
            parse_selection(profile=PROFILE, before=bad, workspace=None, policy=ProfilePolicy())
        assert raised.value.exit_code == 2
    assert not data_root(tmp_path).exists()


def test_cutoff_is_the_utc_day_and_data_on_the_cutoff_day_is_kept() -> None:
    day = parse_selection(
        profile=PROFILE, before="2026-10-10", workspace=None, policy=ProfilePolicy()
    )
    late = parse_selection(
        profile=PROFILE, before="2026-10-10T23:59:59Z", workspace=None, policy=ProfilePolicy()
    )
    shifted = parse_selection(
        profile=PROFILE, before="2026-10-11T01:00:00+05:00", workspace=None, policy=ProfilePolicy()
    )
    assert day.cutoff_day == late.cutoff_day == shifted.cutoff_day == "2026-10-10"


def test_workspace_must_be_absolute_canonical_and_inside_an_approved_root(tmp_path: Path) -> None:
    real = tmp_path / "real"
    (real / "sub").mkdir(parents=True)
    link = tmp_path / "link"
    link.symlink_to(real)
    personal = ProfilePolicy()
    chosen = parse_selection(
        profile=PROFILE, before=CUTOFF, workspace=str(link / "sub"), policy=personal
    )
    assert chosen.workspace == str((real / "sub").resolve())  # symlinks resolved
    for bad in ("relative/dir", "/", ""):
        with pytest.raises(PurgeError) as raised:
            parse_selection(profile=PROFILE, before=CUTOFF, workspace=bad, policy=personal)
        assert (raised.value.condition, raised.value.exit_code) == ("invalid_workspace", 2)
    org = ProfilePolicy(profile_id="acme", roots=(str(real),))
    inside = parse_selection(profile="acme", before=CUTOFF, workspace=str(link / "sub"), policy=org)
    assert inside.workspace == str((real / "sub").resolve())
    sibling = tmp_path / "real-evil"  # a lexical prefix of the root, not contained in it
    sibling.mkdir()
    for outside in (sibling, tmp_path):
        with pytest.raises(PurgeError) as raised:
            parse_selection(profile="acme", before=CUTOFF, workspace=str(outside), policy=org)
        assert (raised.value.condition, raised.value.exit_code) == ("workspace_not_in_scope", 2)
    with pytest.raises(PurgeError) as raised:  # an org profile without roots accepts nothing
        parse_selection(
            profile="acme",
            before=CUTOFF,
            workspace=str(real),
            policy=ProfilePolicy(profile_id="acme"),
        )
    assert raised.value.condition == "workspace_not_in_scope"
    # An invalid (relative) approved root is skipped without ignoring the valid one.
    team = ProfilePolicy(profile_id="team", roots=("relative", ALPHA))
    chosen = purge_select.parse_selection(
        profile="team", before="2026-10-10", workspace=ALPHA + "/child", policy=team
    )
    assert chosen.workspace == ALPHA + "/child"
    with pytest.raises(PurgeError) as caught:
        purge_select.parse_selection(
            profile="team", before="2026-10-10", workspace="/work/beta", policy=team
        )
    assert caught.value.condition == "workspace_not_in_scope"


def test_hostile_profile_ids_are_usage_errors() -> None:
    for bad in ("../etc", "A/B", "x" * 40):
        with pytest.raises(PurgeError) as raised:
            parse_selection(profile=bad, before=CUTOFF, workspace=None, policy=ProfilePolicy())
        assert raised.value.exit_code == 2


# ---- dry run ----------------------------------------------------------------------------------


def test_dry_run_with_no_accounting_data_reports_nothing_and_creates_nothing(
    tmp_path: Path,
) -> None:
    report = purge.dry_run(select())
    assert report.state == "nothing_to_purge" and report.empty and not report.performed
    assert not data_root(tmp_path).exists()


def test_dry_run_changes_nothing_and_predicts_exactly_what_a_real_run_deletes(
    tmp_path: Path,
) -> None:
    build_scene(tmp_path)
    before = tree(data_root(tmp_path))
    preview = purge.dry_run(select())
    assert tree(data_root(tmp_path)) == before  # no file, directory, lease or mtime touched
    assert preview.state == "dry_run" and not preview.performed
    assert preview.local_deleted.evidence_records > 0
    assert (
        preview.local_deleted.ledger_period_rows == 2 and preview.local_deleted.receipt_files == 2
    )
    assert preview.local_deleted.journal_segments == 3  # n-start, n-a1, n-b1 (all old and quiet)
    assert preview.retained.journal_segments_retained == 1  # n-a2 holds newer events

    done = purge.execute(select(), now=PURGE_AT)
    assert done.performed and done.state == "purged"
    assert replace(done.local_deleted, delivery_pins_released=0) == replace(
        preview.local_deleted, delivery_pins_released=0
    )


def test_dry_run_mentions_a_running_worker_through_the_read_only_lease_view(tmp_path: Path) -> None:
    build_scene(tmp_path)
    with Lease.acquire(PROFILE, "serve"):
        before = tree(data_root(tmp_path))
        report = purge.dry_run(select())
        assert tree(data_root(tmp_path)) == before
    assert report.worker is not None and report.worker.running and report.worker.kind == "serve"


# ---- real purge, profile-wide -----------------------------------------------------------------


def test_profile_purge_removes_old_totals_everywhere_and_keeps_newer_data(tmp_path: Path) -> None:
    build_scene(tmp_path)
    assert periods() == {"2026-10-05", "2026-10-06", "2026-10-15"}
    assert set(evidence_by_session()) >= {"a1", "b1", "a2", "src-a1", "src-b1", "src-a2"}
    receipts_before = sorted(p.stem for p in (accounting_dir() / "receipts").iterdir())
    assert len(receipts_before) == 3

    report = purge.execute(select(), now=PURGE_AT)

    assert periods() == {"2026-10-15"}  # purged-day totals vanish from a fresh ledger read
    assert set(evidence_by_session()) == {"a2", "src-a2"}  # evidence of old conversations is gone
    assert len(list((accounting_dir() / "receipts").iterdir())) == 1
    history = (accounting_dir() / "history" / "periods.json").read_text()
    assert "2026-10-05" not in history and "2026-10-06" not in history and "2026-10-15" in history
    envelope = current_envelope()  # the next snapshot read works and carries no old aggregate
    assert envelope.generation == 2 and generation_names() == [
        f"gen-0000000002-{envelope.snapshot_id}.json"
    ]
    assert "shared_frequency_authority_invalidated_requires_explicit_rebuild" in report.notes
    for path in accounting_dir().rglob("*"):  # no byte of purged evidence survives elsewhere
        if path.is_file():
            raw = path.read_bytes()
            assert b"services/alpha/limits.py" not in raw and b"ERR_ALPHA_ONE" not in raw, path
            assert b"services/beta/limits.py" not in raw and b"ERR_BETA_ONE" not in raw, path
    assert not any(
        b"services/alpha/other.py"
        in p.read_bytes()  # newer IDs/totals survive, not copied plaintext
        for p in (accounting_dir() / "collector" / "evidence").iterdir()
    )
    remaining = {p.name for p in (accounting_dir() / "journal").iterdir()}
    assert len(remaining) == 1 and report.local_deleted.journal_segments == 3
    assert report.retained.journal_segments_retained == 1
    assert report.local_deleted.snapshot_generations == 1
    assert report.local_deleted.receipt_files == 2 and report.local_deleted.ledger_period_rows == 2
    assert report.local_deleted.sealed_days == 3  # 10-01 (journal-only), 10-05, 10-06
    assert not (accounting_dir() / "collector" / purge_files.MARKER).exists()
    assert accounting_dir().joinpath("worker", "lease.lock").exists()


def test_a_later_collection_cycle_continues_from_the_purged_state(tmp_path: Path) -> None:
    build_scene(tmp_path)
    purge.execute(select(), now=PURGE_AT)
    again = collect(at(22, 12))
    assert again.state == "published" and again.report is not None
    assert again.report.generation == 3
    assert periods() == {"2026-10-15"}  # nothing old was resurrected from the sources
    assert set(evidence_by_session()) == {"a2", "src-a2"}


def test_the_cutoff_day_itself_is_kept_and_the_day_before_it_goes(tmp_path: Path) -> None:
    build_scene(tmp_path)
    purge.execute(select("2026-10-06"), now=PURGE_AT)  # strictly before 10-06: only 10-05
    assert periods() == {"2026-10-06", "2026-10-15"}
    purge.execute(select("2026-10-06T23:59:59Z"), now=PURGE_AT)  # same UTC day: nothing more
    assert periods() == {"2026-10-06", "2026-10-15"}
    purge.execute(select("2026-10-07"), now=PURGE_AT)
    assert periods() == {"2026-10-15"}


def test_second_run_is_idempotent_and_changes_nothing(tmp_path: Path) -> None:
    build_scene(tmp_path)
    first = purge.execute(select(), now=PURGE_AT)
    assert first.state == "purged"
    after = tree(data_root(tmp_path))
    second = purge.execute(select(), now=PURGE_AT)
    assert second.state == "nothing_to_purge" and second.local_deleted.total() == 0
    changed = {k for k in after.keys() | tree(data_root(tmp_path)).keys()} - {
        "accounting/personal-local/worker/lease.lock"
    }
    now = tree(data_root(tmp_path))
    assert {k: after.get(k) for k in changed} == {k: now.get(k) for k in changed}


def test_a_purge_with_nothing_old_leaves_every_byte_alone(tmp_path: Path) -> None:
    build_scene(tmp_path)
    before = tree(data_root(tmp_path))
    report = purge.execute(select("2026-10-01"), now=PURGE_AT)  # nothing is dated before 10-01
    assert report.state == "nothing_to_purge" and report.local_deleted.total() == 0
    assert periods() == {"2026-10-05", "2026-10-06", "2026-10-15"}
    stable = {k: v for k, v in tree(data_root(tmp_path)).items() if "lease.lock" not in k}
    assert stable == {k: v for k, v in before.items() if "lease.lock" not in k}


# ---- workspace scope --------------------------------------------------------------------------


def test_workspace_purge_removes_only_that_workspace_and_counts_unknown_scope(
    tmp_path: Path,
) -> None:
    build_scene(tmp_path, orphan=True)
    report = purge.execute(select(workspace=ALPHA), now=PURGE_AT)

    assert periods() == {"2026-10-06", "2026-10-15"}  # alpha's 10-05 went; beta and 10-15 stay
    sessions = set(evidence_by_session())
    assert not sessions & {"a1", "src-a1"}
    assert sessions >= {"b1", "src-b1", "a2", "src-a2", "orph"}  # other workspace, newer, unknown
    assert report.retained.unknown_scope_conversations >= 1  # 'orph' has no known directory
    assert report.local_deleted.ledger_period_rows == 1
    assert report.retained.sealed_days_not_splittable == 3  # aggregates cannot be split
    remaining = sorted(
        {d for p in (accounting_dir() / "journal").iterdir() for d in _event_days(p)}
    )
    assert remaining == ["2026-10-01", "2026-10-06", "2026-10-15"]  # only alpha's day-5 file went
    assert report.retained.journal_segments_retained == 3


def test_workspace_by_lexical_prefix_does_not_select_a_sibling_directory(tmp_path: Path) -> None:
    build_scene(tmp_path)
    purge.execute(select(workspace="/work/alph"), now=PURGE_AT)  # prefix of /work/alpha
    assert periods() == {"2026-10-05", "2026-10-06", "2026-10-15"}


# ---- storage safety ---------------------------------------------------------------------------


def _sentinels(tmp_path: Path) -> dict[Path, bytes]:
    data = data_root(tmp_path)
    found = {
        tmp_path / "claude" / "projects" / "x.jsonl": b'{"transcript": 1}\n',
        data / "lancedb" / "table.lance": b"index bytes",
        data / "notes" / "notes-202610.jsonl": b'{"note": "keep"}\n',
        data / "accounting" / "policy.json.sentinel": b"policy sibling",
        data / "accounting" / "other-profile" / "snapshots" / "gen-0000000001.json": b"theirs",
        data / "accounting" / "other-profile" / "journal" / "seg-0123456789abcdef.jsonl": b"x\n",
    }
    for path, body in found.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        old = datetime(2026, 1, 1, tzinfo=UTC).timestamp()
        os.utime(path, (old, old))
    return found


def test_purge_never_touches_transcripts_index_notes_policy_or_other_profiles(
    tmp_path: Path,
) -> None:
    build_scene(tmp_path)
    secrets = [
        accounting_dir() / "fingerprint.key",
        accounting_dir() / "export" / "pseudonym.key",
    ]
    export_ids.load_key(PROFILE, create=True)
    sentinels = _sentinels(tmp_path)
    policy_before = (data_root(tmp_path) / "accounting" / "policy.json").read_bytes()
    keys_before = {p: p.read_bytes() for p in secrets}

    purge.execute(select(), now=PURGE_AT)

    for path, body in sentinels.items():
        assert path.read_bytes() == body, path
        assert path.stat().st_mtime == datetime(2026, 1, 1, tzinfo=UTC).timestamp()
    assert (data_root(tmp_path) / "accounting" / "policy.json").read_bytes() == policy_before
    assert {p: p.read_bytes() for p in secrets} == keys_before
    assert periods() == {"2026-10-15"}  # and the purge itself really happened


def test_a_symlinked_collector_directory_is_refused_and_nothing_outside_is_deleted(
    tmp_path: Path,
) -> None:
    build_scene(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "evidence").mkdir()
    victim = elsewhere / "evidence" / "seg-victim.jsonl"
    victim.write_bytes(b"not ours\n")
    real = accounting_dir() / "collector"
    real.rename(tmp_path / "moved-collector")
    real.symlink_to(elsewhere)
    before = tree(data_root(tmp_path))
    outside = tree(elsewhere)

    for run in (lambda: purge.dry_run(select()), lambda: purge.execute(select(), now=PURGE_AT)):
        with pytest.raises(PurgeError) as raised:
            run()
        assert (raised.value.condition, raised.value.exit_code) == ("storage_unsafe", 1)
        assert str(tmp_path) not in str(raised.value)
    assert tree(data_root(tmp_path)) == before
    assert tree(elsewhere) == outside and victim.read_bytes() == b"not ours\n"


def test_a_symlinked_receipt_file_is_refused_before_any_deletion(tmp_path: Path) -> None:
    build_scene(tmp_path)
    target = tmp_path / "precious.json"
    target.write_text("precious")
    receipt = next((accounting_dir() / "receipts").iterdir())
    receipt.unlink()
    receipt.symlink_to(target)
    before = tree(data_root(tmp_path))
    with pytest.raises(PurgeError) as raised:
        purge.execute(select(), now=PURGE_AT)
    assert raised.value.condition == "storage_unsafe"
    assert tree(data_root(tmp_path)) == before and target.read_text() == "precious"


# ---- the worker lease -------------------------------------------------------------------------


def _hold_lease() -> subprocess.Popen[str]:
    code = (
        "import sys\n"
        "from ssgrep.accounting.serve.lease import Lease\n"
        "lease = Lease.acquire(sys.argv[1], 'collect')\n"
        "print('ready', flush=True)\n"
        "sys.stdin.readline()\n"
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", code, PROFILE],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        env=dict(os.environ),
    )
    assert holder.stdout is not None and holder.stdout.readline().strip() == "ready"
    return holder


def test_another_process_holding_the_lease_blocks_a_real_purge_and_nothing_changes(
    tmp_path: Path,
) -> None:
    build_scene(tmp_path)
    holder = _hold_lease()
    try:
        before = tree(data_root(tmp_path))
        with pytest.raises(PurgeError) as raised:
            purge.execute(select(), now=PURGE_AT)
        assert (raised.value.condition, raised.value.exit_code) == ("already_running", 1)
        assert tree(data_root(tmp_path)) == before
        preview = purge.dry_run(select())  # a preview is still allowed, and says who runs
        assert preview.worker is not None and preview.worker.running
        assert preview.worker.kind == "collect"
    finally:
        assert holder.stdin is not None
        holder.stdin.write("\n")
        holder.stdin.flush()
        assert holder.wait(timeout=30) == 0
    assert purge.execute(select(), now=PURGE_AT).state == "purged"


# ---- export linkage: tombstones, unsent batches, honesty --------------------------------------


def configure_policy(*, logs_permitted: bool, enabled: bool = True) -> None:
    destinations = tuple(
        DestinationPolicy(
            name,
            ENDPOINT,
            (Signal.METRICS, Signal.LOGS) if logs_permitted else (Signal.METRICS,),
            logs_permitted=logs_permitted,
        )
        for name in (DEST, OTHER)
    )
    exports = tuple((name, "logs") for name in (DEST, OTHER)) if logs_permitted and enabled else ()
    policy.save_profile(
        ProfilePolicy(destinations=destinations, enabled_exports=exports, collection_enabled=True)
    )
    if logs_permitted and enabled:
        from ssgrep.accounting.export.consent import apply_enable

        for destination in (DEST, OTHER):
            apply_enable(PROFILE, destination, Signal.LOGS)


def export_all(destination: str, *, ack: bool) -> list[str]:
    """Project the REAL ledger into ``destination``'s registry; ``ack`` = the receiver took it."""
    key = export_ids.load_key(PROFILE, create=True)
    assert key is not None
    rows = export_records.ledger_rows(PROFILE)
    assert rows is not None
    candidates, omitted = project_candidates(rows, key=key, destination=destination)
    assert omitted == 0 and candidates
    planned = plan_revisions(Registry(), candidates)
    mark_queued(PROFILE, destination, planned, "g0000000001-0123456789abcdef-0000")
    if ack:
        mark_acked(PROFILE, destination, "g0000000001-0123456789abcdef-0000")
    return sorted(p.record_id for p in planned)


def old_record_ids(destination: str) -> list[str]:
    return export_registry.records_before(PROFILE, destination, CUTOFF)


def tombstone_batches(destination: str) -> list[export_spool.Meta]:
    return [
        m
        for m in export_spool.list_batches(PROFILE, destination, Signal.LOGS)
        if m.kind == "tombstone"
    ]


def test_previously_exported_records_get_tombstones_queued_while_the_receiver_is_offline(
    tmp_path: Path,
) -> None:
    build_scene(tmp_path)
    configure_policy(logs_permitted=True)
    export_all(DEST, ack=True)
    doomed = old_record_ids(DEST)
    assert len(doomed) == 4  # periods 10-05 and 10-06, two units each
    assert not tombstone_batches(DEST)

    report = purge.execute(select(), now=PURGE_AT)  # no receiver anywhere: nothing is sent

    (batch,) = tombstone_batches(DEST)  # queued, never sent
    assert batch.state == "queued" and batch.attempts == 0 and batch.kind == "tombstone"
    assert set(batch.record_ids) == set(doomed)
    body = export_spool.read_body(PROFILE, DEST, Signal.LOGS, batch.batch_id)
    assert b"deletion" in body
    assert report.queued_records == 4 and report.queued_batches == 1
    (line,) = report.tombstones
    assert (line.destination, line.status, line.records) == (DEST, "queued", 4)
    document = report.to_dict()
    assert document["backend_erased"] == BACKEND_ERASED == "unknown_operator_responsibility"
    assert document["accepted_by_collector"]["count"] == 0
    assert document["queued_tombstones"]["records"] == 4
    assert document["operator_obligations"] == []
    # linkage survives locally as the minimal pending state: a deletion revision per record
    entries = load_registry(PROFILE, DEST).entries
    assert all(entries[i].queued and entries[i].queued[-1].type == "deletion" for i in doomed)
    assert periods() == {"2026-10-15"}  # local deletion still completed


def test_a_dry_run_reports_the_tombstones_it_would_queue_and_writes_none(tmp_path: Path) -> None:
    build_scene(tmp_path)
    configure_policy(logs_permitted=True)
    export_all(DEST, ack=True)
    before = tree(data_root(tmp_path))
    preview = purge.dry_run(select())
    assert tree(data_root(tmp_path)) == before and not tombstone_batches(DEST)
    (line,) = preview.tombstones
    assert (line.status, line.records, line.batches) == ("would_queue", 4, None)
    assert preview.queued_records == 4


def test_when_sending_is_not_permitted_obligations_are_reported_and_no_batch_is_written(
    tmp_path: Path,
) -> None:
    build_scene(tmp_path)
    configure_policy(logs_permitted=False)
    exported = export_all(DEST, ack=True)
    doomed = old_record_ids(DEST)
    assert len(doomed) == 4 and set(doomed) <= set(exported)

    report = purge.execute(select(), now=PURGE_AT)

    assert not tombstone_batches(DEST)
    assert not (accounting_dir() / "spool").exists()  # nothing was spooled at all
    (owed,) = report.obligations
    assert owed.destination == DEST and owed.records == 4
    assert set(owed.record_ids) == set(doomed) and len(owed.record_ids) <= 1000
    assert owed.reason is not None
    (line,) = report.tombstones
    assert line.status == "not_permitted" and report.queued_records == 0
    assert report.to_dict()["backend_erased"] == BACKEND_ERASED
    assert periods() == {"2026-10-15"}


def test_acknowledged_metrics_are_flagged_non_reversible_and_their_pin_is_released(
    tmp_path: Path,
) -> None:
    build_scene(tmp_path)
    configure_policy(logs_permitted=True)
    export_all(DEST, ack=True)
    old = current_envelope()
    stamp = "2026-10-21T12:00:00Z"
    acks.update_delivery(
        PROFILE, DEST, Signal.METRICS, lambda d: acks.ack_vacuous(d, 1, old.snapshot_id, stamp)
    )
    assert old.snapshot_id in pinned_snapshot_ids(PROFILE)

    report = purge.execute(select(), now=PURGE_AT)

    assert report.non_reversible_destinations == (DEST,)
    notes = report.to_dict()["non_reversible_metrics"]
    assert notes[0]["destination"] == DEST and "cannot_be_recalled" in notes[0]["note"]
    assert report.local_deleted.delivery_pins_released == 1
    state = acks.read_delivery(PROFILE, DEST, Signal.METRICS)
    assert state is not None and state.last_acked_snapshot_id is None
    assert state.expired_snapshot_id == old.snapshot_id
    assert state.last_acked_generation == 1  # acknowledgement history is never rewritten
    assert old.snapshot_id not in "".join(generation_names())


def test_unsent_batches_with_selected_records_and_all_unsent_metrics_are_discarded_and_recorded(
    tmp_path: Path,
) -> None:
    build_scene(tmp_path)
    configure_policy(logs_permitted=True)
    envelope = current_envelope()
    full = policy.load_profile_policy(PROFILE)
    dest = next(d for d in full.destinations if d.destination_id == OTHER)
    key = export_ids.load_key(PROFILE, create=True)
    assert key is not None
    logs = export_build.build_logs(PROFILE, dest, full, envelope, key)
    metrics = export_build.build_metrics(PROFILE, dest, full, envelope, key)
    created = "2026-10-21T12:00:00Z"
    for batch, members in zip(logs.batches, logs.planned, strict=True):
        export_spool.write_batch(
            PROFILE, OTHER, Signal.LOGS, batch, created_utc=created, scope=logs.scope
        )
        mark_queued(PROFILE, OTHER, members, batch.batch_id)
    for batch in metrics.batches:
        export_spool.write_batch(
            PROFILE, OTHER, Signal.METRICS, batch, created_utc=created, scope=metrics.scope
        )
    assert export_spool.list_batches(PROFILE, OTHER, Signal.LOGS)
    assert export_spool.list_batches(PROFILE, OTHER, Signal.METRICS)

    report = purge.execute(select(), now=PURGE_AT)

    assert export_spool.list_batches(PROFILE, OTHER, Signal.METRICS) == []  # regenerated later
    assert export_spool.list_batches(PROFILE, OTHER, Signal.LOGS) == []  # held selected records
    assert report.local_deleted.spool_batches == len(logs.batches) + len(metrics.batches)
    for signal in (Signal.LOGS, Signal.METRICS):
        state = acks.read_delivery(PROFILE, OTHER, signal)
        assert state is not None and state.lost_records > 0
        assert state.loss_summary[0].reason == "purge_discarded"
        assert (state.pending_batches, state.unknown_batches) == (0, 0)
    queued_after = [e for e in load_registry(PROFILE, OTHER).entries.values() if e.queued]
    assert queued_after == []  # the registry forgot the discarded queued states


def test_an_unsent_batch_of_records_outside_the_selection_is_kept(tmp_path: Path) -> None:
    build_scene(tmp_path)
    configure_policy(logs_permitted=True)
    envelope = current_envelope()
    full = policy.load_profile_policy(PROFILE)
    dest = next(d for d in full.destinations if d.destination_id == OTHER)
    key = export_ids.load_key(PROFILE, create=True)
    assert key is not None
    logs = export_build.build_logs(PROFILE, dest, full, envelope, key)
    keep_root = root_key_of("claude:a2")
    for batch, members in zip(logs.batches, logs.planned, strict=True):
        keep = [m for m in members if m.root_key == keep_root]
        assert keep
        only = replace(batch, record_ids=tuple(sorted(m.record_id for m in keep)))
        export_spool.write_batch(
            PROFILE, OTHER, Signal.LOGS, only, created_utc="2026-10-21T12:00:00Z", scope=logs.scope
        )
    purge.execute(select(), now=PURGE_AT)
    assert len(export_spool.list_batches(PROFILE, OTHER, Signal.LOGS)) == len(logs.batches)


def test_with_a_workspace_only_that_workspaces_records_get_tombstones(tmp_path: Path) -> None:
    build_scene(tmp_path)
    configure_policy(logs_permitted=True)
    export_all(DEST, ack=True)
    alpha_old = export_registry.records_before(PROFILE, DEST, CUTOFF, [root_key_of("claude:a1")])
    assert len(alpha_old) == 2
    report = purge.execute(select(workspace=ALPHA), now=PURGE_AT)
    (batch,) = tombstone_batches(DEST)
    assert set(batch.record_ids) == set(alpha_old)
    assert report.queued_records == 2


# ---- crash safety -----------------------------------------------------------------------------


def test_a_crash_after_the_checkpoint_commit_is_finished_by_the_next_run(tmp_path: Path) -> None:
    build_scene(tmp_path)

    def crash(profile: str, generation: int) -> None:
        raise OSError("simulated power loss before current.json moved")

    with pytest.MonkeyPatch.context() as local:  # never ``undo`` the isolation fixture's env
        local.setattr(purge_snapshot, "make_current", crash)
        with pytest.raises(PurgeError) as raised:
            purge.execute(select(), now=PURGE_AT)
    assert raised.value.condition == "purge_failed"
    assert (accounting_dir() / "collector" / purge_files.MARKER).exists()  # unfinished is visible
    assert periods() == {"2026-10-15"}  # the commit point was already crossed: privacy first
    committed_snapshot = snapshots.read_generation(PROFILE, 2)
    assert committed_snapshot is not None

    assert purge.dry_run(select()).state != "purged"
    purge.execute(select(), now=PURGE_AT)
    assert not (accounting_dir() / "collector" / purge_files.MARKER).exists()
    assert current_envelope().generation == 2
    assert generation_names() == [f"gen-0000000002-{current_envelope().snapshot_id}.json"]
    assert len(list((accounting_dir() / "receipts").iterdir())) == 1
    assert (
        current_envelope() == committed_snapshot
    )  # exact accepted rows, independent of cleanup phase
    assert snapshots.current_generation(PROFILE) == 2


def test_report_json_never_contains_paths_or_record_contents(tmp_path: Path) -> None:
    import json

    build_scene(tmp_path)
    configure_policy(logs_permitted=False)
    export_all(DEST, ack=True)
    report = purge.execute(select(workspace=ALPHA), now=PURGE_AT)
    text = json.dumps(report.to_dict())
    for forbidden in (str(tmp_path), "services/alpha", "ERR_ALPHA", "fix the deploy", ".jsonl"):
        assert forbidden not in text.replace(ALPHA, "")
    assert not any(isinstance(v, float) for v in _walk(report.to_dict()))


def _walk(value: Any):
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _walk(item)
    else:
        yield value


# ---- helpers, scoping and state plans (from the purge-storage review batch) --------------------


def test_absent_file_helpers_create_nothing_and_present_helpers_delete_only_selected_files(
    tmp_path: Path,
) -> None:
    assert purge_files.receipt_ids(PROFILE) == set()
    assert purge_files.delete_receipts(PROFILE, {"old"}) == 0
    assert purge_files.generation_files(PROFILE) == []
    assert purge_files.has_current(PROFILE) is False
    assert purge_files.delete_generations(PROFILE, []) == 0
    assert purge_files.delete_current(PROFILE) is False
    purge_files.clear_marker(PROFILE)
    assert not (tmp_path / "data").exists()
    first, second = make_envelope(1), make_envelope(2)
    snapshots.publish(PROFILE, first)
    snapshots.publish(PROFILE, second)
    for name, data in (("old.json", b"old receipt"), ("keep.json", b"keep receipt")):
        plainfile.write_private(
            storage.data_root(), "accounting", PROFILE, "receipts", name, data=data
        )
    assert purge_files.receipt_ids(PROFILE) == {"old", "keep"}
    assert purge_files.delete_receipts(PROFILE, {"old", "absent"}) == 1
    assert purge_files.has_current(PROFILE)
    assert purge_files.delete_current(PROFILE)
    assert not purge_files.delete_current(PROFILE)
    generations = purge_files.generation_files(PROFILE)
    assert [g.number for g in generations] == [1, 2]
    assert purge_files.delete_generations(PROFILE, generations[:1]) == 1
    assert snapshots.read_generation(PROFILE, 1) is None
    assert snapshots.read_generation(PROFILE, 2) == second
    assert (accounting_dir() / "receipts" / "keep.json").read_bytes() == b"keep receipt"


def test_scoper_rejects_ambiguous_paths_and_tracks_mixed_conversations() -> None:
    scoper = purge_select.Scoper(ALPHA)
    for invalid in (None, "", "relative", "\0bad"):
        assert not scoper.inside(invalid)
    assert scoper.inside(ALPHA + "/child")
    assert not scoper.inside(ALPHA + "-other")
    records = [
        build_evidence(
            session_id=session,
            record_id=str(index),
            ts_utc="2026-10-01T00:00:00Z",
            payload={"cwd": cwd},
        )
        for index, (session, cwd) in enumerate(
            [
                ("selected", ALPHA),
                ("mixed", ALPHA),
                ("mixed", "/work/beta"),
                ("unknown", None),
                ("other", "/work/beta"),
            ]
        )
    ]
    checkpoint = collector_store.Checkpoint(
        PROFILE, segments=collector_store.write_segments(PROFILE, records)
    )
    scopes = purge_select.classify(PROFILE, checkpoint, None, (), scoper)
    assert scopes.selected == frozenset({"claude:selected"})
    assert (scopes.unknown, scopes.mixed) == (1, 1)
    assert scopes.root(None, "selected") is None
    assert scopes.root("claude", None) is None
    assert [
        (r.identity, r.payload.get("cwd"))
        for r in collector_store.iter_evidence(PROFILE, checkpoint.segments)
    ] == [(r.identity, r.payload.get("cwd")) for r in records]


def test_undated_structure_follows_session_cutoff_and_unselected_segments_keep_identity() -> None:
    def record(session, name, stamp):
        return build_evidence(
            session_id=session, record_id=name, ts_utc=stamp, payload={"cwd": ALPHA}
        )

    gone = [record("old", "structure-old", None), record("old", "old", "2026-10-01T00:00:00Z")]
    live = [record("live", "structure-live", None), record("live", "new", "2026-10-10T00:00:00Z")]
    old_refs = collector_store.write_segments(PROFILE, gone)
    live_refs = collector_store.write_segments(PROFILE, live)
    checkpoint = collector_store.Checkpoint(PROFILE, segments=[*old_refs, *live_refs])
    collector_store.commit_checkpoint(PROFILE, checkpoint)
    planned = purge_state.plan_state(PROFILE, select(), checkpoint)
    assert planned is not None and planned.evidence_records == 2
    assert planned.gone_sessions == frozenset({("claude", "old")})
    rewritten = purge_state.write_purged_state(PROFILE, planned, select())
    assert rewritten.segments == live_refs
    assert [r.identity for r in collector_store.iter_evidence(PROFILE, rewritten.segments)] == [
        r.identity for r in live
    ]
    assert [r.identity for r in collector_store.iter_evidence(PROFILE, checkpoint.segments)] == [
        r.identity for r in gone + live
    ]
    assert collector_store.load_checkpoint(PROFILE) == checkpoint


def test_missing_execution_cwd_uses_proven_session_scope_and_retains_unknowns(
    tmp_path: Path,
) -> None:
    build_scene(tmp_path)
    checkpoint = collector_store.load_checkpoint(PROFILE)
    assert checkpoint is not None
    state = load_state(PROFILE, checkpoint, checkpoint.methodology)
    assert state.reduction is not None and state.reduction.executions
    prototype = state.reduction.executions[0]
    executions = tuple(
        replace(
            prototype,
            execution_id=name,
            day="2026-10-05",
            ts_utc="2026-10-05T00:00:00Z",
            cwd=None,
            runtime="claude",
            session_id=session,
        )
        for name, session in (("proven", "a1"), ("outside", "b1"), ("unknown", None))
    )
    doc = replace(state.reduction, executions=executions)
    checkpoint = replace(
        checkpoint,
        blobs={
            **checkpoint.blobs,
            REDUCTION: collector_store.write_blob(PROFILE, REDUCTION, doc.to_dict()),
        },
    )
    planned = purge_state.plan_state(PROFILE, select(workspace=ALPHA), checkpoint)
    assert planned is not None and planned.reduction_after is not None
    assert planned.executions_removed == 1 and planned.unknown_scope_executions == 2
    assert planned.reduction_after.executions == executions[1:]
    assert collector_store.read_blob(PROFILE, checkpoint.blobs[REDUCTION]) == doc.to_dict()


def test_snapshot_build_requires_current_methodology_instead_of_publishing_empty_totals() -> None:
    checkpoint = collector_store.Checkpoint(PROFILE)
    planned = purge_state.StatePlan(checkpoint, State(), purge_select.Scopes(None))
    with pytest.raises(ValueError, match="no ledger"):
        purge_snapshot.build(PROFILE, planned, replace(checkpoint, generation=1), PURGE_AT)
    assert snapshots.generations(PROFILE) == []


def test_purge_removes_uncommitted_temporary_files_without_deleting_other_collector_data(
    tmp_path: Path,
) -> None:
    build_scene(tmp_path)
    collector = storage.open_profile_dir(PROFILE, "collector")
    try:
        storage.atomic_write(collector, ".scratch.tmp", b"unfinished private data")
    finally:
        os.close(collector)
    nested = storage.open_profile_dir(PROFILE, "collector", "nested")
    try:
        storage.atomic_write(nested, "retained-state.json", b"still required")
    finally:
        os.close(nested)
    cache = storage.open_profile_dir(PROFILE, "collector", "cache")
    try:
        storage.atomic_write(cache, "malformed-cache.json", b"{not-json\n")
        storage.atomic_write(cache, ".pending-cache.tmp", b"uncommitted cache data")
    finally:
        os.close(cache)

    purge.execute(select(), now=PURGE_AT)

    assert not (accounting_dir() / "collector" / ".scratch.tmp").exists()
    assert not (accounting_dir() / "collector" / "cache" / "malformed-cache.json").exists()
    assert not (accounting_dir() / "collector" / "cache" / ".pending-cache.tmp").exists()
    assert (
        accounting_dir() / "collector" / "nested" / "retained-state.json"
    ).read_bytes() == b"still required"


def test_empty_profile_with_frequency_authority_writes_tombstone_before_deletion(
    tmp_path: Path,
) -> None:
    selection = select(before="2026-10-10")
    collector = storage.open_profile_dir(PROFILE, "collector")
    try:
        storage.atomic_write(collector, "frequency.json", b"legacy-frequency")
    finally:
        os.close(collector)

    result = purge_files.purge_derived(selection)

    assert result.frequency_files == 1
    assert not (accounting_dir() / "collector" / "frequency.json").exists()
    raw = (accounting_dir() / "collector" / purge_files.AUTHORITY_TOMBSTONE).read_bytes()
    tombstone = json.loads(raw)
    assert tombstone["reason"] == "frequency_scope_erased"
    assert tombstone["whole_profile"] is True
    assert tombstone["authorized_generation"] is None
