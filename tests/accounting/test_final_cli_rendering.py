"""Consumer-facing quantity, scope, privacy and pagination boundaries in accounting views."""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ssgrep.accounting import export_consent, service
from ssgrep.accounting.collector_types import CollectOptions, CollectResult, CycleReport
from ssgrep.accounting.contracts import Basis, Quantity
from ssgrep.accounting.export_consent_preview import BatchView, PreviewView
from ssgrep.accounting.ledger_contracts import DestinationPolicy, ProfilePolicy, Signal
from ssgrep.accounting.purge_model import (
    LocalDeleted,
    Obligation,
    PurgeReport,
    TombstoneLine,
    WorkerView,
)
from ssgrep.accounting.service_status import read_status
from ssgrep.accounting.service_types import DeliveryLine, HistoryEntry, HistoryView, Scope
from ssgrep.accounting.status_worker import Watermark
from ssgrep.cli import (
    collect_json,
    collect_render,
    export_render,
    metrics_render,
    purge_render,
    savings_render,
    savings_state_render,
)
from ssgrep.cli.savings_text import Doc, fmt_age
from tests.accounting.snapshot_builders import (
    RECEIPT_ID,
    enable_policy,
    write_receipt,
    write_snapshot,
)
from tests.accounting.test_export_consent import DEST, ENDPOINT, configure
from tests.accounting.test_savings_render import flat
from tests.accounting.verify_scenes import env_for

NOW = datetime(2026, 10, 12, 12, tzinfo=UTC)


@pytest.mark.parametrize("width", [40, 80, 120])
@pytest.mark.parametrize("label", ["More:", ""])
def test_scope_command_round_trips_shell_metacharacters_and_child_selection(
    width: int,
    label: str,
    tmp_path: Path,
):
    scope = Scope(
        since="2026-10-01",
        until="2026-10-02",
        project="/w/O'Brien team;$(printf unexpected)/" + "space path/" * 14,
        runtime="omp",
        session="session-a",
        task="turn-a",
        by="task",
        include_children=True,
    )
    command = savings_state_render.scope_command("acme", scope, 7, "cursor-1")
    doc = Doc(width)
    doc.command(label, command)
    rendered = doc.text().plain
    assert all(len(line) <= width for line in rendered.splitlines())
    parsed = subprocess.run(
        ["/bin/sh", "-c", "set -e; ssgrep() { printf '%s\\0' ssgrep \"$@\"; };\n" + rendered],
        cwd=tmp_path,
        capture_output=True,
        check=True,
    )
    tokens = [part.decode() for part in parsed.stdout.split(b"\0")[:-1]]
    assert tokens == shlex.split(command)
    assert "\n" not in command
    assert tokens[:2] == ["ssgrep", "savings"]
    assert tokens[tokens.index("--project") + 1] == scope.project
    for flag, value in (
        ("--since", scope.since),
        ("--until", scope.until),
        ("--runtime", "omp"),
        ("--session", "session-a"),
        ("--task", "turn-a"),
        ("--by", "task"),
        ("--limit", "7"),
        ("--cursor", "cursor-1"),
        ("--profile", "acme"),
    ):
        assert tokens[tokens.index(flag) + 1] == value
    assert "--include-children" in tokens


@pytest.mark.parametrize("width", [40, 80, 120])
def test_receipt_rounds_raw_characters_and_sums_known_lookup_costs(tmp_path: Path, width: int):
    write_receipt(tmp_path / "data")
    base = service.read_receipt("personal-local", RECEIPT_ID)
    assert base is not None
    view = replace(
        base,
        unit="context_tokens:chars_div4_v1",
        period="not-a-day",
        credit_raw=(("conservative", 19), ("central", 31)),
        costs=(
            ("first", Quantity(7, "context_tokens", Basis.ESTIMATED)),
            ("second", Quantity(11, "context_tokens", Basis.ESTIMATED)),
        ),
        exclusions=(
            "atom:source:path:prior_known",
            "missing:unresolved:source_unresolved:opaque:unavailable",
        ),
    )
    text = flat(savings_render.render_receipt(view, profile_hint="acme", width=width).plain)
    assert "+4 tokens" in text and "+7 tokens" in text
    assert "18 tokens" in text
    assert "source:path" in text and "source_unresolved" in text
    assert "unresolved:source_unresolved:opaque" not in text
    assert "ssgrep savings --profile acme" in text
    assert "--since not-a-day" not in text
    missing = flat(savings_render.render_receipt(replace(view, costs=()), width=width).plain)
    assert "unknown" in missing and "0 tokens" not in missing


def test_empty_and_expired_history_do_not_fabricate_a_recomputable_zero():
    unknown = Quantity.unavailable("context_tokens", "periods_unavailable")
    view = HistoryView("acme", (), unknown, None, reason="retention_boundary")
    text = flat(savings_render.render_history(view).plain)
    assert "periods_unavailable" in text and "retention_boundary" in text
    assert "0 tokens" not in text
    retained = replace(
        view,
        entries=(
            HistoryEntry(
                "2026-10-01",
                2,
                "revised",
                True,
                1,
                Quantity(-37, "context_tokens", Basis.ESTIMATED),
                recomputable=False,
            ),
        ),
        current_total=Quantity(-37, "context_tokens", Basis.ESTIMATED),
    )
    text = flat(savings_render.render_history(retained).plain)
    assert "-37 tokens" in text and "expired" in text


def test_missing_measured_fields_and_empty_groups_stay_absent(tmp_path: Path):
    enable_policy()
    write_snapshot(tmp_path / "data")
    view = service.read_summary(now=NOW)
    view = replace(
        view,
        measured=(("searches", Quantity.observed(17, "count")),),
        scope=Scope(by="task"),
        groups=(),
    )
    text = flat(savings_render.render_summary(view).plain)
    measured = text.split("MEASURED", 1)[1].split("ESTIMATED", 1)[0]
    assert "17" in measured and "Shows" not in measured and "0 B" not in measured
    groups = text.split("BY TASK", 1)[1].split("EVIDENCE", 1)[0]
    assert "0 tokens" not in groups and "No rows" in groups


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(None, "age unknown"), (172799, "47h ago"), (172800, "2d ago"), (259199, "2d ago")],
)
def test_age_precision_changes_at_whole_day_boundary(seconds: int | None, expected: str):
    assert fmt_age(seconds) == expected


def test_status_keeps_watermark_count_distinct_from_visible_rows_and_delivery_age():
    base = read_status(now=NOW)
    marks = tuple(Watermark(f"{i:016x}", "omp", 100, 120, 20, 0, "ready") for i in range(10))
    lag = replace(base.lag, lag_bytes=200, watermark_count=10, watermarks=marks)
    line = DeliveryLine(
        DEST,
        "logs",
        True,
        "acknowledged",
        "collector accepted",
        last_acked_at="2026-10-10T12:00:00Z",
    )
    view = replace(
        base,
        lag=lag,
        delivery_lines=(line,),
        delivery_age_seconds=172800,
        policy_error="invalid\x1b[2Jpolicy",
    )
    text = flat(metrics_render.render_status(view, 80).plain)
    assert "200 B" in text and "2d ago" in text
    assert "2026-10-10T12:00:00Z" in text
    assert marks[7].source in text and marks[8].source not in text
    assert "2 more" in text and "--json" in text
    assert "\x1b" not in text and "\\x1b[2J" in text
    unknown = replace(view, lag=replace(lag, watermark_count=None, watermarks=None))
    text = flat(metrics_render.render_status(unknown, 80).plain)
    assert "200 B" in text and "Watermarks unknown" in text


@pytest.mark.parametrize("roots", [(), ("/w/team\x1b[2J",)])
def test_setup_preview_preserves_roots_retention_and_disabled_destinations(roots: tuple[str, ...]):
    p = ProfilePolicy(
        profile_id="acme",
        roots=roots,
        evidence_retention_days=17,
        receipt_retention_days=43,
        destinations=(DestinationPolicy(DEST, ENDPOINT),),
    )
    text = flat(
        metrics_render.render_setup_preview(
            p, width=80, already_enabled=False, source="managed"
        ).plain
    )
    assert "17 days" in text and "43 days" in text and DEST in text
    assert "disabled" in text
    assert "\x1b" not in text
    if roots:
        assert "/w/team\\x1b[2J" in text
    else:
        assert "none approved" in text and "all working directories" not in text


def preview(bodies: tuple[bytes, ...]) -> PreviewView:
    batches = tuple(
        BatchView(f"b-{i}", len(body), i + 1, hashlib.sha256(body).hexdigest(), body)
        for i, body in enumerate(bodies)
    )
    return PreviewView(
        "personal-local",
        DEST,
        Signal.METRICS,
        "ok",
        None,
        1,
        "localhost",
        "metrics-v1",
        False,
        7,
        "a" * 16,
        0,
        (),
        batches,
    )


@pytest.mark.parametrize("cap", [0, 12, 24])
def test_export_preview_print_cap_preserves_whole_first_request_and_summarizes_rest(cap: int):
    bodies = tuple(
        json.dumps({"value": n}, separators=(",", ":")).encode("ascii") for n in (1, 2, 3)
    )
    view = preview(bodies)
    lines = export_render.preview_payload_lines(view, cap)
    shown = [line.encode("ascii") for line in lines if line.startswith("{")]
    count = max(1, min(len(bodies), cap // len(bodies[0])))
    assert shown == list(bodies[:count])
    for batch in view.batches[count:]:
        assert any(
            batch.batch_id in line and batch.sha256 in line and f"{batch.bytes} bytes" in line
            for line in lines
        )
    if count < len(bodies):
        assert any("--json" in line for line in lines)


def test_export_preview_explains_blocked_scope_and_enable_keeps_proxy_and_resume_separate():
    configure(
        destinations=(DestinationPolicy(DEST, ENDPOINT, proxy="http://127.0.0.1:8080"),),
        paused=True,
        paused_at="2026-10-10T00:00:00Z",
    )
    plan = export_consent.plan_enable("personal-local", DEST, Signal.METRICS)
    preview_text = flat(export_render.render_enable_preview(plan, 80).plain)
    assert "127.0.0.1:8080" in preview_text
    result_text = flat(export_render.render_enable_result(plan, True, 80).plain)
    assert "paused" in result_text
    assert (
        f"--destination {DEST}" in result_text
        and "--signal metrics" in result_text
        and "--check" in result_text
    )
    blocked = replace(preview(()), state="policy_blocked", reason="scope\x1b[2J_changed")
    text = flat(export_render.render_preview_header(blocked, 80).plain)
    assert "scope\\x1b[2J changed" in text and "\x1b" not in text


@pytest.mark.parametrize("performed", [False, True])
def test_purge_json_and_text_separate_local_queued_and_operator_erasure(performed: bool):
    report = PurgeReport(
        "acme",
        performed,
        "2026-10-01T00:00:00Z",
        "2026-10-01",
        "/w/team",
        LocalDeleted(receipt_files=13),
        tombstones=(
            TombstoneLine("allowed", 17, 2, "queued" if performed else "would_queue", None),
            TombstoneLine("blocked", 19, 0, "not_permitted", "export_disabled"),
        ),
        obligations=(Obligation("blocked", "export_disabled", 19, ("private-id",)),),
        non_reversible_destinations=("aggregate",),
        pending_retention_days=11,
        worker=WorkerView(True, "serve") if not performed else None,
    )
    document = purge_render.report_document(report, "acme")
    data = document["data"]
    assert isinstance(data, dict)
    assert data["local_deleted"]["receipt_files"] == 13
    assert data["queued_tombstones"]["records"] == 17
    assert data["accepted_by_collector"]["count"] == 0
    assert data["backend_erased"] == "unknown_operator_responsibility"
    warnings = document["warnings"]
    actions = document["next_actions"]
    assert isinstance(warnings, list) and isinstance(actions, list)
    assert "operator_must_erase_exported_records" in warnings
    assert "aggregate_metrics_cannot_be_recalled" in warnings
    assert ("worker_running" in warnings) is (not performed)
    assert all(action["command"].endswith("--profile acme") for action in actions)
    text = flat(purge_render.render_report(report, 80).plain)
    assert "13" in text and "allowed: 17 record(s)" in text and "blocked: 19 record(s)" in text
    assert "11 days" in text and "aggregate" in text and "operator responsibility" in text
    assert "private-id" not in text
    assert "blocked: 19 exported record(s)" in text


def test_missing_backfill_series_never_masquerades_as_zero_imported_rows():
    options = CollectOptions(backfill=True, since="2026-10-01", until="2026-10-02")
    report = CycleReport(
        "published", 1, "a" * 16, False, "byte_budget", bytes_read=64, lag_bytes=128
    )
    result = CollectResult(state="partial", mode="backfill", report=report, epoch=1)
    document, data, actions = collect_json.collect_document("acme", options, result)
    text = flat(collect_render.render_collect(data, (), actions, 80).plain)
    assert data["backfill"] is None
    assert "64 B" in text and "128 B" in text
    assert "--backfill --since 2026-10-01 --until 2026-10-02 --profile acme" in text


def test_stopped_worker_preserves_sanitized_last_failure():
    text = flat(
        collect_render.render_serve_final(
            {
                "url": "http://127.0.0.1:4321",
                "port": 4321,
                "token_file": "/tmp/token",
                "profile": "acme",
                "phase": "stopped",
                "cycles": 2,
                "last_report": None,
                "last_error": "disk\x1b[2Jfailure",
            },
            80,
        ).plain
    )
    assert "4321" in text and "2" in text
    assert "disk\\x1b[2Jfailure" in text and "\x1b" not in text


def test_unreadable_lease_falls_back_to_bounded_valid_progress(tmp_path: Path):
    from ssgrep.accounting import plainfile

    plainfile.write_private(
        tmp_path / "data", "accounting", "personal-local", "worker", data=b"untouched"
    )
    owner = collect_json.owner_view(
        "personal-local",
        {
            "pid": 2345,
            "kind": "collect",
            "phase": "r" * 200,
            "cycles": True,
            "bytes_read": -1,
            "lag_bytes": 2048,
            "sources_done": 2.5,
        },
    )
    assert owner["pid"] == 2345 and owner["alive"] is None
    assert owner["cycles"] is None and owner["bytes_read"] is None
    assert owner["sources_done"] is None
    phase = owner["phase"]
    assert isinstance(phase, str) and len(phase) == 64
    text = flat(collect_render.render_running({"owner": owner}, (), 80).plain)
    assert "2.0 KiB" in text and "unknown" in text
    assert (
        tmp_path / "data" / "accounting" / "personal-local" / "worker"
    ).read_bytes() == b"untouched"


def test_raw_scope_command_preserves_hash_prefixed_selection_in_real_cli(tmp_path: Path) -> None:
    scope = Scope(
        since=None,
        until=None,
        project="#notes",
        runtime="omp",
        session="#session",
        task="#task",
        by=None,
        include_children=True,
    )
    command = savings_state_render.scope_command("personal-local", scope, 20, None)
    binary = Path(sys.executable).parent / "ssgrep"
    script = f'ssgrep() {{ command {shlex.quote(str(binary))} "$@"; }};\n{command} --json'
    result = subprocess.run(
        ["/bin/sh", "-c", script],
        env=env_for(tmp_path),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    selected = json.loads(result.stdout)["data"]["data"]["scope"]
    assert {key: selected[key] for key in ("project", "runtime", "session", "task")} == {
        "project": "#notes",
        "runtime": "omp",
        "session": "#session",
        "task": "#task",
    }
    assert selected["include_children"] is True
