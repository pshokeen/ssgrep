from __future__ import annotations

import json
import re
from datetime import timedelta
from fractions import Fraction
from math import ceil, floor
from pathlib import Path
from typing import Any

import pytest

from ssgrep.accounting.export import acks
from ssgrep.accounting.ledger.ledger_contracts import DestinationPolicy, Signal, SnapshotState
from ssgrep.accounting.privacy import policy
from ssgrep.accounting.views import service
from ssgrep.accounting.views.service import read_status
from tests.accounting.collect import (
    collector_scene as records_scene,
    collector_scene as scene,
    verify_scenes as vs,
)
from tests.accounting.export import delivery_fixtures as fx
from tests.accounting.ledger.reduce_builders import source_session
from tests.accounting.views import ux_datasets as ds
from tests.accounting.views.test_ux_journeys import ANSI_OTHER, SGR, WIDTHS, keep
from tests.accounting.views.test_ux_support import (
    NOW,
    Binary,
    collected_scene,
    current_fact_consumer,
    current_rarity_index,
    fingerprint,
)

DAY12 = records_scene.at(hour=20)
DAY22 = DAY12 + timedelta(days=10)
DAY57 = DAY12 + timedelta(days=45)


def renamed(session: str, records: list[dict]) -> list[dict]:
    return [{**r, "sessionId": session} if "sessionId" in r else dict(r) for r in records]


def put(root: Path, name: str, records: list[dict]) -> None:
    vs.put(vs.projects(root), name, renamed(name, records), records_scene.at(hour=10))


def start(root: Path) -> None:
    vs.enable()
    vs.start_capture(day=11)
    put(root, "src1", source_session("src1", vs.fact(0), when_day=11))
    put(root, "first", current_fact_consumer())
    put(root, "second", current_fact_consumer())  # a second consumer of the SAME source fact


def receipts_dir(root: Path) -> Path:
    return root / "data" / "accounting" / "personal-local" / "receipts"


def receipt_ids(root: Path) -> list[str]:
    return sorted(p.stem for p in receipts_dir(root).glob("*.json"))


@pytest.fixture
def overlap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    current_rarity_index(tmp_path, monkeypatch)
    start(tmp_path)
    assert records_scene.collect(DAY12).state == "published"
    return tmp_path


def test_two_consumers_of_one_source_are_each_explained_and_credited_per_task(
    overlap: Path,
) -> None:
    run = Binary(overlap)
    ids = receipt_ids(overlap)
    assert len(ids) == 2
    views = [service.read_receipt("personal-local", i) for i in ids]
    assert all(v is not None and v.revision == 1 for v in views)
    credits = {v.root: dict(v.credit_raw)["central"] for v in views if v is not None}
    assert set(credits) == {"claude:first", "claude:second"}
    assert len(set(credits.values())) == 1 and all(value > 0 for value in credits.values())
    # overlap is removed WITHIN a task; two separate tasks that reused the same source are each
    # credited (design D6), so the all-task credit is the sum of the two task credits
    grouped = run.data("savings", "--no-refresh", "--group-by", "task")["data"]["groups"]["rows"]
    by_task = {row["key"]: row["estimates"] for row in grouped}
    first = by_task["claude:first#turn:u1"]
    second = by_task["claude:second#turn:u1"]
    both = run.data("savings")["data"]["estimates"]
    cells = [x["context_tokens"]["central"]["credit"]["value"] for x in (first, second, both)]
    assert cells[0] == cells[1] and cells[0] > 0
    assert cells[0] == credits["claude:first"] // 4
    assert cells[1] == credits["claude:second"] // 4
    assert cells[2] == sum(credits.values()) // 4, "convert once after summing per-task characters"
    for view in views:
        assert view is not None and view.scenarios
        assert view.raw_unit == "characters" and view.debit_raw is not None
        assert view.debit_raw_min == view.debit_raw == view.debit_raw_max
        assert view.debit_raw >= 0
        raw_credits = dict(view.credit_raw)
        for estimate in view.scenarios:
            assert estimate.credit.value is not None and estimate.debit.value is not None
            assert estimate.complete
            raw_credit = raw_credits[estimate.scenario.value]
            assert estimate.credit.value == raw_credit // 4
            assert estimate.debit.value == view.debit_raw // 4
            exact_net = Fraction(raw_credit - view.debit_raw, 4)
            assert estimate.net.value == estimate.credit.value - estimate.debit.value
            assert estimate.bounds is not None
            lower, upper = estimate.bounds.net_lower.value, estimate.bounds.net_upper.value
            assert lower is not None and upper is not None
            assert lower == floor(exact_net) and upper == ceil(exact_net)
            assert lower <= exact_net <= upper


def test_explain_shows_costs_exclusions_provenance_and_hides_content_by_default(
    overlap: Path,
) -> None:
    run = Binary(overlap)
    receipt_id = receipt_ids(overlap)[0]
    plain = run("savings", "explain", receipt_id, columns=80)
    assert plain.returncode == 0
    for heading in (
        "CREDIT BY SCENARIO",
        "COST OF THE LOOKUP",
        "NOT CREDITED",
        "EVIDENCE AVAILABILITY",
        "PROVENANCE",
    ):
        assert heading in plain.stdout
    assert "Private content is hidden" in plain.stdout
    stored = json.loads((receipts_dir(overlap) / f"{receipt_id}.json").read_text())
    assert "content" not in stored, "the collector never persists private content"
    assert stored["schema"] == 1 and set(stored["identities"]) == {
        "frequency",
        "tokenizer",
        "pricing",
    }
    data = run.data("savings", "explain", receipt_id)["data"]
    assert data["content"] is None and data["schema"] == 1 and data["unit"]
    assert data["credit_raw"]["central"] >= data["credit_raw"]["conservative"]
    assert data["recomputation_available"] is True
    assert set(data["identities"]) == {"frequency", "pricing", "tokenizer"}


def test_include_content_is_explicit_local_and_never_invented(overlap: Path) -> None:
    run = Binary(overlap)
    receipt_id = receipt_ids(overlap)[0]
    path = receipts_dir(overlap) / f"{receipt_id}.json"
    shown = run("savings", "explain", receipt_id, "--include-content")
    assert (
        "Nothing to show" in " ".join(shown.stdout.split()) and "no private content" in shown.stdout
    )
    document = json.loads(path.read_text())
    document["content"] = "LOCAL-ONLY-CANARY-7731"
    path.write_text(json.dumps(document))
    path.chmod(0o600)
    assert "LOCAL-ONLY-CANARY-7731" not in run("savings", "explain", receipt_id).stdout
    assert "LOCAL-ONLY-CANARY-7731" not in run("savings", "explain", receipt_id, "--json").stdout
    assert (
        "LOCAL-ONLY-CANARY-7731"
        in run("savings", "explain", receipt_id, "--include-content").stdout
    )
    with_json = run.data("savings", "explain", receipt_id, "--include-content")["data"]
    assert with_json["content"] == "LOCAL-ONLY-CANARY-7731"
    for name in ("snapshots", "history"):  # content never leaks into anything exportable
        for file in (overlap / "data" / "accounting" / "personal-local" / name).rglob("*"):
            if file.is_file():
                assert b"LOCAL-ONLY-CANARY-7731" not in file.read_bytes()


def test_unknown_ids_exit_3_in_text_and_json_and_never_look_like_a_zero_receipt(
    overlap: Path,
) -> None:
    run = Binary(overlap)
    for bad in ("0" * 24, "unknown-id", "..%2f..%2fetc", "a" * 65):
        done = run("savings", "explain", bad)
        assert done.returncode == 3 and done.stdout == "" and "No receipt" in done.stderr, bad
        code, body = run.json("savings", "explain", bad)
        assert code == 3 and body["condition"] == "receipt_not_found" and body["ok"] is False


def test_late_evidence_for_a_sealed_day_makes_a_superseding_revision_history_never_double_counts(
    overlap: Path,
) -> None:
    run = Binary(overlap)
    before = run.data("savings", "history")["data"]
    assert [(e["period"], e["revision"], e["current"]) for e in before["entries"]] == [
        ("2026-10-12", 1, True)
    ]
    put(overlap, "late", current_fact_consumer())  # a third consumer arriving after the seal
    assert records_scene.collect(DAY22).state == "published"
    history = run.data("savings", "history")["data"]
    entries = history["entries"]
    revisions = [(e["revision"], e["current"], e["supersedes"], e["kind"]) for e in entries]
    assert revisions == [(1, False, None, "original"), (2, True, 1, "adjustment")]
    original, replacement = entries[0]["net"]["value"], entries[1]["net"]["value"]
    assert original is not None and replacement is not None
    assert history["current_total"]["value"] == replacement, "only the applicable revision counts"
    assert history["current_total"]["value"] != original + replacement
    text = " ".join(run("savings", "history", columns=40).stdout.split())
    assert "superseded" in text and "adjustment" in text and "Applicable revisions only" in text
    ids = receipt_ids(overlap)
    latest = [service.read_receipt("personal-local", i) for i in ids]
    assert all(v is not None for v in latest)


def test_history_pages_are_bound_to_the_data_and_reject_a_moved_cursor(overlap: Path) -> None:
    run = Binary(overlap)
    put(overlap, "late", current_fact_consumer())
    assert records_scene.collect(DAY22).state == "published"
    first = run.data("savings", "history", "--limit", "1")["data"]
    cursor = first["next_cursor"]
    assert cursor and len(first["entries"]) == 1
    second = run.data("savings", "history", "--limit", "1", "--page", cursor)["data"]
    assert [e["revision"] for e in first["entries"] + second["entries"]] == [1, 2]
    assert second["next_cursor"] is None
    done = run("savings", "history", "--limit", "2", "--page", cursor, "--json")
    assert done.returncode == 2 and json.loads(done.stdout)["condition"] == "invalid_cursor"
    put(overlap, "later", current_fact_consumer())
    assert records_scene.collect(DAY22 + timedelta(hours=1)).state == "published"
    done = run("savings", "history", "--limit", "1", "--page", cursor, "--json")
    assert done.returncode == 2, "the data changed: the old cursor must die, not skip rows"


def test_expired_evidence_keeps_totals_marks_not_recomputable_and_says_so_everywhere(
    overlap: Path,
) -> None:
    run = Binary(overlap)
    totals = run.data("savings", "history")["data"]["current_total"]["value"]
    assert records_scene.collect(DAY57).state == "published"
    ids = receipt_ids(overlap)
    expired = [service.read_receipt("personal-local", i) for i in ids]
    assert all(v is not None and not v.evidence_available for v in expired)
    explained = run("savings", "explain", ids[0])
    text = " ".join(explained.stdout.split())
    assert explained.returncode == 0
    assert "EXPIRED" in text and "Detailed evidence has expired" in text
    assert (
        "cannot be recomputed from source" in text and "Full recomputation is unavailable" in text
    )
    data = run.data("savings", "explain", ids[0])
    assert data["state"] == "evidence_expired" and data["data"]["recomputation_available"] is False
    assert any("evidence_expired" in w for w in data["warnings"])
    history = run.data("savings", "history")["data"]
    assert history["current_total"]["value"] == totals, "retained closed totals did not shrink"
    assert all(e["recomputable"] is False for e in history["entries"])
    content = run("savings", "explain", ids[0], "--include-content")
    assert "evidence has expired, so no content can be shown" in " ".join(content.stdout.split())
    summary = run.data("savings")
    assert any(c["code"] == "evidence_expired" for c in summary["data"]["conditions"])


def test_a_newer_schema_history_is_unavailable_not_rewritten(overlap: Path) -> None:
    run = Binary(overlap)
    path = overlap / "data" / "accounting" / "personal-local" / "history" / "periods.json"
    rows = json.loads(path.read_text())
    rows[0]["schema"] = 7
    path.write_text(json.dumps(rows))
    path.chmod(0o600)
    before = path.read_bytes()
    done = run("savings", "history", "--json")
    body = json.loads(done.stdout)["data"]
    assert done.returncode == 0 and body["state"] == "empty"
    assert body["data"]["reason"] == "history_unsupported_schema"
    assert body["data"]["current_total"]["value"] is None
    assert body["next_actions"][0]["command"] == "ssgrep savings status"
    assert path.read_bytes() == before


def test_damaged_receipt_is_a_specific_exit_1_not_a_not_found(overlap: Path) -> None:
    run = Binary(overlap)
    receipt_id = receipt_ids(overlap)[0]
    path = receipts_dir(overlap) / f"{receipt_id}.json"
    path.write_text("{not json")
    path.chmod(0o600)
    code, body = run.json("savings", "explain", receipt_id)
    assert code == 1 and body["condition"] == "receipt_unreadable"
    assert body["command"] == "ssgrep savings status"
    before = fingerprint(overlap / "data")
    run("savings", "explain", receipt_id)
    assert fingerprint(overlap / "data") == before


ENVELOPE = {"schema_version", "state", "data", "warnings", "next_actions", "provenance"}
ERROR = {"ok", "condition", "message", "command"}
ANSI = re.compile(r"\x1b|\x9b")
FLOAT = re.compile(r"(?<![\w.\"])-?\d+\.\d+(?:[eE][+-]?\d+)?(?![\w.\"])")


def no_float(text: str) -> Any:
    raise AssertionError(f"float in JSON output: {text}")


def parse(stdout: str) -> dict[str, Any]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    assert len(lines) == 1, f"exactly one JSON document expected, got {len(lines)}: {stdout!r}"
    assert not ANSI.search(stdout), "ANSI escape in JSON"
    assert "NaN" not in stdout and "Infinity" not in stdout
    return json.loads(lines[0], parse_float=no_float)


def shape_ok(body: dict[str, Any], expected: int) -> None:
    if expected == 0:
        assert body["ok"] is True and ENVELOPE <= set(body["data"]), body
    elif "error" in body:  # usecli's own document for argument parsing errors
        assert body["ok"] is False and {"type", "message", "code"} <= set(body["error"])
        assert body["error"]["code"] == expected
    else:
        assert ERROR <= set(body) and body["ok"] is False, body
        assert isinstance(body["condition"], str) and body["condition"]
        assert isinstance(body["message"], str) and isinstance(body["command"], str)


#: (arguments, expected exit, expected condition-or-state, must leave files unchanged)
EMPTY_ROWS = [
    (("savings",), 0, "disabled", True),
    (("savings", "history"), 0, "empty", True),
    (("savings", "explain", "nope"), 3, "receipt_not_found", True),
    (("savings", "explain", "../../etc/passwd"), 3, "receipt_not_found", True),
    (("savings", "history", "--limit", "101"), 2, "invalid_limit", True),
    (("savings", "history", "--limit", "0"), 2, "invalid_limit", True),
    (("savings", "--from", "yesterday"), 2, "invalid_interval", True),
    (
        ("savings", "--from", "2026-10-02", "--to", "2026-10-01"),
        2,
        "invalid_interval",
        True,
    ),
    (("savings", "--runtime", "vim"), 2, "invalid_filter", True),
    (("savings", "lookups", "--children"), 2, "invalid_filter", True),
    (("savings", "history", "--page", "h1.1.deadbeefdeadbeef"), 2, "invalid_cursor", True),
    (("savings", "--profile", "../x"), 2, "invalid_profile", True),
    (("savings", "--bogus"), 2, None, True),
    (("savings", "explain"), 2, None, True),
    (("savings", "--by", "colour"), 2, None, True),
    (("savings", "--cursor", "garbage"), 2, None, True),
    (("savings", "explain", "x", "--profile", "A/B"), 2, None, True),
    (("metrics", "status"), 2, None, True),
    (("savings", "collect"), 1, "collection_disabled", True),
    (("savings", "collect", "--backfill"), 2, "backfill_requires_interval", True),
    (("savings", "collect", "--since", "2026-01-01"), 2, None, True),
    (
        ("savings", "collect", "--backfill", "--recompute", "2026-01-01", "2026-02-01"),
        2,
        "conflicting_modes",
        True,
    ),
    (("savings", "collect", "--profile", "A/B"), 2, "invalid_profile", True),
    (("savings", "setup"), 2, "confirmation_required", True),
    (("savings", "export", "preview"), 2, None, True),
    (("savings", "export", "preview", "x"), 2, "unknown_destination", True),
    (("savings", "purge", "2026-01-01"), 2, "confirmation_required", True),
    (("savings", "purge"), 2, None, True),
    (("savings", "export", "serve", "--profile", "../x"), 2, "invalid_profile", True),
    (("metrics", "census"), 2, None, True),
    (("savings", "pause", "--profile", "../x"), 2, "invalid_profile", True),
    (("savings", "resume", "--profile", "../x"), 2, "invalid_profile", True),
]


@pytest.fixture
def run(tmp_path: Path) -> Binary:
    return Binary(tmp_path)


@pytest.mark.parametrize(
    ("args", "code", "label", "readonly"), EMPTY_ROWS, ids=lambda v: str(v)[:60]
)
def test_matrix_row_on_an_empty_profile(
    run: Binary, args: tuple[str, ...], code: int, label: str | None, readonly: bool
) -> None:
    before = fingerprint(run.root / "data") if (run.root / "data").exists() else {}
    done = run(*args, "--json")
    assert done.returncode == code, (args, done.stdout, done.stderr)
    body = parse(done.stdout)
    shape_ok(body, code)
    if label is not None:
        got = body["data"]["state"] if code == 0 else body["condition"]
        assert got == label, body
    if readonly:
        after = fingerprint(run.root / "data") if (run.root / "data").exists() else {}
        assert after == before, "a rejected or read-only command changed files"


def test_paused_mutations_exit_1_with_stable_conditions_and_readonly_commands_still_work(
    run: Binary,
) -> None:
    assert run("savings", "setup", "--yes", "--json").returncode == 0
    assert run("savings", "pause", "--json").returncode == 0
    before = fingerprint(run.root / "data")
    code, body = run.json("savings", "collect")
    assert code == 1 and body["condition"] == "collection_paused"
    assert body["command"] == "ssgrep savings resume --profile personal-local"
    for readonly in (("savings",), ("savings", "history"), ("savings", "status")):
        done = run(*readonly, "--json")
        assert done.returncode == 0, readonly
        assert parse(done.stdout)["data"]["state"] in {"paused", "empty"}
    assert fingerprint(run.root / "data") == before, "paused reads and refusals wrote nothing"
    code, resumed = run.json("savings", "resume")
    assert code == 0 and resumed["data"]["state"] == "resumed"


def test_paused_export_send_and_check_exit_1_export_paused_and_preview_stays_usable(
    run: Binary, tmp_path: Path
) -> None:
    from dataclasses import replace

    from ssgrep.accounting.ledger.ledger_contracts import DestinationPolicy, Signal
    from ssgrep.accounting.privacy import policy

    policy.enable_collection("personal-local")
    current = policy.load_profile_policy("personal-local")
    dest = DestinationPolicy(
        "org-primary", "https://collector.example.test:4318", (Signal.METRICS,)
    )
    policy.save_profile(replace(current, destinations=(dest,)))
    assert run("savings", "pause", "--json").returncode == 0
    for mode in ("send", "check"):
        done = run("savings", "export", mode, "org-primary", "--json")
        assert done.returncode == 1, (mode, done.stdout, done.stderr[-300:])
        body = parse(done.stdout)
        shape_ok(body, 1)
        assert body["condition"] == "export_paused"
    preview = run("savings", "export", "preview", "org-primary", "--json")
    assert preview.returncode == 0, preview.stderr[-300:]
    parse(preview.stdout)


def test_successes_over_a_real_collection_are_float_free_ansi_free_one_document(
    tmp_path: Path,
) -> None:
    collected_scene(tmp_path)
    run = Binary(tmp_path)
    receipt = run.data("savings")["provenance"]["snapshot_id"]
    assert receipt
    rows = [
        ("savings",),
        ("savings", "--group-by", "task"),
        ("savings", "--group-by", "day"),
        ("savings", "--runtime", "claude", "--from", "2026-10-12", "--to", "2026-10-13"),
        ("savings", "history"),
        ("savings", "status"),
        ("savings", "status", "--sources"),
    ]
    for args in rows:
        done = run(*args, "--json", env={"FORCE_COLOR": "1", "NO_COLOR": ""})
        assert done.returncode == 0, (args, done.stderr[-300:])
        body = parse(done.stdout)
        shape_ok(body, 0)
        assert not FLOAT.search(done.stdout), (args, FLOAT.search(done.stdout))
    explained = run.data("savings")["data"]["representative_receipt"]
    done = run("savings", "explain", explained, "--json", env={"FORCE_COLOR": "1"})
    assert done.returncode == 0 and ENVELOPE <= set(parse(done.stdout)["data"])
    unknown = run("savings", "explain", "ffffffffffffffffffffffff", "--json")
    assert unknown.returncode == 3
    assert parse(unknown.stdout) == {
        "ok": False,
        "condition": "receipt_not_found",
        "message": "No receipt with that id exists for this profile.",
        "command": "ssgrep savings",
    }


def test_unknown_values_are_null_with_a_reason_never_zero_in_json(tmp_path: Path) -> None:
    collected_scene(tmp_path)
    run = Binary(tmp_path)
    data = run.data("savings", "--runtime", "codex")["data"]
    net = data["estimates"]["context_tokens"]["central"]["net"]
    assert net["value"] is None and net["reason"] and net["basis"] == "unavailable"
    by_task = run.data("savings", "--no-refresh", "--group-by", "task")["data"]["groups"]["rows"]
    unknown = next(r for r in by_task if r["key"] == "task_unknown")
    cell = unknown["estimates"]["context_tokens"]["central"]["net"]
    assert cell["value"] is None and cell["reason"]
    status = run.data("savings", "status")["data"]
    assert status["lag"]["lag_bytes"] is not None or status["lag"]["reason"]
    assert all(
        v is None or isinstance(v, int)
        for v in (
            status["storage_pressure"]["free_bytes"],
            status["storage_pressure"]["total_bytes"],
        )
    )


def test_a_newer_schema_receipt_exits_3_unavailable_and_is_not_rewritten(tmp_path: Path) -> None:
    collected_scene(tmp_path)
    run = Binary(tmp_path)
    receipt_id = run.data("savings")["data"]["representative_receipt"]
    target = tmp_path / "data" / "accounting" / "personal-local" / "receipts" / f"{receipt_id}.json"
    document = json.loads(target.read_text())
    document["schema"] = 99
    target.write_text(json.dumps(document))
    target.chmod(0o600)
    before = target.read_bytes()
    done = run("savings", "explain", receipt_id, "--json")
    assert done.returncode == 3
    body = parse(done.stdout)
    assert body["condition"] == "receipt_unsupported_schema" and "newer schema" in body["message"]
    assert target.read_bytes() == before, "an unsupported receipt is never rewritten"


COLLECTED = scene.at(hour=20)


@pytest.fixture
def collected(tmp_path: Path) -> Path:
    collected_scene(tmp_path)
    return tmp_path


@pytest.mark.parametrize(
    ("age", "stale"),
    [(0, False), (60, False), (179, False), (180, False), (181, True), (3600, True)],
)
def test_staleness_begins_after_exactly_three_intervals_of_180_seconds(
    collected: Path, age: int, stale: bool
) -> None:
    view = service.read_summary(now=COLLECTED + timedelta(seconds=age))
    assert view.stale is stale and view.age_seconds == age
    assert view.state is (SnapshotState.STALE if stale else SnapshotState.FRESH)
    assert view.has_values, "stale keeps the last good values"
    status = read_status("personal-local", now=COLLECTED + timedelta(seconds=age))
    assert status.stale is stale and status.stale_after_seconds == 180


def test_pause_is_a_separate_state_and_stale_age_continues_while_paused(collected: Path) -> None:
    policy.pause("personal-local")
    fresh = service.read_summary(now=COLLECTED + timedelta(seconds=10))
    old = service.read_summary(now=COLLECTED + timedelta(seconds=999))
    assert fresh.state is SnapshotState.PAUSED and not fresh.stale
    assert old.state is SnapshotState.PAUSED and old.stale, "paused AND stale are both visible"
    assert old.age_seconds == 999


def test_status_reports_coverage_lag_storage_and_both_freshness_clocks(collected: Path) -> None:
    status = read_status("personal-local", now=COLLECTED + timedelta(seconds=45))
    assert status.collection_age_seconds == 45
    assert status.delivery_age_seconds is None, "nothing delivered: unknown, not 45"
    assert status.lag.lag_bytes == 0 and status.lag.sources_total
    assert status.missing.missing_sources == 0
    assert status.storage.level in {"ok", "high", "critical"}
    assert {c.name for c in status.storage.components} >= {"snapshots", "receipts", "history"}
    runtimes = {r.runtime for r in status.lag.by_runtime}
    assert "claude" in runtimes


def test_status_json_exposes_the_same_fields_as_integers_and_nulls(collected: Path) -> None:
    body = Binary(collected).data("savings", "status")["data"]
    assert body["snapshot"]["stale_after_seconds"] == 180
    assert isinstance(body["snapshot"]["age_seconds"], int)
    assert body["freshness"]["delivery"]["age_seconds"] is None
    assert body["lag"]["lag_bytes"] == 0 and body["lag"]["by_runtime"]
    assert body["storage_pressure"]["level"] in {"ok", "high", "critical"}
    assert body["missing_evidence"]["missing_sources"] == 0
    assert body["watermarks"]["count"] and body["watermarks"]["items"]
    assert "acknowledgment" in body["freshness"]["delivery"]


# ---- reads change nothing ----

READS = [
    ("savings",),
    ("savings", "--group-by", "task"),
    ("savings", "--runtime", "claude", "--from", "2026-10-01"),
    ("savings", "history"),
    ("savings", "status"),
    ("savings", "status", "--sources"),
]


def test_repeated_real_reads_leave_every_file_and_directory_untouched(collected: Path) -> None:
    run = Binary(collected)
    receipt = run.data("savings")["data"]["representative_receipt"]
    commands = [
        *READS,
        ("savings", "explain", receipt),
        ("savings", "explain", receipt, "--include-content"),
    ]
    root = collected / "data"
    before = fingerprint(root)
    state_before = fingerprint(collected / "claude")
    for _ in range(2):
        for args in commands:
            concrete = tuple(a.replace("{home}", str(collected / "home")) for a in args)
            for mode in ((), ("--json",)):
                done = run(*concrete, *mode)
                assert done.returncode == 0, (concrete, done.stderr[-300:])
    assert fingerprint(root) == before, "reading created, touched or rewrote an accounting file"
    assert fingerprint(collected / "claude") == state_before, "source transcripts were touched"


def test_reads_leave_delivery_acknowledgments_and_checkpoints_exactly_as_they_were(
    collected: Path,
) -> None:
    from dataclasses import replace

    dest = DestinationPolicy(
        "org-primary", "https://collector.example.test:4318", (Signal.METRICS,)
    )
    policy.save_profile(
        replace(
            policy.load_profile_policy("personal-local"),
            destinations=(dest,),
            enabled_exports=(("org-primary", "metrics"),),
        )
    )
    acks.update_delivery(
        "personal-local",
        "org-primary",
        Signal.METRICS,
        lambda d: acks.retry_scheduled(
            acks.set_counts(acks.collected(d, 1), pending_batches=2, unknown_batches=0),
            "2026-10-12T20:05:00Z",
            "network_error",
        ),
    )
    root = collected / "data" / "accounting" / "personal-local"
    watched = ("delivery", "collector", "snapshots", "receipts", "history")
    before = {name: fingerprint(root / name) for name in watched}
    run = Binary(collected)
    for args in (("savings", "--no-refresh"), ("savings", "status"), ("savings", "history")):
        for _ in range(2):
            assert run(*args).returncode == 0 and run(*args, "--json").returncode == 0
    after = {name: fingerprint(root / name) for name in watched}
    assert after == before
    line = read_status("personal-local", now=NOW).delivery_lines[0]
    assert line.pending_batches == 2 and line.next_retry_utc == "2026-10-12T20:05:00Z"


SECRET_FIELDS = re.compile(r"sk-CANARY|Bearer ")


def screens(run: Binary, name: str) -> dict[int, str]:
    out: dict[int, str] = {}
    for width in WIDTHS:
        done = run("savings", "status", columns=width, env=fx.ENV)
        assert done.returncode == 0 and not done.stderr, done.stderr[-300:]
        text = done.stdout
        assert not SGR.search(text) and not ANSI_OTHER.search(text), "NO_COLOR, no controls"
        assert max(len(line) for line in text.splitlines()) <= width
        assert not SECRET_FIELDS.search(text)
        keep(f"{name}-status", width, text)
        summary = run("savings", columns=width, env=fx.ENV)
        assert summary.returncode == 0
        keep(f"{name}-savings", width, summary.stdout)
        out[width] = text
    return out


def delivery_line(run: Binary) -> dict:
    code, body = run.json("savings", "status")
    assert code == 0
    lines = [ln for d in body["data"]["data"]["destinations"] for ln in d["lines"]]
    assert len(lines) == 1, lines
    return lines[0]


@pytest.fixture
def isolated(tmp_path: Path) -> Path:
    return tmp_path


def test_offline_backlog_through_the_real_binary_is_pending_with_retry_never_synced(
    isolated: Path,
) -> None:
    with ds.delivery("offline", isolated) as (run, _):
        line = delivery_line(run)
        assert line["state"] == "offline", line
        assert line["pending_batches"] >= 1 and line["next_retry_utc"]
        assert line["acked_generation"] is None and line["last_acked_at"] is None
        assert line["next_action"]["command"] == f"ssgrep savings export check {fx.DEST}", line
        for width, text in screens(run, "offline").items():
            flat = " ".join(text.split())
            assert "Not synced" in flat and "pending" in flat.lower(), width
            assert "Delivery outcome unknown" not in flat
        wide = screens(run, "offline")[120]
        assert wide.count("Destination unreachable") == 1


def test_credentials_expired_through_the_real_binary_names_rotation_and_no_secret(
    isolated: Path,
) -> None:
    with ds.delivery("auth_error", isolated) as (run, receiver):
        assert receiver is not None and receiver.requests and receiver.requests[0].status == 401
        line = delivery_line(run)
        assert line["state"] == "credentials_expired" and line["blocked_reason"]
        assert line["next_retry_utc"] is None
        assert line["next_action"]["command"] == f"ssgrep savings export check {fx.DEST}", line
        for _width, text in screens(run, "auth").items():
            assert fx.SECRET not in text and "token rejected" not in text


def test_partial_rejection_through_the_real_binary_is_never_called_acknowledged(
    isolated: Path,
) -> None:
    with ds.delivery("partial_export", isolated) as (run, receiver):
        assert receiver is not None and receiver.requests
        line = delivery_line(run)
        assert line["state"] == "partial_export", line
        assert line["rejected_points"] == 4 and line["acked_generation"] is None
        for width, text in screens(run, "partial").items():
            flat = " ".join(text.split())
            assert "Partial export" in flat and "rejected points 4" in flat, width
            assert "synthetic rejection" not in text


def test_paused_profile_with_pending_export_keeps_the_backlog_and_names_resume(
    isolated: Path,
) -> None:
    with ds.delivery("paused_pending", isolated) as (run, _):
        line = delivery_line(run)
        assert line["next_action"]["command"] == "ssgrep savings resume", line
        assert line["pending_batches"] >= 1
        code, body = run.json("savings")
        assert code == 0 and body["data"]["state"] == "paused"
        summary_actions = [a["command"] for a in body["data"]["next_actions"]]
        assert "ssgrep savings resume" in summary_actions
        for width, text in screens(run, "paused-pending").items():
            flat = " ".join(text.split())
            assert "PAUSED" in flat and "resume" in flat, width
        before = fx.tree()
        send = run("savings", "export", "send", fx.DEST, "--json", env=fx.ENV)
        assert send.returncode == 1 and parse(send.stdout)["condition"] == "export_paused"
        assert fx.tree() == before


@pytest.mark.parametrize("mode", ["send", "check"])
def test_paused_export_exits_1_before_any_import_or_network_text_and_json(
    isolated: Path, mode: str
) -> None:
    with ds.delivery("offline", isolated) as (run, receiver):
        assert receiver is None
        assert run("savings", "pause", "--json", env=fx.ENV).returncode == 0
        before = fx.tree()
        base = ("savings", "export", mode, fx.DEST)
        as_json = run(*base, "--json", env=fx.ENV)
        assert as_json.returncode == 1 and as_json.stderr == ""
        body = parse(as_json.stdout)
        assert body["condition"] == "export_paused" and body["ok"] is False
        assert body["command"].endswith("savings resume")
        text = run(*base, env=fx.ENV)
        assert text.returncode == 1 and "paused" in (text.stdout + text.stderr).lower()
        assert "Traceback" not in text.stderr and "ImportError" not in text.stderr
        assert fx.tree() == before


EXPECT = {
    "disabled": (SnapshotState.DISABLED, {"disabled"}),
    "no_snapshot": (SnapshotState.NO_SNAPSHOT, set()),
    "fresh": (SnapshotState.FRESH, {"partial", "negative"}),
    "no_reuse": (SnapshotState.FRESH, {"no_reuse"}),
    "stale": (SnapshotState.STALE, {"stale"}),
    "paused": (SnapshotState.PAUSED, {"paused"}),
    "evidence_expired": (SnapshotState.FRESH, {"evidence_expired"}),
    "overlap": (SnapshotState.FRESH, set()),
    "revised": (SnapshotState.FRESH, set()),
}


@pytest.mark.parametrize("name", ds.STATES)
def test_each_dataset_reproduces_its_state_and_conditions(
    name: str, isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if name == "fresh":
        current_rarity_index(isolated, monkeypatch)
    built = ds.BUILDERS[name](isolated)
    view = service.read_summary(now=built.now)
    expected_state, expected_conditions = EXPECT[name]
    assert view.state is expected_state, (name, view.state)
    assert expected_conditions <= {c.code for c in view.conditions}, name
    for condition in view.conditions:
        assert condition.cause
    assert not any(q.value == 0 and q.basis.value == "unavailable" for _, q in view.measured)
    json.dumps({"state": view.state.value, "conditions": [c.code for c in view.conditions]})


def test_policy_discards_clear_active_offline_status_without_relabeling_loss(
    isolated: Path,
) -> None:
    with ds.delivery("offline", isolated) as (run, _):
        before = delivery_line(run)
        assert before["state"] == "offline" and before["pending_batches"] > 0
        fx.amend(roots=(str(isolated / "narrower-workspace"),))
        run("savings", "export", "send", fx.DEST, "--json", env=fx.ENV)
        durable = fx.delivery()
        assert durable is not None
        assert durable.pending_batches == 0 and durable.lost_records > 0
        assert durable.loss_summary[0].reason == "policy_contraction"
        after = delivery_line(run)
        assert after["state"] == "waiting" and after["lost_records"] == durable.lost_records
        assert "policy_contraction" in after["cause"]
        assert "expired" not in after["cause"] and "unreachable" not in after["cause"]
        assert after["next_action"]["command"] == f"ssgrep savings export preview {fx.DEST}"
