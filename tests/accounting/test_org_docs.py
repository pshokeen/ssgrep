"""Organization dashboard doc + synthetic OTLP examples cannot drift from the code (14.5/14.6).

``docs/examples/otlp-*.json`` are generated from real builders over synthetic current state.
Both generated and checked-in examples retain schema, privacy and receiver invariants.
``docs/organization-metrics.md`` SQL examples execute against synthetic rows. Real
PromQL consumer cases execute in ``test_metrics_scrape.py``.

Regenerate the examples after an intentional serializer change with
``uv run python -m tests.accounting.test_org_docs``.
"""

from __future__ import annotations

import json
import os
import random
import re
import sqlite3
import tempfile
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from ssgrep.accounting import (
    collector_store,
    export,
    export_build,
    export_consent,
    export_records,
    policy as policy_store,
)
from ssgrep.accounting.contracts import TaskId, digest_of
from ssgrep.accounting.export_records_plan import build_deletions
from ssgrep.accounting.ledger_contracts import DestinationPolicy, ProfilePolicy, Signal
from ssgrep.accounting.receiver_contract import (
    Key,
    Outcome,
    ReceiverState,
    apply,
    current_totals,
)
from ssgrep.accounting.valuation_inputs import EvidenceBundle
from tests.accounting.export_canary import build_ledger, decimals, parse_strict, persist_ledger
from tests.accounting.http_builders import (
    make_snapshot,
    synthetic_current_checkpoint,
    window_with_unknown_optimistic,
)
from tests.accounting.valuation_builders import METHOD, candidate, debit

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "organization-metrics.md"
EXAMPLES = ROOT / "docs" / "examples"
METRICS_FILE = "otlp-metrics-request.json"
LOGS_FILE = "otlp-logs-request.json"
TOMBSTONE_FILE = "otlp-logs-tombstone-request.json"
PROFILE, DEST = "personal-local", "acme-org"
KEY = bytes(range(32))  # synthetic pseudonym key
VERSION = "0.0.0-synthetic"
RECEIVER_KEY = ("tenant-a", "device-1")  # what the receiver derives from the enrolled credential
STAMP = 1_791_892_800  # 2026-10-13T12:00:00Z, the tombstone batch time
SCENARIOS = ("conservative", "central", "optimistic")


# ---- example generation (real builders) ------------------------------------------------------


def _refuse(token: str) -> Any:
    raise ValueError(f"non-finite JSON constant {token}")


def _forbid_float(text: str) -> Any:
    raise AssertionError(f"float in payload: {text}")


def _pretty(wire: bytes) -> bytes:
    """Indented deterministic rendering of the wire bytes (OTLP doubles pass through float)."""
    document = json.loads(wire, parse_float=Decimal, parse_constant=_refuse)
    return (json.dumps(document, indent=2, default=float) + "\n").encode()


def _ledger(first_task_debit: int) -> Any:
    """Two synthetic tasks; only the first task's debit changes between revisions."""
    other = TaskId("root-2", "turn-1", "main")
    bundle = EvidenceBundle(
        (candidate("a"), candidate("b", task=other)),
        (debit("x", first_task_debit), debit("y", 800, task=other)),
    )
    return build_ledger(bundle, METHOD)


def render_examples(data_dir: Path) -> dict[str, tuple[bytes, bytes]]:
    """name -> (exact wire bytes, pretty example bytes) for the three request examples."""
    dest = DestinationPolicy(
        DEST, "https://collector.example.test", (Signal.METRICS, Signal.LOGS), logs_permitted=True
    )
    policy = ProfilePolicy(
        destinations=(dest,), enabled_exports=((DEST, "metrics"), (DEST, "logs"))
    )
    with (
        mock.patch.dict(os.environ, {"SSGREP_DATA_DIR": str(data_dir)}),
        mock.patch.object(export_build, "installed_version", return_value=VERSION),
    ):
        current = policy_store.save_profile(policy)
        checkpoint = synthetic_current_checkpoint(PROFILE, current)
        collector_store.commit_checkpoint(PROFILE, checkpoint)
        for signal in (Signal.METRICS, Signal.LOGS):
            export_consent.apply_enable(PROFILE, DEST, signal)
        policy = policy_store.load_profile_policy(PROFILE)
        first = make_snapshot(
            3, methodology_id=METHOD.hash, window=window_with_unknown_optimistic()
        )
        second = make_snapshot(4, extra_searches=1, methodology_id=METHOD.hash)
        metrics = export_build.build_metrics(PROFILE, dest, policy, first, KEY)
        persist_ledger(PROFILE, _ledger(640))
        initial = export_build.build_logs(PROFILE, dest, policy, first, KEY)
        sent = initial.planned[0][0]  # only record A's revision 1 is accepted before it changes
        export_records.mark_queued(PROFILE, DEST, [sent], "g0000000003-sent-0000")
        export_records.mark_acked(PROFILE, DEST, "g0000000003-sent-0000")
        persist_ledger(PROFILE, _ledger(1200))
        revised = export_build.build_logs(PROFILE, dest, policy, second, KEY)
        export_records.mark_queued(PROFILE, DEST, revised.planned[0], revised.batches[0].batch_id)
        export_records.mark_acked(PROFILE, DEST, revised.batches[0].batch_id)
        registry = export_records.load_registry(PROFILE, DEST)
        deletions = build_deletions(registry, [sent.record_id], key=KEY, destination=DEST)
        tomb = export_build.build_tombstones(PROFILE, dest, policy, deletions, KEY, stamp=STAMP)
    wires = {
        METRICS_FILE: metrics.batches[0].body,
        LOGS_FILE: revised.batches[0].body,
        TOMBSTONE_FILE: tomb.batches[0].body,
    }
    return {name: (wire, _pretty(wire)) for name, wire in wires.items()}


@pytest.fixture(scope="module", params=("generated", "checked-in"))
def examples(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, tuple[bytes, bytes]]:
    if request.param == "generated":
        return render_examples(tmp_path_factory.mktemp("org-docs-data"))
    return {
        name: ((EXAMPLES / name).read_bytes(), (EXAMPLES / name).read_bytes())
        for name in (METRICS_FILE, LOGS_FILE, TOMBSTONE_FILE)
    }


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        for name, (_, pretty) in render_examples(Path(tmp)).items():
            (EXAMPLES / name).write_bytes(pretty)


# ---- examples: drift, shape, sanitization ----------------------------------------------------


def _records(name: str, examples: dict[str, tuple[bytes, bytes]]) -> list[dict[str, Any]]:
    document = json.loads(examples[name][1])
    return document["resourceLogs"][0]["scopeLogs"][0]["logRecords"]


def test_log_examples_are_float_free_and_metric_fractions_are_only_histogram_doubles(
    examples: dict[str, tuple[bytes, bytes]],
) -> None:
    for name in (LOGS_FILE, TOMBSTONE_FILE):
        json.loads(examples[name][1], parse_float=_forbid_float, parse_constant=_refuse)
        for record in _records(name, examples):
            json.loads(record["body"]["stringValue"], parse_float=_forbid_float)
    paths = list(decimals(parse_strict(examples[METRICS_FILE][1])))
    assert paths  # the example does show histogram doubles
    for path in paths:
        leaf = [part for part in path if part != "[]"][-1]
        assert leaf in {"sum", "explicitBounds"}, path  # every other number is an int64 string


def _strings(node: Any) -> Iterator[tuple[str, str]]:
    """(key, string) for every string leaf, descending into the embedded ledger bodies."""
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, str):
                yield key, value
                if key == "stringValue" and value.startswith("{"):
                    yield from _strings(json.loads(value))
            else:
                yield from _strings(value)
    elif isinstance(node, list):
        for item in node:
            yield from _strings(item)


SECRET = re.compile(r"secret|password|passwd|bearer|authorization|api[_-]?key|private key", re.I)
PATHLIKE = re.compile(r"(^|\s)(/|~|[A-Za-z]:\\)|://|\.\./|\\")
IDENTITY = {"tenant", "device", "subject", "user", "email", "hostname"}


@pytest.mark.parametrize("name", [METRICS_FILE, LOGS_FILE, TOMBSTONE_FILE])
def test_examples_carry_no_paths_secrets_or_asserted_identity(
    examples: dict[str, tuple[bytes, bytes]], name: str, tmp_path: Path
) -> None:
    text = examples[name][1].decode()
    document = json.loads(text, parse_float=Decimal)
    for key, value in _strings(document):
        assert not PATHLIKE.search(value), (key, value)
        assert not SECRET.search(value), (key, value)
        assert value not in IDENTITY, (key, value)
    assert not SECRET.search(text)
    for marker in ("/Users", "/home", str(tmp_path), KEY.hex(), "root-1", "turn-1"):
        assert marker not in text, marker  # no path, key, or raw conversation-root id
    (resource,) = next(iter(document.values()))
    attributes = {a["key"] for a in resource["resource"]["attributes"]}
    assert attributes <= {
        "service.name",
        "service.version",
        "ssgrep.counter.epoch",
        "ssgrep.installation.id",
        "ssgrep.policy.version",
    }  # the client never asserts tenant, device, or subject


def _metric_points(doc: dict[str, Any]) -> Iterator[tuple[dict[str, Any], str, dict[str, Any]]]:
    for metric in doc["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]:
        kind = next(k for k in ("sum", "gauge", "histogram") if k in metric)
        for point in metric[kind]["dataPoints"]:
            yield metric, kind, point


def _labels(point: dict[str, Any]) -> dict[str, str]:
    return {a["key"]: a["value"]["stringValue"] for a in point["attributes"]}


def test_metrics_example_follows_the_export_catalog_and_shows_the_partial_negative_case(
    examples: dict[str, tuple[bytes, bytes]],
) -> None:
    document = parse_strict(examples[METRICS_FILE][1])
    optimistic = window_with_unknown_optimistic()["estimates"]["context_tokens"]["optimistic"]
    assert optimistic["credit"]["value"] == 20000
    assert optimistic["credit_missing"] == [["synthetic_source", "source_costs_missing"]]
    assert optimistic["net"]["value"] is None and optimistic["complete"] is False
    assert optimistic["known_net"] == 4000
    table = {row[0]: row for row in export._TABLE.values()}
    seen: set[str] = set()
    values: dict[tuple[str, str], int] = {}
    for metric, kind, point in _metric_points(document):
        name = metric["name"]
        _, want_kind, unit, _ = table[name]
        seen.add(name)
        assert (kind, metric["unit"]) == (want_kind, unit)
        labels = _labels(point)
        if kind in {"sum", "histogram"}:
            assert metric[kind]["aggregationTemporality"] == 2  # cumulative
            assert int(point["startTimeUnixNano"]) < int(point["timeUnixNano"])
        else:
            assert "startTimeUnixNano" not in point
        if kind == "sum":
            assert metric[kind]["isMonotonic"] is True
        if name in {
            "ssgrep_estimated_net_tokens",
            "ssgrep_estimated_net_work_seconds",
            "ssgrep_estimated_credit_tokens",
            "ssgrep_known_credit_tokens",
            "ssgrep_known_net_tokens",
        }:
            values[(name, labels["scenario"])] = int(point["asInt"])
        if name == "ssgrep_coverage_records" and labels["kind"] == "estimates":
            values[(name, labels["status"])] = int(point["asInt"])
    assert seen <= set(table) and "ssgrep_accounting_info" in seen
    assert values[("ssgrep_estimated_net_tokens", "conservative")] == -12000  # negative kept
    assert ("ssgrep_estimated_net_tokens", "optimistic") not in values  # unknown: omitted, not 0
    assert ("ssgrep_estimated_credit_tokens", "optimistic") not in values
    assert values[("ssgrep_known_credit_tokens", "optimistic")] == 20000
    assert values[("ssgrep_known_net_tokens", "optimistic")] == 4000
    # Fourteen slots: credit/net for three scenarios and a shared debit, for each
    # of tokens/work_seconds. Incomplete optimistic token credit AND net are absent.
    assert values[("ssgrep_coverage_records", "missing")] == 2
    assert values[("ssgrep_coverage_records", "available")] == 12
    resource = {a["key"] for a in document["resourceMetrics"][0]["resource"]["attributes"]}
    assert "ssgrep.counter.epoch" in resource


def test_log_examples_are_receiver_valid_revisioned_records_and_delete_the_revised_one(
    examples: dict[str, tuple[bytes, bytes]],
) -> None:
    state = ReceiverState()
    outcomes: list[Outcome] = []
    for name in (LOGS_FILE, TOMBSTONE_FILE):
        for record in _records(name, examples):
            body = json.loads(record["body"]["stringValue"])
            attributes = {a["key"]: next(iter(a["value"].values())) for a in record["attributes"]}
            assert attributes["ssgrep.record.digest"] == digest_of(body)
            assert attributes["ssgrep.record.id"] == body["record_id"]
            assert attributes["ssgrep.record.revision"] == str(body["revision"])
            assert attributes["ssgrep.record.type"] == body["record_type"]
            state, outcome = apply(state, Key(*RECEIVER_KEY, body["record_id"]), body)
            outcomes.append(outcome)
    assert outcomes == [Outcome.APPLIED_NEW, Outcome.APPLIED_NEW, Outcome.APPLIED_TOMBSTONE]
    logs = [json.loads(r["body"]["stringValue"]) for r in _records(LOGS_FILE, examples)]
    assert [(b["revision"], b["supersedes_revision"], b["record_type"]) for b in logs] == [
        (2, 1, "adjustment"),
        (1, None, "contribution"),
    ]
    (tomb,) = [json.loads(r["body"]["stringValue"]) for r in _records(TOMBSTONE_FILE, examples)]
    assert (tomb["record_id"], tomb["record_type"]) == (logs[0]["record_id"], "deletion")
    assert (tomb["revision"], tomb["supersedes_revision"]) == (3, 2)
    assert tomb["scenarios"] == tomb["known_costs"] == tomb["coverage"] == {}
    unit_key = ("context_tokens", "chars_div4_v1", None, None)
    totals = current_totals(state, methodology=METHOD.hash, unit_key=unit_key)
    assert totals["rows"] == 1  # the deleted record left the current totals


# ---- the document: structure and catalog -----------------------------------------------------

BLOCK = re.compile(r"```(\w+)(?: ([\w-]+))?\n(.*?)\n```", re.S)


def doc_text() -> str:
    return DOC.read_text()


def blocks(lang: str) -> dict[str, str]:
    return {name: body for kind, name, body in BLOCK.findall(doc_text()) if kind == lang and name}


# ---- worked example: apply + current_totals --------------------------------------------------

UNIT = {"unit": "tokens", "method": "estimate", "tokenizer": None, "currency": None}
UNIT_KEY = ("tokens", "estimate", None, None)
METHODOLOGY = "synthetic-methodology-1"
RECORDS = {"R1": "0" * 31 + "1", "R2": "0" * 31 + "2", "R3": "0" * 31 + "3"}
Nets = tuple[int | None, int | None, int | None]  # (conservative, central, optimistic) net
ARRIVALS: list[tuple[str, int, Nets | None]] = [  # None = a deletion
    ("R1", 1, (-50, 100, 200)),
    ("R1", 3, (-200, 300, 400)),
    ("R1", 3, (-200, 300, 400)),  # duplicate delivery of revision 3
    ("R1", 2, (-80, 150, 250)),  # reordered: an older revision arrives late
    ("R1", 3, (-200, 999, 400)),  # same revision, different digest: conflict
    ("R2", 1, (-10, -20, None)),  # the optimistic estimate is unknown
    ("R3", 1, (5, 10, 15)),
    ("R3", 2, None),  # deletion (tombstone)
    ("R3", 1, (5, 10, 15)),  # replay of the deleted contribution
]
CONFLICT = 4  # index of the same-revision/different-digest arrival


def _quantity(value: int | None) -> dict[str, Any]:
    if value is None:
        return {
            "value": None,
            "unit": "tokens",
            "basis": "unavailable",
            "method": "",
            "reason": "no_evidence",
        }
    return {"value": value, "unit": "tokens", "basis": "estimated", "method": "m", "reason": ""}


def worked_body(record: str, revision: int, nets: Nets | None) -> dict[str, Any]:
    scenarios: dict[str, Any] = {}
    for name, net in zip(SCENARIOS, nets, strict=True) if nets is not None else ():
        credit = None if net is None else max(net, 0)
        loss = None if net is None else max(-net, 0)
        scenarios[name] = {
            "credit": _quantity(credit),
            "debit": _quantity(loss),
            "net": _quantity(net),
        }
    kind = "deletion" if nets is None else ("contribution" if revision == 1 else "adjustment")
    return {
        "schema": 1,
        "record_id": RECORDS[record],
        "revision": revision,
        "supersedes_revision": None if revision == 1 else revision - 1,
        "record_type": kind,
        "task_id": f"synthetic-task-{record[1]}",
        "period": "2026-01-02",
        "unit": dict(UNIT),
        "scenarios": scenarios,
        "known_costs": {} if nets is None else {"output_tokens": 5},
        "coverage": {} if nets is None else {"complete": True},
        "methodology": METHODOLOGY,
    }


def replay(order: list[int]) -> tuple[ReceiverState, list[Outcome]]:
    state, outcomes = ReceiverState(), []
    for index in order:
        record, revision, nets = ARRIVALS[index]
        key = Key(*RECEIVER_KEY, RECORDS[record])
        state, outcome = apply(state, key, worked_body(record, revision, nets))
        outcomes.append(outcome)
    return state, outcomes


def arrival_rows() -> list[str]:
    _, outcomes = replay(list(range(len(ARRIVALS))))
    rows = []
    for number, ((record, revision, nets), outcome) in enumerate(
        zip(ARRIVALS, outcomes, strict=True), 1
    ):
        shown = (
            "-" if nets is None else " / ".join("unknown" if n is None else str(n) for n in nets)
        )
        kind = worked_body(record, revision, nets)["record_type"]
        rows.append(f"| {number} | {record} | {revision} | {kind} | {shown} | `{outcome.value}` |")
    return rows


def expected_result(state: ReceiverState) -> dict[str, Any]:
    totals = current_totals(state, methodology=METHODOLOGY, unit_key=UNIT_KEY)
    return {
        "current_rows": totals["rows"],
        "quarantined": len(state.quarantine),
        "tombstones": len(state.tombstones),
        "net": {
            name: {k: totals["scenarios"][name]["net"][k] for k in ("total", "known", "unknown")}
            for name in SCENARIOS
        },
    }


def test_worked_example_table_and_totals_equal_what_the_code_computes() -> None:
    state, outcomes = replay(list(range(len(ARRIVALS))))
    assert outcomes == [
        Outcome.APPLIED_NEW,
        Outcome.APPLIED_UPSERT,
        Outcome.DUPLICATE_NOOP,
        Outcome.REJECTED_OLD_REVISION,
        Outcome.QUARANTINED,
        Outcome.APPLIED_NEW,
        Outcome.APPLIED_NEW,
        Outcome.APPLIED_TOMBSTONE,
        Outcome.REJECTED_TOMBSTONED,
    ]
    text = doc_text()
    assert "\n".join(arrival_rows()) in text
    shown = json.loads(blocks("json")["worked-totals"])
    assert shown == expected_result(state)
    assert shown["net"]["central"] == {"total": 280, "known": 2, "unknown": 0}
    assert shown["net"]["conservative"] == {"total": -210, "known": 2, "unknown": 0}  # negative
    assert shown["net"]["optimistic"] == {"total": 400, "known": 1, "unknown": 1}  # unknown != 0
    naive = sum(n[1] for _, _, n in ARRIVALS if n is not None and n[1] is not None)
    assert naive == 1849 and f"**{naive}**" in text  # what summing every received row would show


def test_the_reference_receivers_final_state_does_not_depend_on_arrival_order() -> None:
    clean = [i for i in range(len(ARRIVALS)) if i != CONFLICT]  # a conflict keeps the first held
    reference = expected_result(replay(clean)[0])
    rng = random.Random(7)
    for _ in range(25):
        order = clean[:]
        rng.shuffle(order)
        assert expected_result(replay(order)[0]) == reference


def test_conformance_fixture_cases_named_in_the_document_still_hold() -> None:
    fixture = json.loads((EXAMPLES / "receiver-conformance.json").read_text())
    cases = {case["name"]: case for case in fixture["cases"]}
    assert cases["retry_and_reorder"]["expect_net_central"] == 300
    assert cases["conflicting_duplicate_quarantine"]["expect_quarantined"] == 1
    assert cases["negative_net_preserved"]["expect_net_central"] == -150
    assert cases["unknown_net_not_zero"]["expect_unknown_central"] == 1
    for name in ("retry_and_reorder", "tombstone_blocks_replay", "unknown_net_not_zero"):
        assert name in doc_text()


# ---- SQL form: executed with stdlib sqlite3 ---------------------------------------------------

INGESTED = 1_791_000_000  # the receiver's own clock for arrival 0
COLLECTED = 1_790_996_400  # an hour earlier: the log record's timeUnixNano
NOW = 1_791_000_100


def _columns(record: str, revision: int, nets: Nets | None, tick: int) -> dict[str, Any]:
    body = worked_body(record, revision, nets)
    net = {n: body["scenarios"].get(n, {}).get("net", {}).get("value") for n in SCENARIOS}
    return {
        "tenant": RECEIVER_KEY[0],
        "device": RECEIVER_KEY[1],
        "record_id": body["record_id"],
        "task_id": body["task_id"],
        "revision": revision,
        "record_type": body["record_type"],
        "digest": digest_of(body),
        "methodology": body["methodology"],
        "unit": body["unit"]["unit"],
        "unit_method": body["unit"]["method"],
        "unit_tokenizer": body["unit"]["tokenizer"] or "",
        "unit_currency": body["unit"]["currency"] or "",
        "period": body["period"],
        "net_conservative": net["conservative"],
        "net_central": net["central"],
        "net_optimistic": net["optimistic"],
        "collected_at": COLLECTED + tick,
        "ingested_at": INGESTED + tick,
    }


def fresh_database() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(blocks("sql")["schema"])
    return conn


def ingest(conn: sqlite3.Connection, order: list[int]) -> None:
    sql = blocks("sql")
    for tick, index in enumerate(order):
        row = _columns(*ARRIVALS[index], tick)
        for name in ("ingest-seen", "ingest-quarantine", "ingest-insert"):
            conn.execute(sql[name], row)
    conn.commit()


def sql_receiver(order: list[int]) -> sqlite3.Connection:
    conn = fresh_database()
    ingest(conn, order)
    return conn


UNIT_PARAMS = {
    "tenant": RECEIVER_KEY[0],
    "methodology": METHODOLOGY,
    "unit": "tokens",
    "unit_method": "estimate",
    "unit_tokenizer": "",
    "unit_currency": "",
}


def test_sql_totals_equal_the_reference_receiver_on_the_worked_example() -> None:
    state, _ = replay(list(range(len(ARRIVALS))))
    conn = sql_receiver(list(range(len(ARRIVALS))))
    row = dict(conn.execute(blocks("sql")["totals"], UNIT_PARAMS).fetchone())
    want = expected_result(state)
    assert row["records"] == want["current_rows"] == 2
    for name in SCENARIOS:
        assert row[f"net_{name}"] == want["net"][name]["total"]
        assert row[f"known_{name}"] == want["net"][name]["known"]
        assert row[f"unknown_{name}"] == want["net"][name]["unknown"]
    assert row["net_central"] == 280  # the conflicting 999 never replaced the held row
    quarantined = conn.execute("SELECT COUNT(*) FROM ledger_quarantine").fetchone()[0]
    assert quarantined == len(state.quarantine) == 1
    # 9 arrivals minus the revision-3 duplicate, the conflict, and the identical R3 replay
    assert conn.execute("SELECT COUNT(*) FROM ledger_log").fetchone()[0] == 6


def test_sql_current_revision_view_is_order_independent_and_excludes_tombstoned_records() -> None:
    clean = [i for i in range(len(ARRIVALS)) if i != CONFLICT]
    rng = random.Random(11)
    rows = []
    for _ in range(25):
        order = clean[:]
        rng.shuffle(order)
        conn = sql_receiver(order)
        rows.append(dict(conn.execute(blocks("sql")["totals"], UNIT_PARAMS).fetchone()))
        current = {
            r["record_id"]: r["revision"] for r in conn.execute("SELECT * FROM ledger_current")
        }
        assert current == {RECORDS["R1"]: 3, RECORDS["R2"]: 1}  # R3 is tombstoned
    assert all(row == rows[0] for row in rows)


def test_sql_unknown_stays_null_negative_stays_negative_and_no_rows_is_null_not_zero() -> None:
    sql = blocks("sql")
    conn = fresh_database()
    row = dict(conn.execute(sql["totals"], UNIT_PARAMS).fetchone())
    assert row["records"] == 0 and row["net_central"] is None  # no data is not zero
    conn.execute(sql["ingest-insert"], _columns("R2", 1, (-10, -20, None), 0))
    row = dict(conn.execute(sql["totals"], UNIT_PARAMS).fetchone())
    assert (row["net_conservative"], row["net_central"]) == (-10, -20)  # negatives preserved
    assert row["net_optimistic"] is None and row["unknown_optimistic"] == 1  # never 0


def test_sql_selection_never_mixes_methodologies_or_units() -> None:
    sql = blocks("sql")
    conn = sql_receiver([0, 1, 5])
    other = _columns("R2", 1, (1, 2, 3), 0) | {"record_id": RECORDS["R3"], "methodology": "m-2"}
    conn.execute(sql["ingest-insert"], other)
    listed = [dict(r) for r in conn.execute(sql["methodologies"], {"tenant": "tenant-a"})]
    assert sorted((r["methodology"], r["records"]) for r in listed) == [
        ("m-2", 1),
        (METHODOLOGY, 2),
    ]
    row = dict(conn.execute(sql["totals"], UNIT_PARAMS).fetchone())
    assert row["records"] == 2 and row["net_central"] == 280  # m-2 is excluded, never added


def test_sql_subject_and_freshness_blocks_run_and_do_not_rank() -> None:
    sql = blocks("sql")
    conn = sql_receiver(list(range(len(ARRIVALS))))
    subject = {**UNIT_PARAMS, "device": RECEIVER_KEY[1]}
    (period,) = [dict(r) for r in conn.execute(sql["subject"], subject)]
    assert (period["period"], period["records"], period["net_central"]) == ("2026-01-02", 2, 280)
    fresh = [dict(r) for r in conn.execute(sql["freshness"], {"now": NOW, "tenant": "tenant-a"})]
    heard = len(ARRIVALS) - 1  # the last arrival is a no-op replay, but the device was heard
    stored = 7  # the newest STORED record is the deletion (arrival 8 of the table)
    assert fresh == [
        {
            "tenant": "tenant-a",
            "device": "device-1",
            "silent_seconds": NOW - (INGESTED + heard),
            "data_age_seconds": NOW - (COLLECTED + stored),
        }
    ]
    assert fresh[0]["data_age_seconds"] > fresh[0]["silent_seconds"]  # old data, recent contact


if __name__ == "__main__":
    main()
