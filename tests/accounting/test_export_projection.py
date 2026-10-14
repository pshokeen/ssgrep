"""Positive-allowlist export projection (task 12.1): nothing planted can reach a payload.

Envelopes are hand-built with the real ``snapshots.build_envelope`` and the real counters; ledger
rows come from the real ``VersionedLedger`` reducers. A walker replaces every key and leaf of the
envelope, one at a time, with a canary and the serialized OTLP bytes are searched for it; the
ledger side plants canaries in every free-text position the builders accept. The positive side
pins the CLOSED sets of metric names, attribute names, attribute values and number encodings.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import Any

import pytest

from ssgrep.accounting import export, export_build, export_records, otlp
from ssgrep.accounting.contracts import (
    RUNTIMES,
    Attribution,
    Operation,
    Outcome,
    Reuse,
    Scenario,
    Transport as TransportEnum,
)
from ssgrep.accounting.export_model import SCOPE_NAME, SERVICE_NAME, Resource
from ssgrep.accounting.ledger_contracts import (
    WINDOWS,
    DestinationPolicy,
    ProfilePolicy,
    Signal,
    SnapshotEnvelope,
)
from ssgrep.accounting.metrics import FAMILIES, MetricsValidationError
from ssgrep.accounting.receiver_contract import Key, Outcome as Received, ReceiverState, apply
from tests.accounting.export_canary import (
    SHAPES,
    build_ledger,
    envelope_fields,
    envelope_variants,
    leaks,
    ledger_variants,
    number_paths,
    parse_strict,
    persist_ledger,
)
from tests.accounting.http_builders import make_snapshot, window_with_unknown_optimistic
from tests.accounting.valuation_builders import METHOD, candidate, debit

PROFILE = "personal-local"
DEST = "acme-org"
KEY = bytes(range(32))
METHODOLOGY = "19b54b416561f300e32b9e793a8ef5ce"
EPOCH_START = 1_790_812_800_000_000_000  # 2026-10-01T00:00:00Z (live_counters' persisted start)
RESOURCE = Resource("2.0.1", "a" * 32, 1, counter_epoch=1)
SHAPE_ITEMS = sorted(SHAPES.items())
STRUCTURED = {"windows", "coverage", "observations", "estimates"}


def policy() -> ProfilePolicy:
    dest = DestinationPolicy(
        DEST, "https://collector.example.test", (Signal.METRICS, Signal.LOGS), logs_permitted=True
    )
    return ProfilePolicy(destinations=(dest,), enabled_exports=((DEST, "metrics"), (DEST, "logs")))


def rich(**overrides: Any) -> SnapshotEnvelope:
    """Every projected section populated, plus every kind of free text a snapshot may carry."""
    base = make_snapshot(3, methodology_id=METHODOLOGY, window=window_with_unknown_optimistic())
    fields = envelope_fields(base)
    fields["warnings"] = ("collection_incomplete: stopped by budget", "second warning")
    fields["coverage"] = {
        **base.coverage,
        "sources": {**dict(base.coverage["sources"]), "lag_bytes": 12},  # ty: ignore[no-matching-overload]
        "note": "free text",
        "paths": ["/a/b", "/c/d"],
    }
    fields["observations"] = {
        **base.observations,
        "backfill": [
            {
                "kind": "backfill",
                "since": "2026-09-01T00:00:00Z",
                "until": "2026-09-30T00:00:00Z",
                "completed_at": "2026-10-01T00:00:00Z",
                "counts": {"search:path": 4, "show:path": 2},
                "unknown_cardinality": 1,
            }
        ],
        "epoch": {"counter_epoch": 1, "started_at": "2026-10-01T00:00:00Z"},
    }
    fields["windows"] = {
        "7d": fields["windows"]["7d"],
        "30d": {"label": "free", "estimates": {"context_tokens": {"note": "x"}}},
    }
    fields["estimates"] = {"representative_receipt": "rcpt-7f3a91", "method": "x", "reason": "y"}
    fields.update(overrides)
    return SnapshotEnvelope(**fields)


def metrics_bytes(points: Any, resource: Resource = RESOURCE) -> bytes:
    return b"".join(body for body, _ in otlp.split_metrics(resource, points))


def project_bytes(envelope: SnapshotEnvelope) -> tuple[export.Projection, bytes]:
    projection = export.project_metrics(envelope)
    return projection, metrics_bytes(projection.points)


def metric_docs(body: bytes) -> list[dict[str, Any]]:
    document = parse_strict(body)
    return [
        metric
        for resource in document["resourceMetrics"]
        for scope in resource["scopeMetrics"]
        for metric in scope["metrics"]
    ]


def data_points(metric: dict[str, Any]) -> list[dict[str, Any]]:
    kind = next(k for k in ("sum", "gauge", "histogram") if k in metric)
    return list(metric[kind]["dataPoints"])


# ---- closed catalog -------------------------------------------------------------------------

LABELS = {family.name: set(family.labels) for family in FAMILIES}
EXPORTED_FAMILIES = set(export._TABLE)
EXPECTED_NAMES = {entry[0] for entry in export._TABLE.values()}
ALLOWED_ATTRIBUTES = {
    export._TABLE[name][0]: LABELS[name] for name in export._TABLE if name in LABELS
}
COVERAGE_PAIRS = {
    ("lookups", "recorded"),
    ("lookups", "attributed"),
    ("lookups", "unattributed"),
    ("source_costs", "available"),
    ("source_costs", "unavailable"),
    ("candidates", "receipts"),
    ("candidates", "ambiguous"),
    ("sources", "total"),
    ("sources", "complete"),
    ("sources", "missing"),
    ("sources", "unreadable"),
    ("rendered_bytes", "unknown"),
    ("durations", "unknown"),
    ("estimates", "available"),
    ("estimates", "missing"),
}
ENUMS = {
    "operation": {o.value for o in Operation},
    "transport": {t.value for t in TransportEnum},
    "outcome": {o.value for o in Outcome},
    "runtime": {*RUNTIMES, "unknown"},
    "attribution": {a.value for a in Attribution},
    "evidence": {r.value for r in Reuse},
    "scenario": {s.value for s in Scenario},
    "window": set(WINDOWS),
}
METHOD_LABEL = re.compile(
    r"(estimate:)?(chars_div4_v1|capped_gap_v1|transcript_intervals_v1|monotonic_v1|"
    r"provider_usage_v1)|estimate:[0-9a-f]{16,64}|exact:[a-z0-9][a-z0-9_.-]{0,47}|unknown"
)
RESOURCE_KEYS = {
    "service.name",
    "service.version",
    "ssgrep.installation.id",
    "ssgrep.policy.version",
    "ssgrep.counter.epoch",
}


def attribute_ok(name: str, value: str, pair: tuple[str, str] | None) -> bool:
    if name in ENUMS:
        return value in ENUMS[name]
    if name in ("kind", "status"):
        return {k for pair in COVERAGE_PAIRS for k in pair[:1]} >= {value} or value in {
            pair[1] for pair in COVERAGE_PAIRS
        }
    if name == "method":
        return METHOD_LABEL.fullmatch(value) is not None
    if name == "schema_version":
        return value.isdigit() and len(value) <= 6
    if name == "methodology_id":
        return re.fullmatch(r"[0-9a-f]{16,64}", value) is not None
    return False


def assert_closed_payload(body: bytes) -> None:
    """Names, attribute names, attribute values, scope and resource come from closed sets."""
    if not body:
        return  # nothing projected: no request at all
    document = parse_strict(body)
    for resource in document["resourceMetrics"]:
        keys = {a["key"] for a in resource["resource"]["attributes"]}
        assert keys <= RESOURCE_KEYS and {"service.name", "ssgrep.installation.id"} <= keys
        for scope in resource["scopeMetrics"]:
            assert scope["scope"]["name"] == SCOPE_NAME
            for metric in scope["metrics"]:
                assert metric["name"] in EXPECTED_NAMES
                for point in data_points(metric):
                    assert {a["key"] for a in point["attributes"]} <= ALLOWED_ATTRIBUTES[
                        metric["name"]
                    ]
                    attrs = {a["key"]: a["value"] for a in point["attributes"]}
                    for name, value in attrs.items():
                        assert set(value) == {"stringValue"}
                        assert attribute_ok(name, value["stringValue"], None), (name, value)
                    if metric["name"] == "ssgrep_coverage_records":
                        pair = (attrs["kind"]["stringValue"], attrs["status"]["stringValue"])
                        assert pair in COVERAGE_PAIRS


def test_catalog_tables_agree_and_health_families_are_not_exported() -> None:
    assert EXPORTED_FAMILIES <= set(LABELS)
    assert not [f for f in EXPORTED_FAMILIES if f.startswith("ssgrep_export_")]
    for family in export._TABLE:
        assert LABELS[family] >= set()
    assert {n for n in EXPECTED_NAMES if n.endswith("_total")} == set()


# ---- positive shape -------------------------------------------------------------------------


def test_a_complete_snapshot_projects_exactly_the_catalog_names_with_closed_attributes() -> None:
    projection, body = project_bytes(rich())
    assert projection.omitted == 0 and projection.notes == ()
    names = {metric["name"] for metric in metric_docs(body)}
    assert names == EXPECTED_NAMES
    assert_closed_payload(body)


def test_integers_are_json_strings_and_no_float_literal_exists_outside_histograms() -> None:
    _, body = project_bytes(rich())
    allowed_ints = {"aggregationTemporality"}
    for path, kind in number_paths(parse_strict(body)):
        if kind == "int":
            assert path[-1] in allowed_ints, path
        else:
            assert path[-1] in {"sum", "explicitBounds", "[]"} and "histogram" in path, path
    # every decimal literal is the exact text derived from integer microseconds
    text = body.decode("ascii")
    assert '"sum":0.006,' in text and '"explicitBounds":[0.001,0.005,0.01,0.05,0.1,0.5,1,5,30]' in (
        text.replace("1.0", "1").replace("5.0", "5").replace("30.0", "30")
    )
    assert re.search(r'"asInt":"-?[0-9]+"', text)
    assert not re.search(r'"asInt":-?[0-9]', text)  # int64 are strings per the OTLP JSON mapping
    for field in ("startTimeUnixNano", "timeUnixNano", "count"):
        assert not re.search(rf'"{field}":[0-9]', text)


def test_counters_carry_the_persisted_start_and_gauges_carry_none() -> None:
    projection, body = project_bytes(rich())
    by_kind = {
        kind: [p for p in projection.points if p.kind == kind]
        for kind in ("sum", "gauge", "histogram")
    }
    assert by_kind["sum"] and by_kind["histogram"] and by_kind["gauge"]
    assert {p.start_unix_nano for p in by_kind["sum"] + by_kind["histogram"]} == {EPOCH_START}
    assert {p.start_unix_nano for p in by_kind["gauge"]} == {None}
    for metric in metric_docs(body):
        kind = next(k for k in ("sum", "gauge", "histogram") if k in metric)
        for point in data_points(metric):
            assert ("startTimeUnixNano" in point) == (kind != "gauge")
        if kind != "gauge":
            assert metric[kind]["aggregationTemporality"] == 2  # cumulative
    assert {p.time_unix_nano for p in projection.points} == {1_791_806_400_000_000_000}
    resource = export.metrics_resource(
        version="2.0.1", installation="b" * 32, policy_version=3, envelope=rich()
    )
    assert resource.counter_epoch == 1  # the persisted epoch travels as a resource attribute


@pytest.mark.parametrize("where", ["counters", "epoch"])
def test_the_epoch_start_is_read_from_either_persisted_location(where: str) -> None:
    observations = dict(rich().observations)
    counters = dict(observations["counters"])  # ty: ignore[no-matching-overload]
    if where == "epoch":
        counters.pop("epoch_started_at")
        observations["epoch"] = {"counter_epoch": 1, "started_at": "2026-10-01T00:00:00Z"}
    observations["counters"] = counters
    projection = export.project_metrics(replace(rich(), observations=observations))
    sums = [p for p in projection.points if p.kind == "sum"]
    assert sums and {p.start_unix_nano for p in sums} == {EPOCH_START}


def test_a_start_time_is_never_invented_counters_are_omitted_and_counted() -> None:
    observations = {k: v for k, v in rich().observations.items() if k != "epoch"}
    counters = {
        k: v
        for k, v in observations["counters"].items()  # ty: ignore[unresolved-attribute]
        if k != "epoch_started_at"
    }
    envelope = replace(rich(), observations={**observations, "counters": counters})
    projection, body = project_bytes(envelope)
    kinds = {p.kind for p in projection.points}
    assert kinds == {"gauge"}  # no sum, no histogram without a persisted start
    assert projection.omitted > 0
    assert projection.notes == ("counters_omitted_no_epoch_start",)
    assert export.describe_omitted(projection) == {
        "omitted_data_points": projection.omitted,
        "notes": ["counters_omitted_no_epoch_start"],
    }
    assert b"startTimeUnixNano" not in body


def test_missing_numbers_omit_the_sample_and_appear_as_a_coverage_count_never_as_zero() -> None:
    projection, body = project_bytes(rich())
    net = [dict(p.attributes) for p in projection.points if p.name == "ssgrep_estimated_net_tokens"]
    assert {a["scenario"] for a in net} == {"conservative", "central"}  # optimistic is unknown
    coverage = {
        dict(p.attributes)["status"]: p.value
        for p in projection.points
        if p.name == "ssgrep_coverage_records" and dict(p.attributes)["kind"] == "estimates"
    }
    # 7d: 9 of 10 slots known (optimistic net tokens unknown); the empty 30d window: all 10 unknown
    assert coverage == {"available": 9, "missing": 1 + 10}
    estimates = [p for p in projection.points if p.name.startswith("ssgrep_estimated")]
    assert len(estimates) == 3 + 2 + 3  # credit x3, net tokens x2, net work seconds x3
    assert 0 not in {p.value for p in estimates}  # no fabricated zero anywhere


@pytest.mark.parametrize("bad", [1.5, True, "12", None, 2**70])
def test_non_integer_numbers_are_dropped_not_coerced_or_rounded(bad: object) -> None:
    window = window_with_unknown_optimistic()
    window["estimates"]["work_seconds"]["central"]["net"]["value"] = bad
    envelope = replace(rich(), windows={"7d": window})
    try:
        projection, body = project_bytes(envelope)
    except ValueError:
        assert bad == 2**70  # an int64-overflowing integer is refused loudly by the serializer
        return
    work = [p for p in projection.points if p.name == "ssgrep_estimated_net_work_seconds"]
    assert {dict(p.attributes)["scenario"] for p in work} == {"conservative", "optimistic"}
    assert b"1.5" not in body and b'"asInt":"12"' not in body.replace(b'"asInt":"120"', b"")


def test_negative_estimates_and_unit_labels_are_preserved() -> None:
    projection, body = project_bytes(rich())
    by_name: dict[tuple[str, str], Any] = {
        (p.name, dict(p.attributes).get("scenario", "")): p for p in projection.points
    }
    assert by_name[("ssgrep_estimated_net_tokens", "conservative")].value == -12000
    assert by_name[("ssgrep_estimated_net_tokens", "central")].value == -7000
    assert by_name[("ssgrep_estimated_net_work_seconds", "conservative")].value == -60
    assert by_name[("ssgrep_estimated_net_tokens", "conservative")].unit == "{token}"
    assert by_name[("ssgrep_estimated_net_work_seconds", "central")].unit == "s"
    assert by_name[("ssgrep_rendered_bytes", "")].unit == "By"
    assert b'"asInt":"-12000"' in body and b'"asInt":"-60"' in body
    for metric in metric_docs(body):
        if metric["name"].startswith("ssgrep_estimated"):
            assert "gauge" in metric  # estimates are gauges, never counters


def test_the_projection_is_deterministic_bytes_for_the_same_snapshot() -> None:
    envelope = rich()
    assert project_bytes(envelope)[1] == project_bytes(envelope)[1]
    assert project_bytes(envelope)[1] != project_bytes(rich(collected_at="2026-10-12T12:00:01Z"))[1]


def test_a_malformed_collection_time_projects_nothing_and_names_no_value() -> None:
    projection = export.project_metrics(replace(rich(), collected_at=SHAPES["secret"]))
    assert projection.points == () and projection.omitted == 1
    assert projection.notes == ("invalid_collected_at",)
    assert not leaks(SHAPES["secret"], export.describe_omitted(projection), projection)


@pytest.mark.parametrize("subject", ["subject", "tenant", "device", "user", "host", "path"])
def test_resource_identity_is_pseudonymous_and_asserts_no_subject_or_tenant(subject: str) -> None:
    candidate_ = export_build.build_metrics(
        PROFILE, policy().destinations[0], policy(), rich(), KEY
    )
    document = json.loads(candidate_.batches[0].body)
    attributes = {a["key"] for a in document["resourceMetrics"][0]["resource"]["attributes"]}
    assert attributes <= RESOURCE_KEYS
    assert not [key for key in attributes if subject in key]
    name = next(
        a["value"]["stringValue"]
        for a in document["resourceMetrics"][0]["resource"]["attributes"]
        if a["key"] == "service.name"
    )
    assert name == SERVICE_NAME
    assert (
        candidate_.scope["fields"] == "metrics-v1" and candidate_.scope["logs_permitted"] is False
    )


# ---- envelope walker: metrics ---------------------------------------------------------------


def outcome_of(envelope: SnapshotEnvelope) -> tuple[export.Projection | None, bytes, str]:
    """(projection, serialized request, error text) — an error must be the safe validation type."""
    try:
        projection, body = project_bytes(envelope)
    except MetricsValidationError as error:
        return None, b"", f"{error!r} {error}"
    return projection, body, ""


def test_walker_covers_hundreds_of_positions_in_every_section() -> None:
    paths = [path for path, _ in envelope_variants(rich(), "x")]
    assert len(paths) > 450
    sections = {path[0] for path in paths}
    assert sections >= {
        "warnings",
        "coverage",
        "windows",
        "observations",
        "estimates",
        "methodology_id",
        "coverage_through",
        "source_manifest_digest",
        "collected_at",
        "snapshot_id",
    }
    assert any(path[-1] == "<key>" for path in paths)  # keys are planted too, not only values


@pytest.mark.parametrize(("shape", "canary"), SHAPE_ITEMS, ids=[s for s, _ in SHAPE_ITEMS])
def test_a_canary_in_any_envelope_key_or_leaf_never_reaches_metrics_bytes(
    shape: str, canary: str
) -> None:
    survived = refused = 0
    paths = list(envelope_variants(rich(), canary))
    for path, envelope in paths:
        projection, body, error = outcome_of(envelope)
        where = "/".join(path)
        if projection is None:
            refused += 1
            assert not leaks(canary, error), where  # a refusal never quotes the value
            continue
        survived += 1
        assert not leaks(canary, body), where
        assert not leaks(canary, projection.points), where
        assert not leaks(canary, export.describe_omitted(projection)), where
        assert_closed_payload(body)
    assert survived > 300 and survived + refused == len(paths)


def test_planted_canary_in_a_numeric_slot_is_omitted_and_counted_not_exported() -> None:
    canary = SHAPES["fact"]
    window = window_with_unknown_optimistic()
    window["estimates"]["context_tokens"]["central"]["credit"]["value"] = canary
    baseline = export.project_metrics(
        replace(rich(), windows={"7d": window_with_unknown_optimistic()})
    )
    projection, body = project_bytes(replace(rich(), windows={"7d": window}))
    assert not leaks(canary, body)
    assert len(projection.points) == len(baseline.points) - 1  # the sample vanished, nothing else


@pytest.mark.parametrize("walk", ["float", "bool", "huge"])
def test_every_structured_numeric_slot_replaced_by_a_non_integer_stays_integer_only(
    walk: str,
) -> None:
    replacement: Any = {"float": 1.5, "bool": True, "huge": 2**70}[walk]
    for path, envelope in envelope_variants(rich(), replacement):
        if path[0] not in STRUCTURED or path[-1] == "<key>":
            continue
        try:
            body = project_bytes(envelope)[1]
        except MetricsValidationError:
            continue  # a non-integer where a closed label belongs is refused by the catalog
        except ValueError as error:
            assert walk == "huge" and "out of range" in str(error)  # int64 overflow only
            continue
        assert not [
            p
            for p, kind in number_paths(parse_strict(body))
            if kind == "decimal" and not ({"sum", "explicitBounds"} & set(p))
        ], "/".join(path)
        assert b"1.5" not in body, "/".join(path)


# ---- ledger logs ----------------------------------------------------------------------------

LOG_KEYS = {
    "schema",
    "record_id",
    "revision",
    "supersedes_revision",
    "record_type",
    "task_id",
    "period",
    "unit",
    "scenarios",
    "known_costs",
    "coverage",
    "methodology",
}
LOG_ATTRIBUTES = {
    "ssgrep.record.id",
    "ssgrep.record.revision",
    "ssgrep.record.type",
    "ssgrep.record.digest",
}


def good_rows() -> Any:
    from tests.accounting.valuation_builders import TASK  # noqa: F401  (documents the shape)

    return build_ledger(
        __import__(
            "ssgrep.accounting.valuation_inputs", fromlist=["EvidenceBundle"]
        ).EvidenceBundle((candidate("a"),), (debit("x", 640),)),
        METHOD,
    )


def serialized_logs(rows: Any, destination: str = DEST) -> tuple[bytes, list[Any]]:
    candidates, _ = export_records.project_candidates(rows, key=KEY, destination=destination)
    planned = export_records.plan_revisions(export_records.Registry(), candidates)
    records = export_records.log_records(planned, time_unix_nano=1_791_806_400_000_000_000)
    return b"".join(body for body, _ in otlp.split_logs(RESOURCE, records)), planned


def test_log_bodies_attributes_and_resource_are_closed_and_receiver_valid() -> None:
    body, planned = serialized_logs(good_rows().rows())
    assert planned
    document = parse_strict(body)
    resource = document["resourceLogs"][0]
    assert {a["key"] for a in resource["resource"]["attributes"]} <= RESOURCE_KEYS
    assert resource["scopeLogs"][0]["scope"]["name"] == SCOPE_NAME
    state = ReceiverState()
    for record in resource["scopeLogs"][0]["logRecords"]:
        assert {a["key"] for a in record["attributes"]} == LOG_ATTRIBUTES
        inner = json.loads(record["body"]["stringValue"])
        assert set(inner) == LOG_KEYS
        assert "root_key" not in inner and "root" not in inner
        assert re.fullmatch(r"[0-9a-f]{32}", inner["record_id"])
        assert re.fullmatch(r"[0-9a-f]{32}", inner["task_id"])
        state, outcome = apply(state, Key("tenant", "device", inner["record_id"]), inner)
        assert outcome is Received.APPLIED_NEW
    assert [n for p, k in number_paths(document) for n in [p[-1]] if k == "int"] == [
        "severityNumber"
    ] * len(resource["scopeLogs"][0]["logRecords"])
    assert not [p for p, k in number_paths(document) if k == "decimal"]


def test_log_pseudonyms_do_not_depend_on_planted_text_and_never_embed_it() -> None:
    canary = SHAPES["path"]
    bundle, method = next((b, m) for label, b, m in ledger_variants(canary) if label == "task.root")
    body, planned = serialized_logs(build_ledger(bundle, method).rows())
    assert planned and not leaks(canary, body)
    assert all(canary not in str(p.body) for p in planned)
    # root_key is a one-way local handle; it is registry-only and never in a payload
    root_key = export_records.root_key_of(canary)
    assert planned[0].root_key == root_key and root_key.encode() not in body


def ledger_planting_outcomes(canary: str) -> dict[str, str]:
    """label -> exported | omitted | rejected for a canary planted at every ledger position."""
    results: dict[str, str] = {}
    for label, bundle, method in ledger_variants(canary):
        try:
            rows = build_ledger(bundle, method).rows()
        except ValueError as error:
            results[label] = "rejected"  # the reducer itself refuses an unsupported unit
            assert label in ("unit.unit", "unit.method")
            assert not leaks(canary, "closed")  # (the reducer's message is local, not exported)
            del error
            continue
        candidates, omitted = export_records.project_candidates(rows, key=KEY, destination=DEST)
        body, planned = serialized_logs(rows)
        assert not leaks(canary, body), label
        assert not leaks(canary, [p.body for p in planned], [p.body_digest for p in planned]), label
        assert not leaks(canary, [(c.record_id, c.task_id, c.core) for c in candidates]), label
        results[label] = "exported" if candidates else "omitted"
        assert omitted + len(candidates) >= 1
    return results


@pytest.mark.parametrize(("shape", "canary"), SHAPE_ITEMS, ids=[s for s, _ in SHAPE_ITEMS])
def test_a_canary_in_any_ledger_free_text_position_never_reaches_log_bytes(
    shape: str, canary: str
) -> None:
    results = ledger_planting_outcomes(canary)
    assert len(results) >= 27
    assert list(results.values()).count("exported") >= 18  # positions that do not block the leaf
    for label in ("unit.tokenizer", "unit.currency"):
        assert results[label] in ("omitted", "rejected"), label  # unit labels are closed sets


@pytest.mark.parametrize("shape", ["secret", "path", "query", "token_hex"])
def test_build_logs_over_a_persisted_ledger_never_carries_the_canary(shape: str) -> None:
    canary = SHAPES[shape]
    dest = policy().destinations[0]
    exported = 0
    for label, bundle, method in ledger_variants(canary):
        persist_ledger(PROFILE, build_ledger(bundle, method))
        try:
            candidate_ = export_build.build_logs(PROFILE, dest, policy(), rich(), KEY)
        except export_records.RecordsUnavailable as error:
            assert label in ("unit.unit", "unit.method")
            assert not leaks(canary, str(error), repr(error), error.condition)
            continue
        exported += bool(candidate_.batches)
        evidence = [
            [(b.batch_id, b.body, b.record_ids) for b in candidate_.batches],
            candidate_.scope,
            candidate_.notes,
            [[(p.record_id, p.body, p.task) for p in group] for group in candidate_.planned],
            candidate_.omitted,
        ]
        assert not leaks(canary, *[str(e) for e in evidence]), label
        for batch in candidate_.batches:
            assert re.fullmatch(r"g[0-9]{10}-[0-9a-f]{16}-[0-9]{4}", batch.batch_id)
            assert all(re.fullmatch(r"[0-9a-f]{32}", rid) for rid in batch.record_ids)
    assert exported >= 18


def test_log_scope_and_notes_are_closed_documents() -> None:
    dest = policy().destinations[0]
    persist_ledger(PROFILE, good_rows())
    built = export_build.build_logs(PROFILE, dest, policy(), rich(), KEY)
    assert built.batches and built.omitted == 0
    assert set(built.scope) == {
        "roots",
        "collection_scope",
        "endpoint",
        "proxy",
        "fields",
        "policy_version",
        "logs_permitted",
        "identity",
    }
    assert built.scope["fields"] == "logs-v1" and built.scope["logs_permitted"] is True
    assert re.fullmatch(r"[0-9a-f]{32}", str(built.scope["identity"]))


def test_log_projection_refuses_envelope_planting_by_ignoring_the_envelope_free_text() -> None:
    dest = policy().destinations[0]
    persist_ledger(PROFILE, good_rows())
    canary = SHAPES["fact"]
    clean = export_build.build_logs(PROFILE, dest, policy(), rich(), KEY)
    for path, envelope in envelope_variants(rich(), canary):
        if path[:2] == ("coverage", "collection_scope"):
            with pytest.raises(ValueError) as raised:
                export_build.build_logs(PROFILE, dest, policy(), envelope, KEY)
            assert not leaks(canary, str(raised.value))
            continue
        if path[0] in {"warnings", "coverage", "windows", "estimates", "observations"}:
            built = export_build.build_logs(PROFILE, dest, policy(), envelope, KEY)
            assert [b.body for b in built.batches] == [b.body for b in clean.batches], path
    assert not leaks(canary, [b.body for b in clean.batches])
