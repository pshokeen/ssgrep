"""Positive per-signal export projection: snapshot aggregates -> sanitized OTLP data points.

Nothing is "stripped from" the snapshot. Each exported datum is *constructed* from an
explicit family table, and every attribute value must pass a closed validator; whatever does
not match is counted (never copied) in ``Projection.omitted``. Consequently a query, path,
fact, title, ref, transcript text, receipt content, raw error body, token, or arbitrary
``warnings``/``coverage``/``windows`` string planted anywhere in the snapshot cannot reach a
payload: the only strings that can appear are the catalog's fixed family names, the bounded
enumerations of ``metrics``/``contracts``, a method label that matches the method allowlist,
and a lowercase-hex methodology digest. Numbers are integers (histogram seconds are exact
decimal literals derived from integer microseconds).

Metrics identity: resource attributes are pseudonymous service/version/installation/policy
(+ the persisted counter epoch). No subject, tenant, device name, host, or path is ever
asserted; the collector derives authoritative identity from the enrolled credential.
Timestamps are the snapshot's own ``collected_at`` and the persisted counter-epoch start, so
re-projecting the same snapshot yields the same bytes.
"""

import hashlib
import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ssgrep.accounting.export import (
    consent as export_policy,
    ids as export_ids,
    otlp,
    records as export_records,
    registry as export_registry,
)
from ssgrep.accounting.export.consent import (
    Gate,
    ScopeState,
    migration_view,
    scope_document,
    snapshot_eligible,
)
from ssgrep.accounting.export.otlp import Batch, Hist, Point, Resource
from ssgrep.accounting.export.registry import Planned
from ssgrep.accounting.ledger.contracts import Quantity, canonical_bytes
from ssgrep.accounting.ledger.ledger_contracts import (
    WINDOWS,
    DestinationPolicy,
    ProfilePolicy,
    Signal,
    SnapshotEnvelope,
)
from ssgrep.accounting.ledger.valuation import BOUND_ASSUMPTIONS
from ssgrep.accounting.serve.metric_estimates import ADDITIONAL_FAMILIES
from ssgrep.accounting.serve.metrics import Sample, snapshot_samples
from ssgrep.accounting.views import snapshots
from ssgrep.accounting.views.lookups import LOOKUP_WORK_ASSUMPTIONS, LOOKUP_WORK_MAPPING

METRICS_FIELDS = otlp.FIELDS_VERSION[Signal.METRICS.value]
_METHOD = re.compile(
    r"(estimate:)?(chars_div4_v1|capped_gap_v1|transcript_intervals_v1|monotonic_v1|"
    r"provider_usage_v1)|estimate:[0-9a-f]{16,64}|exact:[a-z0-9][a-z0-9_.-]{0,47}|unknown"
)
_HEX = re.compile(r"[0-9a-f]{16,64}")
OTHER = "other"
_UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def safe_method(value: object) -> str:
    """A measurement-method label from the allowlist; anything else is the constant ``other``."""
    return value if isinstance(value, str) and _METHOD.fullmatch(value) else OTHER


# Closed vocabulary of the reasons the reducers/valuation can attach to an unavailable quantity.
# Shape alone is never enough (a token-like string would pass any lexical test), so a reason is
# exported only when it IS one of these codes. Codes that are built with a free value suffix
# (``no_applicable_tariff:<category>``, ``tokenizer_*:<library/version/error>``) lose the suffix.
UNSPECIFIED = "unspecified"
REASON_CODES = frozenset(
    {
        UNSPECIFIED,
        "debit_chars_unavailable",
        "conflicting_duplicate_debit",
        "downstream_exposure_unobserved",
        "evidence_expired",
        "exact_tokens_unavailable",
        "history_unreadable",
        "invalid_timing_in_gaps",
        "lookup_work_seconds_not_modeled",
        "malformed_quantity",
        "mixed_shell_output_unattributable",
        "monotonic_clock_reversed",
        "no_closed_periods",
        "no_dependency_evidence",
        "no_journal_event_in_capture_span",
        "no_render_event",
        "no_runtime_exposure_joined",
        "no_tariff",
        "not_estimated",
        "output_shared_by_several_executions",
        "provider_cost_missing",
        "rendered_bytes_missing_for_some_executions",
        "scenario_missing",
        "source_costs_missing",
        "source_seconds_unavailable",
        "tokenizer_file_digest_mismatch",
        "tokenizer_file_missing",
        "unit_not_in_period",
        "visible_bytes_missing_for_some_executions",
        "journal_monotonic_duration_missing",
        "conflicting_journal_timing",
        "ambiguous_journal_timing",
        "journal_cardinality_unbounded",
        "no_linked_journal_timing",
        "source_missing_clock",
        "source_clock_reversed",
        "source_terminal_gap",
        "source_task_boundary_gap",
        "source_work_method_incompatible",
        "source_ownership_unknown",
        "rarity_unknown",
        "prior_context_unavailable",
        "legacy_valuation_uncertainty_unavailable",
        "credit_upper_unbounded",
        "debit_upper_unbounded",
        "net_lower_unbounded_debit",
        "net_upper_unbounded_credit",
        "measured_service_ms_missing_for_some_executions",
        "lookup_work_nanoseconds_unavailable",
        "journal_nanosecond_precision_unavailable",
        "nanosecond_precision_unavailable",
        "time_precision_namespace_migration_required",
    }
)
_SUFFIXED_CODES = frozenset(
    {
        "no_applicable_tariff",
        "tokenizer_library_not_installed",
        "tokenizer_load_failed",
        "tokenizer_version_mismatch",
        "unsupported_tokenizer_library",
    }
)
_INCOMPLETE = "incomplete:"


def _reason_code(member: str) -> str | None:
    head, _, _ = member.partition(":")
    if head in _SUFFIXED_CODES:
        return head
    return member if member in REASON_CODES else None


def safe_reason(value: object) -> str:
    """A reason from the closed vocabulary (or ``incomplete:`` plus members of it).

    Anything else, including a single unknown member inside ``incomplete:``, collapses the whole
    reason to the constant ``unspecified``: free text never reaches a payload.
    """
    if not isinstance(value, str):
        return UNSPECIFIED
    if value.startswith(_INCOMPLETE):
        members = [_reason_code(m) for m in value[len(_INCOMPLETE) :].split(",")]
        if not members or any(m is None for m in members):
            return UNSPECIFIED
        return _INCOMPLETE + ",".join(sorted({m for m in members if m is not None}))
    return _reason_code(value) or UNSPECIFIED


def unix_nano(stamp: str) -> int:
    """UTC ``YYYY-MM-DDTHH:MM:SSZ`` -> nanoseconds since the epoch (ValueError otherwise)."""
    try:
        moment = datetime.strptime(stamp, _UTC_FORMAT).replace(tzinfo=UTC)
    except (TypeError, ValueError):
        raise ValueError("invalid_timestamp") from None  # never quote the offending value
    return int(moment.timestamp()) * 1_000_000_000


# family -> (OTLP name, kind, unit, description). Sums drop the Prometheus ``_total`` suffix:
# the collector's Prometheus exporter re-adds it, and OTLP names stay idiomatic.
_TABLE: dict[str, tuple[str, str, str, str]] = {
    "ssgrep_executions_total": (
        "ssgrep_executions",
        "sum",
        "{execution}",
        "Captured search/show/note executions",
    ),
    "ssgrep_rendered_bytes_total": (
        "ssgrep_rendered_bytes",
        "sum",
        "By",
        "Rendered response bytes where observable",
    ),
    "ssgrep_operation_duration_seconds": (
        "ssgrep_operation_duration_seconds",
        "histogram",
        "s",
        "Live captured operation duration",
    ),
    "ssgrep_reconciliation_records": (
        "ssgrep_reconciliation_records",
        "gauge",
        "{record}",
        "Lookups by attribution state",
    ),
    "ssgrep_reuse_evidence_records": (
        "ssgrep_reuse_evidence_records",
        "gauge",
        "{record}",
        "Candidates by reuse evidence",
    ),
    "ssgrep_estimated_credit_tokens": (
        "ssgrep_estimated_credit_tokens",
        "gauge",
        "{token}",
        "Modeled credit (not causal proof)",
    ),
    "ssgrep_debit_tokens": ("ssgrep_debit_tokens", "gauge", "{token}", "Known lookup debit"),
    "ssgrep_estimated_net_tokens": (
        "ssgrep_estimated_net_tokens",
        "gauge",
        "{token}",
        "Modeled net context tokens; may be < 0",
    ),
    "ssgrep_estimated_net_work_seconds": (
        "ssgrep_estimated_net_work_seconds",
        "gauge",
        "s",
        "Modeled net agent work; may be < 0",
    ),
    "ssgrep_coverage_records": (
        "ssgrep_coverage_records",
        "gauge",
        "{record}",
        "Evidence coverage counts",
    ),
    "ssgrep_snapshot_collected_timestamp_seconds": (
        "ssgrep_snapshot_collected_timestamp_seconds",
        "gauge",
        "s",
        "Snapshot collection time",
    ),
    "ssgrep_snapshot_available": (
        "ssgrep_snapshot_available",
        "gauge",
        "{state}",
        "Snapshot exists",
    ),
    "ssgrep_accounting_info": (
        "ssgrep_accounting_info",
        "gauge",
        "{info}",
        "Schema and methodology identifiers (value 1)",
    ),
}
for _name, _labels in ADDITIONAL_FAMILIES:
    _unit = (
        "ms"
        if _name.endswith("_milliseconds")
        else "s"
        if _name.endswith("_work_seconds")
        else "{record}"
        if _name.endswith("_records")
        else "{token}"
    )
    _TABLE[_name] = (
        _name,
        "gauge",
        _unit,
        "Known subtotal, interval bound or evidence coverage; not causal or human time savings",
    )
_ENUM_LABELS = frozenset(
    {"operation", "transport", "outcome", "runtime", "attribution", "evidence", "scenario",
     "window", "kind", "status", "quantity", "bound"}
)  # fmt: skip
_BUCKET = re.compile(r"ssgrep_operation_duration_seconds_(bucket|count|sum)")


@dataclass(frozen=True)
class Projection:
    """The sanitized candidate for one snapshot/signal; ``omitted`` counts, never quotes."""

    points: tuple[Point, ...]
    omitted: int
    notes: tuple[str, ...] = ()


def _label_ok(name: str, value: str) -> bool:
    if name == "window":
        return value in WINDOWS
    if name in _ENUM_LABELS:
        return bool(re.fullmatch(r"[a-z][a-z0-9_]{0,31}", value))
    if name == "method":
        return value == safe_method(value)
    if name == "schema_version":
        return value.isdigit() and len(value) <= 6
    assert name == "methodology_id"  # snapshot_samples enforces the fixed label catalog
    return bool(_HEX.fullmatch(value))


def _attributes(sample: Sample) -> tuple[tuple[str, str], ...] | None:
    out = []
    for name, value in sample.labels:
        if name == "le":
            continue
        if not _label_ok(name, value):
            return None
        out.append((name, value))
    return tuple(sorted(out))


def _decimal_seconds(micros: int) -> str:
    seconds, rest = divmod(micros, 1_000_000)
    return str(seconds) if not rest else f"{seconds}.{rest:06d}".rstrip("0")


def _epoch_start(envelope: SnapshotEnvelope) -> int | None:
    counters = envelope.observations.get("counters")
    stamp = counters.get("epoch_started_at") if isinstance(counters, dict) else None
    if not isinstance(stamp, str):
        epoch = envelope.observations.get("epoch")
        stamp = epoch.get("started_at") if isinstance(epoch, dict) else None
    try:
        return unix_nano(stamp) if isinstance(stamp, str) else None
    except ValueError:
        return None


def _histograms(
    groups: dict[tuple[tuple[str, str], ...], list[Sample]], now: int, start: int | None
) -> tuple[list[Point], int]:
    points: list[Point] = []
    omitted = 0
    for attrs, samples in sorted(groups.items()):
        buckets = [s for s in samples if s.family.endswith("_bucket")]
        count = next((s.value for s in samples if s.family.endswith("_count")), None)
        total = next((s.value for s in samples if s.family.endswith("_sum")), None)
        if start is None or count is None or total is None or not buckets:
            omitted += 1
            continue
        cumulative = [b.value for b in buckets]
        bounds = [dict(b.labels)["le"] for b in buckets][:-1]
        counts = [c - p for c, p in zip(cumulative, [0, *cumulative[:-1]], strict=True)]
        hist = Hist(tuple(bounds), tuple(counts), count, _decimal_seconds(total))
        name, _, unit, description = _TABLE["ssgrep_operation_duration_seconds"]
        points.append(Point(name, "histogram", unit, description, attrs, now, None, hist, start))
    return points, omitted


def project_metrics(envelope: SnapshotEnvelope) -> Projection:
    """Every datum the snapshot supports, built only from the family table above."""
    try:
        now = unix_nano(envelope.collected_at)
    except ValueError:
        return Projection((), 1, ("invalid_collected_at",))
    start = _epoch_start(envelope)
    points: list[Point] = []
    groups: dict[tuple[tuple[str, str], ...], list[Sample]] = {}
    omitted = 0
    for sample in snapshot_samples(envelope):
        entry = _TABLE.get(sample.family) or _TABLE.get(
            _BUCKET.sub("ssgrep_operation_duration_seconds", sample.family)
        )
        attrs = _attributes(sample)
        if entry is None or attrs is None or type(sample.value) is not int:
            omitted += 1
            continue
        if entry[1] == "histogram":
            groups.setdefault(attrs, []).append(sample)
            continue
        name, kind, unit, description = entry
        if kind == "sum" and start is None:
            omitted += 1  # a cumulative point without a persisted start time is never invented
            continue
        points.append(
            Point(name, kind, unit, description, attrs, now, sample.value, None,
                  start if kind == "sum" else None)
        )  # fmt: skip
    hist_points, hist_omitted = _histograms(groups, now, start)
    ordered = sorted(
        [*points, *hist_points], key=lambda p: (p.name, p.attributes, p.time_unix_nano)
    )
    notes = ("counters_omitted_no_epoch_start",) if start is None and omitted else ()
    return Projection(tuple(ordered), omitted + hist_omitted, notes)


def metrics_resource(
    *, version: str, installation: str, policy_version: int, envelope: SnapshotEnvelope
) -> Resource:
    """Metrics resource: pseudonymous identity plus the persisted counter epoch, nothing else."""
    return Resource(version, installation, policy_version, counter_epoch=envelope.counter_epoch)


def installed_version() -> str:
    """The ssgrep package version as ``service.version`` (the string ``unknown`` if absent)."""
    try:
        from importlib.metadata import version

        return version("ssgrep")
    except Exception:  # noqa: BLE001 - metadata lookup must never block an export preview
        return "unknown"


def describe_omitted(projection: Projection) -> dict[str, Any]:
    return {"omitted_data_points": projection.omitted, "notes": list(projection.notes)}


"""Build the exact sanitized candidate batches for one destination/signal (no I/O except reads).

Preview, the delivery cycle and the tombstone writer all go through here, so what ``--preview``
prints is byte-for-byte what a send would spool (for the same snapshot, policy and key):

* metrics: ``export.project_metrics`` (positive allowlist) -> ``otlp.split_metrics``;
* logs: ledger rows -> ``export_records.project_candidates`` -> ``plan_revisions`` against the
  registry (only NEW or CHANGED records, as higher revisions) -> ``otlp.split_logs``;
* tombstones: registry-derived deletion bodies -> ``otlp.split_logs`` with ``t...`` batch ids.

Nothing here creates state: the pseudonym key is only read (``ephemeral_key`` stands in when it
does not exist yet, and the candidate says so), the registry is only loaded, and ledger rows are
only read.
"""


TOMBSTONE_GENERATION = 0


@dataclass(frozen=True)
class Candidate:
    """The sanitized candidate for one destination/signal and what it was derived from."""

    signal: Signal
    destination: str
    generation: int
    snapshot_id: str
    batches: tuple[Batch, ...]
    planned: tuple[tuple[Planned, ...], ...]  # per batch (logs only)
    scope: dict[str, object]
    omitted: int = 0
    notes: tuple[str, ...] = ()
    ephemeral_key: bool = False
    policy_version: int = 1


def key_for(profile: str, *, create: bool) -> tuple[bytes, bool]:
    """(pseudonym key, ephemeral?) — read-only unless ``create`` (only the send path creates)."""
    key = export_ids.load_key(profile, create=create)
    if key is None:
        return export_ids.ephemeral_key(), True
    return key, False


def _batch_id(generation: int, snapshot_id: str, index: int) -> str:
    return f"g{generation:010d}-{snapshot_id[:16]}-{index:04d}"


def _resource(
    profile: str,
    destination: str,
    policy: ProfilePolicy,
    key: bytes,
    envelope: SnapshotEnvelope | None,
) -> Resource:
    return Resource(
        installed_version(),
        export_ids.installation_id(key, destination, profile),
        policy.policy_version,
        counter_epoch=envelope.counter_epoch if envelope is not None else None,
    )


def _require_collection_scope(
    profile: str, envelope: SnapshotEnvelope, scope: dict, version: int
) -> None:
    state = ScopeState("", None, scope.get("collection_scope"))
    migration = migration_view(profile)
    if migration.reason:
        raise ValueError(migration.reason)
    if not snapshot_eligible(envelope, state, version):
        raise ValueError("ineligible_scope: recollect under current workspace ownership")


def build_metrics(
    profile: str,
    dest: DestinationPolicy,
    policy: ProfilePolicy,
    envelope: SnapshotEnvelope,
    key: bytes,
    *,
    ephemeral: bool = False,
) -> Candidate:
    resource = _resource(profile, dest.destination_id, policy, key, envelope)
    scope = scope_document(policy, dest, Signal.METRICS, resource.installation_id)
    _require_collection_scope(profile, envelope, scope, policy.policy_version)
    projection: Projection = project_metrics(envelope)
    chunks = otlp.split_metrics(resource, projection.points)
    batches = tuple(
        Batch(
            _batch_id(envelope.generation, envelope.snapshot_id, index),
            Signal.METRICS,
            envelope.generation,
            envelope.snapshot_id,
            index,
            len(chunks),
            "data",
            body,
            count,
        )
        for index, (body, count) in enumerate(chunks)
    )
    return Candidate(
        Signal.METRICS,
        dest.destination_id,
        envelope.generation,
        envelope.snapshot_id,
        batches,
        (),
        scope,
        projection.omitted,
        projection.notes,
        ephemeral,
        policy.policy_version,
    )


def _record_ids(body: bytes) -> list[tuple[str, int]]:
    """(record_id, revision) of every log record in a serialized logs request."""
    document = json.loads(body)
    found: list[tuple[str, int]] = []
    for resource in document.get("resourceLogs", []):
        for scope in resource.get("scopeLogs", []):
            for record in scope.get("logRecords", []):
                inner = json.loads(record["body"]["stringValue"])
                found.append((inner["record_id"], inner["revision"]))
    return found


def _logs_batches(
    resource: Resource,
    planned: Sequence[Planned],
    time_nano: int,
    ids: Sequence[str],
    *,
    generation: int,
    snapshot_id: str,
    kind: str,
) -> tuple[tuple[Batch, ...], tuple[tuple[Planned, ...], ...]]:
    records = export_records.log_records(planned, time_unix_nano=time_nano)
    chunks = otlp.split_logs(resource, records)
    by_key = {(p.record_id, p.revision): p for p in planned}
    batches: list[Batch] = []
    groups: list[tuple[Planned, ...]] = []
    for index, (body, count) in enumerate(chunks):
        members = tuple(by_key[key] for key in _record_ids(body))
        batches.append(
            Batch(
                ids[index],
                Signal.LOGS,
                generation,
                snapshot_id,
                index,
                len(chunks),
                kind,
                body,
                count,
                tuple(sorted({m.record_id for m in members})),
            )
        )
        groups.append(members)
    return tuple(batches), tuple(groups)


def _chunk_count(resource: Resource, planned: Sequence[Planned], time_nano: int) -> int:
    records = export_records.log_records(planned, time_unix_nano=time_nano)
    return len(otlp.split_logs(resource, records))


def build_logs(
    profile: str,
    dest: DestinationPolicy,
    policy: ProfilePolicy,
    envelope: SnapshotEnvelope,
    key: bytes,
    *,
    ephemeral: bool = False,
) -> Candidate:
    """Ledger-record revisions the registry does not know yet (logs; consent required by caller)."""
    resource = _resource(profile, dest.destination_id, policy, key, None)
    scope = scope_document(policy, dest, Signal.LOGS, resource.installation_id)
    _require_collection_scope(profile, envelope, scope, policy.policy_version)
    rows = export_records.ledger_rows(profile, collection_scope=scope["collection_scope"])
    planned: list[Planned] = []
    omitted = 0
    if rows is not None:
        candidates, omitted = export_records.project_candidates(
            rows, key=key, destination=dest.destination_id
        )
        registry = export_registry.load_registry(profile, dest.destination_id)
        planned = export_registry.plan_revisions(
            registry, candidates, current_namespace=rows.namespace
        )
    time_nano = unix_nano(envelope.collected_at)
    count = _chunk_count(resource, planned, time_nano) if planned else 0
    ids = [_batch_id(envelope.generation, envelope.snapshot_id, i) for i in range(count)]
    batches, groups = _logs_batches(
        resource,
        planned,
        time_nano,
        ids,
        generation=envelope.generation,
        snapshot_id=envelope.snapshot_id,
        kind="data",
    )
    return Candidate(
        Signal.LOGS,
        dest.destination_id,
        envelope.generation,
        envelope.snapshot_id,
        batches,
        groups,
        scope,
        omitted,
        (),
        ephemeral,
        policy.policy_version,
    )


def tombstone_ids(planned: Sequence[Planned], stamp: int, count: int) -> list[str]:
    """Deterministic ``t<unix>-<hash8>-<index>`` ids for one tombstone set."""
    digest = hashlib.sha256(
        canonical_bytes(sorted((p.record_id, p.revision) for p in planned))
    ).hexdigest()[:8]
    return [f"t{stamp:010d}-{digest}-{index:04d}" for index in range(count)]


def build_tombstones(
    profile: str,
    dest: DestinationPolicy,
    policy: ProfilePolicy,
    deletions: Sequence[Planned],
    key: bytes,
    *,
    stamp: int,
) -> Candidate:
    """Deletion-revision log batches (``kind='tombstone'``); empty input yields no batches."""
    resource = _resource(profile, dest.destination_id, policy, key, None)
    scope = scope_document(policy, dest, Signal.LOGS, resource.installation_id)
    time_nano = stamp * 1_000_000_000
    count = _chunk_count(resource, deletions, time_nano) if deletions else 0
    batches, groups = _logs_batches(
        resource,
        deletions,
        time_nano,
        tombstone_ids(deletions, stamp, count),
        generation=TOMBSTONE_GENERATION,
        snapshot_id="",
        kind="tombstone",
    )
    return Candidate(
        Signal.LOGS,
        dest.destination_id,
        TOMBSTONE_GENERATION,
        "",
        batches,
        groups,
        scope,
        0,
        (),
        False,
        policy.policy_version,
    )


BOUND_FIELDS = (
    "credit_lower",
    "credit_upper",
    "debit_lower",
    "debit_upper",
    "net_lower",
    "net_upper",
)
_ALLOWED_ASSUMPTIONS = frozenset(BOUND_ASSUMPTIONS)
_LIMIT = 2**63 - 1


def quantity(raw: Quantity, unit: str) -> dict:
    if raw.unit != unit:
        raise ValueError("incompatible_quantity_unit")
    if raw.value is not None and (
        type(raw.value) is not int or not -_LIMIT - 1 <= raw.value <= _LIMIT
    ):
        raise ValueError("quantity_out_of_range")
    return {
        "value": raw.value,
        "unit": unit,
        "basis": raw.basis.value,
        "method": safe_method(raw.method) if raw.method else "",
        "reason": safe_reason(raw.reason) if raw.reason else "",
    }


def missing_counts(missing) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for _, reason in missing:
        code = safe_reason(reason)
        counts[code if len(code) <= 64 else "unspecified"] += 1
    ordered = sorted(counts)
    out = {code: counts[code] for code in ordered[:63]}
    if len(ordered) > 63:
        out["other_missing_reasons"] = sum(counts[code] for code in ordered[63:])
    return out


def _assumptions(values) -> list[str]:
    out = set()
    for value in values:
        if value in _ALLOWED_ASSUMPTIONS:
            out.add(value)
        elif value in LOOKUP_WORK_ASSUMPTIONS:
            out.add(LOOKUP_WORK_MAPPING)
        else:
            out.add("additional_assumptions_not_exported")
    return sorted(out)


def project_scenario(estimate) -> dict:
    unit = estimate.unit.unit
    known = estimate.known_net
    if type(known) is not int or not -_LIMIT - 1 <= known <= _LIMIT:
        raise ValueError("known_subtotal_out_of_range")
    bounds = estimate.bounds
    return {
        "credit": quantity(estimate.credit, unit),
        "debit": quantity(estimate.debit, unit),
        "net": quantity(estimate.net, unit),
        "known_net": known,
        "complete": estimate.complete,
        "credit_missing": missing_counts(estimate.credit_missing),
        "debit_missing": missing_counts(estimate.debit_missing),
        "known_atoms": estimate.known_atoms,
        "known_debits": estimate.known_debits,
        "bounds": (
            {
                **{field: quantity(getattr(bounds, field), unit) for field in BOUND_FIELDS},
                "assumptions": _assumptions(bounds.assumptions),
            }
            if bounds is not None
            else None
        ),
    }


"""Candidate selection shared by ``--preview`` and the enqueue step of a delivery cycle.

``preview_candidate`` is read-only: it observes no scope, mints no pseudonym key, and writes no
registry/spool/delivery state. It goes through the same ``export_build.build_*`` functions as the
enqueue step, so the bytes a preview shows equal the bytes a cycle would spool for the same
snapshot, policy and key (``ephemeral_key`` says when no key exists yet).

States of :class:`PreviewCandidate` (``candidate`` is present for ``ok``, ``no_changes`` and
``no_records``; absent otherwise):

* ``no_snapshot``     no readable current snapshot;
* ``policy_blocked``  the policy gate denies this destination/signal (``destination`` may still
                      name the policy entry); pause and enablement are NOT checked here;
* ``ineligible_scope`` the snapshot predates the last data-affecting scope change or was
                      collected under another policy version: it can never be exported;
* ``ok``              at least one batch would be spooled;
* ``no_changes``      logs only: every current record is already known to the registry;
* ``no_records``      nothing to export at all (no ledger rows, or an empty projection).

Ledger/registry problems propagate as ``RecordsUnavailable`` / ``RegistryCorrupt`` (stable
``condition`` codes, no raw text) rather than being flattened into a state.
"""


STAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
STATES = (
    "ok",
    "no_snapshot",
    "ineligible_scope",
    "no_changes",
    "policy_blocked",
    "no_records",
)


@dataclass(frozen=True)
class PreviewCandidate:
    state: str
    candidate: Candidate | None
    envelope: SnapshotEnvelope | None
    destination: DestinationPolicy | None
    policy: ProfilePolicy
    ephemeral_key: bool


def current_envelope(profile: str) -> SnapshotEnvelope | None:
    """The verified current snapshot; ``None`` when absent or unreadable. Never writes."""
    try:
        generation = snapshots.current_generation(profile)
        return None if generation is None else snapshots.read_generation(profile, generation)
    except (OSError, ValueError):
        return None


def _simulated_scope(
    profile: str, destination: str, signal: Signal, scope: dict[str, object], stamp: str
) -> ScopeState:
    """The scope state a cycle WOULD act on, without persisting anything."""
    known = export_policy.read_scope(profile, destination, signal)
    return export_policy.next_scope(scope, known, stamp)


def _select(
    profile: str, destination: str, signal: Signal, gate: Gate, stamp: str
) -> PreviewCandidate:
    if not gate.ok or gate.destination is None:
        return PreviewCandidate("policy_blocked", None, None, gate.destination, gate.policy, False)
    dest, current = gate.destination, gate.policy
    envelope = current_envelope(profile)
    if envelope is None:
        return PreviewCandidate("no_snapshot", None, None, dest, current, False)
    key, ephemeral = key_for(profile, create=False)
    installation = export_ids.installation_id(key, destination, profile)
    scope = export_policy.scope_document(current, dest, signal, installation)
    state = _simulated_scope(profile, destination, signal, scope, stamp)
    if not export_policy.snapshot_eligible(envelope, state, current.policy_version):
        return PreviewCandidate("ineligible_scope", None, envelope, dest, current, ephemeral)
    if signal is Signal.METRICS:
        built = build_metrics(profile, dest, current, envelope, key, ephemeral=ephemeral)
    else:
        built = build_logs(profile, dest, current, envelope, key, ephemeral=ephemeral)
    if built.batches:
        verdict = "ok"
    elif signal is Signal.LOGS and export_registry.load_registry(profile, destination).entries:
        verdict = "no_changes"
    else:
        verdict = "no_records"
    return PreviewCandidate(verdict, built, envelope, dest, current, ephemeral)


def preview_candidate(
    profile: str, destination: str, signal: Signal, *, now: datetime | None = None
) -> PreviewCandidate:
    """The exact sanitized candidate a send would spool now. Writes NOTHING (no key, no scope)."""
    moment = (now or datetime.now(UTC)).astimezone(UTC)
    sig = Signal(signal)
    gate = export_policy.gate(profile, destination, sig, need_enabled=False, need_unpaused=False)
    return _select(profile, destination, sig, gate, moment.strftime(STAMP_FORMAT))
