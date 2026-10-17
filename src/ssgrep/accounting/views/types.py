"""Typed views returned by the read facade (``service.py``) and their JSON form.

Snapshot payload contract read by the facade (written by the collector):

* ``windows[<w>]["observations"]``: ``{name: Quantity}`` for ``searches``,
  ``shows``, ``failed``, ``visible_bytes`` (extra names are shown after).
* ``windows[<w>]["estimates"]``: ``{unit_key: {scenario: {credit, debit, net}}}``
  with ``unit_key`` in ``context_tokens`` / ``work_seconds`` and each leaf a
  ``Quantity`` object ``{value, unit, basis, method, reason}``.
* ``coverage``: integer counters (``recorded_lookups``, ``attributed_lookups``,
  ``candidate_receipts``, ``source_costs_available``, ``ambiguous_joins``,
  ``unavailable_sources``), ``index`` (``available``/``unavailable``),
  ``exclusions`` (reason -> count), ``partial`` (bool).
* ``estimates["representative_receipt"]``: receipt id that ``savings explain``
  can resolve, or null.

Anything missing or malformed becomes an unavailable quantity with a reason,
never zero. Floats are rejected so no JSON consumer sees locale/NaN hazards.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any

from ssgrep.accounting.ledger.contracts import PERSONAL_PROFILE, RUNTIMES, Basis, Quantity, Scenario
from ssgrep.accounting.ledger.ledger_contracts import (
    ScenarioEstimate,
    SnapshotState,
    valid_profile_id,
)
from ssgrep.accounting.privacy.policy import PolicyError

if TYPE_CHECKING:
    from ssgrep.accounting.views.diagnostics import DiagnosticGroup

MEASURED_ORDER = ("searches", "shows", "failed", "visible_bytes")
ESTIMATE_UNITS = ("context_tokens", "work_seconds")
COVERAGE_COUNTERS = (
    "recorded_lookups",
    "attributed_lookups",
    "candidate_receipts",
    "source_costs_available",
    "ambiguous_joins",
    "unavailable_sources",
    "replay_unavailable_sessions",
    "frequency_revision_unavailable",
)


def command_for(profile: str, text: str) -> str:
    """``ssgrep <text>`` with the selected profile preserved (omitted for the default)."""
    suffix = "" if profile == PERSONAL_PROFILE else f" --profile {profile}"
    return f"ssgrep {text}{suffix}"


def parse_quantity(raw: Any, unit: str, missing_reason: str) -> Quantity:
    """A ``Quantity`` from snapshot JSON; anything unusable is unavailable."""
    if raw is None:
        return Quantity.unavailable(unit, missing_reason)
    try:
        value = raw["value"]
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise TypeError("value must be an integer")
        basis = Basis(raw["basis"])
        return Quantity(
            value=value,
            unit=str(raw.get("unit") or unit),
            basis=basis,
            method=str(raw.get("method", "")),
            reason=str(raw.get("reason", "")),
        )
    except (KeyError, TypeError, ValueError, AttributeError):
        return Quantity.unavailable(unit, "malformed_quantity")


def quantity_json(quantity: Quantity) -> dict[str, object]:
    return {
        "value": quantity.value,
        "unit": quantity.unit,
        "basis": quantity.basis.value,
        "method": quantity.method,
        "reason": quantity.reason or None,
    }


@dataclass(frozen=True)
class ScenarioRow:
    """One unit (e.g. context tokens) across the three scenarios."""

    unit_key: str
    credit: tuple[tuple[Scenario, Quantity], ...]
    debit: tuple[tuple[Scenario, Quantity], ...]
    net: tuple[tuple[Scenario, Quantity], ...]
    estimates: tuple[ScenarioEstimate, ...] = ()


@dataclass(frozen=True)
class NextAction:
    command: str
    why: str


@dataclass(frozen=True)
class Condition:
    """One named state of a view: its cause and the real command that addresses it."""

    code: str
    cause: str
    action: NextAction | None = None


@dataclass(frozen=True)
class GroupRow:
    """One `--group-by` row: measured counts and per-unit scenario estimates of one key."""

    key: str
    measured: tuple[tuple[str, Quantity], ...]
    scenarios: tuple[ScenarioRow, ...]
    receipts: int = 0


@dataclass(frozen=True)
class DeliveryLine:
    """Delivery state of one destination/signal (collector acceptance, never 'synced')."""

    destination_id: str
    signal: str
    enabled: bool
    state: str  # disabled|waiting|pending|offline|partial_export|credentials_expired|blocked|
    # unknown|acknowledged|unreadable
    cause: str
    pending_batches: int = 0
    rejected_points: int = 0
    unknown_batches: int = 0
    partial_batches: int = 0
    lost_records: int = 0
    collected_generation: int | None = None
    attempted_generation: int | None = None
    acked_generation: int | None = None
    last_acked_at: str | None = None
    last_attempt_at: str | None = None
    next_retry_utc: str | None = None
    last_error_category: str | None = None
    blocked_reason: str | None = None
    acked_age_seconds: int | None = None
    attempt_age_seconds: int | None = None
    action: NextAction | None = None


@dataclass(frozen=True)
class Scope:
    window: str = "7d"
    since: str | None = None
    until: str | None = None
    project: str | None = None
    runtime: str | None = None
    session: str | None = None
    task: str | None = None
    by: str | None = None
    include_children: bool = False

    @property
    def is_default(self) -> bool:
        return not (
            self.since
            or self.until
            or self.project
            or self.runtime
            or self.session
            or self.task
            or self.by
        )


@dataclass(frozen=True)
class SummaryView:
    profile: str
    state: SnapshotState
    paused: bool
    paused_at: str | None
    stale: bool
    scope: Scope
    collected_at: str | None
    age_seconds: int | None
    collection_enabled: bool
    exports: tuple[tuple[str, str], ...]
    measured: tuple[tuple[str, Quantity], ...] = ()
    scenarios: tuple[ScenarioRow, ...] = ()
    coverage: dict[str, int | None] = field(default_factory=dict)
    exclusions: tuple[tuple[str, int], ...] = ()
    partial: bool = False
    index: str = "unknown"
    receipt_id: str | None = None
    snapshot_id: str | None = None
    generation: int | None = None
    methodology_id: str | None = None
    policy_version: int | None = None
    coverage_through: str | None = None
    warnings: tuple[str, ...] = ()
    next_actions: tuple[NextAction, ...] = ()
    reason: str | None = None
    conditions: tuple[Condition, ...] = ()
    delivery: tuple[DeliveryLine, ...] = ()
    groups: tuple[GroupRow, ...] = ()
    group_total: int | None = None
    next_cursor: str | None = None
    filtered: bool = False
    estimates_reason: str | None = None
    frozen_roots: int | None = None
    evidence_expired_before: str | None = None
    limit: int = 20

    @property
    def diagnostic_groups(self) -> tuple[DiagnosticGroup, ...]:
        """Stable, severity-ordered groups; no I/O or hidden truncation."""
        from ssgrep.accounting.views.diagnostics import grouped_warnings

        return grouped_warnings(self.warnings)

    @property
    def diagnostic_total(self) -> int:
        return sum(group.count for group in self.diagnostic_groups)

    @property
    def diagnostic_alias_duplicates(self) -> int:
        return len(self.warnings) - self.diagnostic_total

    @property
    def replay_unavailable(self) -> bool:
        """Positive snapshot-wide missingness; unknown counts never become zero."""
        return any(
            value is not None and value > 0
            for name in ("replay_unavailable_sessions", "frequency_revision_unavailable")
            for value in (self.coverage.get(name),)
        )

    @property
    def has_values(self) -> bool:
        return bool(self.measured or self.scenarios)

    @property
    def has_negative(self) -> bool:
        return any(
            q.value is not None and q.value < 0 for row in self.scenarios for _, q in row.net
        )

    @property
    def no_reuse(self) -> bool:
        return self.has_values and self.coverage.get("candidate_receipts") == 0


@dataclass(frozen=True)
class ReceiptView:
    receipt_id: str
    revision: int
    root: str
    task_turn: str | None
    candidate_type: str
    atoms: tuple[str, ...]
    allocations: tuple[tuple[str, tuple[str, ...]], ...]
    exclusions: tuple[str, ...]
    costs: tuple[tuple[str, Quantity], ...]
    methodology: str
    evidence_available: bool
    execution_ids: tuple[str, ...]
    content: str | None = None
    note: str | None = None
    schema: int = 1
    unit: str | None = None
    period: str | None = None
    missing_costs: tuple[tuple[str, str], ...] = ()
    credit_raw: tuple[tuple[str, int], ...] = ()
    identities: tuple[tuple[str, str], ...] = ()
    atom_digests: tuple[tuple[str, str], ...] = ()
    retention_boundary: str | None = None
    reuse: tuple[tuple[str, str], ...] | None = None
    eligibility: str = "unavailable"
    eligibility_reasons: tuple[str, ...] = ()
    provenance: tuple[tuple[str, str], ...] = ()
    scenarios: tuple[ScenarioEstimate, ...] = ()
    exposure_ids: tuple[str, ...] = ()
    raw_unit: str = ""
    debit_raw: int | None = None
    debit_raw_min: int | None = None
    debit_raw_max: int | None = None


@dataclass(frozen=True)
class HistoryEntry:
    period: str
    revision: int
    status: str
    current: bool
    supersedes: int | None
    net: Quantity
    recomputable: bool = True
    kind: str = "original"
    scenarios: tuple[ScenarioEstimate, ...] = ()
    namespace: str = ""
    revision_name: str = ""


@dataclass(frozen=True)
class HistoryView:
    profile: str
    entries: tuple[HistoryEntry, ...]
    current_total: Quantity
    next_cursor: str | None
    reason: str | None = None
    warnings: tuple[str, ...] = ()
    next_actions: tuple[NextAction, ...] = ()
    since: str | None = None
    until: str | None = None
    limit: int = 20
    scenarios: tuple[ScenarioEstimate, ...] = ()


MAX_LIMIT = 100
BY_CHOICES = ("task", "runtime", "project", "day")
_MAX_FILTER = 512
_RFC3339 = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2}(\.\d{1,9})?)?(Z|[+-]\d{2}:\d{2})$")
_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _usage(condition: str, message: str, command: str = "ssgrep savings --help") -> PolicyError:
    return PolicyError(condition, message, command, exit_code=2)


def parse_instant(text: str, name: str) -> str:
    """Normalize an RFC3339 or YYYY-MM-DD boundary to ``YYYY-MM-DDTHH:MM:SSZ``."""
    try:
        if _DAY.fullmatch(text):
            moment = datetime.combine(date.fromisoformat(text), datetime.min.time(), tzinfo=UTC)
        elif _RFC3339.fullmatch(text):
            moment = datetime.fromisoformat(text).astimezone(UTC)
        else:
            raise ValueError(text)
    except ValueError:
        raise _usage(
            "invalid_interval",
            f"{name} must be RFC3339 (2026-10-01T00:00:00Z) or YYYY-MM-DD, not {text[:40]!r}.",
        ) from None
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def check_limit(limit: int) -> int:
    if not 1 <= limit <= MAX_LIMIT:
        raise _usage("invalid_limit", f"--limit must be between 1 and {MAX_LIMIT}, not {limit}.")
    return limit


def check_profile(profile: str) -> str:
    if not valid_profile_id(profile):
        raise _usage(
            "invalid_profile",
            "Profile ids are 1-32 characters of a-z, 0-9, '-' or '_' and never paths.",
            "ssgrep savings status",
        )
    return profile


def _bounded(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    if not value or len(value) > _MAX_FILTER or "\0" in value:
        raise _usage(
            "invalid_filter", f"{name} must be a non-empty value under {_MAX_FILTER} bytes."
        )
    return value


def interval(since: str | None, until: str | None) -> tuple[str | None, str | None]:
    start = parse_instant(since, "start of the UTC interval") if since else None
    end = parse_instant(until, "end of the UTC interval") if until else None
    if start and end and start >= end:
        raise _usage("invalid_interval", "Start must be before end (half-open UTC interval).")
    return start, end


def build_scope(
    *,
    since: str | None = None,
    until: str | None = None,
    project: str | None = None,
    runtime: str | None = None,
    session: str | None = None,
    task: str | None = None,
    by: str | None = None,
    include_children: bool = False,
) -> Scope:
    start, end = interval(since, until)
    if runtime is not None and runtime not in RUNTIMES:
        raise _usage("invalid_filter", f"--runtime must be one of: {', '.join(RUNTIMES)}.")
    if by is not None and by not in BY_CHOICES:
        raise _usage("invalid_filter", f"--group-by must be one of: {', '.join(BY_CHOICES)}.")
    if include_children and not session:
        raise _usage("invalid_filter", "--children only applies together with --session.")
    return Scope(
        window="7d",
        since=start,
        until=end,
        project=_bounded(project, "--project"),
        runtime=runtime,
        session=_bounded(session, "--session"),
        task=_bounded(task, "--task"),
        by=by,
        include_children=include_children,
    )


_CURSOR_RE = re.compile(r"^([a-z]{1,8})1\.(\d{1,7})\.([0-9a-f]{16})$")


def _cursor_tag(kind: str, offset: int, binding: str) -> str:
    return hashlib.sha256(f"{kind}|{offset}|{binding}".encode()).hexdigest()[:16]


def issue(kind: str, offset: int, binding: str) -> str:
    """Issue a page token bound to its view, offset, and selection digest."""
    return f"{kind}1.{offset}.{_cursor_tag(kind, offset, binding)}"


def redeem(kind: str, cursor: str | None, binding: str, command: str) -> int:
    """Return the bound row offset; reject edited or cross-view page tokens."""
    if cursor is None:
        return 0
    match = _CURSOR_RE.fullmatch(cursor)
    if match is None or match.group(1) != kind:
        raise PolicyError("invalid_cursor", "That is not a cursor of this view.", command, 2)
    offset = int(match.group(2))
    if _cursor_tag(kind, offset, binding) != match.group(3):
        raise PolicyError(
            "invalid_cursor",
            "Page token does not match this snapshot and selection (the data or filters "
            "changed, or the token was edited). Start again without the page token.",
            command,
            2,
        )
    return offset
