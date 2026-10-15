"""Ledger-log projection: committed collector ledger -> pseudonymous, revisioned log records.

Three parts, all stdlib-only and float-free:

* ``ledger_rows`` (read-only) restores the committed ledger exactly like a collector cycle but
  never writes and never takes the worker lease;
* ``project_candidates`` (pure) builds one sanitized ``Candidate`` per CURRENT period row x
  leaf. A body is constructed POSITIVELY from the leaf's numeric fields, the unit label, the
  methodology hash and keyed pseudonyms: no atom id, candidate id, digest, locator, receipt
  id, root, turn, branch or any content can reach it, because none is ever read into it;
* revisions, the local registry and the state transitions live in ``export_records_plan``,
  ``export_records_state`` and ``export_registry`` and are re-exported here as the public
  surface used by delivery, purge and tombstones.

Record identity: ``record_id`` pseudonymises (root, turn, branch, period, unit) and
``task_id`` (root, turn, branch), both keyed with the install's pseudonym key and scoped to
the destination, so two destinations see unrelated ids and nothing is recomputable from a
path or conversation id without the key. ``root_key`` (unkeyed hash of the root) exists only
in the local registry so purge can find a conversation's exported records; it is never sent.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import date
from typing import Any

from ssgrep.accounting import collector_state, collector_store, export_ids, storage
from ssgrep.accounting.collector_ledger import VersionedLedger
from ssgrep.accounting.collector_types import CollectError
from ssgrep.accounting.contracts import digest_of
from ssgrep.accounting.export_authority import migration_of
from ssgrep.accounting.export_estimates import project_scenario
from ssgrep.accounting.export_model import LogRecord
from ssgrep.accounting.export_records_model import (
    BATCH_ID,
    CURRENCY,
    INT64_MAX,
    METHOD_NAME,
    METHODOLOGY,
    PERIOD,
    SCENARIOS,
    SCHEMA,
    TOKENIZER,
    UNIT_NAME,
    Candidate,
    Entry,
    Planned,
    RecordsUnavailable,
    Registry,
    RegistryCorrupt,
    State,
)
from ssgrep.accounting.export_records_plan import build_deletions, plan_revisions
from ssgrep.accounting.export_records_state import (
    ack,
    discard,
    ids_before,
    ids_for_roots,
    queue,
    unresolve,
)
from ssgrep.accounting.export_registry import load_registry, update_registry
from ssgrep.accounting.policy_rules import valid_collection_scope
from ssgrep.accounting.replay import LedgerRows
from ssgrep.accounting.revisions import PeriodRow, current_rows
from ssgrep.accounting.valuation import Leaf, ScenarioLeaf, rollup

__all__ = [
    "Candidate",
    "Entry",
    "Planned",
    "RecordsUnavailable",
    "Registry",
    "RegistryCorrupt",
    "State",
    "build_deletions",
    "ledger_rows",
    "load_registry",
    "log_records",
    "mark_acked",
    "mark_partial",
    "mark_quarantined",
    "mark_queued",
    "mark_unknown",
    "plan_revisions",
    "project_candidates",
    "records_before",
    "records_for_roots",
    "reset_queued",
    "root_key_of",
]

_CONDITION_FALLBACK = "state_corrupt"


# --- A. read-only ledger access ------------------------------------------------------------


def _condition(error: CollectError) -> str:
    code = error.condition
    return code if isinstance(code, str) and UNIT_NAME.fullmatch(code) else _CONDITION_FALLBACK


def ledger_rows(profile: str, *, collection_scope: dict | None = None) -> LedgerRows | None:
    """Rows of the committed collector ledger; None before one exists. Never writes.

    Raises ``RecordsUnavailable`` with a stable condition code (no raw error text) when the
    collector storage is unsafe, unsupported, or corrupt.
    """
    try:
        checkpoint = collector_store.load_checkpoint(profile)
        if checkpoint is None or not checkpoint.methodology:
            return None
        migration = migration_of(checkpoint)
        if migration.reason:
            raise RecordsUnavailable(migration.reason)
        if collection_scope is not None and (
            not valid_collection_scope(checkpoint.collection_scope)
            or checkpoint.collection_scope != collection_scope
        ):
            raise RecordsUnavailable("ineligible_scope")
        state = collector_state.load_state(profile, checkpoint, checkpoint.methodology)
        if collection_scope is not None and state.collection_scope != collection_scope:
            raise RecordsUnavailable("ineligible_scope")
        if state.ledger is None:
            return None
        ledger = VersionedLedger.restore_versioned(state.ledger, state.superseded, state.archives)
        if ledger.methodology.hash != checkpoint.methodology:
            raise RecordsUnavailable("state_corrupt")
        return ledger.rows()
    except CollectError as error:
        raise RecordsUnavailable(_condition(error)) from None
    except storage.UnsafeStorageError:
        raise RecordsUnavailable("storage_unsafe") from None
    except (OSError, ValueError, KeyError, TypeError, AttributeError, AssertionError):
        raise RecordsUnavailable(_CONDITION_FALLBACK) from None


# --- B. pure projection --------------------------------------------------------------------


def root_key_of(root: str) -> str:
    """The local-only 16-hex handle of a conversation root (never part of a body)."""
    return hashlib.sha256(root.encode()).hexdigest()[:16]


def _count(value: int) -> int:
    if type(value) is not int or not 0 <= value <= INT64_MAX:
        raise ValueError("count out of range")
    return value


def _scenario_leaf(leaf: Leaf, name: str) -> ScenarioLeaf | None:
    return next((s for s in leaf.scenarios if s.scenario.value == name), None)


def _unit_ok(leaf: Leaf) -> bool:
    unit = leaf.unit
    return (
        UNIT_NAME.fullmatch(unit.unit) is not None
        and METHOD_NAME.fullmatch(unit.method) is not None
        and (unit.tokenizer is None or TOKENIZER.fullmatch(unit.tokenizer) is not None)
        and (unit.currency is None or CURRENCY.fullmatch(unit.currency) is not None)
    )


def _valid_period(period: str) -> bool:
    if not PERIOD.fullmatch(period):
        return False
    try:
        date.fromisoformat(period)
    except ValueError:
        return False
    return True


def _core(
    row: PeriodRow, leaf: Leaf, *, key: bytes, destination: str
) -> tuple[str, str, dict[str, Any]]:
    task, unit = leaf.task, leaf.unit
    estimates = {e.scenario.value: e for e in rollup([leaf], unit)}
    scenarios: dict[str, Any] = {}
    known: dict[str, int | str] = {
        "debit_raw": _count(leaf.debit_raw),
        "debit_known": _count(leaf.debit_known),
        "debit_missing_count": _count(len(leaf.debit_missing)),
        "raw_unit": "nanoseconds"
        if unit.method == "capped_gap_v1"
        else "characters"
        if unit.method == "chars_div4_v1"
        else unit.unit,
    }
    coverage: dict[str, Any] = {"recomputable": bool(row.recomputable)}
    for name in SCENARIOS:
        estimate = estimates[name]
        scenarios[name] = project_scenario(estimate)
        mine = _scenario_leaf(leaf, name)
        known[f"credit_raw_{name}"] = _count(mine.credit_raw if mine else 0)
        coverage[f"known_atoms_{name}"] = _count(mine.known_atoms if mine else 0)
        coverage[f"missing_{name}"] = _count(len(mine.missing) if mine else 0)
    coverage["debit_missing"] = _count(len(leaf.debit_missing))
    turn = task.turn or ""
    record_id = export_ids.pseudonym(
        key,
        "record",
        destination,
        task.root,
        turn,
        task.branch,
        row.period,
        unit.unit,
        unit.method,
        unit.tokenizer or "",
        unit.currency or "",
    )
    task_id = export_ids.pseudonym(key, "task", destination, task.root, turn, task.branch)
    core = {
        "schema": SCHEMA,
        "record_id": record_id,
        "task_id": task_id,
        "period": row.period,
        "unit": {
            "unit": unit.unit,
            "method": unit.method,
            "tokenizer": unit.tokenizer,
            "currency": unit.currency,
        },
        "scenarios": scenarios,
        "known_costs": known,
        "coverage": coverage,
        "methodology": leaf.methodology,
    }
    return record_id, task_id, core


def project_candidates(
    rows: LedgerRows, *, key: bytes, destination: str
) -> tuple[list[Candidate], int]:
    """One candidate per CURRENT period row x leaf, plus how many leaves were omitted.

    Superseded period revisions never contribute. A leaf is omitted (counted, never quoted)
    when its unit, period or methodology label is not allowlist-clean, when its numbers do
    not fit, or when two leaves would claim one record id with different content.
    """
    found: dict[str, Candidate] = {}
    conflicts: set[str] = set()
    omitted = 0
    for row in current_rows(rows.periods, rows.namespace):
        for leaf in row.leaves:
            if (
                not _unit_ok(leaf)
                or leaf.methodology != rows.namespace
                or not _valid_period(row.period)
                or not METHODOLOGY.fullmatch(leaf.methodology)
            ):
                omitted += 1
                continue
            try:
                record_id, task_id, core = _core(row, leaf, key=key, destination=destination)
            except (ValueError, AssertionError):
                omitted += 1
                continue
            unit = leaf.unit
            cand = Candidate(
                record_id,
                task_id,
                root_key_of(leaf.task.root),
                row.period,
                (unit.unit, unit.method, unit.tokenizer, unit.currency),
                leaf.methodology,
                core,
                digest_of(core),
            )
            held = found.setdefault(record_id, cand)
            if held.content_digest != cand.content_digest:
                conflicts.add(record_id)
                omitted += 1
    for record_id in conflicts:
        del found[record_id]
        omitted += 1
    ordered = sorted(found.values(), key=lambda c: (c.period, c.record_id))
    return ordered, omitted


# --- C. registry transitions ---------------------------------------------------------------


def _batch(batch_id: str) -> str:
    if not BATCH_ID.fullmatch(batch_id):
        raise ValueError("batch ids are bounded identifiers")
    return batch_id


def mark_queued(
    profile: str, destination: str, planned: Iterable[Planned], batch_id: str
) -> Registry:
    """Record ``planned`` as spooled (not yet sent) in ``batch_id``."""
    items = list(planned)
    batch = _batch(batch_id)
    return update_registry(profile, destination, lambda reg: queue(reg, items, batch))


def mark_acked(profile: str, destination: str, batch_id: str) -> Registry:
    """The collector fully accepted the batch: its queued states become ``acked``."""
    batch = _batch(batch_id)
    return update_registry(profile, destination, lambda r: ack(r, batch), create=False)


def mark_unknown(profile: str, destination: str, batch_id: str) -> Registry:
    """The batch was sent but its acceptance is unresolved (timeout, restart in flight)."""
    batch = _batch(batch_id)
    return update_registry(profile, destination, lambda r: unresolve(r, batch), create=False)


def mark_partial(profile: str, destination: str, batch_id: str) -> Registry:
    """The batch was partially accepted; item identities are unavailable, so all are unknown."""
    return mark_unknown(profile, destination, batch_id)


def mark_quarantined(profile: str, destination: str, batch_id: str) -> Registry:
    """A quarantined batch is treated like an unresolved one."""
    return mark_unknown(profile, destination, batch_id)


def reset_queued(profile: str, destination: str, batch_id: str) -> Registry:
    """The batch was discarded unsent: its queued states are removed."""
    batch = _batch(batch_id)
    return update_registry(profile, destination, lambda r: discard(r, batch), create=False)


def records_for_roots(profile: str, destination: str, root_keys: Iterable[str]) -> list[str]:
    """Record ids exported for conversations whose ``root_key`` is in ``root_keys``."""
    return ids_for_roots(load_registry(profile, destination), frozenset(root_keys))


def records_before(
    profile: str, destination: str, day: str, root_keys: Iterable[str] | None = None
) -> list[str]:
    """Record ids with a period before ``day`` (optionally only for the given roots)."""
    wanted = None if root_keys is None else frozenset(root_keys)
    return ids_before(load_registry(profile, destination), day, wanted)


# --- D. OTLP log records -------------------------------------------------------------------


def log_records(planned: Iterable[Planned], *, time_unix_nano: int) -> list[LogRecord]:
    """Canonical-JSON log records with the four allowlisted attributes, ordered as given."""
    return [
        LogRecord(
            time_unix_nano,
            json.dumps(item.body, sort_keys=True, separators=(",", ":")),
            (
                ("ssgrep.record.id", item.record_id),
                ("ssgrep.record.revision", item.revision),
                ("ssgrep.record.type", item.record_type),
                ("ssgrep.record.digest", item.body_digest),
            ),
        )
        for item in planned
    ]
