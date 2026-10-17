"""Ledger-log projection: committed collector ledger -> pseudonymous, revisioned log records.

Three parts, all stdlib-only and float-free:

* ``ledger_rows`` (read-only) restores the committed ledger exactly like a collector cycle but
  never writes and never takes the worker lease;
* ``project_candidates`` (pure) builds one sanitized ``Candidate`` per CURRENT period row x
  leaf. A body is constructed POSITIVELY from the leaf's numeric fields, the unit label, the
  methodology hash and keyed pseudonyms: no atom id, candidate id, digest, locator, receipt
  id, root, turn, branch or any content can reach it, because none is ever read into it;
* revisions, the local registry and its state transitions live in ``registry.py``; this module
  exposes the sanitized ledger-log projection and OTLP log-record construction.

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

from ssgrep.accounting.collect import state as collector_state, store as collector_store
from ssgrep.accounting.collect.collector import CollectError
from ssgrep.accounting.collect.replay import LedgerRows
from ssgrep.accounting.collect.state import VersionedLedger
from ssgrep.accounting.export import ids as export_ids
from ssgrep.accounting.export.otlp import LogRecord
from ssgrep.accounting.export.registry import (
    CURRENCY as _CURRENCY,
    INT64_MAX as _INT64_MAX,
    METHOD_NAME as _METHOD_NAME,
    METHODOLOGY as _METHODOLOGY,
    PERIOD as _PERIOD,
    SCENARIOS as _SCENARIOS,
    SCHEMA as _SCHEMA,
    TOKENIZER as _TOKENIZER,
    UNIT_NAME as _UNIT_NAME,
    Candidate as _Candidate,
    Planned as _Planned,
    RecordsUnavailable as _RecordsUnavailable,
)
from ssgrep.accounting.ledger.contracts import digest_of
from ssgrep.accounting.ledger.revisions import PeriodRow, current_rows
from ssgrep.accounting.ledger.valuation import Leaf, ScenarioLeaf, rollup
from ssgrep.accounting.privacy import storage
from ssgrep.accounting.privacy.policy import valid_collection_scope

__all__ = ["ledger_rows", "log_records", "project_candidates", "root_key_of"]

_CONDITION_FALLBACK = "state_corrupt"


# --- A. read-only ledger access ------------------------------------------------------------


def _condition(error: CollectError) -> str:
    code = error.condition
    return code if isinstance(code, str) and _UNIT_NAME.fullmatch(code) else _CONDITION_FALLBACK


def ledger_rows(profile: str, *, collection_scope: dict | None = None) -> LedgerRows | None:
    """Rows of the committed collector ledger; None before one exists. Never writes.

    Raises ``RecordsUnavailable`` with a stable condition code (no raw error text) when the
    collector storage is unsafe, unsupported, or corrupt.
    """
    from ssgrep.accounting.export.consent import migration_of

    try:
        checkpoint = collector_store.load_checkpoint(profile)
        if checkpoint is None or not checkpoint.methodology:
            return None
        migration = migration_of(checkpoint)
        if migration.reason:
            raise _RecordsUnavailable(migration.reason)
        if collection_scope is not None and (
            not valid_collection_scope(checkpoint.collection_scope)
            or checkpoint.collection_scope != collection_scope
        ):
            raise _RecordsUnavailable("ineligible_scope")
        state = collector_state.load_state(profile, checkpoint, checkpoint.methodology)
        if collection_scope is not None and state.collection_scope != collection_scope:
            raise _RecordsUnavailable("ineligible_scope")
        if state.ledger is None:
            return None
        ledger = VersionedLedger.restore_versioned(state.ledger, state.superseded, state.archives)
        if ledger.methodology.hash != checkpoint.methodology:
            raise _RecordsUnavailable("state_corrupt")
        from ssgrep.accounting.collect.steps import archive_retired_rows

        archive_retired_rows(profile, ledger, checkpoint.blobs)
        return ledger.rows()
    except CollectError as error:
        raise _RecordsUnavailable(_condition(error)) from None
    except storage.UnsafeStorageError:
        raise _RecordsUnavailable("storage_unsafe") from None
    except ValueError as error:
        condition = (
            "methodology_retired" if str(error) == "methodology_read_only" else _CONDITION_FALLBACK
        )
        raise _RecordsUnavailable(condition) from None
    except (OSError, KeyError, TypeError, AttributeError, AssertionError):
        raise _RecordsUnavailable(_CONDITION_FALLBACK) from None


# --- B. pure projection --------------------------------------------------------------------


def root_key_of(root: str) -> str:
    """The local-only 16-hex handle of a conversation root (never part of a body)."""
    return hashlib.sha256(root.encode()).hexdigest()[:16]


def _count(value: int) -> int:
    if type(value) is not int or not 0 <= value <= _INT64_MAX:
        raise ValueError("count out of range")
    return value


def _scenario_leaf(leaf: Leaf, name: str) -> ScenarioLeaf | None:
    return next((s for s in leaf.scenarios if s.scenario.value == name), None)


def _unit_ok(leaf: Leaf) -> bool:
    unit = leaf.unit
    return (
        _UNIT_NAME.fullmatch(unit.unit) is not None
        and _METHOD_NAME.fullmatch(unit.method) is not None
        and (unit.tokenizer is None or _TOKENIZER.fullmatch(unit.tokenizer) is not None)
        and (unit.currency is None or _CURRENCY.fullmatch(unit.currency) is not None)
    )


def _valid_period(period: str) -> bool:
    if not _PERIOD.fullmatch(period):
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
    from ssgrep.accounting.export.projection import project_scenario

    for name in _SCENARIOS:
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
        "schema": _SCHEMA,
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
) -> tuple[list[_Candidate], int]:
    """One candidate per CURRENT period row x leaf, plus how many leaves were omitted.

    Superseded period revisions never contribute. A leaf is omitted (counted, never quoted)
    when its unit, period or methodology label is not allowlist-clean, when its numbers do
    not fit, or when two leaves would claim one record id with different content.
    """
    found: dict[str, _Candidate] = {}
    conflicts: set[str] = set()
    omitted = 0
    for row in current_rows(rows.periods, rows.namespace):
        for leaf in row.leaves:
            if (
                not _unit_ok(leaf)
                or leaf.methodology != rows.namespace
                or not _valid_period(row.period)
                or not _METHODOLOGY.fullmatch(leaf.methodology)
            ):
                omitted += 1
                continue
            try:
                record_id, task_id, core = _core(row, leaf, key=key, destination=destination)
            except (ValueError, AssertionError):
                omitted += 1
                continue
            unit = leaf.unit
            cand = _Candidate(
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


# --- D. OTLP log records -------------------------------------------------------------------


def log_records(planned: Iterable[_Planned], *, time_unix_nano: int) -> list[LogRecord]:
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
