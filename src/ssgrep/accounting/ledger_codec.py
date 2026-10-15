"""JSON-safe encodings of the replay ledger's persisted state (design D8).

Thin, typed wrappers over :mod:`plain_codec`. Writers emit exactly the structure
``contracts.canonical_bytes`` hashes, except that a receipt's private ``content``
is NEVER serialized. Readers are strict: unknown fields (a newer schema), a
``schema`` other than this build's, wrong types, and any private ``content`` in a
document are ``ValueError``, so nothing is silently dropped or trusted.
"""

from __future__ import annotations

import dataclasses
import types
from collections.abc import Iterable, Mapping
from functools import cache
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

from ssgrep.accounting.contracts import SCHEMA_VERSION
from ssgrep.accounting.ledger_contracts import ScenarioEstimate
from ssgrep.accounting.plain_codec import from_plain, to_plain
from ssgrep.accounting.receipts import ReceiptRecord
from ssgrep.accounting.replay import FrozenRoot, LedgerState
from ssgrep.accounting.revisions import PeriodRow
from ssgrep.accounting.valuation import UnitEstimate
from ssgrep.accounting.valuation_inputs import EvidenceBundle

T = TypeVar("T")


def _public(record: ReceiptRecord) -> dict[str, Any]:
    """Plain receipt record without the private ``content`` key."""
    plain = to_plain(dataclasses.replace(record, content=None))
    del plain["content"]
    return plain


def _checked(value: T) -> T:
    """Reject documents written by a different schema version or carrying content."""
    if isinstance(value, ReceiptRecord) and value.content is not None:
        raise ValueError("receipt documents must not carry private content")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        schema = getattr(value, "schema", SCHEMA_VERSION)
        if schema != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema {schema!r}; this build reads {SCHEMA_VERSION}")
        if isinstance(value, (UnitEstimate, ScenarioEstimate)):
            if value.complete != (value.net.value is not None):
                raise ValueError("complete_net_metadata_mismatch")
            if value.complete and value.net.value != value.known_net:
                raise ValueError("complete_net_differs_from_known_subtotal")
        for f in dataclasses.fields(value):
            _checked(getattr(value, f.name))
    elif isinstance(value, (tuple, list)):
        for item in value:
            _checked(item)
    return value


_REVISED_FIELDS = {
    "Candidate": {"eligibility", "eligibility_reasons", "provenance", "rarity_lower_bound_milli"},
    "Debit": {"raw_min", "raw_max", "work_ns", "mapping_version", "assumptions"},
    "Leaf": {"debit_min", "debit_max", "eligibility", "provenance", "assumptions", "raw_time_unit"},
    "ScenarioLeaf": {"eligibility"},
    "Methodology": {"valuation_version"},
    "ReceiptRecord": {"raw_unit", "debit_raw", "debit_raw_min", "debit_raw_max"},
    "Receipt": {"eligibility", "eligibility_reasons", "provenance", "scenarios"},
    "ValuationAtom": {"work_method", "cost_reason", "actor", "t_start_ms", "t_end_ms"},
    "UnitEstimate": {"bounds", "eligibility", "provenance"},
    "ScenarioEstimate": {
        "known_net",
        "complete",
        "credit_missing",
        "debit_missing",
        "known_atoms",
        "known_debits",
        "bounds",
        "eligibility",
        "provenance",
        "unit",
    },
}


@cache
def _declared(cls):
    return get_type_hints(cls)


def _revision_fields(cls: Any, raw: Any) -> None:
    """Check trusted declared types before from_plain can apply constructor defaults."""
    if raw is None:
        return
    origin, args = get_origin(cls), get_args(cls)
    if origin in (Union, types.UnionType):
        for member in args:
            if member is not type(None):
                _revision_fields(member, raw)
    elif origin in (tuple, list) and isinstance(raw, list):
        if len(args) == 2 and args[1] is Ellipsis:
            for value in raw:
                _revision_fields(args[0], value)
        else:
            for member, value in zip(args, raw, strict=False):
                _revision_fields(member, value)
    elif origin in (dict, Mapping) and isinstance(raw, Mapping):
        for value in raw.values():
            _revision_fields(args[1], value)
    elif dataclasses.is_dataclass(cls) and isinstance(raw, Mapping):
        required = _REVISED_FIELDS.get(cls.__name__, set())
        if required - raw.keys():
            raise ValueError(
                "legacy_valuation_migration_required:" + ",".join(sorted(required - raw.keys()))
            )
        if cls.__name__ == "Methodology" and raw.get("valuation_version") != "honest_bounds_ns_v3":
            raise ValueError("time_precision_namespace_migration_required")
        for name, member in _declared(cls).items():
            if name in raw:
                _revision_fields(member, raw[name])


def _decode(cls: type[T], raw: Mapping[str, Any]) -> T:
    if not isinstance(raw, Mapping):
        raise ValueError(f"expected an object for {cls.__name__}, got {type(raw).__name__}")
    _revision_fields(cls, raw)
    return _checked(from_plain(cls, dict(raw)))


def _records(records: Iterable[ReceiptRecord]) -> list[dict[str, Any]]:
    return [_public(r) for r in records]


def receipt_record_to_dict(record: ReceiptRecord) -> dict[str, Any]:
    return _public(record)


def receipt_record_from_dict(raw: Mapping[str, Any]) -> ReceiptRecord:
    return _decode(ReceiptRecord, raw)


def frozen_to_dict(frozen: FrozenRoot) -> dict[str, Any]:
    plain = to_plain(dataclasses.replace(frozen, receipts=()))
    plain["receipts"] = _records(frozen.receipts)
    return plain


def frozen_from_dict(raw: Mapping[str, Any]) -> FrozenRoot:
    return _decode(FrozenRoot, raw)


def ledger_state_to_dict(state: LedgerState) -> dict[str, Any]:
    plain = to_plain(dataclasses.replace(state, frozen=()))
    plain["frozen"] = [frozen_to_dict(f) for f in state.frozen]
    return plain


def ledger_state_from_dict(raw: Mapping[str, Any]) -> LedgerState:
    return _decode(LedgerState, raw)


def bundle_to_dict(bundle: EvidenceBundle) -> dict[str, Any]:
    return to_plain(bundle)


def bundle_from_dict(raw: Mapping[str, Any]) -> EvidenceBundle:
    return _decode(EvidenceBundle, raw)


def period_row_to_dict(row: PeriodRow) -> dict[str, Any]:
    return to_plain(row)


def period_row_from_dict(raw: Mapping[str, Any]) -> PeriodRow:
    return _decode(PeriodRow, raw)
