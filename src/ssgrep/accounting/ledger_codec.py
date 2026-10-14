"""JSON-safe encodings of the replay ledger's persisted state (design D8).

Thin, typed wrappers over :mod:`plain_codec`. Writers emit exactly the structure
``contracts.canonical_bytes`` hashes, except that a receipt's private ``content``
is NEVER serialized. Readers are strict: unknown fields (a newer schema), a
``schema`` other than this build's, wrong types, and any private ``content`` in a
document are ``ValueError``, so nothing is silently dropped or trusted.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from typing import Any, TypeVar

from ssgrep.accounting.contracts import SCHEMA_VERSION
from ssgrep.accounting.plain_codec import from_plain, to_plain
from ssgrep.accounting.receipts import ReceiptRecord
from ssgrep.accounting.replay import FrozenRoot, LedgerState
from ssgrep.accounting.revisions import PeriodRow
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
        for f in dataclasses.fields(value):
            _checked(getattr(value, f.name))
    elif isinstance(value, (tuple, list)):
        for item in value:
            _checked(item)
    return value


def _decode(cls: type[T], raw: Mapping[str, Any]) -> T:
    if not isinstance(raw, Mapping):
        raise ValueError(f"expected an object for {cls.__name__}, got {type(raw).__name__}")
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
