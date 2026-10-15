"""Closed JSON admission for exact407 collector publications, not valuation replay.

The historical writer emitted eight history fields, nineteen receipt fields,
and optionally a six-field reuse block. All nested records are closed too. No
unknown object can become a new long-lived original_utf8 plaintext container.
Original bytes are retained by the caller only AFTER this validator succeeds.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

HISTORY_FIELDS = frozenset(
    {"schema", "period", "revision", "status", "supersedes", "net", "kind", "recomputable"}
)
RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "receipt_id",
        "revision",
        "root",
        "task_turn",
        "candidate_type",
        "atoms",
        "allocations",
        "exclusions",
        "costs",
        "methodology",
        "evidence_available",
        "execution_ids",
        "unit",
        "period",
        "missing_costs",
        "atom_digests",
        "credit_raw",
        "identities",
    }
)
QUANTITY_FIELDS = frozenset({"value", "unit", "basis", "method", "reason"})
IDENTITY_FIELDS = frozenset({"frequency", "tokenizer", "pricing"})
REUSE_FIELDS = frozenset({"class", "relay", "report", "action_ok", "novel", "user_supplied"})
BASES = frozenset({"observed", "reported", "estimated", "unavailable"})
REUSE_CLASSES = frozenset(
    {"none_observed", "inspected", "action_reference", "validated", "contradicted"}
)
SCENARIOS = frozenset({"conservative", "central", "optimistic"})


def _fail() -> None:
    # Never echo rejected keys/values, which may themselves contain private data.
    raise ValueError("Original published document does not match its supported closed schema.")


def _object(value: Any, required: frozenset[str], optional: frozenset[str] = frozenset()) -> dict:
    if (
        not isinstance(value, dict)
        or not required <= value.keys()
        or value.keys() - required - optional
    ):
        _fail()
    return value


def _string(value: Any) -> None:
    if not isinstance(value, str):
        _fail()


def _integer(value: Any, *, positive: bool = False) -> None:
    if type(value) is not int or (positive and value < 1):
        _fail()


def _boolean(value: Any) -> None:
    if type(value) is not bool:
        _fail()


def _day(value: Any) -> None:
    if not isinstance(value, str) or len(value) != 10:
        _fail()
    try:
        if date.fromisoformat(value).isoformat() != value:
            _fail()
    except ValueError:
        _fail()


def _schema(value: Any) -> None:
    if type(value) is not int or value != 1:
        _fail()


def _strings(value: Any) -> None:
    if not isinstance(value, list):
        _fail()
    for item in value:
        _string(item)


def _pairs(value: Any, first, second) -> None:
    if not isinstance(value, list):
        _fail()
    for pair in value:
        if not isinstance(pair, list) or len(pair) != 2:
            _fail()
        first(pair[0])
        second(pair[1])


def _scenario(value: Any) -> None:
    if not isinstance(value, str) or value not in SCENARIOS:
        _fail()


def _nonnegative(value: Any) -> None:
    _integer(value)
    if value < 0:
        _fail()


def _quantity(value: Any) -> None:
    quantity = _object(value, QUANTITY_FIELDS)
    for field in ("unit", "basis", "method", "reason"):
        _string(quantity[field])
    if quantity["basis"] not in BASES:
        _fail()
    amount = quantity["value"]
    if amount is not None:
        _integer(amount)
    if (amount is None) != (quantity["basis"] == "unavailable"):
        _fail()
    if amount is None and not quantity["reason"]:
        _fail()


def _history(document: Any) -> None:
    if not isinstance(document, list):
        _fail()
    for value in document:
        row = _object(value, HISTORY_FIELDS)
        _schema(row["schema"])
        _day(row["period"])
        _integer(row["revision"], positive=True)
        if row["supersedes"] is not None:
            _integer(row["supersedes"], positive=True)
        _string(row["status"])
        _string(row["kind"])
        _boolean(row["recomputable"])
        _quantity(row["net"])


def _reuse(value: Any) -> None:
    block = _object(value, REUSE_FIELDS)
    if not isinstance(block["class"], str) or block["class"] not in REUSE_CLASSES:
        _fail()
    for field in REUSE_FIELDS - {"class"}:
        _boolean(block[field])


def _receipt(document: Any) -> None:
    row = _object(document, RECEIPT_FIELDS, frozenset({"reuse"}))
    _schema(row["schema"])
    _day(row["period"])
    _integer(row["revision"], positive=True)
    for field in ("receipt_id", "root", "candidate_type", "methodology", "unit"):
        _string(row[field])
    if row["task_turn"] is not None:
        _string(row["task_turn"])
    _boolean(row["evidence_available"])
    for field in ("atoms", "exclusions", "execution_ids"):
        _strings(row[field])
    _pairs(row["allocations"], _scenario, _strings)
    _pairs(row["costs"], _string, _quantity)
    _pairs(row["missing_costs"], _string, _string)
    _pairs(row["atom_digests"], _string, _string)
    _pairs(row["credit_raw"], _scenario, _nonnegative)
    identities = _object(row["identities"], IDENTITY_FIELDS)
    for value in identities.values():
        _string(value)
    if "reuse" in row:
        _reuse(row["reuse"])


def _unique_object(pairs) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate original publication JSON object key.")
        result[key] = value
    return result


def _no_float(_text):
    raise ValueError("Original publication JSON must be finite integer-valued data.")


def decode_published(raw: bytes, kind: str) -> Any:
    """Decode UTF8 with duplicate-key/float rejection, then validate every nested shape."""
    if not isinstance(raw, bytes):
        _fail()
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_float=_no_float,
            parse_constant=_no_float,
        )
    except (UnicodeError, RecursionError) as error:
        raise ValueError("Original publication UTF8 or nesting is invalid.") from error
    if kind == "history":
        _history(document)
    elif kind == "receipt":
        _receipt(document)
    else:
        raise ValueError("Unknown original publication kind.")
    return document
