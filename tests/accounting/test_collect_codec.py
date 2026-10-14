"""Plain-JSON codecs for the replay ledger: round trips, byte identity, strict decoding."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

import pytest

from ssgrep.accounting.contracts import canonical_bytes
from ssgrep.accounting.ledger_codec import (
    bundle_from_dict,
    bundle_to_dict,
    frozen_from_dict,
    frozen_to_dict,
    ledger_state_from_dict,
    ledger_state_to_dict,
    period_row_from_dict,
    period_row_to_dict,
    receipt_record_from_dict,
    receipt_record_to_dict,
)
from ssgrep.accounting.plain_codec import from_plain, loads, to_plain
from ssgrep.accounting.replay import Ledger
from ssgrep.accounting.valuation_inputs import EvidenceBundle
from tests.accounting.test_replay import corpus
from tests.accounting.valuation_builders import METHOD, atom, candidate


def through_json(raw: Any) -> Any:
    """What a file round trip does to a document (floats are rejected on the way in)."""
    return loads(json.dumps(raw, sort_keys=True))


@pytest.fixture
def ledger() -> Ledger:
    ledger = Ledger(METHOD)
    ledger.ingest(corpus())
    ledger.expire_roots(["root-1"])
    return ledger


# ---- acceptance 1: round trip equality and byte identity -------------------------------


def test_frozen_roots_round_trip_equal_with_identical_canonical_bytes(ledger: Ledger):
    frozen = ledger.state().frozen
    assert frozen and frozen[0].receipts and frozen[0].leaves
    for f in frozen:
        back = frozen_from_dict(through_json(frozen_to_dict(f)))
        assert back == f
        assert canonical_bytes(back) == canonical_bytes(f)


def test_restored_ledger_from_json_state_has_identical_rows_to_the_original(ledger: Ledger):
    state = ledger.state()
    assert state.frozen and state.candidates and state.debits  # mixed frozen + live evidence
    document = through_json(ledger_state_to_dict(state))
    restored_state = ledger_state_from_dict(document)
    assert restored_state == state
    restored = Ledger.restore(restored_state)
    assert restored.rows().canonical_bytes() == ledger.rows().canonical_bytes()
    assert restored.rows().digest == ledger.rows().digest
    assert restored.rows().receipts and restored.rows().periods


def test_restarted_ledger_keeps_ingesting_to_the_same_rows_as_a_full_run(ledger: Ledger):
    first = Ledger(METHOD)
    early = corpus().accepted_through("2026-11-05")
    first.ingest(early)
    first.expire_roots(["root-1"])
    resumed = Ledger.restore(
        ledger_state_from_dict(through_json(ledger_state_to_dict(first.state())))
    )
    late = EvidenceBundle(
        tuple(c for c in corpus().candidates if c not in early.candidates),
        tuple(d for d in corpus().debits if d not in early.debits),
    )
    resumed.ingest(late)
    continuous = Ledger(METHOD)
    continuous.ingest(early)
    continuous.expire_roots(["root-1"])
    continuous.ingest(late)
    assert resumed.rows().canonical_bytes() == continuous.rows().canonical_bytes()


def test_period_rows_round_trip(ledger: Ledger):
    periods = ledger.rows().periods
    assert any(not p.recomputable for p in periods) and any(p.late for p in periods)
    for row in periods:
        back = period_row_from_dict(through_json(period_row_to_dict(row)))
        assert back == row and back.digest == row.digest


# ---- acceptance 2: to_plain agrees with canonical_bytes ---------------------------------


def test_to_plain_is_what_canonical_bytes_hashes(ledger: Ledger):
    rows = ledger.rows()
    bundle = corpus()
    frozen = ledger.state().frozen[0]
    receipts = [r.with_content(None) for r in rows.receipts]
    for subject in (*rows.periods, *receipts, bundle, frozen):
        assert canonical_bytes(to_plain(subject)) == canonical_bytes(subject)
    assert json.dumps(to_plain(bundle), sort_keys=True) == json.dumps(
        json.loads(canonical_bytes(bundle)), sort_keys=True
    )


def test_to_plain_rejects_floats_anywhere():
    with pytest.raises(TypeError):
        to_plain({"a": [1, (2, 1.5)]})
    with pytest.raises(TypeError):
        to_plain(replace(candidate("f"), candidate_id=1.5))  # type: ignore[arg-type]


# ---- acceptance 3: strict decoding -----------------------------------------------------


def base_bundle() -> dict[str, Any]:
    cand = candidate("a", immediate=(atom("s1", 400),))
    return bundle_to_dict(EvidenceBundle((cand,), ()))


def edit(path: list[Any], value: Any) -> dict[str, Any]:
    doc = copy.deepcopy(base_bundle())
    node: Any = doc
    for step in path[:-1]:
        node = node[step]
    if value is DELETE:
        del node[path[-1]]
    else:
        node[path[-1]] = value
    return doc


DELETE = object()
CAND = ["candidates", 0]
BAD_DOCS = {
    "float_value": edit([*CAND, "immediate", 0, "atom", "chars"], 1.5),
    "str_for_int": edit([*CAND, "immediate", 0, "atom", "chars"], "400"),
    "bool_for_int": edit([*CAND, "immediate", 0, "atom", "chars"], True),
    "int_for_bool": edit([*CAND, "novel"], 1),
    "unknown_enum": edit([*CAND, "reuse"], "sorcery"),
    "enum_not_string": edit([*CAND, "reuse"], 3),
    "missing_required": edit([*CAND, "candidate_id"], DELETE),
    "missing_nested_required": edit([*CAND, "task", "root"], DELETE),
    "list_for_dict": edit([*CAND, "task"], ["root-1", "turn-1", "main"]),
    "dict_for_list": edit([*CAND, "immediate"], {"0": {}}),
    "null_for_str": edit([*CAND, "candidate_id"], None),
    "extra_field": edit([*CAND, "surprise"], 1),
    "object_not_candidate": edit(["candidates"], {"a": 1}),
}


@pytest.mark.parametrize("name", sorted(BAD_DOCS))
def test_malformed_documents_raise_value_error(name: str):
    with pytest.raises(ValueError):
        bundle_from_dict(BAD_DOCS[name])


def test_the_untouched_document_decodes_so_each_failure_is_caused_by_its_edit():
    cand = bundle_from_dict(base_bundle()).candidates[0]
    assert cand.candidate_id == "a" and cand.immediate[0].atom.chars == 400


def test_float_literal_in_json_text_is_a_value_error_not_a_silent_coercion():
    text = json.dumps(base_bundle()).replace('"chars": 400', '"chars": 1.5')
    assert '"chars": 1.5' in text
    with pytest.raises(ValueError):
        loads(text)
    with pytest.raises(ValueError):
        loads('{"x": NaN}')
    with pytest.raises(ValueError):
        from_plain(EvidenceBundle, json.loads(text))  # even a lenient parser cannot sneak it in


def test_any_passthrough_still_rejects_floats_and_non_json_values():
    with pytest.raises(ValueError):
        from_plain(dict[str, Any], {"a": {"b": [1.5]}})
    assert from_plain(dict[str, Any], {"a": {"b": [1, "x", None, True]}}) == {
        "a": {"b": [1, "x", None, True]}
    }


class Color(StrEnum):
    RED = "red"


@dataclass(frozen=True)
class Inner:
    n: int


@dataclass(frozen=True)
class Outer:
    pairs: tuple[tuple[str, int], ...]
    inner: Inner | None
    by_name: dict[str, Inner]
    color: Color = Color.RED
    tags: tuple[str, ...] = ()
    meta: dict[str, int] = field(default_factory=dict)


def test_declared_container_shapes_and_defaults_decode_exactly():
    raw = {
        "pairs": [["a", 1], ["b", 2]],
        "inner": {"n": 3},
        "by_name": {"x": {"n": 4}},
    }
    got = from_plain(Outer, raw)
    assert got == Outer((("a", 1), ("b", 2)), Inner(3), {"x": Inner(4)})  # defaults filled in
    assert to_plain(got) == {**raw, "color": "red", "tags": [], "meta": {}}
    assert from_plain(Outer, {**raw, "inner": None}).inner is None
    with pytest.raises(ValueError):
        from_plain(Outer, {**raw, "pairs": [["a", 1, 9]]})  # wrong tuple arity
    with pytest.raises(ValueError):
        from_plain(Outer, {**raw, "pairs": [["a", True]]})


NEWER = "field_from_a_newer_schema"


def test_newer_schema_documents_are_rejected_by_every_ledger_level_decoder(ledger: Ledger):
    state = ledger.state()
    rows = ledger.rows()
    cases = [
        (ledger_state_from_dict, ledger_state_to_dict(state)),
        (frozen_from_dict, frozen_to_dict(state.frozen[0])),
        (receipt_record_from_dict, receipt_record_to_dict(rows.receipts[0])),
        (period_row_from_dict, period_row_to_dict(rows.periods[0])),
        (bundle_from_dict, bundle_to_dict(corpus())),
    ]
    for decode, document in cases:
        assert decode(through_json(document)) is not None  # the control decodes
        with pytest.raises(ValueError):
            decode({**document, NEWER: 1})
    nested = ledger_state_to_dict(state)
    nested["frozen"][0]["receipts"][0][NEWER] = "x"
    with pytest.raises(ValueError):
        ledger_state_from_dict(nested)


def test_a_different_schema_number_is_rejected(ledger: Ledger):
    row = period_row_to_dict(ledger.rows().periods[0])
    with pytest.raises(ValueError):
        period_row_from_dict({**row, "schema": 2})
    receipt = receipt_record_to_dict(ledger.rows().receipts[0])
    receipt["receipt"] = {**receipt["receipt"], "schema": 2}
    with pytest.raises(ValueError):
        receipt_record_from_dict(receipt)


# ---- acceptance 4: private content and accepted_day ------------------------------------


CANARY = "CANARY-private-receipt-body-7d2f"


def test_receipt_content_is_never_serialized(ledger: Ledger):
    record = ledger.rows().receipts[0].with_content(CANARY)
    assert record.content == CANARY
    plain = receipt_record_to_dict(record)
    assert "content" not in plain
    assert CANARY not in json.dumps(plain)
    back = receipt_record_from_dict(through_json(plain))
    assert back.content is None and back == record.with_content(None)
    assert back.digest == record.digest  # content never part of identity


def test_content_never_reaches_frozen_or_state_documents(ledger: Ledger):
    frozen = ledger.state().frozen[0]
    tainted = replace(frozen, receipts=tuple(r.with_content(CANARY) for r in frozen.receipts))
    assert CANARY not in json.dumps(frozen_to_dict(tainted))
    state = replace(ledger.state(), frozen=(tainted,))
    assert CANARY not in json.dumps(ledger_state_to_dict(state))


def test_a_document_that_carries_content_is_rejected_on_read(ledger: Ledger):
    plain = receipt_record_to_dict(ledger.rows().receipts[0])
    with pytest.raises(ValueError):
        receipt_record_from_dict({**plain, "content": CANARY})


def test_bundle_round_trip_keeps_accepted_day_and_optional_fields():
    bundle = corpus()
    assert any(c.accepted_day for c in bundle.candidates)
    assert any(d.accepted_day for d in bundle.debits)
    back = bundle_from_dict(through_json(bundle_to_dict(bundle)))
    assert back == bundle
    assert [c.accepted for c in back.candidates] == [c.accepted for c in bundle.candidates]
    assert {c.accepted_day for c in back.candidates} >= {"2026-11-02", "2026-11-30"}
    assert any(d.partial for d in back.debits) and any(d.chars is None for d in back.debits)


# ---- acceptance 5: smoke guard (generous bound, not a benchmark) ------------------------


def test_two_thousand_candidate_bundle_round_trips_quickly():
    """Smoke guard against accidental quadratic behaviour; the bound is deliberately loose."""
    bundle = EvidenceBundle(
        tuple(
            candidate(f"c{i}", immediate=(atom(f"s{i}", 400 + i, origin=f"e{i}"),))
            for i in range(2000)
        )
    )
    started = time.monotonic()
    text = json.dumps(bundle_to_dict(bundle), sort_keys=True)
    back = bundle_from_dict(loads(text))
    elapsed = time.monotonic() - started
    assert back == bundle and len(back.candidates) == 2000
    assert elapsed < 5.0, f"2,000-candidate round trip took {elapsed:.2f}s"
