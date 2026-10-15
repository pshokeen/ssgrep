"""Plain-JSON codecs for the replay ledger: round trips, byte identity, strict decoding."""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

import pytest

from ssgrep.accounting import plain_codec
from ssgrep.accounting.collector_state import ReductionDoc
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


# ---- declared-schema decoding rejects noncanonical data without importing data-selected types


@dataclass
class Nested:
    rows: list[dict[str, int]]
    derived: int = field(default=17, init=False)


@dataclass
class Checked:
    count: int

    def __post_init__(self) -> None:
        if self.count < 0:
            raise TypeError("count must not be negative")


def test_lists_decode_nested_integer_fields_and_preserve_noninit_defaults() -> None:
    decoded = plain_codec.from_plain(Nested, {"rows": [{"count": 3}]})
    assert decoded == Nested([{"count": 3}]) and decoded.derived == 17
    with pytest.raises(ValueError, match=r"rows\.0\.count: expected int"):
        plain_codec.from_plain(Nested, {"rows": [{"count": True}]})
    with pytest.raises(ValueError, match="unknown field.*derived"):
        plain_codec.from_plain(Nested, {"rows": [], "derived": 99})
    assert plain_codec.to_plain(decoded) == {"rows": [{"count": 3}], "derived": 17}


@pytest.mark.parametrize("declared", [None, type(None)])
def test_null_schema_accepts_only_null(declared: Any) -> None:
    assert plain_codec.from_plain(declared, None) is None
    with pytest.raises(ValueError, match="expected null"):
        plain_codec.from_plain(declared, False)


def test_union_attempts_each_declared_alternative_without_integer_bool_coercion() -> None:
    declared = type(None) | int | str
    assert plain_codec.from_plain(declared, "seven") == "seven"
    assert plain_codec.from_plain(declared, 7) == 7
    assert plain_codec.from_plain(declared, None) is None
    with pytest.raises(ValueError, match="declared union members"):
        plain_codec.from_plain(declared, True)


@pytest.mark.parametrize("declared", [dict[str, int], Mapping[str, int]])
def test_mapping_schema_requires_objects_and_string_keys(declared: Any) -> None:
    assert plain_codec.from_plain(declared, {"answer": 42}) == {"answer": 42}
    with pytest.raises(ValueError, match="expected object"):
        plain_codec.from_plain(declared, [["answer", 42]])
    with pytest.raises(ValueError, match="string key"):
        plain_codec.from_plain(declared, {1: 42})


def test_list_schema_rejects_python_tuple_even_when_elements_fit() -> None:
    assert plain_codec.from_plain(list[int], [1, 2]) == [1, 2]
    with pytest.raises(ValueError, match="expected list"):
        plain_codec.from_plain(list[int], (1, 2))


@pytest.mark.parametrize("declared", [float, bytes, set[int]])
def test_noncanonical_declared_types_are_not_coerced(declared: Any) -> None:
    with pytest.raises(ValueError, match="floats are not allowed|unsupported declared type"):
        plain_codec.from_plain(declared, 1)


@pytest.mark.parametrize("declared", [Any, object])
@pytest.mark.parametrize("data", [{1: "value"}, {"nested": [b"binary"]}])
def test_untyped_fields_still_reject_non_json_data(declared: Any, data: Any) -> None:
    with pytest.raises(ValueError, match="string key|nested.0: expected plain JSON value"):
        plain_codec.from_plain(declared, data)
    assert plain_codec.from_plain(declared, {"nested": [None, True, 7, "value"]}) == {
        "nested": [None, True, 7, "value"]
    }


def test_constructor_validation_is_a_decode_error_with_its_original_cause() -> None:
    assert plain_codec.from_plain(Checked, {"count": 2}) == Checked(2)
    with pytest.raises(ValueError, match="Checked rejected its fields") as caught:
        plain_codec.from_plain(Checked, {"count": -1})
    assert isinstance(caught.value.__cause__, TypeError)
    assert str(caught.value.__cause__) == "count must not be negative"


@pytest.mark.parametrize("value", [b"bytes", {1, 2}, complex(1, 2)])
def test_encoder_refuses_values_without_a_canonical_json_representation(value: Any) -> None:
    with pytest.raises(TypeError, match="cannot encode"):
        plain_codec.to_plain({"payload": value})
    assert plain_codec.to_plain({"payload": (1, None, "text")}) == {"payload": [1, None, "text"]}


# ---- the stored reduction document: additive fields default empty on older blobs ----------

REDUCTION_CWD = "/work/proj"


def test_older_reduction_blob_without_the_new_fields_still_loads_empty() -> None:
    doc = ReductionDoc(
        coverage={"recorded_lookups": 2},
        session_roots=(("omp", "w", "omp:root"),),
        session_last=(("omp", "w", "2026-10-12T10:00:00.000Z"),),
    )
    older = doc.to_dict()
    del older["session_parents"], older["session_cwd"]  # what a pre-change collector wrote
    loaded = ReductionDoc.from_dict(older)
    assert loaded.session_parents == () and loaded.session_cwd == ()
    assert loaded.session_roots == doc.session_roots and loaded.coverage == doc.coverage


def test_new_fields_round_trip_exactly_including_a_null_parent() -> None:
    doc = ReductionDoc(
        session_parents=(("omp", "child", "root"), ("omp", "root", None)),
        session_cwd=(("omp", "child", REDUCTION_CWD),),
    )
    again = ReductionDoc.from_dict(json.loads(json.dumps(doc.to_dict())))
    assert again.session_parents == doc.session_parents
    assert again.session_cwd == doc.session_cwd


def test_older_blob_without_invocation_sessions_loads_empty_and_the_field_round_trips() -> None:
    older = ReductionDoc().to_dict()
    assert "invocation_sessions" in older
    del older["invocation_sessions"]  # what a pre-change collector wrote
    assert ReductionDoc.from_dict(older).invocation_sessions == ()
    doc = ReductionDoc(invocation_sessions=(("inv-a", "omp", "child-a"), ("inv-b", "claude", "s")))
    again = ReductionDoc.from_dict(json.loads(json.dumps(doc.to_dict())))
    assert again.invocation_sessions == doc.invocation_sessions
