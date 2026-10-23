"""Plain-JSON codecs for the replay ledger: round trips, byte identity, strict decoding."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

import pytest

from ssgrep.accounting.capture import fingerprint
from ssgrep.accounting.capture.journal_reader import read_journal
from ssgrep.accounting.collect import (
    collector,
    collector as collector_modes,
    collector as collector_progress,
    collector as maintenance,
    cycle as collector_cycle,
    cycle_publish as publish,
    read as collector_inputs,
    read as collector_live,
    read as collector_read,
    state as collector_state,
    store as store,
)
from ssgrep.accounting.collect.collector import MIB, CollectOptions, CycleBudget
from ssgrep.accounting.collect.read import (
    AdmitRules,
    ClassifyingLedger,
    ReadOutcome,
    baseline_record,
    discover,
    make_readers,
    read_sources,
)
from ssgrep.accounting.collect.replay import EvidenceReplay, Ledger, evidence_line, full_replay
from ssgrep.accounting.collect.state import ReconcileResult, ReductionDoc, State, VersionedLedger
from ssgrep.accounting.collect.steps import source_progress
from ssgrep.accounting.collect.store import (
    CONSUMED_KINDS,
    CollectError,
    SourceRecord,
    restore_cache,
    save_cache,
    source_key,
)
from ssgrep.accounting.ledger import (
    ledger_contracts as counters,
    ledger_contracts as counters_module,
)
from ssgrep.accounting.ledger.contracts import (
    RUNTIMES,
    SCHEMA_VERSION,
    Attribution,
    Basis,
    ExecutionEvent,
    Exposure,
    InvocationClass,
    Operation,
    Outcome,
    Phase,
    Quantity,
    RawEvidence,
    Reuse,
    Scenario,
    TaskId,
    Transport,
    canonical_bytes,
    digest_of,
)
from ssgrep.accounting.ledger.ledger_codec import (
    bundle_from_dict,
    bundle_to_dict,
    frozen_to_dict,
    ledger_state_from_dict,
    ledger_state_to_dict,
    period_row_from_dict,
    period_row_to_dict,
    receipt_record_from_dict,
    receipt_record_to_dict,
)
from ssgrep.accounting.ledger.ledger_contracts import (
    UNKNOWN_RUNTIME,
    BackfillSeries,
    ExecKey,
    LiveCounters,
    OpKey,
    backfill_series,
    bucket_index,
    empty_counters,
    observe,
    prune,
    start_epoch,
    view,
)
from ssgrep.accounting.ledger.reduce_records import (
    ExecutionRecord,
    InvocationRecord,
    summarize,
)
from ssgrep.accounting.ledger.revisions import (
    PeriodRow,
    current_rows,
    current_total,
    history,
    seal_day,
)
from ssgrep.accounting.ledger.valuation_inputs import (
    Candidate,
    Debit,
    Eligibility,
    EvidenceBundle,
)
from ssgrep.accounting.privacy import storage, storage as plain_codec
from ssgrep.accounting.privacy.policy import enable_collection
from ssgrep.accounting.privacy.storage import from_plain, loads, to_plain
from ssgrep.accounting.views import service
from tests.accounting.collect import collector_scene as scene, verify_scenes as vs
from tests.accounting.collect.test_replay import corpus
from tests.accounting.conftest import build_event, build_evidence
from tests.accounting.ledger.reduce_builders import (
    KEY,
    MCP,
    card,
    claude_prompt,
    claude_tool_result,
    claude_tool_use,
    results_json,
    source_session,
    ts,
)
from tests.accounting.ledger.valuation_builders import CHARS, DAY, METHOD, atom, candidate, debit
from tests.accounting.readers.readers_fixtures import (
    T0 as READERS_T0,
    append_bytes,
    jsonl,
    opencode_add_message,
    opencode_add_part,
    opencode_add_session,
    opencode_assistant_data,
    opencode_connect,
    opencode_tool_part,
    pi_header,
    pi_tool_call,
    pi_tool_result,
    write_records,
)
from tests.accounting.views.test_savings_commands import BINARY


def through_json(raw: Any) -> Any:
    """What a file round trip does to a document (floats are rejected on the way in)."""
    return loads(json.dumps(raw, sort_keys=True))


@pytest.fixture(name="ledger")
def codec_ledger() -> Ledger:
    ledger = Ledger(METHOD)
    ledger.ingest(corpus())
    ledger.expire_roots(["root-1"])
    return ledger


# ---- acceptance 1: round trip equality and byte identity -------------------------------


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


@pytest.mark.parametrize("version", ("honest_bounds_ns_v3", "honest_bounds_ns_v4"))
def test_known_ns_method_state_keeps_its_exact_header_and_hash(version: str) -> None:
    from ssgrep.accounting.collect.replay import LedgerState
    from ssgrep.accounting.ledger.contracts import digest_of

    method = replace(METHOD, valuation_version=version)
    document = ledger_state_to_dict(LedgerState(method))
    original = canonical_bytes(document)
    restored = ledger_state_from_dict(through_json(document))
    assert restored.methodology.valuation_version == version
    assert restored.methodology.hash == digest_of(document["methodology"])[:32]
    assert canonical_bytes(ledger_state_to_dict(restored)) == original


@pytest.mark.parametrize("version", ("honest_bounds_v1", "honest_bounds_ns_v2", "future_v5"))
def test_unknown_or_pre_ns_method_state_is_not_relabelled_as_a_known_namespace(
    version: str,
) -> None:
    from ssgrep.accounting.collect.replay import LedgerState

    document = ledger_state_to_dict(LedgerState(replace(METHOD, valuation_version=version)))
    original = canonical_bytes(document)
    with pytest.raises(ValueError, match="time_precision_namespace_migration_required"):
        ledger_state_from_dict(through_json(document))
    assert canonical_bytes(document) == original


@pytest.fixture
def committed_json_state(tmp_path: Path) -> Path:
    vs.enable(pin=True)
    vs.start_capture()
    vs.pair(vs.projects(tmp_path), 0, scene.at(hour=9))
    assert scene.report(scene.collect(scene.at(hour=20))).generation == 1
    assert scene.counter_total(scene.current()) == 2
    checkpoint = scene.committed()
    assert checkpoint.collection_scope and checkpoint.methodology
    assert store.load_checkpoint(PROFILE) == checkpoint
    return scene.accounting_dir() / "collector"


@pytest.mark.parametrize("field", ["generation", "methodology", "scope_digest"])
@pytest.mark.parametrize("conflict_first", [True, False])
def test_duplicate_checkpoint_authority_fails_before_publication(
    committed_json_state: Path, field: str, conflict_first: bool
) -> None:
    path = committed_json_state / store.CHECKPOINT
    original = path.read_bytes()
    checkpoint = scene.committed()
    if field == "scope_digest":
        key = "digest"
        value = checkpoint.collection_scope[key]
        conflict = "0" * 64 if value != "0" * 64 else "1" * 64
    else:
        key = field
        value = getattr(checkpoint, field)
        conflict = checkpoint.generation + 10 if field == "generation" else "foreign-method"
    member = store.dumps(key) + b":" + store.dumps(value)
    other = store.dumps(key) + b":" + store.dumps(conflict)
    assert original.count(member) == 1
    duplicate = other + b"," + member if conflict_first else member + b"," + other
    path.write_bytes(original.replace(member, duplicate, 1))
    corrupted = path.read_bytes()
    before = scene.current()
    before_counters = scene.counter_total(before)
    state_files = scene.file_digests(committed_json_state / store.STATE_DIR)

    with pytest.raises(CollectError) as failed:
        scene.collect(scene.at(hour=20))

    assert failed.value.condition == "state_corrupt"
    assert path.read_bytes() == corrupted
    assert scene.current() == before
    assert scene.counter_total(scene.current()) == before_counters
    assert scene.file_digests(committed_json_state / store.STATE_DIR) == state_files


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("conflict_first", [True, False])
def test_duplicate_referenced_blob_fails_before_publication(
    committed_json_state: Path, nested: bool, conflict_first: bool
) -> None:
    checkpoint = scene.committed()
    name = collector_state.ledger_blob_name(checkpoint.methodology)
    filename = checkpoint.blobs[name]
    path = committed_json_state / store.STATE_DIR / filename
    original = path.read_bytes()
    document = store.read_blob(PROFILE, filename)
    assert isinstance(document, dict)
    assert document["collection_scope"] == checkpoint.collection_scope
    if nested:
        key = "digest"
        value = checkpoint.collection_scope[key]
        conflict = "0" * 64 if value != "0" * 64 else "1" * 64
    else:
        key = "collection_scope"
        value = checkpoint.collection_scope
        conflict = {**checkpoint.collection_scope, "digest": "0" * 64}
    member = store.dumps(key) + b":" + store.dumps(value)
    other = store.dumps(key) + b":" + store.dumps(conflict)
    assert original.count(member) == 1
    duplicate = other + b"," + member if conflict_first else member + b"," + other
    path.write_bytes(original.replace(member, duplicate, 1))
    corrupted = path.read_bytes()
    before = scene.current()
    checkpoint_bytes = (committed_json_state / store.CHECKPOINT).read_bytes()
    state_files = scene.file_digests(committed_json_state / store.STATE_DIR)

    with pytest.raises(CollectError) as failed:
        scene.collect(scene.at(hour=20))

    assert failed.value.condition == "state_corrupt"
    assert path.read_bytes() == corrupted
    assert (committed_json_state / store.CHECKPOINT).read_bytes() == checkpoint_bytes
    assert scene.current() == before
    assert scene.counter_total(scene.current()) == 2
    assert scene.file_digests(committed_json_state / store.STATE_DIR) == state_files


# Live captured counters: once-only counting, labels frozen at capture, epochs, backfill apart.


COUNTERS_T0 = "2026-11-01T00:00:00Z"
SEARCH_CLI = OpKey("search", "cli")


def completed(n: int, **overrides: object) -> ExecutionEvent:
    return build_event(execution_id=f"e{n:04d}", phase=Phase.COMPLETED, **overrides)


def rendered(n: int, value: int | None = 100, **overrides: object) -> ExecutionEvent:
    quantity = (
        Quantity.observed(value, "bytes", "mcp_json")
        if value is not None
        else Quantity.unavailable("bytes", "not_recorded")
    )
    return build_event(
        execution_id=f"e{n:04d}",
        phase=Phase.RENDERED,
        outcome=None,
        mono_ns=None,
        rendered_bytes=quantity,
        **overrides,
    )


def started(n: int) -> ExecutionEvent:
    return build_event(
        execution_id=f"e{n:04d}",
        phase=Phase.STARTED,
        outcome=None,
        mono_ns=None,
        rendered_bytes=None,
    )


def mixed_events() -> list[ExecutionEvent]:
    events: list[ExecutionEvent] = []
    for n in range(40):
        operation = (Operation.SEARCH, Operation.SHOW, Operation.NOTE)[n % 3]
        transport = (Transport.CLI, Transport.MCP)[n % 2]
        outcome = (Outcome.OK, Outcome.EMPTY, Outcome.ERROR, None)[n % 4]
        mono = None if n % 7 == 0 else (n * 997_000) % 40_000_000
        events.append(started(n))
        events.append(
            completed(n, operation=operation, transport=transport, outcome=outcome, mono_ns=mono)
        )
        events.append(
            rendered(n, None if n % 5 == 0 else n * 10, operation=operation, transport=transport)
        )
    return events


def fresh() -> LiveCounters:
    return empty_counters(1, COUNTERS_T0)


def total(counters: LiveCounters) -> int:
    return sum(count for _, count in counters.executions)


def test_property_any_order_batches_and_repeats_equal_one_pass() -> None:
    events = mixed_events()
    reference = observe(fresh(), events, horizon_day=None)
    assert total(reference) == 40  # only completed events count; 80 other events added nothing
    for seed in range(25):
        rng = random.Random(seed)
        shuffled = events + rng.sample(events, 30)  # duplicates inside the stream
        rng.shuffle(shuffled)
        counters = fresh()
        cut = sorted(rng.sample(range(len(shuffled)), 4))
        batches = [shuffled[a:b] for a, b in zip([0, *cut], [*cut, len(shuffled)], strict=True)]
        previous_total = 0
        for batch in batches:
            counters = observe(counters, batch, horizon_day=None)
            counters = observe(counters, batch, horizon_day=None)  # repeated batch
            assert total(counters) >= previous_total  # never decreases
            previous_total = total(counters)
        assert counters == reference
        assert canonical_bytes(counters.to_dict()) == canonical_bytes(reference.to_dict())


def test_only_completed_phase_increments_executions_and_labels_come_from_capture() -> None:
    counters = observe(
        fresh(),
        [
            started(1),
            rendered(1, 50),
            completed(1, outcome=Outcome.EMPTY, transport=Transport.MCP, operation=Operation.SHOW),
            completed(2, outcome=None),
        ],
        horizon_day=None,
    )
    assert counters.executions == (
        (ExecKey("search", "cli", "unknown", UNKNOWN_RUNTIME), 1),
        (ExecKey("show", "mcp", "empty", UNKNOWN_RUNTIME), 1),
    )
    only_started = observe(fresh(), [started(1)], horizon_day=None)
    assert only_started.executions == () and only_started.counted == ()


def test_runtime_label_never_moves_when_an_attribution_appears_elsewhere() -> None:
    event = completed(1)
    counters = observe(fresh(), [event], horizon_day=None)
    persisted = canonical_bytes(counters.to_dict())
    # Later the reducer attributes the very same execution to a runtime (ledger/gauge side).
    record = ExecutionRecord(
        execution_id=event.execution_id, ts_utc=event.ts_utc, day=event.ts_utc[:10],
        operation="search", transport="cli", outcome=Outcome.OK, outcome_basis="journal",
        attribution=Attribution.EXPLICIT, source="both", runtime="claude", session_id="s1",
        invocation_id="inv1", actor="agent", task=None,
        rendered_bytes=Quantity.observed(812, "bytes", "mcp_json"),
        visible_bytes=Quantity.observed(812, "bytes", "mcp_json"),
        exposure=Exposure.OBSERVED, reuse=Reuse.NONE_OBSERVED, timing_known=True, in_scope=True,
    )  # fmt: skip
    gauges = summarize([record])
    assert gauges["runtime.claude"].value == 1 and "runtime.unknown" not in gauges
    again = observe(counters, [event, rendered(1)], horizon_day=None)
    assert [key.runtime for key, _ in again.executions] == [UNKNOWN_RUNTIME]
    assert [count for _, count in again.executions] == [1]
    assert canonical_bytes(observe(counters, [event], horizon_day=None).to_dict()) == persisted
    assert json.loads(persisted)["executions"] == [
        {
            "operation": "search",
            "transport": "cli",
            "outcome": "ok",
            "runtime": "unknown",
            "count": 1,
        }
    ]


@pytest.mark.parametrize(
    ("duration_us", "index"),
    [
        (0, 0),
        (999, 0),
        (1_000, 0),
        (1_001, 1),
        (5_000, 1),
        (5_001, 2),
        (500_000, 5),
        (500_001, 6),
        (5_000_000, 7),
        (5_000_001, 8),
        (30_000_000, 8),
        (30_000_001, 9),
        (900_000_000, 9),
    ],
)
def test_bucket_boundaries_are_inclusive(duration_us: int, index: int) -> None:
    assert bucket_index(duration_us) == index
    counters = observe(fresh(), [completed(1, mono_ns=duration_us * 1000)], horizon_day=None)
    histogram = dict(counters.durations)[SEARCH_CLI]
    assert histogram.buckets[index] == 1 and sum(histogram.buckets) == 1
    assert histogram.sum_us == duration_us


def test_histogram_count_sum_exact_and_unknown_duration_kept_apart() -> None:
    durations_us = [1_000, 1_001, 7, 30_000_001, 123_456]
    events = [
        completed(i, mono_ns=us * 1000 + 999) for i, us in enumerate(durations_us)
    ]  # ns remainder floors
    events.append(completed(90, mono_ns=None))
    counters = observe(fresh(), events, horizon_day=None)
    histogram = dict(counters.durations)[SEARCH_CLI]
    assert histogram.count == sum(histogram.buckets) == 5
    assert histogram.sum_us == sum(durations_us)
    assert histogram.buckets[0] == 2 and histogram.buckets[9] == 1
    assert dict(counters.duration_unknown) == {SEARCH_CLI: 1}
    assert total(counters) == 6  # the unknown-duration execution is still an execution
    only_unknown = observe(fresh(), [completed(1, mono_ns=None)], horizon_day=None)
    assert only_unknown.durations == () and dict(only_unknown.duration_unknown) == {SEARCH_CLI: 1}


def test_rendered_known_adds_bytes_unknown_bumps_marker_never_zero() -> None:
    events = [
        rendered(1, 100),
        rendered(2, 250),
        rendered(3, None),
        rendered(4, None, operation=Operation.SHOW),
        build_event(execution_id="e5", phase=Phase.RENDERED, rendered_bytes=None),
    ]
    counters = observe(fresh(), events, horizon_day=None)
    assert dict(counters.rendered_bytes) == {SEARCH_CLI: 350}
    assert dict(counters.rendered_unknown) == {SEARCH_CLI: 2, OpKey("show", "cli"): 1}


def test_each_execution_phase_counts_once_even_with_conflicting_repeat() -> None:
    first = completed(1, mono_ns=2_000_000)
    counters = observe(fresh(), [first], horizon_day=None)
    redelivered = replace(first, mono_ns=9_000_000_000, outcome=Outcome.ERROR)
    assert observe(counters, [redelivered], horizon_day=None) == counters
    assert ("e0001|completed", "2026-11-01") in counters.counted


def test_roundtrip_is_byte_identical_and_restart_counts_nothing_twice() -> None:
    events = mixed_events()
    counters = observe(fresh(), events, horizon_day="2026-10-01")
    data = counters.to_dict()
    restored = LiveCounters.from_dict(json.loads(canonical_bytes(data)))
    assert restored == counters
    assert canonical_bytes(restored.to_dict()) == canonical_bytes(data)
    after_restart = observe(restored, events, horizon_day="2026-10-01")
    assert after_restart == counters and after_restart.epoch == 1
    extra = completed(500)
    grown = observe(restored, [*events, extra], horizon_day=None)
    assert total(grown) == total(counters) + 1


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(epoch=1.0),
        lambda d: d.update(epoch=0),
        lambda d: d.update(skipped_expired=-1),
        lambda d: d.update(skipped_expired=True),
        lambda d: d.update(epoch_started_at="yesterday"),
        lambda d: d.update(extra=1),
        lambda d: d["executions"][0].update(count=1.5),
        lambda d: d["executions"][0].pop("runtime"),
        lambda d: d["durations"][0]["histogram"].update(count=99),
        lambda d: d["durations"][0]["histogram"]["buckets"].pop(),
        lambda d: d["counted"].append(["no-phase-separator", "2026-11-01"]),
        lambda d: d["counted"].append(["e1|bogus", "2026-11-01"]),
        lambda d: d["counted"].append(["e1|completed", "11/01/2026"]),
        lambda d: d["executions"].append(dict(d["executions"][0])),
        lambda d: d["rendered_bytes"].append(dict(d["rendered_bytes"][0])),
        lambda d: d["durations"].append(dict(d["durations"][0])),
        lambda d: d["counted"].append(list(d["counted"][0])),
        lambda d: d["counted"].append(["e1|completed"]),
        lambda d: d["executions"][0].update(operation=""),
        lambda d: d.update(counted={}),
    ],
)
def test_from_dict_rejects_garbage(mutate) -> None:
    data = json.loads(canonical_bytes(observe(fresh(), mixed_events(), horizon_day=None).to_dict()))
    mutate(data)
    with pytest.raises(ValueError):
        LiveCounters.from_dict(data)


def test_from_dict_rejects_float_literals_in_json_text() -> None:
    text = canonical_bytes(observe(fresh(), [completed(1)], horizon_day=None).to_dict()).decode()
    assert '"count":1,' in text
    assert LiveCounters.from_dict(json.loads(text)).executions  # the untouched text is valid
    for literal in ("1.5", "1.0", "1e0"):
        with pytest.raises(ValueError):
            LiveCounters.from_dict(json.loads(text.replace('"count":1,', f'"count":{literal},', 1)))
    with pytest.raises(ValueError):
        LiveCounters.from_dict("not an object")


def test_start_epoch_zeroes_series_and_known_events_never_count_again() -> None:
    old = mixed_events()
    previous = observe(fresh(), old, horizon_day=None)
    now = datetime(2026, 11, 5, 12, 30, 15, tzinfo=UTC)
    epoch2 = start_epoch(previous, old, now)
    assert epoch2.epoch == 2 and epoch2.epoch_started_at == "2026-11-05T12:30:15Z"
    assert (epoch2.executions, epoch2.rendered_bytes, epoch2.rendered_unknown) == ((), (), ())
    assert (epoch2.durations, epoch2.duration_unknown, epoch2.skipped_expired) == ((), (), 0)
    assert set(previous.counted) <= set(epoch2.counted)
    replayed = observe(epoch2, old, horizon_day=None)
    assert replayed == epoch2  # captured before the call: never counts in the new epoch
    fresh_event = completed(777, mono_ns=3_000_000)
    after = observe(replayed, [*old, fresh_event], horizon_day=None)
    assert dict(after.executions) == {ExecKey("search", "cli", "ok", "unknown"): 1}
    assert dict(after.durations)[SEARCH_CLI].count == 1
    assert previous.epoch == 1 and total(previous) == 40  # previous value untouched (immutable)


def test_start_epoch_counts_events_known_only_through_the_argument_and_first_epoch() -> None:
    known_only = [completed(1), started(2)]
    epoch2 = start_epoch(fresh(), known_only, datetime(2026, 11, 5, tzinfo=UTC))
    assert {key for key, _ in epoch2.counted} == {"e0001|completed", "e0002|started"}
    first = start_epoch(None, known_only, datetime(2026, 11, 5, tzinfo=UTC))
    assert first.epoch == 1 and first.executions == ()
    assert observe(first, known_only, horizon_day=None).executions == ()
    with pytest.raises(ValueError):
        start_epoch(None, [], datetime(2026, 11, 5))  # naive clock value is ambiguous


def test_horizon_skips_old_events_and_counts_them_as_expired() -> None:
    old = completed(1, ts_utc="2026-09-30T23:59:59Z")
    edge = completed(2, ts_utc="2026-10-01T00:00:00Z")
    new = completed(3, ts_utc="2026-11-01T01:00:00Z")
    counters = observe(fresh(), [old, old, edge, new], horizon_day="2026-10-01")
    assert (
        total(counters) == 2 and counters.skipped_expired == 1
    )  # duplicates in one pass skipped once
    assert {key for key, _ in counters.counted} == {"e0002|completed", "e0003|completed"}
    # Expired events are not remembered: with a looser horizon they still count (once).
    loosened = observe(counters, [old], horizon_day=None)
    assert total(loosened) == 3


def test_prune_drops_only_old_counted_keys_and_keeps_totals() -> None:
    events = [
        completed(1, ts_utc="2026-09-01T00:00:00Z"),
        completed(2, ts_utc="2026-10-15T00:00:00Z"),
        completed(3, ts_utc="2026-11-01T00:00:00Z"),
    ]
    counters = observe(fresh(), events, horizon_day=None)
    pruned = prune(counters, "2026-10-15")
    assert [key for key, _ in pruned.counted] == ["e0002|completed", "e0003|completed"]
    assert (
        replace(pruned, counted=counters.counted) == counters
    )  # everything but `counted` unchanged
    assert prune(pruned, "2026-10-15") == pruned
    assert prune(counters, "2026-01-01") == counters
    # Pruned keys behind the horizon cannot recount when the same horizon guards observe.
    again = observe(pruned, events, horizon_day="2026-10-15")
    assert total(again) == 3 and again.skipped_expired == 1


def test_view_is_integer_only_sorted_with_cumulative_le_buckets() -> None:
    events = [
        completed(1, mono_ns=1_000_000),
        completed(2, mono_ns=1_001_000),
        completed(3, mono_ns=40_000_000_000),
        completed(4, operation=Operation.SHOW, transport=Transport.MCP, mono_ns=None),
        rendered(1, 70),
        rendered(2, None, operation=Operation.NOTE),
    ]
    counters = observe(fresh(), events, horizon_day=None)
    document = view(counters)
    assert json.loads(canonical_bytes(document)) == document
    assert document["epoch"] == 1 and document["epoch_started_at"] == COUNTERS_T0
    assert [(r["operation"], r["transport"], r["count"]) for r in document["executions"]] == [
        ("search", "cli", 3),
        ("show", "mcp", 1),
    ]
    assert document["rendered_bytes"] == [{"operation": "search", "transport": "cli", "bytes": 70}]
    assert document["rendered_unknown"] == [{"operation": "note", "transport": "cli", "count": 1}]
    assert document["duration_unknown"] == [{"operation": "show", "transport": "mcp", "count": 1}]
    (series,) = document["duration"]
    assert series["count"] == 3 and series["sum_us"] == 1_000 + 1_001 + 40_000_000
    assert [(b["le_us"], b["count"]) for b in series["buckets"]] == [
        ("1000", 1),
        ("5000", 2),
        ("10000", 2),
        ("50000", 2),
        ("100000", 2),
        ("500000", 2),
        ("1000000", 2),
        ("5000000", 2),
        ("30000000", 2),
        ("+Inf", 3),
    ]
    assert view(observe(fresh(), list(reversed(events)), horizon_day=None)) == document


# --- backfill -----------------------------------------------------------------------------


def invocation(
    n: int,
    ts: str | None,
    operations: tuple[str, ...] = ("search",),
    cls=InvocationClass.TRANSCRIPT_ONLY,
    cardinality: str = "single_call",
) -> InvocationRecord:
    return InvocationRecord(
        invocation_id=f"inv{n}",
        runtime="claude",
        session_id="s",
        call_id=f"c{n}",
        args_hash="h",
        ts_utc=ts,
        operations=operations,
        route="mcp",
        cardinality=cardinality,
        cls=cls,
        attribution=Attribution.UNATTRIBUTED,
        execution_ids=(),
        ambiguous_execution_ids=(),
        outcome=Outcome.UNKNOWN,
        exposure=Exposure.UNAVAILABLE,
        visible_bytes=Quantity(None, "bytes", Basis.UNAVAILABLE, reason="not_recorded"),
        task=None,
        actor="agent",
    )


def test_backfill_counts_half_open_interval_keyed_by_operation_and_class() -> None:
    since, until = "2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z"
    records = [
        invocation(1, since),  # inclusive start
        invocation(2, "2026-01-31T23:59:59Z", ("show",), InvocationClass.MATCHED),
        invocation(3, until),  # exclusive end
        invocation(4, "2025-12-31T23:59:59Z"),
        invocation(5, None),  # cannot be placed in the interval
        invocation(
            6,
            "2026-01-15T00:00:00Z",
            ("search", "show"),
            InvocationClass.TRANSCRIPT_ONLY,
            "unknown",
        ),
        invocation(7, "2026-01-16T00:00:00Z", (), InvocationClass.MENTION),
        invocation(8, "2026-01-17T00:00:00Z", ("search",), InvocationClass.JOURNAL_GAP),
        invocation(1, since),  # same invocation seen twice
    ]
    series = backfill_series(records, since, until, "2026-11-02T00:00:00Z")
    assert dict(series.counts) == {
        "search:transcript_only": 2,
        "show:matched": 1,
        "show:transcript_only": 1,
        "search:journal_gap": 1,
    }
    assert series.counts == tuple(sorted(series.counts))
    assert series.unknown_cardinality == 1
    assert (series.since, series.until, series.completed_at) == (
        since,
        until,
        "2026-11-02T00:00:00Z",
    )
    assert backfill_series(list(reversed(records)), since, until, "2026-11-02T00:00:00Z") == series
    restored = BackfillSeries.from_dict(json.loads(canonical_bytes(series.to_dict())))
    assert restored == series and canonical_bytes(restored.to_dict()) == canonical_bytes(
        series.to_dict()
    )
    with pytest.raises(ValueError):
        BackfillSeries.from_dict(
            json.loads(canonical_bytes(series.to_dict()).replace(b":2", b":2.5", 1))
        )
    with pytest.raises(ValueError):
        backfill_series(records, until, since, "2026-11-02T00:00:00Z")
    with pytest.raises(ValueError):
        LiveCounters.from_dict(series.to_dict())  # a backfill series is never a live counter


def test_backfill_does_not_touch_live_counters_or_create_a_burst() -> None:
    live = observe(fresh(), [completed(1), completed(2)], horizon_day=None)
    before = canonical_bytes(live.to_dict())
    view_before = canonical_bytes(view(live))
    start = datetime(2026, 1, 1, tzinfo=UTC)
    records = [
        invocation(n, (start + timedelta(seconds=n * 60)).strftime("%Y-%m-%dT%H:%M:%SZ"))
        for n in range(10_000)
    ]
    series = backfill_series(
        records, "2026-01-01T00:00:00Z", "2026-03-01T00:00:00Z", "2026-11-02T00:00:00Z"
    )
    assert dict(series.counts) == {"search:transcript_only": 10_000}
    assert not isinstance(series, LiveCounters) and not isinstance(live, BackfillSeries)
    assert canonical_bytes(live.to_dict()) == before and canonical_bytes(view(live)) == view_before
    assert total(live) == 2  # 10,000 imported invocations produced no live executions
    assert total(observe(live, [], horizon_day=None)) == 2


def test_counter_epoch_zero_is_rejected_before_any_event_can_be_counted() -> None:
    with pytest.raises(ValueError):
        counters_module.empty_counters(0, "2026-10-12T00:00:00Z")
    positive = observe(
        counters_module.empty_counters(1, "2026-10-12T00:00:00Z"), [completed(1)], horizon_day=None
    )
    assert sum(count for _, count in positive.executions) == 1


@pytest.mark.parametrize("field", ["kind", "counts"])
def test_backfill_decoder_does_not_accept_a_live_series_or_unkeyed_counts(field: str) -> None:
    value = backfill_series(
        [invocation(1, "2026-10-12T12:00:00Z")],
        "2026-10-12T00:00:00Z",
        "2026-10-13T00:00:00Z",
        "2026-10-14T00:00:00Z",
    )
    raw = value.to_dict()
    assert BackfillSeries.from_dict(raw) == value
    raw[field] = "live" if field == "kind" else []
    with pytest.raises(ValueError):
        BackfillSeries.from_dict(raw)


# ---- live counters through the real collector ------------------------------------------------
# Live counters keep their capture-time ``unknown`` runtime label after attribution, never
# decrease across cycles/restarts, keep the epoch across an ordinary restart, and reset only on a
# declared ``--new-epoch`` (future events only, ledger history kept). Backfill is a separate
# series. Histograms use the fixed ten buckets with count+sum in integer microseconds.

CLOCK = scene.at(hour=9)
COUNTERS_DAY = 12
MARK = 1  # the capture-start marker is itself one real (cli search) execution on day 11, 08:00


def later(days: int, hour: int = 12) -> Any:
    return scene.at(COUNTERS_DAY, hour) + timedelta(days=days)


def observation(envelope: Any, window: str, name: str) -> int | None:
    return scene.window(envelope, window)["observations"][name]["value"]


def live_counters(envelope: Any) -> dict[str, Any]:
    return scene.observations(envelope)["counters"]


def total_executions(envelope: Any) -> int:
    return sum(row["count"] for row in live_counters(envelope)["executions"])


def test_the_runtime_label_stays_unknown_after_the_execution_is_later_attributed(
    tmp_path: Path,
) -> None:
    vs.enable(pin=True)
    vs.start_capture()
    # capture-time: the journal records the execution, the transcript does not exist yet
    scene.journal_events(
        scene.execution("n-0", ts(COUNTERS_DAY, 9, 0, 11), text=vs.query(0), transport="mcp")
    )
    scene.collect(scene.at(hour=10))
    early = scene.current()
    mcp = [r for r in live_counters(early)["executions"] if r["transport"] == "mcp"]
    assert mcp == [
        {
            "operation": "search",
            "transport": "mcp",
            "outcome": "ok",
            "runtime": "unknown",
            "count": 1,
        }
    ]
    assert observation(early, "lifetime", "runtime.unknown") == 1 + MARK
    assert (
        "runtime.claude" not in scene.window(early, "lifetime")["observations"]
    )  # nothing attributed

    # reconciliation later attributes the execution to the claude transcript that called it
    vs.pair(vs.projects(tmp_path), 0, CLOCK)
    scene.collect(scene.at(hour=11))
    attributed = scene.current()
    assert observation(attributed, "lifetime", "runtime.claude") == 1  # the attributed view moved
    assert observation(attributed, "lifetime", "runtime.unknown") == MARK  # only the marker is left
    rows = live_counters(attributed)["executions"]
    assert [r for r in rows if r["transport"] == "mcp"] == mcp  # the frozen label did not move
    assert sum(r["count"] for r in rows) == 1 + MARK and all(
        r["runtime"] == "unknown" for r in rows
    )


def test_counters_never_decrease_across_cycles_restarts_and_expiry(tmp_path: Path) -> None:
    vs.enable(pin=True)
    vs.start_capture()
    totals: list[int] = []
    hist_counts: list[int] = []
    for n in range(4):
        vs.pair(vs.projects(tmp_path), n, CLOCK)
        scene.collect(scene.at(hour=10 + n))
        envelope = scene.current()
        totals.append(total_executions(envelope))
        hist_counts.append(sum(h["count"] for h in live_counters(envelope)["duration"]))
    scene.collect(scene.at(hour=20))  # nothing new at all
    totals.append(total_executions(scene.current()))
    scene.collect(later(45))  # clock jumps far ahead: windows and retention horizon move
    totals.append(total_executions(scene.current()))
    assert totals == [n + MARK for n in (1, 2, 3, 4, 4, 4)]
    assert hist_counts == sorted(hist_counts) and hist_counts[-1] == 4 + MARK


def test_an_ordinary_restart_preserves_the_epoch_and_the_counters(tmp_path: Path) -> None:
    vs.enable(pin=True)
    vs.start_capture()
    vs.pair(vs.projects(tmp_path), 0, CLOCK)
    scene.collect(scene.at(hour=10))
    first = scene.current()
    vs.pair(vs.projects(tmp_path), 1, CLOCK)
    result = scene.collect(
        scene.at(hour=11)
    )  # a fresh process call: a restart as far as state goes
    second = scene.current()
    assert result.epoch == 1 and second.counter_epoch == first.counter_epoch == 1
    assert live_counters(second)["epoch_started_at"] == live_counters(first)["epoch_started_at"]
    assert total_executions(second) == 2 + MARK and total_executions(first) == 1 + MARK


def test_new_epoch_zeroes_future_live_counters_and_keeps_every_ledger_total(
    tmp_path: Path,
) -> None:
    vs.enable(pin=True)
    vs.start_capture()
    for n in (0, 1):
        vs.pair(vs.projects(tmp_path), n, CLOCK)
    scene.collect(scene.at(hour=10))
    before = scene.current()
    ledger_before = vs.ledger_digest()
    assert total_executions(before) == 2 + MARK and before.counter_epoch == 1

    declared = scene.collect(scene.at(hour=11), CollectOptions(new_epoch=True))
    after = scene.current()
    assert declared.epoch == 2 and after.counter_epoch == 2
    assert live_counters(after)["epoch"] == 2
    assert live_counters(after)["executions"] == []  # live counters start at zero
    assert live_counters(after)["duration"] == [] and live_counters(after)["rendered_bytes"] == []
    assert live_counters(after)["epoch_started_at"] == "2026-10-12T11:00:00Z"
    # history, the ledger and the lifetime window are untouched by the declaration
    assert vs.ledger_digest() == ledger_before
    assert observation(after, "lifetime", "searches") == observation(before, "lifetime", "searches")
    assert observation(after, "lifetime", "searches") == 2 + MARK

    # the events captured before the declaration never count in the new epoch ...
    scene.collect(scene.at(hour=12))
    assert live_counters(scene.current())["executions"] == []
    # ... only events captured AFTER it do, from zero
    vs.pair(vs.projects(tmp_path), 2, CLOCK)
    scene.journal_events(
        scene.execution(
            "n-later", ts(COUNTERS_DAY, 13, 0, 0), text="after the reset", transport="cli"
        )
    )
    scene.collect(scene.at(hour=14))
    final = scene.current()
    assert final.counter_epoch == 2
    assert total_executions(final) == 2  # n-2 (09+2 h) and n-late, counted in epoch 2 once
    assert observation(final, "lifetime", "searches") == 4 + MARK  # ledger history kept: all


def test_a_new_epoch_is_not_inherited_by_the_next_ordinary_restart(tmp_path: Path) -> None:
    vs.enable(pin=True)
    vs.start_capture()
    vs.pair(vs.projects(tmp_path), 0, CLOCK)
    scene.collect(scene.at(hour=10))
    scene.collect(scene.at(hour=11), CollectOptions(new_epoch=True))
    scene.collect(scene.at(hour=12))
    scene.collect(scene.at(hour=13))
    assert scene.current().counter_epoch == 2  # not 3, not back to 1


def test_backfill_is_a_separate_series_and_never_a_burst_of_live_executions(
    tmp_path: Path,
) -> None:
    vs.enable(pin=True)
    vs.start_capture()
    vs.pair(vs.projects(tmp_path), 0, CLOCK)
    scene.collect(scene.at(hour=10))
    live = scene.current()
    # an old transcript the live reader pre-baselined as history
    old = vs.put(vs.projects(tmp_path), "old", [], scene.at(COUNTERS_DAY - 10))
    old.write_text("")  # placeholder replaced below by real old records
    records = [
        claude_prompt("old", "old work", uuid="uo", ts=ts(2, 9)),
        claude_tool_use("old", "o1", MCP, {"query": "old q"}, ts=ts(2, 9, 0, 5), message_id="mo"),
        claude_tool_result("old", "o1", results_json(), ts=ts(2, 9, 0, 8)),
    ]
    vs.put(vs.projects(tmp_path), "old", records, scene.at(COUNTERS_DAY - 10))
    backfill = CollectOptions(
        backfill=True, since="2026-10-01T00:00:00Z", until="2026-10-05T00:00:00Z"
    )
    result = scene.collect(scene.at(hour=11), backfill)
    assert result.mode == "backfill" and result.backfill is not None
    after = scene.current()
    series = scene.observations(after)["backfill"]
    assert len(series) == 1 and series[0]["since"] == "2026-10-01T00:00:00Z"
    assert sum(series[0]["counts"].values()) >= 1  # the imported history is counted ...
    assert live_counters(after)["executions"] == live_counters(live)["executions"]  # not live
    assert after.counter_epoch == live.counter_epoch  # backfill never starts an epoch
    assert observation(after, "24h", "searches") == observation(live, "24h", "searches")


def test_duration_histograms_use_the_ten_fixed_buckets_with_exact_count_and_sum(
    tmp_path: Path,
) -> None:
    vs.enable()
    vs.start_capture()
    durations = [500, 1_000, 1_001, 5_000, 99_999, 100_000, 30_000_000, 30_000_001, None]
    for index, micros in enumerate(durations):
        scene.journal_events(vs.timed_event(f"h-{index}", ts(COUNTERS_DAY, 9, index), micros))
    scene.collect(scene.at(hour=10))
    view_doc = live_counters(scene.current())
    (series,) = [d for d in view_doc["duration"] if d["transport"] == "mcp"]
    assert [b["le_us"] for b in series["buckets"]] == [
        "1000",
        "5000",
        "10000",
        "50000",
        "100000",
        "500000",
        "1000000",
        "5000000",
        "30000000",
        "+Inf",
    ]
    known = [d for d in durations if d is not None]
    assert series["count"] == len(known) and series["sum_us"] == sum(known)  # exact integers
    cumulative = [b["count"] for b in series["buckets"]]
    assert cumulative == sorted(cumulative) and cumulative[-1] == series["count"]
    assert cumulative == [2, 4, 4, 4, 6, 6, 6, 6, 7, 8]  # inclusive upper bounds (le)
    unknown = [d for d in view_doc["duration_unknown"] if d["transport"] == "mcp"]
    assert [d["count"] for d in unknown] == [1]  # unknown duration kept apart, not 0
    raw = (vs.accounting_dir() / "snapshots" / "current.json").read_text()
    assert "." not in json.dumps(view_doc["duration"])  # no float anywhere in the series
    assert '"sum_us":' in raw


@pytest.mark.parametrize("ordinary_restart", [True, False])
def test_histogram_count_and_sum_survive_restart_and_epoch_boundaries(
    tmp_path: Path, ordinary_restart: bool
) -> None:
    vs.enable()
    vs.start_capture()
    scene.journal_events(vs.timed_event("h-1", ts(COUNTERS_DAY, 9, 1), 2_000))
    scene.collect(scene.at(hour=10))
    scene.journal_events(vs.timed_event("h-2", ts(COUNTERS_DAY, 9, 2), 3_000))
    options = scene.LIVE if ordinary_restart else CollectOptions(new_epoch=True)
    scene.collect(scene.at(hour=11), options)
    mcp = [d for d in live_counters(scene.current())["duration"] if d["transport"] == "mcp"]
    series = mcp[0] if mcp else {"count": 0, "sum_us": 0}
    if ordinary_restart:
        assert (series["count"], series["sum_us"]) == (2, 5_000)
    else:
        assert (series["count"], series["sum_us"]) == (0, 0)  # reset for the new epoch


# Remaining aggregate boundaries: corrupt series, unit identity, revisions, and late data.


@pytest.mark.parametrize("field", ["operation", "counted"])
def test_corrupt_counter_labels_or_dedupe_container_cannot_be_restored(field: str) -> None:
    valid = counters.observe(fresh(), mixed_events(), horizon_day=None)
    raw = valid.to_dict()
    assert counters.LiveCounters.from_dict(raw) == valid
    if field == "operation":
        raw["executions"][0]["operation"] = ""
    else:
        raw["counted"] = {}
    with pytest.raises(ValueError):
        counters.LiveCounters.from_dict(raw)


def test_naive_source_timestamp_cannot_advance_the_published_evidence_watermark() -> None:
    record = SourceRecord("claude", "claude:/private/source", last_ts="2027-01-01T12:00:00")
    assert (
        collector_cycle.coverage_through(
            {record.key: record}, "2026-10-12T11:00:00Z", "2026-10-12T12:00:00Z"
        )
        == "2026-10-12T11:00:00Z"
    )


# Admission, damaged sources, and advisory inputs retain their distinct accounting meanings.


def test_journal_damage_changes_reduction_signature_without_changing_valid_events(
    tmp_path: Path,
) -> None:
    seeded(tmp_path)
    before = read_journal(PROFILE)
    original = collector_inputs.journal_signature(before)
    path = next((vs.accounting_dir() / "journal").glob("*.jsonl"))
    with path.open("ab") as handle:
        handle.write(b"not-json\n")
    after = read_journal(PROFILE)
    assert after.events == before.events and after.quarantined
    assert collector_inputs.journal_signature(after) != original
    checkpoint_before = scene.committed()
    from ssgrep.accounting.collect.read import read_cached_journal

    observed_signature = read_cached_journal(PROFILE, checkpoint_before).signature
    assert observed_signature is not None
    scene.collect(NOW)
    assert scene.committed().journal_sig == observed_signature
    assert scene.counter_total(scene.current()) == 2


def journal_segment_fixture():
    """One committed journal segment, plus the cache record an accepted cycle would hold."""
    from ssgrep.accounting.privacy.storage import to_plain

    scene.install_key()
    scene.journal_events(scene.execution("n-old", ts(scene.DAY, 9), text="old measurement"))
    journal = read_journal(PROFILE, quarantine=False)
    [proof] = journal.proofs
    assert journal.segment_events[proof.name]
    cached = {
        "proof": to_plain(proof),
        "events": [to_plain(row) for row in journal.segment_events[proof.name]],
        "damage": [],
        "retention_pending": False,
    }
    return journal, proof, cached


def test_journal_cache_round_trip_reuses_segments_and_keeps_pending_cleanup_marks() -> None:
    from ssgrep.accounting.collect.read import (
        JOURNAL_CACHE_BLOB,
        journal_references,
        read_cached_journal,
        save_journal_cache,
    )

    journal, proof, _ = journal_segment_fixture()
    journal.retention_pending.add(proof.name)
    blobs: dict[str, str] = {}
    save_journal_cache(PROFILE, journal, blobs, None)
    checkpoint = store.Checkpoint(PROFILE, generation=1, blobs=blobs)

    reused = read_cached_journal(PROFILE, checkpoint)
    assert reused.events == journal.events and reused.events
    assert reused.retention_pending == {proof.name}
    assert reused.cache_dirty == set()  # nothing was parsed again
    assert set(reused.cache_blobs) == {proof.name}
    assert journal_references(PROFILE, blobs[JOURNAL_CACHE_BLOB]) == set(
        reused.cache_blobs.values()
    )

    fresh = read_cached_journal(PROFILE, None)
    assert fresh.events == journal.events and fresh.cache_dirty == {proof.name}
    assert fresh.retention_pending == set()


@pytest.mark.parametrize("damage", ["schema", "not_a_mapping", "segment_blob", "segment_proof"])
def test_a_damaged_journal_cache_is_typed_state_corruption_never_a_silent_reparse(
    damage: str,
) -> None:
    from ssgrep.accounting.collect.read import (
        JOURNAL_CACHE_BLOB,
        journal_references,
        read_cached_journal,
    )

    _, proof, cached = journal_segment_fixture()
    segment = store.write_blob(PROFILE, "journal-segment", cached)
    manifest = {"schema": 1, "segments": {proof.name: segment}}
    control = store.write_blob(PROFILE, JOURNAL_CACHE_BLOB, manifest)
    assert read_cached_journal(
        PROFILE, store.Checkpoint(PROFILE, 1, blobs={JOURNAL_CACHE_BLOB: control})
    )
    assert journal_references(PROFILE, control) == {segment}

    if damage == "schema":
        manifest = {"schema": 2, "segments": {}}
    elif damage == "not_a_mapping":
        manifest = ["not", "a", "manifest"]
    elif damage == "segment_blob":
        manifest["segments"][proof.name] = store.write_blob(PROFILE, "journal-segment", ["x"])
    else:
        broken = {**cached, "proof": {"name": proof.name}}
        manifest["segments"][proof.name] = store.write_blob(PROFILE, "journal-segment", broken)
    reference = store.write_blob(PROFILE, JOURNAL_CACHE_BLOB, manifest)
    checkpoint = store.Checkpoint(PROFILE, 2, blobs={JOURNAL_CACHE_BLOB: reference})
    with pytest.raises(CollectError) as failed:
        read_cached_journal(PROFILE, checkpoint)
    assert failed.value.condition == "state_corrupt"
    if damage in {"schema", "not_a_mapping"}:
        with pytest.raises(CollectError) as refused:
            journal_references(PROFILE, reference)
        assert refused.value.condition == "state_corrupt"


def test_a_missing_journal_segment_needs_retention_authority_and_every_retained_measurement(
    tmp_path: Path,
) -> None:
    from ssgrep.accounting.capture.journal_reader import Damage
    from ssgrep.accounting.collect.read import _require_cached_segments
    from ssgrep.accounting.privacy.storage import to_plain

    journal, proof, cached = journal_segment_fixture()
    segment = next((vs.accounting_dir() / "journal").glob("*.jsonl"))
    segment.unlink()
    scene.journal_events(scene.execution("n-old", ts(scene.DAY, 9), text="old measurement"))
    survivors = read_journal(PROFILE, quarantine=False)
    assert survivors.proofs and proof.name not in {row.name for row in survivors.proofs}
    assert [row.execution_id for row in survivors.events] == [
        row.execution_id for row in journal.events
    ]
    for path in (vs.accounting_dir() / "journal").glob("*.jsonl"):
        path.unlink()
    nothing = read_journal(PROFILE, quarantine=False)
    assert not nothing.events

    retention = {
        "key": "00" * 32,
        "horizon_day": "2000-01-01",
        "proofs": [to_plain(proof)],
        "retired": {},
        "bridges": [],
    }
    previous = {proof.name: cached}
    # Control: the same measurements survive in another accepted segment.
    _require_cached_segments(survivors, previous, retention)

    def refused(journal_view, accepted, authority):
        with pytest.raises(CollectError) as failed:
            _require_cached_segments(journal_view, accepted, authority)
        assert failed.value.condition == "state_corrupt"
        return str(failed.value)

    assert "without retention authority" in refused(survivors, previous, None)
    other = {**retention, "proofs": []}
    assert "without retention authority" in refused(survivors, previous, other)
    assert "retained measurements" in refused(nothing, previous, retention)
    assert "invalid" in refused(
        survivors, {proof.name: {**cached, "proof": {"name": proof.name}}}, retention
    )
    pending = Damage(proof.name, 0, 4, "digest", "incomplete", "incomplete_tail")
    held = {**cached, "damage": [to_plain(pending)]}
    assert "pending record" in refused(survivors, {proof.name: held}, retention)
    # Pending damage in a segment that was last written before the horizon is expired history.
    future = {**retention, "horizon_day": "2999-12-31"}
    _require_cached_segments(survivors, {proof.name: held}, future)


def test_inaccessible_fingerprint_is_unknown_not_a_newly_minted_key(tmp_path: Path) -> None:
    seeded(tmp_path)
    path = vs.accounting_dir() / fingerprint.KEY_FILE
    before = path.read_bytes()
    fingerprint.forget_keys()
    path.chmod(0o644)
    try:
        assert collector_inputs.load_fingerprint_key(PROFILE) is None
        assert path.read_bytes() == before
    finally:
        path.chmod(0o600)
    assert collector_inputs.load_fingerprint_key(PROFILE) == KEY


# Status distinguishes unsafe, unreadable, and concurrently removed storage from zero usage.


# Collector recovery refuses unsafe or mismatched state without publishing new totals.


# ---- survivor source: test_guard_collector_recovery.py -----------------------------------


guard_recovery_NOW = scene.at(hour=20)


def guard_recovery_seeded(root: Path) -> None:
    vs.enable(pin=True)
    vs.start_capture()
    vs.pair(vs.projects(root), 0, scene.at(hour=9))
    assert scene.report(scene.collect(guard_recovery_NOW)).generation == 1
    assert scene.counter_total(scene.current()) == 2


def guard_recovery_assert_failed_without_publish(
    condition: str, generation: int, snapshot: bytes
) -> None:
    assert (vs.accounting_dir() / "snapshots" / "current.json").read_bytes() == snapshot
    progress = collector_progress.read_progress(PROFILE)
    assert progress is not None
    assert progress["phase"] == "idle"
    assert progress["last_cycle"]["error"] == condition
    assert progress["last_cycle"]["generation"] is None
    assert scene.current().generation == generation


@pytest.mark.parametrize(
    ("damage", "condition"),
    [
        (b'{"schema":', "state_corrupt"),
        (b"[]", "state_corrupt"),
        (b'{"generation":1}', "state_corrupt"),
        (b'{"schema":1,"generation":1.5}', "state_corrupt"),
        (b'{"schema":1}', "state_corrupt"),
        (b'{"schema":true}', "unsupported_schema"),
        (json.dumps({"schema": SCHEMA_VERSION + 1}).encode(), "unsupported_schema"),
    ],
)
def test_invalid_checkpoint_never_replaces_published_totals_guard_recovery(
    tmp_path: Path, damage: bytes, condition: str
) -> None:
    guard_recovery_seeded(tmp_path)
    path = vs.accounting_dir() / "collector" / store.CHECKPOINT
    valid = path.read_bytes()
    snapshot = (vs.accounting_dir() / "snapshots" / "current.json").read_bytes()
    path.write_bytes(damage)
    with pytest.raises(CollectError) as failed:
        scene.collect(guard_recovery_NOW)
    assert failed.value.condition == condition
    assert path.read_bytes() == damage
    guard_recovery_assert_failed_without_publish(condition, 1, snapshot)
    path.write_bytes(valid)
    assert scene.report(scene.collect(guard_recovery_NOW)).generation == 2
    assert scene.counter_total(scene.current()) == 2


@pytest.mark.parametrize("target", ["checkpoint", "counters"])
def test_state_size_limits_fail_closed_and_recover_after_limit_restored_guard_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    guard_recovery_seeded(tmp_path)
    directory = vs.accounting_dir() / "collector"
    checkpoint = scene.committed()
    path = (
        directory / store.CHECKPOINT
        if target == "checkpoint"
        else directory / "state" / checkpoint.blobs["counters"]
    )
    limit = "MAX_CHECKPOINT_BYTES" if target == "checkpoint" else "MAX_BLOB_BYTES"
    original = path.read_bytes()
    snapshot = (vs.accounting_dir() / "snapshots" / "current.json").read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(store, limit, len(original) - 1)
        with pytest.raises(CollectError) as failed:
            scene.collect(guard_recovery_NOW)
        assert failed.value.condition == "state_corrupt"
    guard_recovery_assert_failed_without_publish("state_corrupt", 1, snapshot)
    assert path.read_bytes() == original
    assert scene.report(scene.collect(guard_recovery_NOW)).generation == 2
    assert scene.counter_total(scene.current()) == 2


def test_missing_committed_counter_blob_does_not_start_new_epoch_guard_recovery(
    tmp_path: Path,
) -> None:
    guard_recovery_seeded(tmp_path)
    path = vs.accounting_dir() / "collector" / "state" / scene.committed().blobs["counters"]
    original = path.read_bytes()
    snapshot = (vs.accounting_dir() / "snapshots" / "current.json").read_bytes()
    path.unlink()
    with pytest.raises(CollectError) as failed:
        scene.collect(guard_recovery_NOW)
    assert failed.value.condition == "state_corrupt"
    guard_recovery_assert_failed_without_publish("state_corrupt", 1, snapshot)
    path.write_bytes(original)
    path.chmod(0o600)
    resumed = scene.collect(guard_recovery_NOW)
    assert resumed.epoch == 1 and scene.counter_total(scene.current()) == 2


@pytest.mark.parametrize("blob", ["counters", "sealed", "reduction", "ledger"])
@pytest.mark.parametrize("damage", [b"null", b"{}"])
def test_invalid_referenced_state_is_not_absent_first_run_guard_recovery(
    tmp_path: Path, blob: str, damage: bytes
) -> None:
    guard_recovery_seeded(tmp_path)
    checkpoint = scene.committed()
    key = next(k for k in checkpoint.blobs if k.startswith("ledger-")) if blob == "ledger" else blob
    path = vs.accounting_dir() / "collector" / "state" / checkpoint.blobs[key]
    if blob == "sealed" and damage == b"{}":
        damage = b"[]"
    valid = path.read_bytes()
    snapshot = (vs.accounting_dir() / "snapshots" / "current.json").read_bytes()
    path.write_bytes(damage)
    with pytest.raises(CollectError) as failed:
        scene.collect(guard_recovery_NOW)
    assert failed.value.condition == "state_corrupt"
    assert scene.committed() == checkpoint
    guard_recovery_assert_failed_without_publish("state_corrupt", 1, snapshot)
    path.write_bytes(valid)
    assert scene.report(scene.collect(guard_recovery_NOW)).generation == 2
    assert scene.counter_total(scene.current()) == 2


@pytest.mark.parametrize("damage", ["missing", "truncated", "oversized", "malformed"])
def test_committed_evidence_damage_cannot_be_rebuilt_as_empty_history_guard_recovery(
    tmp_path: Path, damage: str
) -> None:
    guard_recovery_seeded(tmp_path)
    ref = scene.committed().segments[0]
    path = vs.accounting_dir() / "collector" / "evidence" / ref.name
    valid = path.read_bytes()
    snapshot = (vs.accounting_dir() / "snapshots" / "current.json").read_bytes()
    if damage == "missing":
        path.unlink()
    else:
        damaged = {
            "truncated": valid[:-1],
            "oversized": valid + b"extra",
            "malformed": b"!" * len(valid),
        }
        path.write_bytes(damaged[damage])
    with pytest.raises(CollectError) as failed:
        scene.collect(guard_recovery_NOW, CollectOptions(recompute=True))
    assert failed.value.condition == "state_corrupt"
    guard_recovery_assert_failed_without_publish("state_corrupt", 1, snapshot)
    path.write_bytes(valid)
    path.chmod(0o600)
    assert (
        scene.report(scene.collect(guard_recovery_NOW, CollectOptions(recompute=True))).generation
        == 2
    )
    assert scene.counter_total(scene.current()) == 2
    assert len(vs.evidence_identities()) == len(set(vs.evidence_identities()))


def test_interrupted_epoch_commit_recovers_without_recounting_guard_recovery(
    tmp_path: Path,
) -> None:
    guard_recovery_seeded(tmp_path)
    snapshot = (vs.accounting_dir() / "snapshots" / "current.json").read_bytes()

    def interrupt(step: str) -> None:
        if step == "checkpoint_committed":
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        scene.collect(guard_recovery_NOW, CollectOptions(new_epoch=True), on_step=interrupt)
    assert scene.committed().generation == 2
    guard_recovery_assert_failed_without_publish("interrupted", 1, snapshot)
    recovered = scene.collect(guard_recovery_NOW)
    assert recovered.epoch == 2
    assert scene.report(recovered).generation == 3
    assert scene.counter_total(scene.current()) == 0
    scene.journal_events(scene.execution("after-epoch", "2026-10-12T20:00:01Z"))
    scene.collect(scene.at(hour=21))
    assert scene.counter_total(scene.current()) == 1


def test_worker_failed_cycle_reports_corruption_then_recovers_guard_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    guard_recovery_seeded(tmp_path)
    monkeypatch.setattr(collector, "_now", lambda: guard_recovery_NOW)
    path = vs.accounting_dir() / "collector" / store.CHECKPOINT
    valid = path.read_bytes()
    path.write_bytes(b"{")
    cycle = collector.run_worker_cycle(PROFILE)
    failed = cycle(lambda: False)
    assert (failed.status, failed.error, failed.generation) == ("failed", "state_corrupt", None)
    assert scene.current().generation == 1
    path.write_bytes(valid)
    succeeded = cycle(lambda: False)
    assert (succeeded.status, succeeded.generation) == ("published", 2)
    assert scene.counter_total(scene.current()) == 2
    progress = collector_progress.read_progress(PROFILE)
    assert progress is not None and progress["cycles"] == 3
    assert progress["kind"] == "serve" and progress["last_cycle"]["error"] is None


@pytest.mark.parametrize("component", ["worker", "snapshots"])
def test_unsafe_worker_or_snapshot_path_is_refused_without_following_it(
    tmp_path: Path, component: str
) -> None:
    guard_recovery_seeded(tmp_path)
    directory = vs.accounting_dir() / component
    saved = tmp_path / f"saved-{component}"
    current = vs.accounting_dir() / "snapshots" / "current.json"
    before = current.read_bytes()
    directory.rename(saved)
    directory.symlink_to(saved, target_is_directory=True)
    try:
        with pytest.raises(CollectError) as failed:
            scene.collect(guard_recovery_NOW)
        assert failed.value.condition == "storage_unsafe"
        assert current.read_bytes() == before
        assert scene.committed().generation == 1
    finally:
        directory.unlink()
        saved.rename(directory)
    assert scene.report(scene.collect(guard_recovery_NOW)).generation == 2
    assert scene.counter_total(scene.current()) == 2


def test_newer_snapshot_schema_is_not_repaired_or_overwritten(tmp_path: Path) -> None:
    guard_recovery_seeded(tmp_path)
    path = vs.accounting_dir() / "snapshots" / "current.json"
    original = path.read_bytes()
    newer = json.loads(original)
    newer["schema_version"] = SCHEMA_VERSION + 1
    path.write_text(json.dumps(newer))
    before = path.read_bytes()
    with pytest.raises(CollectError) as failed:
        scene.collect(guard_recovery_NOW)
    assert failed.value.condition == "unsupported_schema"
    assert path.read_bytes() == before and scene.committed().generation == 1
    path.write_bytes(original)
    assert scene.report(scene.collect(guard_recovery_NOW)).generation == 2


def test_malformed_current_pointer_is_repaired_from_the_committed_generation(
    tmp_path: Path,
) -> None:
    guard_recovery_seeded(tmp_path)
    path = vs.accounting_dir() / "snapshots" / "current.json"
    path.write_bytes(b"{")
    result = scene.collect(guard_recovery_NOW)
    assert scene.report(result).generation == 2
    assert scene.counter_total(scene.current()) == 2
    assert scene.current().generation == 2


def test_missing_committed_snapshot_is_corruption_not_permission_to_reset(tmp_path: Path) -> None:
    guard_recovery_seeded(tmp_path)
    directory = vs.accounting_dir() / "snapshots"
    generation = next(directory.glob("gen-*.json"))
    original = generation.read_bytes()
    current = (directory / "current.json").read_bytes()
    generation.unlink()
    with pytest.raises(CollectError) as failed:
        scene.collect(guard_recovery_NOW)
    assert failed.value.condition == "state_corrupt"
    assert (directory / "current.json").read_bytes() == current
    generation.write_bytes(original)
    generation.chmod(0o600)
    assert scene.report(scene.collect(guard_recovery_NOW)).generation == 2


@pytest.mark.parametrize("component", ["state", "evidence"])
def test_missing_committed_input_directory_is_not_an_empty_dataset(
    tmp_path: Path, component: str
) -> None:
    guard_recovery_seeded(tmp_path)
    directory = vs.accounting_dir() / "collector" / component
    saved = tmp_path / f"saved-{component}"
    before = scene.current()
    directory.rename(saved)
    vs.filler(vs.projects(tmp_path), "new-input", guard_recovery_NOW, 1)
    try:
        with pytest.raises(CollectError) as failed:
            scene.collect(guard_recovery_NOW)
        assert failed.value.condition == "state_corrupt"
        assert scene.current() == before
    finally:
        if directory.exists():
            directory.rmdir()  # an empty staging directory may have been opened before reading
        saved.rename(directory)
    assert scene.report(scene.collect(guard_recovery_NOW)).generation == 2
    assert scene.counter_total(scene.current()) == 2


# VersionedLedger: one active version per evidence identity, sealed periods append-only.


NS = METHOD.hash
SEAL = seal_day(DAY)  # first UTC day on which acceptance of a DAY period counts as late
LAST_PROVISIONAL = "2026-11-08"


def bundle(cands: Sequence[Candidate], debits: Sequence[Debit]) -> EvidenceBundle:
    return EvidenceBundle(tuple(cands), tuple(debits))


def central(row: PeriodRow, part: str = "net") -> int | None:
    for _, estimates in row.estimates:
        for estimate in estimates:
            if estimate.scenario is Scenario.CENTRAL:
                return getattr(estimate, part).value
    raise AssertionError("no central estimate")


def cand_b() -> Candidate:
    return candidate("b", immediate=(atom("s2", 200, 0, "ev-2"),))  # 50 tokens of credit


# ---- (a) refinement inside the provisional window ---------------------------------


def test_naive_ingest_of_both_debit_versions_is_the_conflict_this_module_prevents():
    naive = Ledger(METHOD)
    naive.ingest(bundle([candidate("a")], [debit("x", None)]))
    naive.ingest(bundle([], [debit("x", 640)]))
    assert b"conflicting_duplicate_debit" in naive.rows().canonical_bytes()


def test_debit_refined_inside_provisional_window_replaces_in_place_without_a_revision():
    ledger = VersionedLedger(METHOD)
    first = ledger.reconcile(
        {"root-1": bundle([candidate("a")], [debit("x", None)])}, today="2026-11-02"
    )
    assert first == ReconcileResult(2, 0, 0, 0, 0, ())
    before = ledger.rows()
    assert central(before.periods[0], "debit") == 0 and central(before.periods[0]) is None

    second = ledger.reconcile(
        {"root-1": bundle([candidate("a")], [debit("x", 640)])}, today="2026-11-03"
    )

    assert second == ReconcileResult(0, 1, 0, 1, 0, ())
    state = ledger.state()
    assert [(d.debit_id, d.chars, d.accepted_day) for d in state.debits] == [
        ("x", 640, "2026-11-02")
    ]
    assert ledger.superseded == {}
    rows = ledger.rows()
    assert b"conflicting_duplicate_debit" not in rows.canonical_bytes()
    assert [p.revision for p in rows.periods] == [1]
    leaf = rows.periods[0].leaves[0]
    assert (leaf.debit_raw, leaf.debit_known, leaf.debit_missing) == (640, 1, ())
    assert central(rows.periods[0]) == 100 - 160


def test_reconciling_the_same_projection_twice_the_same_day_changes_nothing():
    ledger = VersionedLedger(METHOD)
    desired = {"root-1": bundle([candidate("a"), cand_b()], [debit("x", 640)])}
    ledger.reconcile(desired, today="2026-11-02")
    snapshot = ledger.rows().canonical_bytes(), ledger.state(), dict(ledger.superseded)
    again = ledger.reconcile(desired, today="2026-11-02")
    assert again == ReconcileResult(0, 0, 0, 3, 0, ())
    assert (ledger.rows().canonical_bytes(), ledger.state(), ledger.superseded) == snapshot


# ---- (b) refinement after the seal -------------------------------------------------


def test_refinement_after_the_seal_adds_revision_two_and_keeps_revision_one_intact():
    ledger = VersionedLedger(METHOD)
    ledger.reconcile({"root-1": bundle([candidate("a")], [debit("x", None)])}, today="2026-11-02")
    (old_digest,) = ledger._debits
    before = ledger.rows()

    result = ledger.reconcile({"root-1": bundle([candidate("a")], [debit("x", 640)])}, today=SEAL)

    assert result == ReconcileResult(0, 1, 0, 1, 1, ())
    assert ledger.superseded == {old_digest: SEAL}
    assert len(ledger._debits) == 2  # old version retained for history
    rows = ledger.rows()
    assert rows.periods[0] == before.periods[0]  # revision 1 byte-for-byte unchanged
    rev1, rev2 = rows.periods
    assert (rev2.revision, rev2.supersedes, rev2.kind, rev2.late) == (2, 1, "adjustment", True)
    assert central(rev1) is None and central(rev2) == -60
    shown = history(rows.periods, NS, CHARS, today=SEAL)
    assert [(h.revision, h.current) for h in shown] == [(1, False), (2, True)]
    total, _ = current_total(rows.periods, NS, CHARS)
    assert total is not None and total.value == -60  # only the current revision counts


@pytest.mark.parametrize(("today", "revisions"), [(LAST_PROVISIONAL, 1), (SEAL, 2)])
def test_the_seal_boundary_decides_between_rewrite_and_revision(today, revisions):
    ledger = VersionedLedger(METHOD)
    ledger.reconcile({"root-1": bundle([candidate("a")], [debit("x", None)])}, today="2026-11-02")
    ledger.reconcile({"root-1": bundle([candidate("a")], [debit("x", 640)])}, today=today)
    assert len(ledger.rows().periods) == revisions


# ---- (c) retraction -----------------------------------------------------------------


def test_sealed_retraction_supersedes_without_touching_revision_one():
    ledger = VersionedLedger(METHOD)
    ledger.reconcile(
        {"root-1": bundle([candidate("a"), cand_b()], [debit("x", 640)])}, today="2026-11-02"
    )
    before = ledger.rows()
    assert central(before.periods[0], "credit") == 150

    result = ledger.reconcile({"root-1": bundle([candidate("a")], [debit("x", 640)])}, today=SEAL)

    assert result == ReconcileResult(0, 0, 1, 2, 0, ())
    assert list(ledger.superseded.values()) == [SEAL]
    rows = ledger.rows()
    assert rows.periods[0] == before.periods[0]
    assert [p.revision for p in rows.periods] == [1, 2]
    assert central(rows.periods[1], "credit") == 100
    assert central(rows.periods[1]) == 100 - 160
    # the retracted candidate stays in the ledger as history
    assert len(ledger.state().candidates) == 2


def test_provisional_retraction_simply_removes_the_item():
    ledger = VersionedLedger(METHOD)
    ledger.reconcile(
        {"root-1": bundle([candidate("a"), cand_b()], [debit("x", 640)])}, today="2026-11-02"
    )
    result = ledger.reconcile(
        {"root-1": bundle([candidate("a")], [debit("x", 640)])}, today="2026-11-03"
    )
    assert result == ReconcileResult(0, 0, 1, 2, 0, ())
    assert [c.candidate_id for c in ledger.state().candidates] == ["a"]
    assert ledger.superseded == {}
    rows = ledger.rows()
    assert [p.revision for p in rows.periods] == [1]
    assert central(rows.periods[0], "credit") == 100


def test_roots_not_mentioned_in_desired_are_left_untouched():
    ledger = VersionedLedger(METHOD)
    other = TaskId("root-2", "turn-1", "main")
    ledger.reconcile(
        {
            "root-1": bundle([candidate("a")], []),
            "root-2": bundle([candidate("z", task=other)], []),
        },
        today="2026-11-02",
    )
    result = ledger.reconcile({"root-1": bundle([candidate("a")], [])}, today="2026-11-03")
    assert result == ReconcileResult(0, 0, 0, 1, 0, ())
    assert {c.task.root for c in ledger.state().candidates} == {"root-1", "root-2"}


def test_conflicting_variants_of_one_identity_in_a_projection_are_rejected_deterministically():
    ledger = VersionedLedger(METHOD)
    pair = [debit("x", 640), debit("x", 800)]
    result = ledger.reconcile({"root-1": bundle([], pair)}, today="2026-11-02")
    other = VersionedLedger(METHOD)
    other.reconcile({"root-1": bundle([], pair[::-1])}, today="2026-11-02")
    assert result.rejected == (("x", "conflicting_desired_variant"),)
    assert ledger.state() == other.state() and len(ledger.state().debits) == 1


@pytest.mark.parametrize("second_supported", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_distinct_lookups_keep_supported_reuse_and_every_debit_without_double_credit(
    second_supported: bool, reverse: bool
) -> None:
    first = candidate("same-fingerprint", exposure_ids=("lookup-1",))
    second = candidate(
        "same-fingerprint",
        exposure_ids=("lookup-2",),
        novel=second_supported,
        eligibility=Eligibility.SUPPORTED if second_supported else Eligibility.KNOWN_REJECTED,
        eligibility_reasons=() if second_supported else ("already_visible",),
        immediate=first.immediate if second_supported else (),
    )
    candidates = (second, first) if reverse else (first, second)
    debits = (
        debit("debit-1", 640, exposure_id="lookup-1"),
        debit("debit-2", 640, exposure_id="lookup-2"),
    )
    ledger = VersionedLedger(METHOD)
    result = ledger.reconcile({"root-1": bundle(candidates, debits)}, today=DAY)
    assert result.rejected == ()
    assert {row.exposure_ids for row in ledger.state().candidates} == {("lookup-1",), ("lookup-2",)}
    assert len(ledger.state().debits) == 2
    rows = ledger.rows()
    assert central(rows.periods[0], "credit") == 100  # same 400-char source union, once
    assert central(rows.periods[0], "debit") == 320  # two distinct 640-char exposures
    assert central(rows.periods[0]) == -220
    assert (
        rows.canonical_bytes() == full_replay(bundle(candidates, debits), METHOD).canonical_bytes()
    )
    restored = VersionedLedger.restore_versioned(ledger.state(), ledger.superseded_state())
    assert restored.rows().canonical_bytes() == rows.canonical_bytes()


def test_conflicting_candidate_variants_of_the_same_lookup_remain_rejected() -> None:
    first = candidate("same-fingerprint", exposure_ids=("lookup-1", "lookup-2"))
    variant = candidate(
        "same-fingerprint",
        exposure_ids=("lookup-2", "lookup-1"),
        novel=False,
        eligibility=Eligibility.KNOWN_REJECTED,
        eligibility_reasons=("already_visible",),
    )
    ledger = VersionedLedger(METHOD)
    result = ledger.reconcile({"root-1": bundle((first, variant), ())}, today=DAY)
    assert result.rejected == (("same-fingerprint", "conflicting_desired_variant"),)
    assert len(ledger.state().candidates) == 1
    reversed_ledger = VersionedLedger(METHOD)
    reversed_ledger.reconcile({"root-1": bundle((variant, first), ())}, today=DAY)
    assert reversed_ledger.state() == ledger.state()


def test_refining_one_lookup_preserves_other_observations_of_the_same_fingerprint() -> None:
    first = candidate("same-fingerprint", exposure_ids=("lookup-1",))
    second = candidate("same-fingerprint", exposure_ids=("lookup-2",))
    ledger = VersionedLedger(METHOD)
    ledger.reconcile({"root-1": bundle((first, second), ())}, today=DAY)
    held_first = next(row for row in ledger.state().candidates if row.exposure_ids == ("lookup-1",))
    rejected_second = candidate(
        "same-fingerprint",
        exposure_ids=("lookup-2",),
        novel=False,
        eligibility=Eligibility.KNOWN_REJECTED,
        eligibility_reasons=("already_visible",),
        immediate=(),
    )
    result = ledger.reconcile({"root-1": bundle((first, rejected_second), ())}, today=DAY)
    assert (result.replaced, result.unchanged, result.retracted, result.rejected) == (1, 1, 0, ())
    current = {row.exposure_ids: row for row in ledger.state().candidates}
    assert current[("lookup-1",)] == held_first
    assert current[("lookup-2",)].eligibility is Eligibility.KNOWN_REJECTED
    assert central(ledger.rows().periods[0], "credit") == 100


def test_restored_collapsed_projection_recovers_first_valid_credit_without_recounting_costs():
    first = candidate("same-fingerprint", exposure_ids=("lookup-1",))
    later_known = candidate(
        "same-fingerprint",
        exposure_ids=("lookup-2",),
        novel=False,
        eligibility=Eligibility.KNOWN_REJECTED,
        eligibility_reasons=("already_visible",),
        immediate=(),
    )
    costs = (
        debit("debit-1", 640, exposure_id="lookup-1"),
        debit("debit-2", 640, exposure_id="lookup-2"),
    )
    # A legitimate old digest-selected projection could retain only the later
    # known-rejected observation despite accepting both distinct lookup costs.
    old = VersionedLedger(METHOD)
    old.reconcile({"root-1": bundle((later_known,), costs)}, today=DAY)
    assert central(old.rows().periods[0], "credit") == 0
    assert central(old.rows().periods[0], "debit") == 320
    state = ledger_state_from_dict(json.loads(json.dumps(ledger_state_to_dict(old.state()))))
    restored = VersionedLedger.restore_versioned(state, old.superseded_state())
    result = restored.reconcile({"root-1": bundle((first, later_known), costs)}, today=DAY)
    assert (result.added, result.unchanged, result.rejected) == (1, 3, ())
    assert central(restored.rows().periods[0], "credit") == 100
    assert central(restored.rows().periods[0], "debit") == 320
    assert (
        restored.rows().canonical_bytes()
        == full_replay(bundle((first, later_known), costs), METHOD).canonical_bytes()
    )


# ---- (e) frozen roots ----------------------------------------------------------------


def test_frozen_root_items_are_rejected_while_other_roots_proceed():
    ledger = VersionedLedger(METHOD)
    other = TaskId("root-2", "turn-1", "main")
    ledger.reconcile(
        {
            "root-1": bundle([candidate("a")], [debit("x", 640)]),
            "root-2": bundle([candidate("z", task=other)], []),
        },
        today="2026-11-02",
    )
    assert ledger.expire_roots(["root-1"]) == ["root-1"]
    frozen_rows = [p for p in ledger.rows().periods if p.period == DAY]

    result = ledger.reconcile(
        {
            "root-1": bundle([candidate("a"), cand_b()], [debit("x", 800)]),
            "root-2": bundle([candidate("z", task=other), candidate("y", task=other)], []),
        },
        today="2026-11-03",
    )

    assert result.rejected == (
        ("a", "evidence_expired_root"),
        ("b", "evidence_expired_root"),
        ("x", "evidence_expired_root"),
    )
    assert (result.added, result.unchanged) == (1, 1)
    state = ledger.state()
    assert [f.root for f in state.frozen] == ["root-1"]
    assert sorted(c.candidate_id for c in state.candidates) == ["y", "z"]
    assert state.debits == ()
    assert [p for p in ledger.rows().periods if p.period == DAY][0].leaves != ()
    assert frozen_rows  # root-1 totals survived the expiry


# ---- (f) equivalence with Ledger, pruning, restore ---------------------------------------


def test_with_no_supersession_rows_are_byte_identical_to_the_base_ledger():
    versioned = VersionedLedger(METHOD)
    versioned.ingest(corpus())
    assert versioned.superseded == {}
    reference = full_replay(corpus(), METHOD)
    assert versioned.rows().canonical_bytes() == reference.canonical_bytes()
    assert versioned.rows().digest == reference.digest


def _sealed_two_root_ledger() -> VersionedLedger:
    ledger = VersionedLedger(METHOD)
    other = TaskId("root-2", "turn-1", "main")
    first = {
        "root-1": bundle([candidate("a")], [debit("x", None)]),
        "root-2": bundle([candidate("z", task=other)], [debit("w", None, task=other)]),
    }
    second = {
        "root-1": bundle([candidate("a")], [debit("x", 640)]),
        "root-2": bundle([candidate("z", task=other)], [debit("w", 80, task=other)]),
    }
    ledger.reconcile(first, today="2026-11-02")
    ledger.reconcile(second, today=SEAL)
    return ledger


def test_expire_roots_and_purge_prune_superseded_entries_of_removed_items():
    ledger = _sealed_two_root_ledger()
    assert len(ledger.superseded) == 2
    ledger.expire_roots(["root-1"])
    assert len(ledger.superseded) == 1
    kept = next(iter(ledger.superseded))
    assert kept in {digest_of(d) for d in ledger.state().debits}
    ledger.purge(roots=["root-2"])
    assert ledger.superseded == {}
    assert ledger.state().debits == ()


def test_restore_versioned_round_trips_through_json_and_drops_unknown_digests():
    ledger = _sealed_two_root_ledger()
    state = json.loads(json.dumps(ledger_state_to_dict(ledger.state())))
    persisted = {**ledger.superseded_state(), "f" * 64: "2026-11-30"}
    restored = VersionedLedger.restore_versioned(ledger_state_from_dict(state), persisted)
    assert restored.superseded == ledger.superseded
    assert restored.superseded_state() == dict(sorted(ledger.superseded.items()))
    assert restored.rows().canonical_bytes() == ledger.rows().canonical_bytes()
    plain = Ledger.restore(ledger.state())  # forgetting supersession double counts: it must differ
    assert plain.rows().canonical_bytes() != ledger.rows().canonical_bytes()


# ---- (d)/(g) simulated collector runs -----------------------------------------------------

STEPS = ["2026-11-02", "2026-11-04", "2026-11-07", "2026-11-09", "2026-11-10", "2026-11-11"]
EVENT_DAYS = ["2026-11-01", "2026-11-02", "2026-11-03"]
ROOTS = ["root-1", "root-2", "root-3"]


def simulate(seed: int) -> list[dict[str, EvidenceBundle]]:
    """Six full projections: anchors per (root, day) never vanish; the rest churn."""
    rng = random.Random(seed)
    truth: dict[str, dict[str, Candidate | Debit]] = {r: {} for r in ROOTS}
    out = []
    for step in range(len(STEPS)):
        for root in ROOTS:
            task = TaskId(root, "turn-1", "main")
            items = truth[root]
            if step == 0:
                for n, day in enumerate(EVENT_DAYS):
                    items[f"c-anchor-{n}"] = candidate(
                        f"{root}-ca{n}",
                        task=task,
                        day=day,
                        immediate=(atom(f"{root}-s{n}", 400, 0, f"ev-{root}-{n}"),),
                    )
                    items[f"d-anchor-{n}"] = debit(f"{root}-da{n}", None, task=task, day=day)
                continue
            for key in sorted(items):
                roll = rng.random()
                item = items[key]
                if isinstance(item, Debit) and roll < 0.5:
                    chars = (item.chars or 0) + rng.choice([40, 120, 400])
                    items[key] = debit(item.debit_id, chars, task=task, day=item.day)
                elif isinstance(item, Candidate) and roll < 0.2:
                    size = rng.choice([200, 600, 800])
                    atom_id = f"{item.candidate_id}-a{size}"
                    items[key] = candidate(
                        item.candidate_id,
                        task=task,
                        day=item.day,
                        immediate=(atom(atom_id, size, 0, f"ev-{item.candidate_id}"),),
                    )
                elif "anchor" not in key and roll > 0.85:
                    del items[key]
            if rng.random() < 0.7:
                tag, day = f"{root}-n{step}", rng.choice(EVENT_DAYS)
                items[f"c-{tag}"] = candidate(
                    tag,
                    task=task,
                    day=day,
                    immediate=(atom(f"{tag}-s", 300, 0, f"ev-{tag}"),),
                )
                items[f"d-{tag}"] = debit(f"{tag}-d", rng.choice([None, 200]), task=task, day=day)
        out.append(
            {
                root: bundle(
                    [i for i in items.values() if isinstance(i, Candidate)],
                    [i for i in items.values() if isinstance(i, Debit)],
                )
                for root, items in truth.items()
            }
        )
    return out


def persist(ledger: VersionedLedger) -> VersionedLedger:
    state = json.loads(json.dumps(ledger_state_to_dict(ledger.state())))
    superseded = json.loads(json.dumps(ledger.superseded_state()))
    return VersionedLedger.restore_versioned(ledger_state_from_dict(state), superseded)


def shuffled(desired: dict[str, EvidenceBundle], rng: random.Random) -> dict[str, EvidenceBundle]:
    out = {}
    for root, b in sorted(desired.items(), reverse=True):
        cands, debits = list(b.candidates), list(b.debits)
        rng.shuffle(cands)
        rng.shuffle(debits)
        out[root] = bundle(cands, debits)
    return out


def test_incremental_runs_with_restarts_match_continuous_runs_and_final_totals():
    projections = simulate(seed=20261013)
    continuous, restarted = VersionedLedger(METHOD), VersionedLedger(METHOD)
    fresh = VersionedLedger(METHOD)
    totals = {"replaced": 0, "retracted": 0, "superseded": 0, "added": 0}
    for projection, today in zip(projections, STEPS, strict=True):
        result = continuous.reconcile(projection, today=today)
        for name in totals:
            totals[name] += getattr(result, name)
        restarted = persist(restarted)
        assert restarted.reconcile(projection, today=today) == result
        assert restarted.rows().canonical_bytes() == continuous.rows().canonical_bytes()
        assert result.rejected == ()
    assert all(totals.values()), totals  # the run exercised every reconcile branch
    assert continuous.superseded  # at least one sealed period was revised

    # A fresh ledger given only the FINAL projection stamps everything with a single day, so its
    # revision rows are NOT required to equal the incremental history; the current totals must.
    fresh.reconcile(projections[-1], today=STEPS[-1])
    final, current = fresh.rows(), continuous.rows()
    assert current_total(current.periods, NS, CHARS) == current_total(final.periods, NS, CHARS)
    assert [(r.period, r.leaves) for r in current_rows(current.periods, NS)] == [
        (r.period, r.leaves) for r in current_rows(final.periods, NS)
    ]
    # no identity ever has two active versions (the conflict reason cannot appear)
    assert b"conflicting_duplicate_debit" not in current.canonical_bytes()


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_item_order_inside_desired_bundles_never_changes_rows(seed):
    projections = simulate(seed=99)
    ordered, scrambled = VersionedLedger(METHOD), VersionedLedger(METHOD)
    rng = random.Random(seed)
    for projection, today in zip(projections, STEPS, strict=True):
        ordered.reconcile(projection, today=today)
        scrambled.reconcile(shuffled(projection, rng), today=today)
        assert scrambled.rows().canonical_bytes() == ordered.rows().canonical_bytes()
        assert scrambled.state() == ordered.state()
        assert scrambled.superseded == ordered.superseded


# ---- restored conflicts, foreign roots, exhaustive reconcile ----------------------------------


def test_restored_conflicting_variants_retire_the_stale_copy_when_one_matches() -> None:
    old = VersionedLedger(METHOD)
    old.ingest(EvidenceBundle((candidate("one"),), (debit("lookup", None),)))
    old.ingest(EvidenceBundle((), (debit("lookup", 640),)))
    assert b"conflicting_duplicate_debit" in old.rows().canonical_bytes()
    restored = VersionedLedger.restore_versioned(old.state(), {})
    result = restored.reconcile(
        {"root-1": EvidenceBundle((candidate("one"),), (debit("lookup", 640),))},
        today="2026-11-02",
    )
    assert result.replaced == 1
    assert len(restored.state().debits) == 1
    assert restored.state().debits[0].chars == 640
    assert b"conflicting_duplicate_debit" not in restored.rows().canonical_bytes()


def test_projection_rejects_candidates_and_debits_owned_by_a_different_root() -> None:
    ledger = VersionedLedger(METHOD)
    foreign = TaskId("other-root", "turn", "main")
    result = ledger.reconcile(
        {
            "root-1": EvidenceBundle(
                (candidate("good"), candidate("foreign", task=foreign)),
                (debit("foreign-debit", 400, task=foreign),),
            )
        },
        today="2026-11-02",
    )
    assert set(result.rejected) == {
        ("foreign", "root_mismatch"),
        ("foreign-debit", "root_mismatch"),
    }
    assert [item.candidate_id for item in ledger.state().candidates] == ["good"]
    assert ledger.state().debits == ()


def test_exhaustive_reconcile_retracts_roots_the_projection_no_longer_mentions() -> None:
    other = TaskId("root-2", "turn-1", "main")

    def projection(root_task: TaskId, ident: str) -> EvidenceBundle:
        return EvidenceBundle(
            (
                candidate(
                    f"c-{ident}", task=root_task, immediate=(atom(f"s-{ident}", 400, 0, ident),)
                ),
            ),
            (debit(f"d-{ident}", 4000, task=root_task, exposure_id=f"e-{ident}"),),
        )

    first = TaskId("root-1", "turn-1", "main")
    ledger = VersionedLedger(METHOD)
    ledger.reconcile(
        {"root-1": projection(first, "1"), "root-2": projection(other, "2")}, today="2026-11-02"
    )
    assert {c.task.root for c in ledger.state().candidates} == {"root-1", "root-2"}

    ledger.reconcile({"root-2": projection(other, "2")}, today="2026-11-03")  # not mentioned
    assert {c.task.root for c in ledger.state().candidates} == {"root-1", "root-2"}  # untouched

    result = ledger.reconcile(
        {"root-2": projection(other, "2")}, today="2026-11-03", exhaustive=True
    )
    assert result.retracted >= 2  # root-1's candidate and debit
    assert {c.task.root for c in ledger.state().candidates} == {"root-2"}
    assert {d.task.root for d in ledger.state().debits} == {"root-2"}
    again = ledger.reconcile(
        {"root-2": projection(other, "2")}, today="2026-11-03", exhaustive=True
    )
    assert (again.added, again.replaced, again.retracted) == (0, 0, 0)  # idempotent


def test_exhaustive_reconcile_never_touches_frozen_roots() -> None:
    ledger = VersionedLedger(METHOD)
    ledger.reconcile(
        {"root-1": EvidenceBundle((candidate("c1", immediate=(atom("s1", 400, 0, "e1"),)),), ())},
        today="2026-11-02",
    )
    before = ledger.expire_roots(["root-1"])
    assert before == ["root-1"]
    frozen = ledger.rows().digest
    ledger.reconcile({}, today="2026-11-03", exhaustive=True)
    assert ledger.rows().digest == frozen  # closed totals survive an exhaustive reconcile


# ---- a committed ledger blob of a foreign methodology is corruption, not data ---------------

NOW = scene.at(hour=20)


def seeded(root: Path) -> None:
    vs.enable(pin=True)
    vs.start_capture()
    vs.pair(vs.projects(root), 0, scene.at(hour=9))
    assert scene.report(scene.collect(NOW)).generation == 1
    assert scene.counter_total(scene.current()) == 2


def test_cross_namespace_ledger_reference_is_not_published(tmp_path: Path) -> None:
    seeded(tmp_path)
    checkpoint = scene.committed()
    original = checkpoint.to_dict()
    key = collector_state.ledger_blob_name(checkpoint.methodology)
    original_path = vs.accounting_dir() / "collector" / "state" / checkpoint.blobs[key]
    original_bytes = original_path.read_bytes()
    assert METHOD.hash != checkpoint.methodology
    checkpoint.blobs[key] = store.write_blob(
        PROFILE,
        key,
        {
            "state": ledger_state_to_dict(VersionedLedger(METHOD).state()),
            "superseded": {},
            "collection_scope": checkpoint.collection_scope,
        },
    )
    store.commit_checkpoint(PROFILE, checkpoint)
    before = scene.current()
    with pytest.raises(CollectError) as failed:
        scene.collect(NOW)
    assert failed.value.condition == "state_corrupt"
    assert scene.current() == before
    original_path.write_bytes(original_bytes)
    original_path.chmod(0o600)
    store.commit_checkpoint(PROFILE, store.Checkpoint.from_dict(original))
    assert scene.report(scene.collect(NOW)).generation == 2


# The collector idle cache cannot authorize a commit the reader never visited.


def test_append_between_reader_return_and_settlement_is_visited_next_cycle(tmp_path, monkeypatch):
    connection = opencode_connect(oc_db(tmp_path))
    try:
        opencode_add_session(connection, "ses_1")
        oc_add_call(connection, 1, READERS_T0 + 100)
        cycle = Cycle(tmp_path)
        cycle.drain()
        assert any(item.payload.get("call_id") == "call_1" for item in cycle.evidence)
        reader = cycle.readers["opencode"]
        original = reader.read
        appended = False

        def append_after_read(*args, **kwargs):
            nonlocal appended
            result = original(*args, **kwargs)
            if not appended:
                appended = True
                oc_add_call(connection, 3, READERS_T0 + 300)
            return result

        monkeypatch.setattr(reader, "read", append_after_read)
        # Force a legitimate new snapshot, then commit another row after it returns.
        oc_add_call(connection, 2, READERS_T0 + 200)
        first = cycle.run()
        assert appended and first.bytes_read > 0
        assert not any(item.payload.get("call_id") == "call_3" for item in first.evidence)
        cycle.drain()
        assert (
            sum(
                item.kind == "tool_call" and item.payload.get("call_id") == "call_3"
                for item in cycle.evidence
            )
            == 1
        )
        assert all(record.complete for record in cycle.known.values())
    finally:
        connection.close()


# Permanent affected-closure regressions, with real native input positive controls.


def seed(root: Path) -> dict[str, Path]:
    scene.install_key()
    from ssgrep.accounting.privacy.policy import enable_collection

    enable_collection(PROFILE)
    scene.journal_events(scene.start_marker())
    scene.collect(
        scene.at(hour=8, minute=5)
    )  # Establish live watermarks before native input exists.
    paths = {
        name: scene.write_transcript(
            root,
            name,
            scene.search_records(name, name + "-first", "query-" + name, (10, n, 0)),
            scene.at(),
        )
        for n, name in enumerate(("left", "right"))
    }
    first = scene.collect(scene.at())
    assert scene.report(first).new_evidence > 0
    assert len(current_ledger()[1].reduction.invocations) == 2
    return paths


def test_unchanged_cycle_never_hydrates_retained_native_content(tmp_path, monkeypatch):
    from ssgrep.accounting.collect.replay import EvidenceReplay

    seed(tmp_path)
    calls = []
    original = EvidenceReplay.hydrate_many

    def observed(self, items):
        items = list(items)
        calls.extend((item.runtime, item.session_id) for item in items)
        return original(self, items)

    monkeypatch.setattr(EvidenceReplay, "hydrate_many", observed)
    result = scene.collect(scene.at(minute=1))
    assert scene.report(result).complete
    assert calls == []
    assert len(current_ledger()[1].reduction.invocations) == 2


def test_one_native_source_changes_only_its_affected_root(tmp_path, monkeypatch):
    from ssgrep.accounting.collect.replay import EvidenceReplay

    paths = seed(tmp_path)
    calls = []
    original = EvidenceReplay.hydrate_many

    def observed(self, items):
        items = list(items)
        calls.extend(item.session_id for item in items)
        return original(self, items)

    monkeypatch.setattr(EvidenceReplay, "hydrate_many", observed)
    scene.append_transcript(
        paths["left"],
        scene.search_records("left", "left-next", "different-query", (11, 0, 0)),
        scene.at(hour=11),
    )
    result = scene.collect(scene.at(hour=11))
    assert scene.report(result).new_evidence > 0
    assert calls and set(calls) == {"left"}
    assert len(current_ledger()[1].reduction.invocations) == 3


def test_budgeted_reduction_backlog_is_durable_and_never_complete(tmp_path):
    from ssgrep.accounting.collect.store import load_partitions

    paths = seed(tmp_path)
    scene.append_transcript(
        paths["left"],
        scene.search_records("left", "left-next", "different-query", (11, 0, 0)),
        scene.at(hour=11),
    )
    result = scene.collect(scene.at(hour=11), budget=CycleBudget(seconds=0))
    assert not scene.report(result).complete
    assert result.state == "partial"
    # Reader backlog or a durable partition queue must explain every omitted update.
    checkpoint = scene.committed()
    partitions = load_partitions(PROFILE, checkpoint)
    assert partitions.pending or any(source.more for source in checkpoint.sources.values())
    resumed = scene.collect(scene.at(hour=11, minute=1))
    assert scene.report(resumed).complete
    assert len(current_ledger()[1].reduction.invocations) == 3


@pytest.mark.parametrize("operation", ["backfill", "recompute", "repin"])
def test_bounded_replacement_resumes_metadata_without_reseeding_completed_roots(
    tmp_path, monkeypatch, operation
):
    from ssgrep.accounting.collect import partitions

    seed(tmp_path)
    checkpoint = scene.committed()
    original = store.load_partitions(PROFILE, checkpoint)
    expected_parts = len(original.parts)
    original.metas = {}
    original.methodology = ""
    store.save_partitions(PROFILE, original, checkpoint.blobs)
    store.commit_checkpoint(PROFILE, checkpoint)
    counters_before = scene.observations(scene.current())["counters"]
    elapsed = [0.0]
    metadata_calls = []
    reduced = []
    graph_calls = []
    real_metadata = partitions.metadata
    real_components = partitions.components
    real_reducer = collector.default_reducer()

    def metadata(items, **kwargs):
        result = real_metadata(items, **kwargs)
        metadata_calls.append(tuple(sorted({item.session_id for item in items})))
        elapsed[0] += 1.5
        return result

    def reduce_profile(*args, **kwargs):
        result = real_reducer.reduce_profile(*args, **kwargs)
        reduced.append(tuple(result.bundle_roots))
        elapsed[0] += 1.5
        return result

    def components(*args):
        result = real_components(*args)
        graph_calls.append(1)
        elapsed[0] += 2.0
        return result

    def setup(step):
        if step == "started":
            elapsed[0] += 0.6

    monkeypatch.setattr(partitions, "metadata", metadata)
    monkeypatch.setattr(partitions, "components", components)
    options = CollectOptions(
        backfill=operation == "backfill",
        since="2026-10-12T00:00:00Z" if operation == "backfill" else None,
        until="2026-10-13T00:00:00Z" if operation == "backfill" else None,
        recompute=operation == "recompute",
        repin_frequency=operation == "repin",
        budget=CycleBudget(seconds=2),
    )
    observed_pending = []
    for minute in range(8):
        elapsed[0] = 0.0
        result = scene.collect(
            scene.at(hour=11, minute=minute),
            options,
            clock=lambda: elapsed[0],
            on_step=setup,
            reducer=collector.Reducer(reduce_profile, real_reducer.methodology_for),
        )
        accepted = store.load_partitions(PROFILE, scene.committed())
        observed_pending.append(len(accepted.pending))
        if scene.report(result).complete:
            break
    assert scene.report(result).complete, observed_pending
    assert len(metadata_calls) == 2
    assert len(reduced) == expected_parts
    assert accepted.pending == [] and accepted.metadata_pending == []
    if operation != "backfill":
        assert accepted.operation
    assert scene.observations(scene.current())["counters"] == counters_before
    archives = current_ledger()[0].archives_state()
    elapsed[0] = 0.0
    repeated = scene.collect(
        scene.at(hour=11, minute=9),
        options,
        clock=lambda: elapsed[0],
        on_step=setup,
        reducer=collector.Reducer(reduce_profile, real_reducer.methodology_for),
    )
    assert scene.report(repeated).complete
    assert current_ledger()[0].archives_state() == archives
    assert len(reduced) == expected_parts
    assert len(graph_calls) == 1


def test_reduction_without_available_budget_reports_actionable_block(tmp_path):
    from ssgrep.accounting.collect.store import load_partitions

    seed(tmp_path)
    options = CollectOptions(recompute=True, budget=CycleBudget(seconds=0))
    planned = scene.collect(scene.at(hour=11), options)
    assert scene.report(planned).stopped_by == "reduction_budget"
    assert load_partitions(PROFILE, scene.committed()).pending_groups
    stopped = scene.collect(scene.at(hour=11), options)
    report = scene.report(stopped)
    assert report.stopped_by == "reduction_blocked"
    assert report.resumable is False
    assert report.next_action == "increase_cycle_budget"
    assert report.configured_budget_seconds == 0
    assert report.required_budget_seconds is not None
    assert report.required_budget_seconds > report.configured_budget_seconds
    pending = load_partitions(PROFILE, scene.committed()).pending
    assert pending
    resumed = scene.collect(scene.at(hour=11, minute=1), CollectOptions(recompute=True))
    assert scene.report(resumed).complete
    assert load_partitions(PROFILE, scene.committed()).pending == []


def test_targeted_compact_reads_do_not_decode_unrelated_sessions(tmp_path):
    from tests.accounting.conftest import build_evidence

    left = build_evidence(session_id="left", origin_domain="left", record_id="l")
    right = build_evidence(session_id="right", origin_domain="right", record_id="r")
    refs = store.write_segments(PROFILE, [left, right])
    selected = list(store.iter_evidence(PROFILE, refs, sessions={("claude", "left")}))
    assert [item.session_id for item in selected] == ["left"]
    selected = list(store.iter_evidence(PROFILE, refs, native_keys={right.native_key}))
    assert [item.native_key for item in selected] == [right.native_key]


def test_partition_blob_references_survive_crash_recovery(tmp_path):
    from ssgrep.accounting.collect.store import load_partitions

    seed(tmp_path)
    checkpoint = scene.committed()
    before = load_partitions(PROFILE, checkpoint)
    store.sweep_orphans(PROFILE, checkpoint)
    after = load_partitions(PROFILE, checkpoint)
    assert before == after
    assert after.parts


@pytest.mark.parametrize(
    "damage",
    [
        {"pending": {}, "parts": []},
        {"metadata_pending": {}},
        {"metadata_pending": [False]},
        {"pending_groups": []},
        {"pending_groups": {"root": [False]}},
        {"operation": []},
        {"blocked_reason": "arbitrary_reason"},
    ],
)
def test_partition_state_fails_closed_on_invalid_accepted_queue(damage):
    from ssgrep.accounting.collect.store import PARTITIONS_BLOB, CollectError, load_partitions

    valid_checkpoint = store.Checkpoint(PROFILE)
    empty = load_partitions(PROFILE, valid_checkpoint)
    assert empty.pending == [] and empty.parts == {}

    filename = store.write_blob(
        PROFILE,
        PARTITIONS_BLOB,
        {"schema": 1, **damage},
    )
    damaged_checkpoint = store.Checkpoint(PROFILE, blobs={PARTITIONS_BLOB: filename})
    with pytest.raises(CollectError) as failure:
        load_partitions(PROFILE, damaged_checkpoint)
    assert failure.value.condition == "state_corrupt"


def test_unavailable_frequency_probe_uses_typed_state_not_integer_checkpoint():
    from ssgrep.accounting.collect.frequency import FrequencyView
    from ssgrep.accounting.collect.state import State, load_state, save_state

    probe = FrequencyView.unknown("index_absent")
    checkpoint = store.Checkpoint(PROFILE, frequency={"documents": 0})
    save_state(PROFILE, State(frequency_probe=probe), "test-namespace", checkpoint.blobs)
    restored = store.Checkpoint.from_dict(checkpoint.to_dict())
    assert restored.frequency == {"documents": 0}
    assert load_state(PROFILE, restored, "test-namespace").frequency_probe == probe


def current_ledger():
    from ssgrep.accounting.collect.state import VersionedLedger, load_state

    checkpoint = scene.committed()
    state = load_state(PROFILE, checkpoint, checkpoint.methodology)
    assert state.ledger is not None
    return VersionedLedger.restore_versioned(state.ledger, state.superseded, state.archives), state


def assert_full_replay_equal():
    from ssgrep.accounting.capture.fingerprint import key_id
    from ssgrep.accounting.capture.journal_reader import read_journal
    from ssgrep.accounting.collect.frequency import (
        FrequencyAssignments,
        FrequencyView,
        load_pinned,
        load_view,
    )
    from ssgrep.accounting.collect.read import Retained, load_fingerprint_key
    from ssgrep.accounting.collect.state import VersionedLedger
    from ssgrep.accounting.ledger.reduce import reduce_profile
    from ssgrep.accounting.privacy.policy import read_policy_file

    ledger, state = current_ledger()
    checkpoint = scene.committed()
    view = state.frequency_probe or load_pinned(PROFILE)
    assert isinstance(view, FrequencyView)
    key = load_fingerprint_key(PROFILE)
    policy = read_policy_file()
    helper = FrequencyAssignments(
        view,
        assignments=state.frequency_assignments,
        loader=lambda view_id: load_view(PROFILE, view_id),
        fingerprint_key_id=key_id(key),
    )
    reduced = reduce_profile(
        PROFILE,
        Retained(PROFILE, checkpoint.segments).items(),
        read_journal(PROFILE).events,
        policy=policy.profiles[PROFILE],
        ownership=policy,
        frequency=view,
        fingerprint_key=key,
        frequency_assignments=helper,
    )
    reference = VersionedLedger(ledger.methodology)
    reference.reconcile(
        dict(zip(reduced.bundle_roots, reduced.bundles, strict=True)),
        today=scene.at().date().isoformat(),
        exhaustive=True,
    )
    assert reference.rows().canonical_bytes() == ledger.rows().canonical_bytes()


def test_structured_result_refs_and_real_join_survive_partition_restart(tmp_path):
    from dataclasses import replace

    from ssgrep.accounting.capture.fingerprint import key_id
    from ssgrep.accounting.ledger.contracts import Attribution
    from ssgrep.accounting.privacy.policy import enable_collection
    from tests.accounting.ledger.reduce_builders import (
        CWD,
        KEY,
        card,
        claude_tool_result,
        claude_tool_use,
        results_json,
        source_session,
        ts,
    )

    enable_collection(PROFILE)
    scene.install_key()
    scene.journal_events(scene.start_marker())
    scene.collect(scene.at(hour=8, minute=5))
    ref = "claude:witness:ep:0"
    scene.write_transcript(
        tmp_path, "witness", source_session("witness", "E_TRACE_UNIQUE", when_day=12), scene.at()
    )
    scene.write_transcript(
        tmp_path,
        "consumer",
        [
            claude_tool_use(
                "consumer",
                "lookup",
                "Bash",
                {"command": "SSGREP_CORRELATION_ID=cor-refs ssgrep search deployment"},
                ts=ts(12, 11),
                cwd=CWD,
            ),
            {
                **claude_tool_result(
                    "consumer",
                    "lookup",
                    results_json(card(ref, "E_TRACE_UNIQUE")),
                    ts=ts(12, 11, 0, 2),
                ),
                "cwd": CWD,
            },
        ],
        scene.at(hour=11),
    )
    scene.journal_events(
        replace(event, fingerprint_key_id=key_id(KEY), scope_basis="process_cwd")
        for event in scene.execution(
            "e-refs",
            ts(12, 11),
            text="deployment",
            transport="cli",
            refs=(ref,),
            correlation="cor-refs",
        )
    )
    first = scene.collect(scene.at(hour=11, minute=1))
    assert scene.report(first).complete
    before, state = current_ledger()
    assert any(
        row.execution_id == "e-refs" and row.attribution is Attribution.EXPLICIT
        for row in state.reduction.executions
    )
    scene.collect(scene.at(hour=11, minute=2))  # Reload actual durable state, not a codec proxy.
    after, restored = current_ledger()
    assert after.rows().canonical_bytes() == before.rows().canonical_bytes()
    assert restored.reduction.invocations == state.reduction.invocations
    assert_full_replay_equal()


def test_source_witness_change_invalidates_its_dependent_partition(tmp_path, monkeypatch):
    from ssgrep.accounting.collect.replay import EvidenceReplay
    from ssgrep.accounting.privacy.policy import enable_collection
    from tests.accounting.ledger.reduce_builders import (
        card,
        claude_tool_result,
        claude_tool_use,
        results_json,
        source_session,
        ts,
    )

    enable_collection(PROFILE)
    scene.install_key()
    scene.journal_events(scene.start_marker())
    scene.collect(scene.at(hour=8, minute=5))
    source = scene.write_transcript(
        tmp_path, "witness", source_session("witness", "E_SOURCE_VALUE", when_day=12), scene.at()
    )
    scene.write_transcript(
        tmp_path,
        "dependent",
        [
            claude_tool_use(
                "dependent",
                "read",
                "mcp__acme_recall__search_sessions",
                {"query": "deployment"},
                ts=ts(12, 11),
            ),
            claude_tool_result(
                "dependent",
                "read",
                results_json(card("claude:witness:ep:0", "E_SOURCE_VALUE")),
                ts=ts(12, 11, 0, 2),
            ),
        ],
        scene.at(hour=11),
    )
    scene.collect(scene.at(hour=11))
    read = []
    original = EvidenceReplay.hydrate_many

    def observed(self, items):
        items = list(items)
        read.extend(row.session_id for row in items)
        return original(self, items)

    monkeypatch.setattr(EvidenceReplay, "hydrate_many", observed)
    scene.append_transcript(
        source,
        [
            claude_tool_use(
                "witness", "tail", "Bash", {"command": "echo continue"}, ts=ts(12, 10, 1)
            )
        ],
        scene.at(hour=11, minute=1),
    )
    scene.collect(scene.at(hour=11, minute=1))
    assert {"witness", "dependent"} <= set(read)
    assert_full_replay_equal()


def test_copied_native_history_does_not_duplicate_and_restart_matches_full_replay(tmp_path):
    paths = seed(tmp_path)
    original, _ = current_ledger()
    copied = paths["left"].with_name("left-copy.jsonl")
    copied.write_bytes(paths["left"].read_bytes())
    scene.collect(scene.at(minute=1))
    actual, _ = current_ledger()
    assert actual.rows().canonical_bytes() == original.rows().canonical_bytes()
    assert_full_replay_equal()


def test_late_parent_merges_the_old_root_and_survives_restart(tmp_path):
    from ssgrep.accounting.privacy.policy import enable_collection
    from tests.accounting.ledger.reduce_builders import (
        omp_header,
        omp_launch,
        omp_search,
        omp_user,
        results_json,
        ts,
    )
    from tests.accounting.readers.readers_fixtures import write_records

    enable_collection(PROFILE)
    scene.install_key()
    scene.journal_events(scene.start_marker())
    scene.collect(scene.at(hour=8, minute=5))
    parent = Path(os.environ["OMP_SESSIONS_DIR"]) / "parent-root.jsonl"
    child = parent.with_name("child-root.jsonl")
    write_records(
        child,
        [
            omp_header("child-root", ts(12, 10), parent=parent),
            *omp_search("lookup", "lookup", "deployment", results_json(), (12, 10, 0, 1)),
        ],
    )
    scene.collect(scene.at())
    write_records(
        parent,
        [
            omp_header("parent-root", ts(12, 9)),
            omp_user("turn", "delegate investigation", ts(12, 9)),
            *omp_launch("launch", "task", "child-root", ts(12, 9, 1), ts(12, 9, 1, 1)),
        ],
    )
    scene.collect(scene.at(minute=1))
    _, state = current_ledger()
    assert ("omp", "child-root", "omp:parent-root") in state.reduction.session_roots
    scene.collect(scene.at(minute=2))
    assert_full_replay_equal()


def test_clock_only_expiry_never_rehydrates_native_evidence(tmp_path, monkeypatch):
    from datetime import timedelta

    from ssgrep.accounting.collect.replay import EvidenceReplay

    seed(tmp_path)
    read = []
    original = EvidenceReplay.hydrate_many

    def observed(self, items):
        items = list(items)
        read.extend(items)
        return original(self, items)

    monkeypatch.setattr(EvidenceReplay, "hydrate_many", observed)
    scene.collect(scene.at() + timedelta(days=31))
    scene.collect(scene.at() + timedelta(days=31, minutes=1))
    assert read == []
    retention = scene.current().coverage["retention"]
    assert isinstance(retention, dict)
    assert retention["evidence_expired_before"]


def test_legacy_candidate_projection_rebuilds_from_accepted_partitions_without_native_io(
    tmp_path, monkeypatch
):
    from ssgrep.accounting.collect import state as collector_ledger
    from ssgrep.accounting.collect.replay import EvidenceReplay
    from ssgrep.accounting.collect.store import load_partitions, save_partitions
    from ssgrep.accounting.ledger.contracts import Scenario
    from ssgrep.accounting.ledger.reduce_cands import CHARS
    from ssgrep.accounting.ledger.revisions import unit_key
    from ssgrep.accounting.ledger.valuation_inputs import Eligibility
    from tests.accounting.collect import verify_scenes as vs
    from tests.accounting.ledger.reduce_builders import (
        MCP,
        card,
        claude_tool_result,
        claude_tool_use,
        results_json,
        ts,
    )
    from tests.accounting.views.test_ux_support import current_rarity_index

    current_rarity_index(tmp_path, monkeypatch)
    vs.enable(pin=True)
    vs.start_capture()
    vs.pair(vs.projects(tmp_path), 0, scene.at(hour=9))
    path = vs.projects(tmp_path) / "-work" / "cons0.jsonl"
    scene.append_transcript(
        path,
        [
            claude_tool_use("cons0", "repeat", MCP, {"query": vs.query(0)}, ts=ts(12, 10)),
            claude_tool_result(
                "cons0",
                "repeat",
                results_json(card("claude:src0:ep:0", vs.fact(0))),
                ts=ts(12, 10, 0, 2),
            ),
            claude_tool_use(
                "cons0",
                "repeat-action",
                "Bash",
                {"command": f"vim {vs.path(0)}"},
                ts=ts(12, 10, 0, 5),
            ),
            claude_tool_result("cons0", "repeat-action", "ok", ts=ts(12, 10, 0, 8)),
        ],
        scene.at(hour=10),
    )
    # Produce the historical collapse using its actual reconciliation key.
    with monkeypatch.context() as old_binary:
        old_binary.setattr(
            collector_ledger,
            "candidate_identity",
            lambda row: (row.task.root, row.task.turn or "", row.task.branch, row.candidate_id),
        )
        scene.collect(scene.at(hour=11))
    before, accepted = current_ledger()
    assert accepted.reduction is not None
    lookups = {row.call_id: row.invocation_id for row in accepted.reduction.invocations}
    first_lookup, repeat_lookup = lookups["t0a"], lookups["repeat"]
    held = before.state()
    assert held.candidates
    assert {row.exposure_ids[0] for row in held.candidates} <= {first_lookup, repeat_lookup}
    counters = scene.observations(scene.current())["counters"]
    immutable = {
        name: body for name, body in vs.snapshot_files().items() if name.startswith("gen-")
    }
    checkpoint = scene.committed()
    partitions = load_partitions(PROFILE, checkpoint)
    for part in partitions.parts.values():
        part.pop("candidate_identity_method", None)  # Real pre-cutover manifest shape.
    save_partitions(PROFILE, partitions, checkpoint.blobs)
    store.commit_checkpoint(PROFILE, checkpoint)

    def forbidden(*args, **kwargs):
        raise AssertionError("accepted partition reprojection must not hydrate native text")

    with monkeypatch.context() as blocked_native:
        blocked_native.setattr(EvidenceReplay, "hydrate_many", forbidden)
        stopped = scene.collect(scene.at(hour=11, minute=1), budget=CycleBudget(seconds=0))
        assert not scene.report(stopped).complete
        assert load_partitions(PROFILE, scene.committed()).pending_projections
        assert current_ledger()[0].state() == held
        resumed = scene.collect(scene.at(hour=11, minute=2))
        assert scene.report(resumed).complete
        assert scene.report(resumed).bytes_read == scene.report(resumed).new_evidence == 0
        after, _ = current_ledger()
        observations = {
            exposure: row for row in after.state().candidates for exposure in row.exposure_ids
        }
        assert set(observations) == {first_lookup, repeat_lookup}
        assert observations[first_lookup].eligibility is Eligibility.SUPPORTED
        assert observations[repeat_lookup].eligibility is Eligibility.KNOWN_REJECTED
        assert observations[repeat_lookup].eligibility_reasons == ("already_visible",)
        assert observations[first_lookup].candidate_id == observations[repeat_lookup].candidate_id
        assert (
            len({row.debit_id for row in after.state().debits if row.unit.unit == "context_tokens"})
            == 2
        )
        assert any(
            estimate.credit.value is not None and estimate.credit.value > 0
            for period in after.rows().periods
            for unit, estimates in period.estimates
            if unit == unit_key(CHARS)
            for estimate in estimates
            if estimate.scenario is Scenario.CENTRAL
        )
        assert scene.observations(scene.current())["counters"] == counters
        assert not load_partitions(PROFILE, scene.committed()).pending_projections
        frozen = after.rows().canonical_bytes()
        scene.collect(scene.at(hour=11, minute=3))
        assert current_ledger()[0].rows().canonical_bytes() == frozen
    assert_full_replay_equal()
    published = vs.snapshot_files()
    assert all(published[name] == body for name, body in immutable.items())


@pytest.mark.parametrize("selection", ["session", "identity"])
def test_legacy_and_indexed_selection_preserve_exact_native_authority(selection):
    from dataclasses import replace

    from tests.accounting.conftest import build_evidence

    items = [
        build_evidence(session_id=name, record_id=name, origin_domain=name, origin_id=name)
        for name in ("left", "right", "unrelated")
    ]
    (ref,) = store.write_segments(PROFILE, items)
    selected = items[1]
    sessions = {(selected.runtime, selected.session_id)} if selection == "session" else None
    native_keys = {selected.native_key} if selection == "identity" else None
    for evidence_ref in (ref, replace(ref, index=None)):
        actual = list(
            store.iter_evidence(PROFILE, [evidence_ref], sessions=sessions, native_keys=native_keys)
        )
        assert [(row.native_key, row.digest) for row in actual] == [
            (selected.native_key, selected.digest)
        ]
        absent_sessions = {("claude", "absent")} if selection == "session" else None
        absent_keys = {("claude", "absent", "absent")} if selection == "identity" else None
        assert (
            list(
                store.iter_evidence(
                    PROFILE, [evidence_ref], sessions=absent_sessions, native_keys=absent_keys
                )
            )
            == []
        )
        assert [row.native_key for row in store.iter_evidence(PROFILE, [evidence_ref])] == [
            item.native_key for item in items
        ]


def test_opencode_source_signature_includes_wal_changes(tmp_path):
    from ssgrep.accounting.collect.read import _source_signature
    from ssgrep.accounting.readers.base import Source
    from ssgrep.accounting.readers.opencode import OpenCodeReader

    database = tmp_path / "opencode.db"
    wal = tmp_path / "opencode.db-wal"
    database.write_bytes(b"database")
    wal.write_bytes(b"first wal state")
    reader = OpenCodeReader({})
    source = Source("opencode", database)

    first = _source_signature(reader, source)
    assert first is not None and first[1] is not None
    wal.write_bytes(b"updated wal state with appended frame")
    changed = _source_signature(reader, source)
    assert changed is not None and changed != first
    wal.unlink()
    assert _source_signature(reader, source) == [first[0], None]


def test_missing_evidence_directory_is_typed_state_corruption():
    import shutil

    from ssgrep.accounting.collect.store import CollectError
    from ssgrep.accounting.privacy import storage
    from tests.accounting.conftest import build_evidence

    item = build_evidence()
    (ref,) = store.write_segments(PROFILE, [item])
    assert [row.native_key for row in store.read_segment(PROFILE, ref)] == [item.native_key]
    evidence_dir = storage.data_root() / "accounting" / PROFILE / "collector" / store.EVIDENCE_DIR
    shutil.rmtree(evidence_dir)

    with pytest.raises(CollectError) as failure:
        list(store.read_segment(PROFILE, ref))
    assert failure.value.condition == "state_corrupt"


@pytest.mark.parametrize("path_kind", ["missing", "directory"])
def test_source_signature_never_accepts_a_missing_or_nonregular_primary_file(
    tmp_path: Path, path_kind: str
):
    from ssgrep.accounting.collect.read import _source_signature
    from ssgrep.accounting.readers.base import Source
    from ssgrep.accounting.readers.opencode import OpenCodeReader

    path = tmp_path / "opencode.db"
    if path_kind == "directory":
        path.mkdir()
    source = Source("opencode", path)
    assert _source_signature(OpenCodeReader({}), source) is None


@pytest.mark.parametrize("damage", ["missing", "size", "digest", "oversized"])
def test_selected_segment_damage_is_typed_and_does_not_publish_checkpoint(damage):
    from ssgrep.accounting.collect.store import CollectError
    from ssgrep.accounting.privacy import storage
    from tests.accounting.conftest import build_evidence

    item = build_evidence()
    (ref,) = store.write_segments(PROFILE, [item])
    checkpoint = store.Checkpoint(PROFILE, generation=7, segments=[ref])
    store.commit_checkpoint(PROFILE, checkpoint)
    root = storage.data_root() / "accounting" / PROFILE / "collector"
    accepted = (root / store.CHECKPOINT).read_bytes()
    path = root / store.EVIDENCE_DIR / ref.name
    original = path.read_bytes()
    assert [
        row.native_key
        for row in store.iter_evidence(PROFILE, [ref], sessions={(item.runtime, item.session_id)})
    ] == [item.native_key]
    if damage == "missing":
        path.unlink()
    elif damage == "size":
        path.write_bytes(original + b"\n")
    elif damage == "oversized":
        path.write_bytes(original + b"x" * (len(original) + 1))
    else:
        path.write_bytes(b" " + original[1:])
    with pytest.raises(CollectError) as failed:
        list(store.iter_evidence(PROFILE, [ref], sessions={(item.runtime, item.session_id)}))
    assert failed.value.condition == "state_corrupt"
    assert (root / store.CHECKPOINT).read_bytes() == accepted
    path.write_bytes(original)
    path.chmod(0o600)
    assert [
        row.native_key
        for row in store.iter_evidence(PROFILE, [ref], sessions={(item.runtime, item.session_id)})
    ] == [item.native_key]


@pytest.mark.parametrize(
    "damage", ["schema", "tables", "negative", "empty", "overflow", "session", "identity"]
)
def test_corrupt_selection_index_cannot_return_another_native_record(damage):
    from dataclasses import replace

    from ssgrep.accounting.collect.cycle import identity_hash, session_name
    from tests.accounting.conftest import build_evidence

    left = build_evidence(
        session_id="left", record_id="left", origin_domain="left", origin_id="left"
    )
    right = build_evidence(
        session_id="right", record_id="right", origin_domain="right", origin_id="right"
    )
    (ref,) = store.write_segments(PROFILE, [left, right])
    assert [
        row.native_key
        for row in store.iter_evidence(PROFILE, [ref], native_keys={right.native_key})
    ] == [right.native_key]
    raw = store.read_blob(PROFILE, ref.index)
    assert isinstance(raw, dict)
    left_name = session_name(("claude", "left"))
    right_name = session_name(("claude", "right"))
    sessions: set[tuple[str, str]] | None = {("claude", "right")}
    native_keys: set[tuple[str, str, str]] | None = None
    if damage == "schema":
        raw["schema"] = 999
    elif damage == "tables":
        raw["sessions"] = []
    elif damage == "session":
        raw["sessions"][right_name] = raw["sessions"][left_name]
    elif damage == "identity":
        source_hash = identity_hash(left.native_key)
        target_hash = identity_hash(right.native_key)
        source_table = store.read_blob(PROFILE, raw["identities"][source_hash[:2]])
        target_table = store.read_blob(PROFILE, raw["identities"][target_hash[:2]])
        assert isinstance(source_table, dict) and isinstance(target_table, dict)
        target_table[target_hash] = source_table[source_hash]
        raw["identities"][target_hash[:2]] = store.write_blob(
            PROFILE, "segment-identities", target_table
        )
        sessions, native_keys = None, {right.native_key}
    else:
        rows = store.read_blob(PROFILE, raw["sessions"][right_name])
        assert isinstance(rows, list)
        if damage == "negative":
            rows[0][0] = -1
        elif damage == "empty":
            rows[0][1] = 0
        else:
            rows[0][1] = ref.size + 1
        raw["sessions"][right_name] = store.write_blob(PROFILE, "segment-session", rows)
    damaged = replace(ref, index=store.write_blob(PROFILE, "segment-index", raw))
    with pytest.raises(CollectError) as failed:
        list(store.iter_evidence(PROFILE, [damaged], sessions=sessions, native_keys=native_keys))
    assert failed.value.condition == "state_corrupt"
    assert [
        row.native_key
        for row in store.iter_evidence(PROFILE, [ref], native_keys={right.native_key})
    ] == [right.native_key]


@pytest.mark.parametrize("damage", ["null", "integer", "string", "list", "boolean-schema"])
@pytest.mark.parametrize("consumer", ["load", "references"])
def test_partition_manifest_requires_object_and_exact_integer_schema(damage, consumer):
    from ssgrep.accounting.collect.store import CollectError
    from ssgrep.accounting.privacy import storage

    checkpoint = store.Checkpoint(PROFILE, generation=8)
    valid = store.PartitionState()
    store.save_partitions(PROFILE, valid, checkpoint.blobs)
    original = checkpoint.blobs[store.PARTITIONS_BLOB]
    assert store.load_partitions(PROFILE, checkpoint) == valid
    assert store.partition_references(PROFILE, original) == set()
    damaged = {
        "null": None,
        "integer": 7,
        "string": "not a partition manifest",
        "list": [],
        "boolean-schema": {**valid.to_dict(), "schema": True},
    }[damage]
    checkpoint.blobs[store.PARTITIONS_BLOB] = store.write_blob(
        PROFILE, store.PARTITIONS_BLOB, damaged
    )
    store.commit_checkpoint(PROFILE, checkpoint)
    path = storage.data_root() / "accounting" / PROFILE / "collector" / store.CHECKPOINT
    accepted = path.read_bytes()
    with pytest.raises(CollectError) as failed:
        if consumer == "load":
            store.load_partitions(PROFILE, checkpoint)
        else:
            store.partition_references(PROFILE, checkpoint.blobs[store.PARTITIONS_BLOB])
    assert failed.value.condition == "state_corrupt"
    assert path.read_bytes() == accepted
    assert store.load_checkpoint(PROFILE) == checkpoint
    checkpoint.blobs[store.PARTITIONS_BLOB] = original
    assert store.load_partitions(PROFILE, checkpoint) == valid
    assert store.partition_references(PROFILE, original) == set()


# Per-root persistence retains canonical rows and migrates legacy state resumably.


PROFILE = "personal-local"


def ledger_two_roots():
    ledger = VersionedLedger(METHOD)
    for root in ("root-1", "root-2"):
        task = TaskId(root, "turn", "main")
        ledger.reconcile(
            {root: EvidenceBundle((candidate(root, task=task),), (debit(root, 40, task=task),))},
            today="2026-11-02",
        )
    return ledger


def test_single_root_change_reuses_other_canonical_root_blob():
    ledger = ledger_two_roots()
    before = State(ledger=ledger.state(), superseded=ledger.superseded_state())
    blobs = {}
    collector_state.save_state(PROFILE, before, METHOD.hash, blobs)
    name = collector_state.ledger_blob_name(METHOD.hash)
    first = store.read_blob(PROFILE, blobs[name])
    ledger.reconcile(
        {
            "root-1": EvidenceBundle(
                (candidate("root-1", task=TaskId("root-1", "turn")),),
                (debit("root-1", 80, task=TaskId("root-1", "turn")),),
            )
        },
        today="2026-11-03",
    )
    after = State(ledger=ledger.state(), superseded=ledger.superseded_state())
    collector_state.save_state(PROFILE, after, METHOD.hash, blobs, previous=before)
    second = store.read_blob(PROFILE, blobs[name])
    assert first is not None and second is not None
    assert first["roots"]["root-2"]["blob"] == second["roots"]["root-2"]["blob"]
    assert first["roots"]["root-1"]["blob"] != second["roots"]["root-1"]["blob"]
    checkpoint = store.Checkpoint(PROFILE, blobs=blobs)
    restored = collector_state.load_state(PROFILE, checkpoint, METHOD.hash)
    assert restored.ledger is not None
    actual = VersionedLedger.restore_versioned(restored.ledger, restored.superseded)
    assert actual.rows().canonical_bytes() == ledger.rows().canonical_bytes()
    store.sweep_orphans(PROFILE, checkpoint)
    assert collector_state.load_state(PROFILE, checkpoint, METHOD.hash).ledger == restored.ledger


def test_legacy_root_conversion_is_budgeted_without_losing_closed_rows():
    from ssgrep.accounting.collect import state as collector_state
    from ssgrep.accounting.collect.state import (
        expand_ledger_document as expand_document,
        save_ledger_incremental as save_incremental,
    )

    ledger = ledger_two_roots()
    state = State(ledger=ledger.state())
    name = collector_state.ledger_blob_name(METHOD.hash)
    assert state.ledger is not None
    blobs = {
        name: store.write_blob(
            PROFILE,
            name,
            {"state": ledger_state_to_dict(state.ledger), "superseded": {}, "archives": {}},
        )
    }
    remaining = save_incremental(PROFILE, name, state, blobs, previous=state, migration_limit=1)
    assert remaining == 1
    raw = store.read_blob(PROFILE, blobs[name])
    assert raw is not None
    assert raw["base_blob"] and raw["migration_pending_roots"] == 1
    expanded = expand_document(PROFILE, raw)
    assert collector_state.decode_ledger(expanded)[0] == state.ledger
    remaining = save_incremental(PROFILE, name, state, blobs, previous=state, migration_limit=1)
    assert remaining == 0
    final = store.read_blob(PROFILE, blobs[name])
    assert final is not None and final["base_blob"] is None
    restored = collector_state.decode_ledger(
        store.read_blob(PROFILE, blobs[name]), profile=PROFILE
    )[0]
    assert restored == state.ledger


def split_ledger_blobs():
    """Two committed canonical roots, and the manifest + names that reference them."""
    state = State(ledger=ledger_two_roots().state())
    blobs: dict[str, str] = {}
    collector_state.save_state(PROFILE, state, METHOD.hash, blobs)
    name = collector_state.ledger_blob_name(METHOD.hash)
    raw = store.read_blob(PROFILE, blobs[name])
    assert raw is not None
    assert raw["format"] == collector_state.LEDGER_FORMAT and set(raw["roots"]) == {
        "root-1",
        "root-2",
    }
    return state, raw


def test_ledger_manifest_expansion_rejects_inconsistent_chains_and_roots() -> None:
    from ssgrep.accounting.collect.state import expand_ledger_document

    state, raw = split_ledger_blobs()
    whole = expand_ledger_document(PROFILE, raw)
    assert {row["task"]["root"] for row in whole["state"]["candidates"]} == {"root-1", "root-2"}

    def condition(manifest) -> str:
        with pytest.raises(CollectError) as failed:
            expand_ledger_document(PROFILE, manifest)
        return failed.value.condition

    assert condition({**raw, "format": "root-ledger-v9"}) == "unsupported_schema"

    # A tombstoned root contributes nothing; its sibling is untouched.
    dropped = expand_ledger_document(PROFILE, {**raw, "roots": {**raw["roots"], "root-1": None}})
    assert {row["task"]["root"] for row in dropped["state"]["candidates"]} == {"root-2"}

    first = store.write_blob(PROFILE, "ledger-base", raw)
    second = store.write_blob(PROFILE, "ledger-base", {**raw, "base_blob": first})
    assert condition({**raw, "base_blob": second}) == "state_corrupt"  # base chain too deep

    child = store.read_blob(PROFILE, raw["roots"]["root-1"]["blob"])
    assert child is not None
    entry = raw["roots"]["root-1"]

    def with_child(document) -> dict:
        blob = store.write_blob(PROFILE, "ledger-root", document)
        return {**raw, "roots": {**raw["roots"], "root-1": {**entry, "blob": blob}}}

    assert condition(with_child(["not", "a", "root document"])) == "state_corrupt"
    foreign_method = {**child, "state": {**child["state"], "methodology": {"label": "other"}}}
    assert condition(with_child(foreign_method)) == "state_corrupt"
    stolen = {
        **child,
        "state": {
            **child["state"],
            "candidates": [
                {**row, "task": {**row["task"], "root": "root-2"}}
                for row in child["state"]["candidates"]
            ],
        },
    }
    assert child["state"]["candidates"] and condition(with_child(stolen)) == "state_corrupt"
    assert state.ledger is not None


def test_ledger_supersession_marks_rewrite_every_root_and_migration_honours_its_deadline() -> None:
    from ssgrep.accounting.collect.state import (
        expand_ledger_document,
        ledger_blob_name,
        save_ledger_incremental,
    )

    ledger = ledger_two_roots()
    state = State(ledger=ledger.state())
    name = ledger_blob_name(METHOD.hash)
    blobs: dict[str, str] = {}
    save_ledger_incremental(PROFILE, name, state, blobs)
    assert state.ledger is not None
    assert (
        expand_ledger_document(PROFILE, store.read_blob(PROFILE, blobs[name]))["superseded"] == {}
    )

    [victim] = [row for row in state.ledger.candidates if row.task.root == "root-2"]
    marked = State(ledger=state.ledger, superseded={digest_of(victim): "2026-11-03"})
    save_ledger_incremental(PROFILE, name, marked, blobs, previous=state)
    restored = collector_state.load_state(
        PROFILE, store.Checkpoint(PROFILE, blobs=blobs), METHOD.hash
    )
    assert restored.superseded == {digest_of(victim): "2026-11-03"}
    assert restored.ledger == state.ledger

    # A converting write that is already out of time finishes no root, but loses none.
    legacy_blobs = {
        name: store.write_blob(
            PROFILE,
            name,
            {"state": ledger_state_to_dict(state.ledger), "superseded": {}, "archives": {}},
        )
    }
    pending = save_ledger_incremental(
        PROFILE, name, state, legacy_blobs, previous=state, clock=lambda: 1.0, deadline=0.0
    )
    assert pending == 2
    manifest = store.read_blob(PROFILE, legacy_blobs[name])
    assert manifest is not None
    assert manifest["migration_pending_roots"] == 2 and manifest["roots"] == {}
    assert expand_ledger_document(PROFILE, manifest)["state"]["candidates"]
    assert (
        save_ledger_incremental(
            PROFILE, name, state, legacy_blobs, previous=state, migration_limit=2
        )
        == 0
    )


@pytest.mark.parametrize("corrupt", [["not", "a", "ledger"], "text", 7])
def test_incremental_ledger_save_refuses_a_non_object_committed_base_without_replacing_it(
    corrupt,
) -> None:
    from ssgrep.accounting.collect.state import ledger_blob_name, save_ledger_incremental

    state = State(ledger=ledger_two_roots().state())
    name = ledger_blob_name(METHOD.hash)
    blobs: dict[str, str] = {}
    save_ledger_incremental(PROFILE, name, state, blobs)
    save_ledger_incremental(PROFILE, name, state, blobs, previous=state)  # valid base: accepted

    damaged = {name: store.write_blob(PROFILE, name, corrupt)}
    with pytest.raises(store.CollectError) as refused:
        save_ledger_incremental(PROFILE, name, state, damaged, previous=state)

    assert refused.value.condition == "state_corrupt"
    assert store.read_blob(PROFILE, damaged[name]) == corrupt
    assert set(damaged) == {name}


def test_assignment_and_probe_documents_reject_unsupported_or_inconsistent_state() -> None:
    from ssgrep.accounting.collect.state import (
        ASSIGNMENT_FORMAT,
        FREQUENCIES,
        State,
        expand_assignments_document,
        load_state,
        save_state,
    )
    from ssgrep.accounting.ledger.ledger_contracts import FrequencyAssignment
    from tests.accounting.ledger.reduce_builders import rare_view

    def condition(call) -> str:
        with pytest.raises(CollectError) as failed:
            call()
        return failed.value.condition

    assert condition(
        lambda: expand_assignments_document(PROFILE, {"format": "assignments-v9"})
    ) == ("unsupported_schema")
    shard = store.write_blob(PROFILE, "assignment-shard", {"not": "a list"})
    assert (
        condition(
            lambda: expand_assignments_document(
                PROFILE, {"format": ASSIGNMENT_FORMAT, "shards": {"aa": shard}}
            )
        )
        == "state_corrupt"
    )

    row = to_plain(FrequencyAssignment("invocation-1", "view-1"))

    def loaded(document) -> State:
        reference = store.write_blob(PROFILE, FREQUENCIES, document)
        return load_state(PROFILE, store.Checkpoint(PROFILE, blobs={FREQUENCIES: reference}), "ns")

    assert loaded({"schema": 1, "assignments": [row]}).frequency_assignments == (
        FrequencyAssignment("invocation-1", "view-1"),
    )
    for damaged in (
        {"schema": 2, "assignments": [row]},
        {"schema": 1, "assignments": [row, row]},
    ):
        assert condition(lambda damaged=damaged: loaded(damaged)) == "state_corrupt"

    blobs: dict[str, str] = {}
    save_state(PROFILE, State(frequency_probe=rare_view("needle", corpus=3)), "ns", blobs)
    assert (
        condition(lambda: load_state(PROFILE, store.Checkpoint(PROFILE, blobs=blobs), "ns"))
        == "state_corrupt"
    )


def test_retired_methodology_rows_are_frozen_and_must_match_their_namespace() -> None:
    from ssgrep.accounting.collect.state import RETIRED_ARCHIVE_PREFIX
    from ssgrep.accounting.ledger.valuation_inputs import RETIRED_VALUATION_VERSION

    def populated(method) -> VersionedLedger:
        ledger = VersionedLedger(method)
        task = TaskId("root-1", "turn", "main")
        ledger.reconcile(
            {"root-1": EvidenceBundle((candidate("a", task=task),), (debit("x", 40, task=task),))},
            today="2026-11-02",
        )
        return ledger

    current = populated(METHOD)
    assert current.rows().periods
    current.expire_roots(["root-1"])
    frozen_state = current.state()
    assert frozen_state.frozen and not frozen_state.candidates
    retired_method = replace(METHOD, valuation_version=RETIRED_VALUATION_VERSION)
    retired = VersionedLedger.restore_versioned(
        replace(frozen_state, methodology=retired_method), {}
    )
    assert not retired.archives
    frozen_rows = retired.rows()
    assert frozen_rows.periods and not any(row.recomputable for row in frozen_rows.periods)
    assert [row.period for row in frozen_rows.periods] == [
        row.period for row in current.rows().periods
    ]
    # A retired methodology cannot be re-valued: only its archived rows are readable.
    live_retired = populated(METHOD)
    live_retired.methodology = retired_method
    with pytest.raises(ValueError, match="methodology_read_only"):
        live_retired.rows()

    archive = frozen_rows
    retired.archives[RETIRED_ARCHIVE_PREFIX + retired.namespace] = collector_state.ArchivedRows(
        "retired", archive.namespace, archive.periods, archive.receipts
    )
    assert retired.rows().periods == tuple(
        replace(row, recomputable=False) for row in archive.periods
    )
    for ledger, namespace in (
        (retired, "another-namespace"),
        (populated(METHOD), populated(METHOD).namespace),
    ):
        ledger.archives[RETIRED_ARCHIVE_PREFIX + ledger.namespace] = collector_state.ArchivedRows(
            "retired", namespace, (), ()
        )
        with pytest.raises(ValueError, match="retired_methodology_namespace_mismatch"):
            ledger.rows()


def test_a_named_replacement_archive_needs_a_name_and_is_taken_once() -> None:
    ledger = ledger_two_roots()
    with pytest.raises(ValueError, match="nonempty revision name"):
        ledger.archive_current("")
    ledger.archive_current("replacement-1")
    frozen = ledger.archives["replacement-1"]
    assert frozen.periods and frozen.namespace == ledger.namespace
    task = TaskId("root-3", "turn", "main")
    ledger.reconcile(
        {"root-3": EvidenceBundle((candidate("c3", task=task),), (debit("c3", 10, task=task),))},
        today="2026-11-03",
    )
    ledger.archive_current("replacement-1")
    assert ledger.archives["replacement-1"] is frozen


# Original-publication maintenance must expose backlog, never replay all children.


PROFILE = "personal-local"
ARCHIVE = "identity-archive-native_identity_v2"


def retention_prepare(
    checkpoint, budget=1024 * 1024, evidence_before_day="2026-11-01", before_day="2026-10-01"
):
    return maintenance.prepare(
        PROFILE,
        checkpoint,
        rows=None,
        before_day=before_day,
        evidence_before_day=evidence_before_day,
        clock=lambda: 0,
        deadline=1,
        byte_budget=budget,
    )


prepare = retention_prepare


def test_oversized_original_manifest_is_pending_without_decoding_it(monkeypatch):
    reference = store.write_blob(PROFILE, ARCHIVE, {"payload": "x" * 256})
    checkpoint = store.Checkpoint(PROFILE, blobs={ARCHIVE: reference})

    def forbidden(*args):
        raise AssertionError(
            "An oversized child must not be read outside its explicit maintenance budget"
        )

    monkeypatch.setattr(store, "read_blob", forbidden)
    work = retention_prepare(checkpoint, budget=1)
    assert work.pending and not work.changed and not work.delete_receipts
    assert checkpoint.retention[maintenance.KEY]["blocked_child_bytes"] > 1


def test_unchanged_accepted_archive_proof_skips_all_owned_child_reads_after_restart(monkeypatch):
    reference = store.write_blob(PROFILE, ARCHIVE, {"schema": 1, "blobs": {}})
    checkpoint = store.Checkpoint(PROFILE, blobs={ARCHIVE: reference})
    assert not prepare(checkpoint).pending
    restarted = store.Checkpoint.from_dict(checkpoint.to_dict())

    def forbidden(*args):
        raise AssertionError("An unchanged completed proof cannot hide work in a replay hydrator")

    monkeypatch.setattr(store, "read_blob", forbidden)
    assert not retention_prepare(restarted).pending


def test_maintenance_quantum_selects_only_one_original_data_child(monkeypatch):
    from ssgrep.accounting.privacy import retention as published_retention

    a = store.write_blob(PROFILE, "original-a", {"retained": "a"})
    b = store.write_blob(PROFILE, "original-b", {"retained": "b"})
    reference = store.write_blob(
        PROFILE,
        ARCHIVE,
        {
            "schema": 1,
            "blobs": {"a": a, "b": b},
            "segments": [],
            "published_documents": {
                "format": "published-session-v1",
                "history_blob": None,
                "receipt_blobs": {},
            },
        },
    )
    checkpoint = store.Checkpoint(PROFILE, blobs={ARCHIVE: reference})
    selections = []

    def selected(*args, **kwargs):
        selections.append(kwargs["blob_names"])
        return published_retention.ChildResult(False, (), kwargs["manifest"])

    monkeypatch.setattr(published_retention, "stage_children", selected)
    first = prepare(checkpoint)
    assert first.pending and selections == [("a",)]
    restarted = store.Checkpoint.from_dict(checkpoint.to_dict())
    assert not retention_prepare(restarted).pending
    assert selections == [("a",), ("b",)]


def test_public_receipt_cleanup_is_a_bounded_postcommit_obligation(monkeypatch):
    import os

    from ssgrep.accounting.privacy import purge_files, storage

    reference = store.write_blob(PROFILE, ARCHIVE, {"schema": 1, "blobs": {}})
    checkpoint = store.Checkpoint(PROFILE, blobs={ARCHIVE: reference})
    assert not prepare(checkpoint).pending
    names = [f"r{i:03}" for i in range(32)]
    directory = storage.open_profile_dir(PROFILE, "receipts")
    try:
        for name in names:
            storage.atomic_write(directory, name + ".json", b"{}\\n")
    finally:
        os.close(directory)
    checkpoint.retention[maintenance.KEY]["cleanup_ids"] = names

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "A <=32-ID cleanup intent must not enumerate all receipt IDs or read archive bodies"
        )

    monkeypatch.setattr(purge_files, "receipt_ids", forbidden)
    monkeypatch.setattr(store, "read_blob", forbidden)
    work = prepare(checkpoint)
    assert len(work.delete_receipts) == 32
    store.commit_checkpoint(PROFILE, checkpoint)  # Durable intent precedes any unlink.
    restarted = store.load_checkpoint(PROFILE)
    assert restarted is not None
    assert retention_prepare(restarted).delete_receipts == work.delete_receipts
    assert maintenance.finish(PROFILE, work) == 32
    assert not retention_prepare(restarted).delete_receipts
    assert restarted.retention[maintenance.KEY]["cleanup_ids"] == []


def test_counter_linkage_uses_evidence_floor_without_aging_publications_at_that_floor(monkeypatch):
    from ssgrep.accounting.privacy import retention as published_retention

    counter = store.write_blob(
        PROFILE, "original-counter", {"counted": [["e|completed", "2026-10-20"]]}
    )
    ledger = store.write_blob(PROFILE, "original-ledger", {"retained": "unchanged quantities"})
    calls = []

    def selected(*args, **kwargs):
        calls.append((kwargs["blob_names"], kwargs["before_day"]))
        return published_retention.ChildResult(False, (), kwargs["manifest"])

    monkeypatch.setattr(published_retention, "stage_children", selected)
    for counter_name in ("counters", "counters-scope-previous"):
        reference = store.write_blob(
            PROFILE,
            ARCHIVE,
            {
                "schema": 1,
                "blobs": {counter_name: counter, "ledger-old": ledger},
                "segments": [],
                "published_documents": {
                    "format": "published-session-v1",
                    "history_blob": None,
                    "receipt_blobs": {},
                },
            },
        )
        checkpoint = store.Checkpoint(PROFILE, blobs={ARCHIVE: reference})
        assert prepare(checkpoint).pending
        assert calls[-1] == ((counter_name,), "2026-11-01")
        assert not prepare(checkpoint).pending
        assert calls[-1] == (("ledger-old",), "2026-10-01")
        # A changed evidence horizon invalidates the completed counter proof even
        # when receipt retention and every archive pointer are unchanged.
        assert retention_prepare(checkpoint, evidence_before_day="2026-11-02").pending
        assert calls[-1] == ((counter_name,), "2026-11-02")


def test_future_counter_is_indexed_once_across_days_then_only_expired_linkage_changes(monkeypatch):
    counter = {
        "epoch": 7,
        "epoch_started_at": "2026-09-01T00:00:00Z",
        "counted": [["future|invoked", "2026-12-01"]],
        "executions": [],
        "rendered_bytes": [],
        "durations": [],
    }
    reference = store.write_blob(PROFILE, "counters", counter)
    manifest = {
        "schema": 1,
        "blobs": {"counters": reference},
        "segments": [],
        "published_documents": {
            "format": "published-session-v1",
            "history_blob": None,
            "receipt_blobs": {},
        },
    }
    checkpoint = store.Checkpoint(
        PROFILE, blobs={ARCHIVE: store.write_blob(PROFILE, ARCHIVE, manifest)}
    )
    original_read, reads = store.read_blob, []

    def observed(profile, filename):
        if filename == reference:
            reads.append(filename)
        return original_read(profile, filename)

    monkeypatch.setattr(store, "read_blob", observed)
    assert not prepare(checkpoint).pending  # Empty phases finish in this same quantum.
    assert reads == [reference]
    assert not retention_prepare(
        checkpoint, evidence_before_day="2026-11-02", before_day="2026-10-02"
    ).pending
    assert not retention_prepare(
        checkpoint, evidence_before_day="2026-12-01", before_day="2026-10-03"
    ).pending
    assert reads == [reference]  # Equality is retained, so the minimum day still proves clean.
    assert not retention_prepare(
        checkpoint, evidence_before_day="2026-12-02", before_day="2026-10-04"
    ).pending
    assert reads == [reference, reference]
    after = original_read(PROFILE, checkpoint.blobs[ARCHIVE])
    assert after is not None
    expired = original_read(PROFILE, after["blobs"]["counters"])
    assert expired == {**counter, "counted": []}

    def forbidden(profile, filename):
        if filename == after["blobs"]["counters"]:
            raise AssertionError("A known-empty linkage child must not be reread on a later day")
        return original_read(profile, filename)

    monkeypatch.setattr(store, "read_blob", forbidden)
    assert not retention_prepare(
        checkpoint, evidence_before_day="2026-12-03", before_day="2026-10-05"
    ).pending


def test_exact_changed_child_reference_invalidates_its_cached_empty_date_proof(monkeypatch):
    original = {"epoch": 2, "counted": []}
    reference = store.write_blob(PROFILE, "counters", original)
    manifest = {
        "schema": 1,
        "blobs": {"counters": reference},
        "segments": [],
        "published_documents": {
            "format": "published-session-v1",
            "history_blob": None,
            "receipt_blobs": {},
        },
    }
    checkpoint = store.Checkpoint(
        PROFILE, blobs={ARCHIVE: store.write_blob(PROFILE, ARCHIVE, manifest)}
    )
    assert not prepare(checkpoint).pending
    indexed = store.read_blob(PROFILE, checkpoint.blobs[ARCHIVE])
    assert indexed is not None
    replacement = store.write_blob(
        PROFILE, "counters", {**original, "counted": [["late|rendered", "2026-10-01"]]}
    )
    indexed["blobs"]["counters"] = (
        replacement  # Keep the stale index deliberately; only its source ref differs.
    )
    checkpoint.blobs[ARCHIVE] = store.write_blob(PROFILE, ARCHIVE, indexed)
    original_read, reads = store.read_blob, []

    def observed(profile, filename):
        if filename == replacement:
            reads.append(filename)
        return original_read(profile, filename)

    monkeypatch.setattr(store, "read_blob", observed)
    assert not prepare(checkpoint).pending
    assert reads == [replacement]
    after = original_read(PROFILE, checkpoint.blobs[ARCHIVE])
    assert after is not None
    assert original_read(PROFILE, after["blobs"]["counters"]) == original


def test_ledger_reduction_segment_history_and_receipt_indexes_skip_clean_future_days(monkeypatch):
    from collections import Counter

    from tests.accounting.conftest import build_evidence
    from tests.accounting.privacy.test_published_archive_purge import seed

    checkpoint, _, manifest = seed()
    reduction = store.write_blob(
        PROFILE,
        "reduction-original",
        {
            "executions": [
                {
                    "execution_id": "old",
                    "day": "2026-10-13",
                    "runtime": "claude",
                    "session_id": "alpha",
                }
            ],
            "invocations": [],
            "session_roots": [["claude", "alpha", "claude:alpha"]],
            "session_last": [["claude", "alpha", "2026-10-13T10:00:00Z"]],
        },
    )
    manifest["blobs"]["reduction"] = reduction
    segments = store.write_segments(PROFILE, [build_evidence(ts_utc="2026-10-13T10:00:00Z")])
    manifest["segments"] = [ref.to_dict() for ref in segments]
    checkpoint.blobs[ARCHIVE] = store.write_blob(PROFILE, ARCHIVE, manifest)
    data_refs = {
        manifest["blobs"]["ledger-original"],
        reduction,
        manifest["published_documents"]["history_blob"],
        *manifest["published_documents"]["receipt_blobs"].values(),
    }
    original_blob, original_segment = store.read_blob, store.read_segment
    bodies, compact_reads = [], []

    def blob(profile, filename):
        if filename in data_refs:
            bodies.append(filename)
        return original_blob(profile, filename)

    def segment(profile, ref):
        compact_reads.append(ref.name)
        yield from original_segment(profile, ref)

    monkeypatch.setattr(store, "read_blob", blob)
    monkeypatch.setattr(store, "read_segment", segment)
    for _ in range(8):
        if not prepare(checkpoint).pending:
            break
    else:
        raise AssertionError("Finite selected-child indexing backlog did not drain")
    counts = Counter(bodies)
    assert all(counts[ref] == 1 for ref in data_refs)
    assert compact_reads == [segments[0].name]
    before = (list(bodies), list(compact_reads))
    assert not retention_prepare(
        checkpoint, before_day="2026-10-02", evidence_before_day="2026-11-02"
    ).pending
    assert not retention_prepare(
        checkpoint, before_day="2026-10-13", evidence_before_day="2026-11-03"
    ).pending
    assert (
        bodies,
        compact_reads,
    ) == before  # Inclusive minimum is still future under exclusive expiry.
    # Only one newly eligible child is processed; other expired children remain
    # explicit backlog rather than being reread/reduced together.
    assert retention_prepare(
        checkpoint, before_day="2026-10-14", evidence_before_day="2026-11-04"
    ).pending
    assert bodies[-1] == manifest["blobs"]["ledger-original"]
    assert len(bodies) == len(before[0]) + 1 and compact_reads == before[1]


def test_empty_archive_metadata_edges_are_bounded_and_resumable_without_fake_completion():
    references = {}
    for number in range(70):
        name = ARCHIVE + f"-{number:03}"
        references[name] = store.write_blob(PROFILE, name, {"schema": 1, "blobs": {}})
    checkpoint = store.Checkpoint(PROFILE, blobs=references)
    first = prepare(checkpoint)
    assert first.pending
    job = checkpoint.retention[maintenance.KEY]
    assert job["metadata_edges"] == 64 and job["archive_after"] == ARCHIVE + "-063"
    restarted = store.Checkpoint.from_dict(checkpoint.to_dict())
    assert not retention_prepare(restarted).pending
    assert restarted.retention[maintenance.KEY]["metadata_edges"] == 6


def test_legacy_segment_index_budget_defers_then_publishes_queryable_identity_ranges(
    tmp_path, monkeypatch
):
    import os
    from dataclasses import replace

    from ssgrep.accounting.collect import steps as collector_maintenance
    from ssgrep.accounting.privacy import storage
    from ssgrep.accounting.readers.native_identity import with_origin
    from tests.accounting.ledger.test_minimal_evidence import CANARY, evidence

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    items = [
        with_origin(
            replace(evidence(), session_id=f"session-{number}", record_id=f"call:{number}"),
            f"claude:session-{number}",
            f"call:{number}",
            "native",
        )
        for number in range(2)
    ]
    body = b"".join(evidence_line(item) + b"\n" for item in items)
    directory = storage.open_profile_dir(PROFILE, "collector", "evidence")
    try:
        storage.write_once(directory, "seg-legacy-index.jsonl", body)
    finally:
        os.close(directory)
    legacy = store.SegmentRef("seg-legacy-index.jsonl", len(items), len(body), "backfill")
    checkpoint = store.Checkpoint(PROFILE, generation=11, segments=[legacy])
    store.commit_checkpoint(PROFILE, checkpoint)
    root = storage.data_root() / "accounting" / PROFILE / "collector"
    checkpoint_bytes = (root / store.CHECKPOINT).read_bytes()
    before = {
        path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }
    kwargs = {"clock": lambda: 0, "deadline": 1, "should_stop": lambda: False}
    used, pending = collector_maintenance.ensure_indexes(
        PROFILE, checkpoint, byte_budget=0, **kwargs
    )
    assert used == 0 and pending
    assert checkpoint.segments == [legacy]
    assert checkpoint.retention["evidence_index_pending"] == [legacy.name]
    assert {
        path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()
    } == before
    used, pending = collector_maintenance.ensure_indexes(
        PROFILE, checkpoint, byte_budget=len(body), **kwargs
    )
    assert used == len(body) and not pending
    assert "evidence_index_pending" not in checkpoint.retention
    (indexed,) = checkpoint.segments
    assert indexed.index is not None and indexed.name == legacy.name
    assert indexed.origin == "backfill"
    assert (root / "evidence" / legacy.name).read_bytes() == body
    assert (root / store.CHECKPOINT).read_bytes() == checkpoint_bytes
    selected = list(store.iter_evidence(PROFILE, [indexed], native_keys={items[0].native_key}))
    assert [(item.native_key, item.digest) for item in selected] == [
        (items[0].native_key, items[0].digest)
    ]
    selected = list(store.iter_evidence(PROFILE, [indexed], sessions={("claude", "session-1")}))
    assert [(item.native_key, item.digest) for item in selected] == [
        (items[1].native_key, items[1].digest)
    ]
    assert CANARY.encode() not in b"".join(
        path.read_bytes() for path in root.rglob("*") if path.is_file()
    )
    store.commit_checkpoint(PROFILE, checkpoint)
    restarted = store.load_checkpoint(PROFILE)
    assert restarted is not None
    assert restarted.segments == [indexed]
    assert (
        list(store.iter_evidence(PROFILE, restarted.segments))[0].native_key == items[0].native_key
    )


@pytest.mark.parametrize("damage", ["missing", "truncated", "oversized", "unterminated"])
def test_damaged_legacy_index_input_is_typed_corruption_without_checkpoint_publication(
    tmp_path, monkeypatch, damage
):
    import os

    import pytest

    from ssgrep.accounting.collect import steps as collector_maintenance
    from ssgrep.accounting.collect.store import CollectError
    from ssgrep.accounting.privacy import storage
    from tests.accounting.ledger.test_minimal_evidence import evidence

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    body = evidence_line(evidence()) + b"\n"
    directory = storage.open_profile_dir(PROFILE, "collector", "evidence")
    try:
        if damage != "missing":
            damaged = {
                "truncated": body[:-2],
                "oversized": body + b"  ",
                "unterminated": body[:-1],
            }[damage]
            storage.write_once(directory, "seg-damaged.jsonl", damaged)
    finally:
        os.close(directory)
    checkpoint = store.Checkpoint(
        PROFILE,
        generation=12,
        segments=[
            store.SegmentRef(
                "seg-damaged.jsonl",
                1,
                len(body) - 1 if damage == "unterminated" else len(body),
            )
        ],
    )
    store.commit_checkpoint(PROFILE, checkpoint)
    root = storage.data_root() / "accounting" / PROFILE / "collector"
    before = {
        path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }
    with pytest.raises(CollectError) as failed:
        collector_maintenance.ensure_indexes(
            PROFILE,
            checkpoint,
            clock=lambda: 0,
            deadline=1,
            byte_budget=len(body),
            should_stop=lambda: False,
        )
    assert failed.value.condition == "state_corrupt"
    assert {
        path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()
    } == before
    assert store.load_checkpoint(PROFILE) == checkpoint
    assert checkpoint.segments[0].index is None


def test_incremental_compaction_stops_before_second_merge_and_resumes_without_record_loss(
    tmp_path, monkeypatch
):
    from dataclasses import replace

    from ssgrep.accounting.collect import steps as collector_maintenance
    from ssgrep.accounting.privacy import storage
    from ssgrep.accounting.readers.native_identity import with_origin
    from tests.accounting.ledger.test_minimal_evidence import evidence

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    items = [
        with_origin(
            replace(evidence(), record_id=f"call:{number}"),
            "claude:compaction",
            f"call:{number}",
            "native",
        )
        for number in range(4)
    ]
    refs = [store.write_segments(PROFILE, [item], origin="backfill")[0] for item in items]
    monkeypatch.setattr(store, "MAX_SEGMENTS", 2)
    monkeypatch.setattr(store, "SEGMENT_TARGET_BYTES", sum(sorted(ref.size for ref in refs)[-2:]))
    checkpoint = store.Checkpoint(PROFILE, generation=13, segments=refs)
    store.commit_checkpoint(PROFILE, checkpoint)
    directory = storage.data_root() / "accounting" / PROFILE / "collector"
    durable_bytes = (directory / store.CHECKPOINT).read_bytes()
    originals = {ref.name: (directory / "evidence" / ref.name).read_bytes() for ref in refs}
    first_pair = collector_maintenance.compact_pair(refs)
    assert first_pair is not None
    first_cost = sum(ref.size for ref in first_pair)
    kwargs = {
        "expired_sessions": set(),
        "clock": lambda: 0,
        "deadline": 1,
        "should_stop": lambda: False,
    }
    partial, job, used, dropped = collector_maintenance.maintain_evidence(
        PROFILE, refs, previous=None, byte_budget=first_cost + 1, **kwargs
    )
    assert used == first_cost and dropped == 0 and len(partial) == 3
    assert job["compaction_pending"] and not job["pending"]
    assert all(ref.origin == "backfill" for ref in partial)
    expected = sorted((item.native_key, item.digest) for item in items)
    assert (
        sorted((item.native_key, item.digest) for item in store.iter_evidence(PROFILE, partial))
        == expected
    )
    assert {name: (directory / "evidence" / name).read_bytes() for name in originals} == originals
    assert (directory / store.CHECKPOINT).read_bytes() == durable_bytes
    final, job, used, dropped = collector_maintenance.maintain_evidence(
        PROFILE, partial, previous=job, byte_budget=sum(ref.size for ref in partial), **kwargs
    )
    assert used > 0 and dropped == 0 and len(final) == 2
    assert not job["compaction_pending"] and not job["pending"]
    assert (
        sorted((item.native_key, item.digest) for item in store.iter_evidence(PROFILE, final))
        == expected
    )
    assert (directory / store.CHECKPOINT).read_bytes() == durable_bytes
    checkpoint.segments = final
    store.commit_checkpoint(PROFILE, checkpoint)
    store.sweep_orphans(PROFILE, checkpoint)
    restarted = store.load_checkpoint(PROFILE)
    assert restarted is not None
    assert (
        sorted(
            (item.native_key, item.digest)
            for item in store.iter_evidence(PROFILE, restarted.segments)
        )
        == expected
    )


def test_unmergeable_segments_do_not_claim_unfinishable_compaction_backlog(tmp_path, monkeypatch):
    from dataclasses import replace

    from ssgrep.accounting.collect import steps as collector_maintenance
    from ssgrep.accounting.privacy import storage
    from tests.accounting.ledger.test_minimal_evidence import evidence

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    refs = [
        store.write_segments(
            PROFILE, [replace(evidence(), record_id=f"call:{number}", origin_id=f"call:{number}")]
        )[0]
        for number in range(3)
    ]
    monkeypatch.setattr(store, "MAX_SEGMENTS", 2)
    monkeypatch.setattr(store, "SEGMENT_TARGET_BYTES", min(ref.size for ref in refs) - 1)
    directory = storage.data_root() / "accounting" / PROFILE / "collector"
    before = {
        path.relative_to(directory): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }
    assert collector_maintenance.compact_pair(refs) is None
    retained, job, used, dropped = collector_maintenance.maintain_evidence(
        PROFILE,
        refs,
        expired_sessions=set(),
        previous=None,
        clock=lambda: 0,
        deadline=1,
        byte_budget=sum(ref.size for ref in refs),
        should_stop=lambda: False,
    )
    assert retained == refs and used == dropped == 0
    assert not job["compaction_pending"] and not job["pending"]
    assert {
        path.relative_to(directory): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    } == before
    assert [item.record_id for item in store.iter_evidence(PROFILE, retained)] == [
        "call:0",
        "call:1",
        "call:2",
    ]


def test_legacy_index_backlog_advances_one_segment_per_budget_across_restart(tmp_path, monkeypatch):
    from dataclasses import replace

    from ssgrep.accounting.collect import steps as collector_maintenance
    from ssgrep.accounting.privacy import storage
    from tests.accounting.ledger.test_minimal_evidence import CANARY, evidence

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    items = [
        replace(
            evidence(),
            session_id=f"owner-{number}",
            record_id=f"call:{number}",
            origin_domain=f"claude:owner-{number}",
            origin_id=f"call:{number}",
            digest=f"digest-{number}",
        )
        for number in range(3)
    ]
    refs = [replace(store.write_segments(PROFILE, [item])[0], index=None) for item in items]
    checkpoint = store.Checkpoint(PROFILE, generation=14, segments=refs)
    store.commit_checkpoint(PROFILE, checkpoint)
    kwargs = {"clock": lambda: 0, "deadline": 1, "should_stop": lambda: False}
    expected = [(item.native_key, item.digest, item.session_id) for item in items]
    for number, legacy in enumerate(refs):
        checkpoint = store.load_checkpoint(PROFILE)
        assert checkpoint is not None
        used, pending = collector_maintenance.ensure_indexes(
            PROFILE, checkpoint, byte_budget=legacy.size, **kwargs
        )
        assert used == legacy.size
        assert pending == (number < len(refs) - 1)
        assert [ref.index is not None for ref in checkpoint.segments] == [
            index <= number for index in range(len(refs))
        ]
        remaining = [ref.name for ref in refs[number + 1 :]]
        assert checkpoint.retention.get("evidence_index_pending", []) == remaining
        for owner, item in enumerate(items[: number + 1]):
            selected = list(
                store.iter_evidence(
                    PROFILE,
                    checkpoint.segments,
                    sessions={("claude", f"owner-{owner}")},
                    native_keys={item.native_key},
                )
            )
            assert [(row.native_key, row.digest, row.session_id) for row in selected] == [
                expected[owner]
            ]
        store.commit_checkpoint(PROFILE, checkpoint)
    restarted = store.load_checkpoint(PROFILE)
    assert restarted is not None
    assert "evidence_index_pending" not in restarted.retention
    assert [
        (item.native_key, item.digest, item.session_id)
        for item in store.iter_evidence(PROFILE, restarted.segments)
    ] == expected
    root = storage.data_root() / "accounting" / PROFILE / "collector"
    assert CANARY.encode() not in b"".join(
        path.read_bytes() for path in root.rglob("*") if path.is_file()
    )


def test_bounded_expiry_resumes_then_compacts_without_mixing_live_and_backfill_owners(
    tmp_path, monkeypatch
):
    from dataclasses import replace

    from ssgrep.accounting.collect import steps as collector_maintenance
    from ssgrep.accounting.privacy import storage
    from tests.accounting.ledger.test_minimal_evidence import CANARY, evidence

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    active = [
        replace(
            evidence(),
            session_id=f"active-{number}",
            record_id=f"call:{number}",
            origin_domain=f"claude:active-{number}",
            origin_id=f"call:{number}",
            digest=f"active-digest-{number}",
        )
        for number in range(4)
    ]
    expired = [
        replace(
            evidence(),
            session_id="expired",
            record_id=f"expired:{number}",
            origin_domain="claude:expired",
            origin_id=f"expired:{number}",
            digest=f"expired-digest-{number}",
        )
        for number in range(2)
    ]
    refs = [
        store.write_segments(PROFILE, [expired[0], active[0]], origin="live")[0],
        store.write_segments(PROFILE, [active[1]], origin="live")[0],
        store.write_segments(PROFILE, [expired[1], active[2]], origin="backfill")[0],
        store.write_segments(PROFILE, [active[3]], origin="backfill")[0],
    ]
    monkeypatch.setattr(store, "MAX_SEGMENTS", 2)
    monkeypatch.setattr(store, "SEGMENT_TARGET_BYTES", sum(ref.size for ref in refs))
    checkpoint = store.Checkpoint(PROFILE, generation=15, segments=refs)
    store.commit_checkpoint(PROFILE, checkpoint)
    kwargs = {
        "expired_sessions": {("claude", "expired")},
        "clock": lambda: 0,
        "deadline": 1,
        "should_stop": lambda: False,
    }
    partial, job, used, dropped = collector_maintenance.maintain_evidence(
        PROFILE, refs, previous=None, byte_budget=refs[0].size, **kwargs
    )
    assert used == refs[0].size and dropped == 1
    assert job["pending"] == [ref.name for ref in refs[1:]]
    assert job["compaction_pending"]
    assert {item.native_key for item in store.iter_evidence(PROFILE, partial)} == {
        item.native_key for item in [*active, expired[1]]
    }
    checkpoint.segments = partial
    checkpoint.retention["evidence_maintenance"] = job
    store.commit_checkpoint(PROFILE, checkpoint)
    store.sweep_orphans(PROFILE, checkpoint)
    restarted = store.load_checkpoint(PROFILE)
    assert restarted is not None
    final, job, used, dropped = collector_maintenance.maintain_evidence(
        PROFILE,
        restarted.segments,
        previous=restarted.retention["evidence_maintenance"],
        byte_budget=10 * sum(ref.size for ref in refs),
        **kwargs,
    )
    assert used > 0 and dropped == 1
    assert not job["pending"] and not job["compaction_pending"]
    assert len(final) == 2
    expected = {
        "live": {(item.native_key, item.digest, item.session_id) for item in active[:2]},
        "backfill": {(item.native_key, item.digest, item.session_id) for item in active[2:]},
    }
    assert {
        ref.origin: {
            (item.native_key, item.digest, item.session_id)
            for item in store.iter_evidence(PROFILE, [ref])
        }
        for ref in final
    } == expected
    restarted.segments = final
    restarted.retention["evidence_maintenance"] = job
    store.commit_checkpoint(PROFILE, restarted)
    store.sweep_orphans(PROFILE, restarted)
    durable = store.load_checkpoint(PROFILE)
    assert durable is not None
    assert (
        list(store.iter_evidence(PROFILE, durable.segments, sessions={("claude", "expired")})) == []
    )
    for item in active:
        retained = list(
            store.iter_evidence(PROFILE, durable.segments, native_keys={item.native_key})
        )
        assert [(row.native_key, row.digest, row.session_id) for row in retained] == [
            (item.native_key, item.digest, item.session_id)
        ]
    root = storage.data_root() / "accounting" / PROFILE / "collector"
    persisted = b"".join(path.read_bytes() for path in root.rglob("*") if path.is_file())
    assert CANARY.encode() not in persisted
    assert all(item.digest.encode() not in persisted for item in expired)


# 9.x collector_read: discovery, baseline, bounded resumable reading (real files).


BIG = 10**9
SMALL = 1024  # Force the deliberately small synthetic backlog across several bounded cycles.


def admit_all(_: RawEvidence) -> str | None:
    return None


def env_for(tmp_path: Path) -> dict[str, str]:
    return {
        "HOME": str(tmp_path / "home"),
        "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"),
        "CODEX_HOME": str(tmp_path / "codex"),
        "OMP_SESSIONS_DIR": str(tmp_path / "omp"),
        "PI_SESSION_DIR": str(tmp_path / "pi"),
        "PRIME_AGENT_SESSION_DIR": str(tmp_path / "prime" / "sessions"),
        "SSGREP_OPENCODE_DB": str(tmp_path / "oc" / "opencode.db"),
    }


def claude_dir(tmp_path: Path) -> Path:
    return tmp_path / "claude" / "projects" / "-work-demo"


def claude_records(session: str, calls: int, pad: str = "") -> list[dict]:
    records = [claude_prompt(session, "go " + pad)]
    for n in range(calls):
        call = f"{session}-t{n}"
        records.append(claude_tool_use(session, call, "Bash", {"command": f"echo {n} {pad}"}))
        records.append(claude_tool_result(session, call, "out"))
    return records


class Cycle:
    """Drives read_sources the way a collector does: known records carry between calls."""

    def __init__(self, tmp_path: Path) -> None:
        self.env = env_for(tmp_path)
        self.readers = make_readers(self.env)
        self.ledger = ClassifyingLedger()
        self.known: dict[str, SourceRecord] = {}
        self.evidence: list[RawEvidence] = []

    def run(self, **overrides: Any) -> ReadOutcome:
        args: dict[str, Any] = {
            "discovered": discover(self.readers),
            "known": self.known,
            "admit": admit_all,
            "ledger": self.ledger,
            "byte_budget": BIG,
            "time_budget": 3600.0,
            "clock": lambda: 0.0,
        }
        args.update(overrides)
        outcome = read_sources(**args)
        self.known = dict(outcome.sources)
        self.evidence.extend(outcome.evidence)
        return outcome

    def drain(self, *, limit: int = 200, **overrides: Any) -> list[ReadOutcome]:
        outcomes = []
        for _ in range(limit):
            outcomes.append(self.run(**overrides))
            if outcomes[-1].stopped_by == "complete" and all(
                record.complete for record in outcomes[-1].sources.values()
            ):
                return outcomes
        raise AssertionError("did not converge")


def ids(items: list[RawEvidence]) -> list[tuple[str, str, str]]:
    return [(e.runtime, e.session_id, e.record_id) for e in items]


# ---------------------------------------------------------------- discovery


def test_make_readers_covers_every_runtime_and_discover_keys_by_source_key(tmp_path: Path) -> None:
    write_records(claude_dir(tmp_path) / "s1.jsonl", claude_records("s1", 1))
    write_records(tmp_path / "pi" / "p1.jsonl", [pi_header("p1")])
    readers = make_readers(env_for(tmp_path))
    assert set(readers) == set(RUNTIMES)
    found = discover(readers)
    assert list(found) == sorted(found)
    assert len(found) == 2
    for key, (reader, source) in found.items():
        assert key == source_key(source.source_id)
        assert reader is readers[source.runtime]


def test_discover_survives_one_reader_raising_oserror(tmp_path: Path) -> None:
    write_records(claude_dir(tmp_path) / "s1.jsonl", claude_records("s1", 1))
    readers = make_readers(env_for(tmp_path))

    def boom() -> list:
        raise PermissionError("no")

    readers["codex"].discover = boom
    found = discover(readers)
    assert [s.runtime for _, s in found.values()] == ["claude"]


# ------------------------------------------------------------------ budgets


def test_byte_budget_resumes_without_duplicates_and_equals_unbudgeted_union(
    tmp_path: Path,
) -> None:
    write_records(claude_dir(tmp_path) / "s1.jsonl", claude_records("s1", 12, "x" * 40))
    write_records(claude_dir(tmp_path) / "s2.jsonl", claude_records("s2", 7, "y" * 40))
    write_records(
        tmp_path / "pi" / "p1.jsonl",
        [
            pi_header("p1"),
            *[
                r
                for n in range(6)
                for r in (
                    pi_tool_call(f"e{n}", f"c{n}", "bash", {"command": "ls"}),
                    pi_tool_result(f"r{n}", f"c{n}", "ok"),
                )
            ],
        ],
    )
    whole = Cycle(tmp_path)
    one = whole.run()
    assert one.stopped_by == "complete" and one.evidence

    chunked = Cycle(tmp_path)
    outcomes = chunked.drain(byte_budget=300)
    assert len(outcomes) > 3
    assert {o.stopped_by for o in outcomes[:-1]} == {"byte_budget"}
    assert sorted(ids(chunked.evidence)) == sorted(ids(one.evidence))
    assert len(ids(chunked.evidence)) == len(set(ids(chunked.evidence)))
    assert all(r.complete and not r.more and r.pending_tail == 0 for r in chunked.known.values())

    # Reader checkpoints agree; persistent fairness counters describe different schedules.
    def reader_state(record: SourceRecord) -> dict:
        state = dict(record.state)
        state["extra"] = {k: v for k, v in state["extra"].items() if not k.startswith("collector_")}
        return state

    assert {k: reader_state(r) for k, r in chunked.known.items()} == {
        k: reader_state(r) for k, r in whole.known.items()
    }


def test_record_larger_than_remaining_budget_is_read_whole_then_call_stops(
    tmp_path: Path,
) -> None:
    path = claude_dir(tmp_path) / "s1.jsonl"
    records = claude_records("s1", 4, "z" * 600)
    write_records(path, records)
    lines = path.read_bytes().splitlines(keepends=True)
    cycle = Cycle(tmp_path)
    first = cycle.run(byte_budget=10)  # far smaller than the first record
    assert first.stopped_by == "byte_budget"
    assert first.bytes_read == len(lines[0])  # exactly one whole record, not a fragment
    [record] = first.sources.values()
    assert record.offset == len(lines[0]) and record.more and not record.complete
    assert ids(first.evidence) == [("claude", "s1", "turn:u-1")]


def test_time_budget_with_fake_clock_stops_between_sources(tmp_path: Path) -> None:
    for n in range(3):
        write_records(claude_dir(tmp_path) / f"s{n}.jsonl", claude_records(f"s{n}", 2))
    ticks = iter(range(0, 1000))
    outcome = Cycle(tmp_path).run(clock=lambda: float(next(ticks)), time_budget=2.5)
    assert outcome.stopped_by == "time_budget"
    done = [r for r in outcome.sources.values() if r.complete]
    untouched = [r for r in outcome.sources.values() if r.offset == 0]
    assert done and untouched and len(done) + len(untouched) == 3
    assert all(not r.complete for r in untouched)


def test_stop_requested_before_any_source_reads_nothing(tmp_path: Path) -> None:
    write_records(claude_dir(tmp_path) / "s1.jsonl", claude_records("s1", 2))
    cycle = Cycle(tmp_path)
    outcome = cycle.run(should_stop=lambda: True)
    assert outcome.stopped_by == "stop_requested"
    assert outcome.evidence == [] and outcome.bytes_read == 0
    [record] = outcome.sources.values()
    assert record.offset == 0 and not record.complete
    # resuming reads everything
    assert cycle.run().stopped_by == "complete" and len(cycle.evidence) == 5


def test_stop_requested_midway_finishes_current_read_and_skips_the_rest(tmp_path: Path) -> None:
    for n in range(3):
        write_records(claude_dir(tmp_path) / f"s{n}.jsonl", claude_records(f"s{n}", 2))
    flag = {"stop": False}
    cycle = Cycle(tmp_path)
    outcome = cycle.run(
        should_stop=lambda: flag["stop"],
        on_source=lambda key, record: flag.update(stop=True),
    )
    assert outcome.stopped_by == "stop_requested"
    assert sum(r.complete for r in outcome.sources.values()) == 1


def test_unreached_source_stat_refresh_shows_current_lag(tmp_path: Path) -> None:
    first = claude_dir(tmp_path) / "a.jsonl"
    second = claude_dir(tmp_path) / "b.jsonl"
    write_records(first, claude_records("a", 1))
    write_records(second, claude_records("b", 1))
    cycle = Cycle(tmp_path)
    cycle.run()
    key_b = source_key(f"claude:{second}")
    assert cycle.known[key_b].complete
    old_size = cycle.known[key_b].size
    append_bytes(second, jsonl(claude_records("b2", 1)))
    append_bytes(first, jsonl(claude_records("a2", 1)))
    outcome = cycle.run(should_stop=lambda: True)  # reaches nothing
    record = outcome.sources[key_b]
    assert record.size == second.stat().st_size > old_size
    assert record.mtime_ns == second.stat().st_mtime_ns
    assert not record.complete and record.offset == old_size  # lag = size - offset > 0
    assert outcome.bytes_read == 0


def test_unreached_source_that_vanished_is_unreadable_not_an_exception(tmp_path: Path) -> None:
    path = claude_dir(tmp_path) / "a.jsonl"
    write_records(path, claude_records("a", 1))
    cycle = Cycle(tmp_path)
    cycle.run()
    discovered = discover(cycle.readers)
    path.unlink()
    outcome = cycle.run(discovered=discovered, should_stop=lambda: True)
    [record] = outcome.sources.values()
    assert record.status == "unreadable" and not record.complete


# ----------------------------------------------------------------- baseline


def test_baseline_skips_old_bytes_but_reads_later_appended_records(tmp_path: Path) -> None:
    path = claude_dir(tmp_path) / "s1.jsonl"
    write_records(path, claude_records("s1", 3))
    size = path.stat().st_size
    cycle = Cycle(tmp_path)
    outcome = cycle.run(baseline_new=True)
    [record] = outcome.sources.values()
    assert outcome.evidence == [] and outcome.bytes_read == 0
    assert record.baseline == "skipped" and record.complete
    assert record.skipped_bytes == record.offset == size == record.size
    append_bytes(path, jsonl([claude_tool_use("s1", "late", "Bash", {"command": "ls"})]))
    later = cycle.run(baseline_new=True)
    assert ids(later.evidence) == [("claude", "s1", "call:late")]
    assert later.sources[record.key].baseline == "skipped"  # provenance kept
    assert later.sources[record.key].complete


def test_baseline_stops_at_last_newline_so_a_torn_tail_is_read_once_when_finished(
    tmp_path: Path,
) -> None:
    path = claude_dir(tmp_path) / "s1.jsonl"
    write_records(path, claude_records("s1", 1))
    whole = path.stat().st_size
    pending = jsonl([claude_tool_use("s1", "late", "Bash", {"command": "ls"})])
    append_bytes(path, pending[:-1])
    reader = make_readers(env_for(tmp_path))["claude"]
    [source] = reader.discover()
    record = baseline_record(reader, source)
    assert record.offset == whole and record.skipped_bytes == whole
    assert record.pending_tail == len(pending) - 1 and not record.complete
    cycle = Cycle(tmp_path)
    cycle.known = {record.key: record}
    first = cycle.run()
    assert first.evidence == [] and first.sources[record.key].pending_tail == len(pending) - 1
    append_bytes(path, pending[-1:])
    second = cycle.run()
    assert ids(second.evidence) == [("claude", "s1", "call:late")]
    assert second.sources[record.key].complete
    assert cycle.run().evidence == []


def test_baseline_finds_last_newline_across_chunks_of_a_huge_unterminated_tail(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pi" / "p1.jsonl"
    write_records(path, [pi_header("p1")])
    head = path.stat().st_size
    append_bytes(path, b"x" * (3 * 1024 * 1024))  # no newline anywhere in the tail
    reader = make_readers(env_for(tmp_path))["pi"]
    [source] = reader.discover()
    record = baseline_record(reader, source)
    assert record.offset == head and record.pending_tail == 3 * 1024 * 1024
    assert baseline_record(reader, source).state == record.state


def test_baseline_of_empty_and_newline_free_files(tmp_path: Path) -> None:
    empty = tmp_path / "pi" / "e.jsonl"
    empty.parent.mkdir(parents=True)
    empty.write_bytes(b"")
    nonl = tmp_path / "pi" / "n.jsonl"
    nonl.write_bytes(b'{"type":"session"')
    reader = make_readers(env_for(tmp_path))["pi"]
    records = {s.path.name: baseline_record(reader, s) for s in reader.discover()}
    assert records["e.jsonl"].offset == 0 and records["e.jsonl"].complete
    assert records["n.jsonl"].offset == 0 and records["n.jsonl"].pending_tail == 17


def test_baseline_of_missing_source_is_unreadable_without_raising(tmp_path: Path) -> None:
    path = tmp_path / "pi" / "p1.jsonl"
    write_records(path, [pi_header("p1")])
    reader = make_readers(env_for(tmp_path))["pi"]
    [source] = reader.discover()
    path.unlink()
    assert baseline_record(reader, source).status == "unreadable"


# ----------------------------------------------------------- torn tail / reset


def test_torn_tail_never_accepted_then_read_exactly_once(tmp_path: Path) -> None:
    path = claude_dir(tmp_path) / "s1.jsonl"
    write_records(path, claude_records("s1", 1))
    full = jsonl([claude_tool_use("s1", "tail", "Bash", {"command": "ls"})])
    append_bytes(path, full[:-1])
    cycle = Cycle(tmp_path)
    first = cycle.run()
    [record] = first.sources.values()
    assert record.pending_tail == len(full) - 1 and not record.complete and not record.more
    assert ("claude", "s1", "call:tail") not in ids(first.evidence)
    again = cycle.run()
    assert again.evidence == [] and again.sources[record.key].pending_tail == len(full) - 1
    append_bytes(path, b"\n")
    done = cycle.run()
    assert ids(done.evidence) == [("claude", "s1", "call:tail")]
    assert done.sources[record.key].complete and done.sources[record.key].pending_tail == 0
    assert ids(cycle.evidence).count(("claude", "s1", "call:tail")) == 1


def test_truncation_and_rotation_produce_reset_gap_and_reread_without_crash(
    tmp_path: Path,
) -> None:
    path = claude_dir(tmp_path) / "s1.jsonl"
    write_records(path, claude_records("s1", 4))
    cycle = Cycle(tmp_path)
    cycle.run()
    write_records(path, claude_records("s9", 1))  # truncated, different content
    outcome = cycle.run()
    [record] = outcome.sources.values()
    assert outcome.resets == 1 and outcome.gaps["reset_truncated"] == 1
    assert record.resets == 1 and record.gaps["reset_truncated"] == 1
    assert {e.session_id for e in outcome.evidence} == {"s9"} and len(outcome.evidence) == 3
    # rotation: same name, new inode, at least as large
    write_records(path, claude_records("s9", 1))
    rotated = tmp_path / "rot.jsonl"
    rotated.write_bytes(path.read_bytes() + b"")
    os.replace(rotated, path)
    again = cycle.run()
    assert again.resets == 1 and again.gaps["reset_rotated"] == 1
    assert again.evidence == []  # same events: ledger dedupes the re-read copy
    assert again.dropped["duplicate"] == 3


def test_missing_source_keeps_its_record_with_missing_status(tmp_path: Path) -> None:
    path = claude_dir(tmp_path) / "s1.jsonl"
    write_records(path, claude_records("s1", 1))
    cycle = Cycle(tmp_path)
    cycle.run()
    [before] = cycle.known.values()
    path.unlink()
    outcome = cycle.run()
    [after] = outcome.sources.values()
    assert after.status == "missing" and not after.complete
    assert after.state == before.state and after.source_id == before.source_id
    assert outcome.evidence == []


# --------------------------------------------------- dedupe / conflict / admit


def test_duplicate_copy_of_history_file_dedupes_through_ledger(tmp_path: Path) -> None:
    records = claude_records("s1", 3)
    write_records(claude_dir(tmp_path) / "s1.jsonl", records)
    write_records(claude_dir(tmp_path) / "copy-of-s1.jsonl", records)
    outcome = Cycle(tmp_path).run()
    native = [e for e in outcome.evidence if e.kind != "ownership_claim"]
    claims = [e for e in outcome.evidence if e.kind == "ownership_claim"]
    assert len(native) == len({e.native_key for e in native}) == 7
    assert {tuple(e.payload["target_native_key"]) for e in claims} == {e.native_key for e in native}
    assert outcome.dropped == {"duplicate": 7}
    assert len(outcome.sources) == 2 and all(r.complete for r in outcome.sources.values())


def test_conflicting_variant_is_counted_as_conflict_not_duplicate(tmp_path: Path) -> None:
    write_records(
        claude_dir(tmp_path) / "a.jsonl",
        [
            claude_prompt("s1", "hi"),
            claude_tool_use("s1", "t1", "Bash", {"command": "ls"}),
        ],
    )
    write_records(
        claude_dir(tmp_path) / "b.jsonl",
        [
            claude_prompt("s1", "hi"),
            claude_tool_use("s1", "t1", "Bash", {"command": "rm -rf /tmp/other"}),
        ],
    )
    cycle = Cycle(tmp_path)
    outcome = cycle.run()
    assert outcome.dropped == {"duplicate": 1, "conflict": 1}
    native = [e for e in outcome.evidence if e.kind != "ownership_claim"]
    assert len(native) == len({e.native_key for e in native}) == 2
    assert {e.kind for e in native} == {"user_turn", "tool_call"}
    assert cycle.ledger.conflict_count == 1


def test_admit_reasons_are_counted_and_those_records_not_retained(tmp_path: Path) -> None:
    write_records(claude_dir(tmp_path) / "s1.jsonl", claude_records("s1", 3))
    cycle = Cycle(tmp_path)

    def admit(item: RawEvidence) -> str | None:
        return "before_horizon" if item.kind == "tool_result" else None

    outcome = cycle.run(admit=admit)
    assert outcome.dropped == {"before_horizon": 3}
    assert {e.kind for e in outcome.evidence} == {"user_turn", "tool_call"}
    # dropped records never reached the ledger: admitting them later still succeeds
    later = read_again_all(cycle)
    assert [e.kind for e in later] == ["tool_result"] * 3


def read_again_all(cycle: Cycle) -> list[RawEvidence]:
    cycle.known = {}
    outcome = cycle.run()
    return outcome.evidence


def test_usage_and_other_kinds_outside_consumed_set_are_never_retained(tmp_path: Path) -> None:
    usage = {"input_tokens": 5, "output_tokens": 7}
    write_records(
        claude_dir(tmp_path) / "s1.jsonl",
        [
            claude_prompt("s1", "hi"),
            claude_tool_use("s1", "t1", "Bash", {"command": "ls"}, usage=usage),
        ],
    )
    cycle = Cycle(tmp_path)
    outcome = cycle.run()
    assert outcome.dropped == {"kind_not_consumed": 1}
    assert {e.kind for e in outcome.evidence} == {"user_turn", "tool_call"}
    assert all(e.kind in CONSUMED_KINDS for e in outcome.evidence)
    assert list(cycle.ledger.final_usage()) == []  # usage never even reached the ledger


# --------------------------------------------------------------- unreadable


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions (chmod 000)")
def test_unreadable_file_marks_status_and_keeps_state(tmp_path: Path) -> None:
    good = claude_dir(tmp_path) / "good.jsonl"
    bad = claude_dir(tmp_path) / "bad.jsonl"
    write_records(good, claude_records("g", 1))
    write_records(bad, claude_records("b", 1))
    bad.chmod(0)
    try:
        cycle = Cycle(tmp_path)
        outcome = cycle.run()
    finally:
        bad.chmod(0o600)
    assert outcome.unreadable == 1
    by_name = {Path(r.source_id.split(":", 1)[1]).name: r for r in outcome.sources.values()}
    assert by_name["bad.jsonl"].status == "unreadable" and not by_name["bad.jsonl"].complete
    assert by_name["bad.jsonl"].offset == 0
    assert not by_name["bad.jsonl"].state.get("head_digest")
    assert not by_name["bad.jsonl"].state.get("tail_digest")
    assert by_name["good.jsonl"].status == "ok" and by_name["good.jsonl"].complete
    assert {e.session_id for e in outcome.evidence} == {"g"}
    # once readable, the very same record resumes from offset 0 and reads it all
    again = cycle.run()
    assert {e.session_id for e in again.evidence} == {"b"}
    assert again.sources[by_name["bad.jsonl"].key].status == "ok"


# ----------------------------------------------------------------- opencode


def oc_db(tmp_path: Path) -> Path:
    return tmp_path / "oc" / "opencode.db"


def oc_add_call(connection, n: int, at: int) -> None:
    opencode_add_message(connection, f"msg_{n}", "ses_1", opencode_assistant_data(cost=1), at=at)
    opencode_add_part(
        connection,
        f"prt_{n}",
        f"msg_{n}",
        "ses_1",
        opencode_tool_part(f"call_{n}", "bash", {"command": "ls"}, start=at + 1, end=at + 2),
        at=at,
    )


def db_bytes(path: Path) -> dict[str, str]:
    def digest(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "absent"

    return {"db": digest(path), "wal": digest(Path(f"{path}-wal"))}


def test_opencode_unvisited_baseline_reads_history_and_later_wal_once_without_writes(
    tmp_path: Path,
) -> None:
    db = oc_db(tmp_path)
    connection = opencode_connect(db)
    opencode_add_session(connection, "ses_1")
    oc_add_call(connection, 1, READERS_T0 + 100)
    oc_add_call(connection, 2, READERS_T0 + 200)
    cycle = Cycle(tmp_path)
    reader = cycle.readers["opencode"]
    [source] = reader.discover()

    before = db_bytes(db)
    record = baseline_record(reader, source)
    assert db_bytes(db) == before
    assert record.baseline == "skipped" and not record.complete and record.status == "ok"
    assert not record.state["extra"].get("wm")
    assert record.state["extra"]["opencode"]["status"] == "unvisited"
    assert record.state["extra"]["identity"]["inode"] == db.stat().st_ino
    assert record.size == db.stat().st_size and record.skipped_bytes == 0

    # A database baseline has not visited any row; it must not claim complete or
    # invent maxima. An unrestricted read admits history, unlike JSONL byte skipping.
    outcome = cycle.run(baseline_new=True)
    assert sorted(e.record_id for e in outcome.evidence if e.kind == "tool_call") == [
        "call:call_1",
        "call:call_2",
    ]
    assert all(r.complete for r in outcome.sources.values()) and db_bytes(db) == before
    # rows committed after the baseline (still only in the WAL) are read; older ones are not
    wal_before = Path(f"{db}-wal").stat().st_size
    oc_add_call(connection, 3, READERS_T0 + 300)
    assert Path(f"{db}-wal").stat().st_size > wal_before  # really in the WAL
    later = cycle.run(baseline_new=True)
    got = sorted(ids(later.evidence))
    assert got == [("opencode", "ses_1", "call:call_3"), ("opencode", "ses_1", "result:call_3")]
    assert later.bytes_read > 0
    [visited] = later.sources.values()
    assert visited.state["extra"]["wm"]["part"] == [READERS_T0 + 300, "prt_3"]
    assert all(r.complete for r in later.sources.values())
    # reading never writes the database or its WAL
    after_read = db_bytes(db)
    assert cycle.run().evidence == []
    assert db_bytes(db) == after_read
    connection.close()


def test_opencode_new_source_without_baseline_reads_everything(tmp_path: Path) -> None:
    db = oc_db(tmp_path)
    connection = opencode_connect(db)
    opencode_add_session(connection, "ses_1")
    oc_add_call(connection, 1, READERS_T0 + 100)
    cycle = Cycle(tmp_path)
    outcome = cycle.run()
    kinds = sorted(e.kind for e in outcome.evidence)
    assert kinds == ["session_header", "tool_call", "tool_result"]
    assert outcome.dropped["kind_not_consumed"] >= 1  # usage, including maintenance visits
    assert all(e.kind in CONSUMED_KINDS for e in outcome.evidence)
    assert all(r.complete for r in outcome.sources.values())
    connection.close()


def test_opencode_corrupt_database_is_unreadable_not_an_exception(tmp_path: Path) -> None:
    db = oc_db(tmp_path)
    db.parent.mkdir(parents=True)
    db.write_bytes(b"this is not a sqlite database" * 50)
    cycle = Cycle(tmp_path)
    outcome = cycle.run()
    [record] = outcome.sources.values()
    assert record.status == "unreadable" and not record.complete
    assert outcome.unreadable == 1 and outcome.evidence == []
    reader = cycle.readers["opencode"]
    [source] = reader.discover()
    baseline = baseline_record(reader, source)
    assert not baseline.complete and baseline.state["extra"]["opencode"]["status"] == "unvisited"


def test_opencode_pages_resume_across_budgeted_calls_without_duplicates(tmp_path: Path) -> None:
    db = oc_db(tmp_path)
    connection = opencode_connect(db)
    opencode_add_session(connection, "ses_1")
    connection.execute("BEGIN")
    for n in range(1, 1101):  # more than the reader's 1000-row page
        oc_add_call(connection, n, READERS_T0 + 100 + n)
    connection.execute("COMMIT")
    whole = Cycle(tmp_path)
    whole.drain()
    chunked = Cycle(tmp_path)
    # One native row is the cooperative quantum, not the old 1000-row page.
    outcomes = chunked.drain(byte_budget=1, limit=1100 * 8 + 20)
    assert len(outcomes) >= 2 and outcomes[0].stopped_by == "byte_budget"
    assert outcomes[0].bytes_read > 0 and outcomes[0].evidence
    [first_record] = outcomes[0].sources.values()
    assert sum(first_record.state["extra"]["opencode"]["rows_visited"].values()) == 1
    assert sorted(ids(chunked.evidence)) == sorted(ids(whole.evidence))
    assert len(chunked.evidence) == len(set(ids(chunked.evidence))) == 1 + 2 * 1100
    assert all(r.complete and not r.more for r in chunked.known.values())
    [record] = chunked.known.values()
    assert record.state["extra"]["wm"] == next(iter(whole.known.values())).state["extra"]["wm"]
    assert record.state["extra"]["wm"] == {
        "session": [READERS_T0, "ses_1"],
        "message": [READERS_T0 + 1200, "msg_1100"],
        "part": [READERS_T0 + 1200, "prt_1100"],
    }
    assert all(
        record.state["extra"]["opencode"]["rows_visited"][table] >= count
        for table, count in {"session": 1, "message": 1100, "part": 1100}.items()
    )
    connection.close()


def test_opencode_idle_skip_requires_the_accepted_generation_to_match_the_database(
    tmp_path: Path,
) -> None:
    from ssgrep.accounting.collect.read import _unchanged

    db = oc_db(tmp_path)
    connection = opencode_connect(db)
    try:
        opencode_add_session(connection, "ses_1")
        oc_add_call(connection, 1, READERS_T0 + 100)
        cycle = Cycle(tmp_path)
        cycle.drain()
        reader = cycle.readers["opencode"]
        [source] = reader.discover()
        [record] = cycle.known.values()
        assert record.complete and _unchanged(reader, source, record)
        idle = cycle.run()
        assert idle.bytes_read == 0 and idle.evidence == []

        # Without the reader's own accepted census the collector may not skip the database.
        extra = {**record.state["extra"]}
        extra.pop("collector_source_generation")
        unaccepted = replace(record, state={**record.state, "extra": extra})
        assert not _unchanged(reader, source, unaccepted)
        foreign = {**record.state["extra"], "collector_source_generation": {"inode": -1}}
        assert not _unchanged(
            reader, source, replace(record, state={**record.state, "extra": foreign})
        )

        # A failure to check the current generation is never an idle proof.
        aside = db.with_name("aside.db")
        db.rename(aside)
        assert not _unchanged(reader, source, record)  # the database vanished
        db.mkdir()
        assert not _unchanged(reader, source, record)  # not a regular database file
        db.rmdir()
        aside.rename(db)
        # Restoring the file changes its inode metadata, so the old proof is gone for good.
        assert not _unchanged(reader, source, record)
    finally:
        connection.close()


def test_opencode_snapshot_lost_to_a_checkpoint_is_changed_then_read_on_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contextlib import contextmanager

    db = oc_db(tmp_path)
    connection = opencode_connect(db)
    try:
        opencode_add_session(connection, "ses_1")
        oc_add_call(connection, 1, READERS_T0 + 100)
        cycle = Cycle(tmp_path)
        reader = cycle.readers["opencode"]
        snapshot = reader.snapshot

        @contextmanager
        def checkpoint_after_pin(path, *, descriptor=None, tracker=None):
            with snapshot(path, descriptor=descriptor, tracker=tracker) as pinned:
                connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                yield pinned

        with monkeypatch.context() as patch:
            patch.setattr(reader, "snapshot", checkpoint_after_pin)
            lost = cycle.run()
        [record] = lost.sources.values()
        assert record.status == "changed" and not record.complete
        assert lost.evidence == []

        retry = cycle.run()
        assert any(item.payload.get("call_id") == "call_1" for item in retry.evidence)
        assert all(r.status == "ok" and r.complete for r in retry.sources.values())
    finally:
        connection.close()


def test_a_source_that_grows_while_it_is_read_gets_no_idle_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = claude_dir(tmp_path) / "growing.jsonl"
    write_records(path, claude_records("growing", 1))
    stable = Cycle(tmp_path)
    stable.run()
    [settled] = stable.known.values()
    assert settled.state["extra"]["collector_source_signature"]

    grow_once = [True]
    cycle = Cycle(tmp_path)
    reader = cycle.readers["claude"]
    original = reader.read

    def grow_after_read(*args, **kwargs):
        result = original(*args, **kwargs)
        if grow_once:
            grow_once.clear()
            append_bytes(path, jsonl([claude_prompt("growing", "late", uuid="u-late")]))
        return result

    monkeypatch.setattr(reader, "read", grow_after_read)
    first = cycle.run()
    [record] = first.sources.values()
    assert "collector_source_signature" not in record.state["extra"]
    monkeypatch.setattr(reader, "read", original)
    second = cycle.run()
    assert any(item.payload.get("text") == "late" for item in second.evidence)
    [after] = second.sources.values()
    assert after.state["extra"]["collector_source_signature"]


def test_source_signature_is_withheld_when_the_primary_path_cannot_be_inspected(
    tmp_path: Path,
) -> None:
    from ssgrep.accounting.collect.read import _source_signature
    from ssgrep.accounting.readers.base import Source
    from ssgrep.accounting.readers.opencode import OpenCodeReader

    blocker = tmp_path / "blocker"
    blocker.write_bytes(b"a file where a directory is needed")
    reader = OpenCodeReader({})
    assert _source_signature(reader, Source("opencode", blocker / "opencode.db")) is None
    database = tmp_path / "opencode.db"
    database.write_bytes(b"database")
    assert _source_signature(reader, Source("opencode", database)) is not None


def test_on_source_called_per_source_in_sorted_key_order_when_all_known(tmp_path: Path) -> None:
    for n in range(3):
        write_records(claude_dir(tmp_path) / f"s{n}.jsonl", claude_records(f"s{n}", 1))
    cycle = Cycle(tmp_path)
    cycle.run()
    seen: list[str] = []
    cycle.run(on_source=lambda key, record: seen.append(key))
    assert seen == sorted(cycle.known) and len(seen) == 3


def test_backlogged_sources_are_read_before_finished_ones(tmp_path: Path) -> None:
    for n in range(3):
        write_records(claude_dir(tmp_path) / f"s{n}.jsonl", claude_records(f"s{n}", 1))
    cycle = Cycle(tmp_path)
    cycle.run()
    keys = sorted(cycle.known)
    # grow the LAST key's file so it has unread data; it must be read first
    last = cycle.known[keys[-1]]
    path = Path(last.source_id.split(":", 1)[1])
    append_bytes(path, jsonl([claude_prompt("zz", "more", uuid="u-zz")]))
    seen: list[str] = []
    cycle.run(on_source=lambda key, record: seen.append(key))
    assert seen[0] == keys[-1] and seen[1:] == keys[:-1]


def test_same_size_rewrite_is_not_starved_by_an_always_growing_backlog(tmp_path: Path) -> None:
    for n in range(2):
        write_records(claude_dir(tmp_path) / f"s{n}.jsonl", claude_records(f"s{n}", 1))
    cycle = Cycle(tmp_path)
    cycle.run()
    backlog, rewritten = [cycle.known[key] for key in sorted(cycle.known)]
    backlog_path = Path(backlog.source_id.split(":", 1)[1])
    rewritten_path = Path(rewritten.source_id.split(":", 1)[1])
    session = rewritten_path.stem
    replacement = "n" + session[1:]
    write_records(rewritten_path, claude_records(replacement, 1))
    assert rewritten_path.stat().st_size == rewritten.size
    for n in range(4):
        append_bytes(
            backlog_path,
            jsonl([claude_prompt("growing", "more", uuid=f"u-growing-{n}")]),
        )
        cycle.run(byte_budget=1)
        if cycle.known[rewritten.key].resets:
            break
    assert cycle.known[rewritten.key].resets == 1
    assert any(e.session_id == replacement for e in cycle.evidence)


# ---- admission rules -------------------------------------------------------------------------


def test_expired_conversation_and_retention_rejections_do_not_discard_needed_structure() -> None:
    rules = AdmitRules(
        expired_sessions=frozenset({("claude", "expired")}),
        horizon_day="2026-10-10",
        capture_start="2026-10-11T00:00:00Z",
    )
    old = build_evidence(ts_utc="2026-10-09T23:59:59Z")
    assert rules(old) == "beyond_retention"
    structural = replace(old, kind="session_header")
    assert rules(structural) is None
    assert rules(replace(structural, session_id="expired")) == "expired_conversation"
    assert rules(replace(structural, runtime="codex", session_id="expired")) is None
    assert rules(replace(old, ts_utc="2026-10-12T00:00:00Z")) is None


def test_backfill_retains_undated_structure_but_not_undated_prompt_text() -> None:
    rules = AdmitRules(window=("2026-10-11T00:00:00Z", "2026-10-12T00:00:00Z"))
    undated = build_evidence(ts_utc=None)
    assert rules(undated) == "outside_interval"
    assert rules(replace(undated, kind="lineage")) is None
    assert rules(replace(undated, ts_utc="2026-10-11T00:00:00Z")) is None
    assert rules(replace(undated, ts_utc="2026-10-12T00:00:00Z")) == "outside_interval"


@pytest.mark.parametrize("invalid", [True, "not-an-instant"])
def test_invalid_capture_timestamp_is_unknown_not_an_epoch_origin(invalid: object) -> None:
    assert collector_modes.parse_stamp(invalid) is None
    assert collector_modes.parse_stamp("2026-10-12T01:00:00+02:00") == datetime(
        2026, 10, 11, 23, tzinfo=UTC
    )


# ---- the real collector over damaged, vanishing and budgeted sources -----------------------------


def test_complete_malformed_source_line_advances_watermark_and_counts_gap_once(
    tmp_path: Path,
) -> None:
    seeded(tmp_path)
    path = vs.projects(tmp_path) / "-work" / "cons0.jsonl"
    with path.open("ab") as handle:
        handle.write(b"not-json\n")
    scene.collect(NOW)
    checkpoint = scene.committed()
    record = next(row for row in checkpoint.sources.values() if row.source_id.endswith(str(path)))
    assert record.offset == path.stat().st_size and record.gaps["malformed_json"] == 1
    assert scene.coverage(scene.current())["gaps"]["malformed_json"] == 1
    assert scene.counter_total(scene.current()) == 2
    scene.collect(NOW)
    again = scene.committed().sources[record.key]
    assert again.offset == record.offset and again.gaps == record.gaps


def test_disappearing_source_during_prebaseline_does_not_hide_other_sources(tmp_path: Path) -> None:
    vs.enable()
    vs.filler(vs.projects(tmp_path), "gone", scene.at(hour=9), 1)
    vs.filler(vs.projects(tmp_path), "present", scene.at(hour=9), 1)
    discovered = collector_read.discover(collector_read.make_readers())
    vanished = vs.projects(tmp_path) / "-work" / "gone.jsonl"
    vanished.unlink()
    known = {}
    assert collector_modes.pre_baseline(discovered, known, NOW) == 1
    assert len(known) == 1
    record = next(iter(known.values()))
    assert record.source_id.endswith("present.jsonl") and record.baseline == "skipped"
    assert record.offset > 0


def test_inaccessible_index_is_reported_unknown_without_losing_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeded(tmp_path)
    index = storage.data_root() / "lancedb"
    index.mkdir(mode=0o700)
    original = Path.stat

    def denied(path: Path, *args: Any, **kwargs: Any) -> os.stat_result:
        if path == index:
            raise PermissionError("index stat denied")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "stat", denied)
        scene.collect(NOW)
        assert scene.coverage(scene.current())["index"] == "unknown"
        assert scene.counter_total(scene.current()) == 2
    scene.collect(NOW)
    assert scene.coverage(scene.current())["index"] == "available"


def test_partial_backfill_resumes_its_own_offsets_after_restart(tmp_path: Path) -> None:
    vs.enable(pin=True)
    vs.start_capture(day=12)
    vs.pair(vs.projects(tmp_path), 0, scene.at(day=11, hour=9))
    scene.collect(NOW)
    live_offsets = {key: row.offset for key, row in scene.committed().sources.items()}
    before = scene.counter_total(scene.current())
    options = CollectOptions(
        backfill=True,
        since="2026-10-11T00:00:00Z",
        until="2026-10-13T00:00:00Z",
        budget=CycleBudget(bytes=1),
    )
    first = scene.collect(NOW, options)
    assert first.state == "partial"
    job = scene.committed().backfill_job
    assert job["status"] == "running"
    assert any(row["state"]["offset"] > 0 for row in job["sources"].values())
    for _ in range(30):
        result = scene.collect(NOW, options)
        assert scene.counter_total(scene.current()) == before
        assert {key: row.offset for key, row in scene.committed().sources.items()} == live_offsets
        if result.state == "published":
            break
    else:
        pytest.fail("backfill did not finish its bounded resumable scan")
    checkpoint = scene.committed()
    assert not checkpoint.backfill_job
    assert len(checkpoint.backfills) == 1
    assert (
        sum(
            count
            for key, count in checkpoint.backfills[0]["counts"].items()
            if key.startswith("search:")
        )
        == 1
    )
    identities = vs.evidence_identities()
    assert identities and len(identities) == len(set(identities))
    scene.collect(NOW, CollectOptions(backfill=True, since=options.since, until=options.until))
    assert scene.committed().backfills == checkpoint.backfills
    assert vs.evidence_identities() == identities


def test_blocked_backfill_requires_source_repair_before_original_interval_retry(tmp_path):
    from ssgrep.cli.collect_render import collect_document

    vs.enable(pin=True)
    vs.start_capture(day=12)
    vs.pair(vs.projects(tmp_path), 0, scene.at(day=11, hour=9))
    scene.collect(NOW)
    before = scene.counter_total(scene.current())
    source = vs.projects(tmp_path) / "unfinished.jsonl"
    source.touch()
    scene.append_transcript(
        source,
        scene.search_records("unfinished", "restored-call", "history-only", (9, 1, 0)),
        scene.at(day=11, hour=9, minute=1),
    )
    original = source.read_bytes()
    source.write_bytes(original[:20])  # Discovered, but no complete native record yet.
    options = CollectOptions(
        backfill=True,
        since="2026-10-11T00:00:00Z",
        until="2026-10-13T00:00:00Z",
    )
    assert scene.collect(NOW, options).state == "partial"
    source.unlink()
    for _ in range(2):
        blocked = scene.collect(NOW, options)
        assert scene.committed().backfill_job["status"] == "blocked"
        assert scene.report(blocked).stopped_by == "backfill_blocked"
        assert not scene.report(blocked).resumable and not scene.report(blocked).complete
        _, _, actions = collect_document(PROFILE, options, blocked)
        assert actions[0].command == "ssgrep savings status --sources"
        assert "--backfill" in actions[1].command
        assert scene.counter_total(scene.current()) == before
    source.write_bytes(original)
    recovered = scene.collect(NOW, options)
    assert recovered.state == "published" and scene.report(recovered).complete
    assert scene.committed().backfill_job == {}
    assert len(scene.committed().backfills) == 1
    assert (
        sum(
            count
            for key, count in scene.committed().backfills[0]["counts"].items()
            if key.startswith("search:")
        )
        == 2
    )  # Includes the previously unread lookup after its source is restored.
    assert scene.counter_total(scene.current()) == before


# ---- cooperative cycle budgets: a backlog resumes across cycles and converges to single-shot ----
# Convergence is judged on retained evidence identities, the canonical ledger digest, and the
# live counters of an unbudgeted single-shot run of the same scene.


def backlog_scene(root: Path, sessions: int = 5) -> None:
    vs.enable(pin=True)
    vs.start_capture()
    for n in range(sessions):
        vs.pair(vs.projects(root), n, CLOCK)


def cycles_until_done(budget: CycleBudget, limit: int = 300) -> list:
    results = []
    for _ in range(limit):
        result = scene.collect(NOW, budget=budget)
        results.append(result)
        if scene.report(result).stopped_by == "complete":
            return results
    raise AssertionError("never completed")


def test_a_small_budget_backlog_resumes_to_exactly_the_single_shot_ledger(tmp_path: Path) -> None:
    backlog_scene(tmp_path)
    single = scene.collect(NOW)
    assert scene.report(single).complete and scene.report(single).stopped_by == "complete"
    reference = vs.facts()
    vs.wipe(tmp_path)

    backlog_scene(tmp_path)
    results = cycles_until_done(CycleBudget(bytes=SMALL))
    reports = [scene.report(r) for r in results]
    assert len(reports) >= 4  # the backlog genuinely needed several cycles
    assert [r.stopped_by for r in reports[:-1]] == ["byte_budget"] * (len(reports) - 1)
    assert all(r.status == "published" and not r.complete for r in reports[:-1])
    assert [r.generation for r in reports] == list(range(1, len(reports) + 1))
    assert reports[-1].complete and reports[-1].lag_bytes == 0

    final = vs.facts()
    assert final.evidence == reference.evidence  # no duplicate and no missing record
    assert len(set(final.evidence)) == len(final.evidence)
    assert final.ledger == reference.ledger  # final ledger == single-shot
    assert final.counters == reference.counters
    assert final.snapshot_id == reference.snapshot_id  # even the snapshot identity agrees
    # the budget bound each cycle (one bounded record may overshoot, never unbounded further work)
    largest = max(
        len(json.dumps(r).encode()) + 1
        for path in vs.projects(tmp_path).rglob("*.jsonl")
        for r in map(json.loads, path.read_text().splitlines())
    )
    assert all(r.bytes_read <= SMALL + largest for r in reports)
    assert sum(r.bytes_read for r in reports) >= scene.report(single).bytes_read  # all consumed


def test_every_partial_generation_states_its_prefix_lag_and_never_claims_completeness(
    tmp_path: Path,
) -> None:
    backlog_scene(tmp_path)
    total = sum(p.stat().st_size for p in vs.projects(tmp_path).rglob("*.jsonl"))
    lags: list[int] = []
    for _ in range(300):
        result = scene.collect(NOW, budget=CycleBudget(bytes=SMALL))
        envelope = scene.current()
        coverage = scene.coverage(envelope)
        watermarks = coverage["watermarks"]
        report = scene.report(result)
        assert coverage["lag_bytes"] == report.lag_bytes == sum(w["lag_bytes"] for w in watermarks)
        assert all(0 <= w["offset"] <= w["size"] for w in watermarks)
        assert coverage["stopped_by"] == report.stopped_by
        if report.stopped_by == "complete":
            assert coverage["complete"] is True and coverage["lag_bytes"] == 0
            break
        assert coverage["complete"] is False  # a lagging generation is never complete
        # A budget can stop before rechecking an already-consumed source. Zero known
        # byte lag does not prove that this discovery/read pass completed.
        assert any("collection_incomplete" in w for w in envelope.warnings)
        assert all(
            w["partial"] for w in [scene.window(envelope, name)["coverage"] for name in ("24h",)]
        )
        assert result.state == "partial"
        lags.append(report.lag_bytes)
    assert lags == sorted(lags, reverse=True) and lags[0] <= total  # lag only ever shrinks
    assert lags[-1] < lags[0]
    assert scene.coverage(scene.current())["complete"] is True


def test_an_incomplete_tail_never_completes_until_its_newline_arrives(tmp_path: Path) -> None:
    backlog_scene(tmp_path, sessions=2)
    target = vs.projects(tmp_path) / "-work" / "cons1.jsonl"
    whole = target.read_bytes()
    last = whole.rstrip(b"\n").rfind(b"\n") + 1
    torn = whole[: last + 40]  # half a record, no newline
    target.write_bytes(torn)
    before = scene.collect(NOW)
    assert scene.report(before).complete is False  # a torn tail is not a complete source
    envelope = scene.current()
    watermark = next(
        w for w in scene.coverage(envelope)["watermarks"] if w["pending_tail_bytes"] > 0
    )
    assert watermark["offset"] + watermark["pending_tail_bytes"] == watermark["size"]
    assert watermark["offset"] == last  # the watermark stops at the last newline
    assert (
        scene.coverage(envelope)["complete"] is False
        and scene.coverage(envelope)["sources"]["complete"]
        < (scene.coverage(envelope)["sources"]["total"])
    )
    for _ in range(3):  # repeating changes nothing: still pending, still not complete
        again = scene.collect(NOW)
        assert not scene.report(again).complete and scene.report(again).new_evidence == 0
    held = {i for i in vs.evidence_identities() if i[1] == "cons1"}
    names = {record_id for _, _, record_id, _ in held}
    assert "call:t1b" in names and "result:t1b" not in names  # only the torn last line is absent

    target.write_bytes(whole)  # the writer finishes the line
    after = scene.collect(NOW)
    assert scene.report(after).complete and scene.report(after).new_evidence > 0
    done = vs.evidence_identities()
    assert len(set(done)) == len(done)
    # the completed tail now yields exactly what an unbudgeted run of the finished file yields
    reference_root = tmp_path / "ref"
    with vs.isolated(vs.env_for(reference_root)):
        backlog_scene(reference_root, sessions=2)
        scene.collect(NOW)
        assert vs.evidence_identities() == done


def test_a_record_larger_than_the_remaining_budget_completes_then_stops(tmp_path: Path) -> None:
    vs.enable(pin=True)
    vs.start_capture()
    big = "x" * 40_000
    records = [
        claude_prompt("big", big, uuid="u-big", ts=ts(12, 9)),
        claude_prompt("big", "second", uuid="u-two", ts=ts(12, 9, 1)),
        claude_prompt("big", "third", uuid="u-three", ts=ts(12, 9, 2)),
    ]
    vs.put(vs.projects(tmp_path), "big", records, CLOCK)
    first = scene.collect(NOW, budget=CycleBudget(bytes=1_000))
    report = scene.report(first)
    assert report.stopped_by == "byte_budget" and report.new_evidence >= 1
    assert report.bytes_read >= 40_000  # the started record finished (cooperative, not hard)
    assert report.lag_bytes > 0  # but nothing further was started: later records still unread
    rest = cycles_until_done(CycleBudget(bytes=1_000))
    assert scene.report(rest[-1]).complete
    identities = [i[2] for i in vs.evidence_identities()]
    assert len(identities) == len(set(identities)) == 3


def test_time_budget_stops_between_sources_and_the_next_cycle_finishes_the_rest(
    tmp_path: Path,
) -> None:
    backlog_scene(tmp_path)
    ticks = iter(range(0, 10_000, 11))  # every clock read advances 11 s: the 10 s target passes
    first = scene.collect(NOW, clock=lambda: float(next(ticks)))
    report = scene.report(first)
    assert report.stopped_by == "time_budget" and not report.complete
    assert report.lag_bytes > 0 and first.state == "partial"
    rest = cycles_until_done(CycleBudget())
    assert scene.report(rest[-1]).complete
    identities = vs.evidence_identities()
    assert len(identities) == len(set(identities))
    reference = tmp_path / "ref"
    with vs.isolated(vs.env_for(reference)):
        backlog_scene(reference)
        scene.collect(NOW)
        expected = vs.facts()
    assert expected.evidence  # the unbudgeted reference really read everything
    assert tuple(identities) == expected.evidence


def test_a_real_backlog_over_32_mib_resumes_under_the_default_budget(tmp_path: Path) -> None:
    vs.enable(pin=True)
    vs.start_capture()
    pad = "p" * 900
    records = [
        claude_prompt("huge", f"{pad}{i}", uuid=f"u{i}", ts=ts(12, 9, i // 3600 % 60, i % 60))
        for i in range(40_000)
    ]
    target = vs.put(vs.projects(tmp_path), "huge", records, CLOCK)
    assert target.stat().st_size > 32 * MIB
    first = scene.collect(NOW)  # default 32 MiB budget
    one = scene.report(first)
    assert one.stopped_by == "byte_budget" and not one.complete and one.lag_bytes > 0
    assert 30 * MIB <= one.bytes_read <= 33 * MIB  # bounded by the default, not by the file
    seen = len(vs.evidence_identities())
    reports = [one]
    for _ in range(8):
        report = scene.report(scene.collect(NOW))
        reports.append(report)
        if report.complete:
            break
    assert reports[-1].complete and reports[-1].lag_bytes == 0
    identities = [i[2] for i in vs.evidence_identities()]
    assert len(identities) == len(set(identities)) == 40_000 and seen < 40_000  # all of it, once
    assert sum(report.bytes_read for report in reports) >= target.stat().st_size


# ---- conflicting copies of one native event are counted once per dropped record, on any schedule


def variant(command: str) -> list[dict]:
    return [
        claude_prompt("s1", "hello", uuid="u1", ts="2026-10-12T09:00:00.000Z"),
        claude_tool_use(
            "s1", "t1", MCP, {"query": command}, ts="2026-10-12T09:00:05.000Z", message_id="m1"
        ),
    ]


def conflict_scene(root: Path) -> None:
    vs.enable()
    vs.start_capture()
    vs.put(vs.projects(root), "one", variant("alpha"), CLOCK)
    vs.put(vs.projects(root), "two", variant("beta"), CLOCK)
    vs.put(vs.projects(root), "three", variant("beta"), CLOCK)


def finish(budget: CycleBudget | None) -> tuple[dict, str, int]:
    cycles = 0
    while True:
        result = scene.collect(NOW, budget=budget)
        cycles += 1
        if scene.report(result).stopped_by == "complete":
            break
        assert cycles < 200
    envelope = scene.current()
    return scene.coverage(envelope)["gaps"], envelope.snapshot_id, cycles


@pytest.mark.parametrize("name", ["a", "eeeee", "g", "hh", "ffffff", "bb"])
def test_dropped_conflicts_do_not_depend_on_how_the_backlog_was_cut(
    tmp_path: Path, name: str
) -> None:
    root = tmp_path / name
    with vs.isolated(vs.env_for(root)):
        conflict_scene(root)
        one_gaps, one_id, one_cycles = finish(None)
        evidence = vs.evidence_identities()
    vs.wipe(root)
    with vs.isolated(vs.env_for(root)):
        conflict_scene(root)
        cut_gaps, cut_id, cut_cycles = finish(CycleBudget(bytes=1))  # one record per cycle
        assert vs.evidence_identities() == evidence  # same retained variant either way
    assert one_cycles == 1 and cut_cycles >= 3  # the schedules really differ
    assert cut_gaps == one_gaps
    assert cut_id == one_id
    dropped = one_gaps.get("dropped_conflict", 0)
    assert dropped >= 1  # the conflict is reported, never silently swallowed
    assert dropped + one_gaps.get("dropped_duplicate", 0) <= 2


# ---- collection control: validation first, recompute from retained evidence only -------------


def credit_scene(root: Path) -> None:
    vs.enable(pin=True)
    vs.start_capture()
    for n in (0, 1):
        vs.pair(vs.projects(root), n, CLOCK)


def test_the_library_also_refuses_a_backfill_without_an_interval_before_any_io(
    tmp_path: Path,
) -> None:
    vs.enable()
    vs.start_capture()
    before = vs.tree(tmp_path / "data")
    for options in (
        CollectOptions(backfill=True),
        CollectOptions(backfill=True, since="2026-10-01T00:00:00Z"),
        CollectOptions(backfill=True, until="2026-10-02T00:00:00Z"),
    ):
        with pytest.raises(Exception, match="requires_interval|needs both"):
            scene.collect(scene.at(hour=10), options)
    after = vs.tree(tmp_path / "data")
    assert {k: v for k, v in after.items() if "worker/" not in k} == {
        k: v for k, v in before.items() if "worker/" not in k
    }
    assert vs.checkpoint() is None  # no checkpoint, no snapshot: no counters changed
    assert not (vs.accounting_dir() / "snapshots").exists()


def test_recompute_does_not_import_live_appends_and_preserves_canonical_measurements(
    tmp_path: Path,
) -> None:
    from ssgrep.accounting.collect.state import load_state
    from ssgrep.accounting.ledger.contracts import canonical_bytes

    def measurements() -> dict:
        checkpoint = scene.committed()
        state = load_state(PROFILE, checkpoint, checkpoint.methodology)
        assert state.ledger is not None and state.ledger.debits

        def normalized(rows):
            return sorted(
                (
                    replace(
                        row,
                        provenance=tuple(
                            pair for pair in row.provenance if pair[0] != "frequency_revision"
                        ),
                    )
                    for row in rows
                ),
                key=canonical_bytes,
            )

        return {
            "methodology": state.ledger.methodology,
            "candidates": normalized(state.ledger.candidates),
            "debits": sorted(state.ledger.debits, key=canonical_bytes),
            "frozen": state.ledger.frozen,
        }

    credit_scene(tmp_path)
    scene.collect(scene.at(hour=10))
    base = vs.facts()
    before_measurements = measurements()
    vs.pair(vs.projects(tmp_path), 2, CLOCK)  # a NEW live record the recompute must not import
    result = scene.collect(
        scene.at(hour=11),
        CollectOptions(recompute=True, since="2026-10-12T00:00:00Z", until="2026-10-13T00:00:00Z"),
    )
    assert result.mode == "recompute" and result.unavailable_periods == ()  # nothing expired
    after = vs.facts()
    assert after.evidence == base.evidence  # the new transcript was not read
    # An explicit recompute changes only its named frequency revision and archives
    # old rows; every current quantity, footprint, identity and missingness is stable.
    assert measurements() == before_measurements
    assert scene.report(result).new_evidence == 0 and scene.report(result).bytes_read == 0
    follow = scene.collect(scene.at(hour=12))  # a normal collection imports it afterwards
    assert scene.report(follow).new_evidence > 0 and vs.facts().evidence != base.evidence


def test_missing_index_does_not_block_collection_and_is_reported_not_hidden(tmp_path: Path) -> None:
    credit_scene(tmp_path)
    assert not (tmp_path / "data" / "lancedb").exists()  # there is no search index at all
    result = scene.collect(scene.at(hour=10))
    assert result.state == "published" and scene.report(result).new_evidence > 0
    envelope = scene.current()
    assert scene.coverage(envelope)["index"] == "unavailable"  # said out loud ...
    assert (
        scene.window(envelope, "lifetime")["observations"]["searches"]["value"] == 3
    )  # ... usage counted
    assert not (tmp_path / "data" / "lancedb").exists()  # and collection never created one


def test_present_index_directory_is_never_opened_or_modified_by_collection(
    tmp_path: Path,
) -> None:
    credit_scene(tmp_path)
    index = tmp_path / "data" / "lancedb"
    index.mkdir(parents=True)
    (index / "table.lance").write_bytes(b"opaque index bytes")
    stamps = {p: p.stat().st_mtime_ns for p in index.rglob("*")}
    scene.collect(scene.at(hour=10))
    scene.collect(scene.at(hour=11), CollectOptions(recompute=True))
    assert scene.coverage(scene.current())["index"] == "available"
    assert {p: p.stat().st_mtime_ns for p in index.rglob("*")} == stamps
    assert (index / "table.lance").read_bytes() == b"opaque index bytes"


def test_a_corrupt_index_directory_does_not_stop_usage_collection(tmp_path: Path) -> None:
    credit_scene(tmp_path)
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "lancedb").write_text("not a directory at all")  # unusable index
    result = scene.collect(scene.at(hour=10))
    assert result.state == "published" and scene.report(result).complete
    assert scene.coverage(scene.current())["index"] == "unavailable"
    assert scene.window(scene.current(), "lifetime")["observations"]["searches"]["value"] == 3


def test_old_database_drop_diagnostics_count_admission_observations_not_usage(tmp_path: Path):
    from ssgrep.accounting.collect import store as collector_store

    db = oc_db(tmp_path)
    connection = opencode_connect(db)
    opencode_add_session(connection, "ses_1")
    oc_add_call(connection, 1, READERS_T0 + 100)
    oc_add_call(connection, 2, READERS_T0 + 200)
    native_env = env_for(tmp_path)
    assert datetime.fromtimestamp(READERS_T0 / 1000, UTC) < scene.at(day=1, hour=8)

    def drain(budget: CycleBudget):
        for _ in range(100):
            result = scene.collect(NOW, budget=budget, env=native_env)
            assert scene.report(result).status == "published"
            if scene.report(result).complete:
                return scene.current()
        raise AssertionError("old-record census did not converge")

    def part_visits() -> int:
        [record] = [row for row in scene.committed().sources.values() if row.runtime == "opencode"]
        visits = record.state["extra"]["opencode"]["rows_visited"]["part"]
        assert isinstance(visits, int) and not isinstance(visits, bool)
        return visits

    try:
        reference_env = {
            **vs.env_for(tmp_path),
            "SSGREP_DATA_DIR": str(tmp_path / "reference-data"),
        }
        with vs.isolated(reference_env):
            vs.enable()
            vs.start_capture(day=1)
            reference = drain(CycleBudget(bytes=BIG))
            reference_drops = scene.coverage(reference)["gaps"]["dropped_before_capture"]
            assert part_visits() >= 2
            # Each completed native part emits one call and one result. Fresh
            # and census visits are distinct admission observations, not usage.
            assert reference_drops == 2 * part_visits()

        vs.enable()
        vs.start_capture(day=1)
        resumed = drain(CycleBudget(bytes=1))  # Every cycle reloads persisted collector state.
        resumed_drops = scene.coverage(resumed)["gaps"]["dropped_before_capture"]
        assert resumed_drops == 2 * part_visits()
        assert resumed.observations == reference.observations
        assert resumed.windows == reference.windows
        assert resumed.estimates == reference.estimates
        assert resumed.source_manifest_digest == reference.source_manifest_digest
        if resumed_drops != reference_drops:
            assert resumed.snapshot_id != reference.snapshot_id  # Coverage is content, too.
        checkpoint = scene.committed()
        retained = list(collector_store.iter_evidence(PROFILE, checkpoint.segments))
        assert retained and all(item.kind == "session_header" for item in retained)
        idle = drain(CycleBudget(bytes=1))
        assert idle.snapshot_id == resumed.snapshot_id

        # A genuinely additional old tool remains excluded from imported usage,
        # while every actual fresh/census rejection remains visible diagnostically.
        oc_add_call(connection, 3, READERS_T0 + 300)
        appended = drain(CycleBudget(bytes=1))
        appended_drops = scene.coverage(appended)["gaps"]["dropped_before_capture"]
        assert appended_drops == 2 * part_visits()
        assert appended_drops >= resumed_drops + 2
        assert scene.counter_total(appended) == scene.counter_total(resumed)
    finally:
        connection.close()


# The reuse block of stored receipts: how reuse was established, from persisted facts only.
#
# Real end to end: native Claude transcripts, the real reducer through ``collect_once``, then the
# installed ``savings explain`` binary reading the files the collector published.


SECRET_QUERY = "why did the alpha deploy fail"
REUSE_KEYS = {"class", "relay", "report", "action_ok", "novel", "user_supplied"}


def consumer(n: int, kind: str) -> list[dict[str, Any]]:
    """Search, then act on the fact; ``contradicted`` errors, ``validated`` then runs pytest."""
    session = f"cons{n}"
    h, m = 9 + n, 10 * n
    records = [
        claude_prompt(session, "fix the deploy", uuid=f"u{n}", ts=ts(12, h, m)),
        claude_tool_use(
            session, f"t{n}a", MCP, {"query": vs.query(n)}, ts=ts(12, h, m, 10), message_id=f"a{n}"
        ),
        claude_tool_result(
            session,
            f"t{n}a",
            results_json(card(f"claude:src{n}:ep:0", vs.fact(n))),
            ts=ts(12, h, m, 12),
        ),
        claude_tool_use(
            session,
            f"t{n}b",
            "Bash",
            {"command": f"vim {vs.path(n)}"},
            ts=ts(12, h, m, 20),
            message_id=f"b{n}",
        ),
        claude_tool_result(
            session, f"t{n}b", "ok", is_error=kind == "contradicted", ts=ts(12, h, m, 22)
        ),
    ]
    if kind == "validated":
        records += [
            claude_tool_use(
                session,
                f"t{n}c",
                "Bash",
                {"command": "uv run pytest -q"},
                ts=ts(12, h, m, 30),
                message_id=f"c{n}",
            ),
            claude_tool_result(
                session, f"t{n}c", "12 passed in 0.4s", is_error=False, ts=ts(12, h, m, 35)
            ),
        ]
    return records


def reuse_build(root: Path) -> dict[int, str]:
    """Three real consumers (action_reference, validated, contradicted); class by session."""
    vs.enable(pin=True)
    vs.start_capture()
    base = vs.projects(root)
    kinds: dict[int, str] = {0: "action_reference", 1: "validated", 2: "contradicted"}
    for n, kind in kinds.items():
        vs.put(base, f"src{n}", source_session(f"src{n}", vs.fact(n), when_day=11), CLOCK)
        vs.put(base, f"cons{n}", consumer(n, kind), CLOCK)
        scene.journal_events(
            scene.execution(f"n-{n}", ts(12, 9 + n, 10 * n, 11), text=vs.query(n), transport="mcp")
        )
    result = scene.collect(NOW)
    assert result.state == "published" and scene.report(result).complete
    return kinds


def stored_blocks() -> dict[str, dict[str, Any]]:
    out = {}
    for path in sorted((vs.accounting_dir() / "receipts").glob("*.json")):
        document = json.loads(path.read_bytes())
        out[path.stem] = document.get("reuse")
    return out


def test_collected_receipts_record_how_reuse_was_established(tmp_path: Path) -> None:
    kinds = reuse_build(tmp_path)
    blocks = stored_blocks()
    assert sorted(b["class"] for b in blocks.values()) == sorted(kinds.values())
    for block in blocks.values():
        assert set(block) == REUSE_KEYS
        assert all(isinstance(block[k], bool) for k in REUSE_KEYS - {"class"})
        assert block["novel"] is True
    by_class = {b["class"]: b for b in blocks.values()}
    assert by_class["contradicted"]["action_ok"] is False
    assert by_class["validated"]["action_ok"] is True
    assert by_class["action_reference"]["action_ok"] is True


def test_block_holds_no_text_paths_or_facts(tmp_path: Path) -> None:
    reuse_build(tmp_path)
    for path in (vs.accounting_dir() / "receipts").glob("*.json"):
        raw = path.read_text()
        block = json.dumps(json.loads(raw)["reuse"])
        for forbidden in (vs.fact(1), vs.path(1), vs.query(1), "pytest", "/"):
            assert forbidden not in block
        assert vs.query(0) not in raw and vs.fact(0) not in raw


def test_savings_explain_shows_how_reuse_was_established(tmp_path: Path) -> None:
    reuse_build(tmp_path)
    by_class = {b["class"]: rid for rid, b in stored_blocks().items()}
    for kind, succeeded in (("validated", "yes"), ("contradicted", "no")):
        done = subprocess.run(
            [str(BINARY), "savings", "explain", by_class[kind]],
            capture_output=True,
            text=True,
            env={**os.environ, "NO_COLOR": "1", "COLUMNS": "100"},
            timeout=120,
            check=False,
        )
        assert done.returncode == 0, done.stderr
        assert "not recorded" not in done.stdout
        klass = next(x for x in done.stdout.splitlines() if "Reuse class" in x)
        assert klass.split()[-1] == kind
        line = next(x for x in done.stdout.splitlines() if "Acted-on step succeeded" in x)
        assert line.split()[-1] == succeeded


def test_ledger_digest_and_snapshot_id_ignore_the_block(tmp_path: Path) -> None:
    reuse_build(tmp_path)
    facts = vs.facts()
    receipts = vs.accounting_dir() / "receipts"
    for path in receipts.glob("*.json"):  # strip every block: an older collector's files
        document = json.loads(path.read_bytes())
        document.pop("reuse")
        path.write_bytes(canonical_bytes(document) + b"\n")
    assert vs.facts() == facts
    assert all(b is None for b in stored_blocks().values())


def test_older_files_without_a_block_still_read(tmp_path: Path) -> None:
    reuse_build(tmp_path)
    rid = next(iter(stored_blocks()))
    path = vs.accounting_dir() / "receipts" / f"{rid}.json"
    document = json.loads(path.read_bytes())
    document.pop("reuse")
    path.write_bytes(canonical_bytes(document) + b"\n")
    view = service.read_receipt(PROFILE, rid)
    assert view is not None and view.reuse is None


@pytest.mark.parametrize("stale", [b"", b"{"])
def test_rewrite_without_fresh_facts_keeps_the_stored_block(tmp_path: Path, stale: bytes) -> None:
    reuse_build(tmp_path)
    before = stored_blocks()
    state = vs.checkpoint()
    assert state is not None
    from ssgrep.accounting.collect.state import VersionedLedger, load_state

    loaded = load_state(PROFILE, state, state.methodology)
    assert loaded.ledger is not None
    ledger = VersionedLedger.restore_versioned(loaded.ledger, loaded.superseded)
    files = vs.accounting_dir() / "receipts"
    for path in files.glob("*.json"):
        path.unlink()
        path.write_bytes(stale)  # unreadable stale file must not raise nor invent a block
        path.chmod(0o600)
    publish.write_receipts(PROFILE, ledger.rows())
    assert all(b is None for b in stored_blocks().values())
    publish.write_receipts(
        PROFILE,
        ledger.rows(),
        publish.reuse_blocks(loaded.ledger.candidates, ledger.methodology, loaded.superseded),
    )
    assert stored_blocks() == before
    publish.write_receipts(PROFILE, ledger.rows())  # no fresh blocks: stored ones survive
    assert stored_blocks() == before


def test_block_is_deterministic_across_collections(tmp_path: Path) -> None:
    reuse_build(tmp_path)
    first = {p.name: p.read_bytes() for p in (vs.accounting_dir() / "receipts").glob("*.json")}
    scene.collect(NOW, scene.LIVE)
    second = {p.name: p.read_bytes() for p in (vs.accounting_dir() / "receipts").glob("*.json")}
    assert first == second


def test_superseded_reuse_does_not_leak_into_the_current_receipt_block() -> None:
    old = candidate("one")
    newer = replace(old, reuse=Reuse.VALIDATED, report_reference=True, accepted_day="2026-11-02")
    assert publish.reuse_blocks([old], METHOD, {digest_of(old)}) == {}
    blocks = publish.reuse_blocks([old, newer], METHOD, {digest_of(old)})
    assert len(blocks) == 1
    block = next(iter(blocks.values()))
    assert block["class"] == Reuse.VALIDATED.value and block["report"] is True


@pytest.mark.parametrize("field", ["class", "action_ok"])
def test_rewriting_receipts_drops_untrusted_stored_reuse_but_keeps_valid_blocks(
    tmp_path: Path, field: str
) -> None:
    reuse_build(tmp_path)
    ledger, _ = current_ledger()
    rows = ledger.rows()
    blocks = publish.reuse_blocks(ledger.state().candidates, ledger.methodology)
    publish.write_receipts(PROFILE, rows, blocks)
    directory = vs.accounting_dir() / "receipts"
    receipt_id = next(iter(blocks))
    path = directory / f"{receipt_id}.json"
    valid = json.loads(path.read_bytes())
    assert valid["reuse"] == blocks[receipt_id]
    publish.write_receipts(PROFILE, rows)
    assert json.loads(path.read_bytes())["reuse"] == blocks[receipt_id]
    damaged = json.loads(path.read_bytes())
    damaged["reuse"][field] = "bogus" if field == "class" else 1
    path.write_text(json.dumps(damaged))
    publish.write_receipts(PROFILE, rows)
    assert "reuse" not in json.loads(path.read_bytes())


# Canonical root evaluations survive restarts without rerating unrelated roots.


def test_restored_canonical_root_rows_are_identical_without_revaluation(monkeypatch):
    ledger = VersionedLedger(METHOD)
    ledger.reconcile(
        {"root-1": EvidenceBundle((candidate("a"),), (debit("d", 40),))}, today="2026-11-02"
    )
    expected = ledger.rows().canonical_bytes()
    blobs = {}
    save_cache("personal-local", ledger, blobs)
    restored = VersionedLedger.restore_versioned(ledger.state(), ledger.superseded_state())
    restore_cache("personal-local", restored, blobs)
    import ssgrep.accounting.collect.replay as replay

    actual = replay.value_evidence
    calls = []

    def observed(*args, **kwargs):
        calls.append(args)
        return actual(*args, **kwargs)

    monkeypatch.setattr(replay, "value_evidence", observed)
    control = VersionedLedger(METHOD)
    control.reconcile(
        {"root-1": EvidenceBundle((candidate("a"),), (debit("d", 40),))}, today="2026-11-02"
    )
    assert control.rows().canonical_bytes() == expected
    assert calls  # positive control: the spy observes a fresh, uncached valuation
    calls.clear()
    assert restored.rows().canonical_bytes() == expected
    assert calls == []


def test_root_cache_keeps_named_archives_and_is_invalidated_by_privacy_purge():
    ledger = VersionedLedger(METHOD)
    ledger.reconcile({"root-1": EvidenceBundle((candidate("a"),), ())}, today="2026-11-02")
    ledger.archive_current("original")
    original = ledger.archives_state()
    ledger.rows()
    assert ledger._eval_cache
    ledger.purge(roots=("root-1",))
    assert not ledger._eval_cache
    assert not ledger.rows().receipts
    assert not ledger.archives_state()
    assert original["original"].receipts


def test_root_cache_with_malformed_root_rows_is_rejected_as_committed_corruption():
    ledger = VersionedLedger(METHOD)
    reference = store.write_blob(
        PROFILE,
        store.ROOT_EVALUATIONS_BLOB,
        {
            "schema": 1,
            "methodology": ledger.methodology.hash,
            "roots": {"root-1": store.write_blob(PROFILE, "root-evaluation", {})},
        },
    )
    restored = VersionedLedger.restore_versioned(
        ledger.state(), ledger.superseded_state(), ledger.archives_state()
    )

    with pytest.raises(store.CollectError) as error:
        restore_cache(PROFILE, restored, {store.ROOT_EVALUATIONS_BLOB: reference})
    assert error.value.condition == "state_corrupt"


def test_root_cache_manifest_must_be_valid_and_only_its_own_methodology_is_restored() -> None:
    from ssgrep.accounting.collect.store import root_cache_references

    ledger = VersionedLedger(METHOD)
    ledger.reconcile({"root-1": EvidenceBundle((candidate("a"),), ())}, today="2026-11-02")
    ledger.rows()
    blobs: dict[str, str] = {}
    save_cache(PROFILE, ledger, blobs)
    manifest = store.read_blob(PROFILE, blobs[store.ROOT_EVALUATIONS_BLOB])
    assert manifest is not None
    assert root_cache_references(PROFILE, blobs[store.ROOT_EVALUATIONS_BLOB]) == set(
        manifest["roots"].values()
    )

    def restored_cache(reference: str, versioned=None) -> VersionedLedger:
        target = versioned or VersionedLedger.restore_versioned(
            ledger.state(), ledger.superseded_state()
        )
        restore_cache(PROFILE, target, {store.ROOT_EVALUATIONS_BLOB: reference})
        return target

    assert restored_cache(blobs[store.ROOT_EVALUATIONS_BLOB])._eval_cache
    other = store.write_blob(PROFILE, store.ROOT_EVALUATIONS_BLOB, {**manifest, "methodology": "x"})
    assert not restored_cache(other)._eval_cache  # another method's rows are never borrowed
    for damaged in (
        {**manifest, "schema": 2},
        {**manifest, "roots": ["not", "a", "mapping"]},
        ["x"],
    ):
        reference = store.write_blob(PROFILE, store.ROOT_EVALUATIONS_BLOB, damaged)
        with pytest.raises(CollectError) as failed:
            root_cache_references(PROFILE, reference)
        assert failed.value.condition == "state_corrupt"
    for damaged in ({**manifest, "schema": 2}, ["x"]):
        reference = store.write_blob(PROFILE, store.ROOT_EVALUATIONS_BLOB, damaged)
        with pytest.raises(CollectError) as failed:
            restored_cache(reference)
        assert failed.value.condition == "state_corrupt"


def test_a_retained_segment_that_is_oversized_or_undecodable_is_state_corruption() -> None:
    from ssgrep.accounting.privacy import storage

    item = build_evidence()
    (ref,) = store.write_segments(PROFILE, [item])
    path = (
        storage.data_root() / "accounting" / PROFILE / "collector" / store.EVIDENCE_DIR / ref.name
    )
    original = path.read_bytes()
    assert [row.native_key for row in store.read_segment(PROFILE, ref)] == [item.native_key]
    for body in (original + b"x" * (len(original) + 1), b"x" * ref.size):
        path.write_bytes(body)
        with pytest.raises(CollectError) as failed:
            list(store.read_segment(PROFILE, ref))
        assert failed.value.condition == "state_corrupt"


def test_checkpoint_reference_walk_follows_pending_work_and_rejects_invalid_manifests() -> None:
    archive = "identity-archive-native_identity_v2"
    staged = store.SegmentRef("seg-staged.jsonl", 1, 10)
    legacy = store.SegmentRef("seg-legacy.jsonl", 1, 10)
    child = store.write_blob(
        PROFILE, archive + "-child", {"schema": 1, "segments": [], "blobs": {}}
    )
    nested_segment = store.SegmentRef("seg-nested.jsonl", 1, 10)
    top = store.write_blob(
        PROFILE,
        archive,
        {
            "schema": 1,
            "segments": [nested_segment.to_dict()],
            "blobs": {"identity-archive-child": child, "payload": "payload-blob"},
        },
    )
    job = store.write_blob(
        PROFILE, "frequency-maintenance", {"schema": 1, "sources": [["view", "view-blob"]]}
    )
    checkpoint = store.Checkpoint(
        PROFILE,
        blobs={archive: top, "frequency-maintenance": job},
        identity_migration={
            "staged_segments": [staged.to_dict()],
            "compact_legacy_segments": [legacy.to_dict()],
        },
    )
    segments, blobs = store.referenced_files(PROFILE, checkpoint)
    assert {"seg-staged.jsonl", "seg-legacy.jsonl", "seg-nested.jsonl"} <= segments
    assert {top, child, job, "payload-blob", "view-blob"} <= blobs

    def condition(damaged: store.Checkpoint) -> str:
        with pytest.raises(CollectError) as failed:
            store.referenced_files(PROFILE, damaged)
        return failed.value.condition

    assert condition(store.Checkpoint(PROFILE, identity_migration={"published_capture": 7})) == (
        "state_corrupt"
    )
    bad_job = store.write_blob(PROFILE, "frequency-maintenance", ["not", "a", "job"])
    assert condition(store.Checkpoint(PROFILE, blobs={"frequency-maintenance": bad_job})) == (
        "state_corrupt"
    )
    bad_archive = store.write_blob(PROFILE, archive, {"schema": 2})
    assert condition(store.Checkpoint(PROFILE, blobs={archive: bad_archive})) == "state_corrupt"
    same = store.Checkpoint(PROFILE, blobs={archive: top, archive + "-alias": top})
    assert store.referenced_files(PROFILE, same)[1] >= {top, child}
    many = {
        f"identity-archive-{number:02}": store.write_blob(
            PROFILE, archive, {"schema": 1, "segments": [], "blobs": {}, "serial": number}
        )
        for number in range(33)
    }
    assert condition(store.Checkpoint(PROFILE, blobs=many)) == "state_corrupt"


def test_evidence_migration_is_bounded_per_child_and_rejects_unsafe_archives() -> None:
    from ssgrep.accounting.collect.store import MAX_ATOMIC_SEGMENT_BYTES, migrate_evidence

    oversized = store.SegmentRef("seg-too-large.jsonl", 1, MAX_ATOMIC_SEGMENT_BYTES + 1)
    with pytest.raises(CollectError) as failed:
        migrate_evidence(
            PROFILE,
            store.Checkpoint(PROFILE, segments=[oversized]),
            byte_budget=MAX_ATOMIC_SEGMENT_BYTES + 2,
        )
    assert failed.value.condition == "evidence_migration_child_bound"

    archive = "identity-archive-native_identity_v2"
    child = store.SegmentRef("seg-archived.jsonl", 1, 1_000)
    manifest = {"schema": 1, "segments": [child.to_dict()], "blobs": {}}
    reference = store.write_blob(PROFILE, archive, manifest)
    manifest_size = storage_state_size(reference)
    checkpoint = store.Checkpoint(PROFILE, blobs={archive: reference})

    smaller = migrate_evidence(PROFILE, checkpoint, byte_budget=manifest_size - 1)
    assert smaller.retention["evidence_compaction"]["blocked_child_bytes"] == manifest_size
    assert smaller.retention["evidence_migration_pending"]
    tighter = migrate_evidence(PROFILE, checkpoint, byte_budget=manifest_size + child.size - 1)
    assert tighter.retention["evidence_compaction"]["blocked_child_bytes"] == (
        manifest_size + child.size
    )

    damaged = store.write_blob(PROFILE, archive, {"schema": 2, "segments": []})
    with pytest.raises(CollectError) as unsafe:
        migrate_evidence(PROFILE, store.Checkpoint(PROFILE, blobs={archive: damaged}))
    assert unsafe.value.condition == "state_corrupt"


def storage_state_size(reference: str) -> int:
    from ssgrep.accounting.privacy import storage

    return (
        (storage.data_root() / "accounting" / PROFILE / "collector" / "state" / reference)
        .stat()
        .st_size
    )


# Stat-stable source reads and unique-source worker progress.


def test_stable_native_source_is_not_parsed_again_with_positive_ingest(tmp_path, monkeypatch):
    write_records(claude_dir(tmp_path) / "stable.jsonl", claude_records("stable", 2))
    cycle = Cycle(tmp_path)
    first = cycle.run()
    assert first.evidence and first.bytes_read > 0
    calls = []
    original = cycle.readers["claude"].read

    def observed(*args, **kwargs):
        calls.append(args[0].path)
        return original(*args, **kwargs)

    monkeypatch.setattr(cycle.readers["claude"], "read", observed)
    idle = cycle.run()
    assert idle.bytes_read == 0 and idle.evidence == []
    assert all(record.complete for record in idle.sources.values())
    assert calls == []


def test_same_size_native_mutation_is_not_hidden_by_fast_skip(tmp_path, monkeypatch):
    path = claude_dir(tmp_path) / "stable.jsonl"
    write_records(path, claude_records("stable", 2))
    cycle = Cycle(tmp_path)
    cycle.run()
    original_stamp = path.stat().st_mtime_ns
    body = path.read_bytes().replace(b"echo 1", b"echo 9")
    path.write_bytes(body)
    os.utime(path, ns=(original_stamp, original_stamp))
    changed = cycle.run()
    assert changed.bytes_read > 0 and changed.resets > 0
    assert changed.dropped.get("conflict", 0) > 0


def test_unvisited_database_pending_count_is_unknown_not_zero():
    counts = source_progress([SourceRecord("opencode", "opencode:/fixture/db")])
    assert counts["sources_unvisited"] == 1
    assert counts["sources_complete"] == 0
    assert counts["pending_records"] is None
    assert not counts["pending_count_complete"]


def test_read_stage_counts_unique_discovered_sources_not_quantum_callbacks(tmp_path, monkeypatch):
    scene.install_key()
    scene.journal_events(scene.start_marker())
    scene.write_transcript(
        tmp_path, "left", scene.search_records("left", "call", "q", (10, 0, 0)), scene.at()
    )
    original = collector_live.read_sources

    def repeated_callback(**kwargs):
        outcome = original(**kwargs)
        for key, record in outcome.sources.items():
            kwargs["on_source"](key, record)
            kwargs["on_source"](key, record)
        return outcome

    monkeypatch.setattr(collector_live, "read_sources", repeated_callback)
    scene.collect(scene.at())
    # Inspect actual worker output, not the count of callback invocations.
    from ssgrep.accounting.collect.collector import read_progress

    progress = read_progress("personal-local")
    assert progress is not None
    assert progress["sources_done"] <= progress["sources_total"] == 1
    assert progress["sources_complete"] == 1


# Selected native replay is exact, authorized and byte-linear per affected source.


def native_items(root):
    enable_collection(PROFILE)
    records = [claude_prompt("batch", "investigate", ts=ts())]
    for n in range(200):
        records.extend(scene.search_records("batch", f"c-{n}", f"query-{n}", (10, 0, 0)))
    path = scene.write_transcript(root, "batch", records, scene.at())
    found = discover(make_readers())
    reader, source = next(iter(found.values()))
    decoded = reader.read(source, None).evidence
    refs = store.write_segments(PROFILE, decoded)
    compact = list(store.iter_evidence(PROFILE, refs))
    return path, decoded, compact


def test_batch_replays_every_selected_identity_with_linear_native_bytes(tmp_path):
    path, expected, compact = native_items(tmp_path)
    replay = EvidenceReplay(PROFILE)
    actual = replay.hydrate_many(compact)
    assert not replay.unavailable
    assert [(row.identity, row.ts_utc) for row in actual] == [
        (row.identity, row.ts_utc) for row in expected
    ]
    assert [row.payload for row in actual] == [row.payload for row in expected]
    assert replay.adapter.bytes_read <= path.stat().st_size + 64 * 1024 + len(compact)


def test_batch_preserves_per_record_drift_and_missingness(tmp_path):
    path, expected, compact = native_items(tmp_path)
    changed = replace(compact[-1], digest="0" * 64)
    replay = EvidenceReplay(PROFILE)
    actual = replay.hydrate_many([*compact[:-1], changed])
    assert actual[-1].payload["_replay_unavailable"] == "source_digest_drift"
    assert actual[0].identity == expected[0].identity
    assert replay.unavailable[("claude", "batch")]
    path.unlink()
    missing = EvidenceReplay(PROFILE)
    missing.hydrate_many(compact)
    assert missing.unavailable[("claude", "batch")] == "source_missing"


def test_a_started_native_record_finishes_without_starting_the_next_over_budget(tmp_path):
    from ssgrep.accounting.ledger.reduce import native_source_adapter
    from ssgrep.accounting.privacy.policy import load_profile_policy

    path, expected, compact = native_items(tmp_path)
    lines = path.read_bytes().splitlines(keepends=True)
    budget = len(lines[0]) + len(lines[1]) // 2
    for selected_count in (2, 3):
        adapter = native_source_adapter(load_profile_policy(PROFILE), now_ms=0)
        adapter.byte_budget = budget
        replay = EvidenceReplay(PROFILE, adapter=adapter)
        actual = replay.hydrate_many(compact[:selected_count])
        assert [row.identity for row in actual[:2]] == [row.identity for row in expected[:2]]
        assert [row.payload for row in actual[:2]] == [row.payload for row in expected[:2]]
        if selected_count == 3:
            assert actual[2].payload["_replay_unavailable"] == "source_read_bound"
        else:
            assert not replay.unavailable
        assert adapter.bytes_read == len(lines[0]) + len(lines[1])
        assert budget < adapter.bytes_read <= budget + max(map(len, lines))


def test_batch_replay_rejects_unresolved_source_without_losing_valid_control(tmp_path):
    path, expected, compact = native_items(tmp_path)
    unresolved = replace(compact[0], source_id="wrong-runtime:" + compact[0].source_id)
    replay = EvidenceReplay(PROFILE)

    actual = replay.hydrate_many([compact[0], unresolved])

    assert actual[0].identity == expected[0].identity
    assert actual[0].payload.get("_replay_unavailable") is None
    assert actual[1].payload["_replay_unavailable"] == "source_unresolved"
    assert replay.unavailable[("claude", "batch")] == "source_unresolved"
    assert path.exists()


def batch_adapter(**changes):
    from ssgrep.accounting.ledger.reduce import native_source_adapter
    from ssgrep.accounting.privacy.policy import load_profile_policy

    adapter = native_source_adapter(load_profile_policy(PROFILE), now_ms=1_000)
    for name, value in changes.items():
        setattr(adapter, name, value)
    return adapter


def batch_reference(item, **changes):
    from ssgrep.accounting.readers.native_source import NativeSourceReference

    return NativeSourceReference(
        item.source_id,
        item.runtime,
        item.session_id,
        Path(item.source_id[len(item.runtime) + 1 :]),
        **changes,
    )


def batch_statuses(rows):
    return [row.status.value for row in rows]


def test_read_batch_without_rows_reads_nothing_and_expired_references_never_open_the_source(
    tmp_path,
):
    from ssgrep.accounting.collect.replay import read_batch

    _, expected, compact = native_items(tmp_path)
    adapter = batch_adapter()
    assert read_batch(adapter, batch_reference(compact[0]), []) == []
    assert adapter.bytes_read == 0

    expired = read_batch(adapter, batch_reference(compact[0], expires_at_ms=1_000), compact[:2])
    assert batch_statuses(expired) == ["source_expired", "source_expired"]
    assert adapter.bytes_read == 0

    live = read_batch(adapter, batch_reference(compact[0], expires_at_ms=1_001), compact[:2])
    assert batch_statuses(live) == ["ok", "ok"]
    assert [row.evidence.identity for row in live] == [row.identity for row in expected[:2]]


def test_read_batch_reports_an_exhausted_byte_budget_before_opening_the_source(tmp_path):
    from ssgrep.accounting.collect.replay import read_batch

    _, _, compact = native_items(tmp_path)
    spent = batch_adapter(byte_budget=100, bytes_read=100)
    rows = read_batch(spent, batch_reference(compact[0]), compact[:2])
    assert batch_statuses(rows) == ["source_read_bound", "source_read_bound"]
    assert spent.bytes_read == 100
    fresh = read_batch(batch_adapter(), batch_reference(compact[0]), compact[:2])
    assert batch_statuses(fresh) == ["ok", "ok"]


def test_read_batch_rejects_foreign_locators_and_only_misses_offsets_past_the_source(tmp_path):
    from ssgrep.accounting.collect.replay import read_batch

    path, expected, compact = native_items(tmp_path)
    reference = batch_reference(compact[0])
    garbled = replace(compact[1], locator="line:2")
    rows = read_batch(batch_adapter(), reference, [compact[0], garbled])
    assert batch_statuses(rows) == ["source_unresolved", "source_unresolved"]

    past_end = replace(compact[1], locator=f"offset:{path.stat().st_size + 10}")
    rows = read_batch(batch_adapter(), reference, [compact[0], past_end])
    assert batch_statuses(rows) == ["ok", "source_missing"]
    assert rows[0].evidence.identity == expected[0].identity


def test_read_batch_stops_at_the_budget_before_reaching_a_distant_target(tmp_path):
    from ssgrep.accounting.collect.replay import read_batch

    path, expected, compact = native_items(tmp_path)
    first_line = len(path.read_bytes().splitlines(keepends=True)[0])
    target = compact[4]
    assert int(target.locator.split(":")[1]) > first_line
    adapter = batch_adapter(byte_budget=first_line)
    rows = read_batch(adapter, batch_reference(target), [target])
    assert batch_statuses(rows) == ["source_read_bound"]
    assert adapter.bytes_read == first_line
    ok = read_batch(batch_adapter(), batch_reference(target), [target])
    assert ok[0].evidence.identity == expected[4].identity


def test_read_batch_distinguishes_oversized_and_unterminated_records(tmp_path):
    from ssgrep.accounting.collect.replay import read_batch
    from ssgrep.accounting.ledger.contracts import MAX_RAW_RECORD_BYTES

    path, _, compact = native_items(tmp_path)
    lines = path.read_bytes().splitlines(keepends=True)
    target = compact[2]
    assert int(target.locator.split(":")[1]) == len(lines[0]) + len(lines[1])
    assert read_batch(batch_adapter(), batch_reference(target), [target])[0].evidence is not None

    # A mid-record locator cannot see the end of the intervening record.
    midway = replace(target, locator=f"offset:{len(lines[0]) // 2}")
    assert batch_statuses(read_batch(batch_adapter(), batch_reference(midway), [midway])) == [
        "source_incomplete"
    ]

    huge = b'{"padding":"' + b"x" * (MAX_RAW_RECORD_BYTES + 16) + b'"}\n'
    path.write_bytes(huge + b"".join(lines[1:]))
    far = replace(target, locator=f"offset:{len(huge) + len(lines[1])}")
    # The oversized record sits between the file start and the target.
    assert batch_statuses(read_batch(batch_adapter(), batch_reference(far), [far])) == [
        "source_read_bound"
    ]
    # The oversized record IS the target.
    first = replace(compact[0], locator="offset:0")
    assert batch_statuses(read_batch(batch_adapter(), batch_reference(first), [first])) == [
        "source_read_bound"
    ]

    path.write_bytes(b"".join(lines)[:-1])
    last = compact[-1]
    assert int(last.locator.split(":")[1]) == len(b"".join(lines[:-1]))
    assert batch_statuses(read_batch(batch_adapter(), batch_reference(last), [last])) == [
        "source_incomplete"
    ]


def test_read_batch_refuses_records_that_are_not_the_selected_native_identity(tmp_path):
    from ssgrep.accounting.collect.replay import read_batch

    _, expected, compact = native_items(tmp_path)
    reference = batch_reference(compact[0])
    stranger = replace(compact[1], origin_id="unrelated-record")
    rows = read_batch(batch_adapter(), reference, [compact[0], stranger])
    assert batch_statuses(rows) == ["ok", "source_identity_mismatch"]
    assert rows[0].evidence.identity == expected[0].identity

    # Same native identity, but the retained row claims another owning session.
    reowned = replace(compact[2], session_id="forked", origin_domain=compact[2].session_id)
    assert reowned.native_identity == compact[2].native_identity
    assert reowned.identity == compact[2].identity
    rows = read_batch(batch_adapter(), reference, [compact[0], reowned])
    assert batch_statuses(rows) == ["ok", "source_identity_mismatch"]


def pi_native_items(tmp_path):
    enable_collection(PROFILE)
    records = [
        pi_header("p1"),
        pi_tool_call("e1", "c1", "bash", {"command": "ls"}),
        pi_tool_result("e2", "c1", "listing"),
    ]
    path = write_records(tmp_path / "pi-sessions" / "-work" / "p1.jsonl", records)
    found = discover(make_readers())
    reader, source = next(iter(found.values()))
    decoded = reader.read(source, None).evidence
    refs = store.write_segments(PROFILE, decoded)
    compact = [
        row
        for row in store.iter_evidence(PROFILE, refs)
        if row.kind in {"tool_call", "tool_result"}
    ]
    assert [row.runtime for row in compact] == ["pi", "pi"]
    return path, compact


def test_read_batch_requires_the_native_session_header_to_own_the_selected_record(tmp_path):
    from ssgrep.accounting.collect.replay import read_batch

    path, compact = pi_native_items(tmp_path)
    reference = batch_reference(compact[0])
    rows = read_batch(batch_adapter(), reference, compact)
    assert batch_statuses(rows) == ["ok", "ok"]

    foreign = replace(reference, session_id="someone-else")
    assert batch_statuses(read_batch(batch_adapter(), foreign, compact)) == [
        "source_identity_mismatch",
        "source_identity_mismatch",
    ]

    # Overwrite the header in place (same byte length) so the file declares no owner.
    lines = path.read_bytes().splitlines(keepends=True)
    blank = b'{"type":"note"}'.ljust(len(lines[0]) - 1) + b"\n"
    path.write_bytes(blank + b"".join(lines[1:]))
    assert len(blank) == len(lines[0])
    assert batch_statuses(read_batch(batch_adapter(), reference, compact)) == [
        "source_identity_mismatch",
        "source_identity_mismatch",
    ]


class GrowingSource:
    """Scope authority whose approval check coincides with a writer appending to the source."""

    def __init__(self, scoper, path, tail):
        self.scoper, self.path, self.tail, self.appended = scoper, path, tail, False

    def source_status(self, cwd):
        if not self.appended:
            append_bytes(self.path, self.tail)
            self.appended = True
        return self.scoper.source_status(cwd)


def test_a_source_growing_during_replay_is_accepted_only_within_the_verification_budget(tmp_path):
    from ssgrep.accounting.collect.replay import read_batch

    path, expected, compact = native_items(tmp_path)
    first_line = len(path.read_bytes().splitlines(keepends=True)[0])
    tail = jsonl([claude_prompt("batch", "later", uuid="u-late", ts=ts())])
    target = compact[0]

    generous = batch_adapter()
    generous.scoper = GrowingSource(generous.scoper, path, tail)
    rows = read_batch(generous, batch_reference(target), [target])
    assert generous.scoper.appended and path.read_bytes().endswith(tail)
    assert batch_statuses(rows) == ["ok"]
    assert rows[0].evidence.identity == expected[0].identity
    assert generous.bytes_read == 2 * first_line  # read once, re-read once to prove the prefix

    path.write_bytes(path.read_bytes()[: -len(tail)])
    tight = batch_adapter(byte_budget=first_line)
    tight.scoper = GrowingSource(tight.scoper, path, tail)
    rows = read_batch(tight, batch_reference(target), [target])
    assert tight.scoper.appended
    assert batch_statuses(rows) == ["source_read_bound"]


def test_closed_periods_are_unavailable_only_when_evidence_has_expired():
    from ssgrep.accounting.collect.steps import unavailable_periods

    assert unavailable_periods(
        frozen_days={"2026-01-01"},
        sealed_days={"2026-01-02", "2026-01-03", "2026-01-04"},
        retained_days={"2026-01-03"},
        since="2026-01-01",
        until="2026-01-04",
        horizon_day="2026-01-04",
    ) == ("2026-01-01=evidence_expired", "2026-01-02=evidence_expired")


def test_frequency_reference_walk_finds_only_named_frozen_views():
    from ssgrep.accounting.collect.state import frequency_refs

    value = {
        "view_id": "freq-" + "a" * 24,
        "nested": [
            ["frequency", "freq-" + "b" * 24],
            ["frequency_view", "freq-" + "c" * 24],
            ["token", "freq-" + "d" * 24],
            {"frequency_view_id": "not-a-view"},
        ],
        "ordinary": "freq-" + "e" * 24,
    }
    assert frequency_refs(value) == {
        "freq-" + "a" * 24,
        "freq-" + "b" * 24,
        "freq-" + "c" * 24,
    }


def test_retained_evidence_compact_measurements_filter_unknown_and_boolean_values():
    from ssgrep.accounting.collect.replay import _measurements, compact_payload

    assert _measurements(None) is None
    assert _measurements(True) is True
    assert _measurements("private text") is None
    assert _measurements(
        {"input_tokens": 10, "input": 4, "total": True, "output": 6, "secret": 100}
    ) == {"input_tokens": 10, "input": 4, "output": 6}
    payload = compact_payload(
        {
            "call_id": "kept",
            "prompt": "discarded",
            "input_tokens": 10,
            "usage": {"input": 4, "output": 6, "secret": "discarded"},
            "target_native_key": ("runtime", "session", "record"),
        }
    )
    assert payload["call_id"] == "kept"
    assert "prompt" not in payload and payload["input_tokens"] == 10
    assert payload["usage"] == {"input": 4, "output": 6}
    assert payload["target_native_key"] == ["runtime", "session", "record"]


def test_plaintext_and_compact_evidence_line_round_trip_keep_their_contract():
    from ssgrep.accounting.collect.replay import evidence_from_line

    item = build_evidence(payload={"cwd": "/approved", "prompt": "not persisted in compact"})
    compact = evidence_from_line(evidence_line(item))
    plain = evidence_from_line(evidence_line(item, plaintext=True))
    assert compact.identity == plain.identity == item.identity
    assert compact.payload["_evidence_format"] == "compact-v1"
    assert "prompt" not in compact.payload
    assert plain.payload["_evidence_format"] == "plaintext-cache-v1"
    assert plain.payload["prompt"] == "not persisted in compact"


def test_retention_child_index_proves_only_matching_clean_cutoffs():
    assert collector._clean({}, "receipts", "original", "2026-01-02") is False
    docs = {
        "retention_child_dates": {
            "receipts": {
                "schema": 1,
                "source": "original",
                "clean_before": "2026-01-03",
                "complete": False,
                "minimum": None,
            }
        }
    }
    assert collector._clean(docs, "receipts", "different", "2026-01-01") is False
    assert collector._clean(docs, "receipts", "original", "2026-01-02") is True
    assert collector._clean(docs, "receipts", "original", "2026-01-04") is False
    docs["retention_child_dates"]["receipts"].update(
        clean_before=None, complete=True, minimum="2026-01-05"
    )
    assert collector._clean(docs, "receipts", "original", "2026-01-04") is True
    assert collector._clean(docs, "receipts", "original", "2026-01-06") is False


def test_day_range_helper_handles_fallback_empty_and_partial_last_day():
    from ssgrep.accounting.collect.steps import _days

    assert _days(None, "2026-01-02", ["fallback"]) == ["fallback"]
    assert _days("2026-01-02", "2026-01-02", []) == []
    assert _days("2026-01-02", "2026-01-02T12:00:00Z", []) == ["2026-01-02"]


def test_segment_iteration_and_indexed_selection_round_trip_evidence():
    item = build_evidence()
    expected = replace(item, payload={"_evidence_format": "compact-v1"})
    refs = store.write_segments(PROFILE, [item])
    assert list(store.iter_evidence(PROFILE, refs)) == [expected]
    from ssgrep.accounting.collect.cycle import iter_selected

    assert list(iter_selected(PROFILE, refs[0])) == [expected]


def test_partition_combine_rejects_two_accepted_documents_for_one_root():
    from ssgrep.accounting.collect.cycle_publish import combine
    from ssgrep.accounting.collect.frequency import FrequencyAssignments, FrequencyView
    from ssgrep.accounting.collect.replay import Reduction
    from ssgrep.accounting.ledger.valuation_inputs import EvidenceBundle

    first = Reduction("personal-local", bundles=(EvidenceBundle(),), bundle_roots=("root",))
    second = Reduction("personal-local", bundles=(EvidenceBundle(),), bundle_roots=("root",))
    parts = {
        name: {"blob": store.write_blob(PROFILE, "partition-test", to_plain(reduction))}
        for name, reduction in (("a", first), ("b", second))
    }
    assignments = FrequencyAssignments(FrequencyView.unknown())
    with pytest.raises(CollectError, match="partition roots overlap"):
        combine(PROFILE, parts, assignments)


def test_replay_cache_policy_enforces_enabled_bounded_retention():
    from dataclasses import replace

    from ssgrep.accounting.collect.replay_evidence import _bounded_policy
    from ssgrep.accounting.ledger.ledger_contracts import ProfilePolicy

    disabled = ProfilePolicy()
    bounded = _bounded_policy(disabled)
    assert not bounded.evidence_cache_enabled
    assert bounded.evidence_cache_retention_days == 0
    valid = replace(
        disabled,
        collection_enabled=True,
        evidence_cache_enabled=True,
        evidence_cache_retention_days=7,
        evidence_cache_max_bytes=1024,
    )
    assert _bounded_policy(valid) is valid


def test_reduction_grouping_uses_session_roots_and_journal_fallbacks():
    from hashlib import sha256

    from ssgrep.accounting.collect.partitions import _groups

    document = {
        "session_roots": [["claude", "session-1", "root-1"]],
        "executions": [
            {"execution_id": "journal-1", "runtime": None, "session_id": None},
            {"execution_id": "native-1", "runtime": "claude", "session_id": "session-1"},
        ],
        "invocations": [{"runtime": "claude", "session_id": "session-1"}],
        "session_parents": [["claude", "session-1", None]],
        "invocation_sessions": [["invocation-1", "claude", "session-1"]],
    }
    grouped = _groups(document)
    journal_root = "@journal:" + sha256(b"journal-1").hexdigest()[:2]
    assert grouped[journal_root]["executions"] == [document["executions"][0]]
    assert grouped["root-1"]["executions"] == [document["executions"][1]]
    assert grouped["root-1"]["invocations"] == document["invocations"]
    assert grouped["root-1"]["session_parents"] == document["session_parents"]
    assert grouped["root-1"]["invocation_sessions"] == document["invocation_sessions"]


def test_ledger_manifest_decoding_requires_profile_and_expands_with_it() -> None:
    from ssgrep.accounting.ledger.ledger_codec import ledger_state_to_dict

    ledger = VersionedLedger(METHOD)
    manifest = {
        "format": "root-ledger-v1",
        "state": ledger_state_to_dict(ledger.state()),
        "roots": {},
        "base_blob": None,
    }
    with pytest.raises(CollectError) as error:
        collector_state.decode_ledger(manifest)
    assert error.value.condition == "state_corrupt"

    decoded, superseded = collector_state.decode_ledger(manifest, profile=PROFILE)
    assert decoded == ledger.state()
    assert superseded == {}


def test_expire_journal_events_keeps_live_start_and_drops_expired_events() -> None:
    from ssgrep.accounting.collect.collector import expire_journal_events

    live_start = build_event(
        execution_id="live", phase=Phase.STARTED, ts_utc="2026-01-01T12:00:00Z"
    )
    old_complete = build_event(
        execution_id="old", phase=Phase.COMPLETED, ts_utc="2025-12-31T12:00:00Z"
    )
    old_start = build_event(
        execution_id="old-only", phase=Phase.STARTED, ts_utc="2025-12-31T12:00:00Z"
    )
    recent = build_event(
        execution_id="recent", phase=Phase.COMPLETED, ts_utc="2026-01-02T00:00:00Z"
    )

    kept, dropped = expire_journal_events(
        [live_start, old_complete, old_start, recent], "2026-01-01"
    )

    assert [event.execution_id for event in kept] == ["live", "recent"]
    assert kept[0] == live_start
    assert dropped == 2


def test_segment_reader_rejects_damaged_committed_bytes_and_size_mismatch() -> None:
    item = build_evidence()
    (ref,) = store.write_segments(PROFILE, [item])
    path = scene.accounting_dir() / "collector" / store.EVIDENCE_DIR / ref.name
    original = path.read_bytes()
    expected = replace(item, payload={"_evidence_format": "compact-v1"})
    assert list(store.read_segment(PROFILE, ref)) == [expected]

    path.write_bytes(original[:-1])
    with pytest.raises(CollectError) as damaged:
        list(store.read_segment(PROFILE, ref))
    assert damaged.value.condition == "state_corrupt"
    path.write_bytes(original)
    path.chmod(0o600)
    path.write_bytes(original.replace(b"compact-v1", b"broken-v1"))
    with pytest.raises(CollectError) as invalid:
        list(store.read_segment(PROFILE, ref))
    assert invalid.value.condition == "state_corrupt"


def test_reduction_document_expands_root_children_and_rejects_unknown_format() -> None:
    from ssgrep.accounting.collect.partitions import (
        REDUCTION_FORMAT,
        expand_reduction_document,
    )

    child = {
        "executions": [],
        "invocations": [],
        "session_roots": [],
        "session_parents": [],
        "invocation_sessions": [],
        "session_cwd": [],
    }
    filename = store.write_blob(PROFILE, "reduction-root", child)
    manifest = {
        "format": REDUCTION_FORMAT,
        "summary": {"schema": 1},
        "roots": {"root": filename},
    }
    expanded = expand_reduction_document(PROFILE, manifest)
    assert expanded["schema"] == 1
    assert all(expanded[field] == [] for field in child)
    assert expand_reduction_document(PROFILE, {"summary": {}}) == {"summary": {}}
    with pytest.raises(CollectError) as unsupported:
        expand_reduction_document(PROFILE, {"format": "future"})
    assert unsupported.value.condition == "unsupported_schema"


def test_reduction_expansion_rejects_missing_root_blob_instead_of_partial_history() -> None:
    from ssgrep.accounting.collect.partitions import REDUCTION_FORMAT, expand_reduction_document
    from ssgrep.accounting.collect.store import CollectError

    child = {
        "executions": [{"ts_utc": "2026-11-01T00:00:00Z", "execution_id": "e1"}],
        "invocations": [],
        "session_roots": [],
        "session_parents": [],
        "invocation_sessions": [],
        "session_cwd": [],
    }
    filename = store.write_blob(PROFILE, "reduction-root", child)
    manifest = {
        "format": REDUCTION_FORMAT,
        "summary": {"schema": 1},
        "roots": {"root": filename},
    }
    assert expand_reduction_document(PROFILE, manifest)["executions"] == child["executions"]

    with pytest.raises(CollectError) as missing:
        expand_reduction_document(PROFILE, {**manifest, "roots": {"root": "missing.json"}})
    assert missing.value.condition == "state_corrupt"


def test_partition_coverage_tracks_pending_and_unavailable_replay() -> None:
    from ssgrep.accounting.collect.steps import partition_coverage
    from ssgrep.accounting.collect.store import PartitionState

    doc = ReductionDoc(coverage={"complete": True}, warnings=())
    pending = partition_coverage(doc, PartitionState(pending=["session"]))
    assert pending.coverage["complete"] is True
    assert pending.coverage["reduction_pending_nodes"] == 1
    assert "reduction_backlog" in pending.warnings
    unavailable = partition_coverage(doc, PartitionState(unavailable={"root": "source_missing"}))
    assert unavailable.coverage["complete"] is True
    assert "historical_replay_unavailable" in unavailable.warnings


def test_cycle_reports_partition_work_before_claiming_completion(tmp_path: Path) -> None:
    vs.enable(pin=True)
    vs.start_capture()
    vs.pair(vs.projects(tmp_path), 0, scene.at(hour=9))
    scene.collect(scene.at(hour=9))
    scene.append_transcript(
        vs.projects(tmp_path) / "-work" / "cons0.jsonl",
        scene.search_records("cons0", "t0-next", "different query", (10, 0, 0)),
        scene.at(hour=10),
    )
    result = scene.collect(
        scene.at(hour=10),
        CollectOptions(budget=CycleBudget(bytes=1)),
    )
    assert result.state == "partial"
    assert result.report is not None
    assert result.report.stopped_by in {
        "byte_budget",
        "reduction_budget",
        "evidence_index_budget",
        "evidence_maintenance_budget",
    }


def test_public_receipt_cleanup_rejects_invalid_persisted_identity_and_keeps_valid_file() -> None:
    import os

    from ssgrep.accounting.privacy import storage

    receipt_id = "cleanup-valid"
    directory = storage.open_profile_dir(PROFILE, "receipts")
    try:
        storage.atomic_write(directory, receipt_id + ".json", b"{}\n")
    finally:
        os.close(directory)

    assert collector._remaining_public(PROFILE, None, [receipt_id]) == (receipt_id,)
    with pytest.raises(CollectError) as invalid:
        collector._remaining_public(PROFILE, None, ["../outside"])
    assert invalid.value.condition == "state_corrupt"
