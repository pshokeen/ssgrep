"""Independent v2 expectations: literals, not production valuation outputs."""

from dataclasses import replace
from typing import Any

import pytest

from eval.accounting_airline import accounting_oracle as oracle
from eval.accounting_airline.oracle_estimates import bounded_estimate, interval_work_ms
from eval.accounting_airline.oracle_inputs import IndexStats, freeze_lookups, rarity_bands
from tests.accounting.airline.oracle_fixtures import TOKEN, base_world


def stats(n=511, **changes):
    return replace(
        IndexStats(
            generation="g1",
            documents=n,
            df=(("mct", 1),),
            corpus_digest="digest",
            authority_id="lab-project",
            key_id="key-id",
            profile_scope="personal-local",
        ),
        **changes,
    )


@pytest.mark.parametrize(
    "n,bands",
    [
        (30, (False, False, False)),
        (31, (False, False, True)),
        (126, (False, False, True)),
        (127, (False, True, True)),
        (199, (False, True, True)),
        (200, (False, True, True)),
        (511, (True, True, True)),
    ],
)
def test_lexical_superset_feasibility_without_document_floor(n, bands):
    assert rarity_bands(stats(n), "identifier", TOKEN) == bands


@pytest.mark.parametrize(
    "evidence",
    [
        stats(df=()),
        stats(normalization="other"),
        stats(document_unit="conversation"),
        stats(authority_id=""),
        stats(key_id=""),
        stats(coverage="partial"),
        stats(unavailable_reason="generation_changed"),
    ],
)
def test_incompatible_or_absent_stats_cannot_prove_positive_credit(evidence):
    assert rarity_bands(evidence, "identifier", TOKEN) == (False, False, False)


def test_frozen_lookup_keeps_old_generation_new_identifier_uses_new_generation():
    frozen, _, _ = base_world()
    old = freeze_lookups({}, frozen.calls, stats(199))
    new_call = replace(frozen.calls[0], call_id="later", native_call_id="later-native")
    newer = freeze_lookups(old, (*frozen.calls, new_call), stats(511, generation="g2"))
    assert newer[frozen.calls[0].call_id].generation == "g1"
    assert newer["later"].generation == "g2"
    assert old.keys() == {frozen.calls[0].call_id}


def test_production_profile_uses_stats_and_marks_unproved_band_unknown():
    frozen, _, _ = base_world()
    inputs = freeze_lookups({}, frozen.calls, stats(199))
    result = oracle.evaluate(frozen, oracle.production_profile(inputs))
    low, mid, high = (result.estimates[s] for s in oracle.SCENARIOS)
    assert low.credit == 0 and low.net is None and low.known_net == -50
    assert low.bounds["credit_lower"] == 0 and low.bounds["credit_upper"] is None
    assert (mid.credit, high.credit) == (39, 74)
    assert (mid.net, high.net) == (-11, 24)


def test_missing_stats_never_become_known_zero_or_positive_credit():
    frozen, _, _ = base_world()
    result = oracle.evaluate(frozen, oracle.production_profile({}))
    assert all(
        e.credit == 0 and e.net is None and e.known_net == -50 for e in result.estimates.values()
    )


def test_unreferenced_visible_token_does_not_require_frequency_on_a_debit_only_leaf():
    from tests.accounting.airline.oracle_fixtures import consumer_b, frozen, search_call, source_a
    from tests.accounting.airline.test_lab_oracle import savings_doc

    world = frozen([source_a(), consumer_b(action=None)], [search_call()])
    inputs = freeze_lookups({}, world.calls, stats())
    result = oracle.evaluate(world, oracle.production_profile(inputs))
    assert result.verdicts[0].reuse == "none_observed"
    assert all(row.credit == 0 and row.net == -50 for row in result.estimates.values())
    assert oracle.compare_with_snapshot(result, savings_doc(result)) == []


@pytest.mark.parametrize("source_present", [False, True])
def test_missing_frozen_frequency_authority_is_not_excused_by_source_uncertainty(source_present):
    from tests.accounting.airline.oracle_fixtures import consumer_b, frozen, search_call, source_a
    from tests.accounting.airline.test_lab_oracle import savings_doc

    world = frozen([*([source_a()] if source_present else []), consumer_b()], [search_call()])
    result = oracle.evaluate(
        world, oracle.production_profile(freeze_lookups({}, world.calls, stats()))
    )
    assert result.verdicts[0].reuse == "action_reference"
    problems = oracle.compare_with_snapshot(result, savings_doc(result))
    assert any("provenance: missing frozen frequency_generation" in p for p in problems)
    assert any("provenance: missing frozen frequency_corpus_digest" in p for p in problems)
    unavailable = oracle.evaluate(world, oracle.production_profile({}))
    assert all(row.credit == 0 and row.net is None for row in unavailable.estimates.values())
    assert not any(
        "missing frozen frequency_" in p
        for p in oracle.compare_with_snapshot(unavailable, savings_doc(unavailable))
    )


@pytest.mark.parametrize("debit_max,lower", [(None, None), (14, -1)])
def test_independent_bounds_subtract_separate_endpoints(debit_max, lower):
    result = bounded_estimate("central", 15, True, 10, 10, debit_max, 4, ())
    assert result.known_net == 1
    assert result.bounds == {
        "credit_lower": 3,
        "credit_upper": None,
        "debit_lower": 2,
        "debit_upper": None if debit_max is None else 4,
        "net_lower": lower,
        "net_upper": None,
    }
    assert result.net is None


def test_active_proxy_union_and_sum_before_floor_are_not_service_latency():
    assert interval_work_ms({"actor-a": [(0, 900), (500, 1400)], "actor-b": [(0, 900)]}) == 2300
    assert interval_work_ms({"actor-a": [(0, 900000)]}) == 300000


def test_snapshot_comparison_rejects_doctored_bounds_subtotals_and_nulls():
    from tests.accounting.airline.test_lab_oracle import savings_doc

    frozen, _, _ = base_world()
    result = oracle.evaluate(frozen, oracle.arithmetic_profile([("identifier", TOKEN)]))
    doc = savings_doc(result)
    row = doc["data"]["data"]["estimates"]["context_tokens"]["central"]
    row["known_net"] = 999
    row["bounds"]["net_lower"]["value"] = None
    problems = oracle.compare_with_snapshot(result, doc)
    assert any("known_net" in p for p in problems)
    assert any("net_lower" in p for p in problems)


def test_later_identifier_gets_new_generation_without_revaluing_old_lookup():
    from tests.accounting.airline.oracle_fixtures import (
        OTHER,
        command,
        consumer_b,
        frozen,
        search_call,
        search_json,
        source_a,
    )

    source = source_a(witness_tail=OTHER)
    first = consumer_b()
    old_world = frozen([source, first], [search_call()])
    old = freeze_lookups({}, old_world.calls, stats(199))
    second = consumer_b(sid="sc", cid="new", excerpt=OTHER, action=None)
    second.call("new2", command(f"psql -c select '{OTHER}'", 60))
    second.result("new2", "ok".ljust(20, "."))
    newer = frozen(
        [source, first, second],
        [search_call(), search_call(sid="sc", native="new1", stdout=search_json(OTHER, 200))],
    )
    inputs = freeze_lookups(old, newer.calls, stats(511, generation="g2"))
    result = oracle.evaluate(newer, oracle.production_profile(inputs))
    by_task = {v.task: v for v in result.verdicts}
    assert by_task["sb#0"].proven_bands == (False, True, True)
    assert by_task["sc#0"].proven_bands == (True, True, True)
    assert dict(by_task["sb#0"].provenance)["generation"] == "g1"
    assert dict(by_task["sc#0"].provenance)["generation"] == "g2"
    assert result.estimates["conservative"].credit == 35
    assert result.estimates["conservative"].net is None
    assert (
        oracle.evaluate(old_world, oracle.production_profile(old)).estimates["conservative"].credit
        == 0
    )


def test_actual_readonly_metadata_adapter_does_not_rebuild_or_guess_scope(tmp_path):
    import json
    from types import SimpleNamespace

    import pyarrow as pa

    from eval.accounting_airline.oracle_inputs import read_index_inputs

    project = tmp_path / "project"
    project.mkdir()
    key = tmp_path / "accounting/personal-local/fingerprint.key"
    key.parent.mkdir(parents=True)
    key.write_bytes(b"x" * 32)

    class RawProjection:
        def __init__(self, rows):
            self.rows = rows

        def select(self, columns):
            self.columns = columns
            return self

        def to_arrow(self):
            return pa.Table.from_pylist(self.rows).select(self.columns)

    class RawRepository:
        def __init__(self):
            self.metadata = {
                "index_state": "ready",
                "last_index_time": "2026-10-14",
                "lexical_stats": json.dumps({"n_docs": 199, "df": {"mct": 1}, "avgdl": 1}),
            }
            self.chunks = [{"chunk_id": str(i), "cwd": str(project)} for i in range(199)]

        def exists(self):
            return True

        def get_meta(self, name):
            return self.metadata.get(name)

        def table(self, name):
            return SimpleNamespace(
                version=1,
                take_offsets=lambda offsets: RawProjection(
                    [{**self.chunks[i], "_rowoffset": i} for i in reversed(offsets)]
                ),
            )

        def count(self, name):
            return len(self.chunks)

    repo = RawRepository()
    good = read_index_inputs(tmp_path, project, repository=repo)
    assert not good.unavailable_reason and good.documents == 199
    assert rarity_bands(good, "identifier", TOKEN) == (False, True, True)
    raw = repo.metadata["lexical_stats"]
    repo.metadata["lexical_stats"] = json.dumps(json.loads(raw), indent=2)
    reformatted = read_index_inputs(tmp_path, project, repository=repo)
    assert reformatted.df == good.df and reformatted.documents == good.documents
    assert reformatted.corpus_digest != good.corpus_digest
    repo.metadata["lexical_stats"] = raw
    repo.chunks[0], repo.chunks[1] = repo.chunks[1], repo.chunks[0]
    reordered = read_index_inputs(tmp_path, project, repository=repo)
    assert reordered.corpus_digest != good.corpus_digest
    repo.chunks[0], repo.chunks[1] = repo.chunks[1], repo.chunks[0]
    repo.chunks[0]["cwd"] = str(tmp_path / "outside")
    assert (
        "scope_unapproved"
        in read_index_inputs(tmp_path, project, repository=repo).unavailable_reason
    )
    repo.chunks[0]["cwd"] = str(project)
    repo.metadata["lexical_stats"] = '{"n_docs":199,"df":{"mct":0},"avgdl":1}'
    assert (
        "stats_invalid" in read_index_inputs(tmp_path, project, repository=repo).unavailable_reason
    )
    assert key.read_bytes() == b"x" * 32


def test_raw_service_latency_is_unique_observed_bounded_active_proxy():
    from datetime import UTC, datetime

    from eval.accounting_airline.oracle_scan import scan_frozen
    from eval.accounting_airline.oracle_time import reconcile_timing

    frozen, _, _ = base_world()
    call = replace(frozen.calls[0], wall_ns=1000000000000, duration_ns=500000000000)
    scan = scan_frozen(replace(frozen, calls=(call,)))
    stamp = datetime.fromtimestamp(1400, UTC).isoformat()
    event = {
        "execution_id": "e1",
        "phase": "completed",
        "operation": "search",
        "ts_utc": stamp,
        "mono_ns": 400000000000,
    }
    assert reconcile_timing(scan, (event, event)) == (400000000000, 300000000000, ())
    wrong = {**event, "mono_ns": 700000000000}
    assert reconcile_timing(scan, (wrong,))[2] == ("no_unique_observed_journal_timing",)
    assert "conflicting_journal_timing" in reconcile_timing(scan, (event, wrong))[2]
    assert reconcile_timing(scan, ())[2] == ("no_unique_observed_journal_timing",)


@pytest.mark.parametrize(
    "unit,method", [("context_tokens", "chars_div4_v1"), ("work_seconds", "capped_gap_v1")]
)
def test_receipt_negative_subtotals_bounds_and_coverage_are_not_hidden(unit, method):
    import copy
    from types import SimpleNamespace

    from eval.accounting_airline.oracle_receipts import compare_receipt

    scale = 1 if unit == "context_tokens" else 250000000
    expected = bounded_estimate(
        "central",
        15 * scale,
        True,
        10 * scale,
        10 * scale,
        None,
        4 * scale,
        ("credit:missing", "debit:missing"),
    )

    def q(value):
        return {"value": value, "unit": unit, "reason": "unbounded" if value is None else None}

    row = {
        "scenario": "central",
        "credit": q(3),
        "debit": q(2),
        "net": q(None),
        "known_net": 1,
        "complete": False,
        "credit_missing": [["c", "missing"]],
        "debit_missing": [["d", "missing"]],
        "bounds": {
            "assumptions": list(expected.bound_assumptions),
            **{k: q(v) for k, v in expected.bounds.items()},
        },
    }
    result = SimpleNamespace(estimates={"central": expected}, time_estimates={"central": expected})
    doc: dict[str, Any] = {
        "unit": f"{unit}:{method}",
        "scenarios": [row],
        "raw_unit": "characters" if unit == "context_tokens" else "nanoseconds",
        "credit_raw": {"central": 15 * scale},
        "debit_raw": 10 * scale,
        "debit_raw_min": 10 * scale,
        "debit_raw_max": None,
    }
    assert compare_receipt(result, doc) == []
    bad = copy.deepcopy(doc)
    bad["scenarios"][0]["bounds"]["net_lower"]["value"] = 1
    bad["scenarios"][0]["net"]["value"] = 1
    assert any("unbounded debit" in p for p in compare_receipt(result, bad))
    assert any("incomplete evidence" in p for p in compare_receipt(result, bad))
    bad = copy.deepcopy(doc)
    bad["scenarios"][0]["credit"]["value"] = 4
    assert any("independently observed" in p for p in compare_receipt(result, bad))


def test_source_time_excludes_terminal_task_and_lookup_boundary_gaps():
    from types import SimpleNamespace

    from eval.accounting_airline.observer import NativeRecord
    from eval.accounting_airline.oracle_estimates import source_work

    records = tuple(
        NativeRecord(i, kind, f"r{i}", "text", ts_ms=stamp)
        for i, (kind, stamp) in enumerate(
            (
                ("tool_call", 1000),
                ("tool_result", 1900),
                ("tool_call", 2800),
                ("tool_result", 3700),
                ("user_turn", 900000),
            )
        )
    )
    exposure = SimpleNamespace(task="consumer", session="b", result=0, index=0, is_lookup=True)
    scan = SimpleNamespace(
        records={"a": records, "b": (NativeRecord(0, "tool_result", "b0", ts_ms=1000000),)},
        invocations=[exposure],
    )
    assert source_work(scan, {"consumer": ["a:r0", "a:r1"]}) == (1800, False)
    assert source_work(scan, {"consumer": ["a:r0", "a:r1"]})[0] // 1000 == 1
    assert source_work(scan, {"consumer": ["a:r3", "a:r4"]}) == (0, True)
    lookup = SimpleNamespace(task="source", session="a", result=3, index=2, is_lookup=True)
    scan.invocations.append(lookup)
    assert source_work(scan, {"consumer": ["a:r1"]}) == (0, True)


def test_snapshot_rejects_incomparable_units_missing_coverage_and_active_proxy():
    import copy

    from tests.accounting.airline.test_lab_oracle import savings_doc

    frozen, _, _ = base_world()
    result = oracle.evaluate(frozen, oracle.arithmetic_profile([("identifier", TOKEN)]))
    doc = savings_doc(result)
    row = doc["data"]["data"]["estimates"]["context_tokens"]["central"]
    row["unit"]["method"] = "provider_tokens"
    row.pop("credit_missing")
    problems = oracle.compare_with_snapshot(result, doc)
    assert any("comparison method" in p for p in problems)
    assert any("credit_missing" in p for p in problems)
    doc = savings_doc(result)
    result = replace(result, timing={**result.timing, "compare_snapshot": True})
    work = copy.deepcopy(doc["data"]["data"]["estimates"]["context_tokens"])
    for scenario, row in work.items():
        expected = result.time_estimates[scenario]
        row.update(
            known_net=expected.known_net,
            complete=expected.complete,
            credit_missing=[m for m in expected.missing if m.startswith("credit:")],
            debit_missing=[m for m in expected.missing if m.startswith("debit:")],
            unit={"unit": "work_seconds", "method": "capped_gap_v1"},
        )
        for field in ("credit", "debit", "net"):
            row[field].update(value=getattr(expected, field), unit="work_seconds", reason="unknown")
        row["bounds"] = {
            name: {"value": value, "unit": "work_seconds", "reason": "unknown"}
            for name, value in expected.bounds.items()
        }
        row["bounds"]["assumptions"] = list(expected.bound_assumptions)
    doc["data"]["data"]["estimates"]["work_seconds"] = work
    assert any("active proxy" in p for p in oracle.compare_with_snapshot(result, doc))


def test_independent_candidate_vocabulary_keeps_domain_signals_not_refs_or_personal_paths():
    from eval.accounting_airline.oracle_inputs import rarity_upper_df
    from eval.accounting_airline.oracle_rules import candidate_tokens

    found = set(
        candidate_tokens(
            "MCT-DEN-045 policy_key --rare-mode ./src/policy_rules.py ERR_POLICY_TIMEOUT "
            "/home/alice/private.txt 2026-10-14 --json omp:source:ep:1"
        )
    )
    assert found == {
        ("identifier", TOKEN),
        ("identifier", "policy_key"),
        ("flag", "--rare-mode"),
        ("path", "src/policy_rules.py"),
        ("error_code", "ERR_POLICY_TIMEOUT"),
    }
    evidence = stats(199, df=(("src", 199), ("policy_rules", 1)))
    assert rarity_upper_df(evidence, "path", "src/policy_rules.py") == 1
    assert rarity_bands(evidence, "path", "src/policy_rules.py") == (False, True, True)
    assert rarity_upper_df(evidence, "path", "missing/absent.txt") is None
    assert rarity_bands(
        replace(evidence, mapping="legacy_exact_df"), "path", "src/policy_rules.py"
    ) == (False, False, False)


@pytest.mark.parametrize(
    "command,expected",
    [
        ("ssgrep search rules", True),
        ("cd /lab && ssgrep show omp:a:ep:1", True),
        ("echo 'ssgrep search rules'", False),
        ("echo ssgrep search rules", False),
    ],
)
def test_native_lookup_boundaries_are_not_narrative_mentions(command, expected):
    from eval.accounting_airline.observer import NativeRecord
    from eval.accounting_airline.oracle_estimates import lookup_boundary

    assert lookup_boundary(NativeRecord(0, "tool_call", "r0", command)) is expected


@pytest.mark.parametrize(
    "argv,expected",
    [
        (("ssgrep", "search", "--json", "rules", "--where=x = 1"), ("rules", "x = 1")),
        (("ssgrep", "search", "--limit", "3", "rules", "--json"), ("rules", None)),
        (("ssgrep", "show", "--json", "omp:a:ep:1"), ("omp:a:ep:1", None)),
        (("ssgrep", "search", "--unknown-option", "rules"), None),
    ],
)
def test_timing_join_fingerprints_parse_actual_argv_not_option_order(argv, expected):
    from eval.accounting_airline.oracle_time import lookup_arguments

    assert lookup_arguments(argv) == expected


def with_native_origins(world):
    sessions = {}
    for key, (actor, task, native) in world.sessions.items():
        records = tuple(
            replace(record, origin_id=f"entry-{record.record_id}") for record in native.records
        )
        sessions[key] = (actor, task, replace(native, records=records))
    return replace(world, sessions=sessions)


def test_identical_native_copy_deduplicates_but_changed_same_origin_output_is_unavailable():
    from tests.accounting.airline.oracle_fixtures import OTHER

    world = with_native_origins(base_world()[0])
    call = replace(world.calls[0], wall_ns=1000000000000, duration_ns=2000000000)
    world = replace(world, calls=(call,))
    profile = oracle.arithmetic_profile(
        [("identifier", TOKEN)],
        [
            {
                "execution_id": "one-real-execution",
                "phase": "completed",
                "operation": "search",
                "ts_utc": "1970-01-01T00:16:41Z",
                "mono_ns": 1200000000,
            }
        ],
    )
    baseline = oracle.evaluate(world, profile)
    actor, task, native = world.sessions["sb"]
    copied = replace(native, path="/lab/copied-sb.jsonl")
    same = replace(world, sessions={**world.sessions, "sb@copy": (actor, task, copied)})
    identical = oracle.evaluate(same, profile)
    assert identical.estimates == baseline.estimates
    assert identical.tasks == baseline.tasks and identical.counts["executions"] == 1
    assert identical.counts["native_conflicts"] == 0
    changed = replace(
        copied,
        records=tuple(
            replace(record, text=record.text.replace(TOKEN, OTHER))
            if record.kind == "tool_result" and record.call_id == "cs1"
            else record
            for record in copied.records
        ),
    )
    corrupt = replace(world, sessions={**world.sessions, "sb@copy": (actor, task, changed)})
    after = oracle.evaluate(corrupt, profile)
    assert after.counts["executions"] == 1 and after.counts["native_conflicts"] == 1
    assert after.verdicts == ()
    for estimate in after.estimates.values():
        assert estimate.debit == 0 and estimate.net is None
        assert estimate.bounds["debit_upper"] is None and estimate.bounds["net_lower"] is None
        assert "debit:conflicting_native_output" in estimate.missing
    assert after.timing["known_service_ns"] == 1200000000
    assert all(row.debit == 1 and row.net == -1 for row in after.time_estimates.values())
    reversed_inputs = replace(corrupt, sessions=dict(reversed(tuple(corrupt.sessions.items()))))
    assert oracle.evaluate(reversed_inputs, profile).estimates == after.estimates


def test_conflicted_source_does_not_choose_first_or_smallest_digest_for_positive_credit():
    world = with_native_origins(base_world()[0])
    actor, task, native = world.sessions["sa"]
    changed = replace(
        native,
        path="/lab/source-variant.jsonl",
        records=tuple(
            replace(record, text=record.text.replace(TOKEN, "MCT-XXX-999"))
            if record.kind == "tool_result" and TOKEN in record.text
            else record
            for record in native.records
        ),
    )
    world = replace(world, sessions={**world.sessions, "sa@variant": (actor, task, changed)})
    result = oracle.evaluate(world, oracle.arithmetic_profile([("identifier", TOKEN)]))
    assert result.counts["native_conflicts"] == 1
    assert result.verdicts[0].reason == "source_evidence_conflict"
    assert all(
        row.credit == 0 and row.net is None and row.bounds["credit_upper"] is None
        for row in result.estimates.values()
    )


def test_native_message_origin_is_distinct_from_reused_tool_call_identifier(tmp_path):
    import json

    from eval.accounting_airline.observer import _parse_native_bytes
    from eval.accounting_airline.oracle_scan import scan_frozen

    rows = [{"type": "session", "id": "same-session", "cwd": "/lab"}]
    for index in range(2):
        rows.extend(
            [
                {
                    "type": "message",
                    "id": f"native-call-{index}",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "toolCall",
                                "id": "reused-id",
                                "name": "bash",
                                "arguments": {"command": "echo ok"},
                            }
                        ],
                    },
                },
                {
                    "type": "message",
                    "id": f"native-result-{index}",
                    "message": {
                        "role": "toolResult",
                        "toolCallId": "reused-id",
                        "isError": False,
                        "content": [{"type": "text", "text": f"answer-{index}"}],
                    },
                },
            ]
        )
    prefix = b"".join(json.dumps(row).encode() + b"\n" for row in rows)
    tail = b'{"type":"message","id":"pending","message":{"role":"toolResult"'
    native = _parse_native_bytes(tmp_path / "native.jsonl", prefix + tail)
    assert native.complete_bytes == len(prefix) and native.partial_tail_bytes == len(tail)
    calls = [record for record in native.records if record.kind == "tool_call"]
    assert calls[0].call_id == calls[1].call_id == "reused-id"
    assert calls[0].origin_id != calls[1].origin_id
    world = replace(
        base_world()[0], calls=(), sessions={"same-session": ("solver-b", "task", native)}
    )
    scan = scan_frozen(world)
    assert len(scan.records["same-session"]) == 4
    assert scan.counts["native_conflicts"] == 0


def test_two_unique_lookups_keep_269_known_chars_proxy_after_conflicting_1364_byte_show():
    from tests.accounting.airline.oracle_fixtures import (
        OTHER,
        REF_A1,
        SHOW_CMD,
        TOKEN,
        command,
        consumer_b,
        frozen,
        lookup,
        observed,
        search_call,
        search_json,
        show_json,
        source_a,
    )

    source, consumer = (
        source_a(),
        consumer_b(visible=search_json("no reusable facts", 1077), action=None),
    )
    shown = show_json(TOKEN, 1364)
    lookup(consumer, "show-call", shown, cmd=SHOW_CMD)
    consumer.call("action", command(f"psql -c select '{TOKEN}'", 60))
    consumer.result("action", "ok".ljust(20, "."))
    calls = [
        search_call(stdout=search_json("no reusable facts", 1077)),
        observed(
            "observed-show",
            "solver-b",
            ("ssgrep", "show", REF_A1),
            native="show-call",
            stdout=shown,
        ),
    ]
    calls = [
        replace(call, wall_ns=(1000 + i * 3) * 1000000000, duration_ns=2000000000)
        for i, call in enumerate(calls)
    ]
    world = with_native_origins(frozen([source, consumer], calls))
    events = [
        {
            "execution_id": f"exec-{i}",
            "phase": "completed",
            "operation": operation,
            "ts_utc": stamp,
            "mono_ns": 1200000000,
        }
        for i, operation, stamp in (
            (0, "search", "1970-01-01T00:16:41Z"),
            (1, "show", "1970-01-01T00:16:44Z"),
        )
    ]
    profile = oracle.arithmetic_profile([("identifier", TOKEN)], events)
    assert oracle.evaluate(world, profile).estimates["central"].debit == 610
    actor, task, native = world.sessions["sb"]
    identical = replace(world, sessions={**world.sessions, "sb@copy": (actor, task, native)})
    assert oracle.evaluate(identical, profile).estimates["central"].debit == 610
    variant = replace(
        native,
        records=tuple(
            replace(record, text=record.text.replace(TOKEN, OTHER))
            if record.kind == "tool_result" and record.call_id == "show-call"
            else record
            for record in native.records
        ),
    )
    conflict = replace(
        identical, sessions={**identical.sessions, "sb@variant": (actor, task, variant)}
    )
    result = oracle.evaluate(conflict, profile)
    assert result.counts["executions"] == 2 and result.counts["native_conflicts"] == 1
    assert all(
        row.debit == 269
        and row.net is None
        and row.bounds["net_lower"] is None
        and row.bounds["debit_upper"] is None
        for row in result.estimates.values()
    )
    assert result.timing["known_service_ns"] == 2400000000
    assert all(row.debit == 2 and row.net == -2 for row in result.time_estimates.values())


def test_conflicted_action_result_cannot_prove_success_or_choose_a_contradiction_variant():
    world = with_native_origins(base_world()[0])
    actor, task, native = world.sessions["sb"]
    changed = replace(
        native,
        records=tuple(
            replace(record, text="Traceback: changed output", is_error=True)
            if record.kind == "tool_result" and record.call_id == "cs2"
            else record
            for record in native.records
        ),
    )
    world = replace(world, sessions={**world.sessions, "sb@action-variant": (actor, task, changed)})
    result = oracle.evaluate(world, oracle.arithmetic_profile([("identifier", TOKEN)]))
    assert result.verdicts[0].reuse == "action_reference"
    assert result.verdicts[0].action_outcome_proven is False
    assert tuple(result.estimates[name].credit for name in oracle.SCENARIOS) == (0, 39, 74)
    assert result.estimates["conservative"].debit == 50
