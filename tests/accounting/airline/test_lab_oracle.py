"""The lab accounting oracle, checked against numbers derived BY HAND from literal evidence.

Reference world (``oracle_fixtures.base_world``), all lengths are characters:
  source ``sa`` (solver-a): L1 = r0 20, r1 24, r2 30; L2 = r3 16, r4 call 40, r5 witness 100
  (carries MCT-DEN-045), r6 60, r7 80.   consumer ``sb`` (solver-b): brief 30, search call 50,
  result 200 (one card, ref omp:sa:ep:1, carries the token), action call 60, action result 20.
  immediate = r4+r5 = 140 -> 35; discovery = r3+r4+r5 = 156 -> 39; episode = r3..r7 = 296 -> 74.
  debit = 200 // 4 = 50.  Nets: 35-50=-15, 39-50=-11, 74-50=24.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from typing import Any

import pytest

from eval.accounting_airline import accounting_oracle as oracle
from eval.accounting_airline.accounting_oracle import (
    ARITHMETIC,
    PRODUCTION,
    AggregationRefused,
    arithmetic_profile,
    compare_with_snapshot,
    evaluate,
    production_profile,
    refuse_aggregate,
    review_candidates,
)
from tests.accounting.airline.oracle_fixtures import (
    OTHER,
    REF_A1,
    SHOW_CMD,
    TOKEN,
    SessionBuilder,
    base_world,
    command,
    consumer_b,
    frozen,
    lookup,
    observed,
    search_call,
    search_json,
    show_json,
    source_a,
    text,
)

PROFILE = arithmetic_profile([("identifier", TOKEN)])


def run(world, profile=PROFILE):
    return evaluate(world, profile)


def nets(result) -> tuple[Any, ...]:
    return tuple(result.estimates[s].net for s in oracle.SCENARIOS)


def credits(result) -> tuple[Any, ...]:
    return tuple(result.estimates[s].credit for s in oracle.SCENARIOS)


def show_call(native: str, actor: str = "solver-b", *, stdout: str = "") -> Any:
    return observed(f"{actor}-{native}", actor, ("ssgrep", "show", REF_A1, "--json"),
                    native=native, stdout=stdout)  # fmt: skip


# ---- 1. nesting and exact integers --------------------------------------------------------


def test_arithmetic_profile_nested_scenarios_exact_integers() -> None:
    world, _sa, _sb = base_world()
    result = run(world)
    assert result.profile == ARITHMETIC and result.methodology_label.endswith("chars_div4_v1")
    assert credits(result) == (35, 39, 74)
    assert [result.estimates[s].debit for s in oracle.SCENARIOS] == [50, 50, 50]
    assert nets(result) == (-15, -11, 24)
    assert result.tasks == ("sa#0", "sa#1", "sb#0")
    (verdict,) = result.verdicts
    assert verdict.eligible and verdict.reuse == "action_reference"
    assert verdict.source_session == "sa" and verdict.witness_record == "sa-r5"
    cons, cent, opt = (set(verdict.atoms[s]) for s in oracle.SCENARIOS)
    assert cons == {"sa:sa-r4", "sa:sa-r5"}
    assert cons < cent < opt and "sa:sa-r3" in cent and "sa:sa-r7" in opt
    assert result.counts["eligible_candidates"] == 1 and result.counts["exposures"] == 1


# ---- 2. union once ------------------------------------------------------------------------


def _two_tokens_one_witness():
    sa = source_a(witness_tail=OTHER)  # witness result: TOKEN + OTHER in one 100-char record
    sb = consumer_b(excerpt=f"{TOKEN} {OTHER}", action=None)
    sb.call("cs2", command(f"psql -c select '{TOKEN}' '{OTHER}'", 60))
    sb.result("cs2", text(20, "ok"))
    stdout = search_json(f"{TOKEN} {OTHER}", 200)
    return frozen([sa, sb], [search_call(stdout=stdout)])


def union_body() -> None:
    both = arithmetic_profile([("identifier", TOKEN), ("identifier", OTHER)])
    result = run(_two_tokens_one_witness(), both)
    assert result.counts["eligible_candidates"] == 2
    # one witness step exposed by two facts counts ONCE: 140 // 4 = 35, never 280 // 4 = 70
    assert credits(result) == (35, 39, 74)
    assert nets(result) == (-15, -11, 24)


def test_two_tokens_sharing_one_witness_count_the_source_step_once() -> None:
    union_body()


def test_mutation_non_unioning_sum_makes_the_union_test_fail(monkeypatch) -> None:
    def no_union(per_task, chars):  # the broken variant: sums every atom of every token
        return sum(sum(chars[a] for a in atoms) for atoms in per_task)

    union_body()  # genuine implementation passes
    monkeypatch.setattr(oracle, "union_chars", no_union)
    with pytest.raises(AssertionError):
        union_body()
    monkeypatch.undo()
    union_body()  # restored: green again


# ---- 3. same atoms, two tasks -------------------------------------------------------------


def test_same_source_atoms_are_credited_per_task_and_summed_before_dividing() -> None:
    sa = source_a(call_len=42)  # immediate = 42 + 100 = 142 chars per task
    sb, sc = consumer_b(), consumer_b(sid="sc", cid="cx")
    calls = [search_call(), search_call("sc", native="cx1", call_id="sc-o1")]
    result = run(frozen([sa, sb, sc], calls))
    verdicts = {v.task: v for v in result.verdicts}
    assert set(verdicts) == {"sb#0", "sc#0"} and all(v.eligible for v in verdicts.values())
    # credited to EACH task: (142 + 142) // 4 = 71, not 35 + 35 = 70; debit (200+200)//4 = 100
    assert result.estimates["conservative"].credit == 71
    assert result.estimates["conservative"].debit == 100
    assert nets(result)[0] == -29
    assert [d.task for d in result.debits] == ["sb#0", "sc#0"]


# ---- 4. repeats debit again, credit once --------------------------------------------------


def test_repeated_search_and_show_debit_every_time_credit_once() -> None:
    sa, sb = source_a(), consumer_b(action=None)
    lookup(sb, "cs3", search_json(f"{TOKEN} min 45", 200))
    lookup(sb, "cs4", show_json(f"{TOKEN} rule", 120), cmd=SHOW_CMD)
    lookup(sb, "cs5", show_json(f"{TOKEN} rule", 120), cmd=SHOW_CMD)
    sb.call("cs6", command(f"psql -c select '{TOKEN}'", 60))
    sb.result("cs6", text(20, "ok"))
    search_again = search_call(call_id="o2", native="cs3")
    calls = [
        search_call(), search_again,
        show_call("cs4", stdout=show_json(f"{TOKEN} rule", 120)),
        show_call("cs5", stdout=show_json(f"{TOKEN} rule", 120)),
    ]  # fmt: skip
    result = run(frozen([sa, sb], calls))
    # 200 + 200 + 120 + 120 = 640 // 4 = 160 debit; credit only the first exposure's 140 // 4
    assert result.estimates["conservative"].debit == 160
    assert credits(result)[0] == 35 and nets(result)[0] == -125
    assert result.counts["exposures"] == 4 and result.counts["eligible_candidates"] == 1
    assert [v.reason for v in result.verdicts] == [
        None,
        "prior_known",
        "prior_known",
        "prior_known",
    ]
    assert result.counts["solver-b.search"] == 2 and result.counts["solver-b.show"] == 2


# ---- 5. verifier lookups ------------------------------------------------------------------


def test_verifier_lookups_add_counts_but_never_exposure_debit_or_credit() -> None:
    world, sa, sb = base_world()
    baseline = run(world)
    spy = observed("v1", "verifier", ("ssgrep", "search", "connection rules", "--json"),
                   native="cs1", stdout=search_json(f"{TOKEN} min 45", 200))  # fmt: skip
    shown = observed("v2", "verifier", ("ssgrep", "show", REF_A1), stdout=text(500))
    world2 = frozen(
        [sa, sb], [search_call(), spy, shown]
    )  # even a (mis)linked native id cannot make it solver evidence
    result = run(world2)
    assert credits(result) == credits(baseline) and nets(result) == nets(baseline)
    assert result.debits == baseline.debits and result.verdicts == baseline.verdicts
    assert result.counts["verifier_search"] == 1 and result.counts["verifier_show"] == 1
    assert result.counts["verifier.search"] == 1
    assert result.counts["executions"] == 1 and result.counts["journaled_search"] == 2
    assert result.counts["journaled_show"] == 1 and result.counts["invocations"] == 1


def test_same_lookup_by_solver_b_does_debit_positive_twin() -> None:
    sa, sb = source_a(), consumer_b()
    lookup(sb, "cs3", search_json(f"{TOKEN} min 45", 200))
    again = search_call(call_id="o2", native="cs3")
    result = run(frozen([sa, sb], [search_call(), again]))
    assert result.estimates["conservative"].debit == 100  # 400 // 4, not 50
    assert result.counts["executions"] == 2 and result.counts["solver-b.search"] == 2


# ---- 6. quoted text is not an execution ---------------------------------------------------


def test_quoted_ssgrep_text_without_a_dispatch_is_not_an_execution() -> None:
    sa = source_a()
    sb = SessionBuilder("sb", "solver-b", "L3", 5000)
    sb.user(text(30))
    sb.call("q1", 'echo "ssgrep search connection rules"')
    sb.result("q1", text(77, "ssgrep search connection rules"))
    sb.call("q2", "read xd://mcp__ssgrep_search_sessions")
    sb.result("q2", text(400, "search_sessions: ssgrep search"))
    sb.call("q3", command("ssgrep search x --json", 30))
    sb.result("q3", text(77, "plain output"))
    result = run(frozen([sa, sb], []))
    assert result.counts["invocations"] == 0 and result.counts["executions"] == 0
    assert result.estimates["conservative"].debit == 0 and result.debits == ()
    assert nets(result) == (0, 0, 0)
    # positive twin: one observed dispatch linked to q3 makes it one execution that debits
    twin = run(
        frozen([sa, sb], [observed("o1", "solver-b", ("ssgrep", "search", "x"), native="q3")])
    )
    assert twin.counts["invocations"] == 1 and twin.counts["executions"] == 1
    assert twin.estimates["conservative"].debit == 77 // 4  # 19


# ---- 7. shell loop and mixed shell --------------------------------------------------------


def test_shell_loop_is_one_invocation_two_executions_with_known_aggregate_debit() -> None:
    sa, sb = source_a(), consumer_b()
    loop = 'for q in a b; do ssgrep search "$q" --json; done'
    sb.call("cs3", loop)
    visible = text(300, "first-document second-document")
    sb.result("cs3", visible)
    two = [
        observed("o2", "solver-b", ("ssgrep", "search", "a", "--json"), native="cs3"),
        observed("o3", "solver-b", ("ssgrep", "search", "b", "--json"), native="cs3"),
    ]
    result = run(frozen([sa, sb], two))
    assert result.counts["invocations"] == 1 and result.counts["executions"] == 2
    (loop_debit,) = result.debits
    assert loop_debit.chars == len(visible) and not loop_debit.partial
    assert loop_debit.reason is None
    est = result.estimates["conservative"]
    assert (est.credit, est.debit, est.net, est.known_net) == (0, len(visible) // 4, -75, -75)
    assert est.missing == ()


def test_single_lookup_in_mixed_shell_with_parsable_cards_is_attributed() -> None:
    # positive twin of the mixed-shell denial: output that IS one whole parsable document
    sa, sb = source_a(), consumer_b(cmd="cd /lab && ssgrep search q --json")
    result = run(frozen([sa, sb], [search_call()]))
    assert nets(result) == (-15, -11, 24) and not [d for d in result.debits if d.chars is None]


def test_unlinked_dispatch_has_unavailable_exposure_never_guessed() -> None:
    sa, sb = source_a(), consumer_b()
    ghost = observed("g1", "solver-b", ("ssgrep", "search", "x"), stdout=text(900))  # no native id
    result = run(frozen([sa, sb], [search_call(), ghost]))
    missing = next(d for d in result.debits if d.call == "g1")
    assert missing.chars is None and missing.reason == "downstream_exposure_unobserved"
    est = result.estimates["central"]
    assert est.net is None and est.known_net == 39 - 50 and est.debit == 50
    assert result.counts["unlinked_executions"] == 1


# ---- 8. hidden content --------------------------------------------------------------------


def _show_world(cut: int):
    full = show_json(text(100, "lead") + " " + TOKEN, 300)
    assert full.index(TOKEN) > 150
    sa = source_a()
    sb = consumer_b(visible=full[:cut], cmd=SHOW_CMD)
    return frozen([sa, sb], [show_call("cs1", stdout=full)]), full


def test_token_hidden_by_truncation_earns_nothing_but_visible_chars_are_debited() -> None:
    world, full = _show_world(80)
    assert TOKEN in full  # present in the raw stdout ...
    result = run(world)
    assert result.counts["eligible_candidates"] == 0 and result.verdicts == ()  # ... not visible
    assert credits(result) == (0, 0, 0)
    assert result.estimates["conservative"].debit == 80 // 4  # 20
    assert nets(result) == (-20, -20, -20) and result.counts["truncated_exposures"] == 1


def test_token_visible_inside_truncated_unparsable_json_earns_credit_positive_twin() -> None:
    world, _full = _show_world(200)  # still cut mid-JSON, but the token is now visible
    result = run(world)
    assert result.counts["eligible_candidates"] == 1  # ref found by regex in the visible text
    assert result.estimates["conservative"].debit == 50 and credits(result)[0] == 35
    assert result.counts["truncated_exposures"] == 1


# ---- 9. novelty ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("why", "build"),
    [
        ("query_mention", lambda: (consumer_b(), {"argv": ("ssgrep", "search", TOKEN, "--json")})),
        ("prior_known", lambda: (consumer_b(brief=text(30, f"brief {TOKEN}")), {})),
    ],
)
def test_novelty_exclusions_survive_a_later_matching_action(why, build) -> None:
    sb, kw = build()
    result = run(frozen([source_a(), sb], [search_call(**kw)]))
    (verdict,) = result.verdicts
    assert verdict.reason == why and not verdict.eligible
    assert verdict.reuse == "action_reference"  # the later action exists ...
    assert credits(result) == (0, 0, 0) and nets(result) == (-50, -50, -50)  # ... earns nothing


def test_prior_known_from_an_earlier_task_of_the_same_session() -> None:
    sb = SessionBuilder("sb", "solver-b", "L3", 5000)
    sb.user(text(30))
    sb.call("e1", "psql -c 'select 1'")
    sb.result("e1", text(40, TOKEN))  # task 0 already saw it
    sb.user(text(30))
    lookup(sb, "cs1", search_json(f"{TOKEN} min 45", 200))
    sb.call("cs2", command(f"psql -c select '{TOKEN}'", 60))
    sb.result("cs2", text(20, "ok"))
    result = run(frozen([source_a(), sb], [search_call()]))
    assert result.verdicts[0].reason == "prior_known" and result.verdicts[0].task == "sb#1"
    assert credits(result) == (0, 0, 0)
    # twin: the same lookup in a FRESH session earns the credit
    assert credits(run(base_world()[0]))[0] == 35


def test_review_candidates_matches_the_exposed_novel_acted_set() -> None:
    world, _sa, _sb = base_world()
    (found,) = review_candidates(world, prefix=r"MCT-[A-Z]{3}-\d{3}")
    assert (found.token, found.task, found.kind) == (TOKEN, "sb#0", "identifier")
    assert review_candidates(world, prefix=r"XYZ-\d+") == ()  # prefix filters
    assert [r.token for r in review_candidates(world)] == [TOKEN]
    excluded = frozen([source_a(), consumer_b()], [search_call(argv=("ssgrep", "search", TOKEN))])
    assert review_candidates(excluded) == ()  # query mention
    unused = frozen([source_a(), consumer_b(action=None)], [search_call()])
    assert review_candidates(unused) == ()  # no later action
    bad = frozen([source_a(), consumer_b(action="error_text")], [search_call()])
    assert review_candidates(bad) == ()  # contradicted


# ---- 10. source resolution ----------------------------------------------------------------


def _verdict(sa, sb=None, **kw):
    sb = sb or consumer_b()
    return run(frozen([sa, sb], [search_call(**kw)])).verdicts[0], sa


def test_source_user_supplied_earns_no_credit() -> None:
    verdict, _ = _verdict(source_a(l2_user=text(16, f"use {TOKEN}")))
    assert verdict.reason == "source_user_supplied" and not verdict.eligible
    result = run(
        frozen([source_a(l2_user=text(16, f"use {TOKEN}")), consumer_b()], [search_call()])
    )
    assert credits(result) == (0, 0, 0)


def test_source_known_before_witness() -> None:
    verdict, _ = _verdict(source_a(call_tag=f"{TOKEN} "))  # the producing call text names it first
    assert verdict.reason == "source_known_before_witness"


def test_source_self_or_later_never_earns_credit() -> None:
    own = consumer_b(excerpt=f"{TOKEN} min 45")
    stdout = search_json(f"{TOKEN} min 45", 200, ref="omp:sb:ep:1")
    own.records[2] = replace(own.records[2], text=stdout)
    result = run(frozen([source_a(), own], [search_call(stdout=stdout)]))
    assert result.verdicts[0].reason == "source_self_or_later"
    later = run(
        frozen([source_a(t0=9000), consumer_b()], [search_call()])
    )  # witness after exposure
    assert later.verdicts[0].reason == "source_self_or_later" and credits(later) == (0, 0, 0)


def test_source_unresolved_and_not_observed() -> None:
    sb = consumer_b(visible=search_json(f"{TOKEN} min 45", 200, ref="omp:ghost:ep:9"))
    result = run(frozen([source_a(), sb], [search_call()]))
    assert result.verdicts[0].reason == "source_unresolved"
    blank = source_a()
    blank.records[5] = replace(blank.records[5], text=text(100))
    result = run(frozen([blank, consumer_b()], [search_call()]))
    assert result.verdicts[0].reason == "source_not_observed" and credits(result) == (0, 0, 0)


# ---- 11. reuse states ---------------------------------------------------------------------


@pytest.mark.parametrize("action", ["error_flag", "error_text"])
def test_contradicted_action_earns_no_credit_in_any_scenario(action) -> None:
    result = run(frozen([source_a(), consumer_b(action=action)], [search_call()]))
    (verdict,) = result.verdicts
    assert verdict.reuse == "contradicted" and verdict.eligible
    assert all(v == () for v in verdict.atoms.values())
    assert credits(result) == (0, 0, 0) and nets(result) == (-50, -50, -50)


def test_observed_nonzero_exit_of_the_action_contradicts() -> None:
    failed = observed("a1", "solver-b", ("psql", "x"), tool="sql", native="cs2", exit_code=1)
    result = run(frozen([source_a(), consumer_b()], [search_call(), failed]))
    assert result.verdicts[0].reuse == "contradicted" and credits(result) == (0, 0, 0)
    ok = observed("a1", "solver-b", ("psql", "x"), tool="sql", native="cs2", exit_code=0)
    assert credits(run(frozen([source_a(), consumer_b()], [search_call(), ok])))[0] == 35


def test_opening_a_card_is_inspection_never_adoption() -> None:
    sb = consumer_b(action=None)
    lookup(sb, "cs3", show_json(f"{TOKEN} rule", 120), cmd=SHOW_CMD)
    result = run(frozen([source_a(), sb], [search_call(), show_call("cs3")]))
    assert result.verdicts[0].reuse == "inspected" and credits(result) == (0, 0, 0)
    assert result.estimates["conservative"].debit == (200 + 120) // 4  # the open is debited
    none = run(frozen([source_a(), consumer_b(action=None)], [search_call()]))
    assert none.verdicts[0].reuse == "none_observed" and credits(none) == (0, 0, 0)


def test_a_later_lookup_quoting_the_token_is_not_an_action() -> None:
    sb = consumer_b(action=None)
    lookup(
        sb, "cs3", search_json(f"{OTHER} only", 150), cmd=f'ssgrep search "{TOKEN} again" --json'
    )
    again = search_call(call_id="o2", native="cs3", argv=("ssgrep", "search", f"{TOKEN} again"))
    again = observed("o2", "solver-b", again.argv, native="cs3", stdout=search_json(OTHER, 150))
    result = run(frozen([source_a(), sb], [search_call(), again]))
    assert result.verdicts[0].reuse == "none_observed" and credits(result) == (0, 0, 0)


# ---- 12. production profile ---------------------------------------------------------------


@pytest.mark.parametrize("docs", [5, None, 199])
def test_production_policy_small_corpus_has_zero_eligible_credit_and_honest_net(docs) -> None:
    world, _sa, _sb = base_world()
    result = run(world, production_profile(docs))
    assert result.profile == PRODUCTION
    assert result.counts["eligible_candidates"] == 0
    assert result.verdicts and all(v.reason == "rarity_unknown" for v in result.verdicts)
    assert credits(result) == (0, 0, 0)
    assert nets(result) == (-50, -50, -50)  # net == -debit
    assert f"frequency_insufficient_corpus:{docs}/200" in result.unknowns


def test_production_policy_with_a_pinned_view_is_not_evaluated_not_invented() -> None:
    world, _sa, _sb = base_world()
    result = run(world, production_profile(200))
    assert "production_rarity_not_evaluated" in result.unknowns
    assert credits(result) == (None, None, None) and nets(result) == (None, None, None)
    assert result.estimates["central"].debit == 50 and result.counts["eligible_candidates"] == 0


# ---- 13. aggregation ----------------------------------------------------------------------


def test_refuse_aggregate_rejects_mixed_profiles_and_accepts_one() -> None:
    world, _sa, _sb = base_world()
    prod, arith = run(world, production_profile(5)), run(world)
    with pytest.raises(AggregationRefused):
        refuse_aggregate([prod, arith])
    refuse_aggregate([arith, run(world)])
    refuse_aggregate([prod, run(world, production_profile(7))])
    refuse_aggregate([])
    assert prod.non_aggregation.startswith(PRODUCTION) and arith.non_aggregation.startswith(
        ARITHMETIC
    )


# ---- 14. snapshot comparison --------------------------------------------------------------


def savings_doc(result) -> dict[str, Any]:
    def quantity(value):
        return {"value": value, "unit": "tokens", "basis": "estimated", "method": "chars_div4_v1"}

    estimates = {
        s: {
            "credit": quantity(e.credit), "debit": quantity(e.debit), "net": quantity(e.net),
        }
        for s, e in result.estimates.items()
    }  # fmt: skip
    measured = {
        "searches": {"value": result.counts["journaled_search"], "unit": "count"},
        "shows": {"value": result.counts["journaled_show"], "unit": "count"},
    }
    inner = {"estimates": {"context_tokens": estimates}, "measured": measured, "coverage": {}}
    return {"ok": True, "data": {"state": "fresh", "data": inner}}


def test_compare_with_snapshot_detects_doctored_numbers_and_accepts_the_real_one() -> None:
    world, _sa, _sb = base_world()
    result = run(world)
    doc = savings_doc(result)
    assert compare_with_snapshot(result, doc) == []
    assert compare_with_snapshot(result, copy.deepcopy(doc)) == []  # identical dict: positive twin
    flat = {
        "estimates": doc["data"]["data"]["estimates"],
        "measured": doc["data"]["data"]["measured"],
    }
    assert compare_with_snapshot(result, flat) == []  # tolerates the nesting

    def mutate(path: tuple[str, ...], value) -> list[str]:
        bad = copy.deepcopy(doc)
        node = bad["data"]["data"]
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        return compare_with_snapshot(result, bad)

    base = ("estimates", "context_tokens")
    debit = mutate((*base, "central", "debit", "value"), 51)
    assert len(debit) == 1 and "central.debit" in debit[0] and "50" in debit[0]
    assert "conservative.credit" in mutate((*base, "conservative", "credit", "value"), 36)[0]
    assert "optimistic.net" in mutate((*base, "optimistic", "net", "value"), 25)[0]
    assert "optimistic.net" in mutate((*base, "optimistic", "net", "value"), None)[0]
    assert "measured.searches" in mutate(("measured", "searches", "value"), 2)[0]
    assert "measured.shows" in mutate(("measured", "shows", "value"), 1)[0]
    assert "non-integer" in mutate((*base, "central", "debit", "value"), 50.0)[0]
    missing = {"ok": True, "data": {"data": {"measured": doc["data"]["data"]["measured"]}}}
    assert any("missing" in m for m in compare_with_snapshot(result, missing))


def test_compare_allows_none_only_when_oracle_says_unknown() -> None:
    sa, sb = source_a(), consumer_b()
    sb.call("cs3", 'for q in a b; do ssgrep search "$q"; done')
    sb.result("cs3", text(100, "mixed"))
    result = run(
        frozen(
            [sa, sb],
            [search_call(), observed("x", "solver-b", ("ssgrep", "search", "a"), native="cs3")],
        )
    )
    assert nets(result) == (None, None, None)
    doc = savings_doc(result)
    assert doc["data"]["data"]["estimates"]["context_tokens"]["central"]["net"]["value"] is None
    assert compare_with_snapshot(result, doc) == []
    known = savings_doc(run(base_world()[0]))
    assert compare_with_snapshot(result, known)  # the known-net snapshot disagrees with unknown


# ---- 15. counts and outcomes --------------------------------------------------------------


def test_outcome_counts_and_errored_lookups_debit_without_candidates() -> None:
    sa = source_a()
    empty_doc = json.dumps({"ok": True, "data": {"results": []}}, separators=(",", ":"))
    err_doc = json.dumps({"ok": False, "error": f"boom {TOKEN}"}, separators=(",", ":"))
    sb = SessionBuilder("sb", "solver-b", "L3", 5000)
    sb.user(text(30))
    lookup(sb, "e1", empty_doc)
    lookup(sb, "e2", text(60, f"search failed {TOKEN}"), error=True)
    lookup(sb, "e3", err_doc)
    lookup(sb, "e4", text(40, "no results"))
    calls = [
        observed("o1", "solver-b", ("ssgrep", "search", "a"), native="e1", stdout=empty_doc),
        observed("o2", "solver-b", ("ssgrep", "search", "b"), native="e2", exit_code=2),
        observed("o3", "solver-b", ("ssgrep", "search", "c"), native="e3", stdout=err_doc),
        observed("o4", "solver-b", ("ssgrep", "search", "d"), native="e4", exit_code=3),
    ]
    result = run(frozen([sa, sb], calls))
    assert result.counts["empty"] == 2 and result.counts["error"] == 2
    assert result.counts["exposures"] == 4 and result.verdicts == ()
    visible = len(empty_doc) + 60 + len(err_doc) + 40
    assert result.estimates["central"].debit == visible // 4 and credits(result) == (0, 0, 0)


def test_execution_states_and_journaled_expectation() -> None:
    sa, sb = source_a(), consumer_b()
    extra = [
        observed("a1", "solver-b", ("ssgrep", "search", "x"), native="cs1", state="attempted"),
        observed("a2", "solver-b", ("ssgrep", "show", "r"), native="cs1", state="interrupted"),
        observed("a3", "solver-b", ("ssgrep", "note", "n"), native="cs1", state="unknown"),
        observed("a4", "control", ("ssgrep", "show", "r"), state="verified"),
        observed("a5", "solver-b", ("sql", "q"), tool="sql", native="cs1"),
    ]
    result = run(frozen([sa, sb], [search_call(), *extra]))
    c = result.counts
    assert c["executions"] == 4 and c["executions_verified"] == 1
    assert (c["executions_attempted"], c["executions_interrupted"], c["executions_unknown"]) == (
        1,
        1,
        1,
    )
    assert c["journaled_search"] == 1 and c["journaled_show"] == 1  # verified only, all actors
    assert c["control_show"] == 1 and c["solver-b.note"] == 1 and c["journaled_note"] == 0


# ---- 16. integers only --------------------------------------------------------------------


def _walk(value: Any) -> None:
    assert not isinstance(value, float), value
    if isinstance(value, dict):
        for item in value.values():
            _walk(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _walk(item)


@pytest.mark.parametrize("profile", [PROFILE, production_profile(5), production_profile(500)])
def test_results_hold_no_float_and_negative_net_is_a_negative_int(profile) -> None:
    world, _sa, _sb = base_world()
    result = run(world, profile)
    _walk(result.as_dict())
    json.dumps(result.as_dict())
    for est in result.estimates.values():
        assert est.net is None or type(est.net) is int
    assert type(run(world).estimates["conservative"].net) is int
    assert run(world).estimates["conservative"].net < 0
