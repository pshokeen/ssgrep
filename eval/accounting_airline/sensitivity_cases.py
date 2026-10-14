"""The sensitivity cases themselves (see ``sensitivity`` for the fixture and result types).

Each case runs REAL binaries, builds its expectation from the independent observer/oracle, reads
the product's own public outputs (``metrics collect``/``savings --json``/``metrics status
--json``) and records every mismatch with the evidence it points to. A mismatch between product
and oracle is never bent either way: it becomes a ``DISAGREEMENT`` whose diagnosis names which
side the design (D3-D7) supports.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import accounting_oracle as oracle, lab_product as lp, observer as obs
from .sensitivity import (
    DEN,
    ORD,
    QUERY,
    REVIEWED,
    SOURCE_ID,
    SOURCE_REF,
    CaseResult,
    Check,
    Fixture,
    Scene,
    arithmetic,
    classify_call,
    estimate,
    finish,
    journal_bytes,
    journal_expectation,
    measured_values,
    oracle_estimate,
    production,
    snapshot_id,
)
from .standin_agent import Bash, DocRead, Say, Step

LOOKUPS = ("search", "show")
ACT = Bash(f"printf '%s\\n' {DEN} > notes.txt")


def look() -> list[Step]:
    return [Bash(f'ssgrep search "{QUERY}" --json'), Bash(f"ssgrep show {SOURCE_REF} --json")]


# ------------------------------------------------------------------------------ evidence


def solver_lookups(frozen: obs.Frozen) -> list[obs.ObservedCall]:
    return [
        c
        for c in frozen.calls
        if c.actor == "solver-b" and c.tool == "ssgrep" and len(c.argv) > 1 and c.argv[1] in LOOKUPS
    ]


def visible_chars(frozen: obs.Frozen) -> int:
    """Characters of consumer-visible native lookup responses, including usage errors."""
    hosts = {c.native_call_id for c in solver_lookups(frozen) if c.native_call_id}
    seen: set[tuple[str, str]] = set()
    total = 0
    for sid, (actor, _task, native) in frozen.sessions.items():
        if actor != "solver-b":
            continue
        for r in native.records:
            if r.kind == "tool_result" and r.call_id in hosts and (sid, r.call_id) not in seen:
                seen.add((sid, r.call_id))
                total += len(r.text)
    return total


def native_text(frozen: obs.Frozen, session_id: str, kind: str) -> list[str]:
    return [r.text for r in frozen.sessions[session_id][2].records if r.kind == kind]


def journal_rows(data: Path) -> Counter[tuple[str, str]]:
    """(operation, outcome) of every COMPLETED journal event (the product's own record)."""
    rows: Counter[tuple[str, str]] = Counter()
    root = data / "accounting" / lp.PROFILE / "journal"
    for path in sorted(root.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("phase") == "completed":
                rows[(str(row.get("operation")), str(row.get("outcome")))] += 1
    return rows


def expected_failed(frozen: obs.Frozen) -> int:
    """Executions a reader of the transcript sees fail: ``ok:false``/non-zero exit for solvers."""
    total = 0
    for call in frozen.calls:
        if call.tool != "ssgrep" or len(call.argv) < 2 or call.argv[1] not in (*LOOKUPS, "note"):
            continue
        kind = classify_call(call)
        if kind in ("usage", "open"):
            continue
        if call.actor in ("solver-a", "solver-b"):
            total += call.exit_code != 0
        else:
            total += kind == "error"
    return total


def reduction_state(data: Path) -> dict[str, Any]:
    """The newest retained reduction document of the isolated profile (product's own state)."""
    state = data / "accounting" / lp.PROFILE / "collector" / "state"
    files = sorted(state.glob("reduction-*.json"), key=lambda p: p.stat().st_mtime)
    return json.loads(files[-1].read_text()) if files else {}


def find_key(doc: Any, key: str) -> Any:
    if isinstance(doc, dict):
        if key in doc:
            return doc[key]
        for value in doc.values():
            found = find_key(value, key)
            if found is not None:
                return found
    return None


def net_reason(collected: lp.Collected, scenario: str = "central") -> str | None:
    row = (collected.estimates.get("context_tokens") or {}).get(scenario) or {}
    return (row.get("net") or {}).get("reason")


def scoped_estimate(
    sc: Scene, profile: oracle.ProfileSpec, session_id: str, scenario: str = "central"
) -> dict[str, int | None]:
    document = sc.savings_for_session(profile, session_id)
    estimates = find_key(document, "estimates") or {}
    row = (estimates.get(oracle.UNIT) or {}).get(scenario) or {}
    return {key: (row.get(key) or {}).get("value") for key in ("credit", "debit", "net")}


def agree(
    check: Check,
    label: str,
    sc: Scene,
    profile: Any,
    collected: lp.Collected,
    skip: tuple[str, ...] = (),
) -> None:
    mism = sc.compare(profile, collected, skip=skip)
    check.ok(f"{label}: oracle vs snapshot", not mism, "; ".join(mism))


# ---------------------------------------------------------------------------- the cases


def empty_error_missing_lookups(fx: Fixture) -> CaseResult:
    name, check = "empty_error_missing_lookups", Check()
    sc = Scene(fx, "c1")
    base = journal_rows(fx.base_data)
    steps: list[Step] = [
        Bash('ssgrep search "zzqx nothing matches" --where "project = \'/nonexistent\'" --json'),
        Bash("ssgrep show omp:nope:ep:0 --json"),
        Bash('ssgrep search x --where "bogus (((" --json'),
        Bash("ssgrep search x --limit abc --json"),
        Say("done"),
    ]
    sc.consume("c1probe01", "Probe empty, missing and invalid lookups.", steps)
    frozen = sc.frozen()
    mine = solver_lookups(frozen)
    kinds = [classify_call(c) for c in mine]
    check.eq("observed kinds", ["empty", "empty", "error", "usage"], kinds)
    check.eq("observed exits", [3, 3, 2, 2], [c.exit_code for c in mine])
    expect = journal_expectation(frozen)
    col = sc.collect()
    measured = measured_values(col)
    rows = journal_rows(sc.data) - base
    check.eq(
        "journal rows added",
        Counter({("search", "empty"): 1, ("show", "empty"): 1, ("search", "error"): 1}),
        rows,
    )
    check.eq("measured.searches", expect["search"], measured.get("searches"))
    check.eq("measured.shows", expect["show"], measured.get("shows"))
    check.eq("measured.failed", expected_failed(frozen), measured.get("failed"))
    debit = visible_chars(frozen) // 4
    scoped = scoped_estimate(sc, production(), "c1probe01")
    check.eq("consumer-root debit excludes pre-API usage", debit, scoped["debit"])
    result = sc.oracle_result(production())
    check.eq(
        "oracle journaled counts exclude pre-API usage",
        (expect["search"], expect["show"]),
        (result.counts["journaled_search"], result.counts["journaled_show"]),
    )
    check.eq(
        "oracle execution count excludes --limit usage, includes invalid --where",
        sum(classify_call(call) not in ("usage", "open") for call in mine),
        result.counts["executions"],
    )
    check.eq("oracle sees service semantic errors", expected_failed(frozen), result.counts["error"])
    agree(check, "production debit", sc, production(), col)
    check.ok(
        "positive twin: valid lookups are journaled as ok",
        base[("search", "ok")] >= 1 and base[("show", "ok")] >= 1,
        str(dict(base)),
    )
    exp = {
        "measured.searches": expect["search"],
        "measured.shows": expect["show"],
        "measured.failed": expected_failed(frozen),
        "debit": debit,
        "usage_not_journaled": 1,
        "journal_rows_added": 3,
    }
    act = {
        "measured.searches": measured.get("searches"),
        "measured.shows": measured.get("shows"),
        "measured.failed": measured.get("failed"),
        "debit": scoped["debit"],
        "global_profile_debit": estimate(col)["debit"],
        "journal_rows_added": sum(rows.values()),
    }
    return finish(
        name,
        check,
        exp,
        act,
        sc.evidence(col),
        "journal records the scope/missing-ref lookups as empty; the reducer overrides "
        "to error from the visible ok:false payload (measured.failed counts them)",
    )


def tiny_show_truncation(fx: Fixture) -> CaseResult:
    name, check = "tiny_show_truncation", Check()
    hidden = Scene(fx, "c2h")
    hidden.consume(
        "c2hid001",
        "Show with a tiny output budget.",
        [Bash(f"ssgrep show {SOURCE_REF} --json | head -c 120"), ACT, Say("done")],
    )
    frozen = hidden.frozen()
    (shown,) = solver_lookups(frozen)
    visible = native_text(frozen, "c2hid001", "tool_result")[0]
    check.ok("the identifier is in the real full output", DEN in shown.stdout)
    check.ok("the identifier is NOT in what the model saw", DEN not in visible)
    actions = [t for t in native_text(frozen, "c2hid001", "tool_call") if "notes.txt" in t]
    check.ok("a later action uses the hidden identifier", bool(actions) and DEN in actions[0])
    prod_col = hidden.collect()
    arith_col, _ = hidden.arithmetic()
    debit = visible_chars(frozen) // 4
    check.ok(
        "debit = visible chars // 4 > 0",
        debit > 0 and estimate(arith_col)["debit"] == debit,
    )
    check.eq("hidden credit (product, arithmetic)", 0, estimate(arith_col)["credit"])
    check.eq("hidden credit (product, production)", 0, estimate(prod_col)["credit"])
    ours = hidden.oracle_result(arithmetic())
    check.eq("hidden credit (oracle)", 0, oracle_estimate(ours)["credit"])
    agree(check, "hidden arithmetic", hidden, arithmetic(), arith_col, skip=("net",))
    agree(check, "hidden production", hidden, production(), prod_col, skip=("net",))
    reason = net_reason(arith_col)
    want_net = oracle_estimate(ours)["net"]
    check.eq(
        "measured partial exposure has a known negative net",
        want_net,
        estimate(arith_col)["net"],
    )
    if estimate(arith_col)["net"] != want_net:
        check.problems.append(
            f"mismatch evidence: native toolResult chars={len(visible)}; output format=truncated "
            "JSON/text from 'ssgrep show --json | head -c 120'; observed comparable "
            f"chars_div4_v1 debit={len(visible)}//4={len(visible) // 4}, candidate absent from "
            f"visible text, product net={estimate(arith_col)['net']} "
            f"({reason}), independent oracle net={want_net}. D6 subtracts the same comparable "
            "known debit; partial exposure limits credit, not measured response cost."
        )
    twin = Scene(fx, "c2f")
    twin.consume(
        "c2ful001",
        "Show without a budget.",
        [Bash(f"ssgrep show {SOURCE_REF} --json"), ACT, Say("done")],
    )
    twin_col, _ = twin.arithmetic()
    twin_oracle = twin.oracle_result(arithmetic())
    check.ok("positive twin: full show earns credit", (estimate(twin_col)["credit"] or 0) > 0)
    agree(check, "full-show twin", twin, arithmetic(), twin_col)
    exp = {
        "credit": 0,
        "debit": debit,
        "twin_credit": oracle_estimate(twin_oracle)["credit"],
        "net": want_net,
    }
    act = {
        "credit": estimate(arith_col)["credit"],
        "debit": estimate(arith_col)["debit"],
        "twin_credit": estimate(twin_col)["credit"],
        "net": estimate(arith_col)["net"],
    }
    return finish(name, check, exp, act, (*hidden.evidence(arith_col), *twin.evidence(twin_col)))


def repeated_lookups(fx: Fixture) -> CaseResult:
    name, check = "repeated_lookups", Check()
    single, double = Scene(fx, "c3s"), Scene(fx, "c3d")
    single.consume("c3sin001", "One lookup.", [*look(), ACT, Say("done")])
    double.consume("c3dbl001", "The same lookup twice.", [*look(), *look(), ACT, Say("done")])
    one, _ = single.arithmetic()
    two, _ = double.arithmetic()
    f1, f2 = single.frozen(), double.frozen()
    one_chars, two_chars = visible_chars(f1), visible_chars(f2)
    check.eq("single debit", one_chars // 4, estimate(one)["debit"])
    check.eq("double debit (each exposure charged)", two_chars // 4, estimate(two)["debit"])
    check.ok("debit increased", (estimate(two)["debit"] or 0) > (estimate(one)["debit"] or 0))
    for scen in ("conservative", "central", "optimistic"):
        a, b = estimate(one, scen)["credit"], estimate(two, scen)["credit"]
        check.eq(f"credit unchanged ({scen})", a, b)
        check.ok(f"credit positive ({scen})", bool(a) and a > 0)
    agree(check, "single", single, arithmetic(), one)
    agree(check, "double", double, arithmetic(), two)
    prod_two = double.collect()
    check.eq(
        "measured.searches",
        journal_expectation(f2)["search"],
        measured_values(prod_two)["searches"],
    )
    agree(check, "double production", double, production(), prod_two)
    exp = {
        "debit_single": one_chars // 4,
        "debit_double": two_chars // 4,
        "credit_central": estimate(one)["credit"],
    }
    act = {
        "debit_single": estimate(one)["debit"],
        "debit_double": estimate(two)["debit"],
        "credit_central": estimate(two)["credit"],
    }
    return finish(name, check, exp, act, (*single.evidence(one), *double.evidence(two)))


def mentions_not_executions(fx: Fixture) -> CaseResult:
    name, check = "mentions_not_executions", Check()
    sc = Scene(fx, "c4")
    before = journal_bytes(sc.data)
    sc.consume(
        "c4men001",
        "Describe the lookup tool.",
        [
            Bash('echo "run: ssgrep search x --json then ssgrep show omp:x:ep:0 --json"'),
            Bash("cat <<'EOF' >/dev/null\nssgrep search \"quoted only\" --json\nEOF"),
            DocRead("~/tool-docs/ssgrep.md", "Use `ssgrep search <query> --json` to find history."),
            Say("done"),
        ],
    )
    frozen = sc.frozen()
    check.eq("observed solver ssgrep dispatches", 0, len(solver_lookups(frozen)))
    check.eq("journal bytes unchanged", before, journal_bytes(sc.data))
    base_counts = journal_expectation(frozen)
    col = sc.collect()
    measured = measured_values(col)
    check.eq("measured.searches", base_counts["search"], measured["searches"])
    check.eq("measured.shows", base_counts["show"], measured["shows"])
    check.eq("no solver lookup is hosted by a mention", 0, len(solver_lookups(frozen)))
    debit = estimate(col)["debit"]
    check.ok("no debit for mentions", debit in (None, 0))
    result = sc.oracle_result(production())
    check.eq("oracle invocations", 0, result.counts["invocations"])
    check.eq("oracle debits", (), result.debits)
    note = (
        f"product estimate debit={debit!r} ({(col.estimates.get('context_tokens') or {}).get('central', {}).get('debit', {}).get('reason')}),"  # noqa: E501
        " oracle 0: both mean no debit"
    )
    twin = Scene(fx, "c4t")
    twin.consume("c4twin01", "A real lookup.", [*look(), Say("done")])
    tf = twin.frozen()
    tcol = twin.collect()
    check.ok(
        "positive twin: a real lookup changes the journal",
        journal_bytes(twin.data) != journal_bytes(fx.base_data),
    )
    check.eq(
        "twin measured.searches",
        journal_expectation(tf)["search"],
        measured_values(tcol)["searches"],
    )
    check.eq("twin debit", visible_chars(tf) // 4, estimate(tcol)["debit"])
    agree(check, "twin", twin, production(), tcol)
    exp = {"solver_dispatches": 0, "debit": 0, "twin_debit": visible_chars(tf) // 4}
    act = {
        "solver_dispatches": len(solver_lookups(frozen)),
        "debit": debit,
        "twin_debit": estimate(tcol)["debit"],
    }
    return finish(name, check, exp, act, (*sc.evidence(col), *twin.evidence(tcol)), note)


def shell_loop_two_searches(fx: Fixture) -> CaseResult:
    name, check = "shell_loop_two_searches", Check()
    loop = Scene(fx, "c5l")
    loop.consume(
        "c5loop01",
        "Two searches in one shell loop.",
        [
            Bash(
                'for q in "minimum connection" "airport rules"; do ssgrep search "$q" --json; done'
            ),
            ACT,
            Say("done"),
        ],
    )
    frozen = loop.frozen()
    mine = solver_lookups(frozen)
    check.eq("observed executions", 2, len(mine))
    check.eq("one native tool call hosts both", 1, len({c.native_call_id for c in mine}))
    result = loop.oracle_result(production())
    check.eq("oracle invocations", 1, result.counts["invocations"])
    check.eq("oracle executions", 2, result.counts["executions"])
    expect = journal_expectation(frozen)
    col = loop.collect()
    check.eq(
        "measured.searches (journal has both)", expect["search"], measured_values(col)["searches"]
    )
    arith_col, _ = loop.arithmetic()
    chars = visible_chars(frozen)
    one_exposure = chars // 4
    check.eq("product charges the one aggregate output once", one_exposure, estimate(col)["debit"])
    ours = loop.oracle_result(arithmetic())
    unknown = [d.reason for d in ours.debits]
    if oracle_estimate(ours)["debit"] != estimate(arith_col)["debit"]:
        check.disagree(
            "debit of a pure ssgrep loop",
            f"oracle debit={oracle_estimate(ours)['debit']} (reason {unknown}: any invocation "
            f"hosting >1 dispatch is 'mixed') vs product debit={estimate(arith_col)['debit']}, "
            f"credit {estimate(arith_col)['credit']} vs oracle {oracle_estimate(ours)['credit']}: "
            "reduce_calls/exposure.shell_shape classifies a loop of ONLY ssgrep commands as "
            "'pure', so the whole concatenated output is one observed exposure (charged once, "
            "never as two complete exposures) and tokens are scanned from the unparsed text. "
            "D6 reserves unknown/partial for output mixed with non-retrieval work, so the debit "
            "is defensible (exact visible chars, charged once); crediting from an unparsable "
            "aggregate (D5 'aggregate is not automatically two complete exposures') is the "
            "questionable half. Needs a design decision; the oracle rule is stricter than D6",
        )
    mixed = Scene(fx, "c5m")
    mixed.consume(
        "c5mix001",
        "A search mixed with other shell output.",
        [Bash(f'ssgrep search "{QUERY}" --json; echo unrelated-output'), ACT, Say("done")],
    )
    mcol, _ = mixed.arithmetic()
    agree(check, "mixed twin (unknown on both sides)", mixed, arithmetic(), mcol)
    check.eq("mixed twin net unavailable", None, estimate(mcol)["net"])
    check.ok(
        "mixed twin reason",
        net_reason(mcol) == "incomplete:mixed_shell_output_unattributable",
        str(net_reason(mcol)),
    )
    separate = Scene(fx, "c5s")
    separate.consume(
        "c5sep001",
        "The same two searches as two calls.",
        [
            Bash('ssgrep search "minimum connection" --json'),
            Bash('ssgrep search "airport rules" --json'),
            ACT,
            Say("done"),
        ],
    )
    scol, _ = separate.arithmetic()
    check.eq(
        "separate calls: each response debited",
        visible_chars(separate.frozen()) // 4,
        estimate(scol)["debit"],
    )
    agree(check, "separate-calls twin", separate, arithmetic(), scol)
    exp = {
        "invocations": 1,
        "executions": 2,
        "measured_searches": expect["search"],
        "debit_once": one_exposure,
        "oracle_debit": oracle_estimate(ours)["debit"],
    }
    act = {
        "invocations": 1,
        "executions": len(mine),
        "measured_searches": measured_values(col)["searches"],
        "debit_once": estimate(col)["debit"],
        "product_debit_arith": estimate(arith_col)["debit"],
        "product_credit_arith": estimate(arith_col)["credit"],
    }
    return finish(
        name,
        check,
        exp,
        act,
        (*loop.evidence(col), *mixed.evidence(mcol), *separate.evidence(scol)),
    )


def prior_known_and_query(fx: Fixture) -> CaseResult:
    name, check = "prior_known_and_query", Check()
    known = Scene(fx, "c6k")
    known.consume(
        "c6prior1",
        f"Plan using policy {DEN}; check history for the rest.",
        [
            Bash(f'ssgrep search "{QUERY} {ORD}" --json'),
            Bash(f"ssgrep show {SOURCE_REF} --json"),
            Bash(f"printf '%s\\n' {DEN} {ORD} > notes.txt"),
            Say("done"),
        ],
    )
    kcol, _ = known.arithmetic()
    ours = known.oracle_result(arithmetic())
    reasons = {
        v.token[1]: v.reason for v in ours.verdicts if v.exposure_call == ours.debits[0].call
    }
    check.eq("DEN is prior_known", "prior_known", reasons.get(DEN))
    check.eq("ORD is a query mention", "query_mention", reasons.get(ORD))
    for scen in ("conservative", "central", "optimistic"):
        check.eq(f"no credit ({scen}, product)", 0, estimate(kcol, scen)["credit"])
        check.eq(f"no credit ({scen}, oracle)", 0, oracle_estimate(ours, scen)["credit"])
    agree(check, "prior-known", known, arithmetic(), kcol)
    clean = Scene(fx, "c6c")
    clean.consume(
        "c6clean1", "Plan the reaccommodation using airline history.", [*look(), ACT, Say("done")]
    )
    ccol, _ = clean.arithmetic()
    check.ok(
        "positive twin: the same action without prior knowledge earns credit",
        (estimate(ccol)["credit"] or 0) > 0,
    )
    agree(check, "clean twin", clean, arithmetic(), ccol)
    exp = {"credit": 0, "twin_credit": oracle_estimate(clean.oracle_result(arithmetic()))["credit"]}
    act = {"credit": estimate(kcol)["credit"], "twin_credit": estimate(ccol)["credit"]}
    return finish(name, check, exp, act, (*known.evidence(kcol), *clean.evidence(ccol)))


def _relay_scene(
    fx: Fixture, tag: str, *, payload: bool, header: bool, sibling: bool
) -> tuple[Scene, dict[str, str]]:  # noqa: E501
    sc = Scene(fx, tag)
    parent, child, sib = f"{tag}par1", f"{tag}chd1", f"{tag}sib1"
    parent_file = sc.start_session(parent)
    declared = sc.view / parent_file.name

    def run_child(_ctx: Any) -> str:
        sc.consume(
            child,
            "Child: look up the connection rules.",
            [*look(), Say("child done")],
            parent=declared if header else None,
        )
        return f'echo "{child} reported {DEN}"' if payload else f'echo "{child} finished"'

    sc.consume(
        parent,
        "Parent: delegate the rule lookup.",
        [Bash(f"echo launching subagent {child}"), Bash(run_child), ACT, Say("done")],
    )
    if sibling:
        sc.consume(sib, "Unrelated sibling.", [ACT, Say("done")])
    return sc, {"parent": parent, "child": child, "sibling": sib}


def relay_vs_sibling(fx: Fixture) -> CaseResult:
    name, check = "relay_vs_sibling", Check()
    relay, ids = _relay_scene(fx, "c7a", payload=True, header=True, sibling=False)
    with_sib, sib_ids = _relay_scene(fx, "c7b", payload=True, header=True, sibling=True)
    no_payload, _ = _relay_scene(fx, "c7c", payload=False, header=True, sibling=False)
    no_header, _ = _relay_scene(fx, "c7d", payload=True, header=False, sibling=False)
    rcol, rprod = relay.arithmetic()
    scol, sprod = with_sib.arithmetic()
    ncol, _ = no_payload.arithmetic()
    hcol, _ = no_header.arithmetic()
    frozen = relay.frozen()
    parent_calls = [t for t in native_text(frozen, ids["parent"], "tool_call") if "notes.txt" in t]
    results = " ".join(native_text(frozen, ids["parent"], "tool_result"))
    check.ok(
        "evidence: parent result names the child and carries the token",
        ids["child"] in results and DEN in results,
    )
    check.ok(
        "evidence: parent action uses the token", bool(parent_calls) and DEN in parent_calls[0]
    )
    check.eq("evidence: the child's lookups are observed", 2, len(solver_lookups(frozen)))
    code, sdoc = sprod.json("savings", "--session", sib_ids["sibling"])
    check.eq("sibling savings read exits 0", 0, code)
    sib_credit = lp.find(sdoc, "estimates") or {}
    sib_row = ((sib_credit.get("context_tokens") or {}).get("central") or {}).get("credit") or {}
    check.ok(
        "sibling coincidence earns no credit (--session sibling)",
        sib_row.get("value") in (None, 0),
        str(sib_row),
    )
    for scen in ("conservative", "central", "optimistic"):
        check.eq(
            f"sibling adds no credit ({scen})",
            estimate(rcol, scen)["credit"],
            estimate(scol, scen)["credit"],
        )
    check.eq("no relay payload -> no credit", 0, estimate(ncol)["credit"])
    check.eq("no declared parent -> no credit", 0, estimate(hcol)["credit"])
    relay_credit = estimate(rcol)["credit"]
    check.ok(
        "explicit relay with matching payload is credited by the product",
        (relay_credit or 0) > 0,
        str(relay_credit),
    )
    ours = relay.oracle_result(arithmetic())
    oracle_credit = oracle_estimate(ours)["credit"]
    if relay_credit != oracle_credit:
        check.disagree(
            "explicit parent relay",
            f"product credit={relay_credit} (conservative "
            f"{estimate(rcol, 'conservative')['credit']}, optimistic "
            f"{estimate(rcol, 'optimistic')['credit']}) vs oracle {oracle_credit}: the "
            "product joins the child's exposure to the parent's later action through the "
            "declared parentSession plus a parent result that names the child and carries the "
            "token (reuse._relay). D5: 'Explicit parent relay with matching payload evidence "
            "can join child exposure to parent action; coincidental occurrence in another "
            "sibling cannot', so the product is right and oracle_scan.reuse only looks inside "
            "the exposing session (no relay model)",
        )
    agree(check, "no-payload twin", no_payload, arithmetic(), ncol)
    agree(check, "no-header twin", no_header, arithmetic(), hcol)
    exp = {
        "relay_credit": oracle_credit,
        "sibling_extra_credit": 0,
        "no_payload_credit": 0,
        "no_header_credit": 0,
    }
    act = {
        "relay_credit": relay_credit,
        "sibling_extra_credit": (estimate(scol)["credit"] or 0) - (relay_credit or 0),
        "no_payload_credit": estimate(ncol)["credit"],
        "no_header_credit": estimate(hcol)["credit"],
    }
    return finish(name, check, exp, act, (*relay.evidence(rcol), *with_sib.evidence(scol)))


def _mutated_copy(source: Path, target: Path) -> None:
    out, done = [], False
    for line in source.read_text().splitlines():
        record = json.loads(line)
        message = record.get("message") or {}
        if not done and message.get("role") == "toolResult" and DEN in json.dumps(message):
            message["content"][0]["text"] = message["content"][0]["text"].replace(
                DEN, "MCT-XXX-999"
            )
            done = True
        out.append(json.dumps(record))
    target.write_text("\n".join(out) + "\n")


def duplicate_mutated_partial_histories(fx: Fixture) -> CaseResult:
    name, check = "duplicate_mutated_partial_histories", Check()
    sc = Scene(fx, "c8")
    path = sc.consume(
        "c8dup0001",
        "Lookups whose history is later copied and damaged.",
        [*look(), ACT, Say("done")],
    )
    before_oracle = sc.oracle_result(arithmetic())
    complete_before = sc.frozen().sessions["c8dup0001"][2].complete_bytes
    clean, _ = sc.arithmetic()
    clean_prod = sc.collect()
    base_digest, prod_digest = lp.savings_digest_view(clean), lp.savings_digest_view(clean_prod)
    shutil.copyfile(path, sc.view / "copy-of-session.jsonl")
    _mutated_copy(path, sc.view / "mutated-copy.jsonl")
    tail = (
        b'{"type":"message","id":"deadbeef","parentId":null,"timestamp":"2026-10-13T18:00:00.000Z",'
        b'"message":{"role":"toolResult","toolCallId":"x","toolName":"bash","content":'
        b'[{"type":"text","text":"' + ORD.encode() + b" partial"
    )
    with path.open("ab") as handle:  # no trailing newline: a partial JSONL tail
        handle.write(tail)
    sc.sync("c8dup0001", path, "c8dup0001#0")
    frozen = sc.frozen()
    native = frozen.sessions["c8dup0001"][2]
    check.eq("oracle: partial tail bytes", len(tail), native.partial_tail_bytes)
    check.eq(
        "oracle: complete prefix unchanged by the tail", complete_before, native.complete_bytes
    )
    check.ok(
        "oracle: no record from the partial tail",
        all("partial" not in r.text for r in native.records),
    )
    after, _ = sc.arithmetic()
    after_prod = sc.collect()
    after_oracle = sc.oracle_result(arithmetic())
    check.eq(
        "oracle estimates unchanged",
        before_oracle.estimates["central"].as_dict(),
        after_oracle.estimates["central"].as_dict(),
    )
    check.eq("product arithmetic digest unchanged", base_digest, lp.savings_digest_view(after))
    check.eq("product production digest unchanged", prod_digest, lp.savings_digest_view(after_prod))
    check.eq("usage not doubled (debit)", estimate(clean)["debit"], estimate(after)["debit"])
    check.eq(
        "measured.searches",
        journal_expectation(frozen)["search"],
        measured_values(after_prod)["searches"],
    )
    agree(check, "after duplicate/mutated/partial", sc, arithmetic(), after)
    gaps = after.status["data"]["missing_evidence"]["gaps"]
    check.ok("conflicting variant reported", (gaps.get("dropped_conflict") or 0) >= 1, str(gaps))
    pending = [w["pending_tail_bytes"] for w in after.status["data"]["watermarks"]["items"]]
    check.ok(
        "partial tail excluded from complete bytes (product watermark)",
        len(tail) in pending,
        str(pending),
    )
    check.eq("product lag counts the tail", len(tail), after.status["data"]["lag"]["lag_bytes"])
    exp = {"debit": estimate(clean)["debit"], "partial_tail_bytes": len(tail), "conflicts_min": 1}
    act = {
        "debit": estimate(after)["debit"],
        "partial_tail_bytes": max(pending or [0]),
        "conflicts_min": gaps.get("dropped_conflict"),
    }
    return finish(
        name, check, exp, act, (*sc.evidence(after), f"snapshot:{snapshot_id(clean)[:16]}")
    )


def journal_write_loss(fx: Fixture) -> CaseResult:
    name, check = "journal_write_loss", Check()
    sc = Scene(fx, "c9")
    jdir = sc.data / "accounting" / lp.PROFILE / "journal"
    sc.consume(
        "c9norm01",
        "A normal lookup.",
        [Bash('ssgrep search "connection policies denver" --json'), Say("done")],
    )
    time.sleep(1.1)
    before = journal_bytes(sc.data)
    os.chmod(jdir, 0o500)
    try:
        sc.consume(
            "c9lost01",
            "A lookup whose journal write is lost.",
            [Bash(f'ssgrep search "{QUERY}" --json'), Say("done")],
        )
    finally:
        os.chmod(jdir, 0o700)
    check.eq("journal gained nothing for the lost lookup", before, journal_bytes(sc.data))
    time.sleep(1.1)
    sc.consume(
        "c9norm02",
        "Another normal lookup.",
        [Bash('ssgrep search "airport rules chicago" --json'), Say("done")],
    )
    frozen = sc.frozen()
    lost = next(c for c in solver_lookups(frozen) if c.argv[2] == QUERY)
    verifier = next(c for c in frozen.calls if c.actor == "verifier" and c.argv[1] == "search")
    check.eq("solver behavior: exit code", verifier.exit_code, lost.exit_code)
    check.eq("solver behavior: stdout", verifier.stdout, lost.stdout)
    expect = journal_expectation(frozen)
    col = sc.collect()
    measured = measured_values(col)
    gap = expect["search"] - (measured["searches"] or 0)
    check.eq("journal holds one fewer search than observed executions", 1, gap)
    state = reduction_state(sc.data)
    check.eq("product classifies it as a journal_gap", 1, find_key(state, "inv_journal_gap"))
    complete = col.status["data"]["missing_evidence"]["capture_completeness"]
    check.eq("product never claims exact completeness", "unknown", complete)
    check.ok("coverage is partial", col.coverage.get("partial") is True)
    check.ok(
        "no fabricated loss count anywhere in the public snapshot",
        "lost_count" not in json.dumps(col.savings_document),
    )
    debit = visible_chars(frozen) // 4
    check.eq(
        "the visible response is still debited from the transcript", debit, estimate(col)["debit"]
    )
    agree(check, "production", sc, production(), col)
    exp = {
        "observed_searches": expect["search"],
        "journal_gap_min": 1,
        "capture_completeness": "unknown",
    }
    act = {
        "observed_searches": measured["searches"] if gap == 0 else expect["search"],
        "journal_searches": measured["searches"],
        "journal_gap": find_key(state, "inv_journal_gap"),
        "capture_completeness": complete,
    }
    return finish(
        name,
        check,
        exp,
        act,
        (*sc.evidence(col), f"journal_gap_vs_observed:{gap}"),
        "journal_gap evidence = observed verified executions minus journal executions; "
        "the public snapshot exposes only capture_completeness=unknown and partial "
        "coverage, the per-class count lives in the reducer's retained state",
    )


def source_expiry(fx: Fixture) -> CaseResult:
    name, check = "source_expiry", Check()
    sc = Scene(fx, "c10")
    sc.consume("c10exp01", "Lookups whose source later expires.", [*look(), ACT, Say("done")])
    before, product = sc.arithmetic()
    base = lp.savings_digest_view(before)
    oracle = oracle_estimate(sc.oracle_result(arithmetic()))
    agree(check, "before expiry", sc, arithmetic(), before)
    check.ok("credit is positive before expiry", (estimate(before)["credit"] or 0) > 0)
    next(sc.view.glob(f"*_{SOURCE_ID}.jsonl")).unlink()
    after = lp.collect(product)
    check.eq("closed totals kept after the source disappeared", base, lp.savings_digest_view(after))
    check.eq("no shrink (central credit)", estimate(before)["credit"], estimate(after)["credit"])
    check.eq("no shrink (central debit)", estimate(before)["debit"], estimate(after)["debit"])
    check.eq("product still equals the oracle over the frozen evidence", oracle, estimate(after))
    missing = after.status["data"]["missing_evidence"]["missing_sources"]
    check.eq("the missing source is reported", 1, missing)
    recomputed = lp.collect(product, "--recompute")
    check.eq(
        "recompute from retained evidence keeps the totals",
        base,
        lp.savings_digest_view(recomputed),
    )
    dest = sc.dir / "replay-without-source"
    lp.copy_accounting(sc.data, dest, journal_only=True)
    fresh = lp.Product({**sc.product.env, "SSGREP_DATA_DIR": str(dest)}, sc.product.cwd)
    lp.pin_arithmetic_view(fresh, [t for _, t in REVIEWED])
    gone = lp.collect(fresh)
    check.eq(
        "fresh replay: unavailable source reported", 1, gone.coverage.get("unavailable_sources")
    )
    check.eq(
        "fresh replay: no source cost available", 0, gone.coverage.get("source_costs_available")
    )
    check.eq("fresh replay: net is not claimed", None, estimate(gone)["net"])
    check.ok(
        "fresh replay: labeled non-recomputable",
        (net_reason(gone) or "").startswith("incomplete"),
        str(net_reason(gone)),
    )
    check.ok(
        "fresh replay: totals never negative-shrunk silently",
        estimate(gone)["debit"] == estimate(before)["debit"],
    )
    exp = {"credit": oracle["credit"], "debit": oracle["debit"], "missing_sources": 1}
    act = {
        "credit": estimate(after)["credit"],
        "debit": estimate(after)["debit"],
        "missing_sources": missing,
        "fresh_replay_net": estimate(gone)["net"],
    }
    return finish(
        name,
        check,
        exp,
        act,
        (*sc.evidence(before, after, gone),),
        "the period is still inside the 7-day provisional window, so the sealed-receipt "
        "branch of D7 is not reachable without clock manipulation; the retained-evidence"
        " branch (totals kept, missing source reported, replay without the source "
        "labeled incomplete) is what is asserted",
    )


def recollection_replay_no_extra_usage(fx: Fixture) -> CaseResult:
    name, check = "recollection_replay_no_extra_usage", Check()
    sc = Scene(fx, "c11")
    sc.consume("c11rep001", "Lookups that are then recollected.", [*look(), ACT, Say("done")])
    first = sc.collect()
    digest = lp.savings_digest_view(first)
    bytes_before = journal_bytes(sc.data)
    second = sc.collect()
    recomputed = sc.collect("--recompute")
    check.eq("recollect: identical canonical view", digest, lp.savings_digest_view(second))
    check.eq("--recompute: identical canonical view", digest, lp.savings_digest_view(recomputed))
    check.eq("no new evidence on recollect", 0, second.collect["data"]["new_evidence"])
    check.eq("collect does not touch the journal", bytes_before, journal_bytes(sc.data))
    for args in (("savings", "--json"), ("metrics", "status", "--json")):
        done = sc.product.run(*args)
        check.eq(f"read {' '.join(args)} exits 0", 0, done.returncode)
    reads_unchanged = bytes_before == journal_bytes(sc.data)
    check.ok("reads create no lookup (journal bytes unchanged)", reads_unchanged)
    dest = sc.dir / "replay"
    lp.copy_accounting(sc.data, dest, journal_only=True)
    replay = lp.collect(
        lp.Product({**sc.product.env, "SSGREP_DATA_DIR": str(dest)}, sc.product.cwd)
    )
    fresh = lp.savings_digest_view(replay)
    check.eq("fresh journal_only replay: estimates", digest["estimates"], fresh["estimates"])
    check.eq("fresh journal_only replay: measured", digest["measured"], fresh["measured"])
    left = {k: v for k, v in digest["coverage"].items() if k != "index"}
    check.eq(
        "fresh journal_only replay: coverage (index availability aside)",
        left,
        {k: v for k, v in fresh["coverage"].items() if k != "index"},
    )
    agree(check, "recollected snapshot", sc, production(), second)
    agree(check, "replayed snapshot", sc, production(), replay)
    sc.lookup("verifier", "search", QUERY, "--json")
    check.ok(
        "positive twin: a real lookup changes the journal", journal_bytes(sc.data) != bytes_before
    )
    exp = {"digest_equal": True, "journal_bytes_unchanged_by_reads": True}
    act = {
        "digest_equal": digest
        == lp.savings_digest_view(second)
        == lp.savings_digest_view(recomputed),
        "journal_bytes_unchanged_by_reads": reads_unchanged,
    }
    return finish(name, check, exp, act, sc.evidence(first, second, recomputed, replay))


def omp_trailer_exposure(fx: Fixture) -> CaseResult:
    """Real omp appends ``Wall time: N seconds`` to bash results; does the product still see it?"""
    name, check = "omp_trailer_exposure", Check()
    rows: dict[str, dict[str, Any]] = {}
    notes: list[str] = []
    snapshots: list[lp.Collected] = []
    scenes: list[Scene] = []
    for trailer in ("none", "omp"):
        sc = Scene(fx, f"t{trailer}", trailer=trailer)
        sc.consume(f"trl{trailer}0001", "Plan using airline history.", [*look(), ACT, Say("done")])
        col, _ = sc.arithmetic()
        snapshots.append(col)
        scenes.append(sc)
        frozen = sc.frozen()
        visible = native_text(frozen, f"trl{trailer}0001", "tool_result")[0]
        check.eq(
            f"{trailer}: trailer present as configured", trailer == "omp", "Wall time:" in visible
        )
        check.eq(
            f"{trailer}: debit = visible chars // 4",
            visible_chars(frozen) // 4,
            estimate(col)["debit"],
        )
        agree(check, f"{trailer}: arithmetic", sc, arithmetic(), col)
        check.ok(f"{trailer}: credit positive", (estimate(col)["credit"] or 0) > 0)
        rows[trailer] = {
            **estimate(col),
            "exclusions": col.savings_document["data"]["data"]["exclusions"],
        }
    check.eq(
        "credit identical with and without the trailer",
        rows["none"]["credit"],
        rows["omp"]["credit"],
    )
    notes.append(
        f"trailer none: {rows['none']}; trailer omp: {rows['omp']}. With the trailer the result is "
        "no longer a single JSON document, so exposure.interpret reports outcome=unknown, "
        "structured=False and zero cards and the reducer falls back to scanning the whole visible "
        "text (exclusion counts differ); credit and debit still equal the oracle because the "
        "identifiers are present as plain text and the trailer chars are visible context"
    )
    exp = {"credit_equal": True, "oracle_equal": True}
    act = {"credit_none": rows["none"]["credit"], "credit_omp": rows["omp"]["credit"]}
    return finish(
        name,
        check,
        exp,
        act,
        tuple(e for s, c in zip(scenes, snapshots, strict=True) for e in s.evidence(c)),
        " ".join(notes),
    )


REGISTRY: dict[str, Callable[[Fixture], CaseResult]] = {
    "empty_error_missing_lookups": empty_error_missing_lookups,
    "tiny_show_truncation": tiny_show_truncation,
    "repeated_lookups": repeated_lookups,
    "mentions_not_executions": mentions_not_executions,
    "shell_loop_two_searches": shell_loop_two_searches,
    "prior_known_and_query": prior_known_and_query,
    "relay_vs_sibling": relay_vs_sibling,
    "duplicate_mutated_partial_histories": duplicate_mutated_partial_histories,
    "journal_write_loss": journal_write_loss,
    "source_expiry": source_expiry,
    "recollection_replay_no_extra_usage": recollection_replay_no_extra_usage,
    "omp_trailer_exposure": omp_trailer_exposure,
}
