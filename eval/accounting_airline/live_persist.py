"""Live C3: genuine source resolution with semantic, not Markdown-byte, evidence checks."""

from __future__ import annotations

from . import domain_oracle, lab_persist
from .gates import PASS
from .lab_checks import check, equal, gate_status_of
from .lab_context import Run, Stage, dig
from .lab_product import index, index_status
from .lab_session import grade_l2, tagged_tables
from .lab_stages_a import L1_QUERY, L2_QUERY, account, gate_entry, reconcile_journal
from .live_grading import missing_facts, normalize_tables
from .observer import parse_native_omp


def _resolve(run: Run, native, task: str, query: str, expected: list[str]) -> dict:
    searched = lab_persist.verifier_lookup(
        run.observer, run.product, run.roots, "search", query, task=task
    )
    cards = {card["ref"]: card for card in lab_persist.data_of(searched).get("results", [])}
    prefix = f"omp:{native.session_id}:ep:"
    refs = [ref for ref in lab_persist.refs_of(searched) if ref.startswith(prefix)]
    attempts = []
    for ref in refs[:4]:
        shown = lab_persist.verifier_lookup(
            run.observer, run.product, run.roots, "show", ref, task=task
        )
        detail = lab_persist.data_of(shown)
        problems, evidence = lab_persist._judge_shown(
            detail, cards[ref], native, run.roots.project, ()
        )
        text = f"{detail.get('prompt_text', '')}\n{detail.get('response_text', '')}"
        absent = missing_facts(text, expected)
        if absent:
            problems.append(f"shown source lacks verified facts: {absent}")
        if searched.returncode or shown.returncode:
            problems.append("real search/show exited nonzero")
        attempt = {"ref": ref, "problems": problems, "evidence": evidence}
        attempts.append(attempt)
        if not problems:
            return {**attempt, "opened": len(attempts), "attempts": attempts}
    return {
        "ref": None,
        "problems": ["no returned native episode contained the verified task facts"],
        "opened": len(attempts),
        "attempts": attempts,
    }


def stage_c3(run: Run) -> dict:
    stage = Stage(run, "C3", "native A persistence and restarted retrieval")
    native = parse_native_omp(run.native["a"])
    stage.runtime_ids = {"solver-a": run.a_sid}
    stage.task_ids = [f"{run.a_sid}#0", f"{run.a_sid}#1"]
    stage.input_paths = [run.native["a"]]
    cash = normalize_tables(tagged_tables(run.frozen(), run.observer, "solver-a", task="A/L1"))
    l2 = normalize_tables(tagged_tables(run.frozen(), run.observer, "solver-a", task="A/L2"))
    cash_errors = domain_oracle.verify_cash(cash.get("cash", []), run.base.state)
    rule_errors = [error for errors in grade_l2(l2, run.base.state).values() for error in errors]
    stage.add_domain(
        check(
            "a.verified_cash_source_facts",
            cash_errors,
            expected="oracle cash rows",
            actual=cash.get("cash"),
            method="independent SQL domain oracle",
        ),
        check(
            "a.verified_rule_source_facts",
            rule_errors,
            expected="oracle L2 rules",
            actual=l2,
            method="independent SQL domain oracle",
        ),
        equal(
            "a.native_complete",
            0,
            native.partial_tail_bytes,
            method="newline-complete native persistence",
        ),
    )
    # L1 fixture-contract failures remain visible, including a raw_free mismatch.
    # This transport gate relies only on independently verified cash and L2 rules.
    expected_cash = [
        f"{booking} {currency} {amount}"
        for (booking, currency), amount in sorted(
            domain_oracle.settled_cash(run.base.state).items()
        )
    ]
    run.collect("pre-c3")  # Refresh the private source view before indexing.
    code, elapsed, detail = index(run.product)
    stage.add_domain(
        equal("a.real_index", 0, code, method="real cached offline index", detail=detail)
    )
    run.notes["index_a"] = {"ms": elapsed, "status": index_status(run.product)}
    first, restarted = {}, {}
    for task, query, expected in (
        ("A/L1", L1_QUERY, expected_cash),
        ("A/L2", L2_QUERY, lab_persist.l2_evidence(run.base.state)),
    ):
        first[task] = _resolve(run, native, task, query, expected)
        restarted[task] = _resolve(run, native, task + "/restart", query, expected)
        stage.add_domain(
            check(
                f"{task}.source_resolves",
                first[task]["problems"],
                expected="returned ref, matching digest, independently verified facts",
                actual=first[task],
                method="real verifier search and bounded show",
            ),
            check(
                f"{task}.restart_resolves",
                restarted[task]["problems"],
                expected="same native evidence after new CLI processes",
                actual=restarted[task],
                method="fresh search/show processes",
            ),
            equal(
                f"{task}.restart_identity",
                first[task].get("ref"),
                restarted[task].get("ref"),
                method="actual returned ref comparison",
            ),
        )
    run.notes["resolution_a"] = first
    status = gate_status_of(stage.domain)
    for group in ("a_native_persistence", "restart_retrieval"):
        gate_entry(
            group,
            run,
            "native.semantic_source_barrier",
            status,
            "; ".join(a.detail for a in stage.domain if a.detail),
        )
    collected = run.collect("c3")
    stage.snapshot_after = str(dig(collected.collect, "snapshot_id"))
    account(run, stage, "c3", collected)
    reconcile_journal(run, stage, "c3")
    stage.finish("B starts only after this real persistence/source barrier passes")
    return {"ok": status == PASS, "native": native, "resolution": first}
