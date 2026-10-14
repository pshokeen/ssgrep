"""Grade the native agents' observed SQL, never execute reference answers for them."""

from __future__ import annotations

import re
from typing import Any

from .checkpoints import Assertion
from .gates import FAIL, PASS
from .lab_checks import check, equal, gate_status_of
from .lab_context import Run, Stage
from .lab_session import (
    grade_l1,
    grade_l2,
    grade_l3,
    grade_l4,
    sql_attempts,
    statement_of,
    tagged_tables,
)
from .lab_stages_a import account, gate_entry, reconcile_journal
from .live_bridge import Bridge
from .live_grading import normalize_tables
from .live_native import native_turn


def _rows(run: Run, actor: str, task: str) -> dict[str, list[dict[str, str]]]:
    return normalize_tables(tagged_tables(run.frozen(), run.observer, actor, task=task))


def _checks(stage: Stage, task: str, grades: dict[str, list[str]], tables: Any) -> None:
    for name, problems in grades.items():
        stage.add_domain(
            check(
                f"{task}.{name}",
                problems,
                expected="independent oracle",
                actual=tables,
                method="native solver's observed SQL result rows",
            )
        )


def stage_a(run: Run, bridge: Bridge, task: str, model: str) -> None:
    stage = Stage(run, "C1" if task == "L1" else "C2", f"native A {task}")
    done = native_turn(run, bridge, task, model)
    stage.runtime_ids = {"solver-a": run.a_sid}
    stage.task_ids = [f"{run.a_sid}#{0 if task == 'L1' else 1}"]
    stage.input_paths = [run.native["a"]]
    tables = _rows(run, "solver-a", f"A/{task}")
    try:
        grades = (
            grade_l1(tables, run.base.state) if task == "L1" else grade_l2(tables, run.base.state)
        )
        if task == "L1":
            naive = grades.pop("naive_cash_detected")
            stage.add_domain(
                equal(
                    "l1.actual_naive_join_rejected",
                    True,
                    bool(tables.get("cash_naive")) and bool(naive),
                    method="independent oracle rejects observed nonempty naive result",
                )
            )
        _checks(stage, task.lower(), grades, tables)
    except (ValueError, KeyError, TypeError) as exc:
        stage.add_domain(
            check(
                f"{task}.result_schema",
                [str(exc)],
                expected="declared columns",
                actual=tables,
                method="observed rows schema validation",
            )
        )
    stage.add_domain(
        equal(f"{task}.process_exit", 0, done.returncode, method="actual native runtime process")
    )
    stage.actual_delta = {
        "sql_attempts": sql_attempts(run.frozen(), run.observer, "solver-a"),
        "native": done.native.to_dict(),
    }
    gate_entry(
        task.lower(),
        run,
        f"native.{task}",
        gate_status_of(stage.domain),
        "; ".join(a.detail for a in stage.domain if a.detail),
    )
    collected = run.collect(stage.cid.lower())
    account(run, stage, stage.cid.lower(), collected)
    reconcile_journal(run, stage, stage.cid.lower())
    stage.add_accounting(
        equal(
            f"{task}.native_user_turns",
            1 if task == "L1" else 2,
            done.native.user_turns,
            method="native conversation user-turn records",
        )
    )
    stage.finish("next native turn or A persistence barrier")


def stage_b_l3(run: Run, bridge: Bridge, model: str) -> dict[str, list[str]]:
    c5 = Stage(run, "C5", "native B retrieval")
    done = native_turn(run, bridge, "L3", model)
    c5.runtime_ids = {"solver-b": run.b_sid}
    c5.task_ids = [f"{run.b_sid}#0"]
    c5.input_paths = [run.native["b"]]
    calls = [c for c in run.frozen().calls if c.actor == "solver-b"]
    lookups = [c for c in calls if c.tool == "ssgrep"]
    searches = [c for c in lookups if len(c.argv) > 1 and c.argv[1] == "search"]
    shows = [c for c in lookups if len(c.argv) > 2 and c.argv[1] == "show"]
    prefix = f"omp:{run.a_sid}:ep:"
    opened = any(c.exit_code == 0 and c.argv[2].startswith(prefix) for c in shows)
    first_search = min((c.dispatch_seq for c in searches), default=None)
    first_rules = min(
        (
            c.dispatch_seq
            for c in calls
            if c.tool == "sql" and "connection_rules" in statement_of(run.observer, c)
        ),
        default=None,
    )
    c5.add_domain(
        equal(
            "b.search_exposes_A",
            True,
            any(prefix in c.stdout for c in searches),
            method="real B search output",
        ),
        equal("b.show_opens_A", True, opened, method="real B show dispatch and return"),
        equal(
            "b.retrieval_precedes_rule_read",
            True,
            first_search is not None and (first_rules is None or first_search < first_rules),
            method="independent observer dispatch order",
        ),
        equal(
            "b.distinct_native_identity",
            True,
            run.a_sid != run.b_sid,
            method="native session headers",
        ),
    )
    c5.actual_delta = {
        "lookups": [
            {"argv": list(c.argv), "exit": c.exit_code, "chars": c.stdout_chars} for c in lookups
        ]
    }
    collected = run.collect("c5")
    account(run, c5, "c5", collected)
    reconcile_journal(run, c5, "c5")
    gate_entry("cross_session_exposure", run, "native.B.sees_A", gate_status_of(c5.domain))
    c5.finish("independently grade B's L3 SQL")
    c6 = Stage(run, "C6", "native B L3 validation")
    tables = _rows(run, "solver-b", "B/L3")
    plan = {
        row["booking_id"]: row["legs"].strip("{}").split(",")
        for row in tables.get("plan", [])
        if "booking_id" in row and "legs" in row
    }
    run.notes["plan"] = plan
    try:
        _checks(c6, "l3", grade_l3(tables, plan, run.base.state), tables)
    except (ValueError, KeyError, TypeError) as exc:
        _checks(c6, "l3", {"result_schema": [str(exc)]}, tables)
    c6.add_domain(equal("l3.native_exit", 0, done.returncode, method="native runtime exit"))
    gate_entry("l3", run, "native.L3", gate_status_of(c6.domain))
    production = account(run, c6, "c6", collected)
    arithmetic = account(run, c6, "c6", collected, profile="arithmetic")
    run.notes["production"], run.notes["arithmetic"] = production, arithmetic
    reuse = [
        v
        for v in arithmetic.verdicts
        if v.eligible and v.source_session == run.a_sid and v.task.startswith(run.b_sid)
    ]
    c6.add_accounting(
        Assertion(
            "live.reuse",
            PASS if reuse else "UNKNOWN",
            "traceable A exposure followed by supported B action",
            [v.token[1] for v in reuse],
            "independent native evidence oracle",
            "" if reuse else "no supported reuse observed",
        )
    )
    gate_entry(
        "reuse_evidence",
        run,
        "native.reuse",
        PASS if reuse else FAIL,
        "" if reuse else "Domain correctness never substitutes for traceable reuse",
    )
    run.notes["reuse_candidates"] = [v.token[1] for v in reuse]
    c6.finish("native B L4 in the same conversation")
    return plan


def stage_b_l4(run: Run, bridge: Bridge, model: str, plan: dict[str, list[str]]) -> None:
    stage = Stage(run, "C7", "native B L4 attempts")
    before = run.instance.read_state(run.b_db)
    done = native_turn(run, bridge, "L4", model)
    after = run.instance.read_state(run.b_db)
    try:
        problems = grade_l4(before, after, plan)
    except (ValueError, KeyError, TypeError) as exc:
        problems = [str(exc)]
    stage.add_domain(
        check(
            "l4.native_joint_apply",
            problems,
            expected="independent post-state",
            actual=after,
            method="independent verifier queries actual business rows",
        ),
        equal("l4.native_exit", 0, done.returncode, method="native runtime exit"),
    )
    calls = [c for c in run.frozen().calls if c.actor == "solver-b" and c.task == "B/L4"]
    replayed = any(re.search(r'"replayed":\s*true', c.stdout) for c in calls)
    stage.add_domain(equal("l4.replay_observed", True, replayed, method="actual SQL responses"))
    stage.runtime_ids = {"solver-b": run.b_sid}
    stage.task_ids = [f"{run.b_sid}#1"]
    stage.input_paths = [run.native["b"]]
    stage.actual_delta = {
        "native": done.native.to_dict(),
        "sql_attempts": sql_attempts(run.frozen(), run.observer, "solver-b"),
    }
    gate_entry("l4_scenarios", run, "native.L4", gate_status_of(stage.domain), "; ".join(problems))
    collected = run.collect("c7")
    account(run, stage, "c7", collected)
    reconcile_journal(run, stage, "c7")
    stage.add_accounting(
        equal("b.two_turns_one_root", 2, done.native.user_turns, method="native user turn count")
    )
    stage.finish("C8-C10 real replay/scrape/export/closure")
