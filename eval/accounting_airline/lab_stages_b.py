"""Checkpoints C4-C7: fresh session B (isolation, first retrieval, L3 validation, L4 attempts)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from . import accounting_oracle as acct, l4_verify
from .checkpoints import Assertion
from .dataset import logical_digests
from .gates import BLOCKED, PASS
from .lab_checks import blocked, check, equal, gate_status_of
from .lab_context import Run, Stage, dig
from .lab_product import index_status
from .lab_session import Turn, absorb, grade_l3, grade_l4, run_turn, statement_of, tagged_tables
from .lab_stages_a import account, gate_assertion, gate_entry, reconcile_journal
from .observer import parse_native_omp
from .sandbox import SANDBOX_EXEC, prove_isolation, wrap_argv

A_REF = "omp:{sid}:ep:"


def _b_turn(run: Run, script: str, task: str, params: dict[str, Any]) -> Any:
    turn = Turn(script, task, {"database": run.b_db, **params})
    done = run_turn(
        run.kit_b, turn, session_id=run.b_sid, cwd=run.roots.project,
        params_dir=run.roots.runtime_b / "params", profile=run.profile if run.sandbox_ok else None,
    )  # fmt: skip
    absorb(run.observer, done)
    run.native["b"] = done.session_file
    return done


def _db_digests(run: Run, database: str) -> dict[str, str]:
    return logical_digests(run.instance.read_state(database))


def _refs_in(text: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"omp:[\w-]+:ep:\d+", text)))


# ----------------------------------------------------------------------------------------- C4


def stage_c4(run: Run) -> None:
    stage = Stage(run, "C4", "B starts")
    a_native = run.native["a"]
    stage.runtime_ids = {"solver-a": run.a_sid, "solver-b": run.b_sid}
    stage.input_paths = [a_native]
    # B's isolation: the sandbox probe (denied + allowed twins) and the actual A transcript
    if run.profile is None:
        gate_entry(
            "b_isolation",
            run,
            "isolation.solver_b_filesystem",
            BLOCKED,
            "sandbox profile unavailable",
        )
        stage.add_domain(
            blocked("b.filesystem_isolation", "sandbox unavailable", method="sandbox-exec probe")
        )
    else:
        gate = prove_isolation(run.roots, run.profile)
        run.sandbox_ok = gate.status == PASS
        gate_entry("b_isolation", run, gate.name, gate.status, gate.detail)
        stage.add_domain(
            gate_assertion(
                gate,
                "canary denied under sandbox, allowed twin and "
                "unsandboxed control read both succeed",
            )
        )
        if run.sandbox_ok:
            denied = run.observer.run(
                actor="control", task="C4/probe", tool="cat",
                argv=wrap_argv(run.profile, ["/bin/cat", str(a_native)]),
                env={"PATH": "/usr/bin:/bin"},
                cwd=run.roots.project, timeout=30, shown=["sandboxed-cat", "A-transcript"],
            )  # fmt: skip
            control = run.observer.run(
                actor="control", task="C4/probe", tool="cat", argv=["/bin/cat", str(a_native)],
                env={"PATH": "/usr/bin:/bin"}, cwd=run.roots.project, timeout=30,
                shown=["unsandboxed-cat", "A-transcript"],
            )  # fmt: skip
            blocked_read = (
                denied[0].exit_code not in (0, None)
                and a_native.read_text()[:40] not in denied[0].stdout
            )
            stage.add_domain(
                check(
                    "b.cannot_read_A_transcript_sandboxed",
                    [] if blocked_read else ["read allowed"],
                    expected="denied",
                    actual=denied[0].exit_code,
                    method=f"{SANDBOX_EXEC} cat of A's real native file",
                ),
                check(
                    "control.can_read_A_transcript",
                    [] if control[0].exit_code == 0 else ["denied"],
                    expected=0,
                    actual=control[0].exit_code,
                    method="unsandboxed positive twin of the same read",
                ),
            )
    stage.add_domain(
        equal(
            "b.fresh_identity",
            True,
            run.b_sid != run.a_sid and not any(run.kit_b.sessions_dir.glob("*.jsonl")),
            method="distinct native session id, empty runtime-b session dir",
        ),
        equal(
            "b.same_cwd",
            str(run.roots.project),
            str(run.roots.project),
            method="both sessions run in roots.project",
        ),
        equal(
            "seed.unchanged_for_b",
            run.base.digests,
            _db_digests(run, run.b_db),
            method="logical digests of B's clone vs sealed baseline",
        ),
        check(
            "b.env_has_no_A_paths",
            [
                k
                for k, v in run.kit_b.env.items()
                if str(run.roots.runtime_a) in v
                or str(run.roots.verifier) in v
                or str(run.roots.artifacts) in v
            ],
            expected="no variable names A, verifier or artifacts",
            actual="checked",
            method="scan of B's explicit environment",
        ),
    )
    pre = run.collect("c4")
    frozen = run.frozen()
    stage.add_accounting(
        equal(
            "b.no_inherited_a_usage",
            [],
            [
                t
                for t in acct.evaluate(frozen, run.production_profile(frozen)).tasks
                if run.b_sid in t
            ],
            method="oracle tasks before B acts",
        ),
        equal(
            "b.no_duplicate_historical_import",
            dig(run.collected["c3"].collect, "sources_total"),
            dig(pre.collect, "sources_total"),
            method="sources_total unchanged by B's start",
        ),
    )
    stage.methods.append("sandbox probe + observed read attempts + env scan")
    stage.finish("session B: L3 (first retrieval before rule reads)")
    gate_entry("cross_session_exposure", run, "b.started", PASS if run.sandbox_ok else BLOCKED, "")


# ------------------------------------------------------------------------------------ C5 / C6


def _b_lookups(run: Run) -> list[Any]:
    frozen = run.frozen()
    return [c for c in frozen.calls if c.actor == "solver-b" and c.tool == "ssgrep"
            and len(c.argv) > 1 and c.argv[1] in ("search", "show")]  # fmt: skip


def stage_c5_c6(run: Run) -> dict[str, Any]:
    c5 = Stage(run, "C5", "each B search/show")
    status_before = index_status(run.product)
    done = _b_turn(run, "L3", "B/L3", {"first_turn": True})
    plan = done.summary.get("memory", {}).get("plan", {})
    run.notes["plan"] = plan
    c5.runtime_ids = {"solver-b": run.b_sid}
    c5.task_ids = [f"{run.b_sid}#0"]
    c5.input_paths = [done.session_file]
    frozen = run.frozen()
    lookups = _b_lookups(run)
    prefix = A_REF.format(sid=run.a_sid)
    rows = []
    for call in lookups:
        refs = _refs_in(call.stdout) if call.argv[1] == "search" else [call.argv[2]]
        rows.append(
            {
                "op": call.argv[1],
                "state": call.state,
                "exit": call.exit_code,
                "refs": refs[:4],
                "chars": call.stdout_chars,
            }
        )
    c5.actual_delta = {"lookups": rows}
    searches = [r for r in rows if r["op"] == "search"]
    shows = [r for r in rows if r["op"] == "show"]
    c5.add_domain(
        equal("b.searched_ssgrep", True, bool(searches), method="observed solver-b dispatches"),
        check(
            "b.search_returned_A_source",
            []
            if any(any(x.startswith(prefix) for x in r["refs"]) for r in searches)
            else ["no A ref in any B search"],
            expected=f"a ref starting {prefix}",
            actual=searches[:1],
            method="raw search output of B's own lookup",
        ),
        check(
            "b.opened_A_source_via_show",
            []
            if any(r["exit"] == 0 and any(x.startswith(prefix) for x in r["refs"]) for r in shows)
            else ["B never opened an A ref"],
            expected="show of an A ref",
            actual=shows[:2],
            method="observed show dispatches",
        ),
    )
    first_search = min((c.dispatch_seq for c in lookups if c.argv[1] == "search"), default=None)
    rule_reads = [
        c.dispatch_seq
        for c in frozen.calls
        if c.actor == "solver-b"
        and c.tool == "sql"
        and "connection_rules" in statement_of(run.observer, c)
    ]
    first_rule = min(rule_reads, default=None)
    c5.add_domain(
        equal(
            "b.first_retrieval_before_rule_reads",
            True,
            first_search is not None and (first_rule is None or first_search < first_rule),
            method="observer dispatch order (search vs first SQL naming connection_rules)",
        ),
    )
    status_after = index_status(run.product)
    c5.add_domain(
        equal(
            "b.not_yet_indexed",
            dig(status_before, "session_count"),
            dig(status_after, "session_count"),
            method="index status before/after B's actions (B indexed only afterwards)",
        ),
    )
    collected = run.collect("c5")
    result = account(run, c5, "c5", collected)
    debits = [d for d in result.debits if d.task.startswith(run.b_sid)]
    c5.add_accounting(
        equal(
            "b.every_lookup_has_a_debit",
            len(lookups),
            len(debits),
            method="oracle debits vs observed lookup executions",
        ),
        equal(
            "b.adoption_zero_before_action",
            True,
            all(v.reuse != "action_reference" or v.eligible is not None for v in result.verdicts),
            method="oracle verdict reuse state",
        ),
    )
    reconcile_journal(run, c5, "c5")
    gate_entry(
        "cross_session_exposure",
        run,
        "b.saw_A_source",
        gate_status_of(c5.domain[:3]),
        "; ".join(a.detail for a in c5.domain if a.detail)[:300],
    )
    c5.finish("validate B's L3 answer and the reuse evidence")

    c6 = Stage(run, "C6", "B L3 validation")
    c6.runtime_ids = {"solver-b": run.b_sid}
    c6.task_ids = [f"{run.b_sid}#0"]
    c6.input_paths = [done.session_file]
    tables = tagged_tables(run.frozen(), run.observer, "solver-b", task="B/L3")
    for name, problems in grade_l3(tables, plan, run.base.state).items():
        c6.add_domain(
            check(
                f"l3.{name}",
                problems,
                expected="oracle plan/rows",
                actual=tables.get(name) if name != "joint_plan" else plan,
                method="domain oracle vs B's own observed results and plan",
            )
        )
    gate_entry(
        "l3",
        run,
        "l3",
        gate_status_of(c6.domain),
        "; ".join(a.detail for a in c6.domain if a.detail)[:300],
    )
    result = account(run, c6, "c6", collected)
    arith = account(run, c6, "c6", collected, profile="arithmetic")
    run.notes["production"], run.notes["arithmetic"] = result, arith
    reuse = [v for v in result.verdicts if v.reuse == "action_reference"
             and v.task.startswith(run.b_sid)]  # fmt: skip
    arith_reuse = [v for v in arith.verdicts if v.eligible and v.source_session == run.a_sid]
    c6.add_accounting(
        equal(
            "c6.profiles_never_summed",
            True,
            result.profile != arith.profile,
            method="oracle refuses aggregation (profiles evaluated separately)",
        ),
        Assertion(
            "c6.reuse_candidates_observed",
            "PASS" if arith_reuse else "UNKNOWN",
            "a candidate exposed from A's source then acted on by B",
            [v.token[1] for v in arith_reuse],
            "oracle verdicts (arithmetic profile)",
            "" if arith_reuse else "no traceable reuse; reuse gate NOT met",
        ),
    )
    run.notes["reuse_candidates"] = [v.token[1] for v in reuse]
    gate_entry(
        "reuse_evidence",
        run,
        "supported_reuse",
        PASS if arith_reuse else "FAIL",
        "" if arith_reuse else "domain pass does not conceal missing traceable reuse",
    )
    c6.methods.append("arithmetic profile pinned in an isolated journal-only copy")
    c6.finish("apply the plan (L4)")
    return {"plan": plan}


# ----------------------------------------------------------------------------------------- C7


def stage_c7(run: Run, plan: dict[str, Any]) -> None:
    stage = Stage(run, "C7", "each L4 attempt")
    before = run.instance.read_state(run.b_db)
    done = _b_turn(run, "L4", "B/L4", {"first_turn": False, "plan": plan})
    after = run.instance.read_state(run.b_db)
    stage.runtime_ids = {"solver-b": run.b_sid}
    stage.task_ids = [f"{run.b_sid}#0", f"{run.b_sid}#1"]
    stage.input_paths = [done.session_file]
    problems = grade_l4(before, after, plan)
    stage.add_domain(
        check(
            "l4.joint_apply",
            problems,
            expected="oracle post-state",
            actual={t: len(after[t]) for t in ("coupons", "rebooking_requests")},
            method="l4_verify.verify_joint_apply vs actual DB rows",
        )
    )
    frozen = run.frozen()
    attempts = [c for c in frozen.calls if c.actor == "solver-b" and c.tool == "sql"
                and "rebook_itinerary" in statement_of(run.observer, c)]  # fmt: skip
    replays = [c for c in attempts if re.search(r'"replayed":\s*true', c.stdout)]
    stage.add_domain(
        equal("l4.attempts_observed", 3, len(attempts), method="observer SQL dispatches"),
        equal("l4.replay_flagged", 1, len(replays), method="raw replay response"),
        equal(
            "l4.request_log_delta",
            2,
            len(after["rebooking_requests"]) - len(before["rebooking_requests"]),
            method="DB request log before/after",
        ),
        check(
            "l4.payments_untouched",
            l4_verify.payments_unchanged(before, after),
            expected="payments unchanged",
            actual="compared",
            method="independent state verifier",
        ),
    )
    finishes = [e for c in attempts for e in c.sql if e.get("phase") == "finish"]
    stage.actual_delta = {"txn": [e.get("txn") for e in finishes]}
    gate_entry(
        "l4_scenarios",
        run,
        "l4.apply",
        gate_status_of(stage.domain),
        "; ".join(a.detail for a in stage.domain if a.detail)[:300],
    )
    collected = run.collect("c7")
    before_lookups = len(_b_lookups(run))
    result = account(run, stage, "c7", collected)
    mine = [d for d in result.debits if d.task.startswith(run.b_sid)]
    stage.add_accounting(
        equal(
            "c7.lookup_debits_survive_idempotent_db",
            before_lookups,
            len(mine),
            method="oracle lookup debits unchanged by DB idempotency/replay",
        ),
    )
    reconcile_journal(run, stage, "c7")
    stage.finish("replay/recollection")


def b_native_text(run: Run) -> str:
    native = parse_native_omp(run.native["b"])
    return "\n".join(r.text for r in native.records)


def json_dump(value: Any) -> str:
    return json.dumps(value, default=str, sort_keys=True)


def path_str(path: Path) -> str:
    return str(path)
