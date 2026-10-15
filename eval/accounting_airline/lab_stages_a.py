"""Checkpoints C0-C3: seed/setup, session A (L1, L2), persistence barrier and restart."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import (
    accounting_oracle as acct,
    domain_lane,
    domain_oracle as oracle,
    lab_journal,
    lab_persist,
)
from .checkpoints import Assertion
from .gates import BLOCKED, PASS, Gate
from .lab_checks import blocked, check, equal, estimate_values, gate_status_of
from .lab_context import Run, Stage, dig
from .lab_product import (
    Collected,
    index,
    index_status,
    journaled_expected,
    native_source_proof,
    savings_digest_view,
)
from .lab_session import (
    Turn,
    absorb,
    grade_l1,
    grade_l2,
    run_turn,
    sql_attempts,
    tagged_tables,
)
from .observer import parse_native_omp
from .report import GateEntry


def gate_assertion(gate: Gate, method: str) -> Assertion:
    status = gate.status if gate.status in (PASS, "FAIL", BLOCKED) else "UNKNOWN"
    return Assertion(gate.name, status, "no violation", gate.detail or "no violation", method,
                     gate.detail)  # fmt: skip


def gate_entry(group: str, run: Run, name: str, status: str, detail: str = "") -> None:
    run.report.add_gate(group, GateEntry(name, status, detail))


# --------------------------------------------------------------------- shared accounting check


def account(
    run: Run, stage: Stage, label: str, collected: Collected, *, profile: str = "production"
) -> acct.OracleResult:
    """Independent oracle (observer + native evidence only) vs the REAL ``savings --json``."""
    frozen = run.frozen()
    if profile == "production":
        spec = run.production_profile(frozen)
        snapshot = collected
    else:
        spec = run.arithmetic_profile(frozen)
        snapshot, _tokens = run.collect_arithmetic(frozen)
        stage.actual_delta["arithmetic_fixture"] = {
            "methodology": snapshot.savings_document["lab_valuation_methodology"],
            "frozen_inputs": snapshot.savings_document["lab_fixture_inputs"],
        }
    result = acct.evaluate(frozen, spec)
    if not result.debits and not result.verdicts:
        # nothing was looked up in scope: the product publishes no estimate (value None, reason
        # not_estimated) rather than a zero; the oracle's rule is the same statement
        mismatches = [
            f"{unit}.{scenario}.{field}: expected unknown/not_estimated, got {row.get(field)}"
            for unit in ("context_tokens", "work_seconds")
            for rows in [(snapshot.estimates.get(unit) or {}) if snapshot.estimates else {}]
            for scenario in ("conservative", "central", "optimistic")
            for row in [rows.get(scenario) or {}]
            for field in ("credit", "debit", "net")
            if (row.get(field) or {}).get("value") is not None
            or (row.get(field) or {}).get("reason") != "not_estimated"
        ]
    else:
        mismatches = acct.compare_with_snapshot(result, snapshot.savings_document)
    run.notes[f"{'production' if profile == 'production' else 'arithmetic'}_mismatch"] = mismatches
    expected = {
        "context_tokens": {s: e.as_dict() for s, e in result.estimates.items()},
        "work_seconds": {s: e.as_dict() for s, e in result.time_estimates.items()},
        "timing": dict(result.timing),
    }
    stage.add_accounting(
        check(
            f"{label}.{result.profile}.oracle_vs_snapshot",
            mismatches,
            expected=expected,
            actual=estimate_values(snapshot.estimates),
            method=f"independent oracle {result.methodology_label} vs real `ssgrep savings --json`",
        )
    )
    stage.methods.append(f"oracle {result.profile} vs savings --json")
    return result


def reconcile_journal(run: Run, stage: Stage, label: str) -> None:
    """Observed completed lookups vs the executions the product journal actually recorded."""
    frozen = run.frozen()
    view = lab_journal.read_journal(run.roots.ssgrep_data)
    problems = lab_journal.reconcile(frozen, view)
    want = lab_journal.observed_expectation(frozen)
    stage.add_accounting(
        check(
            f"{label}.journal_matches_observed",
            problems,
            expected=want["completed"],
            actual=dict(view.completed),
            method="observer dispatch+result vs journal completed events (read-only)",
        )
    )
    stage.methods.append("observed dispatch vs journal completions")


def lookup_names(frozen: Any, actor: str) -> list[str]:
    return [
        f"{c.argv[1]}:{c.state}" for c in frozen.calls
        if c.actor == actor and c.tool == "ssgrep" and len(c.argv) > 1
    ]  # fmt: skip


# ----------------------------------------------------------------------------------------- C0


def stage_c0(run: Run, lane_note: str) -> None:
    stage = Stage(run, "C0", "seed/setup")
    inst, base = run.instance, run.base
    scratch = inst.clone("scratch")
    try:
        gate = domain_lane.gate_seed(inst, base, scratch)
    finally:
        inst.drop_clone("scratch")
    stage.add_domain(
        gate_assertion(gate, "counts, constraints, logical digests (domain_lane.gate_seed)")
    )
    stage.add_domain(
        equal("clock.frozen", "2026-11-01T07:00:00Z", __import__("eval.accounting_airline.dataset",
              fromlist=["AS_OF"]).AS_OF, method="fixture definition (application clock)")
    )  # fmt: skip
    gate_entry("seed_integrity", run, gate.name, gate.status, gate.detail)
    stage.methods.append("seed gate on the run's own baseline")
    setup = run.notes["setup"]
    stage.add_accounting(
        equal("profile.collection_enabled", True, dig(setup, "collection_enabled"),
              method="`metrics setup --yes --json`"),
        equal("export.off", [], dig(setup, "enabled_exports"), method="`metrics setup --json`"),
    )  # fmt: skip
    pre = run.notes["pre_collect_state"]
    stage.add_accounting(
        equal(
            "ledger.empty_state_honest",
            "no_snapshot",
            pre,
            method="`savings --json` before any collection",
        ),
    )
    first = run.collected["initial"]
    stage.actual_delta = {
        "snapshot_id": dig(first.collect, "snapshot_id"),
        "epoch": dig(first.collect, "epoch"),
        "generation": dig(first.collect, "generation"),
        "sources_total": dig(first.status, "sources_total"),
    }
    stage.expected_delta = {"sources_total": 0, "note": "no native session exists yet"}
    stage.snapshot_after = str(stage.actual_delta["snapshot_id"])
    stage.add_accounting(
        equal(
            "initial.no_sources",
            0,
            dig(first.status, "sources_total"),
            method="initial `metrics collect` after profile setup",
        ),
    )
    stage.methods.append(f"lane: {lane_note}")
    stage.finish("session A: L1")


# ------------------------------------------------------------------------------ C1 / C2 (A)


def _run_a(run: Run, script: str, task: str, first: bool) -> Any:
    turn = Turn(script, task, {"database": run.a_db, "first_turn": first})
    done = run_turn(
        run.kit_a, turn, session_id=run.a_sid, cwd=run.roots.project,
        params_dir=run.roots.runtime_a / "params", profile=None,
    )  # fmt: skip
    absorb(run.observer, done)
    run.native["a"] = done.session_file
    return done


def _attempts_assertion(run: Run, stage: Stage, task: str, expected_failed: int) -> None:
    frozen = run.frozen()
    rows = [r for r in sql_attempts(frozen, run.observer, "solver-a") if r["task"] == task]
    failed = [r for r in rows if r["exit"] not in (0, None)]
    stage.add_accounting(
        equal(
            f"{task}.failed_sql_attempts_recorded",
            expected_failed,
            len(failed),
            method="observer SQL start/finish with transaction outcome",
        ),
    )
    stage.actual_delta["sql_attempts"] = [
        {k: r[k] for k in ("tag", "state", "exit", "txn", "sqlstate")} for r in rows
    ]


def stage_c1(run: Run) -> None:
    stage = Stage(run, "C1", "A L1 completes")
    done = _run_a(run, "L1", "A/L1", True)
    stage.runtime_ids = {"solver-a": run.a_sid}
    stage.task_ids = [f"{run.a_sid}#0"]
    stage.input_paths = [done.session_file]
    frozen = run.frozen()
    tables = tagged_tables(frozen, run.observer, "solver-a", task="A/L1")
    graded = grade_l1(tables, run.base.state)
    naive = graded.pop("naive_cash_detected")
    for name, problems in graded.items():
        stage.add_domain(
            check(
                f"l1.{name}",
                problems,
                expected="oracle rows",
                actual=tables.get(name),
                method="domain oracle vs ACTUAL rows",
            )
        )
    stage.add_domain(
        Assertion.negative(
            "l1.naive_fanout_join_detected",
            detected=bool(naive),
            expected="violations",
            actual=naive[:2],
            method="oracle rejects the solver's own fan-out attempt",
        ),
        equal(
            "a.process_exit",
            0,
            done.returncode,
            method="observed stand-in process exit",
            detail=str(done.summary.get("stderr_tail", "")),
        ),
    )
    gate_entry(
        "l1",
        run,
        "l1",
        gate_status_of(stage.domain),
        "; ".join(a.detail for a in stage.domain if a.detail)[:300],
    )
    stage.methods.append("domain oracle vs solver's own observed SQL results")
    _attempts_assertion(run, stage, "A/L1", expected_failed=1)
    collected = run.collect("c1")
    stage.snapshot_after = str(dig(collected.collect, "snapshot_id"))
    result = account(run, stage, "c1", collected)
    stage.add_accounting(
        check(
            "c1.no_reuse_credit_fabricated",
            [str(v.as_dict()) for v in result.verdicts if v.eligible],
            expected="no eligible reused candidate",
            actual=len(result.verdicts),
            method="oracle verdicts (A never searched ssgrep)",
        ),
    )
    reconcile_journal(run, stage, "c1")
    stage.finish("session A: L2")


def stage_c2(run: Run) -> None:
    stage = Stage(run, "C2", "A L2 completes")
    done = _run_a(run, "L2", "A/L2", False)
    stage.runtime_ids = {"solver-a": run.a_sid}
    stage.task_ids = [f"{run.a_sid}#0", f"{run.a_sid}#1"]
    stage.input_paths = [done.session_file]
    frozen = run.frozen()
    tables = tagged_tables(frozen, run.observer, "solver-a", task="A/L2")
    for name, problems in grade_l2(tables, run.base.state).items():
        stage.add_domain(
            check(
                f"l2.{name}",
                problems,
                expected="oracle rows",
                actual=tables.get(name),
                method="domain oracle vs ACTUAL rows",
            )
        )
    gate_entry(
        "l2",
        run,
        "l2",
        gate_status_of(stage.domain),
        "; ".join(a.detail for a in stage.domain if a.detail)[:300],
    )
    collected = run.collect("c2")
    result = account(run, stage, "c2", collected)
    a_tasks = [t for t in result.tasks if t.startswith(run.a_sid)]
    stage.add_accounting(
        equal(
            "a.two_distinct_tasks_one_root",
            [f"{run.a_sid}#0", f"{run.a_sid}#1"],
            a_tasks,
            method="oracle task ordinals from native user turns",
        ),
    )
    reconcile_journal(run, stage, "c2")
    stage.methods.append("task ordinals from native user turns")
    stage.finish("persist + index A, resolve both tasks as the verifier")


# ----------------------------------------------------------------------------------------- C3

L1_QUERY = "settled cash per booking by currency and sellable economy seats per flight"
L2_QUERY = "travellers needing reaccommodation and minimum connection rules per airport"


def blocked_persistence(run: Run, stage: Stage, reason: str) -> None:
    for group in ("a_native_persistence", "restart_retrieval"):
        gate_entry(group, run, f"{group}.model_cache", BLOCKED, reason)
    stage.add_domain(blocked("a.sources_resolve", reason, method="real ssgrep index + search/show"))
    stage.add_accounting(blocked("a.visibility_latency", reason, method="real ssgrep index"))
    stage.coverage_missing.append("embedding model not cached locally; nothing was downloaded")


def stage_c3(run: Run) -> dict[str, Any]:
    stage = Stage(run, "C3", "A persisted/indexed/restarted")
    stage.runtime_ids = {"solver-a": run.a_sid}
    stage.task_ids = [f"{run.a_sid}#0", f"{run.a_sid}#1"]
    native = parse_native_omp(run.native["a"])
    stage.input_paths = [run.native["a"]]
    if run.cache_blocked_reason:
        blocked_persistence(run, stage, run.cache_blocked_reason)
        stage.finish("persistence BLOCKED; session B not started")
        return {"ok": False}
    stage.add_domain(
        equal(
            "a.partial_tail_bytes",
            0,
            native.partial_tail_bytes,
            method="parse of the runtime file: newline-complete through the last turn",
        ),
        equal(
            "a.final_message_present",
            True,
            any(r.kind == "usage" for r in native.records) and native.malformed_lines == 0,
            method="native records parsed, no malformed lines",
        ),
    )
    native_sources = [
        str(path.relative_to(run.roots.native_sessions))
        for directory in run.native_dirs().values()
        for path in sorted(directory.rglob("*.jsonl"))
    ]
    code, index_ms, tail = index(run.product)
    stage.add_domain(
        equal("index.completed", 0, code, method="real `ssgrep index --quiet`", detail=tail)
    )
    status = index_status(run.product)
    run.notes["index_a"] = {"ms": index_ms, "status": status, "native_sources": native_sources}
    searched: dict[str, Any] = {}
    problems_all: list[str] = []
    for task, query, expected in (
        ("A/L1", L1_QUERY, lab_persist.l1_evidence(run.base.state)),
        ("A/L2", L2_QUERY, lab_persist.l2_evidence(run.base.state)),
    ):
        res, _s, opened = lab_persist.resolve_task(
            run.observer,
            run.verify_product(),
            run.roots,
            task=task,
            query=query,
            session=native,
            expected_text=expected,
            project=run.roots.project,
        )
        searched[task] = {
            "ref": res.ref,
            "problems": res.problems,
            "evidence": res.evidence,
            "opened": len(opened),
        }
        problems_all += [f"{task}: {p}" for p in res.problems]
        stage.add_domain(
            check(
                f"a.{task}.resolves_to_native_source",
                res.problems,
                expected="ref of A's session with verified facts",
                actual={
                    "ref": res.ref,
                    **{k: v for k, v in res.evidence.items() if k in ("source_digest", "identity")},
                },
                method="verifier search -> show vs oracle evidence + file digest",
            )
        )
        proof = (
            native_source_proof(
                run.product,
                res.ref,
                outside_root=run.roots.marker,
            )
            if res.ref is not None
            else {"status": "source_unresolved"}
        )
        expected_source = {
            "status": "ok",
            "session": ["omp", native.session_id],
            "path": str(run.native["a"]),
            "source_digest": native.file_digest,
            "source_bytes": native.complete_bytes,
            "outside_root_status": "source_unapproved",
        }
        source_problems = [
            f"{key}: expected {value!r}, got {proof.get(key)!r}"
            for key, value in expected_source.items()
            if proof.get(key) != value
        ]
        if not proof.get("record_digest_count") or not proof.get("native_identities"):
            source_problems.append("original native identities/digests are unavailable")
        stage.add_accounting(
            check(
                f"a.{task}.production_native_source_authority",
                source_problems,
                expected=expected_source,
                actual=proof,
                method="unchanged production native_source_adapter: actual index locator "
                "-> original "
                "native ID/scope/digests; same-root outside-source negative, independent of rarity",
            )
        )
    gate_entry(
        "a_native_persistence",
        run,
        "a.persisted_indexed",
        gate_status_of(stage.domain),
        "; ".join(problems_all)[:300],
    )
    # restart: every CLI call is a fresh process; repeat both resolutions and compare identity
    again: list[str] = []
    for task, query, expected in (
        ("A/L1", L1_QUERY, lab_persist.l1_evidence(run.base.state)),
        ("A/L2", L2_QUERY, lab_persist.l2_evidence(run.base.state)),
    ):
        res, _s, _o = lab_persist.resolve_task(
            run.observer,
            run.verify_product(),
            run.roots,
            task=f"{task}/restart",
            query=query,
            session=native,
            expected_text=expected,
            project=run.roots.project,
        )
        again += [f"{task}: {p}" for p in res.problems]
        if res.ref != searched[task]["ref"] or res.evidence.get("source_digest") != searched[task][
            "evidence"
        ].get("source_digest"):
            again.append(f"{task}: ref/digest changed across fresh processes")
    stage.add_domain(
        check(
            "a.restart_resolution_identical",
            again,
            expected="same ref+digest",
            actual="fresh CLI process",
            method="repeat search+show in new process",
        )
    )
    gate_entry(
        "restart_retrieval",
        run,
        "restart_retrieval",
        gate_status_of(stage.domain[-1:]),
        "; ".join(again)[:300],
    )
    frozen = run.frozen()
    verifier_calls = [c for c in frozen.calls if c.actor == "verifier" and c.tool == "ssgrep"]
    stage.add_accounting(
        equal(
            "verifier.lookups_separate_actor",
            True,
            len(verifier_calls) >= 8 and all(c.actor == "verifier" for c in verifier_calls),
            method="observer actor field on every verifier lookup",
        ),
    )
    collected = run.collect("c3")
    result = account(run, stage, "c3", collected)
    stage.add_accounting(
        check(
            "verifier.lookups_never_solver_exposure",
            [v.task for v in result.verdicts if not v.task.startswith(run.a_sid)]
            + [f"{d.task}" for d in result.debits if not d.task.startswith(run.a_sid)],
            expected="no verdict/debit outside A tasks",
            actual=len(result.debits),
            method="oracle verdicts and task debits",
        ),
        equal(
            "verifier.recorded_as_global_executions",
            True,
            result.counts.get("verifier_search", 0) >= 2,
            method="oracle counts keep verifier lookups visible",
        ),
    )
    reconcile_journal(run, stage, "c3")
    stage.actual_delta = {
        "index_ms": index_ms,
        "index_status_sessions": dig(status, "session_count"),
        "chunks": dig(status, "chunk_count"),
        "native_sources": native_sources,
    }
    stage.expected_delta = {"sessions_visible": 1}
    stage.methods.append("capture->parse->index->search visibility latency recorded as metrics")
    stage.snapshot_after = str(dig(collected.collect, "snapshot_id"))
    stage.finish("session B: fresh process in the same project")
    run.notes["resolution_a"] = searched
    return {"ok": True, "native": native, "resolution": searched}


def savings_view(collected: Collected) -> Mapping[str, Any]:
    return savings_digest_view(collected)


def expected_lookups(frozen: Any) -> Mapping[str, int]:
    return journaled_expected(frozen)


def now_ms() -> int:
    return int(time.monotonic() * 1000)


def short(path: Path, root: Path) -> str:
    return str(path.relative_to(root)) if root in path.parents else path.name


def as_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


__all__ = ["blocked", "oracle", "Sequence"]
