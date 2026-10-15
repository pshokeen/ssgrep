"""The scripted-standin pipeline: deterministic lane + sessions A then B + checkpoints C0-C10.

Runtime label is ``scripted-standin`` in every artifact; the result can never count toward the
live gate. Real parts: PostgreSQL in docker, restricted roles, ``ssgrep`` binaries, the real
embedding model (only when already cached; nothing is downloaded), the real index, sandbox-exec
for B. Teardown is ownership-checked.
"""

from __future__ import annotations

import secrets
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import accounting_oracle as acct, domain_lane, lab_metrics, lab_persist, prompts
from .checkpoints import Assertion
from .database import (
    READER,
    WRITER,
    LabInstance,
    list_containers,
    preflight as db_preflight,
    removal_mark,
    removals_since,
)
from .gates import BLOCKED, FAIL, NOT_RUN, PASS, Gate
from .lab_checks import blocked, check, equal, gate_status_of
from .lab_context import Run, Stage, dig
from .lab_product import Product, collect, copy_accounting, real_model_cache, setup_profile
from .lab_session import install_tools, make_kit
from .lab_stages_a import (
    gate_assertion,
    gate_entry,
    reconcile_journal,
    stage_c0,
    stage_c1,
    stage_c2,
    stage_c3,
)
from .lab_stages_b import stage_c4, stage_c5_c6, stage_c7
from .manifest import STANDIN_LABEL, _gate_model_cache, build_manifest
from .observer import Observer, parse_native_omp
from .report import GateEntry, Report, canary_scan, gate_table
from .roots import LabRoots, runtime_env
from .sandbox import availability, profile_for_solver_b

LANE_GROUPS = (
    ("seed.", "seed_integrity"),
    ("oracle.", "seed_integrity"),
    ("sensitivity.domain", "seed_integrity"),
    ("isolation.database_roles", "seed_integrity"), ("l1.", "l1"), ("l2.", "l2"), ("l3.", "l3"),
    ("l4.", "l4_scenarios"), ("sensitivity.l4", "l4_scenarios"), ("cleanup.", "cleanup"),
)  # fmt: skip


def _group(name: str) -> str | None:
    return next((g for prefix, g in LANE_GROUPS if name.startswith(prefix)), None)


def preflight_standin(model_cache: Path) -> tuple[list[Gate], str | None]:
    """Hard prerequisites (docker, psql) and the soft one (model cache -> persistence BLOCKED)."""
    gates = [db_preflight()]
    gates.append(
        Gate(
            "psql_client",
            PASS if shutil.which("psql") else BLOCKED,
            "" if shutil.which("psql") else "host psql client not found",
        )
    )
    cache = _gate_model_cache(model_cache)
    return gates, (None if cache.status == PASS else cache.detail)


def _lane(run_dir: Path, report: Report, emit: Callable[[str], None]) -> None:
    root, rid = domain_lane.make_run_root(run_dir, secrets.token_hex(6))
    lane = domain_lane.run_lane(root, rid, None, report_name="domain-lane.json")
    for gate in lane.gates:
        group = _group(gate.name)
        if group is not None:
            report.add_gate(group, GateEntry(f"lane:{gate.name}", gate.status, gate.detail))
    emit(f"[deterministic lane] gates={len(lane.gates)} status={lane.domain_status}")


def build_run(roots: LabRoots, emit: Callable[[str], None], model_cache: Path) -> Run:
    gates, blocked_reason = preflight_standin(model_cache)
    for gate in gates:
        if gate.status != PASS:
            raise PrerequisiteBlocked(gate)
    instance = LabInstance.start(roots.run_id.replace("-", "")[:12])
    try:
        base = instance.provision_baseline(
            __import__("eval.accounting_airline.faults", fromlist=["x"]).baseline_sql("correct")
        )
        a_db, b_db = instance.clone("a"), instance.clone("b")
        report = Report(roots.run_id, "standin", STANDIN_LABEL, False, STANDIN_LABEL)
        passwords = {"reader": instance._passwords[READER], "writer": instance._passwords[WRITER]}
        observer = Observer(roots.verifier, roots.run_id)
        shim = install_tools(roots)
        cache = None if blocked_reason else model_cache
        env = runtime_env(roots, "control", runtime_sessions=roots.native_sessions,
                          model_cache=model_cache if cache else None)  # fmt: skip
        product = Product(env, roots.project)
        kits = {}
        for actor, home, label in (("solver-a", roots.runtime_a, "a"),
                                   ("solver-b", roots.runtime_b, "b")):  # fmt: skip
            kits[actor] = make_kit(
                roots,
                actor,
                run_id=roots.run_id,
                sessions_dir=roots.native_sessions / label,
                model_cache=model_cache,
                sql_port=instance.host_port,
                passwords=passwords,
                shim_dir=shim,
                spool_dir=home / "spool",
            )
        sandbox_gap = availability()
        profile = None if sandbox_gap else profile_for_solver_b(roots)
        manifest = build_manifest(
            run_id=roots.run_id, mode="standin", runtime=STANDIN_LABEL, roots=roots,
            prompts=prompts.PROMPTS,
            limitations=[*(profile.limitations if profile else ()),
                         *([sandbox_gap] if sandbox_gap else [])],
        )  # fmt: skip
        manifest.write(roots.artifacts / "manifest.json")
        return Run(
            roots,
            observer,
            report,
            manifest,
            instance,
            base,
            cache,
            blocked_reason,
            [*passwords.values()],
            emit,
            product,
            kits["solver-a"],
            kits["solver-b"],
            a_db,
            b_db,
            "sess-a",
            "sess-b",
            profile=profile,
        )
    except BaseException:
        instance.teardown()
        LabInstance.remove_run(instance.run_id)
        raise


class PrerequisiteBlocked(RuntimeError):
    def __init__(self, gate: Gate) -> None:
        super().__init__(f"{gate.name}: {gate.detail}")
        self.gate = gate


# ------------------------------------------------------------------------------ C8 / C9 / C10


def _mutation_snapshot(run: Run) -> dict[str, Any]:
    from .dataset import logical_digests

    return {
        "a_db": logical_digests(run.instance.read_state(run.a_db)),
        "b_db": logical_digests(run.instance.read_state(run.b_db)),
        "sources": {k: v.stat().st_size for k, v in run.native.items()},
    }


def stage_c8(run: Run) -> None:
    stage = Stage(run, "C8", "replay/recollection")
    before = _mutation_snapshot(run)
    frozen = run.frozen()
    final = run.collect("c8")
    again = run.collect("c8-again")  # duplicate collect: must add no usage
    fresh_dir = run.roots.artifacts / "full-recompute"
    copy_accounting(run.roots.ssgrep_data, fresh_dir, journal_only=True)
    fresh = collect(Product({**run.product.env, "SSGREP_DATA_DIR": str(fresh_dir)},
                            run.product.cwd, run.product.binary))  # fmt: skip
    view = lambda c: {"estimates": c.estimates, "measured": c.measured}  # noqa: E731
    stage.add_accounting(
        equal(
            "c8.duplicate_collect_adds_no_usage",
            view(final),
            view(again),
            method="savings estimates+measured after a second `metrics collect`",
        ),
        equal(
            "c8.full_vs_incremental_estimates",
            view(final),
            view(fresh),
            method="journal-only fresh full collect vs incremental state (canonical view)",
        ),
        equal(
            "c8.epoch_preserved",
            dig(final.collect, "epoch"),
            dig(again.collect, "epoch"),
            method="counter epoch across repeated collection",
        ),
    )
    result = acct.evaluate(frozen, run.production_profile(frozen))
    run.notes["oracle_counts"] = dict(result.counts)
    stage.add_accounting(
        check(
            "c8.oracle_equals_recollected",
            acct.compare_with_snapshot(result, again.savings_document),
            expected="oracle estimates",
            actual="recollected snapshot",
            method="independent oracle vs recollected `savings --json`",
        ),
    )
    after = _mutation_snapshot(run)
    stage.add_domain(
        equal(
            "c8.business_state_unchanged",
            before,
            after,
            method="logical DB digests of both clones + native source sizes around collection",
        ),
    )
    gate_entry(
        "deterministic_replay",
        run,
        "replay",
        gate_status_of(stage.accounting),
        "; ".join(a.detail for a in stage.accounting if a.detail)[:300],
    )
    stage.finish("scrape/export")


def stage_c9(run: Run) -> None:
    from .lab_c9 import run_c9, source_digests

    stage = Stage(run, "C9", "scrape/export")
    source_digests_before = source_digests(run.native_dirs())
    before = _mutation_snapshot(run)
    previous = run.collected.get("c8-again") or run.collected.get("c8")
    if previous is None:
        raise RuntimeError("C9 requires the completed C8 snapshot")

    canary = f"c9-private-canary-{secrets.token_hex(8)}"

    def collect_latest() -> Any:
        query = f"airline reservation connection capacity {canary}"
        argv = [str(run.product.binary), "search", query, "--json"]
        shown = ["ssgrep", "search", query, "--json"]
        call, process = run.observer.run(
            actor="control",
            task="C9-scrape-export",
            tool="ssgrep",
            argv=argv,
            env=run.product.env,
            cwd=run.roots.project,
            timeout=120,
            shown=shown,
        )
        if call.state != "verified" or process is None or process.returncode != 0:
            raise RuntimeError("C9 control search did not complete")
        return run.collect("c9")

    outcome = run_c9(run.product, previous, collect_latest, secrets=tuple((*run.secrets, canary)))
    frozen = run.frozen()
    oracle = acct.evaluate(frozen, run.production_profile(frozen))
    oracle_problems = acct.compare_with_snapshot(oracle, outcome.collected.savings_document)
    stage.snapshot_before = outcome.snapshot_before_id
    stage.snapshot_after = outcome.snapshot_id
    stage.actual_delta["c9_transport"] = {
        **outcome.evidence,
        "scrape_gate": outcome.scrape.status,
        "export_gate": outcome.export.status,
    }
    stage.add_accounting(
        check(
            "c9.snapshot_matches_observer_journal",
            oracle_problems,
            expected="independent observer/journal oracle",
            actual="latest production-policy snapshot",
            method=(
                "independent observer counts and arithmetic versus the real "
                "collected savings document"
            ),
        ),
    )
    for name, gate in (
        ("c9.openmetrics_scrape", outcome.scrape),
        ("c9.export_delivery", outcome.export),
    ):
        if gate.status == BLOCKED:
            assertion = blocked(name, gate.detail, method="real C9 HTTP/OTLP acceptance")
        else:
            assertion = check(
                name,
                [] if gate.status == PASS else [gate.detail or gate.status],
                expected=PASS,
                actual=gate.status,
                method="real C9 HTTP/OTLP acceptance",
            )
        stage.add_accounting(assertion)
    after = _mutation_snapshot(run)
    source_digests_after = source_digests(run.native_dirs())
    stage.add_domain(
        equal(
            "c9.no_sql_or_source_mutation",
            before,
            after,
            method="logical DB digests + native source sizes around real collection/scrape/export",
        ),
        equal(
            "c9.native_source_contents_unchanged",
            source_digests_before,
            source_digests_after,
            method="per-file SHA-256 digests of every native source file before/after C9",
        ),
    )
    scrape_status = FAIL if oracle_problems else outcome.scrape.status
    scrape_detail = "; ".join(oracle_problems) if oracle_problems else outcome.scrape.detail
    gate_entry("scrape", run, "scrape", scrape_status, scrape_detail)
    gate_entry(
        "export_privacy", run, "export_privacy", outcome.export.status, outcome.export.detail
    )
    if outcome.scrape.status == BLOCKED:
        stage.coverage_missing.append(outcome.scrape.detail)
    stage.finish("closure: index and resolve B")


def stage_c10(run: Run) -> None:
    stage = Stage(run, "C10", "closure")
    from .lab_product import index

    code, ms, tail = index(run.product)
    stage.add_domain(
        equal("b.index.completed", 0, code, method="real `ssgrep index --quiet`", detail=tail)
    )
    native = parse_native_omp(run.native["b"])
    problems: list[str] = []
    for task, query in (("B/L3", "joint reaccommodation plan with shared capacity and priority"),
                        ("B/L4", "applied rebooking requests with idempotency keys")):  # fmt: skip
        res, _s, _o = lab_persist.resolve_task(
            run.observer,
            run.product,
            run.roots,
            task=task,
            query=query,
            session=native,
            expected_text=["rebook_itinerary"] if task == "B/L4" else ["Joint allocation"],
            project=run.roots.project,
        )
        problems += [f"{task}: {p}" for p in res.problems]
    stage.add_domain(
        check(
            "b.sources_resolve",
            problems,
            expected="B's L3/L4 resolvable",
            actual="resolved",
            method="verifier search+show vs B's native digest",
        )
    )
    final = run.collect("final")
    result = acct.evaluate(run.frozen(), run.production_profile(run.frozen()))
    stage.add_accounting(
        check(
            "c10.b_history_creates_no_credit_for_its_own_work",
            acct.compare_with_snapshot(result, final.savings_document),
            expected="oracle",
            actual="final snapshot",
            method="oracle vs final `savings --json` after B indexed",
        ),
    )
    stage.add_accounting(
        equal(
            "c10.profiles_kept_separate",
            True,
            run.notes["production"].profile != run.notes["arithmetic"].profile,
            method="production-policy and airline-lab-arithmetic-v1 reported separately",
        ),
    )
    reconcile_journal(run, stage, "c10")
    stage.actual_delta = {"b_index_ms": ms}
    stage.finish("teardown and report")


def valuation_payloads(run: Run) -> None:
    for key, name in (
        ("production", "production-policy"),
        ("arithmetic", "airline-lab-arithmetic-v1"),
    ):
        result = run.notes.get(key)
        group = "valuation_production" if key == "production" else "valuation_arithmetic"
        if result is None:
            run.report.add_gate(group, GateEntry(name, NOT_RUN, "profile was not evaluated"))
            continue
        run.report.valuation(name, {"profile": name, **result.as_dict()})
        mismatch = run.notes.get(f"{key}_mismatch") or []
        run.report.add_gate(
            group,
            GateEntry(
                name,
                FAIL if mismatch else PASS,
                "; ".join(mismatch) or "independent oracle equals the real `savings --json`",
            ),
        )


# -------------------------------------------------------------------------------- the run


def run_standin(
    parent: Path | None, emit: Callable[[str], None], *, model_cache: Path | None = None
) -> tuple[Report, Path, dict[str, Any]]:
    """Run the whole stand-in pipeline; returns (report, artifact dir copy, teardown record)."""
    cache = model_cache or real_model_cache()
    roots = LabRoots.create(parent)
    mark, started = removal_mark(), time.monotonic()
    run: Run | None = None
    teardown: dict[str, Any] = {}
    try:
        run = build_run(roots, emit, cache)
        _lane(roots.artifacts, run.report, emit)
        mark = removal_mark()  # the lane audits its own container; ours starts here
        setup = setup_profile(run.product)
        run.notes["setup"] = setup
        run.notes["pre_collect_state"] = dig(run.product.json("savings")[1], "state")
        # an empty index and one control lookup so capture "begins" before A's transcripts exist
        from .lab_product import index

        index(run.product)
        run.observer.run(actor="control", task=None, tool="ssgrep",
                         argv=[str(run.product.binary), "search", "early control probe", "--json"],
                         env=run.product.env, cwd=roots.project, timeout=600,
                         shown=["ssgrep", "search", "early control probe", "--json"])  # fmt: skip
        run.collect("initial")
        stage_c0(run, "domain lane ran first in its own owned container")
        stage_c1(run)
        stage_c2(run)
        persisted = stage_c3(run)
        if persisted.get("ok") and run.sandbox_ok is not None:
            stage_c4(run)
            plan = stage_c5_c6(run)["plan"]
            stage_c7(run, plan)
            stage_c8(run)
            stage_c9(run)
            stage_c10(run)
        else:
            run.say("B not started: persistence gates did not pass")
        lab_metrics.fill(run)
        valuation_payloads(run)
    except PrerequisiteBlocked as exc:
        emit(f"[BLOCKED ] {exc}")
        teardown["blocked"] = str(exc)
        report = Report(roots.run_id, "standin", STANDIN_LABEL, False, STANDIN_LABEL)
        report.add_gate("cleanup", GateEntry("prerequisite", BLOCKED, str(exc)))
        run_report = report
    except BaseException:
        if run is not None:
            run.failed = True
        raise
    finally:
        if run is not None:
            try:
                run.observer.finalize()
                run.instance.teardown()
            finally:
                LabInstance.remove_run(run.instance.run_id)
            problems = [r for r in removals_since(mark)
                        if not r.owned_by(run.instance.run_id) and r.identified]  # fmt: skip
            leftover = sorted(list_containers(f"ssgrep-lab-run={run.instance.run_id}"))
            run.report.add_gate(
                "cleanup",
                GateEntry(
                    "cleanup.owned_resources",
                    FAIL if (problems or leftover) else PASS,
                    f"touched foreign: {len(problems)}; leftover containers: {len(leftover)}",
                ),
            )
            run_report = run.report
    out = _finalize(run, roots, run_report, teardown, started)
    return out


def _finalize(run: Run | None, roots: LabRoots, report: Report, teardown: dict[str, Any],
              started: float) -> tuple[Report, Path, dict[str, Any]]:  # fmt: skip
    keep = Path(roots.root.parent) / f"{roots.root.name}-report"
    secrets_list = list(run.secrets) if run else []
    if run is not None:
        texts = {p.name: p.read_text() for p in roots.artifacts.glob("*.json")}
        texts.update({f"checkpoints/{p.name}": p.read_text()
                      for p in (roots.artifacts / "checkpoints").glob("*.json")})  # fmt: skip
        leaks = canary_scan(texts, secrets_list)
        run.notes["canary_leaks"] = len(leaks)
        report.add_gate(
            "accounting_checkpoints",
            GateEntry("report.no_secret_canary", FAIL if leaks else PASS, "; ".join(leaks)),
        )
        for cid, item in report.checkpoints.as_dict().items():
            status = item["status"]
            if status != NOT_RUN:
                report.add_gate("accounting_checkpoints", GateEntry(cid, status, ""))
    keep.mkdir(mode=0o700, exist_ok=True)
    report.write_private(keep / "report.json")
    report.checkpoints.write(keep / "checkpoints.json")
    (keep / "gates.txt").write_text(gate_table(report) + "\n")
    (keep / "checkpoints.txt").write_text(report.checkpoints.table() + "\n")
    teardown["elapsed_s"] = int(time.monotonic() - started)
    teardown["root"] = roots.remove(retain_failure_to=None)
    return report, keep, teardown


__all__ = ["run_standin", "gate_assertion", "Assertion", "WRITER", "blocked", "NOT_RUN"]
