"""Deterministic domain lane: real PostgreSQL, independent oracle, fault injection.

``run_lane`` provisions one run-owned container, checks every domain gate against ACTUAL
query results and database state, proves the verifiers reject deliberately broken
formulations/implementations, and always tears down only that run's container. Infrastructure
failures keep a bounded private failure manifest before cleanup and re-raise.
"""

from __future__ import annotations

import json
import os
import secrets
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import dataset, domain_oracle as oracle, faults, mutations, queries
from .database import (  # fmt: skip
    ADMIN_DB,
    READER,
    ROLE_LIMITS,
    RUN_LABEL,
    WRITER,
    BaselineInfo,
    LabInstance,
    RemovalRecord,
    list_containers,
    preflight,
    removal_mark,
    removals_since,
)
from .gates import FAIL, PASS, Gate, from_violations, not_run

# variant -> scenarios that MUST detect it (the verifier fails the broken implementation)
BROKEN_L4: dict[str, tuple[str, ...]] = {
    "no_lock": ("race",),
    "no_revalidation": ("race",),
    "no_capacity_check": ("race", "stale_plan"),
    "half_applied": ("joint_apply", "replay"),
    "swallow_failure": ("rollback",),
}

Emit = Callable[[Gate], None]


def make_run_root(base: Path | None = None, run_id: str | None = None) -> tuple[Path, str]:
    """Owner-only run root: verifier/ (oracle answers) and artifacts/ (reports)."""
    run_id = run_id or secrets.token_hex(6)
    parent = base if base is not None else Path(os.environ.get("TMPDIR", "/tmp"))
    root = parent / f"ssgrep-airline-run-{run_id}"
    for sub in ("", "verifier", "artifacts"):
        (root / sub).mkdir(mode=0o700, parents=True, exist_ok=True)
        (root / sub).chmod(0o700)
    return root, run_id


def write_private(path: Path, payload: Any) -> None:
    data = json.dumps(payload, indent=2, sort_keys=True, default=str)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(data + "\n")


@dataclass
class LaneReport:
    run_id: str
    run_root: Path
    gates: list[Gate] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)
    started_utc: str = ""
    elapsed_s: float = 0.0

    def add(self, gate: Gate, emit: Emit | None = None) -> Gate:
        self.gates.append(gate)
        if emit:
            emit(gate)
        return gate

    @property
    def domain_status(self) -> str:
        statuses = {g.status for g in self.gates if not g.name.startswith("live.")}
        for status in (FAIL, "BLOCKED"):
            if status in statuses:
                return status
        return PASS if statuses == {PASS} else "UNKNOWN"

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_utc": self.started_utc,
            "elapsed_s": round(self.elapsed_s, 3),
            "domain_status": self.domain_status,
            "manifest": self.manifest,
            "gates": [g.as_dict() for g in self.gates],
        }


# ----------------------------------------------------------------------- gate builders


def _actual_rows(lab: LabInstance, database: str, sql: str) -> list[dict[str, Any]]:
    return lab.rows(database, sql, role=READER)


def gate_seed(lab: LabInstance, base: BaselineInfo, scratch: str) -> Gate:
    problems: list[str] = []
    if base.counts != dataset.EXPECTED_COUNTS:
        problems.append(f"table counts {base.counts} != {dataset.EXPECTED_COUNTS}")
    if sum(base.counts.values()) != dataset.TOTAL_ROWS:
        problems.append(f"total rows {sum(base.counts.values())} != {dataset.TOTAL_ROWS}")
    if not all(5 <= n <= 10 for n in base.counts.values()):
        problems.append("a table is outside the 5-10 row range")
    if base.digests != dataset.seed_digests():
        problems.append("database logical digests differ from the fixture definition")
    kinds = ", ".join(
        f"count(*) FILTER (WHERE c.contype = '{kind}') AS {name}"
        for kind, name in (("p", "pk"), ("f", "fk"), ("u", "uq"), ("c", "ck"))
    )
    catalog = lab.rows(
        scratch,
        f"SELECT {kinds}, count(*) FILTER (WHERE NOT c.convalidated) AS unvalidated "
        "FROM pg_constraint c JOIN pg_namespace n ON n.oid = c.connamespace "
        f"WHERE n.nspname = '{dataset.SCHEMA}'",
    )[0]
    if (catalog["pk"], catalog["fk"], catalog["uq"]) != (12, 13, 8) or catalog["ck"] < 20:
        problems.append(f"constraint inventory too small: {catalog}")
    if catalog["unvalidated"]:
        problems.append("a constraint is NOT VALID")
    refused: dict[str, str | None] = {}
    for probe in dataset.CONSTRAINT_PROBES:
        res = lab.psql(
            scratch, f"BEGIN;\n{probe.sql}\nROLLBACK;\n", check=False, app="lab-constraint-probe"
        )
        refused[probe.name] = res.sqlstate
        if res.ok or res.sqlstate != probe.sqlstate:
            problems.append(f"{probe.name}: expected SQLSTATE {probe.sqlstate}, got {res.sqlstate}")
    clone_state = lab.read_state(scratch)
    if dataset.logical_digests(clone_state) != base.digests:
        problems.append("scratch clone (or probes run in it) differs from the sealed baseline")
    evidence = {
        "counts": base.counts, "total_rows": sum(base.counts.values()),
        "overall_digest": base.digest, "table_digests": base.digests,
        "constraints": catalog, "rejected_probes": refused, "as_of": dataset.AS_OF,
        "schema_digest": dataset.schema_digest(),
    }  # fmt: skip
    return from_violations("seed.integrity", problems, evidence)


def gate_oracle_canon(state: oracle.State) -> Gate:
    """The reference model reproduces every sealed canonical answer from fixture rows alone."""
    problems: list[str] = []
    if oracle.settled_cash(state) != oracle.CANON_CASH:
        problems.append(f"cash model {oracle.settled_cash(state)}")
    totals = oracle.currency_totals(oracle.settled_cash(state))
    if totals != {"USD": 195000, "EUR": 45000}:
        problems.append(f"currency totals {totals}")
    seats = {r.flight_id: r.key() for r in oracle.seat_rows(state)}
    if seats != oracle.CANON_SEATS:
        problems.append(f"seat model {seats}")
    if oracle.sellable_map(state) != oracle.CANON_SELLABLE:
        problems.append("sellable model differs from 0,0,1,1,1,0,1,2")
    if tuple(oracle.affected_travellers(state)) != oracle.CANON_AFFECTED:
        problems.append(f"affected model {oracle.affected_travellers(state)}")
    routes = tuple(r.legs for r in oracle.route_candidates(state)[0])
    if set(routes) != set(oracle.CANON_ROUTES):
        problems.append(f"route model {routes}")
    if oracle.joint_plan(state) != oracle.CANON_PLAN:
        problems.append(f"joint plan model {oracle.joint_plan(state)}")
    return from_violations("oracle.canonical_answers", problems, {"cash": totals})


def gate_l1(lab: LabInstance, database: str, state: oracle.State) -> list[Gate]:
    cash = _actual_rows(lab, database, queries.CASH_PREAGGREGATED)
    seats = _actual_rows(lab, database, queries.seats())
    advanced = _actual_rows(lab, database, queries.seats(mutations.ADVANCED_AS_OF))
    later = datetime(2026, 11, 1, 7, 20, tzinfo=UTC)
    return [
        from_violations("l1.cash", oracle.verify_cash(cash, state), {"actual": cash}),
        from_violations("l1.seats", oracle.verify_seats(seats, state), {"actual": seats}),
        from_violations(
            "l1.clock_advance",
            oracle.verify_seats(advanced, state, later),
            {"as_of": mutations.ADVANCED_AS_OF, "actual": advanced},
        ),
    ]


def gate_l2(lab: LabInstance, database: str, state: oracle.State) -> list[Gate]:
    affected = _actual_rows(lab, database, queries.affected())
    conns = _actual_rows(lab, database, queries.connections())
    clocks = _actual_rows(lab, database, queries.local_clock_listing())
    instants = _actual_rows(lab, database, queries.flight_instants())
    pairs = _actual_rows(lab, database, queries.connectable_pairs())
    names = [r["display_name"] for r in state["travellers"]]
    problems = oracle.verify_connectable(pairs, state)
    if ("F8", "F2") in {(r["from_flight"], r["to_flight"]) for r in pairs}:
        problems.append("F8 connects to F2")
    clock_problems = oracle.verify_local_clock(clocks, state) + oracle.verify_flight_instants(
        instants, state
    )
    offsets = {r["flight_id"]: (r["local_clock"], r["utc_offset_hours"]) for r in clocks}
    if offsets["F1"] != ("01:30", -7) or offsets["F8"] != ("01:30", -8):
        clock_problems.append(f"F1/F8 local clocks {offsets['F1']} {offsets['F8']}")
    return [
        from_violations(
            "l2.affected",
            oracle.verify_affected(affected, state),
            {"actual": affected, "duplicate_names": names.count("Sam Lee")},
        ),  # fmt: skip
        from_violations(
            "l2.connections", oracle.verify_connections(conns, state), {"actual": conns}
        ),
        from_violations("l2.local_time_dst", clock_problems, {"actual": clocks}),
        from_violations("l2.connectable_pairs", problems, {"actual": len(pairs)}),
    ]


def gate_l3(lab: LabInstance, database: str, state: oracle.State) -> list[Gate]:
    routes = _actual_rows(lab, database, queries.candidate_routes())
    affected = _actual_rows(lab, database, queries.affected())
    seats = _actual_rows(lab, database, queries.seats())
    plan = queries.greedy_priority_plan(affected, routes, seats)
    policies = _actual_rows(lab, database, queries.policy_ids_used(("F3", "F4")))
    policy_problems = []
    if [r["rule_id"] for r in policies] != oracle.policy_ids(state, ("F3", "F4")):
        policy_problems.append(f"policy ids {policies}")
    return [
        from_violations(
            "l3.candidate_routes", oracle.verify_routes(routes, state), {"actual": routes}
        ),
        from_violations(
            "l3.joint_plan",
            oracle.plan_violations(state, plan),
            {"plan": {b: list(legs) for b, legs in plan.items()}},
        ),  # fmt: skip
        from_violations("l3.policy_ids", policy_problems, {"actual": policies}),
    ]


def gate_domain_sensitivity(lab: LabInstance, database: str, state: oracle.State) -> Gate:
    problems: list[str] = []
    detected: dict[str, list[str]] = {}
    for formulation in mutations.MUTATIONS:
        _, violations = mutations.evaluate(lab, database, formulation, state)
        detected[formulation.name] = violations[:3]
        if not violations:
            problems.append(f"{formulation.name}: oracle ACCEPTED a wrong formulation")
        for missing in mutations.missing_flags(formulation, violations):
            problems.append(f"{formulation.name}: expected oracle to flag {missing!r}")
    for formulation in mutations.CORRECT:
        _, violations = mutations.evaluate(lab, database, formulation, state)
        if violations:
            problems.append(f"{formulation.name}: correct formulation rejected {violations[:2]}")
    return from_violations(
        "sensitivity.domain", problems,
        {"wrong_rejected": len(mutations.MUTATIONS), "correct_accepted": len(mutations.CORRECT),
         "violations": detected},
    )  # fmt: skip


def gate_roles(lab: LabInstance, database: str) -> Gate:
    problems: list[str] = []
    refusals: dict[str, str | None] = {}

    def refuse(name: str, role: str, sql: str, expect: str, db: str = database) -> None:
        res = lab.psql(db, sql, role=role, check=False, app="lab-role-probe")
        if expect == "connect":  # connection-time refusals carry no SQLSTATE in psql output
            refused = not res.ok and "permission denied for database" in res.stderr
            refusals[name] = "permission denied for database" if refused else res.sqlstate
            if not refused:
                problems.append(f"{name}: connection to {db} was not refused")
            return
        refusals[name] = res.sqlstate
        if res.ok or res.sqlstate != expect:
            problems.append(f"{name}: expected {expect}, got {res.sqlstate}")

    s = dataset.SCHEMA
    refuse("reader.update", READER, f"UPDATE {s}.coupons SET status = 'void';", "25006")
    refuse("reader.insert", READER, f"INSERT INTO {s}.airports VALUES ('XXX', 'UTC');", "25006")
    refuse("writer.payments", WRITER, f"UPDATE {s}.payment_events SET status = 'failed';", "42501")
    refuse("writer.tickets", WRITER, f"UPDATE {s}.tickets SET face_value_minor = 1;", "42501")
    refuse("writer.delete_coupon", WRITER, f"DELETE FROM {s}.coupons;", "42501")
    refuse("writer.capacity", WRITER, f"UPDATE {s}.inventory SET capacity = 99;", "42501")
    refuse("writer.ddl", WRITER, f"CREATE TABLE {s}.x (a int);", "42501")
    refuse("writer.admin_db", WRITER, "SELECT 1;", "connect", db=ADMIN_DB)
    refuse("reader.admin_db", READER, "SELECT 1;", "connect", db=ADMIN_DB)
    for role, limits in ROLE_LIMITS.items():
        for key, value in limits.items():
            actual = lab.psql(database, f"SHOW {key};", role=role).stdout.strip()
            if actual != value:
                problems.append(f"{role} {key} is {actual}, expected {value}")
    read_ok = lab.rows(database, f"SELECT count(*) AS n FROM {s}.coupons", role=READER)[0]["n"]
    if read_ok != 10:
        problems.append("reader cannot read coupons")
    return from_violations("isolation.database_roles", problems, {"refused": refusals})


def gate_l4(prep: faults.Prepared) -> list[Gate]:
    gates = []
    for name in faults.SCENARIOS:
        scenario = faults.run(prep, name, "correct")
        gates.append(
            from_violations(
                f"l4.{name}",
                scenario.violations,
                {"attempts": scenario.attempts, **scenario.evidence},
            )
        )
    return gates


def gate_l4_sensitivity(lab: LabInstance, base: BaselineInfo) -> Gate:
    """Broken L4 implementations must FAIL the named scenarios; the correct one must not."""
    prep = faults.Prepared(lab, base.state)
    problems: list[str] = []
    caught: dict[str, dict[str, list[str]]] = {}
    for variant, scenarios in BROKEN_L4.items():
        caught[variant] = {}
        for name in scenarios:
            result = faults.run(prep, name, variant)
            caught[variant][name] = result.violations[:3]
            if result.passed:
                problems.append(f"{variant}: verifier ACCEPTED it in {name}")
    return from_violations("sensitivity.l4", problems, {"caught": caught})


def gate_cleanup(
    run_id: str,
    lab: LabInstance | None,
    removals: Sequence[RemovalRecord],
    foreign_before: set[str],
) -> Gate:
    """Cleanup gate over what the lab ACTUALLY removed, not over unrelated daemon churn.

    FAIL only if (a) the lab asked docker to remove/stop/kill a container that is not this
    run's own (ownership read before the call), (b) run-labelled containers remain, or
    (c) the lab's own container was never removed. Foreign containers that appeared or
    vanished without the lab touching them (shared daemon, other projects' reapers) are
    reported as information only.
    """
    name = "cleanup.owned_resources"
    problems: list[str] = []
    for record in removals:
        if record.owned_by(run_id):
            continue
        if record.identified or record.succeeded:
            problems.append(
                f"lab {record.verb} touched a container it does not own: "
                f"{record.container_id or record.ref} name={record.name} "
                f"run_label={record.run_label}"
            )
    if lab is not None and not lab.removed:
        problems.append("owned container was not removed")
    remaining = sorted(list_containers(f"{RUN_LABEL}={run_id}"))
    if remaining:
        problems.append(f"run-labelled containers remain: {remaining}")
    foreign_after = other_containers()
    removed_ids = {r.container_id for r in removals if r.succeeded and r.container_id}
    evidence = {
        "lab_removals": [r.as_dict() for r in removals],
        "foreign_before": len(foreign_before),
        "foreign_after": len(foreign_after),
        "foreign_disappeared_informational": sorted(foreign_before - foreign_after - removed_ids),
        "foreign_appeared_informational": sorted(foreign_after - foreign_before),
    }
    return from_violations(name, problems, evidence)


def other_containers() -> set[str]:
    """IDs of every container NOT carrying this lab's run label (daemon-wide snapshot)."""
    return list_containers() - list_containers(RUN_LABEL)


# ------------------------------------------------------------------------------- lane


def run_lane(
    run_root: Path, run_id: str, emit: Emit | None = None, *, report_name: str = "domain-lane.json"
) -> LaneReport:
    report = LaneReport(
        run_id, run_root, started_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    started = time.monotonic()
    gate = report.add(preflight(), emit)
    if gate.status != PASS:
        report.elapsed_s = time.monotonic() - started
        write_private(run_root / "artifacts" / report_name, report.as_dict())
        return report
    foreign_before = other_containers()
    removal_start = removal_mark()
    lab: LabInstance | None = None
    try:
        lab = LabInstance.start(run_id)
        report.manifest = lab.manifest()
        base = lab.provision_baseline(faults.baseline_sql("correct"))
        write_private(
            run_root / "verifier" / "seed.json",
            {"state": base.state, "digests": base.digests, "overall": base.digest},
        )
        (run_root / "verifier" / "seed.sql").write_text(dataset.seed_sql())
        (run_root / "verifier" / "seed.sql").chmod(0o600)
        scratch = lab.clone("scratch")
        state = base.state
        report.add(gate_seed(lab, base, scratch), emit)
        report.add(gate_oracle_canon(state), emit)
        for g in gate_l1(lab, scratch, state) + gate_l2(lab, scratch, state):
            report.add(g, emit)
        for g in gate_l3(lab, scratch, state):
            report.add(g, emit)
        report.add(gate_domain_sensitivity(lab, scratch, state), emit)
        report.add(gate_roles(lab, scratch), emit)
        prep = faults.Prepared(lab, base.state)
        for g in gate_l4(prep):
            report.add(g, emit)
        report.add(gate_l4_sensitivity(lab, base), emit)
        report.manifest["final_container"] = lab.inspect()["State"]["Status"]
    except Exception as exc:  # noqa: BLE001 - recorded as a FAIL gate + private manifest, never hidden
        failure = {
            "run_id": run_id,
            "error_type": type(exc).__name__,
            "message": str(exc)[:2000],
            "gates_so_far": [g.as_dict() for g in report.gates][-40:],
            "container": lab.container_name if lab else None,
        }
        write_private(run_root / "artifacts" / "failure-manifest.json", failure)
        report.add(Gate("lane.execution", FAIL, f"{type(exc).__name__}: {str(exc)[:300]}"), emit)
    finally:
        if lab is not None:
            try:
                lab.teardown()
            finally:
                LabInstance.remove_run(run_id)
    report.add(gate_cleanup(run_id, lab, removals_since(removal_start), foreign_before), emit)
    report.elapsed_s = time.monotonic() - started
    write_private(run_root / "artifacts" / report_name, report.as_dict())
    return report


def live_gate(runtime: str | None) -> Gate:
    """Live mode is implemented by a different task; never a PASS and never silent."""
    detail = (
        "live two-session lane is not implemented in this lane; "
        f"requested runtime={runtime!r}; no runtime was contacted and nothing was faked"
    )
    return not_run("live.two_session_lab", detail)
