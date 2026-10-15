"""Scenario drivers that exercise the L4 operation against the real database.

Every scenario runs on its own clone of the recorded baseline (``CREATE DATABASE ...
TEMPLATE``), drives real ``psql`` clients, and hands the ACTUAL before/after states to the
independent verifiers in ``l4_verify``. Synchronisation is by PostgreSQL advisory-lock
gates observed through ``pg_locks``; nothing relies on a fixed sleep to order clients.

Fault/barrier seams are installed by the lab as a trigger on ``coupons`` that is inert
unless the *client session* sets ``lab.barrier_key`` / ``lab.fail_coupon``. The operation
under test contains no test hooks.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from . import dataset, domain_oracle as oracle, l4_reference as l4, l4_verify
from .bounded_capture import AsyncCapture, OutputLimitExceeded, decode_text
from .database import ADMIN_DB, SQLSTATE_RE, VERIFIER, WRITER, LabError, LabInstance

START_GATE_KEY = 7_000_001
BARRIER_KEY = 7_000_002
RACE_LOCK_TIMEOUT = "20s"  # recorded: gate waits count against lock_timeout
POLL_DEADLINE_S = 25.0

KIT_SQL = f"""
CREATE OR REPLACE FUNCTION {dataset.SCHEMA}.lab_fault_hook() RETURNS trigger
  LANGUAGE plpgsql AS $fn$
DECLARE
  v_barrier text := nullif(current_setting('lab.barrier_key', true), '');
  v_fail text := nullif(current_setting('lab.fail_coupon', true), '');
BEGIN
  IF v_barrier IS NOT NULL THEN
    PERFORM pg_advisory_xact_lock_shared(v_barrier::bigint);
  END IF;
  IF v_fail IS NOT NULL AND NEW.coupon_id = v_fail THEN
    RAISE EXCEPTION 'injected failure before updating coupon %', NEW.coupon_id
      USING ERRCODE = 'P0001';
  END IF;
  RETURN NEW;
END $fn$;
DROP TRIGGER IF EXISTS lab_fault_hook ON {dataset.SCHEMA}.coupons;
CREATE TRIGGER lab_fault_hook BEFORE UPDATE ON {dataset.SCHEMA}.coupons
  FOR EACH ROW EXECUTE FUNCTION {dataset.SCHEMA}.lab_fault_hook();
"""


def baseline_sql(variant: str = "correct") -> list[str]:
    """Extra SQL executed into the baseline: the operation under test plus the fault kit."""
    return [l4.install_sql(variant), KIT_SQL]


@dataclass
class ClientResult:
    app: str
    returncode: int | None
    stdout: str
    stderr: str
    duration_ns: int
    response: dict[str, Any] | None = None

    @property
    def sqlstate(self) -> str | None:
        match = SQLSTATE_RE.search(self.stderr)
        return match.group(1) if match else None

    def evidence(self) -> dict[str, Any]:
        return {
            "app": self.app,
            "returncode": self.returncode,
            "duration_ns": self.duration_ns,
            "response": self.response,
            "error": self.stderr.strip().splitlines()[0][:200] if self.stderr.strip() else None,
        }


@dataclass
class Scenario:
    """Outcome of one scenario: violations (empty = invariants hold) plus recorded evidence."""

    name: str
    violations: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    attempts: list[dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.violations


def parse_response(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    return None


def _script(
    request: l4.RebookRequest,
    *,
    digest: str | None = None,
    start_gate: bool = False,
    barrier: bool = False,
    fail_coupon: str | None = None,
) -> str:
    lines = [f"SET lock_timeout = '{RACE_LOCK_TIMEOUT}';"]
    if barrier:
        lines.append(f"SET lab.barrier_key = '{BARRIER_KEY}';")
    if fail_coupon:
        lines.append(f"SET lab.fail_coupon = '{fail_coupon}';")
    if start_gate:  # blocks while the verifier holds the exclusive gate
        lines += [
            f"SELECT pg_advisory_lock_shared({START_GATE_KEY});",
            f"SELECT pg_advisory_unlock_shared({START_GATE_KEY});",
        ]
    lines += ["BEGIN ISOLATION LEVEL READ COMMITTED;", request.call(digest=digest), "COMMIT;"]
    return "\n".join(lines) + "\n"


def run_client(
    lab: LabInstance, database: str, script: str, *, app: str, timeout: float = 60.0
) -> ClientResult:
    """One independent writer-role psql client; failures are returned, never swallowed."""
    started = time.monotonic_ns()
    res = lab.psql(database, script, role=WRITER, app=app, timeout=timeout, check=False)
    return ClientResult(
        app, res.returncode, res.stdout, res.stderr, time.monotonic_ns() - started,
        parse_response(res.stdout),
    )  # fmt: skip


def apply_request(
    lab: LabInstance, database: str, request: l4.RebookRequest, *, app: str, **options: Any
) -> ClientResult:
    return run_client(lab, database, _script(request, **options), app=app)


# --------------------------------------------------------------------- gates and clients


class AdvisoryGate:
    """A verifier-held exclusive advisory lock that clients block on until released."""

    def __init__(self, lab: LabInstance, database: str, key: int, app: str) -> None:
        self.lab, self.key, self.app = lab, key, app
        self.proc = lab.popen(database, role=VERIFIER, app=app)
        self.capture = AsyncCapture(self.proc, timeout=None)
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write(f"SELECT pg_advisory_lock({key});\n")
            self.proc.stdin.flush()
            wait_until(lambda: self.held(), f"gate {app} to be held")
        except BaseException:
            self.capture.abort()
            raise

    def held(self) -> bool:
        self.capture.check_output_limit()
        rows = self.lab.rows(
            ADMIN_DB,
            "SELECT 1 AS n FROM pg_locks l JOIN pg_stat_activity a ON a.pid = l.pid "
            f"WHERE l.locktype = 'advisory' AND l.objid = {self.key}::oid AND l.granted "
            f"AND a.application_name = '{self.app}'",
        )
        return bool(rows)

    def release(self) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(f"SELECT pg_advisory_unlock({self.key});\n")
        _close_stdin(self.proc)
        captured = self.capture.finish(timeout=30)
        if captured.timed_out:
            raise LabError(f"gate {self.app} exceeded capture timeout")
        out, err = _captured_text(captured)
        if self.proc.returncode != 0:
            raise LabError(f"gate {self.app} failed: {err.strip()[:200]}")

    def abort(self) -> None:
        self.capture.abort()


def _close_stdin(proc: subprocess.Popen[str]) -> None:
    """Close and detach stdin so ``communicate`` does not flush a closed file."""
    if proc.stdin is not None:
        proc.stdin.close()
        proc.stdin = None


def _captured_text(captured) -> tuple[str, str]:
    return decode_text(captured.stdout), decode_text(captured.stderr)


def wait_until(predicate: Callable[[], bool], what: str, deadline: float = POLL_DEADLINE_S) -> None:
    """Poll an OBSERVED condition (never a guess-the-timing sleep) with a hard deadline."""
    end = time.monotonic() + deadline
    while True:
        if predicate():
            return
        if time.monotonic() > end:
            raise LabError(f"timed out waiting for {what}")
        time.sleep(0.05)


def _waiters(lab: LabInstance, apps: Sequence[str], gate_key: int) -> dict[str, dict[str, bool]]:
    names = ", ".join("'" + a + "'" for a in apps)
    rows = lab.rows(
        ADMIN_DB,
        "SELECT a.application_name AS app, "
        f"coalesce(bool_or(NOT l.granted AND l.locktype = 'advisory' "
        f"AND l.objid = {gate_key}::oid), false) AS at_start_gate, "
        f"coalesce(bool_or(NOT l.granted AND NOT (l.locktype = 'advisory' "
        f"AND l.objid = {gate_key}::oid)), false) AS blocked_elsewhere "
        "FROM pg_stat_activity a LEFT JOIN pg_locks l ON l.pid = a.pid "
        f"WHERE a.application_name IN ({names}) GROUP BY a.application_name",
    )
    return {r["app"]: r for r in rows}


class _ClientLab(Protocol):
    def popen(self, database: str, /, *, role: str, app: str) -> subprocess.Popen[str]: ...


class _Client:
    def __init__(self, lab: _ClientLab, database: str, app: str, script: str) -> None:
        self.app = app
        self.started = time.monotonic_ns()
        self.proc = lab.popen(database, role=WRITER, app=app)
        self.capture = AsyncCapture(self.proc, timeout=None)
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write(script)
            _close_stdin(self.proc)
        except BaseException:
            self.capture.abort()
            raise

    def finish(self, timeout: float = 60.0) -> ClientResult:
        captured = self.capture.finish(timeout=timeout)
        out, err = _captured_text(captured)
        return ClientResult(
            self.app, self.proc.returncode, out, err, time.monotonic_ns() - self.started,
            parse_response(out),
        )  # fmt: skip

    def abort(self) -> None:
        self.capture.abort()


def run_barriered_pair(
    lab: LabInstance, database: str, requests: Sequence[l4.RebookRequest]
) -> tuple[list[ClientResult], dict[str, Any]]:
    """Release two real concurrent clients through a start gate and a commit barrier.

    1. both clients are observed blocked on the verifier-held start gate;
    2. the gate is released, so both begin at once;
    3. the verifier-held commit barrier keeps both inside their transactions until each is
       observed waiting on a lock (the barrier or the competitor's row lock), so the
       overlap is real for every variant and not a timing accident;
    4. the barrier opens and both finish.
    """
    apps = [f"lab-race-{i}" for i in range(len(requests))]
    observed: dict[str, Any] = {"apps": apps}
    gates: list[AdvisoryGate] = []
    clients: list[_Client] = []
    try:
        start_gate = AdvisoryGate(lab, database, START_GATE_KEY, "lab-gate-start")
        gates.append(start_gate)
        barrier = AdvisoryGate(lab, database, BARRIER_KEY, "lab-gate-barrier")
        gates.append(barrier)
        for app, request in zip(apps, requests, strict=True):
            script = _script(request, start_gate=True, barrier=True)
            clients.append(_Client(lab, database, app, script))
        wait_until(
            lambda: (
                all(w.get("at_start_gate") for w in _waiters(lab, apps, START_GATE_KEY).values())
                and len(_waiters(lab, apps, START_GATE_KEY)) == len(apps)
            ),
            "every client to block on the start gate",
        )
        observed["clients_blocked_at_start_gate"] = len(apps)
        start_gate.release()

        def overlapped() -> bool:
            seen = _waiters(lab, apps, START_GATE_KEY)
            return all(
                c.proc.poll() is not None or seen.get(c.app, {}).get("blocked_elsewhere")
                for c in clients
            )

        wait_until(overlapped, "every client to be mid-transaction and waiting")
        observed["clients_mid_transaction_waiting"] = [
            c.app for c in clients if c.proc.poll() is None
        ]
        barrier.release()
        results = [c.finish() for c in clients]
    except BaseException:
        capture_error = None
        for owner in [*gates, *clients]:
            try:
                owner.abort()
            except BaseException as exc:
                if capture_error is None or isinstance(exc, OutputLimitExceeded):
                    capture_error = exc
        if capture_error is not None:
            raise capture_error from None
        raise
    return results, observed


# --------------------------------------------------------------------- scenarios


@dataclass
class Prepared:
    lab: LabInstance
    baseline_state: dict[str, list[dict[str, Any]]]
    template: str = "base"  # label of the database scenario clones are cut from
    created: list[str] = field(default_factory=list)

    def clone(
        self, label: str, variant: str = "correct"
    ) -> tuple[str, dict[str, list[dict[str, Any]]]]:
        """Cut a scenario database from the baseline; swap in a broken operation if asked."""
        database = self.lab.clone(label, self.template)
        self.created.append(label)
        if variant != "correct":
            self.lab.psql(database, l4.install_sql(variant))
        state = self.lab.read_state(database)
        if dataset.logical_digests(state) != dataset.logical_digests(self.baseline_state):
            raise LabError(f"clone {label} does not equal the recorded baseline")
        return database, state


def _attempt(label: str, result: ClientResult) -> dict[str, Any]:
    return {"label": label, **result.evidence()}


def joint_apply(prep: Prepared, variant: str = "correct") -> Scenario:
    scenario = Scenario(f"l4.joint_apply[{variant}]")
    database, before = prep.clone(f"joint_{variant}", variant)
    plan = oracle.joint_plan(before)
    requests = [
        l4.request_for(before, f"l4:joint:{b}", b, legs) for b, legs in sorted(plan.items())
    ]
    for request in requests:
        result = apply_request(prep.lab, database, request, app=f"lab-joint-{request.booking_id}")
        scenario.attempts.append(_attempt(request.request_key, result))
        if result.returncode != 0 or result.response is None:
            scenario.violations.append(
                f"{request.request_key} failed: {result.stderr.strip()[:160]}"
            )
    after = prep.lab.read_state(database)
    scenario.violations += l4_verify.verify_joint_apply(before, after, requests)
    scenario.evidence = {
        "plan": {b: list(legs) for b, legs in plan.items()},
        "digest_after": dataset.overall_digest(dataset.logical_digests(after)),
    }
    return scenario


def race(prep: Prepared, variant: str = "correct") -> Scenario:
    scenario = Scenario(f"l4.race[{variant}]")
    database, before = prep.clone(f"race_{variant}", variant)
    requests = [l4.request_for(before, f"l4:race:{b}", b, ("F3", "F4")) for b in ("B1", "B2")]
    results, observed = run_barriered_pair(prep.lab, database, requests)
    for request, result in zip(requests, results, strict=True):
        scenario.attempts.append(_attempt(request.request_key, result))
    responses = {
        r.request_key: res.response or {"outcome": "error", "reason": res.stderr.strip()[:120]}
        for r, res in zip(requests, results, strict=True)
    }
    after = prep.lab.read_state(database)
    scenario.violations += l4_verify.verify_race(before, after, requests, responses)
    try:
        final_occupancy: Any = {
            f: n for f, n in l4_verify.occupancy(after).items() if f in ("F3", "F4")
        }
    except oracle.IntegrityError as exc:  # recorded as evidence, never hidden
        final_occupancy = f"integrity failure: {exc}"
    scenario.evidence = {
        "synchronisation": observed, "responses": responses, "final_occupancy": final_occupancy,
    }  # fmt: skip
    return scenario


def replay(prep: Prepared, variant: str = "correct") -> Scenario:
    """Winner's response is lost; a fresh client replays the identical key and digest."""
    scenario = Scenario(f"l4.replay[{variant}]")
    database, before = prep.clone(f"replay_{variant}", variant)
    request = l4.request_for(before, "l4:replay:B1", "B1", ("F5",))
    first = apply_request(prep.lab, database, request, app="lab-replay-first")
    scenario.attempts.append(_attempt("first (response discarded: lost in transit)", first))
    mid = prep.lab.read_state(database)
    stored = next(r for r in mid["rebooking_requests"] if r["request_key"] == request.request_key)
    original = {**stored["response_json"], "replayed": False}
    second = apply_request(prep.lab, database, request, app="lab-replay-restarted-client")
    scenario.attempts.append(_attempt("replay after client restart", second))
    after = prep.lab.read_state(database)
    if first.returncode != 0 or second.response is None:
        scenario.violations.append("replay driver did not obtain responses")
    else:
        scenario.violations += l4_verify.verify_replay(mid, after, original, second.response)
    committed = oracle.apply_moves(
        before, [oracle.MoveRequest(request.request_key, "B1", request.itinerary)]
    )
    scenario.violations += oracle.verify_state_delta(after, committed)
    return scenario


def key_conflict(prep: Prepared, variant: str = "correct") -> Scenario:
    scenario = Scenario(f"l4.key_conflict[{variant}]")
    database, before = prep.clone(f"keyconflict_{variant}", variant)
    original = l4.request_for(before, "l4:conflict:B1", "B1", ("F5",))
    first = apply_request(prep.lab, database, original, app="lab-conflict-first")
    scenario.attempts.append(_attempt("original", first))
    mid = prep.lab.read_state(database)
    altered = original.with_itinerary(("F3", "F4"))  # same key, different payload digest
    other = apply_request(prep.lab, database, altered, app="lab-conflict-altered")
    scenario.attempts.append(_attempt("same key, different digest", other))
    after = prep.lab.read_state(database)
    if other.response is None:
        scenario.violations.append("conflicting request produced no response")
    else:
        scenario.violations += l4_verify.verify_key_conflict(mid, after, other.response)
    if altered.digest == original.digest:
        scenario.violations.append("test payload did not actually change the digest")
    return scenario


def rollback(prep: Prepared, variant: str = "correct") -> Scenario:
    """Fail between the two coupon updates of an ORD move; roll back; retry once."""
    scenario = Scenario(f"l4.rollback[{variant}]")
    database, before = prep.clone(f"rollback_{variant}", variant)
    request = l4.request_for(before, "l4:rollback:B1", "B1", ("F3", "F4"))
    failed = apply_request(prep.lab, database, request, app="lab-rollback-fail", fail_coupon="C2")
    scenario.attempts.append(_attempt("interrupted between coupon updates", failed))
    after_failure = prep.lab.read_state(database)
    if failed.returncode == 0:
        scenario.violations.append(
            "injected failure did not abort the transaction (swallowed or never reached)"
        )
    elif failed.sqlstate != "P0001":
        scenario.violations.append(f"failed with {failed.sqlstate!r}, not the injected P0001")
    scenario.violations += l4_verify.verify_rollback(before, after_failure)
    retry = apply_request(prep.lab, database, request, app="lab-rollback-retry")
    scenario.attempts.append(_attempt("clean retry", retry))
    after_retry = prep.lab.read_state(database)
    if retry.response is None:
        scenario.violations.append("clean retry produced no response")
    else:
        scenario.violations += l4_verify.verify_clean_retry(
            before, after_retry, request, retry.response
        )
        again = apply_request(prep.lab, database, request, app="lab-rollback-again")
        scenario.attempts.append(_attempt("retry replay (must be a no-op)", again))
        final = prep.lab.read_state(database)
        scenario.violations += [
            f"third attempt mutated {t}" for t in l4_verify.changed_tables(after_retry, final)
        ]
    return scenario


def stale_plan(prep: Prepared, variant: str = "correct") -> Scenario:
    scenario = Scenario(f"l4.stale_plan[{variant}]")
    database, before = prep.clone(f"stale_{variant}", variant)
    plan = oracle.joint_plan(before)
    stale = l4.request_for(before, "l4:stale:B1", "B1", plan["B1"])
    hold_id = "H-STALE-1"
    prep.lab.psql(
        database,
        l4.place_hold_call(hold_id, "F5", 1, dataset.FUTURE_HOLD_EXPIRY),
        role=VERIFIER,
    )
    with_hold = prep.lab.read_state(database)
    scenario.violations += l4_verify.verify_hold_delta(before, with_hold, hold_id, "F5")
    if oracle.sellable_map(with_hold)["F5"] != 0 or not oracle.plan_violations(with_hold, plan):
        scenario.violations.append("oracle does not consider the stale plan infeasible")
    result = apply_request(prep.lab, database, stale, app="lab-stale-apply")
    scenario.attempts.append(_attempt("apply stale plan", result))
    after = prep.lab.read_state(database)
    if result.response is None:
        scenario.violations.append("stale apply produced no response")
    else:
        scenario.violations += l4_verify.verify_stale_plan(
            with_hold, after, stale, result.response, hold_id
        )
    scenario.evidence = {
        "baseline_digest": dataset.overall_digest(dataset.logical_digests(before)),
        "changed_baseline_digest": dataset.overall_digest(dataset.logical_digests(with_hold)),
        "expected_delta": {"seat_holds": f"+{hold_id} on F5"},
    }
    return scenario


def run(prep: Prepared, name: str, variant: str = "correct") -> Scenario:
    """Run one scenario and drop its clone afterwards (evidence lives in the Scenario)."""
    try:
        return SCENARIOS[name](prep, variant)
    finally:
        while prep.created:
            prep.lab.drop_clone(prep.created.pop())


SCENARIOS: Mapping[str, Callable[[Prepared, str], Scenario]] = {
    "joint_apply": joint_apply,
    "race": race,
    "replay": replay,
    "key_conflict": key_conflict,
    "rollback": rollback,
    "stale_plan": stale_plan,
}
