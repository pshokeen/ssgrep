"""Accounting sensitivity cases against the REAL ssgrep binaries (validation-only).

Every case drives the real ``.venv/bin/ssgrep`` (``index``/``search``/``show``/``metrics
collect``/``savings --json``/``metrics status --json``) in an isolated lab root, using the real
cached embedding model, against SCRIPTED-STANDIN consumer sessions, and compares the product's
snapshot with the independent oracle (``accounting_oracle``). The oracle never reads product
output; the product never reads oracle output; expected values come from the observer's own
dispatch/native evidence.

One shared fixture (one lab root, one real index of one finished SOURCE session, one early
control lookup so capture "begins") is built once. Each case works in a ``Scene``: a cheap copy
of that fixture's data/verifier state plus its own view directory, so cases cannot contaminate
each other's cumulative journal while still sharing the expensive index.

Results are ``CaseResult`` rows: PASS, FAIL (the lab/product broke an expectation the design
fixes), DISAGREEMENT (product and oracle differ; the diagnosis names which side the design D3-D7
supports) or BLOCKED (a prerequisite cannot be reproduced; never faked).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from . import accounting_oracle as oracle, lab_product as lp, observer as obs
from .roots import LabRoots, runtime_env
from .standin_agent import NativeWriter, Step, run_turn

PASS, FAIL, DISAGREEMENT, BLOCKED = "PASS", "FAIL", "DISAGREEMENT", "BLOCKED"
CASES: tuple[str, ...] = (
    "empty_error_missing_lookups",
    "tiny_show_truncation",
    "repeated_lookups",
    "mentions_not_executions",
    "shell_loop_two_searches",
    "prior_known_and_query",
    "relay_vs_sibling",
    "duplicate_mutated_partial_histories",
    "journal_write_loss",
    "source_expiry",
    "recollection_replay_no_extra_usage",
    "omp_trailer_exposure",  # extra: real omp appends a "Wall time" trailer to bash results
)
SOURCE_ID = "srcdoc01"
SOURCE_REF = f"omp:{SOURCE_ID}:ep:0"
QUERY = "minimum connection rules per airport"
DEN, ORD, SFO, JFK = "MCT-DEN-045", "MCT-ORD-045", "MCT-SFO-045", "MCT-JFK-060"
REVIEWED = tuple(("identifier", t) for t in (DEN, ORD, SFO, JFK))
TINY_CORPUS = 8  # any count below the 200-document frequency minimum: production credit unknown
SOURCE_PROMPT = (
    f"Find the minimum connection rules per airport. Known user fact {JFK} applies at JFK."
)
SOURCE_SAY = f"Policies found: {DEN} (Denver) and {ORD} (Chicago); also {SFO}."


@dataclass(frozen=True)
class CaseResult:
    name: str
    status: str
    expected: dict[str, Any]
    actual: dict[str, Any]
    evidence: tuple[str, ...]
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "expected": self.expected,
            "actual": self.actual,
            "evidence": list(self.evidence),
            "detail": self.detail,
        }


class InfrastructureError(RuntimeError):
    """The lab itself (not a case expectation) could not do what the case needs."""


@dataclass
class Check:
    """Collects violated expectations; ``disagree`` records a product/oracle split, never hidden."""

    problems: list[str] = field(default_factory=list)
    disagreements: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)

    def ok(self, label: str, condition: bool, why: str = "") -> bool:
        if not condition:
            self.problems.append(f"{label}: {why}" if why else label)
        return condition

    def eq(self, label: str, expected: Any, actual: Any) -> bool:
        return self.ok(label, expected == actual, f"expected {expected!r}, got {actual!r}")

    def disagree(self, label: str, diagnosis: str) -> None:
        self.disagreements.append(f"{label}: {diagnosis}")

    def status(self) -> str:
        if self.problems:
            return FAIL
        if self.blocked:
            return BLOCKED
        return DISAGREEMENT if self.disagreements else PASS

    def detail(self) -> str:
        parts = [*self.problems, *(f"DISAGREEMENT {d}" for d in self.disagreements)]
        return " | ".join([*parts, *(f"BLOCKED {b}" for b in self.blocked)])


# --------------------------------------------------------------------------------- journal


def journal_bytes(data: Path) -> dict[str, int]:
    """Size of every journal segment (the product's only lookup record)."""
    root = data / "accounting" / lp.PROFILE / "journal"
    return {p.name: p.stat().st_size for p in sorted(root.glob("*.jsonl"))} if root.is_dir() else {}


def _stdout_doc(call: obs.ObservedCall) -> dict[str, Any]:
    for line in call.stdout.strip().splitlines():
        if line.startswith("{"):
            try:
                doc = json.loads(line)
            except ValueError:
                return {}
            return doc if isinstance(doc, dict) else {}
    return {}


def classify_call(call: obs.ObservedCall) -> str:
    """``usage`` (pre-API, never journaled), ``ok``/``empty``/``error`` (journaled), else ``open``.

    Decided from the OBSERVED exit code and stdout only: a Click/usecli argument failure prints
    ``error.type == "BadParameter"``; an invalid metadata predicate also exits 2 but prints a
    ``condition`` and DOES enter the service API.
    """
    if call.state != "verified":
        return "open"
    doc = _stdout_doc(call)
    error = doc.get("error")
    if isinstance(error, dict) and error.get("type") in ("BadParameter", "UsageError"):
        return "usage"
    if call.exit_code == 0 and doc.get("ok") is not False:
        results = (doc.get("data") or {}).get("results")
        return "empty" if results == [] else "ok"
    return "empty" if call.exit_code == 3 else "error"


def journal_expectation(frozen: obs.Frozen) -> dict[str, int]:
    """What the product journal must hold for every observed, completed ssgrep execution."""
    out = {
        "search": 0,
        "show": 0,
        "note": 0,
        "ok": 0,
        "empty": 0,
        "error": 0,
        "usage": 0,
        "open": 0,
    }
    for call in frozen.calls:
        if (
            call.tool != "ssgrep"
            or len(call.argv) < 2
            or call.argv[1] not in ("search", "show", "note")
        ):
            continue
        kind = classify_call(call)
        out[kind] += 1
        if kind not in ("usage", "open"):
            out[call.argv[1]] += 1
    return out


def measured_values(collected: lp.Collected) -> dict[str, int | None]:
    return {k: v.get("value") for k, v in collected.measured.items() if isinstance(v, Mapping)}


# ----------------------------------------------------------------------------------- fixture


@dataclass
class Fixture:
    """The shared, expensive part: one root, one real index of one finished source session."""

    roots: LabRoots
    model_cache: Path
    base_data: Path
    base_view: Path
    base_verifier: Path
    shim_dir: Path
    source_file: Path
    counter: int = 0

    @classmethod
    def create(cls, workdir: Path, *, model_cache: Path) -> Fixture:
        roots = LabRoots.create(workdir)
        try:
            return cls._build(roots, model_cache)
        except BaseException:
            roots.remove()
            raise

    @classmethod
    def _build(cls, roots: LabRoots, model_cache: Path) -> Fixture:
        for sub in ("cases", "base"):
            (roots.artifacts / sub).mkdir(mode=0o700)
        view = roots.artifacts / "base" / "view"
        view.mkdir(mode=0o700)
        shim_dir = roots.tools / "solver"
        obs.install_shims(shim_dir, python=sys.executable, repo_root=lp.REPO_ROOT)
        fx = cls(
            roots,
            model_cache,
            roots.ssgrep_data,
            view,
            roots.verifier / "base-observer",
            shim_dir,
            view,
        )
        scene = Scene(fx, "base", fresh=True)
        ctl = scene.product
        lp.setup_profile(ctl)
        code, _, diagnostic = lp.index(ctl)  # an EMPTY index: capture needs a first journaled call
        if code != 0:
            raise InfrastructureError(f"initial index failed: {diagnostic} (exit {code})")
        call, _ = scene.lookup("control", "search", "early control probe", "--json")
        if call.state != "verified" or classify_call(call) != "empty":
            raise InfrastructureError("early control search did not complete as an empty lookup")
        scene.write_source()
        code, _, diagnostic = lp.index(ctl)
        if code != 0:
            raise InfrastructureError(f"source index failed: {diagnostic} (exit {code})")
        found, _ = scene.lookup("verifier", "search", QUERY, "--json")
        shown, _ = scene.lookup("verifier", "show", SOURCE_REF, "--json")
        if SOURCE_REF not in found.stdout or DEN not in shown.stdout:
            raise InfrastructureError("the indexed source session is not retrievable")
        fx.source_file = next(view.glob(f"*_{SOURCE_ID}.jsonl"))
        return fx

    def destroy(self) -> dict[str, Any]:
        return self.roots.remove()


class Scene:
    """One case's working copy: data dir, view dir, observer and consumer-session runner."""

    def __init__(self, fx: Fixture, name: str, *, trailer: str = "omp", fresh: bool = False):
        self.fx, self.name, self.trailer = fx, name, trailer
        fx.counter += 1
        self.dir = fx.roots.artifacts / ("base" if fresh else f"cases/{name}-{fx.counter}")
        self.data = fx.base_data if fresh else self.dir / "data"
        self.view = fx.base_view if fresh else self.dir / "view"
        self.verifier = fx.base_verifier if fresh else self.dir / "verifier"
        self.sessions = self.dir / "runtime-b"
        self.dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.sessions.mkdir(mode=0o700, exist_ok=True)
        if not fresh:
            ignore = shutil.ignore_patterns("*.lock")
            shutil.copytree(fx.base_data, self.data, ignore=ignore)
            shutil.copytree(fx.base_verifier, self.verifier)
            self.view.mkdir(mode=0o700)
            for path in fx.base_view.glob("*.jsonl"):
                shutil.copyfile(path, self.view / path.name)
        self.observer = obs.Observer(self.verifier, fx.roots.run_id)
        # This lane executes reviewed scripted steps, not untrusted native model actions.
        # Its supervised producer writes the reserved private spool used for attestation.
        self.spool = self.verifier / "solver-b" / "spool.ndjson"
        self.product = lp.Product(self.env("control"), fx.roots.project)
        self.linked: dict[str, Path] = {}
        self.copy_digests: dict[str, str] = {}
        self.native: dict[str, tuple[str, Path]] = {}  # session id -> (task, file)
        self.associations: dict[str | None, obs.NativeSession] = {}
        self.arith_count = 0

    # -- environments -----------------------------------------------------------------------
    def env(self, actor: str) -> dict[str, str]:
        return runtime_env(
            self.fx.roots,
            actor,
            runtime_sessions=self.view,
            model_cache=self.fx.model_cache,
            extra={"SSGREP_DATA_DIR": str(self.data)},
        )

    def shim_env(self, session_id: str) -> dict[str, str]:
        env = self.env("solver-b")
        env["PATH"] = f"{self.fx.shim_dir}{os.pathsep}{env['PATH']}"
        env.update(
            {
                obs.ENV_SPOOL: str(self.spool),
                obs.ENV_ACTOR: "solver-b",
                obs.ENV_RUN: self.fx.roots.run_id,
                obs.ENV_TASK_FILE: str(self.dir / f"task-{session_id}.txt"),
                obs.ENV_REAL + "SSGREP": str(lp.BINARY),
                "LAB_STANDIN_TRAILER": self.trailer,
            }
        )
        return env

    # -- observed control / verifier lookups (NOT solver exposures) ---------------------------
    def lookup(self, actor: str, *args: str) -> tuple[obs.ObservedCall, Any]:
        return self.observer.run(
            actor=actor,
            task=None,
            tool="ssgrep",
            argv=[str(lp.BINARY), *args],
            env=self.env("verifier"),
            cwd=self.fx.roots.project,
            timeout=600,
            shown=["ssgrep", *args],
        )

    # -- sessions ----------------------------------------------------------------------------
    def link(self, path: Path) -> Path:
        """Refresh a private collection copy without giving native files extra links."""
        target = self.view / path.name
        content = obs.read_source(
            path,
            self.sessions
            if path.parent == self.sessions
            else self.fx.roots.runtime_a / "home" / "omp-sessions",
        )
        temporary = target.with_suffix(".snapshot")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(content)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        self.linked[str(path)] = target
        self.copy_digests[str(path)] = obs.sha(content)
        return target

    def write_source(self) -> Path:
        from .standin_agent import Bash, Say

        steps: list[Step] = [
            Bash("printf 'rule_id minutes\\n%s-%s-045 45\\n' MCT DEN"),
            Bash("printf 'rule_id minutes\\n%s-%s-045 45\\n' MCT ORD"),
            Bash(f"echo {SFO}"),  # the call itself names the fact: known before its witness
            Say(SOURCE_SAY),
        ]
        env = runtime_env(
            self.fx.roots, "solver-a", runtime_sessions=self.view, model_cache=self.fx.model_cache
        )
        summary = run_turn(
            session_dir=self.fx.roots.runtime_a / "home" / "omp-sessions",
            session_id=SOURCE_ID,
            cwd=self.fx.roots.project,
            prompt=SOURCE_PROMPT,
            steps=steps,
            env=env,
            params={},
            task_file=None,
            task=None,
        )
        path = Path(summary["file"])
        self.link(path)
        self.observer.record_native(
            "solver-a",
            f"{SOURCE_ID}#0",
            path,
            source_root=self.fx.roots.runtime_a / "home" / "omp-sessions",
            source_aliases={str(self.linked[str(path)]): self.copy_digests[str(path)]},
        )
        return path

    def start_session(self, session_id: str, parent: Path | None = None) -> Path:
        """Create the native file (optionally declaring ``parentSession``) before its first turn."""
        writer = NativeWriter(self.sessions, session_id, str(self.fx.roots.project))
        if parent is not None:
            lines = writer.path.read_bytes().split(b"\n")
            header = json.loads(lines[0])
            header["parentSession"] = str(parent)
            lines[0] = json.dumps(header).encode()
            with writer.path.open("r+b") as handle:
                handle.write(b"\n".join(lines))
        self.link(writer.path)
        return writer.path

    def consume(
        self, session_id: str, prompt: str, steps: Sequence[Step], *, parent: Path | None = None
    ) -> Path:
        """One scripted consumer turn with REAL shell commands behind the observer shims."""
        existing = list(self.sessions.glob(f"*_{session_id}.jsonl"))
        path = existing[0] if existing else self.start_session(session_id, parent)
        task = f"{session_id}#0"
        run_turn(
            session_dir=self.sessions,
            session_id=session_id,
            cwd=self.fx.roots.project,
            prompt=prompt,
            steps=steps,
            env=self.shim_env(session_id),
            params={},
            task_file=self.dir / f"task-{session_id}.txt",
            task=task,
        )
        self.sync(session_id, path, task)
        return path

    def sync(self, session_id: str, path: Path, task: str) -> None:
        self.native[session_id] = (task, path)
        self.link(path)
        self.associations[task] = self.observer.record_native(
            "solver-b",
            task,
            path,
            source_root=self.sessions,
            source_aliases={str(self.linked[str(path)]): self.copy_digests[str(path)]},
        )
        self.observer.ingest_spool(
            self.spool,
            self.spool.parent / (self.spool.stem + "-blobs"),
            expected_actor="solver-b",
            expected_task=None,
            native_association=self.associations,
        )
        if self.spool.exists():
            self.observer.attest_supervisor_spool(
                self.spool,
                self.spool.parent / (self.spool.stem + "-blobs"),
                expected_actor="solver-b",
                expected_tasks=set(self.associations),
            )

    # -- collection and comparison -------------------------------------------------------------
    def frozen(self) -> obs.Frozen:
        self.observer.ingest_spool(
            self.spool,
            self.spool.parent / (self.spool.stem + "-blobs"),
            expected_actor="solver-b",
            expected_task=None,
            native_association=self.associations,
        )
        if self.spool.exists():
            self.observer.attest_supervisor_spool(
                self.spool,
                self.spool.parent / (self.spool.stem + "-blobs"),
                expected_actor="solver-b",
                expected_tasks=set(self.associations),
            )
        return self.observer.freeze()

    def collect(self, *flags: str) -> lp.Collected:
        return lp.collect(self.product, *flags)

    def arithmetic(self) -> tuple[lp.Collected, lp.Product]:
        """Fresh isolated journal-only copy, test view pinned, collected there (never summed)."""
        self.arith_count += 1
        dest = self.dir / f"arith-{self.arith_count}"
        lp.copy_accounting(self.data, dest, journal_only=True)
        product = lp.Product({**self.product.env, "SSGREP_DATA_DIR": str(dest)}, self.product.cwd)
        lp.pin_arithmetic_view(product, [t for _, t in REVIEWED])
        return lp.collect(product), product

    def evidence(self, *snapshots: lp.Collected) -> tuple[str, ...]:
        """Observed solver ssgrep call ids, native tool-call record ids and snapshot ids only."""
        frozen = self.frozen()
        calls = [c.call_id for c in frozen.calls if c.actor == "solver-b" and c.tool == "ssgrep"]
        records = [
            f"{sid}:{r.record_id}"
            for sid, (actor, _task, native) in frozen.sessions.items()
            if actor == "solver-b"
            for r in native.records
            if r.kind == "tool_call" and "ssgrep" in r.text
        ]
        snaps = [f"snapshot:{snapshot_id(c)[:16]}" for c in snapshots]
        return (*calls, *records, *snaps)

    def product_for_profile(self, profile: oracle.ProfileSpec) -> lp.Product:
        if profile.name != oracle.ARITHMETIC:
            return self.product
        scoped_data = self.dir / f"arith-{self.arith_count}"
        return lp.Product(
            {**self.product.env, "SSGREP_DATA_DIR": str(scoped_data)}, self.product.cwd
        )

    def savings_for_session(self, profile: oracle.ProfileSpec, session: str) -> dict[str, Any]:
        code, document = self.product_for_profile(profile).json("savings", "--session", session)
        if code != 0:
            raise InfrastructureError(f"session-scoped savings failed ({code}) for {session}")
        return document

    def compare(
        self, profile: oracle.ProfileSpec, collected: lp.Collected, skip: Sequence[str] = ()
    ) -> list[str]:
        """Compare matching consumer roots; verifier/control history is not solver debit."""
        frozen = self.frozen()
        entries = {
            native.session_id: (label, native)
            for actor, label, native in frozen.sessions.values()
            if actor == "solver-b" and native.session_id in self.native
        }
        parents: dict[str, str] = {}
        for child_id, (_child_label, child) in entries.items():
            parent_path = child.parent_session
            if not isinstance(parent_path, str):
                continue
            for parent_id, (_parent_label, parent) in entries.items():
                same = os.path.abspath(parent_path) in {
                    os.path.abspath(path) for path in (parent.path, *parent.source_aliases)
                }
                if same:
                    parents[child_id] = parent_id
                    break
        scopes = []
        for session_id, (label, native) in entries.items():
            host_ids = {
                record.call_id
                for record in native.records
                if record.kind == "tool_call" and record.call_id
            }
            if any(
                call.actor == "solver-b"
                and call.task == label
                and call.native_call_id in host_ids
                and call.tool == "ssgrep"
                and len(call.argv) > 1
                and call.argv[1] in ("search", "show")
                and classify_call(call) not in ("usage", "open")
                for call in frozen.calls
            ):
                scopes.append(session_id)
        product = self.product_for_profile(profile)
        mismatches: list[str] = []
        for scope in scopes:
            members = {scope}
            parent = parents.get(scope)
            while parent is not None and parent not in members:
                members.add(parent)
                parent = parents.get(parent)
            sessions = {
                key: entry
                for key, entry in frozen.sessions.items()
                if entry[0] == "solver-a"
                or (entry[0] == "solver-b" and entry[2].session_id in members)
            }
            native_calls = {
                (actor, label, record.call_id)
                for actor, label, native in sessions.values()
                for record in native.records
                if record.kind == "tool_call" and record.call_id
            }
            calls = tuple(
                call
                for call in frozen.calls
                if (call.actor, call.task, call.native_call_id) in native_calls
            )
            scoped = replace(frozen, calls=calls, sessions=sessions)
            result = oracle.evaluate(scoped, profile)
            code, document = product.json("savings", "--session", scope)
            if code != 0:
                mismatches.append(f"session {scope}: savings exited {code}")
                continue
            found = oracle.compare_with_snapshot(result, document)
            mismatches.extend(
                f"session {scope}: {m}"
                for m in found
                if not m.startswith("measured.") and not any(f".{field}:" in m for field in skip)
            )
        return mismatches

    def oracle_result(self, profile: oracle.ProfileSpec) -> oracle.OracleResult:
        return oracle.evaluate(self.frozen(), profile)


def snapshot_id(collected: lp.Collected) -> str:
    body = collected.collect.get("data", collected.collect)
    return str(body.get("snapshot_id", "?"))


def production() -> oracle.ProfileSpec:
    return oracle.production_profile(corpus_docs=TINY_CORPUS)


def arithmetic() -> oracle.ProfileSpec:
    return oracle.arithmetic_profile(REVIEWED)


def estimate(collected: lp.Collected, scenario: str = "central") -> dict[str, int | None]:
    row = (collected.estimates.get(oracle.UNIT) or {}).get(scenario) or {}
    return {k: (row.get(k) or {}).get("value") for k in ("credit", "debit", "net")}


def oracle_estimate(
    result: oracle.OracleResult, scenario: str = "central"
) -> dict[str, int | None]:
    est = result.estimates[scenario]
    return {"credit": est.credit, "debit": est.debit, "net": est.net}


def finish(
    name: str,
    check: Check,
    expected: dict[str, Any],
    actual: dict[str, Any],
    evidence: Sequence[str],
    note: str = "",
) -> CaseResult:
    detail = " | ".join(x for x in (check.detail(), note) if x)
    return CaseResult(name, check.status(), expected, actual, tuple(evidence), detail)


# ------------------------------------------------------------------------------------ drive


def run_case(fixture: Fixture, name: str) -> CaseResult:
    from . import sensitivity_cases

    if name not in CASES:
        raise KeyError(f"unknown sensitivity case {name!r}")
    return sensitivity_cases.REGISTRY[name](fixture)


def run_all(
    workdir: Path, *, model_cache: Path, emit: Callable[[CaseResult], None] | None = None
) -> list[CaseResult]:
    """Build the shared fixture in ``workdir``, run every case, tear down what was created."""
    fixture = Fixture.create(workdir, model_cache=model_cache)
    results: list[CaseResult] = []
    try:
        for name in CASES:
            try:
                row = run_case(fixture, name)
            except InfrastructureError:
                raise
            except Exception as exc:  # noqa: BLE001 - a crashed case is a FAIL row, never silent
                row = CaseResult(name, FAIL, {}, {}, (), f"{type(exc).__name__}: {exc}"[:400])
            results.append(row)
            if emit is not None:
                emit(row)
    finally:
        fixture.destroy()
    return results
