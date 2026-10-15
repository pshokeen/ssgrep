"""Authorized native airline lab. Every attempt and every non-PASS gate stays private.

No retries, no synthetic native transcript, no copying A answers into B. The deterministic
scenario lane is labeled separately from native SQL and cannot satisfy the live gate.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Callable
from pathlib import Path

from . import lab_run, live_metrics
from .domain_lane import write_private
from .gates import BLOCKED, FAIL, NOT_RUN, PASS, Gate
from .lab_checks import equal
from .lab_context import Run, Stage, dig
from .lab_product import REPO_ROOT, index, setup_profile
from .lab_stages_a import gate_assertion, stage_c0
from .live_bridge import Bridge
from .live_checks import prove_bridge, prove_native_tools
from .live_native import READER_ENV, LiveBlocked, seal_kit
from .live_persist import stage_c3
from .live_prompts import prompt_for
from .live_security import live_profile, prove_live_isolation
from .live_stages import stage_a, stage_b_l3, stage_b_l4
from .manifest import CredentialRef, build_manifest, preflight_live
from .report import GateEntry, Report, gate_table
from .roots import LabRoots
from .runtime_omp import verify_protocol
from .sandbox import account_home

DEFAULT_MODEL = "openai-codex/gpt-6-luna"


def _source_pins() -> dict[str, str]:
    paths = [
        *REPO_ROOT.joinpath("src").rglob("*.py"),
        *REPO_ROOT.joinpath("eval/accounting_airline").rglob("*.py"),
    ]
    return {
        str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def _check_sources(run: Run) -> None:
    if _source_pins() != run.notes["source_pins"]:
        raise LiveBlocked("shared source changed after preregistration; no further native turns")


def _archive_project(run: Run) -> None:
    """No shared answer files survive A; B sees the same path, empty public workspace."""
    archive = run.roots.verifier / "a-workspace"
    archive.mkdir(mode=0o700)
    for child in run.roots.project.iterdir():
        child.rename(archive / child.name)


def _isolation(run: Run, *, before_b: bool) -> None:
    profile = live_profile(run.roots, "solver-b")
    probe_source = run.native.get("a")
    if probe_source is None:
        probe_source = run.roots.runtime_a / "preflight-native-canary.txt"
        probe_source.write_text("private A native source boundary canary")
    gate = prove_live_isolation(run.roots, profile, probe_source)
    write_private(
        run.roots.artifacts / ("b-isolation.json" if before_b else "preflight-isolation.json"),
        gate.as_dict(),
    )
    run.report.add_gate("b_isolation", GateEntry(gate.name, gate.status, gate.detail))
    if gate.status != PASS:
        raise LiveBlocked(f"sandbox prerequisite: {gate.detail}")
    run.profile, run.sandbox_ok = profile, True
    if before_b:
        stage = Stage(run, "C4", "B isolation before native launch")
        stage.add_domain(
            gate_assertion(gate, "actual denied/allowed reads and credential write denial"),
            equal(
                "b.empty_native_state",
                [],
                list(run.kit_b.sessions_dir.glob("*.jsonl")),
                method="fresh native runtime-b directory",
            ),
            equal(
                "b.baseline_unchanged",
                run.base.digests,
                __import__(
                    "eval.accounting_airline.dataset", fromlist=["logical_digests"]
                ).logical_digests(run.instance.read_state(run.b_db)),
                method="independent state hashes",
            ),
        )
        run.collect("c4")
        stage.add_accounting(
            equal(
                "b.no_native_usage_yet",
                False,
                "b" in run.native,
                method="no B model invocation before isolation proof",
            )
        )
        stage.finish("native B L3, new conversation at same canonical project")


def native_turns_completed(turns: dict) -> bool:
    """Acceptance follows the supervisor status, including future stop reasons."""
    return set(turns) == {"L1", "L2", "L3", "L4"} and all(
        turn["returncode"] == 0
        and turn.get("capture_status") == "completed"
        and all(
            attempt.get("capture_status") == "completed"
            for attempt in turn.get("bridge_attempts", [])
        )
        for turn in turns.values()
    )


def _close(run: Run, error: str | None, started: float) -> int:
    run.observer.finalize()
    try:
        run.instance.teardown()
        run.report.add_gate(
            "cleanup",
            GateEntry(
                "owned_database_removed", PASS, "Private evidence retained; only owned DB stopped"
            ),
        )
    except Exception as exc:
        run.report.add_gate("cleanup", GateEntry("owned_database_removed", FAIL, str(exc)))
    # Retain native evidence, but remove synthetic database passwords. Provider
    # credentials never entered a run-owned file; gateway capabilities have expired.
    for kit in (run.kit_a, run.kit_b):
        from .observer import ENV_SQL_PGPASS

        Path(kit.env[ENV_SQL_PGPASS]).unlink(missing_ok=True)
    live_metrics.fill(run)
    lab_run.valuation_payloads(run)
    for task, group in (("L1", "l1"), ("L2", "l2"), ("L3", "l3"), ("L4", "l4_scenarios")):
        if task not in run.notes.get("turns", {}):
            run.report.add_gate(group, GateEntry(f"native.{task}", NOT_RUN, error or "not reached"))
    for cid, item in run.report.checkpoints.as_dict().items():
        run.report.add_gate("accounting_checkpoints", GateEntry(cid, item["status"], ""))
    turns = run.notes.get("turns", {})
    live_ok = native_turns_completed(turns) and run.a_sid != run.b_sid and not error
    run.report.add_gate(
        "live_session_gate",
        GateEntry(
            "native.two_conversations_four_turns",
            PASS if live_ok else BLOCKED,
            error or "All native attempts retained; no retries",
        ),
    )
    write_private(
        run.roots.artifacts / "live-evidence.json",
        {
            "runtime": "omp-live",
            "native_ids": {
                "a": run.a_sid if "a" in run.native else None,
                "b": run.b_sid if "b" in run.native else None,
            },
            "turns": turns,
            "source_resolution": run.notes.get("resolution_a"),
            "cost_provenance": "Native OMP catalog-derived cost estimate for Codex, "
            "not provider billing; no ssgrep repricing",
            "blocked_prerequisite": error,
            "elapsed_s": time.monotonic() - started,
            "private_root_retained": str(run.roots.root),
        },
    )
    run.report.write_private(run.roots.artifacts / "report.json")
    run.report.checkpoints.write(run.roots.artifacts / "checkpoints.json")
    run.say(run.report.checkpoints.table())
    run.say(gate_table(run.report))
    run.say(f"private_artifacts={run.roots.artifacts} native_sources_retained={run.roots.root}")
    statuses = [run.report.gate_status(g) for g in run.report.gates]
    return 1 if FAIL in statuses else (3 if any(s != PASS for s in statuses) else 0)


def run_live(
    parent: Path | None,
    emit: Callable[[str], None],
    *,
    authorized: bool,
    runtime: str = "omp",
    model: str = DEFAULT_MODEL,
    thinking: str = "low",
) -> int:
    if not authorized:
        raise ValueError("live authorization is required")
    roots = LabRoots.create(parent)
    started = time.monotonic()
    emit(
        f"run_id={roots.run_id} private_root={roots.root} "
        f"model={model} thinking={thinking} max_time=10m/turn"
    )
    cache = Path(os.environ.get("HF_HOME") or account_home() / ".cache/huggingface")
    credential = account_home() / ".omp/agent/agent.db"
    gates = preflight_live(
        runtime=runtime, authorize_live=True, model_cache=cache, credential_paths=(credential,)
    )
    from .live_auth import catalog_metadata

    catalog = None
    try:
        catalog = catalog_metadata(model)
        efforts = catalog["model"].get("thinking", {}).get("efforts", [])
        if thinking not in efforts:
            raise ValueError(f"thinking {thinking!r} not supported by selected model: {efforts}")
        gates.append(Gate("live.model_catalog", PASS, f"{model}; thinking={thinking}"))
    except (OSError, ValueError, KeyError) as exc:
        gates.append(Gate("live.model_catalog", BLOCKED, str(exc)))
    for gate in gates:
        emit(f"[{gate.status}] {gate.name}: {gate.detail}")
    write_private(roots.artifacts / "preflight.json", [g.as_dict() for g in gates])
    failures = [g for g in gates if g.status != PASS]
    if failures:
        report = Report(roots.run_id, "live", "omp", True, "omp-live")
        report.add_gate(
            "live_session_gate",
            GateEntry("preflight", BLOCKED, "; ".join(f"{g.name}: {g.detail}" for g in failures)),
        )
        report.write_private(roots.artifacts / "report.json")
        emit(gate_table(report))
        return 3
    protocol = verify_protocol()
    run = lab_run.build_run(roots, emit, cache)
    run.report = Report(roots.run_id, "live", "omp", True, "omp-live")
    run.a_sid, run.b_sid = "not-started-a", "not-started-b"
    run.manifest = build_manifest(
        run_id=roots.run_id,
        mode="live",
        runtime="omp",
        roots=roots,
        authorize_live=True,
        runtime_version=protocol.version,
        model=model,
        provider=model.split("/", 1)[0],
        sampling={"thinking": thinking, "max_time": "10m", **READER_ENV},
        credentials=(
            CredentialRef("omp-provider", str(credential), "supervised-readonly-gateway"),
        ),
        prompts={
            task: prompt_for(task, run.a_db if task in ("L1", "L2") else run.b_db)
            for task in ("L1", "L2", "L3", "L4")
        },
        limitations=live_profile(roots, "solver-b").limitations,
    )
    run.manifest.write(roots.artifacts / "manifest.json")
    run.notes["source_pins"] = _source_pins()
    write_private(
        roots.artifacts / "preregistration.json",
        {
            "model_catalog": catalog,
            "sampling": dict(run.manifest.sampling),
            "prompts": {
                task: prompt_for(task, run.a_db if task in ("L1", "L2") else run.b_db)
                for task in ("L1", "L2", "L3", "L4")
            },
            "prompt_hashes": dict(run.manifest.prompt_hashes),
            "baseline_digests": run.base.digests,
            "source_sha256": run.notes["source_pins"],
            "native_limits": {
                "user_turns": 4,
                "max_time": "10m",
                "timeout_seconds": 620,
                "gateway_lifetime_seconds": 3600,
                "gateway_requests_per_actor": 64,
            },
            "reader_environment": READER_ENV,
            "sandbox": {
                actor: live_profile(roots, actor).text for actor in ("solver-a", "solver-b")
            },
            "criteria": "Correct L1-L4, real A persistence/restart, fresh B exposure and supported "
            "later action, independent accounting at every checkpoint; no retries",
        },
    )
    error = None
    gateway = None
    try:
        for kit in (run.kit_a, run.kit_b):
            seal_kit(run, kit)
        from .live_auth import AuthGateway

        gateway = AuthGateway(roots, model)
        gateway.__enter__()
        run.notes["auth_gateway"] = gateway
        run.secrets.extend(gateway.capabilities.values())
        _isolation(run, before_b=False)
        boundary = prove_bridge(run)
        write_private(roots.artifacts / "bridge-isolation.json", boundary.as_dict())
        run.report.add_gate(
            "b_isolation", GateEntry(boundary.name, boundary.status, boundary.detail)
        )
        if boundary.status != PASS:
            raise LiveBlocked(f"supervisor boundary prerequisite: {boundary.detail}")
        tools = prove_native_tools(run)
        write_private(roots.artifacts / "native-tools-isolation.json", tools.as_dict())
        run.report.add_gate("b_isolation", GateEntry(tools.name, tools.status, tools.detail))
        if tools.status != PASS:
            raise LiveBlocked(f"native tool boundary prerequisite: {tools.detail}")
        # Real concurrency/rollback fixtures run as the VERIFIER, never B's answer.
        lab_run._lane(roots.artifacts, run.report, emit)
        run.notes["setup"] = setup_profile(run.product)
        run.notes["pre_collect_state"] = dig(run.product.json("savings")[1], "state")
        code, _ms, detail = index(run.product)
        if code:
            raise LiveBlocked(f"isolated offline empty index failed ({code}): {detail}")
        run.collect("initial")
        stage_c0(run, "separate real PostgreSQL deterministic verifier scenarios")
        with Bridge(run.kit_a, run.a_db, roots.project) as bridge:
            _check_sources(run)
            stage_a(run, bridge, "L1", model)
            _check_sources(run)
            stage_a(run, bridge, "L2", model)
        persisted = stage_c3(run)
        # The shared stage records failures but its return's ok flag is not a gate verdict.
        if not persisted.get("ok") or any(
            run.report.gate_status(g) != PASS for g in ("a_native_persistence", "restart_retrieval")
        ):
            raise LiveBlocked("A persistence/source evidence barrier did not PASS; B not started")
        _archive_project(run)
        _isolation(run, before_b=True)
        with Bridge(run.kit_b, run.b_db, roots.project) as bridge:
            _check_sources(run)
            plan = stage_b_l3(run, bridge, model)
            _check_sources(run)
            stage_b_l4(run, bridge, model, plan)
        _check_sources(run)
        for stage in (lab_run.stage_c8, lab_run.stage_c9, lab_run.stage_c10):
            try:
                stage(run)
            except Exception as exc:
                error = f"{stage.__name__}: {type(exc).__name__}: {exc}"
                run.say(f"[FAIL] {error}")
                run.report.add_gate(
                    "accounting_checkpoints", GateEntry(stage.__name__, FAIL, error)
                )
        _check_sources(run)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        write_private(roots.artifacts / "blocked.json", {"prerequisite": error})
        run.say(f"[BLOCKED] {error}")
    finally:
        if gateway is not None:
            gateway.__exit__(None, None, None)
        result = _close(run, error, started)
    return result
