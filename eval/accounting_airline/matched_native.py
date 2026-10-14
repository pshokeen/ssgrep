"""Real isolated paired-arm execution; never used implicitly by the main live lab."""

from __future__ import annotations

import os
from pathlib import Path

from . import dataset, lab_run
from .compare import Measurement
from .database import ADMIN_DB
from .domain_lane import write_private
from .gates import FAIL, PASS
from .lab_product import index
from .lab_session import grade_l3, grade_l4, tagged_tables
from .live_auth import AuthGateway
from .live_grading import normalize_tables
from .live_native import LiveBlocked, seal_kit
from .live_security import live_profile
from .matched_bridge import MatchedBridge
from .matched_security import install_source, remove_credentials
from .roots import runtime_env
from .runtime_omp import TurnSpec, run_turn, verify_protocol
from .sandbox import SandboxProfile, account_home, sbpl_quote


def prepare(roots, emit):
    run = None
    try:
        run = lab_run.build_run(roots, emit, account_home() / ".cache/huggingface")
        run.instance.psql(ADMIN_DB, f"ALTER DATABASE {run.b_db} RENAME TO airline;")
        run.b_db = "airline"
        state = run.instance.read_state(run.b_db)
        if dataset.overall_digest(dataset.logical_digests(state)) != run.base.digest:
            raise LiveBlocked("matched.clone_integrity: actual B rows differ from sealed baseline")
        seal_kit(run, run.kit_b)
        return run
    except BaseException:
        try:
            if run is not None:
                run.instance.teardown()
        finally:
            remove_credentials(roots)
        raise


def profile_for(run, sealed):
    base = live_profile(run.roots, "solver-b")
    additions = tuple(Path(p).resolve() for p in sealed)
    text = base.text
    text += "".join(f"(deny file-read* file-write* (subpath {sbpl_quote(p)}))\n" for p in additions)
    return SandboxProfile(text, (*base.denied, *additions), base.limitations)


def execute(run, *, source, source_digest, source_sid, history, prompts, model, runtime, sealed):
    if verify_protocol().version != runtime:
        raise LiveBlocked("matched.runtime_pin: runtime version changed")
    run.a_sid = source_sid
    if history:
        install_source(run.roots, source, source_digest)
        code, _, detail = index(run.product)
        if code:
            raise LiveBlocked(f"matched.history_index: {detail}")
    # No collect/reindex occurs after this point: B can never enter the history corpus.
    profile = profile_for(run, sealed)
    results = []
    with AuthGateway(run.roots, model) as gateway:
        run.secrets.extend(gateway.capabilities.values())
        with MatchedBridge(run.kit_b, run.b_db, run.roots.project, history=history) as bridge:
            clients = run.roots.tools / "solver-b"
            bridge.install_clients(clients)
            env = runtime_env(
                run.roots,
                "solver-b",
                runtime_sessions=run.kit_b.sessions_dir,
                model_cache=run.cache,
            )
            gateway.configure("solver-b", env)
            env["PATH"] = str(clients) + os.pathsep + env["PATH"]
            scratch = run.roots.actor_home("solver-b") / "tmp"
            scratch.mkdir(mode=0o700, exist_ok=True)
            env["TMPDIR"] = str(scratch)
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            before = run.instance.read_state(run.b_db)
            for task in ("L3", "L4"):
                bridge.writer = task == "L4"
                run.kit_b.task_file.write_text(f"B/{task}")
                spec = TurnSpec(
                    prompts[task],
                    run.kit_b.sessions_dir,
                    run.roots.project,
                    ("bash",),
                    model=model,
                    continue_session=task == "L4",
                    max_time="10m",
                    thinking="low",
                    extra_flags=("--auto-approve", "--no-lsp", "--no-pty"),
                )
                output = run.roots.artifacts / "native-attempts" / task
                done = run_turn(
                    spec, env=env, profile=profile, timeout=620, authorized=True, output_dir=output
                )
                write_private(output / "result.json", done.to_dict())
                results.append(done)
                native = done.native
                if len(native.session_files) != 1 or not native.session_id:
                    raise LiveBlocked(f"matched.{task}.native_source: missing unique conversation")
                if native.session_id == source_sid or (
                    task == "L4" and native.session_id != run.b_sid
                ):
                    raise LiveBlocked("matched.native_identity: conversation isolation failed")
                if not native.usage_records or any(native.partial_tail_bytes.values()):
                    raise LiveBlocked(f"matched.{task}.native_usage: incomplete native evidence")
                if not native.models or any(
                    m not in (model, model.split("/", 1)[-1]) for m in native.models
                ):
                    raise LiveBlocked(f"matched.{task}.model_pin: native model differs from plan")
                run.b_sid = native.session_id
                association = run.observer.record_native(
                    "solver-b",
                    f"B/{task}",
                    native.session_files[0],
                    source_root=run.kit_b.sessions_dir,
                )
                kit = run.kit_b
                run.observer.ingest_spool(
                    kit.spool,
                    kit.spool.parent / (kit.spool.stem + "-blobs"),
                    expected_actor="solver-b",
                    expected_task=f"B/{task}",
                    native_association=association,
                )
                if task == "L3":
                    tables = normalize_tables(
                        tagged_tables(run.frozen(), run.observer, "solver-b", task="B/L3")
                    )
                    plan = {
                        r["booking_id"]: r["legs"].strip("{}").split(",")
                        for r in tables.get("plan", [])
                        if "booking_id" in r and "legs" in r
                    }
                    l3 = grade_l3(tables, plan, run.base.state)
            after = run.instance.read_state(run.b_db)
            l4 = grade_l4(before, after, plan)
    calls = [c for c in run.frozen().calls if c.actor == "solver-b" and c.task == "B/L4"]
    replay = any('"replayed": true' in c.stdout or '"replayed":true' in c.stdout for c in calls)
    correctness = {
        "l3": PASS if not any(l3.values()) else FAIL,
        "l4": PASS if not l4 and replay else FAIL,
        "native": PASS
        if all(d.returncode == 0 and not d.timed_out for d in results)
        and results[-1].native.user_turns == 2
        else FAIL,
    }
    final = results[-1].native
    measured = {
        f"tokens.{k}": Measurement(v, "tokens", "reported") for k, v in final.usage_total.items()
    }
    measured["elapsed"] = Measurement(sum(d.elapsed_ms for d in results), "ms", "observed")
    measured["runtime_catalog_cost"] = Measurement(
        final.cost_nano_usd,
        "nano_usd",
        "estimated" if final.cost_nano_usd is not None else "unavailable",
        "Runtime catalog estimate, not provider billing"
        if final.cost_nano_usd is not None
        else "native runtime did not report catalog cost",
    )
    write_private(run.roots.artifacts / "domain.json", {"l3": l3, "l4": l4, "replay": replay})
    return correctness, measured
