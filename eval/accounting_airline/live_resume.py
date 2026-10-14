"""Continue a retained, completed native A after a harness-only grading correction.

Never invokes A again. The read-only A source must match its recorded digest. A fresh
owned database is recreated from identical pinned seed/schema, and B remains a fresh
conversation at the original canonical project path. Original reports are archived intact.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import fields
from pathlib import Path

from . import lab_run
from .checkpoints import Assertion, Checkpoint
from .domain_lane import write_private
from .gates import FAIL, PASS
from .lab_checks import check, gate_status_of
from .lab_session import grade_l2, tagged_tables
from .live_auth import AuthGateway
from .live_bridge import Bridge
from .live_checks import prove_bridge, prove_native_tools
from .live_grading import normalize_tables
from .live_native import LiveBlocked, seal_kit
from .live_persist import stage_c3
from .live_prompts import prompt_for
from .live_run import _archive_project, _close, _isolation
from .live_security import live_profile
from .live_stages import stage_b_l3, stage_b_l4
from .manifest import CredentialRef, build_manifest
from .observer import parse_native_omp
from .report import GateEntry, Report
from .roots import LabRoots
from .runtime_omp import verify_protocol
from .sandbox import account_home


def _checkpoint(document: dict) -> Checkpoint:
    allowed = {field.name for field in fields(Checkpoint)}
    values = {key: value for key, value in document.items() if key in allowed}
    assertion_keys = {field.name for field in fields(Assertion)}
    for kind in ("domain", "accounting"):
        values[kind] = [
            Assertion(**{key: value for key, value in item.items() if key in assertion_keys})
            for item in values.get(kind, [])
        ]
    return Checkpoint(**values)


def resume_live(
    root: Path, emit, *, authorized: bool, model: str, thinking: str | None = None
) -> int:
    if not authorized:
        raise ValueError("live authorization is required for continuation")
    root = root.resolve()
    roots = LabRoots(root, root.name.removeprefix("ssgrep-airline-lab-"))
    roots.verify_ownership()
    evidence = json.loads((roots.artifacts / "live-evidence.json").read_text())
    original_manifest = json.loads((roots.artifacts / "manifest.json").read_text())
    original_report = json.loads((roots.artifacts / "report.json").read_text())
    if set(evidence.get("turns", {})) != {"L1", "L2"} or evidence["native_ids"].get("b"):
        raise LiveBlocked("continuation requires exactly completed A L1/L2 and no B attempt")
    if original_manifest["model"] != model:
        raise LiveBlocked("continuation cannot change the chosen native model")
    if thinking is not None and thinking != original_manifest["sampling"].get("thinking", "low"):
        raise LiveBlocked("continuation cannot change the pinned thinking effort")
    source = Path(evidence["turns"]["L2"]["native"]["session_files"][0])
    native = parse_native_omp(source)
    expected_digest = next(iter(evidence["turns"]["L2"]["native"]["file_digests"].values()))
    if native.file_digest != expected_digest or native.partial_tail_bytes:
        raise LiveBlocked("retained A native source changed or is incomplete")
    if list((roots.runtime_b / "home/omp-sessions").rglob("*.jsonl")):
        raise LiveBlocked("B already has native history; refusing a second B attempt")
    stamp = str(time.time_ns())
    prior = roots.verifier / f"attempt-before-correction-{stamp}"
    shutil.copytree(roots.artifacts, prior, ignore=shutil.ignore_patterns("collect-view"))
    prior.chmod(0o700)
    old_gateway = roots.verifier / "auth-gateway"
    if old_gateway.exists():
        old_gateway.rename(roots.verifier / f"auth-gateway-before-correction-{stamp}")
    started = time.monotonic()
    cache = account_home() / ".cache/huggingface"
    run = lab_run.build_run(roots, emit, cache)
    run.report = Report(roots.run_id, "live", "omp", True, "omp-live")
    run.native["a"], run.a_sid, run.b_sid = source, native.session_id, "not-started-b"
    run.notes["turns"] = evidence["turns"]
    run.observer.attest_supervisor_spool(
        roots.verifier / "solver-a/spool.ndjson",
        roots.verifier / "solver-a/spool-blobs",
        expected_actor="solver-a",
        expected_tasks={"A/L1", "A/L2"},
    )
    run.notes["index_a"] = {
        "ms": original_report["checkpoints"]["C3"]["actual_delta"].get("index_ms")
    }
    for group, outcome in original_report["gates"].items():
        for entry in outcome["entries"]:
            if entry["name"].startswith("lane:") or entry["name"] in (
                "native.L1",
                "seed.integrity",
            ):
                run.report.add_gate(group, GateEntry(**entry))
    for cid in ("C0", "C1", "C2"):
        checkpoint = _checkpoint(original_report["checkpoints"][cid])
        if cid == "C2":
            tables = normalize_tables(
                tagged_tables(run.frozen(), run.observer, "solver-a", task="A/L2")
            )
            checkpoint.domain = [
                check(
                    f"l2.{name}",
                    errors,
                    expected="independent oracle rows",
                    actual=tables,
                    method="regrade immutable native SQL; integral DECIMAL is an integer",
                )
                for name, errors in grade_l2(tables, run.base.state).items()
            ]
            checkpoint.comparison_method += (
                "; original numeric-parser failure preserved in prior report"
            )
            run.report.add_gate(
                "l2",
                GateEntry(
                    "native.L2.regraded",
                    gate_status_of(checkpoint.domain),
                    "No new model attempt; original failed harness report at "
                    f"{prior / 'report.json'}",
                ),
            )
        run.report.checkpoints.add(checkpoint)
    error, gateway = None, None
    try:
        if (
            run.manifest.seed_digest != original_manifest["seed_digest"]
            or run.manifest.schema_digest != original_manifest["schema_digest"]
        ):
            raise LiveBlocked("recreated baseline fixture/schema differs from original A")
        write_private(
            roots.artifacts / "continuation.json",
            {
                "prior_report": str(prior / "report.json"),
                "native_a_sha256": native.file_digest,
                "a_model_calls_repeated": 0,
                "project": str(roots.project),
                "database": "fresh owned clone of identical pinned baseline; A was read-only",
                "harness_corrections": [
                    "exact integral SQL DECIMAL normalization",
                    "Markdown-independent source facts",
                ],
                "preserved_l1_domain_outcome": original_report["gates"]["l1"],
            },
        )
        run.manifest = build_manifest(
            run_id=roots.run_id,
            mode="live",
            runtime="omp",
            roots=roots,
            authorize_live=True,
            model=model,
            provider=model.split("/", 1)[0],
            runtime_version=verify_protocol().version,
            sampling=original_manifest["sampling"],
            prompts={
                **{
                    key: json.loads((prior / "prompts" / f"{key}.json").read_text())["prompt"]
                    for key in ("L1", "L2")
                },
                **{key: prompt_for(key, run.b_db) for key in ("L3", "L4")},
            },
            limitations=live_profile(roots, "solver-b").limitations,
            credentials=(
                CredentialRef(
                    "omp-provider",
                    str(account_home() / ".omp/agent/agent.db"),
                    "supervised-readonly-gateway",
                ),
            ),
        )
        run.manifest.write(roots.artifacts / "manifest.json")
        for kit in (run.kit_a, run.kit_b):
            seal_kit(run, kit)
        gateway = AuthGateway(roots, model)
        gateway.__enter__()
        run.notes["auth_gateway"] = gateway
        run.secrets.extend(gateway.capabilities.values())
        boundary = prove_bridge(run)
        run.report.add_gate(
            "b_isolation", GateEntry(boundary.name, boundary.status, boundary.detail)
        )
        if boundary.status != PASS:
            raise LiveBlocked(boundary.detail)
        tools = prove_native_tools(run)
        write_private(roots.artifacts / "native-tools-isolation.json", tools.as_dict())
        run.report.add_gate("b_isolation", GateEntry(tools.name, tools.status, tools.detail))
        if tools.status != PASS:
            raise LiveBlocked(tools.detail)
        if not stage_c3(run)["ok"]:
            raise LiveBlocked("corrected semantic persistence barrier did not pass")
        _archive_project(run)
        _isolation(run, before_b=True)
        with Bridge(run.kit_b, run.b_db, roots.project) as bridge:
            plan = stage_b_l3(run, bridge, model)
            stage_b_l4(run, bridge, model, plan)
        for stage in (lab_run.stage_c8, lab_run.stage_c9, lab_run.stage_c10):
            try:
                stage(run)
            except Exception as exc:
                error = f"{stage.__name__}: {type(exc).__name__}: {exc}"
                run.report.add_gate(
                    "accounting_checkpoints", GateEntry(stage.__name__, FAIL, error)
                )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        write_private(roots.artifacts / "continuation-blocked.json", {"prerequisite": error})
        run.say(f"[BLOCKED] {error}")
    finally:
        if gateway is not None:
            gateway.__exit__(None, None, None)
        result = _close(run, error, started)
    return result
