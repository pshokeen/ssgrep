"""Opt-in real paired comparison, never A-versus-B savings.

Run ``python -m eval.accounting_airline.matched --help``. Each invocation is a new
immutable plan; failed/interrupted slots are retained and never retried in place.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
from datetime import UTC, datetime
from pathlib import Path

from . import matched_native, prompts
from .compare import WITH_HISTORY, Attempt, build_plan, compare
from .domain_lane import write_private
from .gates import BLOCKED, PASS
from .live_native import LiveBlocked
from .live_prompts import CONTRACTS, L3_DOMAIN_CONTRACT
from .matched_security import close_run, freeze_source, remove_credentials
from .observer import _parse_native_bytes
from .roots import LabRoots
from .runtime_omp import verify_protocol


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def matched_prompts():
    # Share the domain contract without requiring history in the no-history arm.
    objective = "Produce a joint plan" + prompts.L3.split("produce a joint plan", 1)[1]
    contract = L3_DOMAIN_CONTRACT
    common = (
        prompts.BRIEF + "\nAssigned database: airline. Use bash and installed commands "
        "sql --role reader|writer --database airline with SQL on stdin. "
        "If history capability is available, you may use ssgrep search <query> --json "
        "and ssgrep show <returned-ref>. History is optional; solve from the "
        "current database when unavailable. Do not read raw histories, credentials, "
        "other projects, or operation source. No task delegation.\n"
    )
    return {
        "L3": common + objective + "\n" + contract,
        "L4": common + prompts.L4 + "\n" + CONTRACTS["L4"],
    }


def verified_source(root, model):
    root = Path(root).resolve()
    roots = LabRoots(root, root.name.removeprefix("ssgrep-airline-lab-"))
    roots.verify_ownership()
    evidence = json.loads((roots.artifacts / "live-evidence.json").read_text())
    manifest = json.loads((roots.artifacts / "manifest.json").read_text())
    report = json.loads((roots.artifacts / "report.json").read_text())
    if manifest["model"] != model:
        raise LiveBlocked("matched.source_model: differs from requested model")
    for gate in ("a_native_persistence", "restart_retrieval"):
        if report["gates"].get(gate, {}).get("status") != PASS:
            raise LiveBlocked(f"matched.source_{gate}: verified A persistence required")
    record = evidence["turns"]["L2"]["native"]
    if len(record["session_files"]) != 1:
        raise LiveBlocked("matched.source_unique: one A native source required")
    source = Path(record["session_files"][0]).absolute()
    if not source.is_relative_to(roots.runtime_a):
        raise LiveBlocked("matched.source_location: source must belong to retained A")
    digests = list(record["file_digests"].values())
    if len(digests) != 1:
        raise LiveBlocked("matched.source_unique: one A native source digest required")
    snapshot = freeze_source(source, roots.runtime_a, digests[0])
    native = _parse_native_bytes(source, snapshot)
    if (
        list(record["file_digests"].values()) != [native.file_digest]
        or native.partial_tail_bytes
        or native.malformed_lines
        or sum(r.kind == "user_turn" for r in native.records) != 2
        or native.session_id != evidence["native_ids"]["a"]
    ):
        raise LiveBlocked("matched.source_integrity: changed/incomplete/non-A history")
    return snapshot, native, manifest


def run_matched(source_root, output, *, authorized, model, repetitions=1, order_seed="matched-v1"):
    if authorized is not True:
        raise LiveBlocked("matched.authorization: explicit --authorize-live required")
    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or not 1 <= repetitions <= 3
    ):
        raise LiveBlocked("matched.bound: repetitions must be between 1 and 3")
    output = Path(output).resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    attempts, plan, runs = [], None, []
    try:
        source, native, source_manifest = verified_source(source_root, model)
        version = verify_protocol().version
        common = matched_prompts()
        roots = [LabRoots.create(output, secrets.token_hex(6)) for _ in range(2 * repetitions)]
        # Prepare a real pinned baseline before committing the immutable execution plan.
        first = matched_native.prepare(roots[0], lambda _: None)
        runs.append(first)
        if (
            first.manifest.seed_digest != source_manifest["seed_digest"]
            or first.manifest.schema_digest != source_manifest["schema_digest"]
        ):
            raise LiveBlocked("matched.source_baseline: seed/schema differs from A")
        plan = build_plan(
            repetitions=repetitions,
            baseline_digest=first.base.digest,
            runtime=f"omp/{version}",
            model=model,
            prompt_hashes={k: sha(v) for k, v in common.items()},
            order_seed=order_seed,
        )
        declaration = {
            "plan": plan.as_dict(),
            "prompts": common,
            "source_sha256": native.file_digest,
            "source_session": native.session_id,
            "sampling": {"thinking": "low", "max_time": "10m", "timeout_seconds": 620},
            "seed_digest": first.manifest.seed_digest,
            "schema_digest": first.manifest.schema_digest,
            "cost_basis": "runtime catalog estimate, not provider billing",
        }
        write_private(output / "declaration.json", declaration)
        (output / "declaration.json").chmod(0o400)
        declaration_hash = hashlib.sha256((output / "declaration.json").read_bytes()).hexdigest()
        write_private(output / "declaration.sha256.json", {"sha256": declaration_hash})
        for number, (arm, repetition) in enumerate(plan.slots):
            root = roots[number]
            started = datetime.now(UTC).isoformat()
            status, correctness, measured = "failed", {"execution": BLOCKED}, {}
            baseline = "unavailable"
            run = None
            interrupted = False
            # Persist intention before any model call; a killed process leaves unknown evidence.
            write_private(
                root.artifacts / "attempt-started.json",
                {"arm": arm, "repetition": repetition, "started_utc": started, "status": "unknown"},
            )
            attempts.append(
                Attempt(
                    arm,
                    repetition,
                    root.run_id,
                    baseline,
                    plan.runtime,
                    model,
                    plan.prompt_hashes,
                    correctness,
                    {},
                    "unknown",
                    started,
                )
            )
            write_private(output / "comparison.json", compare(plan, attempts).as_dict())
            try:
                if (
                    hashlib.sha256((output / "declaration.json").read_bytes()).hexdigest()
                    != declaration_hash
                ):
                    raise LiveBlocked("matched.declaration_integrity: immutable plan changed")
                run = first if number == 0 else matched_native.prepare(root, lambda _: None)
                if number:
                    runs.append(run)
                baseline = run.base.digest
                if baseline != plan.baseline_digest:
                    raise LiveBlocked(
                        "matched.clone_baseline: clone differs from declared baseline"
                    )
                sealed = [r.root for r in roots if r != root]
                sealed += [
                    Path(source_root).resolve(),
                    output / "declaration.json",
                    output / "declaration.sha256.json",
                    output / "comparison.json",
                ]
                correctness, measured = matched_native.execute(
                    run,
                    source=source,
                    source_digest=native.file_digest,
                    source_sid=native.session_id,
                    history=arm == WITH_HISTORY,
                    prompts=common,
                    model=model,
                    runtime=version,
                    sealed=sealed,
                )
                correctness["execution"] = PASS
                status = "completed"
            except BaseException as exc:
                interrupted = isinstance(exc, (KeyboardInterrupt, SystemExit))
                status = "interrupted" if interrupted else "failed"
                write_private(
                    root.artifacts / "failure.json",
                    {
                        "gate": "matched.execution",
                        "exception": type(exc).__name__,
                        "detail": str(exc),
                    },
                )
            finally:
                if run is not None:
                    close_run(run)
            attempts[-1] = Attempt(
                arm,
                repetition,
                root.run_id,
                baseline,
                plan.runtime,
                model,
                plan.prompt_hashes,
                correctness,
                measured,
                status,
                started,
            )
            report = compare(plan, attempts).as_dict()
            write_private(output / "comparison.json", report)
            write_private(root.artifacts / "attempt.json", report["attempts"][-1])
            if interrupted:
                break
        return compare(plan, attempts)
    except BaseException as exc:
        write_private(output / "blocked.json", {"gate": "matched.prerequisite", "detail": str(exc)})
        if plan is not None:
            write_private(output / "comparison.json", compare(plan, attempts).as_dict())
        raise
    finally:
        for run in runs:
            try:
                if not run.instance.removed:
                    run.instance.teardown()
            finally:
                remove_credentials(run.roots)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path, required=True, help="retained verified native A lab root"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="new private directory; never reused"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--repetitions", type=int, default=1, choices=range(1, 4))
    parser.add_argument("--order-seed", default="matched-v1")
    parser.add_argument(
        "--authorize-live", action="store_true", help="authorize up to 4 model turns per repetition"
    )
    args = parser.parse_args(argv)
    try:
        report = run_matched(
            args.source_root,
            args.output,
            authorized=args.authorize_live,
            model=args.model,
            repetitions=args.repetitions,
            order_seed=args.order_seed,
        )
    except (Exception, KeyboardInterrupt) as exc:
        print(json.dumps({"status": "BLOCKED", "gate": "matched.execution", "detail": str(exc)}))
        return 3
    print(json.dumps(report.as_dict()))
    return 0 if report.pairs_compared == report.pairs_planned else 1


if __name__ == "__main__":
    raise SystemExit(main())
