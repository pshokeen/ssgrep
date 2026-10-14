"""Positive/negative boundary probes run once before any native provider request."""

from __future__ import annotations

import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from .gates import Gate, from_violations
from .lab_context import Run
from .lab_product import REPO_ROOT
from .live_bridge import TRUSTED_SHIM, Bridge
from .live_security import live_profile
from .observer import ENV_ACTOR, ENV_SPOOL, ENV_TASK_FILE, _child_env, _sql_argv
from .sandbox import run_sandboxed


def prove_bridge(run: Run) -> Gate:
    kit = run.kit_a
    argv = _sql_argv(["--role", "reader", "--database", run.a_db], kit.env)
    env = _child_env("sql", kit.env)
    marker = run.roots.verifier / "forbidden-client-shell-output"
    observations = []
    for label, sql, expected in (
        ("protocol_query", "SELECT 1 AS boundary;", 0),
        ("client_shell_denied", f"\\! touch {marker}\n", 1),
        ("client_file_include_denied", f"\\i {run.roots.marker}\n", 1),
    ):
        call, _done = run.observer.run(
            actor="control",
            task="live-boundary",
            tool="sql",
            argv=argv,
            env=env,
            cwd=run.roots.project,
            timeout=30,
            stdin=sql,
            shown=["sql-protocol-probe", label],
        )
        observations.append(
            {"name": label, "exit": call.exit_code, "expected": expected, "output": call.stdout}
        )
    problems = [
        f"{o['name']}: exit {o['exit']} != {o['expected']}"
        for o in observations
        if o["exit"] != o["expected"]
    ]
    if str(observations[0]["output"]).strip() != "boundary\n1":
        problems.append("SQL positive twin did not return expected server value")
    if marker.exists():
        problems.append("SQL client-side shell executed outside the sandbox")
        marker.unlink()
    # A solver-writable package with a side effect must not become supervisor code.
    package = run.roots.project / "eval"
    package.mkdir()
    init = package / "__init__.py"
    init.write_text(f"from pathlib import Path; Path({str(marker)!r}).write_text('escaped')\n")
    try:
        code = TRUSTED_SHIM.rsplit("raise SystemExit", 1)[0] + "print('trusted-observer')"
        call, _done = run.observer.run(
            actor="control",
            task="live-boundary",
            tool="python",
            argv=[sys.executable, "-I", "-c", code],
            env=kit.env,
            cwd=run.roots.project,
            timeout=30,
            shown=["isolated-python-import-probe"],
        )
        safe = (
            call.exit_code == 0
            and call.stdout.strip() == "trusted-observer"
            and not marker.exists()
        )
        observations.append({"name": "workspace_import_denied", "safe": safe})
        if not safe:
            problems.append("solver workspace controlled supervisor imports")
    finally:
        init.unlink()
        package.rmdir()
        marker.unlink(missing_ok=True)
    return from_violations("live.supervisor_boundary", problems, {"probes": observations})


def prove_native_tools(run: Run) -> Gate:
    """Exercise both native boundaries, including future neighboring evidence and real SQL."""
    problems, observations = [], []
    profiles = {actor: live_profile(run.roots, actor) for actor in ("solver-a", "solver-b")}
    with tempfile.TemporaryDirectory(
        prefix="ssgrep-airline-future-report-", dir=run.roots.root.parent
    ) as directory:
        neighbor = Path(directory) / "private.json"
        neighbor.write_text("future private evidence")
        for actor, profile in profiles.items():
            scratch = run.roots.actor_home(actor) / "tmp"
            scratch.mkdir(mode=0o700, exist_ok=True)
            env = {
                "PATH": "/usr/bin:/bin",
                "HOME": str(run.roots.actor_home(actor)),
                "TMPDIR": str(scratch),
            }
            for path in (
                neighbor,
                run.roots.marker,
                REPO_ROOT / "eval/accounting_airline/domain_oracle.py",
            ):
                if not path.read_bytes():
                    problems.append(f"empty positive source: {path}")
                result = run_sandboxed(
                    profile, ["/bin/cat", str(path)], env=env, cwd=run.roots.project, timeout=10
                )
                denied = result.returncode != 0 and not result.stdout
                observations.append({"actor": actor, "path": str(path), "read_denied": denied})
                if not denied:
                    problems.append(f"private source readable by {actor}: {path}")
            code = (
                "from pathlib import Path; import sys; p=Path(sys.argv[1]); "
                "p.write_text('own'); print(p.read_text()); p.unlink()"
            )
            result = run_sandboxed(
                profile,
                [sys.executable, "-I", "-c", code, str(scratch / "positive")],
                env=env,
                cwd=run.roots.project,
                timeout=10,
            )
            positive = result.returncode == 0 and result.stdout.strip() == "own"
            observations.append({"actor": actor, "own_scratch_write_read": positive})
            if not positive:
                problems.append(f"own scratch unavailable to {actor}: {result.stderr}")
            original = run.kit_a if actor == "solver-a" else run.kit_b
            task = run.roots.verifier / f"preflight-task-{actor}"
            task.write_text("native-tool-boundary")
            spool = run.roots.verifier / f"preflight-spool-{actor}.ndjson"
            control = replace(
                original,
                actor="control",
                spool=spool,
                task_file=task,
                env={
                    **original.env,
                    ENV_ACTOR: "control",
                    ENV_SPOOL: str(spool),
                    ENV_TASK_FILE: str(task),
                },
            )
            database = run.a_db if actor == "solver-a" else run.b_db
            with Bridge(control, database, run.roots.project) as bridge:
                clients = run.roots.tools / actor
                bridge.install_clients(clients)
                query = 'printf "SELECT 1 AS boundary;" | "$1" --role reader --database "$2"'
                result = run_sandboxed(
                    profile,
                    ["/bin/sh", "-c", query, "probe", str(clients / "sql"), database],
                    env=env,
                    cwd=run.roots.project,
                    timeout=30,
                )
            positive = result.returncode == 0 and result.stdout.strip() == "boundary\n1"
            observations.append(
                {"actor": actor, "real_sql_client": positive, "stdout": result.stdout}
            )
            if not positive:
                problems.append(f"real SQL client unavailable to {actor}: {result.stderr}")
    return from_violations("live.native_tools_boundary", problems, {"probes": observations})
