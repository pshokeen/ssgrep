"""Actual DB + sandboxed control-process smoke, NOT a native/provider comparison.

Run with ``python -m eval.accounting_airline.matched_smoke``. No AuthGateway or
model runtime is opened. This proves transport/isolation only, not model outcomes.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

from .domain_lane import write_private
from .matched_bridge import MatchedBridge
from .matched_native import prepare, profile_for
from .matched_security import close_run
from .roots import LabRoots, runtime_env
from .sandbox import run_sandboxed


def main():
    parent = Path(tempfile.mkdtemp(prefix="ssgrep-matched-control-"))
    roots = [LabRoots.create(parent) for _ in range(2)]
    outcomes = []
    for number, root in enumerate(roots):
        run = prepare(root, lambda _: None)
        try:
            other = roots[1 - number].root
            canary = other / "history-canary.txt"
            canary.write_text("must not cross arms")
            clients = root.tools / "solver-b"
            run.kit_b.task_file.write_text("B/L3")
            with MatchedBridge(run.kit_b, run.b_db, root.project, history=False) as bridge:
                bridge.install_clients(clients)
                env = runtime_env(
                    root, "solver-b", runtime_sessions=run.kit_b.sessions_dir, model_cache=run.cache
                )
                env["PATH"] = str(clients) + os.pathsep + env["PATH"]
                code = (
                    "import subprocess,pathlib,json; "
                    "sql=subprocess.run(['sql','--role','reader','--database','airline'],"
                    "input='SELECT count(*) AS bookings FROM bookings;',"
                    "text=True,capture_output=True); "
                    "history=subprocess.run(['ssgrep','search','rules','--json'],"
                    "text=True,capture_output=True); "
                    f"p=pathlib.Path({str(canary)!r}); "
                    "denied=subprocess.run(['/bin/cat',str(p)],text=True,capture_output=True); "
                    "print(json.dumps({'sql_exit':sql.returncode,'sql_stdout':sql.stdout,"
                    "'history_exit':history.returncode,'cross_arm_read_exit':denied.returncode})); "
                    "raise SystemExit(0 if sql.returncode==0 and history.returncode!=0 "
                    "and denied.returncode!=0 else 1)"
                )
                done = run_sandboxed(
                    profile_for(run, [other]),
                    [sys.executable, "-c", code],
                    env=env,
                    cwd=root.project,
                    timeout=240,
                )
                outcome = {
                    "arm": number,
                    "baseline_digest": run.base.digest,
                    "control_exit": done.returncode,
                    "stdout": done.stdout,
                    "stderr": done.stderr,
                }
                outcomes.append(outcome)
                if done.returncode:
                    raise RuntimeError(f"controlled transport smoke failed: {outcome}")
        finally:
            close_run(run)
    evidence = {
        "label": "actual PostgreSQL + sandboxed control process; NO model/provider calls",
        "empirical_matched_comparison": "UNRUN",
        "outcomes": outcomes,
        "identical_baselines": outcomes[0]["baseline_digest"] == outcomes[1]["baseline_digest"],
    }
    write_private(parent / "control-smoke.json", evidence)
    print(json.dumps({"artifact": str(parent / "control-smoke.json"), **evidence}))
    return 0 if evidence["identical_baselines"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
