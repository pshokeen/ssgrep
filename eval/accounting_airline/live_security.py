"""Live-only enforcement: seal raw sources/oracles and make real stores read-only."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from .gates import Gate, from_violations
from .lab_product import REPO_ROOT
from .roots import LabRoots
from .sandbox import (
    Limitation,
    SandboxProfile,
    account_home,
    profile_for_solver_b,
    run_sandboxed,
    sbpl_quote,
)


def live_profile(roots: LabRoots, actor: str, *, real_home: Path | None = None) -> SandboxProfile:
    home = real_home or account_home()
    sealed = [REPO_ROOT / name for name in ("eval", "tests", "openspec", "docs", ".git")]
    sealed += [REPO_ROOT / "README.md", REPO_ROOT / "CHANGELOG.md", roots.ssgrep_data]
    base = profile_for_solver_b(roots, extra_denied=sealed, real_home=home)
    denied = list(base.denied)
    if actor == "solver-a":
        denied.remove(roots.runtime_a)
        denied.append(roots.runtime_b)
    other = "solver-b" if actor == "solver-a" else "solver-a"
    denied.extend((roots.tools / other, roots.tmp, home / ".omp"))
    lines = ["(version 1)", "(allow default)"]
    for path in denied:
        lines.append(f"(deny file-read* file-write* (subpath {sbpl_quote(path)}))")
    # No solver or inherited child can mutate actual account/config/history stores,
    # the repository, or the model cache. All intended writes live under the run.
    writable = roots.runtime_a if actor == "solver-a" else roots.runtime_b
    lines += [
        "(deny file-write*)",
        f"(deny file-write* (subpath {sbpl_quote(home)}))",
        f"(allow file-write* (subpath {sbpl_quote(writable)}))",
        f"(allow file-write* (subpath {sbpl_quote(roots.project)}))",
        '(allow file-write* (literal "/dev/null"))',
        "(deny process-info*)",
        "(allow process-info* (target self))",
        "(deny signal)",
    ]
    # Deny parents, not enumerated siblings: new pytest/report/lab roots must stay sealed.
    temporary = {
        roots.root.parent,
        Path(tempfile.gettempdir()).resolve(),
        Path("/tmp").resolve(),
        Path("/var/tmp").resolve(),
    }
    for path in sorted(temporary):
        lines.append(f"(deny file-read* file-write* (subpath {sbpl_quote(path)}))")
    for path in (writable, roots.project, roots.tools / actor):
        lines.append(f"(allow file-read* (subpath {sbpl_quote(path)}))")
    for path in (writable, roots.project):
        lines.append(f"(allow file-write* (subpath {sbpl_quote(path)}))")
    # The other actor's native/scratch/tools and all verifier stores stay sealed.
    for path in denied:
        lines.append(f"(deny file-read* file-write* (subpath {sbpl_quote(path)}))")
    for socket in (home / ".docker/run/docker.sock", Path("/var/run/docker.sock")):
        lines.append(
            f"(deny network-outbound (remote unix-socket (path-literal {sbpl_quote(socket)})))"
        )
    limitations = (
        Limitation(
            "provider_gateway",
            "provider traffic uses a supervised readonly auth gateway",
            "The native actor cannot read real OMP credentials. The supervised gateway "
            "loads one credential read-only into RAM, disables refresh, and restricts "
            "each actor to its own router cache and the selected model.",
        ),
        Limitation(
            "provider_network",
            "provider and loopback network remain enabled",
            "The bridge accepts only assigned SQL roles/database and bounded real search/show.",
        ),
    )
    return SandboxProfile(
        "\n".join(lines) + "\n", tuple(denied) + tuple(sorted(temporary)), limitations
    )


def prove_live_isolation(roots: LabRoots, profile: SandboxProfile, a_native: Path) -> Gate:
    """Use real denied reads and harmless O_WRONLY opens (no bytes written) as probes."""
    problems, evidence = [], []
    env = {"PATH": "/usr/bin:/bin", "HOME": str(roots.actor_home("solver-b"))}
    paths = [
        a_native,
        REPO_ROOT / "eval/accounting_airline/domain_oracle.py",
        REPO_ROOT / "openspec/changes/add-savings-ledger/validation-lab.md",
        roots.marker,
    ]
    for path in paths:
        control = subprocess.run(["/bin/cat", str(path)], capture_output=True, timeout=10)
        denied = run_sandboxed(
            profile, ["/bin/cat", str(path)], env=env, cwd=roots.project, timeout=10
        )
        ok = control.returncode == 0 and denied.returncode != 0 and not denied.stdout
        evidence.append({"path": str(path), "denied": ok})
        if not ok:
            problems.append(f"read boundary failed: {path}")
    allowed = roots.project / "isolation-positive.txt"
    allowed.write_text("public business brief canary")
    twin = run_sandboxed(
        profile, ["/bin/cat", str(allowed)], env=env, cwd=roots.project, timeout=10
    )
    allowed.unlink()
    if twin.returncode != 0 or twin.stdout != "public business brief canary":
        problems.append("public project positive twin failed")
    # Test open-for-write without O_TRUNC/O_CREAT and without writing any byte.
    import sys

    source = account_home() / ".omp/agent/agent.db"
    probe = run_sandboxed(
        profile,
        [
            sys.executable,
            "-c",
            "import os,sys; fd=os.open(sys.argv[1],os.O_WRONLY); os.close(fd)",
            str(source),
        ],
        env=env,
        cwd=roots.project,
        timeout=10,
    )
    if probe.returncode == 0:
        problems.append("real credential store remained writable")
    evidence.append({"credential_write_denied": probe.returncode != 0})
    read_probe = run_sandboxed(
        profile,
        [
            sys.executable,
            "-I",
            "-c",
            "import os,sys; fd=os.open(sys.argv[1],os.O_RDONLY); os.close(fd)",
            str(source),
        ],
        env=env,
        cwd=roots.project,
        timeout=10,
    )
    evidence.append({"credential_read_denied": read_probe.returncode != 0})
    if read_probe.returncode == 0:
        problems.append("native actor can read real provider credentials")
    for label, path, permitted in (
        ("shared_scratch", roots.tmp / "write-boundary", False),
        ("shared_tools", roots.tools / "write-boundary", False),
        ("own_runtime", roots.runtime_b / "write-boundary", True),
    ):
        code = "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('boundary')"
        result = run_sandboxed(
            profile,
            [sys.executable, "-I", "-c", code, str(path)],
            env=env,
            cwd=roots.project,
            timeout=10,
        )
        correct = (result.returncode == 0 and path.exists()) == permitted
        evidence.append(
            {
                "write_probe": label,
                "correct": correct,
                "exit": result.returncode,
                "stderr": result.stderr[:500],
            }
        )
        if not correct:
            problems.append(f"{label} write boundary failed")
        path.unlink(missing_ok=True)
    return from_violations("live.isolation", problems, {"probes": evidence})
