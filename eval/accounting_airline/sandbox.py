"""macOS ``sandbox-exec`` boundary for solver B, with a probe that proves it works.

The profile is ``allow default`` plus explicit denials of file reads AND writes for every path
B must not see (A's runtime state, the verifier, artifacts, DB admin credentials, the docker
sockets, and the real history stores). It is deliberately honest about what it does not
enforce: each gap is a named ``Limitation``, never a silent omission. ``prove_isolation`` does
not trust the profile text: it plants canaries, reads them sandboxed and unsandboxed, and
reports PASS only if the denials AND the allowed twins behave as declared.
"""

from __future__ import annotations

import os
import secrets
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .bounded_capture import OutputLimitExceeded, run_bounded
from .gates import BLOCKED, Gate, from_violations
from .roots import LabRoots

GATE_NAME = "isolation.solver_b_filesystem"
SANDBOX_EXEC = "/usr/bin/sandbox-exec"


def account_home() -> Path:
    """Actual account home, independent of a shell's or solver's HOME override."""
    import pwd

    return Path(pwd.getpwuid(os.getuid()).pw_dir)


_PROBE_TIMEOUT = 30

# Real history/config stores (relative to the real home) solver B must not read or write.
# The omp credential store (~/.omp/agent/agent.db, config.yml) is deliberately NOT here.
REAL_HISTORY_STORES = (
    ".omp/agent/sessions",
    ".claude",
    ".codex",
    ".pi",
    ".prime",
    ".prime-agent",
    ".local/share/opencode",
    ".ssgrep",
)
DOCKER_SOCKETS = ("/var/run/docker.sock", "~/.docker/run/docker.sock")


@dataclass(frozen=True)
class Limitation:
    name: str
    reason: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {"name": self.name, "reason": self.reason, "detail": self.detail}


@dataclass(frozen=True)
class SandboxProfile:
    text: str
    denied: tuple[Path, ...]
    limitations: tuple[Limitation, ...]


# ------------------------------------------------------------------------- availability


def _sandbox_binary() -> str | None:
    return shutil.which("sandbox-exec") or (SANDBOX_EXEC if os.path.exists(SANDBOX_EXEC) else None)


def availability() -> Limitation | None:
    """``None`` when sandbox-exec exists and a trivial allow-default probe runs."""
    binary = _sandbox_binary()
    if binary is None:
        return Limitation(
            "sandbox_exec_unavailable",
            "sandbox-exec is not installed",
            "no sandbox-exec on PATH or at /usr/bin/sandbox-exec; solver B filesystem "
            "isolation cannot be enforced or proven on this host",
        )
    try:
        probe = run_bounded(
            [binary, "-p", "(version 1)(allow default)", "/usr/bin/true"],
            text=True,
            timeout=_PROBE_TIMEOUT,
            check=False,
            stdout_limit=1024 * 1024,
            stderr_limit=1024 * 1024,
        )
    except OutputLimitExceeded as exc:
        return Limitation(
            "sandbox_exec_output_limit", str(exc), "sandbox-exec probe exceeded byte caps"
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return Limitation(
            "sandbox_exec_unavailable",
            f"sandbox-exec probe could not run: {type(exc).__name__}",
            f"{binary} -p '(version 1)(allow default)' /usr/bin/true did not complete",
        )
    if probe.returncode != 0:
        return Limitation(
            "sandbox_exec_unavailable",
            f"sandbox-exec probe exited {probe.returncode}",
            probe.stderr.strip()[:300],
        )
    return None


# ----------------------------------------------------------------------------- profile


def sbpl_quote(path: str | Path) -> str:
    """SBPL string literal for ``path``; control characters are rejected, not escaped."""
    text = os.fspath(path)
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        raise ValueError(f"path contains control characters: {text!r}")
    if "\x00" in text:
        raise ValueError("path contains NUL")
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def profile_for_solver_b(
    roots: LabRoots,
    *,
    extra_denied: Sequence[Path] = (),
    deny_docker_sockets: bool = True,
    real_home: Path | None = None,
) -> SandboxProfile:
    """Deny read+write of everything solver B must not touch; allow the rest."""
    home = real_home if real_home is not None else account_home()
    subpaths: list[Path] = [
        Path(os.path.realpath(roots.runtime_a)),
        Path(os.path.realpath(roots.native_sessions / "a")),
        Path(os.path.realpath(roots.verifier)),
        Path(os.path.realpath(roots.artifacts)),
        Path(os.path.realpath(roots.tools / "admin")),
    ]
    for relative in REAL_HISTORY_STORES:
        # Deny future-created stores as well: environment overrides are not a boundary.
        subpaths.append(Path(os.path.realpath(home / relative)))
    from platformdirs import user_data_path

    subpaths.append(Path(os.path.realpath(user_data_path("ssgrep", appauthor=False))))
    actual_default = home / (
        "Library/Application Support/ssgrep" if sys.platform == "darwin" else ".local/share/ssgrep"
    )
    subpaths.append(Path(os.path.realpath(actual_default)))
    configured = os.environ.get("SSGREP_DATA_DIR")
    if configured:
        subpaths.append(Path(os.path.realpath(Path(configured).expanduser())))
    for extra in extra_denied:
        subpaths.append(Path(os.path.realpath(extra)))
    sockets: list[Path] = []
    if deny_docker_sockets:
        for candidate in DOCKER_SOCKETS:
            raw = home / candidate[2:] if candidate.startswith("~/") else Path(candidate)
            if os.path.lexists(raw):
                for variant in {raw, Path(os.path.realpath(raw))}:
                    sockets.append(variant)
    lines = ["(version 1)", "(allow default)"]
    seen: set[Path] = set()
    for path in subpaths:
        if path in seen:
            continue
        seen.add(path)
        lines.append(f"(deny file-read* file-write* (subpath {sbpl_quote(path)}))")
    # The provider credential reference is read-only, including SQLite sidecars and
    # every other real runtime setting. Runtime-owned writes belong in the lab root.
    lines.append(f"(deny file-write* (subpath {sbpl_quote(home / '.omp')}))")
    for path in sorted(set(sockets)):
        lines.append(f"(deny file-read* file-write* (literal {sbpl_quote(path)}))")
        lines.append(
            f"(deny network-outbound (remote unix-socket (path-literal {sbpl_quote(path)})))"
        )
    denied = tuple(dict.fromkeys([*subpaths, *sorted(set(sockets))]))
    limitations = (
        Limitation(
            "network_not_restricted",
            "the profile is allow-default for network access",
            "loopback and provider network are open by design; B reaches PostgreSQL through "
            "a loopback TCP restricted role, and only the docker unix sockets are denied",
        ),
        Limitation(
            "process_inspection_not_blocked",
            "process listing and signalling are not denied",
            "B can observe other processes' command lines; only filesystem access to the "
            "listed paths is enforced",
        ),
        Limitation(
            "provider_credential_store_readable_by_design",
            "the omp credential store is a brokered read-only reference",
            "~/.omp/agent/agent.db and config.yml stay readable because the runtime must "
            "read them to authenticate; B can therefore read provider credentials",
        ),
    )
    return SandboxProfile("\n".join(lines) + "\n", denied, limitations)


# ------------------------------------------------------------------------------ running


def wrap_argv(profile: SandboxProfile, argv: Sequence[str]) -> list[str]:
    """argv list only; the profile text is a single argument, never shell-interpolated."""
    if isinstance(argv, str):
        raise TypeError("argv must be a sequence of arguments, not a shell string")
    return ["sandbox-exec", "-p", profile.text, *argv]


def run_sandboxed(
    profile: SandboxProfile,
    argv: Sequence[str],
    *,
    env: dict[str, str],
    cwd: Path | str,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    command = wrap_argv(profile, argv)
    command[0] = _sandbox_binary() or SANDBOX_EXEC
    return run_bounded(command, env=env, cwd=cwd, text=True, timeout=timeout, check=False)


# ------------------------------------------------------------------------------- proof


def _probe_env() -> dict[str, str]:
    return {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"}


def _plant(path: Path, token: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(token)


def prove_isolation(roots: LabRoots, profile: SandboxProfile) -> Gate:
    unavailable = availability()
    if unavailable is not None:
        return Gate(
            GATE_NAME,
            BLOCKED,
            f"{unavailable.name}: {unavailable.reason}",
            {"limitation": unavailable.as_dict()},
        )
    token = secrets.token_hex(12)
    denied_dirs = {
        "runtime_a": roots.runtime_a,
        "native_a": roots.native_sessions / "a",
        "verifier": roots.verifier,
        "artifacts": roots.artifacts,
        "tools_admin": roots.tools / "admin",
    }
    allowed_dirs = {
        "project": roots.project,
        "runtime_b": roots.runtime_b,
        "native_b": roots.native_sessions / "b",
        "ssgrep_data": roots.ssgrep_data,
        "tools_solver": roots.tools / "solver",
    }
    canaries: dict[str, tuple[Path, str]] = {}
    violations: list[str] = []
    evidence: dict[str, object] = {"denied": [], "allowed": []}
    env = _probe_env()
    cwd = Path(tempfile.gettempdir())
    write_probe_denied = denied_dirs["runtime_a"] / f"write-probe-{token}"
    write_probe_allowed = allowed_dirs["runtime_b"] / f"write-probe-{token}"
    planted: list[Path] = []
    try:
        for label, directory in {**denied_dirs, **allowed_dirs}.items():
            path = directory / f"canary-{label}-{token}"
            canary = f"CANARY-{label}-{secrets.token_hex(8)}"
            _plant(path, canary)
            planted.append(path)
            canaries[label] = (path, canary)

        def cat(label: str, sandboxed: bool) -> subprocess.CompletedProcess[str]:
            path = canaries[label][0]
            if sandboxed:
                return run_sandboxed(
                    profile, ["/bin/cat", str(path)], env=env, cwd=cwd, timeout=_PROBE_TIMEOUT
                )
            return run_bounded(
                ["/bin/cat", str(path)],
                env=env,
                cwd=cwd,
                text=True,
                timeout=_PROBE_TIMEOUT,
                check=False,
            )

        # (2) control: every canary is readable without the sandbox
        for label, (_path, canary) in canaries.items():
            control = cat(label, sandboxed=False)
            if control.returncode != 0 or canary not in control.stdout:
                violations.append(
                    f"control: {label} canary not readable unsandboxed (file permissions, not "
                    "the sandbox, would be the cause of any denial)"
                )
        # (1) sandboxed: denied ones fail without leaking, allowed ones succeed
        for label in denied_dirs:
            result = cat(label, sandboxed=True)
            leaked = canaries[label][1] in result.stdout + result.stderr
            evidence["denied"].append(  # type: ignore[union-attr]
                {"label": label, "returncode": result.returncode, "leaked": leaked}
            )
            if result.returncode == 0:
                violations.append(f"{label}: sandboxed read of the canary succeeded (not denied)")
            if leaked:
                violations.append(f"{label}: canary content reached sandboxed stdout/stderr")
        for label in allowed_dirs:
            result = cat(label, sandboxed=True)
            ok = result.returncode == 0 and canaries[label][1] in result.stdout
            evidence["allowed"].append(  # type: ignore[union-attr]
                {"label": label, "returncode": result.returncode, "readable": ok}
            )
            if not ok:
                violations.append(
                    f"{label}: sandboxed read of an allowed canary failed "
                    "(profile denies more than declared)"
                )
        # writes
        write = ["/bin/sh", "-c", 'printf written > "$1"', "sh"]
        denied_write = run_sandboxed(
            profile, [*write, str(write_probe_denied)], env=env, cwd=cwd, timeout=_PROBE_TIMEOUT
        )
        if denied_write.returncode == 0 or write_probe_denied.exists():
            violations.append("runtime_a: sandboxed write was not denied")
        allowed_write = run_sandboxed(
            profile, [*write, str(write_probe_allowed)], env=env, cwd=cwd, timeout=_PROBE_TIMEOUT
        )
        if (
            allowed_write.returncode != 0
            or not write_probe_allowed.exists()
            or write_probe_allowed.read_text() != "written"
        ):
            violations.append("runtime_b: sandboxed write to an allowed directory failed")
        # the parent of the denied directories stays listable
        listing = run_sandboxed(
            profile, ["/bin/ls", str(roots.root)], env=env, cwd=cwd, timeout=_PROBE_TIMEOUT
        )
        if listing.returncode != 0 or "runtime-b" not in listing.stdout:
            violations.append("lab root listing failed in the sandbox (parent must stay readable)")
        evidence["parent_listing_ok"] = listing.returncode == 0
    finally:
        for path in (*planted, write_probe_denied, write_probe_allowed):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    evidence["limitations"] = [item.as_dict() for item in profile.limitations]
    return from_violations(GATE_NAME, violations, evidence)
