"""Owner-only lab roots, explicit per-actor environments, and ownership-checked removal.

One lab run owns exactly one ``ssgrep-airline-lab-<run_id>`` directory (mode 0700, marker in
``verifier/``). Nothing here touches a directory it did not create and verify; ``remove`` never
follows a symlink out of the root. ``runtime_env`` builds a COMPLETE explicit environment: the
ambient process environment is never copied wholesale, so real credentials, real session
history and real settings cannot reach a lab runtime by inheritance.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SUBDIRS = ("project", "runtime-a", "runtime-b", "ssgrep-data", "verifier", "artifacts", "tools")
ACTORS = ("solver-a", "solver-b", "verifier", "control")
ROOT_PREFIX = "ssgrep-airline-lab-"
MARKER_NAME = ".lab-owner.json"
COLLECT_VIEW = "collect-view"

# Per-file bound for retained failure evidence; the total bound is ``retain_cap_bytes``.
MAX_RETAINED_FILE_BYTES = 256_000
MAX_RETAINED_FILES = 200

_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,40}$")

# Names that may carry a secret. Secrets reach tools only through files, by reference.
_SECRET_FRAGMENTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
_SECRET_PREFIXES = ("PG", "AWS_")
_SECRET_NAMES = frozenset({"DATABASE_URL"})

# Variables the lab refuses to export: they would override or bypass the six discovery roots.
FORBIDDEN_NAMES = frozenset(
    {
        "SSGREP_TRANSCRIPT_DIRS",
        "SSGREP_PI_SESSIONS_DIR",
        "SSGREP_PRIME_AGENT_SESSIONS_DIR",
        "SSGREP_CODEX_SESSIONS_DIR",
    }
)

# Every variable that tells a runtime or ssgrep where history/config/data lives.
DISCOVERY_VARS = (
    "HOME",
    "TMPDIR",
    "SSGREP_DATA_DIR",
    "CLAUDE_CONFIG_DIR",
    "CODEX_HOME",
    "CODEX_SESSIONS_DIR",
    "PI_CODING_AGENT_DIR",
    "PI_SESSION_DIR",
    "PI_CODING_AGENT_SESSION_DIR",
    "PRIME_AGENT_CODING_AGENT_DIR",
    "PRIME_AGENT_SESSION_DIR",
    "PRIME_AGENT_CODING_AGENT_SESSION_DIR",
    "SSGREP_OMP_SESSIONS_DIR",
    "OMP_SESSIONS_DIR",
    "OMP_AGENT_DIR",
    "SSGREP_OPENCODE_DB",
    "XDG_DATA_HOME",
    "XDG_CONFIG_HOME",
    "OPENCODE_CONFIG_DIR",
)
_CORE_VARS = ("HOME", "TMPDIR", "SSGREP_DATA_DIR")
_READ_ONLY_REFERENCES = frozenset({"HF_HOME"})


class RootOwnershipError(RuntimeError):
    """The directory is not provably this run's private root; nothing was changed."""


class IsolationError(RuntimeError):
    """An environment points outside the lab root or carries a forbidden variable."""


def _mkdir_private(path: Path) -> None:
    os.mkdir(path, 0o700)
    os.chmod(path, 0o700)


def _owned_private_dir(path: Path) -> str | None:
    """Reason ``path`` is not an owned, real, 0700 directory; ``None`` when it is."""
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return "does not exist"
    if stat.S_ISLNK(info.st_mode):
        return "is a symlink"
    if not stat.S_ISDIR(info.st_mode):
        return "is not a directory"
    if info.st_uid != os.getuid():
        return f"is owned by uid {info.st_uid}, not {os.getuid()}"
    if stat.S_IMODE(info.st_mode) != 0o700:
        return f"has mode {stat.S_IMODE(info.st_mode):04o}, not 0700"
    return None


@dataclass(frozen=True)
class LabRoots:
    root: Path
    run_id: str

    # ------------------------------------------------------------------ layout
    @property
    def project(self) -> Path:
        return self.root / "project"

    @property
    def runtime_a(self) -> Path:
        return self.root / "runtime-a"

    @property
    def runtime_b(self) -> Path:
        return self.root / "runtime-b"

    @property
    def ssgrep_data(self) -> Path:
        return self.root / "ssgrep-data"

    @property
    def verifier(self) -> Path:
        return self.root / "verifier"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def tools(self) -> Path:
        return self.root / "tools"

    @property
    def collect_view(self) -> Path:
        return self.artifacts / COLLECT_VIEW

    @property
    def tmp(self) -> Path:
        return self.root / "tmp"

    @property
    def marker(self) -> Path:
        return self.verifier / MARKER_NAME

    def actor_home(self, actor: str) -> Path:
        if actor == "solver-a":
            return self.runtime_a / "home"
        if actor == "solver-b":
            return self.runtime_b / "home"
        if actor in ("verifier", "control"):
            return self.verifier / "home"
        raise ValueError(f"unknown actor {actor!r}; expected one of {ACTORS}")

    # --------------------------------------------------------------- lifecycle
    @classmethod
    def create(cls, parent: Path | None = None, run_id: str | None = None) -> LabRoots:
        run_id = run_id or secrets.token_hex(6)
        if not _RUN_ID.match(run_id):
            raise ValueError(f"unsafe run id {run_id!r}")
        base = parent if parent is not None else Path(os.environ.get("TMPDIR") or "/tmp")
        real_parent = Path(os.path.realpath(base))
        real_parent.mkdir(parents=True, exist_ok=True)
        root = real_parent / f"{ROOT_PREFIX}{run_id}"
        try:
            os.mkdir(root, 0o700)
        except FileExistsError:
            problem = _owned_private_dir(root)
            if problem is None and os.listdir(root):
                problem = "is not empty"
            if problem is not None:
                raise RootOwnershipError(
                    f"refusing to reuse existing lab root {root}: it {problem}"
                ) from None
        os.chmod(root, 0o700)
        roots = cls(root=root, run_id=run_id)
        for name in SUBDIRS:
            _mkdir_private(root / name)
        for path in (
            roots.collect_view,
            roots.tools / "admin",
            roots.tools / "solver",
            roots.tmp,
            roots.actor_home("solver-a"),
            roots.actor_home("solver-b"),
            roots.actor_home("verifier"),
        ):
            _mkdir_private(path)
        marker = {
            "run_id": run_id,
            "uid": os.getuid(),
            "created_utc": datetime.now(UTC).isoformat(),
            "pid": os.getpid(),
            "root": str(root),
        }
        fd = os.open(roots.marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps(marker, indent=2, sort_keys=True) + "\n")
        return roots

    def verify_ownership(self) -> None:
        """Raise ``RootOwnershipError`` unless this is provably the run's own private root."""
        if self.root.name != f"{ROOT_PREFIX}{self.run_id}" or not _RUN_ID.match(self.run_id):
            raise RootOwnershipError(f"{self.root} is not named {ROOT_PREFIX}<run_id>")
        problem = _owned_private_dir(self.root)
        if problem is not None:
            raise RootOwnershipError(f"lab root {self.root} {problem}")
        if os.path.realpath(self.root) != str(self.root):
            raise RootOwnershipError(f"lab root {self.root} is not a canonical real path")
        problem = _owned_private_dir(self.verifier)
        if problem is not None:
            raise RootOwnershipError(f"verifier dir {self.verifier} {problem}")
        try:
            fd = os.open(self.marker, os.O_RDONLY | os.O_NOFOLLOW)
        except OSError as exc:
            raise RootOwnershipError(f"ownership marker unreadable: {exc.strerror}") from None
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                raise RootOwnershipError("ownership marker is not a regular file owned by us")
            with os.fdopen(fd, "r") as handle:
                fd = -1
                try:
                    marker = json.load(handle)
                except ValueError:
                    raise RootOwnershipError("ownership marker is not valid JSON") from None
        finally:
            if fd >= 0:
                os.close(fd)
        if not isinstance(marker, dict):
            raise RootOwnershipError("ownership marker is not a JSON object")
        expected = {"run_id": self.run_id, "uid": os.getuid(), "root": str(self.root)}
        for key, value in expected.items():
            if marker.get(key) != value:
                raise RootOwnershipError(f"ownership marker {key} does not match this run")

    def remove(
        self, *, retain_failure_to: Path | None = None, retain_cap_bytes: int = 4_000_000
    ) -> dict[str, Any]:
        """Delete this run's root; ownership is verified BEFORE anything is read or deleted."""
        try:
            os.lstat(self.root)
        except FileNotFoundError:
            return {"removed": False, "retained_files": [], "retained_bytes": 0}
        self.verify_ownership()
        retained: list[str] = []
        skipped: list[str] = []
        retained_bytes = 0
        retained_dir: str | None = None
        if retain_failure_to is not None:
            destination = Path(os.path.realpath(retain_failure_to))
            if destination == self.root or self.root in destination.parents:
                raise ValueError("failure retention directory must be outside the lab root")
            retained_dir, retained, retained_bytes, skipped = _retain_artifacts(
                self, destination, retain_cap_bytes
            )
        _reopen_owned_dirs(self.root)
        shutil.rmtree(self.root)
        return {
            "removed": not os.path.lexists(self.root),
            "retained_files": retained,
            "retained_bytes": retained_bytes,
            "retained_dir": retained_dir,
            "skipped": skipped,
        }


def _reopen_owned_dirs(root: Path) -> None:
    """Make every real directory under ``root`` traversable so rmtree can finish.

    Symlinks are never followed or chmod'ed (``lstat`` gate), so nothing outside the root
    can be touched.
    """
    pending = [os.fspath(root)]
    while pending:
        directory = pending.pop()
        try:
            info = os.lstat(directory)
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
                continue
            if stat.S_IMODE(info.st_mode) & 0o700 != 0o700:
                os.chmod(directory, 0o700)
            with os.scandir(directory) as entries:
                pending.extend(
                    entry.path for entry in entries if entry.is_dir(follow_symlinks=False)
                )
        except OSError:
            continue


def _retain_artifacts(
    roots: LabRoots, destination_parent: Path, cap: int
) -> tuple[str, list[str], int, list[str]]:
    """Copy a bounded set of small regular files from artifacts/ into a new 0700 directory."""
    destination_parent.mkdir(parents=True, exist_ok=True)
    target = destination_parent / f"lab-failure-{roots.run_id}-{secrets.token_hex(3)}"
    _mkdir_private(target)
    retained: list[str] = []
    skipped: list[str] = []
    total = 0
    pending = [roots.artifacts]
    while pending:
        directory = pending.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError:
            skipped.append(os.path.relpath(directory, roots.artifacts))
            continue
        for entry in entries:
            relative = os.path.relpath(entry.path, roots.artifacts)
            if entry.is_symlink():
                skipped.append(relative)
            elif entry.is_dir(follow_symlinks=False):
                if relative == COLLECT_VIEW:
                    skipped.append(relative)
                else:
                    pending.append(Path(entry.path))
            elif not entry.is_file(follow_symlinks=False):
                skipped.append(relative)
            else:
                size = entry.stat(follow_symlinks=False).st_size
                if (
                    size > MAX_RETAINED_FILE_BYTES
                    or size > cap
                    or total + size > cap
                    or len(retained) >= MAX_RETAINED_FILES
                ):
                    skipped.append(relative)
                    continue
                if _copy_private(Path(entry.path), target / relative, cap - total):
                    retained.append(relative)
                    total += size
                else:
                    skipped.append(relative)
    return str(target), retained, total, skipped


def _copy_private(source: Path, destination: Path, remaining: int) -> bool:
    try:
        fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError:
        return False
    with os.fdopen(fd, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            return False
        data = handle.read(remaining + 1)
    if len(data) > remaining:
        return False
    _ensure_private_dirs(destination.parent)
    out = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(out, "wb") as handle:
        handle.write(data)
    return True


def _ensure_private_dirs(path: Path) -> None:
    """Create ``path`` and any missing ancestors below it, each 0700 (never above the target)."""
    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        _mkdir_private(directory)


# ------------------------------------------------------------------------- environment


def is_secret_name(name: str) -> bool:
    upper = name.upper()
    return (
        upper in _SECRET_NAMES
        or upper.startswith(_SECRET_PREFIXES)
        or any(fragment in upper for fragment in _SECRET_FRAGMENTS)
    )


def forbidden_env_names(env: Mapping[str, str]) -> list[str]:
    """Names in ``env`` that look like secrets or bypass the lab's discovery roots."""
    return sorted(name for name in env if is_secret_name(name) or name in FORBIDDEN_NAMES)


def env_limitations(roots: LabRoots, credential_home: Path | None) -> tuple[Any, ...]:
    """Named limitations an environment built with ``credential_home`` carries."""
    from .sandbox import Limitation

    if credential_home is None:
        return ()
    return (
        Limitation(
            "omp_requires_real_home_for_credentials",
            "omp reports credentials not_configured under an isolated HOME",
            f"HOME={credential_home} (real); the credential store is read by reference. "
            f"Session dir and every discovery root stay under {roots.root}; real history "
            "stores are denied by the solver B sandbox profile, not by the environment.",
        ),
    )


def runtime_env(
    roots: LabRoots,
    actor: str,
    *,
    runtime_sessions: Path | None = None,
    model_cache: Path | None = None,
    credential_home: Path | None = None,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Complete explicit environment for one actor; the ambient environment is not copied.

    ``credential_home`` (the real home, read by reference for the runtime's credential store)
    replaces only ``HOME``; every discovery root still lives under the actor's lab home.
    """
    home = roots.actor_home(actor)
    for name, value in (extra or {}).items():
        if is_secret_name(name) or name in FORBIDDEN_NAMES:
            raise ValueError(
                f"refusing environment variable {name!r}: secrets reach tools only through "
                "files by reference, and discovery roots are set by the lab"
            )
        if not isinstance(value, str):
            raise ValueError(f"environment variable {name!r} must be a string")
    if not roots.tmp.is_dir():
        _mkdir_private(roots.tmp)
    sessions = runtime_sessions if runtime_sessions is not None else home / "omp-sessions"
    data_home = home / ".local" / "share"
    config_home = home / ".config"
    env: dict[str, str] = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        "TMPDIR": str(roots.tmp),
        "HOME": str(credential_home) if credential_home is not None else str(home),
        "SSGREP_DATA_DIR": str(roots.ssgrep_data),
        "CLAUDE_CONFIG_DIR": str(home / ".claude"),
        "CODEX_HOME": str(home / ".codex"),
        "CODEX_SESSIONS_DIR": str(home / ".codex" / "sessions"),
        "PI_CODING_AGENT_DIR": str(home / ".pi" / "agent"),
        "PI_SESSION_DIR": str(home / ".pi" / "agent" / "sessions"),
        "PI_CODING_AGENT_SESSION_DIR": str(home / ".pi" / "agent" / "sessions"),
        "PRIME_AGENT_CODING_AGENT_DIR": str(home / ".prime-agent" / "agent"),
        "PRIME_AGENT_SESSION_DIR": str(home / ".prime-agent" / "agent" / "sessions"),
        "PRIME_AGENT_CODING_AGENT_SESSION_DIR": str(home / ".prime-agent" / "agent" / "sessions"),
        "SSGREP_OMP_SESSIONS_DIR": str(sessions),
        "OMP_SESSIONS_DIR": str(sessions),
        "SSGREP_OPENCODE_DB": str(data_home / "opencode" / "opencode.db"),
        "XDG_DATA_HOME": str(data_home),
        "XDG_CONFIG_HOME": str(config_home),
        "OPENCODE_CONFIG_DIR": str(config_home / "opencode"),
    }
    if model_cache is not None:
        env.update(
            {"HF_HOME": str(model_cache), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        )
    env.update(extra or {})
    assert_isolated(env, roots, credential_home=credential_home)
    return env


def assert_isolated(
    env: Mapping[str, str],
    roots: LabRoots,
    *,
    allow: Sequence[str] = (),
    credential_home: Path | None = None,
) -> None:
    """Raise ``IsolationError`` if a discovery variable escapes the lab root.

    ``HF_HOME`` (a read-only model-cache reference) and any name in ``allow`` are exempt.
    ``HOME`` may equal ``credential_home`` (the brokered real home) and nothing else outside.
    """
    problems: list[str] = []
    for name in _CORE_VARS:
        if name not in env:
            problems.append(f"{name} is not set")
    for name in forbidden_env_names(env):
        problems.append(f"{name} must not be exported to a lab runtime")
    exempt = _READ_ONLY_REFERENCES | set(allow)
    if credential_home is not None and env.get("HOME") == str(credential_home):
        exempt = exempt | {"HOME"}
    real_root = os.path.realpath(roots.root)
    for name in DISCOVERY_VARS:
        value = env.get(name)
        if value is None or name in exempt:
            continue
        if not os.path.isabs(value):
            problems.append(f"{name}={value!r} is not absolute")
            continue
        real = os.path.realpath(value)
        if real != real_root and not real.startswith(real_root + os.sep):
            problems.append(f"{name} points outside the lab root")
    if problems:
        raise IsolationError("; ".join(problems))
