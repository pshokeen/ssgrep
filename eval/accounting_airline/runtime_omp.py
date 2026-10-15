"""Verified non-interactive ``omp`` adapter for the live lane (task 15.8, runtime half).

``omp`` is driven only through an argv list: the prompt is one argv element and is never
interpolated into a shell string. Flags are discovered from ``omp --help`` and refused by name
when absent; nothing about the CLI is assumed. Credentials never travel in argv or in the
environment passed here (the runtime reads its own credential store by reference).

A turn is deliberate and runs once: ``run_turn`` requires ``authorized=True``, never retries,
and returns only digests, counts and the parsed native-session summary, never prompt or
transcript text.
"""

from __future__ import annotations

import os
import re
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .bounded_capture import capture_process
from .observer import NativeSession, parse_native_omp, sha
from .roots import forbidden_env_names
from .sandbox import SANDBOX_EXEC, SandboxProfile, wrap_argv

RUNTIME = "omp"
LABEL = "omp-live"

_PROTOCOL_TIMEOUT = 30
NATIVE_STDOUT_LIMIT = 32 * 1024 * 1024
NATIVE_STDERR_LIMIT = 4 * 1024 * 1024
_PROBE_OUTPUT_LIMIT = 1024 * 1024
_REQUIRED_FLAGS = (
    "--session-dir",
    "--tools",
    "-p",
    "--print",
    "--cwd",
    "--continue",
    "--model",
    "--thinking",
    "--no-skills",
    "--no-rules",
    "--no-extensions",
    "--no-title",
    "--max-time",
    "--mode",
)
_VERSION = re.compile(r"\bomp(?:/| v)(\d+\.\d+\.\d+[0-9A-Za-z.+-]*)")
SAFE_TOOLS = frozenset({"bash", "read", "write", "edit", "grep", "glob", "find"})
_THINKING = frozenset({"off", "minimal", "low", "medium", "high", "xhigh", "max", "auto"})
_MAX_TIME = re.compile(r"^[1-9][0-9]*[smh]?$")
_EXTRA_FLAG = re.compile(r"^--[a-z][a-z0-9-]*(=.*)?$", re.DOTALL)

# Flags this adapter owns (a second copy would be ambiguous) or that route credentials,
# profiles, config overlays, code loading, or persistence away from the isolated session dir.
_MANAGED_FLAGS = frozenset(
    {
        "--print",
        "--session-dir",
        "--cwd",
        "--tools",
        "--no-tools",
        "--continue",
        "--model",
        "--thinking",
        "--no-skills",
        "--no-rules",
        "--no-extensions",
        "--no-title",
        "--max-time",
        "--mode",
        "--no-session",
        "--resume",
        "--restore",
        "--export",
        "--from-claude",
        "--from-codex",
    }
)
_FORBIDDEN_FLAGS = frozenset(
    {
        "--api-key",
        "--provider",
        "--profile",
        "--alias",
        "--config",
        "--hook",
        "--extension",
        "--add-dir",
        "--system-prompt",
        "--system-prompt-template",
        "--append-system-prompt",
        "--allow-home",
    }
)
_CREDENTIAL_FRAGMENTS = ("key", "token", "secret", "password", "credential", "auth")


class AdapterError(RuntimeError):
    """The omp CLI does not match the verified protocol, or a turn was refused."""


@dataclass(frozen=True)
class OmpProtocol:
    version: str
    flags_verified: tuple[str, ...]


def _flag_present(help_text: str, flag: str) -> bool:
    return re.search(rf"(?<![\w-]){re.escape(flag)}(?![\w-])", help_text) is not None


def _run_probe(argv: list[str]) -> str:
    try:
        done = capture_process(
            argv,
            env=os.environ,
            cwd=Path.cwd(),
            timeout=_PROTOCOL_TIMEOUT,
            stdout_limit=_PROBE_OUTPUT_LIMIT,
            stderr_limit=_PROBE_OUTPUT_LIMIT,
        )
    except OSError as exc:
        raise AdapterError(f"{argv[1]} probe failed: {type(exc).__name__}") from exc
    if done.status != "completed":
        raise AdapterError(f"{argv[1]} probe failed: {done.status}")
    if done.returncode != 0:
        raise AdapterError(f"{' '.join(argv)} exited {done.returncode}")
    return (done.stdout + done.stderr).decode("utf-8", errors="replace")


def verify_protocol(omp: str | None = None) -> OmpProtocol:
    """Discover the version and required flags from ``omp --version`` / ``omp --help``."""
    binary = shutil.which(omp or "omp")
    if binary is None:
        raise AdapterError(f"omp executable not found: {omp or 'omp'!r}")
    version_text = _run_probe([binary, "--version"])
    found = _VERSION.search(version_text)
    if found is None:
        raise AdapterError("omp --version did not report 'omp/<semver>' or 'omp v<semver>'")
    help_text = _run_probe([binary, "--help"])
    for flag in _REQUIRED_FLAGS:
        if not _flag_present(help_text, flag):
            raise AdapterError(f"omp --help does not list required flag {flag}")
    return OmpProtocol(found.group(1), _REQUIRED_FLAGS)


@dataclass(frozen=True)
class TurnSpec:
    prompt: str
    session_dir: Path
    cwd: Path
    tools: tuple[str, ...]
    model: str | None = None
    continue_session: bool = False
    max_time: str = "10m"
    thinking: str | None = None
    extra_flags: tuple[str, ...] = ()


def _check_extra_flag(flag: str) -> None:
    if not isinstance(flag, str) or not _EXTRA_FLAG.match(flag):
        raise AdapterError("extra flags must be '--name' or '--name=value'")
    name = flag.split("=", 1)[0]
    if name in _MANAGED_FLAGS:
        raise AdapterError(f"flag {name} is managed by the adapter and cannot be passed as extra")
    if name in _FORBIDDEN_FLAGS or any(part in name for part in _CREDENTIAL_FRAGMENTS):
        raise AdapterError(f"flag {name} is refused: credentials and config never travel in argv")


def build_argv(spec: TurnSpec, omp: str = "omp") -> list[str]:
    """The exact argv list for one turn; the prompt is a single element."""
    prompt = spec.prompt
    if not isinstance(prompt, str) or not prompt.strip():
        raise AdapterError("refusing an empty prompt")
    if "\x00" in prompt:
        raise AdapterError("refusing a prompt containing NUL")
    if prompt[0] in "-@":
        raise AdapterError("refusing a prompt starting with '-' or '@' (flag or file injection)")
    tools = tuple(spec.tools)
    if not tools:
        raise AdapterError("refusing an empty tools allowlist")
    for tool in tools:
        if tool not in SAFE_TOOLS:
            raise AdapterError(f"tool {tool!r} is outside the safe set {sorted(SAFE_TOOLS)}")
    if len(set(tools)) != len(tools):
        raise AdapterError("duplicate tool in allowlist")
    for label, path in (("session_dir", spec.session_dir), ("cwd", spec.cwd)):
        if not Path(path).is_absolute():
            raise AdapterError(f"{label} must be an absolute path")
    if not _MAX_TIME.match(spec.max_time):
        raise AdapterError(f"max_time {spec.max_time!r} is not <int>[s|m|h]")
    if spec.thinking is not None and spec.thinking not in _THINKING:
        raise AdapterError(f"thinking level {spec.thinking!r} is not one of {sorted(_THINKING)}")
    if spec.model is not None and (not spec.model or spec.model.startswith("-")):
        raise AdapterError("model must be a non-empty name not starting with '-'")
    for flag in spec.extra_flags:
        _check_extra_flag(flag)
    argv = [
        omp,
        "-p",
        prompt,
        "--session-dir",
        str(spec.session_dir),
        "--cwd",
        str(spec.cwd),
        "--tools",
        ",".join(tools),
        "--no-skills",
        "--no-rules",
        "--no-extensions",
        "--no-title",
        "--max-time",
        spec.max_time,
    ]
    if spec.continue_session:
        argv.append("--continue")
    if spec.model is not None:
        argv += ["--model", spec.model]
    if spec.thinking is not None:
        argv += ["--thinking", spec.thinking]
    argv.extend(spec.extra_flags)
    return argv


def describe_transport(tools: Sequence[str]) -> str:
    """How ssgrep reaches the model: the live run uses the ssgrep CLI on PATH via bash."""
    mcp = any("mcp" in tool.lower() for tool in tools)
    bash = "bash" in tools
    if bash and not mcp:
        return "bash-cli"
    if mcp and bash:
        return "bash-cli+mcp"
    if mcp:
        return "mcp"
    return "none"


def refuse_live_without_authorization(authorize_live: bool) -> None:
    if authorize_live is not True:
        raise AdapterError("live omp turn refused: explicit authorization was not given")


# --------------------------------------------------------------------------- native capture


@dataclass
class NativeCapture:
    session_id: str | None
    session_files: tuple[Path, ...]
    file_digests: dict[str, str]
    complete_bytes: dict[str, int]
    partial_tail_bytes: dict[str, int]
    usage_records: int
    usage_total: dict[str, int]
    cost_nano_usd: int | None
    models: tuple[str, ...]
    tool_calls: int
    user_turns: int
    runtime_labels: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "session_files": [str(path) for path in self.session_files],
            "file_digests": dict(self.file_digests),
            "complete_bytes": dict(self.complete_bytes),
            "partial_tail_bytes": dict(self.partial_tail_bytes),
            "usage_records": self.usage_records,
            "usage_total": dict(self.usage_total),
            "cost_nano_usd": self.cost_nano_usd,
            "models": list(self.models),
            "tool_calls": self.tool_calls,
            "user_turns": self.user_turns,
            "runtime_labels": list(self.runtime_labels),
        }


def _session_files(session_dir: Path) -> list[Path]:
    found: list[Path] = []
    if not session_dir.is_dir():
        return found
    for current, dirs, names in os.walk(session_dir, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not Path(current, d).is_symlink())
        for name in sorted(names):
            path = Path(current, name)
            if name.endswith(".jsonl") and not path.is_symlink() and path.is_file():
                found.append(path)
    return sorted(found)


def capture_native(session_dir: Path) -> NativeCapture:
    """Summarize every native ``*.jsonl`` under ``session_dir`` via the observer's parser."""
    files = _session_files(session_dir)
    sessions: list[tuple[Path, NativeSession]] = [(p, parse_native_omp(p)) for p in files]
    digests: dict[str, str] = {}
    complete: dict[str, int] = {}
    partial: dict[str, int] = {}
    usage_total: dict[str, int] = {}
    usage_records = tool_calls = user_turns = 0
    cost: int | None = None
    models: list[str] = []
    labels: list[str] = []
    for path, session in sessions:
        key = path.relative_to(session_dir).as_posix()
        digests[key] = session.file_digest
        complete[key] = session.complete_bytes
        partial[key] = session.partial_tail_bytes
        if session.model is not None and session.model not in models:
            models.append(session.model)
        if session.runtime_label is not None and session.runtime_label not in labels:
            labels.append(session.runtime_label)
        for record in session.records:
            if record.kind == "usage":
                usage_records += 1
                for name, value in record.usage.items():
                    usage_total[name] = usage_total.get(name, 0) + value
                if record.cost_nano_usd is not None:
                    cost = (cost or 0) + record.cost_nano_usd
            elif record.kind == "tool_call":
                tool_calls += 1
            elif record.kind == "user_turn":
                user_turns += 1
    return NativeCapture(
        session_id=sessions[0][1].session_id if sessions else None,
        session_files=tuple(files),
        file_digests=digests,
        complete_bytes=complete,
        partial_tail_bytes=partial,
        usage_records=usage_records,
        usage_total=usage_total,
        cost_nano_usd=cost,
        models=tuple(models),
        tool_calls=tool_calls,
        user_turns=user_turns,
        runtime_labels=tuple(labels),
    )


# ------------------------------------------------------------------------------- one turn


@dataclass
class TurnResult:
    """Public metadata only: chars count retained prefixes, bytes count observed pipe data.

    First-stop status is independent of observed truncation and pipe EOF. Native persistence
    and billing are independent of captured stdout; stopped captures are never complete.
    """

    argv_digest: str
    prompt_sha256: str
    returncode: int | None
    timed_out: bool
    elapsed_ms: int
    stdout_chars: int
    stderr_chars: int
    native: NativeCapture
    label: str = field(default=LABEL)
    pid: int | None = None
    stdout_bytes: int = 0
    stderr_bytes: int = 0
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    output_limit: bool = False
    output_limit_streams: tuple[str, ...] = ()
    capture_status: str = "completed"
    capture_complete: bool = True
    pipes_drained: bool = True
    termination_cause: str | None = None
    observed_over_limit_streams: tuple[str, ...] = ()
    stdout_limit_bytes: int = NATIVE_STDOUT_LIMIT
    stderr_limit_bytes: int = NATIVE_STDERR_LIMIT

    def to_dict(self) -> dict[str, Any]:
        return {
            "argv_digest": self.argv_digest,
            "prompt_sha256": self.prompt_sha256,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "elapsed_ms": self.elapsed_ms,
            "stdout_chars": self.stdout_chars,
            "stderr_chars": self.stderr_chars,
            "native": self.native.to_dict(),
            "label": self.label,
            "pid": self.pid,
            "stdout_bytes": self.stdout_bytes,
            "stderr_bytes": self.stderr_bytes,
            "stdout_truncated": self.stdout_truncated,
            "stderr_truncated": self.stderr_truncated,
            "output_limit": self.output_limit,
            "output_limit_streams": list(self.output_limit_streams),
            "capture_status": self.capture_status,
            "capture_complete": self.capture_complete,
            "pipes_drained": self.pipes_drained,
            "termination_cause": self.termination_cause,
            "observed_over_limit_streams": list(self.observed_over_limit_streams),
            "stdout_limit_bytes": self.stdout_limit_bytes,
            "stderr_limit_bytes": self.stderr_limit_bytes,
        }


def argv_digest(argv: Sequence[str]) -> str:
    return sha("\x00".join(argv))


def _sandbox_binary(env: Mapping[str, str]) -> str:
    if os.path.exists(SANDBOX_EXEC):
        return SANDBOX_EXEC
    return shutil.which("sandbox-exec", path=env.get("PATH")) or SANDBOX_EXEC


def _chars(data: bytes | None) -> int:
    return len(data.decode("utf-8", errors="replace")) if data else 0


def run_turn(
    spec: TurnSpec,
    *,
    env: Mapping[str, str],
    profile: SandboxProfile | None,
    timeout: float,
    omp: str = "omp",
    authorized: bool = False,
    output_dir: Path | None = None,
) -> TurnResult:
    """Run one omp turn exactly once; native session files are captured afterwards."""
    refuse_live_without_authorization(authorized)
    problems = forbidden_env_names(env)
    if problems:
        raise AdapterError(f"refusing environment variables {problems}: credentials stay in files")
    argv = build_argv(spec, omp)
    digest = argv_digest(argv)
    command = wrap_argv(profile, argv) if profile is not None else list(argv)
    if profile is not None:
        command[0] = _sandbox_binary(env)
    spec.session_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    started = time.monotonic_ns()
    try:
        captured = capture_process(
            command,
            env=env,
            cwd=spec.cwd,
            timeout=timeout,
            stdout_limit=NATIVE_STDOUT_LIMIT,
            stderr_limit=NATIVE_STDERR_LIMIT,
            output_dir=output_dir,
        )
    except OSError as exc:
        raise AdapterError(f"could not capture {command[0]!r}: {type(exc).__name__}") from exc
    elapsed_ms = (time.monotonic_ns() - started) // 1_000_000
    return TurnResult(
        argv_digest=digest,
        prompt_sha256=sha(spec.prompt),
        returncode=None if captured.timed_out else captured.returncode,
        timed_out=captured.timed_out,
        elapsed_ms=elapsed_ms,
        stdout_chars=_chars(captured.stdout),
        stderr_chars=_chars(captured.stderr),
        native=capture_native(spec.session_dir),
        pid=captured.pid,
        stdout_bytes=captured.stdout_bytes,
        stderr_bytes=captured.stderr_bytes,
        stdout_truncated=captured.stdout_truncated,
        stderr_truncated=captured.stderr_truncated,
        output_limit=captured.output_limit,
        output_limit_streams=captured.output_limit_streams,
        capture_status=captured.status,
        capture_complete=captured.capture_complete,
        pipes_drained=captured.pipes_drained,
        termination_cause=captured.termination_cause,
        observed_over_limit_streams=captured.observed_over_limit_streams,
        stdout_limit_bytes=captured.stdout_limit_bytes,
        stderr_limit_bytes=captured.stderr_limit_bytes,
    )
