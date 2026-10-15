"""Hard byte-bounded subprocess capture for verifier-owned native/tool processes.

Counts are actual pipe bytes observed, not reconstructed child-produced totals. Retained
memory and exclusive 0600 artifacts never exceed either stream ceiling. Newly owned groups
are cleaned even after a normal leader exit; overflow triggers immediate kill/reaping.
Nested bridge tools remain in the outer owner's group: their direct child is reaped before
a private marker requests whole-group cleanup. Normal nested exits never signal that group.
No stdout parsing, model retry, credential handling, or persisted-native-session mutation.
These are required-native-lane boundaries, not a global boundedness claim for optional
matched/scripted-stand-in entry points.
The first termination cause stays authoritative while post-kill bytes remain counted.
``capture_complete`` excludes stopped/truncated captures; ``pipes_drained`` reports EOF
separately. Early shim snapshots are explicitly distinguished from final observations.
"""

from __future__ import annotations

import os
import selectors
import signal
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Event, Thread
from typing import IO, Any, cast

_CHUNK = 64 * 1024
_REAP_TIMEOUT = 10


@dataclass(frozen=True)
class CaptureResult:
    returncode: int | None
    pid: int
    stdout: bytes = field(repr=False)
    stderr: bytes = field(repr=False)
    stdout_bytes: int
    stderr_bytes: int
    stdout_limit_bytes: int
    stderr_limit_bytes: int
    timed_out: bool
    capture_complete: bool
    external_output_limit: bool = False
    cancelled: bool = False
    termination_cause: str | None = None
    limit_trigger_streams: tuple[str, ...] = ()
    pipes_drained: bool = False

    @property
    def stdout_truncated(self) -> bool:
        return self.stdout_bytes > len(self.stdout)

    @property
    def stderr_truncated(self) -> bool:
        return self.stderr_bytes > len(self.stderr)

    @property
    def observed_over_limit_streams(self) -> tuple[str, ...]:
        return tuple(name for name in ("stdout", "stderr") if getattr(self, name + "_truncated"))

    @property
    def output_limit_streams(self) -> tuple[str, ...]:
        return self.limit_trigger_streams if self.output_limit else ()

    @property
    def output_limit(self) -> bool:
        return self.termination_cause == "output_limit"

    @property
    def status(self) -> str:
        return self.termination_cause or "completed"

    def metadata(self) -> dict[str, Any]:
        return {
            "returncode": self.returncode,
            "pid": self.pid,
            "timed_out": self.timed_out,
            "cancelled": self.cancelled,
            "capture_status": self.status,
            "capture_complete": self.capture_complete,
            "termination_cause": self.termination_cause,
            "pipes_drained": self.pipes_drained,
            "observed_over_limit_streams": list(self.observed_over_limit_streams),
            "output_limit": self.output_limit,
            "output_limit_streams": list(self.output_limit_streams),
            **{
                name: getattr(self, name)
                for name in (
                    "stdout_bytes",
                    "stderr_bytes",
                    "stdout_limit_bytes",
                    "stderr_limit_bytes",
                    "stdout_truncated",
                    "stderr_truncated",
                )
            },
        }


def _kill_owned_group(proc: subprocess.Popen[bytes], start_new_session: bool = True) -> None:
    if not start_new_session:
        proc.kill()  # Nested tools remain in the outer supervisor's owned group.
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        proc.kill()


def _reap_after_setup_failure(proc: subprocess.Popen[bytes], start_new_session: bool) -> None:
    if proc.returncode is None:
        _kill_owned_group(proc, start_new_session)
        proc.wait(timeout=_REAP_TIMEOUT)


def _read_failure_marker(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None
    with os.fdopen(fd, "rb") as handle:
        data = handle.read(65)
    if data in (
        b"output_limit",
        b"timeout",
        b"cancelled",
        b"invalid_output_encoding",
        b"interrupted",
    ):
        return data.decode("ascii")
    return "capture_error"


def capture_process(
    argv: str
    | bytes
    | os.PathLike[str]
    | os.PathLike[bytes]
    | Sequence[str | bytes | os.PathLike[str] | os.PathLike[bytes]],
    *,
    env: Mapping[str, str],
    cwd: Path,
    timeout: float | None,
    stdout_limit: int,
    stderr_limit: int,
    input_bytes: bytes | None = None,
    output_dir: Path | None = None,
    start_new_session: bool = True,
    on_output_limit: Callable[[CaptureResult], None] | None = None,
    failure_marker: Path | None = None,
    cancel_event: Event | None = None,
    process: subprocess.Popen[Any] | None = None,
    capture_stdin: bool = True,
) -> CaptureResult:
    """Run once, fairly drain both pipes and nonblocking stdin, and always reap our child."""
    if stdout_limit < 0 or stderr_limit < 0 or (timeout is not None and timeout <= 0):
        raise ValueError("capture limits must be nonnegative and any timeout positive")
    limits = {"stdout": stdout_limit, "stderr": stderr_limit}
    counts = dict.fromkeys(limits, 0)
    buffers = {name: bytearray() for name in limits}
    with ExitStack() as stack:
        if process is not None:
            stack.callback(_reap_after_setup_failure, process, start_new_session)
        files = {}
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            output_dir.chmod(0o700)
            for name in limits:
                fd = os.open(
                    output_dir / f"{name}.txt",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                )
                files[name] = stack.enter_context(os.fdopen(fd, "wb"))
        if process is None:
            proc = subprocess.Popen(
                argv,
                env=dict(env),
                cwd=cwd,
                start_new_session=start_new_session,
                stdin=subprocess.PIPE if input_bytes else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stack.callback(_reap_after_setup_failure, proc, start_new_session)
        else:
            proc = process
        selector = stack.enter_context(selectors.DefaultSelector())
        for name in limits:
            pipe = stack.enter_context(getattr(proc, name))
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, selectors.EVENT_READ, name)
        pending = memoryview(input_bytes or b"")
        if capture_stdin and proc.stdin is not None:
            stack.enter_context(proc.stdin)
            os.set_blocking(proc.stdin.fileno(), False)
            selector.register(proc.stdin, selectors.EVENT_WRITE, "stdin")
        deadline = time.monotonic() + timeout if timeout is not None else None
        cleanup_deadline = None
        timed_out = False
        external_limit = False
        cancelled = False
        complete = False
        termination_cause = None
        limit_trigger_streams = ()
        try:
            while selector.get_map() or proc.poll() is None:
                now = time.monotonic()
                if cleanup_deadline is None and cancel_event is not None and cancel_event.is_set():
                    termination_cause = "cancelled"
                    cancelled = True
                    _kill_owned_group(proc, start_new_session)
                    cleanup_deadline = now + _REAP_TIMEOUT
                marker_cause = _read_failure_marker(failure_marker)
                if marker_cause is not None:
                    external_limit = external_limit or marker_cause == "output_limit"
                    if termination_cause is None:
                        termination_cause = marker_cause
                        timed_out = marker_cause == "timeout"
                        cancelled = marker_cause == "cancelled"
                        limit_trigger_streams = (
                            ("inner_tool",) if marker_cause == "output_limit" else ()
                        )
                        _kill_owned_group(proc, start_new_session)
                        cleanup_deadline = now + _REAP_TIMEOUT
                child_code = proc.poll()
                if termination_cause is None and child_code is not None and child_code < 0:
                    termination_cause = "interrupted"
                    _kill_owned_group(proc, start_new_session)
                    cleanup_deadline = now + _REAP_TIMEOUT
                if cleanup_deadline is None and deadline is not None and now >= deadline:
                    timed_out = True
                    termination_cause = "timeout"
                    _kill_owned_group(proc, start_new_session)
                    cleanup_deadline = now + _REAP_TIMEOUT
                if cleanup_deadline is not None and now >= cleanup_deadline:
                    break
                until = cleanup_deadline if cleanup_deadline is not None else deadline
                for key, _mask in selector.select(
                    0.1 if until is None else min(0.1, max(0, until - now))
                ):
                    pipe, name = cast(IO[Any], key.fileobj), key.data
                    if name == "stdin":
                        try:
                            written = os.write(pipe.fileno(), pending[:_CHUNK])
                            pending = pending[written:]
                        except BrokenPipeError:
                            pending = memoryview(b"")
                        except BlockingIOError:
                            continue
                        if not pending:
                            selector.unregister(pipe)
                            pipe.close()
                        continue
                    amount = (
                        _CHUNK
                        if cleanup_deadline is not None
                        else min(_CHUNK, max(1, limits[name] - counts[name] + 1))
                    )
                    try:
                        chunk = os.read(pipe.fileno(), amount)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(pipe)
                        pipe.close()
                        continue
                    counts[name] += len(chunk)
                    kept = chunk[: max(0, limits[name] - len(buffers[name]))]
                    buffers[name].extend(kept)
                    if name in files:
                        files[name].write(kept)
                    if cleanup_deadline is None and counts[name] > limits[name]:
                        termination_cause = "output_limit"
                        limit_trigger_streams = (name,)
                        _kill_owned_group(proc, start_new_session)
                        proc.wait(timeout=_REAP_TIMEOUT)
                        if on_output_limit is not None:
                            for handle in files.values():
                                handle.flush()
                            on_output_limit(
                                CaptureResult(
                                    proc.returncode,
                                    proc.pid,
                                    bytes(buffers["stdout"]),
                                    bytes(buffers["stderr"]),
                                    counts["stdout"],
                                    counts["stderr"],
                                    stdout_limit,
                                    stderr_limit,
                                    timed_out,
                                    False,
                                    termination_cause=termination_cause,
                                    limit_trigger_streams=limit_trigger_streams,
                                )
                            )
                        cleanup_deadline = time.monotonic() + _REAP_TIMEOUT
            marker_cause = _read_failure_marker(failure_marker)
            if marker_cause is not None:
                external_limit = external_limit or marker_cause == "output_limit"
                if termination_cause is None:
                    termination_cause = marker_cause
                    timed_out = marker_cause == "timeout"
                    cancelled = marker_cause == "cancelled"
                    limit_trigger_streams = (
                        ("inner_tool",) if marker_cause == "output_limit" else ()
                    )
                    _kill_owned_group(proc, start_new_session)
            complete = not any(key.data != "stdin" for key in selector.get_map().values())
        finally:
            # A successful leader can leave DEVNULL-background descendants behind.
            # Only newly owned groups are cleaned here; nested tools inherit the outer owner.
            if start_new_session or not complete or proc.poll() is None:
                _kill_owned_group(proc, start_new_session)
            # Closing after SIGKILL cannot leave a blocked writer alive in our owned group.
            for key in list(selector.get_map().values()):
                selector.unregister(key.fileobj)
                cast(IO[Any], key.fileobj).close()
            proc.wait(timeout=_REAP_TIMEOUT)
        if termination_cause is None and proc.returncode is not None and proc.returncode < 0:
            termination_cause = "interrupted"
        return CaptureResult(
            proc.returncode,
            proc.pid,
            bytes(buffers["stdout"]),
            bytes(buffers["stderr"]),
            counts["stdout"],
            counts["stderr"],
            stdout_limit,
            stderr_limit,
            timed_out,
            complete
            and termination_cause is None
            and proc.returncode is not None
            and proc.returncode >= 0,
            external_limit,
            cancelled,
            termination_cause=termination_cause,
            limit_trigger_streams=limit_trigger_streams,
            pipes_drained=complete,
        )


DEFAULT_STDOUT_LIMIT = 32 * 1024 * 1024
DEFAULT_STDERR_LIMIT = 4 * 1024 * 1024


class OutputLimitExceeded(subprocess.SubprocessError):
    """A bounded execution failed its capture boundary, irrespective of child exit code."""

    def __init__(self, capture: CaptureResult):
        self.capture = capture
        super().__init__(
            f"subprocess output_limit ({','.join(capture.output_limit_streams)}); "
            f"stdout_bytes={capture.stdout_bytes}/{capture.stdout_limit_bytes}; "
            f"stderr_bytes={capture.stderr_bytes}/{capture.stderr_limit_bytes}"
        )


def text_encoding() -> str:
    """Match stdlib text=True, including UTF-8 Mode, without changing process locale."""
    import locale

    return locale.getpreferredencoding(False)


def decode_text(data: bytes, *, encoding: str | None = None) -> str:
    """Decode a subprocess text stream with stdlib universal-newline handling."""
    return (
        data.decode(text_encoding() if encoding is None else encoding)
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )


def run_bounded(
    argv: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    timeout: float = 180,
    input: str | bytes | None = None,
    text: bool = False,
    check: bool = False,
    stdout_limit: int | None = None,
    stderr_limit: int | None = None,
    output_dir: Path | None = None,
) -> subprocess.CompletedProcess:
    """Bounded subprocess.run equivalent; oversize is an explicit failure, not truncation."""
    encoding = text_encoding()
    captured = capture_process(
        argv,
        env=os.environ if env is None else env,
        cwd=Path.cwd() if cwd is None else Path(cwd),
        timeout=timeout,
        input_bytes=input.encode(encoding) if isinstance(input, str) else input,
        stdout_limit=DEFAULT_STDOUT_LIMIT if stdout_limit is None else stdout_limit,
        stderr_limit=DEFAULT_STDERR_LIMIT if stderr_limit is None else stderr_limit,
        output_dir=output_dir,
    )
    if captured.output_limit:
        raise OutputLimitExceeded(captured)
    if captured.timed_out:
        raise subprocess.TimeoutExpired(
            argv, timeout, output=captured.stdout, stderr=captured.stderr
        )
    stdout: str | bytes = captured.stdout
    stderr: str | bytes = captured.stderr
    if text:
        stdout = decode_text(captured.stdout, encoding=encoding)
        stderr = decode_text(captured.stderr, encoding=encoding)
    assert captured.returncode is not None  # capture_process reaped the child before returning.
    result = subprocess.CompletedProcess(argv, captured.returncode, stdout, stderr)
    if check:
        result.check_returncode()
    return result


class AsyncCapture:
    """Drain immediately; the owner must choose a launch timeout or explicit ``None``.

    With ``None``, only the owner's finish/abort deadlines terminate the capture.
    """

    def __init__(
        self,
        process: subprocess.Popen[Any],
        *,
        timeout: float | None,
        stdout_limit: int | None = None,
        stderr_limit: int | None = None,
    ):
        self.process = process
        self.stop = Event()
        self.result: CaptureResult | None = None
        self.error: BaseException | None = None

        def drain() -> None:
            try:
                self.result = capture_process(
                    process.args,
                    env={},
                    cwd=Path.cwd(),
                    timeout=timeout,
                    stdout_limit=DEFAULT_STDOUT_LIMIT if stdout_limit is None else stdout_limit,
                    stderr_limit=DEFAULT_STDERR_LIMIT if stderr_limit is None else stderr_limit,
                    process=process,
                    capture_stdin=False,
                    cancel_event=self.stop,
                )
            except BaseException as exc:
                self.error = exc

        self.thread = Thread(target=drain, daemon=True)
        self.thread.start()

    def check_output_limit(self) -> None:
        if self.error is not None:
            raise self.error
        if self.result is not None and self.result.output_limit:
            raise OutputLimitExceeded(self.result)

    def finish(self, timeout: float = 60, *, cancel: bool = False) -> CaptureResult:
        if cancel:
            self.stop.set()
        self.thread.join(timeout)
        timed_out = self.thread.is_alive()
        if timed_out:
            self.stop.set()
            self.thread.join(_REAP_TIMEOUT + 1)
        if self.thread.is_alive():
            raise RuntimeError("bounded capture could not finish owned-child cleanup")
        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass  # This child is already reaped; no input can still be delivered.
            self.process.stdin = None
        self.check_output_limit()
        if self.result is None:
            raise RuntimeError("bounded capture finished without a process result")
        if timed_out and not cancel and self.result.termination_cause in (None, "cancelled"):
            self.result = replace(
                self.result,
                timed_out=True,
                cancelled=False,
                termination_cause="timeout",
                capture_complete=False,
            )
        return self.result

    def abort(self) -> CaptureResult:
        return self.finish(timeout=_REAP_TIMEOUT + 1, cancel=True)
