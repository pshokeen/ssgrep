"""omp adapter: real protocol discovery, exact argv, and a labelled test-double for every run.

No test starts a real omp agent turn. Every execution below uses ``fake-omp-test-double``, a
script created in ``tmp_path`` whose every artifact records ``labRuntime: "test-double"``.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from eval.accounting_airline import runtime_omp
from eval.accounting_airline.roots import LabRoots, runtime_env
from eval.accounting_airline.runtime_omp import (
    LABEL,
    RUNTIME,
    AdapterError,
    OmpProtocol,
    TurnSpec,
    argv_digest,
    build_argv,
    capture_native,
    describe_transport,
    refuse_live_without_authorization,
    run_turn,
    verify_protocol,
)
from eval.accounting_airline.sandbox import availability, profile_for_solver_b

_LIMITATION = availability()
needs_sandbox = pytest.mark.skipif(
    _LIMITATION is not None,
    reason=f"sandbox-exec unavailable: {_LIMITATION.reason if _LIMITATION else ''}",
)

DOUBLE_NAME = "fake-omp-test-double"
PROMPT_CANARY = "PROMPT-CANARY-7f3a9c"
HOSTILE = 'it\'s "q"\nline2 $(touch pwned) `touch pwned2`; rm -rf / && echo $HOME \\ \t end'

DOUBLE_SOURCE = r'''#!{python}
"""TEST DOUBLE for omp (labRuntime: test-double). Never contacts a model."""
import json
import os
import pathlib
import sys
import time

args = sys.argv[1:]
here = pathlib.Path(__file__).resolve().parent
if args == ["--version"]:
    print("omp/0.0.0-test-double")
    print("omp v0.0.0-test-double")
    sys.exit(0)
if args == ["--help"]:
    print((here / "help.txt").read_text())
    sys.exit(0)


def opt(name):
    return args[args.index(name) + 1] if name in args else None


mode = os.environ.get("FAKE_OMP_MODE", "ok")
counter = os.environ.get("FAKE_OMP_COUNTER")
if counter:
    with open(counter, "a") as handle:
        handle.write("invoked\n")
prompt = args[args.index("-p") + 1]
sdir = pathlib.Path(opt("--session-dir"))
sdir.mkdir(parents=True, exist_ok=True)
observed = {{
    "labRuntime": "test-double",
    "argv": args,
    "env_keys": sorted(os.environ),
    "cwd": os.getcwd(),
    "canary_seen": "AMBIENT_CANARY_SECRET_TOKEN" in os.environ,
}}


def write_observed():
    path = sdir / f"observed-{{time.time_ns()}}.json"
    path.write_text(json.dumps(observed))


target = os.environ.get("FAKE_OMP_READ")
if target:
    try:
        with open(target) as handle:
            observed["read"] = "ok:" + handle.read()
    except OSError as exc:
        observed["read"] = "denied:" + type(exc).__name__
        write_observed()
        sys.exit(7)


def line(obj):
    return json.dumps(obj, separators=(",", ":")) + "\n"


stamp = "2026-10-13T00:00:0{{n}}.000Z"
if "--continue" in args:
    files = sorted(sdir.glob("*.jsonl"))
    if not files:
        observed["error"] = "no session to continue"
        write_observed()
        sys.exit(9)
    path = files[0]
    existing = path.read_text().count("\n")
else:
    path = sdir / "2026-10-13T00-00-00-000Z_double-session-1.jsonl"
    existing = 0
    with open(path, "a") as handle:
        handle.write(line({{"type": "session", "id": "double-session-1", "cwd": os.getcwd(),
                           "labRuntime": "test-double"}}))
n = existing + 1
user = {{"type": "message", "id": f"u{{n}}", "timestamp": stamp.format(n=n % 10),
        "message": {{"role": "user", "content": [{{"type": "text", "text": prompt}}]}}}}
usage = {{"input": 100, "output": 20, "cacheRead": 5, "cacheWrite": 3, "totalTokens": 128,
         "reasoningTokens": 7}}
if os.environ.get("FAKE_OMP_COST"):
    usage["cost"] = {{"total": float(os.environ["FAKE_OMP_COST"])}}
assistant = {{"type": "message", "id": f"a{{n}}", "timestamp": stamp.format(n=n % 10),
             "message": {{"role": "assistant", "model": "test-double-model", "usage": usage,
                         "content": [{{"type": "text", "text": "ok"}}]}}}}
with open(path, "a") as handle:
    handle.write(line(user))
    if mode == "timeout":
        handle.write('{{"type":"message","id":"partial-tail"')
        handle.flush()
        os.fsync(handle.fileno())
        write_observed()
        time.sleep(60)
    handle.write(line(assistant))
write_observed()
sys.exit(3 if mode == "fail" else 0)
'''


HELP_ROWS = {
    "--model": "      --model=<value>   Model to use",
    "--thinking": "      --thinking=<value> Reasoning effort",
    "--cwd": "      --cwd=<value>     Directory to start in",
    "--mode": "      --mode=<value>    Output mode",
    "-p": "  -p, --print           Non-interactive mode",
    "--continue": "  -c, --continue        Continue previous session",
    "--session-dir": "      --session-dir=<value>  Directory for session storage",
    "--no-session": "      --no-session      Don't save session",
    "--tools": "      --tools=<value>   Comma-separated list of tools",
    "--no-extensions": "      --no-extensions   Disable extension discovery",
    "--no-skills": "      --no-skills       Disable skills",
    "--no-rules": "      --no-rules        Disable rules",
    "--no-title": "      --no-title        Disable title",
    "--max-time": "      --max-time=<value>  Stop after this duration",
    "--plan-yolo": "      --plan-yolo       Unrelated flag (prefix of nothing)",
}


def _help_text(omit: str | None = None) -> str:
    rows = [row for flag, row in HELP_ROWS.items() if flag != omit or flag == "-p"]
    if omit == "-p":  # -p stays absent while --print remains listed
        rows = [row.replace("-p, ", "    ") for row in rows]
    return "\n".join(rows) + "\n"


def make_double(directory: Path, help_text: str | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / DOUBLE_NAME
    script.write_text(DOUBLE_SOURCE.format(python=sys.executable))
    script.chmod(0o755)
    (directory / "help.txt").write_text(help_text if help_text is not None else _help_text())
    return script


def _env(**extra: str) -> dict[str, str]:
    base = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"}
    base.update(extra)
    return base


def _spec(tmp_path: Path, prompt: str = "do the thing", **overrides: Any) -> TurnSpec:
    session_dir = tmp_path / "sessions"
    cwd = tmp_path / "work"
    cwd.mkdir(exist_ok=True)
    fields: dict[str, Any] = {
        "prompt": prompt,
        "session_dir": session_dir,
        "cwd": cwd,
        "tools": ("bash", "read"),
        "model": None,
        "continue_session": False,
        "max_time": "10m",
        "thinking": None,
    }
    fields.update(overrides)
    return TurnSpec(**fields)


def _observed(session_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(session_dir.glob("observed-*.json"))]


# ------------------------------------------------------------------- verify_protocol


def test_verify_protocol_on_the_real_omp() -> None:
    if shutil.which("omp") is None:
        pytest.skip("omp is not installed on this host")
    protocol = verify_protocol()
    assert isinstance(protocol, OmpProtocol)
    assert protocol.version and protocol.version[0].isdigit()
    assert RUNTIME == "omp"
    for flag in (
        "--session-dir", "--tools", "-p", "--print", "--cwd", "--continue", "--model",
        "--no-skills", "--no-rules", "--no-extensions", "--no-title", "--max-time", "--mode",
    ):  # fmt: skip
        assert flag in protocol.flags_verified


def test_verify_protocol_with_a_complete_fake_double(tmp_path: Path) -> None:
    double = make_double(tmp_path / "bin")
    protocol = verify_protocol(str(double))
    assert protocol.version == "0.0.0-test-double"
    assert "--session-dir" in protocol.flags_verified


@pytest.mark.parametrize(
    "missing", ["--session-dir", "--tools", "--no-skills", "--max-time", "--mode", "-p"]
)
def test_verify_protocol_names_the_missing_flag(tmp_path: Path, missing: str) -> None:
    double = make_double(tmp_path / "bin", _help_text(missing))
    with pytest.raises(AdapterError) as caught:
        verify_protocol(str(double))
    assert f"required flag {missing}" in str(caught.value)


def test_verify_protocol_rejects_unparseable_version_and_missing_binary(tmp_path: Path) -> None:
    script = tmp_path / DOUBLE_NAME
    script.write_text("#!/bin/sh\necho 'something 1.2'\n")
    script.chmod(0o755)
    with pytest.raises(AdapterError, match="version"):
        verify_protocol(str(script))
    with pytest.raises(AdapterError, match="not found"):
        verify_protocol(str(tmp_path / "nope"))


# ----------------------------------------------------------------------- build_argv


def test_build_argv_exact_shape(tmp_path: Path) -> None:
    spec = _spec(tmp_path, model="some/model", continue_session=True, thinking="low")
    argv = build_argv(spec, "/x/omp")
    assert argv == [
        "/x/omp", "-p", "do the thing",
        "--session-dir", str(tmp_path / "sessions"),
        "--cwd", str(tmp_path / "work"),
        "--tools", "bash,read",
        "--no-skills", "--no-rules", "--no-extensions", "--no-title",
        "--max-time", "10m",
        "--continue", "--model", "some/model", "--thinking", "low",
    ]  # fmt: skip
    bare = build_argv(_spec(tmp_path))
    assert bare[0] == "omp"
    assert "--continue" not in bare and "--model" not in bare and "--thinking" not in bare


def test_hostile_prompt_is_one_unmodified_argv_element(tmp_path: Path) -> None:
    argv = build_argv(_spec(tmp_path, prompt=HOSTILE))
    assert argv[2] == HOSTILE
    assert argv.count(HOSTILE) == 1
    assert all(isinstance(item, str) for item in argv)
    assert argv[:2] == ["omp", "-p"]
    assert argv[3] == "--session-dir"  # nothing was split off the prompt


def test_extra_flag_auto_approve_allowed_only_when_explicit(tmp_path: Path) -> None:
    assert "--auto-approve" not in build_argv(_spec(tmp_path))
    argv = build_argv(_spec(tmp_path, extra_flags=("--auto-approve",)))
    assert argv[-1] == "--auto-approve"
    assert build_argv(_spec(tmp_path, extra_flags=("--approval-mode=write",)))[-1] == (
        "--approval-mode=write"
    )


@pytest.mark.parametrize(
    "flag",
    [
        "--api-key", "--api-key=sk-secret", "--provider=openai", "--profile=x", "--config=/x.yml",
        "--hook=/x.ts", "--system-prompt=hi", "--add-dir=/", "--session-dir=/elsewhere",
        "--no-session", "--tools=all", "--resume=abc", "-x", "auto-approve", "--Upper", "",
    ],
)  # fmt: skip
def test_forbidden_extra_flags_are_refused(tmp_path: Path, flag: str) -> None:
    with pytest.raises(AdapterError):
        build_argv(_spec(tmp_path, extra_flags=(flag,)))


@pytest.mark.parametrize(
    "tools", [("bash", "rm"), ("mcp__ssgrep",), ("browser",), (), ("bash", "bash")]
)
def test_unsafe_tools_are_refused(tmp_path: Path, tools: tuple[str, ...]) -> None:
    with pytest.raises(AdapterError):
        build_argv(_spec(tmp_path, tools=tools))


def test_every_safe_tool_is_accepted(tmp_path: Path) -> None:
    tools = ("bash", "read", "write", "edit", "grep", "glob", "find")
    assert build_argv(_spec(tmp_path, tools=tools))[8] == ",".join(tools)


@pytest.mark.parametrize("prompt", ["", "   \n\t", "-p evil", "@/etc/passwd", "a\x00b"])
def test_bad_prompts_are_refused(tmp_path: Path, prompt: str) -> None:
    with pytest.raises(AdapterError):
        build_argv(_spec(tmp_path, prompt=prompt))


def test_other_spec_fields_are_validated(tmp_path: Path) -> None:
    cases: list[dict[str, Any]] = [
        {"max_time": "10 minutes"}, {"max_time": "0"}, {"thinking": "extreme"},
        {"model": "--api-key"}, {"model": ""}, {"session_dir": Path("relative")},
    ]  # fmt: skip
    for overrides in cases:
        with pytest.raises(AdapterError):
            build_argv(_spec(tmp_path, **overrides))
    assert build_argv(_spec(tmp_path, max_time="600"))[-1] == "600"


def test_describe_transport() -> None:
    assert describe_transport(("bash", "read")) == "bash-cli"
    assert describe_transport(("read",)) == "none"
    assert describe_transport(("bash", "mcp__ssgrep__search")) == "bash-cli+mcp"
    assert describe_transport(("mcp__ssgrep__search",)) == "mcp"


def test_argv_digest_is_nul_joined_sha256() -> None:
    import hashlib

    argv = ["omp", "-p", "x y"]
    assert argv_digest(argv) == hashlib.sha256(b"omp\x00-p\x00x y").hexdigest()


# ------------------------------------------------------------------------- run_turn


def test_refusal_without_authorization_and_positive_twin(tmp_path: Path) -> None:
    counter = tmp_path / "counter"
    double = make_double(tmp_path / "bin")
    env = _env(FAKE_OMP_COUNTER=str(counter))
    with pytest.raises(AdapterError, match="authorization"):
        run_turn(_spec(tmp_path), env=env, profile=None, timeout=30, omp=str(double))
    with pytest.raises(AdapterError, match="authorization"):
        run_turn(
            _spec(tmp_path), env=env, profile=None, timeout=30, omp=str(double), authorized=False
        )
    assert not counter.exists()  # the double never ran
    with pytest.raises(AdapterError):
        refuse_live_without_authorization(False)
    assert refuse_live_without_authorization(True) is None
    result = run_turn(
        _spec(tmp_path), env=env, profile=None, timeout=30, omp=str(double), authorized=True
    )
    assert result.returncode == 0 and counter.read_text() == "invoked\n"


def test_secret_looking_env_is_refused_before_running(tmp_path: Path) -> None:
    counter = tmp_path / "counter"
    double = make_double(tmp_path / "bin")
    env = _env(FAKE_OMP_COUNTER=str(counter), OPENAI_API_KEY="sk-nope")
    with pytest.raises(AdapterError, match="OPENAI_API_KEY"):
        run_turn(
            _spec(tmp_path), env=env, profile=None, timeout=30, omp=str(double), authorized=True
        )
    assert not counter.exists()


def test_run_turn_records_argv_explicit_env_and_no_ambient_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AMBIENT_CANARY_SECRET_TOKEN", "ambient-canary-value")
    double = make_double(tmp_path / "bin")
    env = _env(FAKE_OMP_COST="0.0123")
    spec = _spec(tmp_path, prompt=HOSTILE, model="m/x")
    result = run_turn(spec, env=env, profile=None, timeout=30, omp=str(double), authorized=True)

    expected = build_argv(spec, str(double))
    assert result.argv_digest == argv_digest(expected)
    assert result.returncode == 0 and not result.timed_out and result.label == LABEL == "omp-live"
    (seen,) = _observed(spec.session_dir)
    assert seen["labRuntime"] == "test-double"
    assert seen["argv"] == expected[1:]  # hostile prompt arrived element-identical
    assert seen["argv"][1] == HOSTILE
    assert not (spec.cwd / "pwned").exists() and not (spec.cwd / "pwned2").exists()
    assert seen["cwd"] == str(spec.cwd.resolve())
    assert set(seen["env_keys"]) - set(env) <= {"__CF_USER_TEXT_ENCODING", "LC_CTYPE"}
    assert set(env) <= set(seen["env_keys"])
    assert seen["canary_seen"] is False
    assert "AMBIENT_CANARY_SECRET_TOKEN" not in seen["env_keys"]

    native = result.native
    assert native.session_id == "double-session-1"
    assert native.user_turns == 1 and native.usage_records == 1
    assert native.usage_total == {
        "input": 100, "output": 20, "cache_read": 5, "cache_write": 3, "total": 128,
        "reasoning": 7,
    }  # fmt: skip
    assert native.cost_nano_usd == 12_300_000
    assert native.models == ("test-double-model",)
    assert native.runtime_labels == ("test-double",)
    assert result.stdout_chars == 0 and result.stderr_chars == 0


def test_result_contains_no_raw_prompt_text(tmp_path: Path) -> None:
    double = make_double(tmp_path / "bin")
    prompt = f"please remember {PROMPT_CANARY} for later"
    spec = _spec(tmp_path, prompt=prompt)
    result = run_turn(spec, env=_env(), profile=None, timeout=30, omp=str(double), authorized=True)
    # the double's own native transcript does hold it (as a real runtime's would) ...
    assert PROMPT_CANARY in next(iter(spec.session_dir.glob("*.jsonl"))).read_text()
    # ... but nothing the adapter returns does.
    assert PROMPT_CANARY not in json.dumps(result.to_dict())
    assert PROMPT_CANARY not in json.dumps(asdict(result), default=str)
    assert PROMPT_CANARY not in repr(result)
    import hashlib

    assert result.prompt_sha256 == hashlib.sha256(prompt.encode()).hexdigest()


def test_continue_appends_a_second_user_turn_to_the_same_session_file(tmp_path: Path) -> None:
    double = make_double(tmp_path / "bin")
    first = _spec(tmp_path, prompt="first turn")
    one = run_turn(first, env=_env(), profile=None, timeout=30, omp=str(double), authorized=True)
    second = _spec(tmp_path, prompt="second turn", continue_session=True)
    two = run_turn(second, env=_env(), profile=None, timeout=30, omp=str(double), authorized=True)

    assert one.native.user_turns == 1
    assert two.native.user_turns == 2 and two.native.usage_records == 2
    assert len(two.native.session_files) == 1
    assert two.native.session_files == one.native.session_files
    assert two.native.session_id == one.native.session_id == "double-session-1"
    assert two.native.usage_total["input"] == 200
    assert two.native.cost_nano_usd is None  # no record carried a cost
    (name,) = two.native.file_digests
    assert two.native.file_digests[name] != one.native.file_digests[name]
    assert two.native.complete_bytes[name] > one.native.complete_bytes[name]
    assert two.native.partial_tail_bytes[name] == 0
    assert "--continue" in _observed(first.session_dir)[1]["argv"]
    assert "--continue" not in _observed(first.session_dir)[0]["argv"]


def test_timeout_kills_and_keeps_the_partial_capture(tmp_path: Path) -> None:
    counter = tmp_path / "counter"
    double = make_double(tmp_path / "bin")
    spec = _spec(tmp_path)
    env = _env(FAKE_OMP_MODE="timeout", FAKE_OMP_COUNTER=str(counter))
    result = run_turn(spec, env=env, profile=None, timeout=3, omp=str(double), authorized=True)
    assert result.timed_out is True and result.returncode is None
    assert result.elapsed_ms < 30_000
    assert counter.read_text() == "invoked\n"  # never retried
    (name,) = result.native.file_digests
    assert result.native.user_turns == 1 and result.native.usage_records == 0
    assert result.native.partial_tail_bytes[name] == len('{"type":"message","id":"partial-tail"')
    assert result.native.complete_bytes[name] > 0


def test_nonzero_exit_is_retained_and_never_retried(tmp_path: Path) -> None:
    counter = tmp_path / "counter"
    double = make_double(tmp_path / "bin")
    env = _env(FAKE_OMP_MODE="fail", FAKE_OMP_COUNTER=str(counter))
    result = run_turn(
        _spec(tmp_path), env=env, profile=None, timeout=30, omp=str(double), authorized=True
    )
    assert result.returncode == 3 and not result.timed_out
    assert counter.read_text() == "invoked\n"  # exactly once
    assert result.native.usage_records == 1  # evidence of the failed attempt is retained


def test_missing_executable_is_an_adapter_error(tmp_path: Path) -> None:
    with pytest.raises(AdapterError, match="could not start"):
        run_turn(
            _spec(tmp_path), env=_env(), profile=None, timeout=5,
            omp=str(tmp_path / "absent" / DOUBLE_NAME), authorized=True,
        )  # fmt: skip


# ---------------------------------------------------------------------------- sandbox


@pytest.fixture
def lab(tmp_path: Path):
    roots = LabRoots.create(parent=tmp_path)
    try:
        yield roots
    finally:
        if roots.root.exists():
            roots.remove()


@needs_sandbox
def test_sandboxed_double_writes_runtime_b_but_cannot_read_runtime_a(
    tmp_path: Path, lab: LabRoots
) -> None:
    secret = lab.runtime_a / "solver-a-transcript.jsonl"
    secret.write_text("SOLVER-A-TRANSCRIPT-CANARY")
    double = make_double(tmp_path / "bin")
    profile = profile_for_solver_b(lab, real_home=tmp_path / "no-home")
    sessions = lab.runtime_b / "home" / "omp-sessions"
    base_env = runtime_env(lab, "solver-b")

    # positive twin: sandboxed, writing only below runtime-b, succeeds
    ok = TurnSpec("hello", sessions, lab.project, ("bash",), None, False, "10m", None)
    allowed = run_turn(
        ok, env=base_env, profile=profile, timeout=60, omp=str(double), authorized=True
    )
    assert allowed.returncode == 0
    assert allowed.native.user_turns == 1
    assert all(path.is_relative_to(lab.runtime_b) for path in allowed.native.session_files)

    # control: the same read, unsandboxed, succeeds (the denial below is the sandbox's doing)
    read_env = {**base_env, "FAKE_OMP_READ": str(secret)}
    control_dir = lab.runtime_b / "control-sessions"
    control = TurnSpec("probe", control_dir, lab.project, ("bash",), None, False, "10m", None)
    result = run_turn(
        control, env=read_env, profile=None, timeout=60, omp=str(double), authorized=True
    )
    assert result.returncode == 0
    assert _observed(control_dir)[0]["read"] == "ok:SOLVER-A-TRANSCRIPT-CANARY"

    # sandboxed: the read of runtime-a is a real denial
    denied_dir = lab.runtime_b / "denied-sessions"
    denied = TurnSpec("probe", denied_dir, lab.project, ("bash",), None, False, "10m", None)
    blocked = run_turn(
        denied, env=read_env, profile=profile, timeout=60, omp=str(double), authorized=True
    )
    assert blocked.returncode == 7
    assert _observed(denied_dir)[0]["read"] == "denied:PermissionError"
    assert "SOLVER-A-TRANSCRIPT-CANARY" not in json.dumps(blocked.to_dict())


@needs_sandbox
def test_sandboxed_timeout_kills_the_whole_process_group(tmp_path: Path, lab: LabRoots) -> None:
    double = make_double(tmp_path / "bin")
    profile = profile_for_solver_b(lab, real_home=tmp_path / "no-home")
    sessions = lab.runtime_b / "home" / "omp-sessions"
    spec = TurnSpec("slow", sessions, lab.project, ("bash",), None, False, "10m", None)
    env = {**runtime_env(lab, "solver-b"), "FAKE_OMP_MODE": "timeout"}
    result = run_turn(spec, env=env, profile=profile, timeout=4, omp=str(double), authorized=True)
    assert result.timed_out and result.returncode is None
    assert result.elapsed_ms < 30_000
    assert result.native.partial_tail_bytes[next(iter(result.native.file_digests))] > 0


# ------------------------------------------------------------------------ capture_native


def _record(kind: str, ident: str, **parts: object) -> str:
    base: dict[str, object] = {"type": "message", "id": ident,
                               "timestamp": "2026-10-13T00:00:00.000Z"}  # fmt: skip
    if kind == "assistant":
        base["message"] = {"role": "assistant", **parts}
    else:
        base["message"] = {"role": "user", "content": "hi"}
    return json.dumps(base) + "\n"


def _header(session_id: str) -> str:
    return json.dumps({"type": "session", "id": session_id, "cwd": "/w"}) + "\n"


def test_capture_sums_usage_exactly_and_cost_is_none_when_absent(tmp_path: Path) -> None:
    session = tmp_path / "s" / "nested" / "a.jsonl"
    session.parent.mkdir(parents=True)
    session.write_text(
        _header("sess-1")
        + _record("user", "u1")
        + _record(
            "assistant", "a1", model="m1", usage={"input": 10, "output": 2, "totalTokens": 12}
        )
        + _record(
            "assistant",
            "a2",
            model="m2",
            usage={
                "input": 7,
                "output": 3,
                "cacheRead": 4,
                "cacheWrite": 1,
                "totalTokens": 10,
                "reasoningTokens": 2,
            },
            content=[
                {"type": "toolCall", "id": "c1", "name": "bash", "arguments": {"command": "ls"}}
            ],
        )  # fmt: skip
    )
    capture = capture_native(tmp_path / "s")
    assert capture.session_id == "sess-1"
    assert capture.usage_records == 2
    assert capture.usage_total == {
        "input": 17, "output": 5, "total": 22, "cache_read": 4, "cache_write": 1, "reasoning": 2,
    }  # fmt: skip
    assert capture.cost_nano_usd is None
    assert capture.tool_calls == 1 and capture.user_turns == 1
    assert capture.models == ("m2",)  # observer keeps the last model per file
    assert capture.session_files == (session,)
    assert list(capture.file_digests) == ["nested/a.jsonl"]


def test_capture_cost_present_is_summed_to_integer_nano_usd(tmp_path: Path) -> None:
    session = tmp_path / "a.jsonl"
    session.write_text(
        _header("sess-2")
        + _record("assistant", "a1", model="m", usage={"input": 1, "cost": {"total": 0.000123456}})
        + _record("assistant", "a2", model="m", usage={"input": 1, "cost": {"total": 0.5}})
        + _record("assistant", "a3", model="m", usage={"input": 1})
    )
    capture = capture_native(tmp_path)
    assert capture.cost_nano_usd == 123_456 + 500_000_000
    assert isinstance(capture.cost_nano_usd, int)
    assert capture.usage_records == 3 and capture.usage_total == {"input": 3}


def test_capture_zero_cost_is_distinct_from_absent(tmp_path: Path) -> None:
    (tmp_path / "a.jsonl").write_text(
        _header("s") + _record("assistant", "a1", model="m", usage={"cost": {"total": 0}})
    )
    assert capture_native(tmp_path).cost_nano_usd == 0


def test_capture_counts_a_partial_tail_apart_from_complete_bytes(tmp_path: Path) -> None:
    complete = _header("sess-3") + _record("user", "u1")
    tail = '{"type":"message","id":"cut'
    (tmp_path / "a.jsonl").write_text(complete + tail)
    capture = capture_native(tmp_path)
    assert capture.complete_bytes == {"a.jsonl": len(complete.encode())}
    assert capture.partial_tail_bytes == {"a.jsonl": len(tail)}
    assert capture.user_turns == 1


def test_capture_empty_missing_and_symlinks(tmp_path: Path) -> None:
    empty = capture_native(tmp_path / "missing")
    assert empty.session_id is None and empty.session_files == ()
    assert empty.usage_records == 0 and empty.cost_nano_usd is None and empty.usage_total == {}
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "x.jsonl").write_text(_header("outside") + _record("user", "u"))
    inside = tmp_path / "inside"
    inside.mkdir()
    (inside / "link.jsonl").symlink_to(outside / "x.jsonl")
    (inside / "dirlink").symlink_to(outside, target_is_directory=True)
    capture = capture_native(inside)
    assert capture.session_files == () and capture.user_turns == 0


def test_module_does_not_touch_the_ambient_environment_or_real_stores() -> None:
    source = Path(runtime_omp.__file__).read_text()
    assert "os.environ" not in source  # the env passed to run_turn is the only env
    assert "shell=True" not in source
    assert os.path.basename(runtime_omp.__file__) == "runtime_omp.py"
