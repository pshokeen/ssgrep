"""Solver B filesystem sandbox: real ``sandbox-exec`` denials with positive twins."""

from __future__ import annotations

import os
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from eval.accounting_airline import sandbox
from eval.accounting_airline.gates import BLOCKED, FAIL, PASS
from eval.accounting_airline.roots import LabRoots
from eval.accounting_airline.sandbox import (
    GATE_NAME,
    Limitation,
    SandboxProfile,
    availability,
    profile_for_solver_b,
    prove_isolation,
    run_sandboxed,
    sbpl_quote,
    wrap_argv,
)

_LIMITATION = availability()
needs_sandbox = pytest.mark.skipif(
    _LIMITATION is not None,
    reason=f"sandbox-exec unavailable: {_LIMITATION.reason if _LIMITATION else ''}",
)
ENV = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"}


@pytest.fixture
def lab(tmp_path: Path):
    roots = LabRoots.create(parent=tmp_path)
    try:
        yield roots
    finally:
        if roots.root.exists():
            roots.remove()


def _cat(profile: SandboxProfile | None, path: Path) -> subprocess.CompletedProcess[str]:
    if profile is None:
        return subprocess.run(
            ["/bin/cat", str(path)], env=ENV, capture_output=True, text=True, check=False
        )
    return run_sandboxed(profile, ["/bin/cat", str(path)], env=ENV, cwd="/", timeout=30)


# ------------------------------------------------------------------------ availability


def test_availability_reports_a_named_limitation_or_none() -> None:
    result = availability()
    assert result is None or (
        isinstance(result, Limitation)
        and result.name == "sandbox_exec_unavailable"
        and result.reason
    )


def test_unavailable_sandbox_yields_a_blocked_gate_naming_the_limitation(
    lab: LabRoots, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = profile_for_solver_b(lab)
    monkeypatch.setenv("PATH", "/nonexistent")
    monkeypatch.setattr(sandbox, "SANDBOX_EXEC", "/nonexistent/sandbox-exec")
    limitation = availability()  # real discovery, with the binary genuinely absent
    assert limitation is not None and limitation.name == "sandbox_exec_unavailable"
    gate = prove_isolation(lab, profile)
    assert gate.name == GATE_NAME and gate.status == BLOCKED
    assert "sandbox_exec_unavailable" in gate.detail
    assert gate.evidence["limitation"]["name"] == "sandbox_exec_unavailable"


# --------------------------------------------------------------------------- profile


def test_profile_text_and_limitations(lab: LabRoots, tmp_path: Path) -> None:
    profile = profile_for_solver_b(lab, real_home=tmp_path / "no-home")
    assert profile.text.startswith("(version 1)\n(allow default)\n")
    expected = {
        Path(os.path.realpath(p))
        for p in (lab.runtime_a, lab.verifier, lab.artifacts, lab.tools / "admin")
    }
    assert expected <= set(profile.denied)
    for path in expected:
        assert f"(deny file-read* file-write* (subpath {sbpl_quote(path)}))" in profile.text
    # allowed trees are never mentioned
    for allowed in (lab.project, lab.runtime_b, lab.ssgrep_data, lab.tools / "solver"):
        assert str(allowed) not in profile.text
    assert f'"{lab.tools}"' not in profile.text  # tools/ itself stays readable
    names = {item.name for item in profile.limitations}
    assert {
        "network_not_restricted",
        "process_inspection_not_blocked",
        "provider_credential_store_readable_by_design",
    } <= names
    assert all(item.reason and item.detail for item in profile.limitations)


def test_sbpl_quote_escapes_and_rejects_control_characters() -> None:
    assert sbpl_quote('/tmp/a"b') == '"/tmp/a\\"b"'
    assert sbpl_quote("/tmp/a\\b") == '"/tmp/a\\\\b"'
    assert sbpl_quote(Path("/plain/path")) == '"/plain/path"'
    hostile = '/tmp/x")(allow file-read* (subpath "/'
    quoted = sbpl_quote(hostile)
    assert quoted == '"/tmp/x\\")(allow file-read* (subpath \\"/"'
    # nothing inside the literal can close it: every interior quote is preceded by a backslash
    interior = quoted[1:-1]
    assert all(interior[i - 1] == "\\" for i, ch in enumerate(interior) if ch == '"')
    for bad in ("/tmp/a\nb", "/tmp/a\rb", "/tmp/a\x00b", "/tmp/a\tb", "/tmp/a\x7fb"):
        with pytest.raises(ValueError, match="control characters|NUL"):
            sbpl_quote(bad)


def test_profile_rejects_control_characters_in_extra_denied(lab: LabRoots) -> None:
    with pytest.raises(ValueError, match="control"):
        profile_for_solver_b(lab, extra_denied=[Path("/tmp/evil\nname")])
    profile_for_solver_b(lab, extra_denied=[Path("/tmp/fine-name")])  # positive twin


def test_wrap_argv_is_an_argv_list_and_never_a_shell_string(lab: LabRoots) -> None:
    profile = profile_for_solver_b(lab)
    argv = wrap_argv(profile, ["/bin/echo", "a b; rm -rf /"])
    assert argv == ["sandbox-exec", "-p", profile.text, "/bin/echo", "a b; rm -rf /"]
    with pytest.raises(TypeError):
        wrap_argv(profile, "/bin/echo hi")  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- proof


@needs_sandbox
def test_prove_isolation_passes_on_this_machine(lab: LabRoots) -> None:
    gate = prove_isolation(lab, profile_for_solver_b(lab))
    assert gate.name == "isolation.solver_b_filesystem"
    assert gate.status == PASS, gate.detail
    assert [item["label"] for item in gate.evidence["denied"]] == [
        "runtime_a",
        "verifier",
        "artifacts",
        "tools_admin",
    ]
    assert all(item["returncode"] != 0 and not item["leaked"] for item in gate.evidence["denied"])
    assert all(item["readable"] for item in gate.evidence["allowed"])
    assert gate.evidence["parent_listing_ok"] is True
    names = {item["name"] for item in gate.evidence["limitations"]}
    assert "network_not_restricted" in names
    # canaries are deleted afterwards
    leftovers = [p.name for p in lab.root.rglob("*") if "canary" in p.name or "probe" in p.name]
    assert leftovers == []


@needs_sandbox
def test_weakened_profile_fails_naming_each_denied_directory(lab: LabRoots) -> None:
    weak = SandboxProfile("(version 1)(allow default)", denied=(), limitations=())
    gate = prove_isolation(lab, weak)
    assert gate.status == FAIL
    for label in ("runtime_a", "verifier", "artifacts", "tools_admin"):
        assert f"{label}: sandboxed read of the canary succeeded" in gate.detail
    assert "runtime_a: sandboxed write was not denied" in gate.detail
    # the control reads all worked, so FAIL is about the profile, not file permissions
    assert "control:" not in gate.detail
    # canaries and write probes are still cleaned up after a failing proof
    assert [p for p in lab.root.rglob("*") if "canary" in p.name or "probe" in p.name] == []


@needs_sandbox
def test_overbroad_profile_fails_its_allowed_twins(lab: LabRoots) -> None:
    broad = SandboxProfile(
        f'(version 1)(allow default)(deny file-read* file-write* (subpath "{lab.root}"))',
        denied=(),
        limitations=(),
    )
    gate = prove_isolation(lab, broad)
    assert gate.status == FAIL
    assert "project: sandboxed read of an allowed canary failed" in gate.detail
    assert "runtime_b: sandboxed write to an allowed directory failed" in gate.detail


@needs_sandbox
def test_partial_profile_failure_names_only_the_missing_denial(lab: LabRoots) -> None:
    full = profile_for_solver_b(lab)
    admin = os.path.realpath(lab.tools / "admin")
    kept = "\n".join(line for line in full.text.splitlines() if admin not in line)
    gate = prove_isolation(lab, replace(full, text=kept + "\n"))
    assert gate.status == FAIL
    assert "tools_admin: sandboxed read of the canary succeeded" in gate.detail
    assert "runtime_a: sandboxed read" not in gate.detail  # the intact denials still hold


@needs_sandbox
def test_denied_and_allowed_reads_directly(lab: LabRoots) -> None:
    profile = profile_for_solver_b(lab)
    secret = lab.runtime_a / "transcript.jsonl"
    secret.write_text("A-PRIVATE-TRANSCRIPT")
    admin_cred = lab.tools / "admin" / "pgpass"
    admin_cred.write_text("ADMIN-CREDENTIAL")
    solver_tool = lab.tools / "solver" / "psql-wrapper"
    solver_tool.write_text("SOLVER-TOOL")
    brief = lab.project / "brief.txt"
    brief.write_text("PUBLIC-BRIEF")
    for path, text in ((secret, "A-PRIVATE-TRANSCRIPT"), (admin_cred, "ADMIN-CREDENTIAL")):
        assert text in _cat(None, path).stdout  # file permissions alone would allow it
        denied = _cat(profile, path)
        assert denied.returncode != 0 and text not in denied.stdout + denied.stderr
        assert "not permitted" in denied.stderr
    for path, text in ((solver_tool, "SOLVER-TOOL"), (brief, "PUBLIC-BRIEF")):
        allowed = _cat(profile, path)
        assert allowed.returncode == 0 and text in allowed.stdout


@needs_sandbox
def test_real_history_stores_are_denied_but_credential_store_stays_readable(
    lab: LabRoots, tmp_path: Path
) -> None:
    home = tmp_path / "fake-real-home"
    stores = {
        ".omp/agent/sessions": "omp-history",
        ".claude": "claude-history",
        ".codex": "codex-history",
        ".pi": "pi-history",
        ".prime": "prime-history",
        ".local/share/opencode": "opencode-history",
        ".ssgrep": "ssgrep-index",
    }
    for relative, text in stores.items():
        (home / relative).mkdir(parents=True)
        (home / relative / "history.txt").write_text(text)
    credential = home / ".omp" / "agent" / "agent.db"
    credential.write_text("BROKERED-CREDENTIAL-REFERENCE")
    config = home / ".omp" / "agent" / "config.yml"
    config.write_text("model: x")
    profile = profile_for_solver_b(lab, real_home=home)
    for relative, text in stores.items():
        target = home / relative / "history.txt"
        assert _cat(None, target).stdout == text
        denied = _cat(profile, target)
        assert denied.returncode != 0 and text not in denied.stdout + denied.stderr, relative
    assert Path(os.path.realpath(home / ".claude")) in profile.denied
    for path, text in ((credential, "BROKERED-CREDENTIAL-REFERENCE"), (config, "model: x")):
        allowed = _cat(profile, path)
        assert allowed.returncode == 0 and text in allowed.stdout
    # A protected store created after policy construction must still be denied.
    future_home = tmp_path / "empty-home"
    sparse = profile_for_solver_b(lab, real_home=future_home)
    future_history = future_home / ".claude" / "history.txt"
    future_history.parent.mkdir(parents=True)
    future_history.write_text("FUTURE-HISTORY-CANARY")
    assert _cat(None, future_history).stdout == "FUTURE-HISTORY-CANARY"
    denied = _cat(sparse, future_history)
    assert denied.returncode != 0
    assert "FUTURE-HISTORY-CANARY" not in denied.stdout + denied.stderr


@needs_sandbox
def test_docker_sockets_are_denied_unless_opted_out(lab: LabRoots, tmp_path: Path) -> None:
    home = tmp_path / "sock-home"
    (home / ".docker" / "run").mkdir(parents=True)
    fake_socket = home / ".docker" / "run" / "docker.sock"
    fake_socket.write_text("DOCKER-SOCKET-STAND-IN")
    real = Path(os.path.realpath(fake_socket))
    profile = profile_for_solver_b(lab, real_home=home)
    assert real in profile.denied
    assert _cat(None, fake_socket).stdout == "DOCKER-SOCKET-STAND-IN"
    denied = _cat(profile, fake_socket)
    assert denied.returncode != 0 and "DOCKER-SOCKET-STAND-IN" not in denied.stdout
    opened = profile_for_solver_b(lab, real_home=home, deny_docker_sockets=False)
    assert real not in opened.denied
    twin = _cat(opened, fake_socket)
    assert twin.returncode == 0 and "DOCKER-SOCKET-STAND-IN" in twin.stdout


@needs_sandbox
def test_extra_denied_paths_are_enforced(lab: LabRoots, tmp_path: Path) -> None:
    private = tmp_path / "extra-private"
    private.mkdir()
    (private / "f.txt").write_text("EXTRA")
    profile = profile_for_solver_b(lab, extra_denied=[private])
    assert _cat(None, private / "f.txt").stdout == "EXTRA"
    assert _cat(profile, private / "f.txt").returncode != 0
    assert _cat(profile_for_solver_b(lab), private / "f.txt").stdout == "EXTRA"


@needs_sandbox
def test_hostile_path_components_cannot_break_out_of_the_profile(tmp_path: Path) -> None:
    hostile_parent = tmp_path / 'we"ird\\par ent'
    hostile_parent.mkdir()
    roots = LabRoots.create(parent=hostile_parent)
    try:
        profile = profile_for_solver_b(roots)
        runtime_a = os.path.realpath(roots.runtime_a)
        assert sbpl_quote(runtime_a) in profile.text
        assert 'we\\"ird\\\\par ent' in profile.text
        # every line is still exactly one well-formed deny form (no injected extra rules)
        rules = [line for line in profile.text.splitlines()[2:] if line]
        assert all(line.startswith("(deny ") and line.endswith("))") for line in rules)
        gate = prove_isolation(roots, profile)
        assert gate.status == PASS, gate.detail
    finally:
        roots.remove()


@needs_sandbox
def test_run_sandboxed_uses_the_given_env_and_cwd(lab: LabRoots) -> None:
    profile = profile_for_solver_b(lab)
    env = {**ENV, "LAB_PROBE": "visible"}
    result = run_sandboxed(
        profile,
        ["/bin/sh", "-c", 'echo "$LAB_PROBE:$(pwd -P)"'],
        env=env,
        cwd=lab.project,
        timeout=30,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == f"visible:{lab.project}"
    # the sandbox applies to children of the shell too
    nested = run_sandboxed(
        profile,
        ["/bin/sh", "-c", f'/bin/cat "{lab.runtime_a}/x" 2>&1; ls "{lab.verifier}" 2>&1'],
        env=env,
        cwd=lab.project,
        timeout=30,
    )
    assert "not permitted" in nested.stdout
