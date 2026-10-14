"""Lab roots: owner-only layout, ownership-checked removal, explicit isolated environments."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from eval.accounting_airline import roots as roots_mod
from eval.accounting_airline.roots import (
    ACTORS,
    DISCOVERY_VARS,
    SUBDIRS,
    IsolationError,
    LabRoots,
    RootOwnershipError,
    assert_isolated,
    forbidden_env_names,
    runtime_env,
)

SIX_ROOT_VARS = (
    "CLAUDE_CONFIG_DIR",
    "CODEX_HOME",
    "CODEX_SESSIONS_DIR",
    "PI_CODING_AGENT_DIR",
    "PI_SESSION_DIR",
    "PRIME_AGENT_CODING_AGENT_DIR",
    "PRIME_AGENT_SESSION_DIR",
    "SSGREP_OMP_SESSIONS_DIR",
    "OMP_SESSIONS_DIR",
    "SSGREP_OPENCODE_DB",
    "XDG_DATA_HOME",
    "OPENCODE_CONFIG_DIR",
)


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


@pytest.fixture
def lab(tmp_path: Path):
    roots = LabRoots.create(parent=tmp_path)
    try:
        yield roots
    finally:
        if roots.root.exists():
            roots.remove()


# ------------------------------------------------------------------------------ create


def test_create_is_owner_only_everywhere_and_writes_marker(lab: LabRoots) -> None:
    assert lab.root == Path(os.path.realpath(lab.root))
    assert lab.root.name == f"ssgrep-airline-lab-{lab.run_id}"
    paths = [lab.root, lab.collect_view, lab.tools / "admin", lab.tools / "solver", lab.tmp]
    paths += [lab.root / name for name in SUBDIRS]
    paths += [lab.actor_home(actor) for actor in ACTORS]
    for path in paths:
        assert path.is_dir() and not path.is_symlink(), path
        assert _mode(path) == 0o700, path
    marker = json.loads(lab.marker.read_text())
    assert marker["run_id"] == lab.run_id
    assert marker["uid"] == os.getuid()
    assert marker["root"] == str(lab.root)
    assert marker["pid"] == os.getpid() and marker["created_utc"]
    assert _mode(lab.marker) == 0o600
    lab.verify_ownership()  # positive twin of every refusal below


def test_actor_homes_follow_the_layout(lab: LabRoots) -> None:
    assert lab.actor_home("solver-a") == lab.runtime_a / "home"
    assert lab.actor_home("solver-b") == lab.runtime_b / "home"
    assert lab.actor_home("verifier") == lab.actor_home("control") == lab.verifier / "home"
    with pytest.raises(ValueError, match="unknown actor"):
        lab.actor_home("nobody")


def test_default_parent_is_the_real_tmpdir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    link = tmp_path / "via-link"
    real = tmp_path / "real"
    real.mkdir()
    link.symlink_to(real)
    monkeypatch.setenv("TMPDIR", str(link))
    roots = LabRoots.create()
    try:
        assert roots.root.parent == Path(os.path.realpath(real))
    finally:
        roots.remove()


def test_create_refuses_nonempty_symlinked_or_loose_existing_roots(tmp_path: Path) -> None:
    run_id = "reuse0001"
    existing = tmp_path / f"ssgrep-airline-lab-{run_id}"
    existing.mkdir(mode=0o700)
    (existing / "stranger.txt").write_text("not ours")
    with pytest.raises(RootOwnershipError, match="not empty"):
        LabRoots.create(parent=tmp_path, run_id=run_id)
    assert (existing / "stranger.txt").read_text() == "not ours"

    outside = tmp_path / "elsewhere"
    outside.mkdir(mode=0o700)
    link_id = "link00001"
    (tmp_path / f"ssgrep-airline-lab-{link_id}").symlink_to(outside)
    with pytest.raises(RootOwnershipError, match="symlink"):
        LabRoots.create(parent=tmp_path, run_id=link_id)
    assert list(outside.iterdir()) == []  # nothing was created through the link

    loose_id = "loose0001"
    loose = tmp_path / f"ssgrep-airline-lab-{loose_id}"
    loose.mkdir()
    loose.chmod(0o755)
    with pytest.raises(RootOwnershipError, match="0700"):
        LabRoots.create(parent=tmp_path, run_id=loose_id)

    # positive twin: an existing EMPTY, owned, 0700 directory is accepted
    empty_id = "empty0001"
    (tmp_path / f"ssgrep-airline-lab-{empty_id}").mkdir(mode=0o700)
    roots = LabRoots.create(parent=tmp_path, run_id=empty_id)
    roots.verify_ownership()
    roots.remove()


def test_create_rejects_unsafe_run_ids(tmp_path: Path) -> None:
    for bad in ("../escape", "has space", "UPPER", "a"):
        with pytest.raises(ValueError, match="unsafe run id"):
            LabRoots.create(parent=tmp_path, run_id=bad)


# --------------------------------------------------------------------------- ownership


def test_verify_ownership_detects_marker_tamper_and_recovers(lab: LabRoots) -> None:
    original = lab.marker.read_text()
    marker = json.loads(original)
    for key, bad in (("run_id", "someone-else"), ("uid", os.getuid() + 1), ("root", "/nowhere")):
        lab.marker.write_text(json.dumps({**marker, key: bad}))
        with pytest.raises(RootOwnershipError, match=key):
            lab.verify_ownership()
    lab.marker.write_text("{not json")
    with pytest.raises(RootOwnershipError, match="JSON"):
        lab.verify_ownership()
    lab.marker.write_text(original)
    lab.verify_ownership()


def test_verify_ownership_rejects_loose_mode_and_missing_marker(lab: LabRoots) -> None:
    lab.root.chmod(0o755)
    with pytest.raises(RootOwnershipError, match="0700"):
        lab.verify_ownership()
    lab.root.chmod(0o700)
    lab.marker.rename(lab.marker.with_name("moved"))
    with pytest.raises(RootOwnershipError, match="marker"):
        lab.verify_ownership()
    lab.marker.with_name("moved").rename(lab.marker)
    lab.verify_ownership()


def test_verify_ownership_rejects_a_symlink_root(tmp_path: Path) -> None:
    real = LabRoots.create(parent=tmp_path)
    try:
        link_parent = tmp_path / "links"
        link_parent.mkdir()
        link = link_parent / real.root.name
        link.symlink_to(real.root)
        fake = LabRoots(root=link, run_id=real.run_id)
        with pytest.raises(RootOwnershipError, match="symlink"):
            fake.verify_ownership()
        real.verify_ownership()
    finally:
        real.remove()


def test_verify_ownership_rejects_wrong_name(lab: LabRoots) -> None:
    with pytest.raises(RootOwnershipError, match="named"):
        LabRoots(root=lab.root, run_id="otherrun1").verify_ownership()


def test_remove_refuses_and_deletes_nothing_when_marker_mismatched(lab: LabRoots) -> None:
    (lab.project / "keep.txt").write_text("precious")
    marker = json.loads(lab.marker.read_text())
    original = lab.marker.read_text()
    lab.marker.write_text(json.dumps({**marker, "run_id": "forged-run"}))
    with pytest.raises(RootOwnershipError):
        lab.remove()
    assert (lab.project / "keep.txt").read_text() == "precious"
    assert lab.marker.exists()
    lab.marker.write_text(original)
    result = lab.remove()  # positive twin: once ownership holds, removal succeeds
    assert result["removed"] is True and not lab.root.exists()


def test_remove_never_follows_a_symlink_out_of_the_root(lab: LabRoots, tmp_path: Path) -> None:
    outside = tmp_path / "outside-dir"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("outside data")
    outside_file = tmp_path / "outside-file.txt"
    outside_file.write_text("outside file")
    (lab.project / "dirlink").symlink_to(outside, target_is_directory=True)
    (lab.runtime_b / "filelink").symlink_to(outside_file)
    (lab.project / "inside.txt").write_text("inside")
    result = lab.remove()
    assert result["removed"] is True
    assert not lab.root.exists()
    assert sentinel.read_text() == "outside data"
    assert outside_file.read_text() == "outside file"


def test_remove_repairs_unreadable_subdirs_it_owns(lab: LabRoots) -> None:
    locked = lab.project / "locked"
    locked.mkdir()
    (locked / "f").write_text("x")
    locked.chmod(0o000)
    assert lab.remove()["removed"] is True
    assert not lab.root.exists()


def test_remove_of_missing_root_is_a_noop(tmp_path: Path) -> None:
    roots = LabRoots.create(parent=tmp_path)
    roots.remove()
    assert roots.remove() == {"removed": False, "retained_files": [], "retained_bytes": 0}


# ---------------------------------------------------------------------------- retention


def test_retention_copies_bounded_small_private_files_never_collect_view(
    lab: LabRoots, tmp_path: Path
) -> None:
    (lab.artifacts / "report.json").write_text('{"ok": true}')
    (lab.artifacts / "nested").mkdir(mode=0o700)
    (lab.artifacts / "nested" / "trace.txt").write_text("trace")
    (lab.artifacts / "big.bin").write_bytes(b"x" * (roots_mod.MAX_RETAINED_FILE_BYTES + 1))
    (lab.collect_view / "private-transcript.jsonl").write_text("PRIVATE TRANSCRIPT")
    (lab.verifier / "oracle.json").write_text("SEALED ORACLE")
    outside = tmp_path / "secret-outside.txt"
    outside.write_text("OUTSIDE SECRET")
    (lab.artifacts / "link.txt").symlink_to(outside)
    keep = tmp_path / "keep"
    result = lab.remove(retain_failure_to=keep)
    assert result["removed"] is True and not lab.root.exists()
    assert sorted(result["retained_files"]) == ["nested/trace.txt", "report.json"]
    assert result["retained_bytes"] == len('{"ok": true}') + len("trace")
    retained = Path(result["retained_dir"])
    assert _mode(retained) == 0o700 and retained.parent == keep.resolve()
    assert (retained / "report.json").read_text() == '{"ok": true}'
    assert _mode(retained / "report.json") == 0o600
    assert _mode(retained / "nested") == 0o700
    copied = {p.name for p in retained.rglob("*") if p.is_file()}
    assert copied == {"report.json", "trace.txt"}  # no big file, no collect-view, no symlink target
    assert not any(
        "PRIVATE TRANSCRIPT" in p.read_text() for p in retained.rglob("*") if p.is_file()
    )
    assert "big.bin" in result["skipped"] and "collect-view" in result["skipped"]


def test_retention_total_cap_is_enforced(lab: LabRoots, tmp_path: Path) -> None:
    for index in range(5):
        (lab.artifacts / f"f{index}.txt").write_bytes(b"y" * 100)
    result = lab.remove(retain_failure_to=tmp_path / "keep", retain_cap_bytes=250)
    assert result["retained_bytes"] <= 250
    assert len(result["retained_files"]) == 2
    assert sorted(result["skipped"]) == ["collect-view", "f2.txt", "f3.txt", "f4.txt"]


def test_retention_never_happens_when_ownership_fails(lab: LabRoots, tmp_path: Path) -> None:
    (lab.artifacts / "report.json").write_text("{}")
    marker = json.loads(lab.marker.read_text())
    original = lab.marker.read_text()
    lab.marker.write_text(json.dumps({**marker, "uid": os.getuid() + 7}))
    keep = tmp_path / "keep"
    with pytest.raises(RootOwnershipError):
        lab.remove(retain_failure_to=keep)
    assert not keep.exists()
    lab.marker.write_text(original)


def test_retention_destination_inside_the_root_is_refused(lab: LabRoots) -> None:
    with pytest.raises(ValueError, match="outside the lab root"):
        lab.remove(retain_failure_to=lab.artifacts / "retained")
    assert lab.root.exists()


# ------------------------------------------------------------------------ environments


def test_runtime_env_never_leaks_ambient_secrets(
    lab: LabRoots, monkeypatch: pytest.MonkeyPatch
) -> None:
    canaries = {
        "ANTHROPIC_API_KEY": "sk-ant-CANARY",
        "PGPASSWORD": "pg-CANARY",
        "AWS_SECRET_ACCESS_KEY": "aws-CANARY",
        "DATABASE_URL": "postgresql://u:CANARY@h/db",
        "GITHUB_TOKEN": "gh-CANARY",
        "SSGREP_TRANSCRIPT_DIRS": "/real/history",
        "SSGREP_PI_SESSIONS_DIR": "/real/pi",
    }
    for name, value in canaries.items():
        monkeypatch.setenv(name, value)
    for actor in ACTORS:
        env = runtime_env(lab, actor)
        assert not set(canaries) & set(env), actor
        assert not any("CANARY" in value for value in env.values())
        assert forbidden_env_names(env) == []
    # positive twin: the helper does recognise the ambient environment as dirty
    assert set(canaries) <= set(forbidden_env_names(dict(os.environ)))


def test_runtime_env_is_complete_and_confined_to_the_lab_root(lab: LabRoots) -> None:
    env = runtime_env(lab, "solver-b")
    home = lab.actor_home("solver-b")
    assert env["HOME"] == str(home)
    assert env["SSGREP_DATA_DIR"] == str(lab.ssgrep_data)
    assert env["TMPDIR"] == str(lab.tmp) and _mode(lab.tmp) == 0o700
    assert env["PATH"] == os.environ["PATH"]
    assert env["LANG"] and env["LC_ALL"]
    for name in SIX_ROOT_VARS:
        value = env[name]
        assert value.startswith(str(home) + os.sep), (name, value)
    assert env["SSGREP_OMP_SESSIONS_DIR"] == env["OMP_SESSIONS_DIR"] == str(home / "omp-sessions")
    for name in DISCOVERY_VARS:
        if name in env:
            assert Path(os.path.realpath(env[name])).is_relative_to(lab.root), name
    assert not {"HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"} & set(env)
    for forbidden in roots_mod.FORBIDDEN_NAMES:
        assert forbidden not in env
    assert_isolated(env, lab)
    # actor separation: A and B never share a home or session root
    other = runtime_env(lab, "solver-a")
    assert other["HOME"] != env["HOME"]
    assert other["SSGREP_OMP_SESSIONS_DIR"] != env["SSGREP_OMP_SESSIONS_DIR"]


def test_runtime_sessions_override_and_model_cache(lab: LabRoots, tmp_path: Path) -> None:
    sessions = lab.runtime_a / "sessions"
    cache = tmp_path / "model-cache"  # does not exist: it is a reference, never created
    env = runtime_env(lab, "solver-a", runtime_sessions=sessions, model_cache=cache)
    assert env["SSGREP_OMP_SESSIONS_DIR"] == env["OMP_SESSIONS_DIR"] == str(sessions)
    assert env["HF_HOME"] == str(cache)
    assert env["HF_HUB_OFFLINE"] == "1" and env["TRANSFORMERS_OFFLINE"] == "1"
    assert not cache.exists()
    assert_isolated(env, lab)  # HF_HOME is the one permitted outside reference
    with pytest.raises(IsolationError, match="OMP_SESSIONS_DIR"):
        runtime_env(lab, "solver-a", runtime_sessions=tmp_path / "outside-sessions")


def test_extra_secret_keys_are_refused_and_innocuous_keys_pass(lab: LabRoots) -> None:
    for name in (
        "ANTHROPIC_API_KEY",
        "MY_TOKEN",
        "DB_SECRET",
        "ADMIN_PASSWORD",
        "PGPASSWORD",
        "PGHOST",
        "DATABASE_URL",
        "AWS_PROFILE",
        "SSGREP_TRANSCRIPT_DIRS",
    ):
        with pytest.raises(ValueError, match="refusing environment variable"):
            runtime_env(lab, "solver-a", extra={name: "x"})
    env = runtime_env(lab, "solver-a", extra={"LAB_RUN_LABEL": "airline", "OMP_SOMETHING": "1"})
    assert env["LAB_RUN_LABEL"] == "airline" and env["OMP_SOMETHING"] == "1"


def test_extra_cannot_move_a_discovery_root_outside_the_lab(lab: LabRoots) -> None:
    with pytest.raises(IsolationError, match="CLAUDE_CONFIG_DIR"):
        runtime_env(lab, "solver-a", extra={"CLAUDE_CONFIG_DIR": "/Users/someone/.claude"})
    with pytest.raises(IsolationError, match="HOME"):
        runtime_env(lab, "solver-a", extra={"HOME": "/Users/someone"})


def test_assert_isolated_detects_each_escape_and_symlinks(lab: LabRoots, tmp_path: Path) -> None:
    good = runtime_env(lab, "solver-b")
    assert_isolated(good, lab)
    for name in ("HOME", "SSGREP_DATA_DIR", "CODEX_HOME", "SSGREP_OPENCODE_DB", "TMPDIR"):
        bad = {**good, name: "/etc"}
        with pytest.raises(IsolationError, match=name):
            assert_isolated(bad, lab)
    escape = lab.project / "escape"
    escape.symlink_to(tmp_path)
    with pytest.raises(IsolationError, match="PI_SESSION_DIR"):
        assert_isolated({**good, "PI_SESSION_DIR": str(escape / "x")}, lab)
    with pytest.raises(IsolationError, match="not set"):
        assert_isolated({k: v for k, v in good.items() if k != "HOME"}, lab)
    with pytest.raises(IsolationError, match="ANTHROPIC_API_KEY"):
        assert_isolated({**good, "ANTHROPIC_API_KEY": "k"}, lab)


def test_credential_home_replaces_only_home_and_names_a_limitation(
    lab: LabRoots, tmp_path: Path
) -> None:
    real_home = tmp_path / "real-home"
    real_home.mkdir()
    env = runtime_env(lab, "solver-b", credential_home=real_home)
    assert env["HOME"] == str(real_home)
    for name in SIX_ROOT_VARS + ("SSGREP_DATA_DIR", "TMPDIR"):
        assert Path(env[name]).is_relative_to(lab.root), name  # discovery stays isolated
    assert_isolated(env, lab, credential_home=real_home)
    with pytest.raises(IsolationError, match="HOME"):
        assert_isolated(env, lab)  # without the declared credential home it is an escape
    names = [item.name for item in roots_mod.env_limitations(lab, real_home)]
    assert names == ["omp_requires_real_home_for_credentials"]
    assert roots_mod.env_limitations(lab, None) == ()
    # a discovery var pointing into the real home is still refused even with credential_home
    with pytest.raises(IsolationError, match="CLAUDE_CONFIG_DIR"):
        runtime_env(
            lab,
            "solver-b",
            credential_home=real_home,
            extra={"CLAUDE_CONFIG_DIR": str(real_home / ".claude")},
        )
