"""CLAUDE_CONFIG_DIR: one resolver for Claude Code's config root.

Claude Code lets a user move its entire config tree with the
``CLAUDE_CONFIG_DIR`` environment variable. ssgrep touches that tree in two
places, and both have to follow it or the product is a total failure for
that buyer:

1. ``<root>/projects`` -- the transcripts. Ignoring the variable here means
   indexing finds nothing.
2. ``<root>/settings.json`` -- where the SessionEnd hook is installed.
   Ignoring the variable here is worse: ``hooks install`` reports success
   after writing a hook into a file Claude Code never reads, so the hook
   never fires and there is no error, no warning, and no output at all.

Every test below drives the real code path (``indexer.index`` /
``hooks._install_hook``) rather than calling ``resolve_claude_dir``
directly, so deleting the wiring at any single site fails a test even
though the resolver itself still exists.

Isolation: HOME is redirected to a tmp_path-rooted fake home (both the
``HOME`` env var and ``Path.home``), so the fallback branch never reads the
developer's real ``~/.claude``. Index output always goes to a tmp_path via
``index_dir=``.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ssgrep import embed, indexer, observability, store
from ssgrep.cli.commands import hooks
from ssgrep.types import IndexNotFoundError
from ssgrep.workqueue import WorkQueue

SESSION_A = "aaaaaaaa-2222-4333-8444-555555555555"
SESSION_B = "bbbbbbbb-2222-4333-8444-555555555555"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A HOME that is not the developer's, with a real ~/.claude tree in it.

    The ``~/.claude/projects`` directory is deliberately created and
    populated in most tests: it is the decoy. If the transcript root stops
    honouring CLAUDE_CONFIG_DIR it will silently fall back here, and the
    tests assert on *which* corpus got indexed, not merely that something
    did.
    """
    home = tmp_path / "home"
    (home / ".claude" / "projects").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    return home


@pytest.fixture(autouse=True)
def _no_inherited_config_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test from "not relocated", whatever the dev's shell says."""
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)


def _fake_encode(monkeypatch: pytest.MonkeyPatch) -> None:
    def encode(texts: list[str]) -> np.ndarray:
        return np.full((len(texts), 256), 1.0 / (256**0.5), dtype=np.float32)

    monkeypatch.setattr(embed, "encode", encode)


def _write_transcript(path: Path, cwd: Path, session_id: str, marker: str) -> None:
    """A minimal but real two-record episode whose prompt carries `marker`."""
    records: list[dict[str, Any]] = [
        {
            "parentUuid": None,
            "isSidechain": False,
            "type": "user",
            "message": {"role": "user", "content": f"How do I fix the {marker} problem?"},
            "uuid": "u1",
            "timestamp": "2026-07-01T10:00:00.000Z",
            "cwd": str(cwd),
            "sessionId": session_id,
            "gitBranch": "main",
        },
        {
            "parentUuid": "u1",
            "isSidechain": False,
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": f"Restart the {marker} daemon."}],
            },
            "uuid": "a1",
            "timestamp": "2026-07-01T10:01:00.000Z",
            "cwd": str(cwd),
            "sessionId": session_id,
            "gitBranch": "main",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _indexed_markers(
    index_dir: Path, markers: tuple[str, ...] = ("relocated", "decoy", "homeonly")
) -> set[str]:
    """Which sentinel words actually made it into the index's FTS content."""
    conn = sqlite3.connect(str(index_dir / "index.db"))
    try:
        found = set()
        for marker in markers:
            if store.search_fts(conn, marker):
                found.add(marker)
        return found
    finally:
        conn.close()


def _installed_command(settings_path: Path) -> str:
    """The SessionEnd hook command Claude Code would actually run.

    Walks the real settings schema rather than trusting that a file exists:
    a settings.json written in the wrong shape is as dead as one written to
    the wrong path.
    """
    settings = json.loads(settings_path.read_text())
    commands = []
    for group in settings.get("hooks", {}).get("SessionEnd", []):
        for hook in group.get("hooks", []):
            commands.append(hook.get("command", ""))
    assert len(commands) == 1, f"expected exactly one hook, got {commands}"
    return commands[0]


# ---------------------------------------------------------------------------
# Transcript root
# ---------------------------------------------------------------------------


def test_indexing_reads_transcripts_from_claude_config_dir(
    fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The relocated corpus is indexed and the ~/.claude one is not.

    Both corpora record the same cwd, so scope matching cannot be what
    distinguishes them -- only the choice of transcript root can.
    """
    project_dir = tmp_path / "project"
    alt_root = tmp_path / "alt-config"

    _write_transcript(
        alt_root / "projects" / "proj" / f"{SESSION_A}.jsonl",
        project_dir,
        SESSION_A,
        "relocated",
    )
    _write_transcript(
        fake_home / ".claude" / "projects" / "proj" / f"{SESSION_B}.jsonl",
        project_dir,
        SESSION_B,
        "decoy",
    )

    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(alt_root))
    _fake_encode(monkeypatch)
    index_dir = tmp_path / "idx"

    stats = indexer.index(project_dir, index_dir=index_dir, quiet=True)

    assert stats.session_count == 1
    assert _indexed_markers(index_dir) == {"relocated"}


def test_indexing_falls_back_to_home_claude_when_var_unset(
    fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no CLAUDE_CONFIG_DIR, ~/.claude/projects is still the root.

    The relocated corpus exists on disk here too, so a resolver that always
    read some other root (or an empty default) would be caught.
    """
    project_dir = tmp_path / "project"
    alt_root = tmp_path / "alt-config"

    _write_transcript(
        fake_home / ".claude" / "projects" / "proj" / f"{SESSION_A}.jsonl",
        project_dir,
        SESSION_A,
        "homeonly",
    )
    _write_transcript(
        alt_root / "projects" / "proj" / f"{SESSION_B}.jsonl",
        project_dir,
        SESSION_B,
        "relocated",
    )

    _fake_encode(monkeypatch)
    index_dir = tmp_path / "idx"

    stats = indexer.index(project_dir, index_dir=index_dir, quiet=True)

    assert stats.session_count == 1
    assert _indexed_markers(index_dir) == {"homeonly"}


def test_missing_transcript_root_error_names_the_configured_root(
    fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Indexing must fail against the relocated root, and say where it looked.

    ``~/.claude/projects`` exists (the fake_home fixture makes it), so a
    resolver that ignored CLAUDE_CONFIG_DIR would find a perfectly valid
    root and raise nothing at all.

    Also pins the removal of the ``mkdir -p ~/.claude/projects`` remedy: a
    manufactured empty root is indistinguishable at runtime from a real
    one, so following that advice would permanently silence the only signal
    a mis-configured user ever gets.
    """
    alt_root = tmp_path / "alt-config"
    alt_root.mkdir()  # exists, but has no projects/ inside
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(alt_root))

    with pytest.raises(IndexNotFoundError) as excinfo:
        indexer.index(tmp_path / "project", index_dir=tmp_path / "idx", quiet=True)

    error = excinfo.value
    assert str(alt_root / "projects") in str(error)
    assert "mkdir" not in f"{error} {error.command}"
    # The machine-readable remedy must not be the command that just failed.
    # IndexNotFoundError's default is "ssgrep index", so dropping the explicit
    # `mkdir -p` argument left `command` pointing at an infinite loop for any
    # agent or script that follows it. The human message names
    # CLAUDE_CONFIG_DIR; the structured field has to as well.
    assert error.command != "ssgrep index", (
        "the remedy field is the command that just failed -- following it " "loops forever"
    )
    assert "CLAUDE_CONFIG_DIR" in (
        error.command or ""
    ), f"the structured remedy must name the actual fix: {error.command!r}"


def test_workqueue_drain_classifies_against_the_configured_root(
    fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The drain's session classification must use the same root as discovery.

    _drain_workqueue() re-classifies each queued transcript as main-vs-subagent
    by path position relative to the transcript root. Hardcoding
    ``Path.home()/".claude"/"projects"`` there makes ``path.relative_to()``
    fail for a relocated buyer, and classify_session's except-branch then
    treats EVERY queued subagent session as a main session -- so subagent
    identity (parent_session_id, agent_hash) is silently lost for exactly the
    sessions hooks enqueue.
    """
    project_dir = tmp_path / "project"
    alt_root = tmp_path / "alt-config"
    parent = "cccccccc-2222-4333-8444-555555555555"
    subagent_path = alt_root / "projects" / "proj" / parent / "subagents" / f"{SESSION_A}.jsonl"
    _write_transcript(subagent_path, project_dir, SESSION_A, "relocated")

    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(alt_root))
    _fake_encode(monkeypatch)
    index_dir = tmp_path / "idx"

    queue = WorkQueue(index_dir)
    queue.open()
    try:
        queue.enqueue(
            transcript_path=str(subagent_path), session_id=SESSION_A, reason="session_end"
        )
    finally:
        queue.close()

    indexer.index(project_dir, index_dir=index_dir, quiet=True)

    conn = sqlite3.connect(str(index_dir / "index.db"))
    try:
        row = conn.execute(
            "SELECT is_main, parent_session_id FROM sessions WHERE session_id = ?",
            (SESSION_A,),
        ).fetchone()
    finally:
        conn.close()

    assert row is not None, "the queued session must have been indexed"
    assert row[0] == 0, (
        "a subagent transcript under the RELOCATED root was classified as a main "
        "session -- the drain resolved the transcript root without honouring "
        "CLAUDE_CONFIG_DIR"
    )
    assert row[1] == parent, f"subagent identity must survive the drain: {row}"


def test_init_hook_warning_names_the_configured_settings_path(
    fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """When hook installation fails, the manual instructions must name the
    settings.json Claude Code actually reads.

    This is the fallback path a relocated buyer lands on, and telling them to
    edit ``~/.claude/settings.json`` sends them to a file Claude Code never
    reads -- the same silent no-op the automatic install used to produce, only
    now performed by hand and believed to have worked.
    """
    from unittest.mock import MagicMock

    from ssgrep.cli.commands.init import InitCommand

    alt_root = tmp_path / "alt-config"
    (alt_root / "projects").mkdir(parents=True)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(alt_root))
    _fake_encode(monkeypatch)

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    monkeypatch.setattr(
        "ssgrep.cli.commands.init._install_hook",
        lambda _project: (_ for _ in ()).throw(OSError("permission denied")),
    )

    InitCommand(MagicMock()).handle(project_dir=str(project_dir))

    stderr = capsys.readouterr().err
    assert (
        str(alt_root / "settings.json") in stderr
    ), f"the manual remedy must name the relocated settings file:\n{stderr}"
    assert str(fake_home / ".claude" / "settings.json") not in stderr


def test_hooks_command_resolves_a_quoted_tilde_project_dir(
    fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`ssgrep hooks install --project-dir "~/code/x"` must mean the home dir.

    HooksCommand.handle() resolves through resolve_project_dir (i.e.
    paths.resolve_live), not Path(...).resolve(). The difference is the
    tilde: a quoted "~" is never expanded by the shell, so bare resolve()
    anchors it under the CURRENT working directory and writes a hook whose
    --project-dir names a directory that does not exist -- so the hook fires
    and indexes nothing, forever, silently.
    """
    from unittest.mock import MagicMock

    from ssgrep.cli.commands.hooks import HooksCommand

    project = fake_home / "code" / "x"
    project.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    HooksCommand(MagicMock()).handle(action="install", project_dir="~/code/x")

    command = _installed_command(fake_home / ".claude" / "settings.json")
    assert str(project.resolve()) in command, f"tilde was not expanded: {command}"
    assert "~" not in command, f"a literal tilde reached the installed hook: {command}"


# ---------------------------------------------------------------------------
# settings.json -- the site most likely to be missed
# ---------------------------------------------------------------------------


def test_hook_install_writes_to_claude_config_dir_settings(
    fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hook lands in the settings.json Claude Code actually reads.

    Asserting the absence of ~/.claude/settings.json is the half that
    matters for the silent failure: writing to both would look like success
    while the relocated user's hook still never fires from the wrong file.
    """
    alt_root = tmp_path / "alt-config"
    alt_root.mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(alt_root))
    project_dir = tmp_path / "project"

    hooks._install_hook(project_dir)

    assert str(project_dir) in _installed_command(alt_root / "settings.json")
    assert not (fake_home / ".claude" / "settings.json").exists()


def test_hook_install_falls_back_to_home_claude_settings(fake_home: Path, tmp_path: Path) -> None:
    """With no CLAUDE_CONFIG_DIR, the hook still lands in ~/.claude."""
    project_dir = tmp_path / "project"

    hooks._install_hook(project_dir)

    settings_path = fake_home / ".claude" / "settings.json"
    assert str(project_dir) in _installed_command(settings_path)


def test_config_dir_is_read_per_call_not_captured_at_import(
    fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing CLAUDE_CONFIG_DIR mid-process changes where the hook goes.

    A module-level constant computed at import time would freeze whichever
    value existed when ssgrep was first imported -- which, for a user who
    exports the variable in their shell profile after install, is the wrong
    one forever. Two installs under two roots: both files must exist and
    each must name its own project.
    """
    first_root = tmp_path / "config-one"
    second_root = tmp_path / "config-two"
    first_root.mkdir()
    second_root.mkdir()
    project_one = tmp_path / "project-one"
    project_two = tmp_path / "project-two"

    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(first_root))
    hooks._install_hook(project_one)

    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(second_root))
    hooks._install_hook(project_two)

    assert str(project_one) in _installed_command(first_root / "settings.json")
    assert str(project_two) in _installed_command(second_root / "settings.json")


def test_empty_config_dir_var_means_not_relocated(
    fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exported-but-empty variable must not point ssgrep at the FS root.

    Shells leave ``CLAUDE_CONFIG_DIR=`` behind when a user clears it;
    treating that as a path would make the settings file ``/settings.json``.
    """
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "  ")
    project_dir = tmp_path / "project"

    hooks._install_hook(project_dir)

    settings_path = fake_home / ".claude" / "settings.json"
    assert str(project_dir) in _installed_command(settings_path)


# ---------------------------------------------------------------------------
# Symlinked vs physical spelling of the transcript root (issue #2)
# ---------------------------------------------------------------------------


def test_symlinked_vs_physical_config_dir_does_not_double_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-indexing via a symlink to the same root must not duplicate chunks.

    resolve_claude_dir() did not canonicalize its result, so discovery built
    every stored path identity (session_files' cursor primary key, sessions'
    path column) from whatever string spelling CLAUDE_CONFIG_DIR happened to
    carry. Indexing once via a physical root and again via a symlink to that
    *same* root re-inserted the one transcript's chunks under a second cursor
    key: chunk_count doubled, and the cursor for one spelling could never
    match the disk stat recorded under the other, so staleness never cleared.

    Uses an explicit symlink (rather than relying on macOS's /tmp ->
    /private/tmp) so the property holds on any platform, mirroring
    test_path_canonicalization.py's own real/link fixture pattern.
    """
    project_dir = tmp_path / "project"
    physical_root = tmp_path / "physical-config"
    symlinked_root = tmp_path / "symlinked-config"
    physical_root.mkdir()
    symlinked_root.symlink_to(physical_root, target_is_directory=True)

    _write_transcript(
        physical_root / "projects" / "proj" / f"{SESSION_A}.jsonl",
        project_dir,
        SESSION_A,
        "onlyonce",
    )

    _fake_encode(monkeypatch)
    index_dir = project_dir / ".ssgrep"

    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(physical_root))
    stats_physical = indexer.index(project_dir, index_dir=index_dir, quiet=True)

    assert stats_physical.session_count == 1
    assert _indexed_markers(index_dir, markers=("onlyonce",)) == {"onlyonce"}

    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(symlinked_root))
    stats_symlinked = indexer.index(project_dir, index_dir=index_dir, quiet=True)

    assert stats_symlinked.session_count == 1, (
        "re-indexing via a symlinked spelling of the same root must still see "
        f"exactly one session, got {stats_symlinked.session_count}"
    )
    assert stats_symlinked.chunk_count == stats_physical.chunk_count, (
        "re-indexing via a symlinked spelling of the same transcript root "
        f"changed chunk_count: {stats_physical.chunk_count} -> "
        f"{stats_symlinked.chunk_count}"
    )

    status = observability.status(project_dir)
    assert status.stale is False, (
        "the index must not report permanently stale after indexing via a "
        f"symlinked spelling of the same root: {status}"
    )
