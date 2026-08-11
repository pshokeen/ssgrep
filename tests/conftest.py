"""Pytest fixture loaders and builder helpers for ssgrep tests.

Every wave-2 task tests against these identical inputs.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from ssgrep.types import (
    Chunk,
    ContentType,
    Episode,
    EpisodeDetail,
    ErrorResponse,
    FileCursor,
    IndexStats,
    ResultCard,
    SearchFilters,
    SearchResponse,
    SessionFile,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Builder helpers — each returns a valid, complete instance of the type
# ---------------------------------------------------------------------------


def build_session_file(
    path: Path | str | None = None,
    session_id: str = "test-session-001",
    is_main: bool = True,
    size: int = 1024,
    mtime: float = 1700000000.0,
    parent_session_id: str | None = None,
    agent_hash: str | None = None,
    agent_type: str | None = None,
    agent_name: str | None = None,
    agent_description: str | None = None,
    agent_model: str | None = None,
    **overrides: Any,
) -> SessionFile:
    obj = SessionFile(
        path=Path(path) if path else Path(f"/tmp/test-{session_id}.jsonl"),
        session_id=session_id,
        is_main=is_main,
        size=size,
        mtime=mtime,
        parent_session_id=parent_session_id,
        agent_hash=agent_hash,
        agent_type=agent_type,
        agent_name=agent_name,
        agent_description=agent_description,
        agent_model=agent_model,
    )
    return dataclasses.replace(obj, **overrides) if overrides else obj


def build_episode(
    episode_id: str = "ep-001",
    session_id: str = "test-session-001",
    prompt_text: str = "Test prompt",
    response_text: str = "Test response",
    title: str = "Test Episode",
    timestamp: datetime | None = None,
    git_branch: str | None = "main",
    cwd: str | None = "/Users/test/project",
    files_touched: tuple[str, ...] = ("src/test.py",),
    tool_names: tuple[str, ...] = ("Read", "Edit"),
    is_subagent: bool = False,
    agent_type: str | None = None,
    agent_name: str | None = None,
    agent_description: str | None = None,
    parent_session_id: str | None = None,
    **overrides: Any,
) -> Episode:
    obj = Episode(
        episode_id=episode_id,
        session_id=session_id,
        prompt_text=prompt_text,
        response_text=response_text,
        title=title,
        timestamp=timestamp or datetime.now(UTC),
        git_branch=git_branch,
        cwd=cwd,
        files_touched=files_touched,
        tool_names=tool_names,
        is_subagent=is_subagent,
        agent_type=agent_type,
        agent_name=agent_name,
        agent_description=agent_description,
        parent_session_id=parent_session_id,
    )
    return dataclasses.replace(obj, **overrides) if overrides else obj


def build_chunk(
    chunk_id: str = "chunk-001",
    episode_id: str = "ep-001",
    session_id: str = "test-session-001",
    text: str = "Test chunk text content",
    content_type: ContentType = ContentType.PROMPT,
    byte_offset: int = 0,
    vec_row: int | None = None,
    **overrides: Any,
) -> Chunk:
    obj = Chunk(
        chunk_id=chunk_id,
        episode_id=episode_id,
        session_id=session_id,
        text=text,
        content_type=content_type,
        byte_offset=byte_offset,
        vec_row=vec_row,
    )
    return dataclasses.replace(obj, **overrides) if overrides else obj


def build_file_cursor(
    path: Path | str | None = None,
    size: int = 1024,
    mtime: float = 1700000000.0,
    byte_offset: int = 0,
    first_line_hash: str = "abc123",
    **overrides: Any,
) -> FileCursor:
    obj = FileCursor(
        path=Path(path) if path else Path("/tmp/test-session.jsonl"),
        size=size,
        mtime=mtime,
        byte_offset=byte_offset,
        first_line_hash=first_line_hash,
    )
    return dataclasses.replace(obj, **overrides) if overrides else obj


def build_search_filters(
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    file_path: str | None = None,
    content_type: ContentType | None = None,
    branch: str | None = None,
    **overrides: Any,
) -> SearchFilters:
    obj = SearchFilters(
        date_from=date_from,
        date_to=date_to,
        file_path=file_path,
        content_type=content_type,
        branch=branch,
    )
    return dataclasses.replace(obj, **overrides) if overrides else obj


def build_error_response(
    ok: bool = False,
    condition: str | None = "test_condition",
    message: str = "Test error message",
    command: str | None = None,
    **overrides: Any,
) -> ErrorResponse:
    obj = ErrorResponse(
        ok=ok,
        condition=condition,
        message=message,
        command=command,
    )
    return dataclasses.replace(obj, **overrides) if overrides else obj


def build_result_card(
    ref: str = "ep-001",
    title: str = "Test Result",
    timestamp: datetime | None = None,
    score: float = 0.85,
    excerpt: str = "Test excerpt text",
    files_touched: tuple[str, ...] = ("src/test.py",),
    is_subagent: bool = False,
    agent_name: str | None = None,
    agent_description: str | None = None,
    parent_session_id: str | None = None,
    content_type: ContentType | None = None,
    **overrides: Any,
) -> ResultCard:
    obj = ResultCard(
        ref=ref,
        title=title,
        timestamp=timestamp or datetime.now(UTC),
        score=score,
        excerpt=excerpt,
        files_touched=files_touched,
        is_subagent=is_subagent,
        agent_name=agent_name,
        agent_description=agent_description,
        parent_session_id=parent_session_id,
        content_type=content_type,
    )
    return dataclasses.replace(obj, **overrides) if overrides else obj


def build_search_response(
    results: list[ResultCard] | None = None,
    omitted_count: int = 0,
    index_exists: bool = True,
    index_empty: bool = False,
    total_matches: int = 0,
    excerpts_truncated: bool = False,
    clamped: bool = False,
    **overrides: Any,
) -> SearchResponse:
    obj = SearchResponse(
        results=results or [],
        omitted_count=omitted_count,
        index_exists=index_exists,
        index_empty=index_empty,
        total_matches=total_matches,
        excerpts_truncated=excerpts_truncated,
        clamped=clamped,
    )
    return dataclasses.replace(obj, **overrides) if overrides else obj


def build_episode_detail(
    episode_id: str = "ep-001",
    session_id: str = "test-session-001",
    title: str = "Test Episode",
    timestamp: datetime | None = None,
    git_branch: str | None = "main",
    cwd: str | None = "/Users/test/project",
    prompt_text: str = "Test prompt",
    response_text: str = "Test response",
    files_touched: tuple[str, ...] = ("src/test.py",),
    tool_names: tuple[str, ...] = ("Read", "Edit"),
    is_subagent: bool = False,
    agent_type: str | None = None,
    agent_name: str | None = None,
    agent_description: str | None = None,
    parent_session_id: str | None = None,
    prompt_truncated: bool = False,
    response_truncated: bool = False,
    **overrides: Any,
) -> EpisodeDetail:
    obj = EpisodeDetail(
        episode_id=episode_id,
        session_id=session_id,
        title=title,
        timestamp=timestamp or datetime.now(UTC),
        git_branch=git_branch,
        cwd=cwd,
        prompt_text=prompt_text,
        response_text=response_text,
        files_touched=files_touched,
        tool_names=tool_names,
        is_subagent=is_subagent,
        agent_type=agent_type,
        agent_name=agent_name,
        agent_description=agent_description,
        parent_session_id=parent_session_id,
        prompt_truncated=prompt_truncated,
        response_truncated=response_truncated,
    )
    return dataclasses.replace(obj, **overrides) if overrides else obj


def build_index_stats(
    session_count: int = 5,
    episode_count: int = 20,
    chunk_count: int = 150,
    index_size_bytes: int = 1_000_000,
    last_index_time: datetime | None = None,
    model_id: str = "minishlab/potion-base-8M",
    vector_dimension: int = 256,
    skipped_records: int = 0,
    malformed_records: int = 0,
    schema_version: int = 1,
    index_exists: bool = True,
    cwd_cache_degraded: bool = False,
    cwd_cache_fallback_scans: int = 0,
    **overrides: Any,
) -> IndexStats:
    obj = IndexStats(
        session_count=session_count,
        episode_count=episode_count,
        chunk_count=chunk_count,
        index_size_bytes=index_size_bytes,
        last_index_time=last_index_time or datetime.now(UTC),
        model_id=model_id,
        vector_dimension=vector_dimension,
        skipped_records=skipped_records,
        malformed_records=malformed_records,
        schema_version=schema_version,
        index_exists=index_exists,
        cwd_cache_degraded=cwd_cache_degraded,
        cwd_cache_fallback_scans=cwd_cache_fallback_scans,
    )
    return dataclasses.replace(obj, **overrides) if overrides else obj


# ---------------------------------------------------------------------------
# Fixture loaders — parse JSONL fixtures into lists of dicts
# ---------------------------------------------------------------------------


def load_jsonl_fixture(filename: str) -> list[dict[str, Any]]:
    """Load a JSONL fixture file and return parsed records."""
    path = FIXTURES_DIR / filename
    records: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    # Malformed lines are expected in some fixtures
                    records.append({"_malformed": True, "_raw": line})
    return records


def load_fixture_lines(filename: str) -> list[str]:
    """Load raw lines from a fixture file."""
    path = FIXTURES_DIR / filename
    with open(path) as f:
        return [line.rstrip("\n") for line in f]


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to the test fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture
def main_session_records() -> list[dict[str, Any]]:
    """Records from the main session fixture."""
    return load_jsonl_fixture("main-session.jsonl")


@pytest.fixture
def compaction_session_records() -> list[dict[str, Any]]:
    """Records from the compaction session fixture."""
    return load_jsonl_fixture("compaction-session.jsonl")


@pytest.fixture
def thinking_signatures_records() -> list[dict[str, Any]]:
    """Records from the thinking signatures fixture."""
    return load_jsonl_fixture("thinking-signatures.jsonl")


@pytest.fixture
def zero_prose_records() -> list[dict[str, Any]]:
    """Records from the zero-prose session fixture."""
    return load_jsonl_fixture("zero-prose-session.jsonl")


@pytest.fixture
def malformed_line_records() -> list[dict[str, Any]]:
    """Records from the malformed-line fixture."""
    return load_jsonl_fixture("malformed-line.jsonl")


@pytest.fixture
def subagent_records() -> list[dict[str, Any]]:
    """Records from the subagent session fixture."""
    return load_jsonl_fixture("agent-aexample-agent-deadbeef01.jsonl")


@pytest.fixture
def old_schema_records() -> list[dict[str, Any]]:
    """Records from the old-schema session fixture."""
    return load_jsonl_fixture("old-schema-session.jsonl")


@pytest.fixture
def new_schema_records() -> list[dict[str, Any]]:
    """Records from the new-schema session fixture."""
    return load_jsonl_fixture("new-schema-session.jsonl")


@pytest.fixture
def subagent_meta() -> dict[str, Any]:
    """Subagent metadata from the sidecar file."""
    path = FIXTURES_DIR / "agent-aexample-agent-deadbeef01.meta.json"
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Typed instance fixtures — valid, complete instances of each contract type
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_session_file() -> SessionFile:
    return build_session_file()


@pytest.fixture
def sample_episode() -> Episode:
    return build_episode()


@pytest.fixture
def sample_chunk() -> Chunk:
    return build_chunk()


@pytest.fixture
def sample_file_cursor() -> FileCursor:
    return build_file_cursor()


@pytest.fixture
def sample_search_filters() -> SearchFilters:
    return build_search_filters()


@pytest.fixture
def sample_result_card() -> ResultCard:
    return build_result_card()


@pytest.fixture
def sample_search_response() -> SearchResponse:
    return build_search_response()


@pytest.fixture
def sample_episode_detail() -> EpisodeDetail:
    return build_episode_detail()


@pytest.fixture
def sample_index_stats() -> IndexStats:
    return build_index_stats()


# ---------------------------------------------------------------------------
# CLI test fixtures — project directories with optional indexing
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_path_empty(tmp_path: Path) -> Path:
    """Create an empty temporary project directory (no index).

    Returns:
        Path to the temporary directory that is an empty project.
    """
    return tmp_path


@pytest.fixture
def tmp_path_project_indexed(tmp_path: Path, monkeypatch) -> Path:
    """Create a temporary project directory with an empty but valid index.

    This fixture creates .ssgrep/index.db with no sessions discovered from
    ~/.claude/projects/. This is a valid state per the spec and allows testing
    of commands that require an existing index, including the "no results"
    exit code for search operations.

    Uses an isolated HOME directory with .claude/projects to avoid relying on
    the developer's real ~/.claude/projects, ensuring the test works in CI.

    Returns:
        Path to the temporary directory with .ssgrep/index.db present.
    """
    from ssgrep import api

    # Create an isolated HOME with the required .claude/projects directory
    isolated_home = tmp_path / "isolated_home"
    isolated_home.mkdir()
    (isolated_home / ".claude" / "projects").mkdir(parents=True, exist_ok=True)

    # Temporarily set HOME to the isolated directory for api.index()
    monkeypatch.setenv("HOME", str(isolated_home))

    # Index the empty project, which creates .ssgrep/index.db with 0 sessions
    api.index(tmp_path, quiet=True)

    return tmp_path


def get_ssgrep_binary() -> Path:
    """Resolve the ssgrep binary from the running interpreter's venv.

    Derives the binary path from sys.executable using with_name(), ensuring
    tests always use the ssgrep built in the venv actually running the tests,
    never a stale system or homebrew binary. Asserts the binary exists with
    a clear error message.

    Returns:
        Path to the ssgrep binary.

    Raises:
        AssertionError if the binary does not exist.
    """
    binary_path = Path(sys.executable).with_name("ssgrep")
    assert binary_path.exists(), (
        f"ssgrep binary not found at {binary_path}. "
        f"Ensure ssgrep is installed in the venv (sys.executable={sys.executable})."
    )
    return binary_path


# ---------------------------------------------------------------------------
# Session-scoped guards — prevent test suite from polluting real settings
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    """Fixture: create an isolated HOME with .claude/projects for in-process tests.

    This fixture creates a temporary HOME directory containing the .claude/projects
    directory structure required by indexer.py's first-run guard, while keeping
    test state isolated from the real ~/.claude/settings.json.

    The fixture temporarily sets the HOME environment variable to the isolated
    directory for the duration of the test, then restores it afterward.

    Returns:
        Tuple of (isolated_home_path, projects_path) for use in tests.

    Example:
        def test_something(isolated_home: tuple[Path, Path]) -> None:
            home, projects = isolated_home
            # HOME is now pointing to the isolated directory
            # Create a test transcript in projects/
            session_dir = projects / "test-session"
            session_dir.mkdir()
            ...
    """
    isolated_home_dir = tmp_path / "isolated_home"
    isolated_home_dir.mkdir()
    projects_dir = isolated_home_dir / ".claude" / "projects"
    projects_dir.mkdir(parents=True)

    # Set HOME for this test
    monkeypatch.setenv("HOME", str(isolated_home_dir))

    return isolated_home_dir, projects_dir


@pytest.fixture
def has_real_corpus() -> bool:
    """Check whether the real Claude Code corpus exists.

    Returns:
        True if the corpus exists and is not empty, False otherwise.

    Used by discovery tests that require real sessions to validate behavior.
    Resolves via resolve_claude_dir() to respect CLAUDE_CONFIG_DIR overrides.
    """
    from ssgrep.paths import resolve_claude_dir

    corpus_root = resolve_claude_dir() / "projects"
    if not corpus_root.exists():
        return False
    # Check if it's not empty (has at least one session directory)
    try:
        for item in corpus_root.iterdir():
            if item.is_dir() or item.is_file():
                return True
    except (OSError, PermissionError):
        pass
    return False


@pytest.fixture
def corpus_scope_with_renamed_dir() -> tuple[str, str] | None:
    """A (scope, project_dir_name) pair observed in the real corpus.

    Discovery matches a transcript to a scope by the ``cwd`` recorded inside
    its records, not by the encoded project-directory name the transcript
    happens to live under. Exercising that needs a scope whose transcripts sit
    under a directory named for some *other* path -- the shape Claude Code
    produces when a session's working directory differs from the directory its
    project dir was named for. Which paths exhibit it is a property of
    whoever's corpus is on the machine, so the pair is located by scanning the
    corpus instead of naming one developer's projects.

    Returns:
        (scope, project_dir_name) for the best-evidenced such pair, or None
        when the corpus contains none, so callers can skip.
    """
    from ssgrep import paths as _paths
    from ssgrep.discovery import cwd_index_for

    claude_dir = _paths.resolve_claude_dir() / "projects"
    if not claude_dir.exists():
        return None

    def _encode(path: str) -> str:
        return path.replace("/", "-").replace(".", "-")

    # scope -> (number of transcripts under a differently-named dir, dir names)
    file_counts: dict[str, int] = {}
    dir_names: dict[str, set[str]] = {}
    for file_path, cwds in cwd_index_for(claude_dir).items():
        parts = Path(file_path).relative_to(claude_dir).parts
        if not parts:
            continue
        project_dir = parts[0]
        for cwd in cwds:
            encoded = _encode(cwd)
            if project_dir == encoded or project_dir.startswith(encoded + "-"):
                continue
            file_counts[cwd] = file_counts.get(cwd, 0) + 1
            dir_names.setdefault(cwd, set()).add(project_dir)

    if not file_counts:
        return None
    # Deterministic pick: most evidence first, ties broken lexically so the
    # same machine always yields the same pair across runs.
    scope = max(sorted(file_counts), key=lambda cwd: file_counts[cwd])
    return scope, sorted(dir_names[scope])[0]


@pytest.fixture(scope="session", autouse=True)
def guard_real_settings_unchanged() -> None:
    """Guard: ensure the test suite does not modify the real settings.json.

    This fixture runs at session start and end, capturing the hash of the real
    settings.json file. If the file changes during test execution, the session
    fails with a clear error message.

    This is a session-scoped guard (not per-test) to catch cross-file damage
    that per-invocation guards might miss.

    The guarded path is resolved through ``_real_settings_file()``, i.e. the
    same ``resolve_claude_dir()`` production writes through, NOT a hardcoded
    ``~/.claude``. Hardcoding it made the guard vacuous for exactly the
    developers and CI runners this change added support for: with
    ``CLAUDE_CONFIG_DIR`` exported, ``hooks._install_hook`` writes to
    ``$CLAUDE_CONFIG_DIR/settings.json`` while this hashed a file nothing
    touches, so a test that escaped isolation could rewrite the real settings
    and the session would still end green.
    """
    real_settings = _real_settings_file()

    # Record state at session start
    before_exists = real_settings.exists()
    before_hash = None
    before_mtime = None

    if before_exists:
        before_mtime = real_settings.stat().st_mtime
        before_hash = hashlib.md5(real_settings.read_bytes()).hexdigest()

    # Yield to allow tests to run
    yield

    # Check state at session end
    after_exists = real_settings.exists()

    if before_exists != after_exists:
        raise AssertionError(
            f"Settings.json existence changed during test session. "
            f"Before: exists={before_exists}, After: exists={after_exists}. "
            f"Tests must use isolated HOME directories to avoid modifying "
            f"~/.claude/settings.json."
        )

    if before_exists:
        after_mtime = real_settings.stat().st_mtime
        after_hash = hashlib.md5(real_settings.read_bytes()).hexdigest()

        if before_mtime != after_mtime or before_hash != after_hash:
            raise AssertionError(
                f"{real_settings} was modified during test session. "
                f"Before hash: {before_hash}, After hash: {after_hash}. "
                f"Tests must use isolated HOME directories via env= parameter "
                f"to avoid modifying the real settings.json."
            )


def require_scoped_real_corpus(project_dir) -> None:
    """Skip when the real corpus resolves to zero sessions for this checkout.

    has_real_corpus() alone is a proxy: it checks that a projects/ root
    exists, not that any transcript's recorded cwd falls under THIS
    checkout's path. A git worktree (or clone) at a non-canonical path
    legitimately scopes to zero sessions, and every real-corpus test that
    asserts non-zero counts would then fail on corpus *availability*, not
    on the behavior it tests. Skipping with an explicit reason keeps those
    tests meaningful on the canonical checkout and hermetic everywhere
    else -- the same fix test_per_turn_segmentation_mirror received.
    """
    import pytest as _pytest

    from ssgrep import discovery as _discovery
    from ssgrep.paths import resolve_claude_dir as _resolve_claude_dir

    corpus_root = _resolve_claude_dir() / "projects"
    if not corpus_root.exists() or not any(corpus_root.iterdir()):
        _pytest.skip("real Claude Code corpus unavailable")
    if not _discovery.discover_sessions(Path(project_dir)):
        _pytest.skip(
            f"real corpus present but no session's recorded cwd falls under "
            f"{project_dir} (worktree/clone at a non-canonical path)"
        )


def _real_settings_file() -> Path:
    """The settings.json a *non-isolated* test would write into, right now.

    Resolved through ssgrep's own ``resolve_claude_dir()`` -- the exact
    expression ``hooks._settings_path()`` uses -- so a developer who relocated
    Claude Code with ``CLAUDE_CONFIG_DIR`` gets their real settings guarded
    rather than an untouched ``~/.claude/settings.json``. Mirrors
    ``_real_projects_dir()``.
    """
    from ssgrep.paths import resolve_claude_dir

    return resolve_claude_dir() / "settings.json"


@pytest.fixture(scope="session", autouse=True)
def guard_no_settings_debris_in_cwd() -> None:
    """Guard: no settings.json may appear next to the working directory.

    The shape this catches is a hook install whose config root resolved to a
    RELATIVE path -- ``Path("")/"settings.json"`` and ``Path("  ")/
    "settings.json"`` both land under the process CWD, i.e. the repo root.
    That is not hypothetical: this repo's working tree carried a stray
    ``settings.json`` and a directory literally named two spaces, both holding
    real SessionEnd hook documents naming a pytest ``tmp_path`` project, left
    behind by exactly that failure mode. No guard noticed, because
    ``guard_real_settings_unchanged`` watches one absolute path and the corpus
    guard watches another.

    Session-scoped and CWD-relative on purpose: it does not care which test
    did it, only that nothing writes Claude Code settings into the tree the
    suite runs from.
    """
    cwd = Path.cwd()
    before = {path for path in cwd.glob("*/settings.json")} | {
        path for path in cwd.glob("settings.json")
    }

    yield

    after = {path for path in cwd.glob("*/settings.json")} | {
        path for path in cwd.glob("settings.json")
    }
    new = sorted(str(path) for path in after - before)
    if new:
        raise AssertionError(
            f"A settings.json appeared under the working directory during the test "
            f"session: {new}. This is a hook install whose config root resolved to a "
            f"relative path (an empty or whitespace-only CLAUDE_CONFIG_DIR is the "
            f"known cause). Tests must isolate HOME/CLAUDE_CONFIG_DIR before calling "
            f"hooks._install_hook. The listed files are debris and safe to delete."
        )


@pytest.fixture(autouse=True)
def _no_inherited_claude_config_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts from "Claude Code is not relocated".

    ``resolve_claude_dir()`` consults ``CLAUDE_CONFIG_DIR`` BEFORE ``HOME``, so
    the usual isolation moves -- patching ``Path.home`` or setting ``HOME`` --
    have no effect at all on a machine where that variable is exported. That
    made 19 tests fail outright for any developer or CI runner running the
    configuration this feature exists to support, and left subprocess tests
    that copy ``os.environ`` running against the developer's real corpus.

    Fixed here rather than in each test file because it is an invariant of the
    whole suite, not a property of four files: any test written from now on
    inherits the isolation without knowing this hazard exists. Tests that are
    ABOUT relocation still call ``monkeypatch.setenv("CLAUDE_CONFIG_DIR", ...)``
    in their own body, which runs after this and wins; because monkeypatch
    mutates ``os.environ`` itself, subprocesses inherit the right value either
    way.
    """
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)


@pytest.fixture(autouse=True)
def _no_inherited_ssgrep_transcript_dirs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts with SSGREP_TRANSCRIPT_DIRS unset.

    ``discovery_roots.discover_external()`` reads ``os.environ`` directly, so a
    developer who has SSGREP_TRANSCRIPT_DIRS configured in their shell (exactly
    the scenario the README now documents) would have those external files folded
    into every test's synthetic corpus, silently corrupting session_count,
    episode_count, and staleness assertions. Live-reproduced by the 2026-08-07
    transcript-dirs blind review: SSGREP_TRANSCRIPT_DIRS=/tmp/leak-root produced
    72 FAILED / 789 passed with a single-file external root.

    The fix mirrors ``_no_inherited_claude_config_dir`` above -- both are
    suite-wide invariants, not per-file concerns. Tests that ARE about external
    roots set ``monkeypatch.setenv(discovery_roots.ENV_VAR, ...)`` in their own
    body; because monkeypatch mutates ``os.environ``, subprocesses inherit the
    right value either way.
    """
    monkeypatch.delenv("SSGREP_TRANSCRIPT_DIRS", raising=False)


def _real_projects_dir() -> Path:
    """The transcript root a *non-isolated* test would write into, right now.

    Resolved through ssgrep's own ``resolve_claude_dir()`` rather than a
    hardcoded ``~/.claude``: that is the exact expression production code
    uses to find the corpus, so a developer who has relocated Claude Code
    with ``CLAUDE_CONFIG_DIR`` gets their real corpus guarded, not an
    empty path that makes the guard vacuous.
    """
    from ssgrep.paths import resolve_claude_dir

    return resolve_claude_dir() / "projects"


def _project_entry_names(projects_dir: Path) -> frozenset[str] | None:
    """Names of the top-level entries in ``projects_dir``, or None if unreadable.

    Top level only, deliberately. Claude Code is usually *running* while
    this suite runs — it appends to transcripts and creates new session
    files inside the current project's directory continuously — so a
    recursive snapshot would fail the run on the tool's own normal writes.
    A new or vanished top-level entry is a different animal: that is a
    whole project appearing or disappearing, which the live tool does not
    do mid-run, and it is exactly the shape test pollution takes (a test
    computing its own project directory name and calling mkdir under the
    real root).
    """
    try:
        return frozenset(entry.name for entry in projects_dir.iterdir())
    except OSError:
        return None


@pytest.fixture(scope="session", autouse=True)
def guard_real_projects_unchanged():
    """Guard: fail the run if the real ~/.claude/projects gains or loses entries.

    The third member of the family, alongside
    :func:`guard_real_settings_unchanged` and
    :func:`guard_real_index_unchanged`, and added for the same reason they
    exist: a test escaped isolation and wrote into the buyer's real corpus.
    Concretely, a since-removed ``test_prune_e2e_with_real_api`` helper did

        encoded = hashlib.md5(str(proj_dir).encode()).hexdigest()
        (Path.home() / ".claude" / "projects" / encoded).mkdir(parents=True)

    with no HOME isolation at all, leaving two synthetic project
    directories in the user's history. The settings guard and the index
    guard both looked straight past it — the corpus itself was the one
    thing nothing watched, and silently corrupting a buyer's indexed
    history is the worst outcome this product has.

    Note what this guard is NOT: it is not a check that some particular
    test behaves. It compares the real corpus root before and after the
    entire session, so it fires for any test — existing, new, or written
    years from now — that reaches the real transcript root by any route:
    ``Path.home()``, an unset ``CLAUDE_CONFIG_DIR``, a subprocess that
    inherits the developer's environment, or a fixture whose monkeypatch
    was undone too early.

    Tests that need a corpus must build one under ``tmp_path`` and point
    HOME (``isolated_home``, ``monkeypatch.setenv("HOME", ...)``) or
    ``CLAUDE_CONFIG_DIR`` at it. Tests that only *read* the real corpus
    (discovery, perf) are unaffected: reading adds and removes nothing.

    One benign way to trip this: opening Claude Code in a *new* project
    directory for the first time while the suite is running creates a
    top-level entry that no test wrote. The failure message prints the
    entry's full path, so that case is a two-second read — an unfamiliar
    hex-looking name is test debris, a path-encoded project name of your
    own is not.
    """
    projects_dir = _real_projects_dir()
    before = _project_entry_names(projects_dir)

    yield

    after = _project_entry_names(projects_dir)

    if before is None or after is None:
        if before is not None or after is not None:
            raise AssertionError(
                f"The real transcript root {projects_dir} became "
                f"{'unreadable' if after is None else 'readable'} during the test "
                f"session (readable before={before is not None}, "
                f"after={after is not None}). A test created, deleted, or changed "
                f"the permissions of the real corpus root. Tests must point HOME or "
                f"CLAUDE_CONFIG_DIR at a tmp_path corpus (see the isolated_home "
                f"fixture)."
            )
        return

    if before == after:
        return

    added = sorted(after - before)
    removed = sorted(before - after)
    raise AssertionError(
        f"The real Claude Code corpus at {projects_dir} changed during the test "
        f"session. Added: {[str(projects_dir / name) for name in added]}. "
        f"Removed: {[str(projects_dir / name) for name in removed]}. "
        f"A test escaped HOME isolation and wrote to (or deleted from) the user's "
        f"real transcript history — this is the buyer's data and the entire product. "
        f"Build the corpus under tmp_path and point HOME at it (the isolated_home "
        f"fixture), or set CLAUDE_CONFIG_DIR; never derive a corpus path from "
        f"Path.home() without isolating HOME first. Any directory listed under "
        f"'Added' above is test debris and is safe to delete."
    )


@pytest.fixture(scope="session")
def isolated_real_corpus_index(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-scoped fixture: a real-corpus index, fully isolated from the repo.

    Builds exactly one index of this repo's real session history and never
    touches `<repo>/.ssgrep/` to do it — not to back it up, move it, or
    restore it. That "swap the real path and restore it after" pattern used
    to live in this fixture's place and was the root cause of a defect
    that survived five rounds of fixing: any interruption between its rmtree()
    of the real .ssgrep/ and its restore in `finally` left the developer's
    real index permanently deleted. Restoration is exactly what fails under
    interruption, so a test that manipulates the real path is not isolated
    no matter how carefully it restores afterwards.

    The actual mechanism: indexer.index() (unlike api.index(), which is the
    frozen contract and has no such override) accepts an explicit index_dir
    that decouples "where to write the index" from "which project's corpus
    to discover." discover_sessions() matches session records' recorded cwd
    against the literal project_dir argument, so project_dir must stay this
    repo's real path for the real corpus to be found at all — but index_dir
    can, and here does, point anywhere else. We build straight into
    <isolated tmp dir>/.ssgrep/, so the real .ssgrep/ is never opened,
    copied, moved, or deleted at any point.

    Session-scoped and read-only for every consumer: a full corpus index
    costs ~15s (see test_perf.py), so it is built once and shared by every
    test that only reads it via search/show/status — through the API
    in-process, the CLI's --project-dir, or the MCP server's cwd-based
    resolution, all three of which resolve <project_dir>/.ssgrep with no
    override, so pointing any of them at the returned directory reads this
    isolated index without any of them ever touching the real one. Safe to
    share: search()/show()/status() only ever write to the index they read
    (via search()'s bounded tail-repair of appended files), and discovery
    scoped to this isolated project_dir can never find real recorded
    sessions here — cursors for every indexed file show as VANISHED, never
    APPENDED, so that repair path can never trigger. A test that needs to
    build (not just read) an index — e.g. to measure indexing itself — must
    call indexer.index(..., index_dir=...) directly with its own tmp_path,
    never write into this shared fixture's directory.

    Returns:
        Path to an isolated project directory (never the repo root) whose
        .ssgrep/ holds a real-corpus-built index.
    """
    from ssgrep import indexer
    from ssgrep.paths import resolve_claude_dir

    # Skip if the real corpus is not available
    corpus_root = resolve_claude_dir() / "projects"
    if not corpus_root.exists():
        pytest.skip(
            reason=(
                f"Real Claude Code corpus not available at {corpus_root}; "
                "isolated_real_corpus_index fixture skipped"
            )
        )

    real_project_dir = Path(__file__).resolve().parent.parent
    isolated_project_dir = tmp_path_factory.mktemp("real-corpus-project")
    index_stats = indexer.index(
        real_project_dir, index_dir=isolated_project_dir / ".ssgrep", quiet=True
    )

    # Re-pin the persisted scope to the ISOLATED path. The index was built
    # with project_dir=real repo (so discovery finds the real corpus), which
    # persists the repo path as the index's scope -- but this fixture's
    # safety contract (docstring above) depends on every consumer seeing the
    # indexed files as VANISHED-never-APPENDED so search()'s tail repair can
    # never write into the shared index. With scope-aware staleness
    # (search._read_persisted_scope), a repo-scoped index consumed at the
    # isolated path would resolve the LIVE corpus again and could see
    # APPENDED files. Persisting the isolated path restores the
    # vanished-by-design semantics explicitly, using the scope feature's own
    # mechanism instead of relying on scope re-derivation.
    import sqlite3 as _sqlite3

    from ssgrep import store as _store

    _gen = _store.GenerationalStore(isolated_project_dir / ".ssgrep")
    _conn = _sqlite3.connect(str(_gen.get_index_path()))
    try:
        _store.set_meta(_conn, "scope", str(isolated_project_dir))
        _conn.commit()
    finally:
        _conn.close()

    # Skip if the index is empty (zero sessions or chunks)
    if index_stats.session_count == 0 or index_stats.chunk_count == 0:
        pytest.skip(
            reason=(
                f"Real corpus index is empty (sessions={index_stats.session_count}, "
                f"chunks={index_stats.chunk_count}); isolated_real_corpus_index fixture skipped"
            )
        )

    return isolated_project_dir


@pytest.fixture(scope="session", autouse=True)
def guard_real_index_unchanged() -> None:
    """Guard: ensure test suite does not modify the repo's .ssgrep/ directory.

    This fixture runs at session start and end, capturing the hash of the
    .ssgrep/.manifest file. If the file changes during test execution, the
    session fails with a clear error message.

    This prevents tests from polluting the real repository index, which can
    produce false fast-search latency readings and confuse performance testing.

    The failure message names what changed (generation, vector_row_count,
    timestamp), not just that something did, and distinguishes a rebuild
    (generation moves) from a bounded tail repair (generation holds, only
    vector_row_count/timestamp move — search()'s incremental self-heal on an
    appended session). Session-scoped teardown otherwise attaches this error
    to whichever test happens to run last, which cost real diagnostic time
    once: a tail repair triggered by tests/test_mcp_server.py surfaced as a
    failure on tests/test_workqueue_drain.py, which was innocent.
    """
    real_repo = Path(__file__).resolve().parent.parent
    real_index_dir = real_repo / ".ssgrep"
    manifest_path = real_index_dir / ".manifest"

    # Record state at session start
    before_exists = real_index_dir.exists()
    before_manifest_exists = manifest_path.exists() if before_exists else False
    before_hash = None
    before_mtime = None
    before_generation = None
    before_row_count = None
    before_timestamp = None

    if before_manifest_exists:
        try:
            before_mtime = manifest_path.stat().st_mtime
            manifest_text = manifest_path.read_text()
            before_hash = hashlib.md5(manifest_text.encode()).hexdigest()
            before_manifest_data = json.loads(manifest_text)
            before_generation = before_manifest_data.get("generation")
            before_row_count = before_manifest_data.get("vector_row_count")
            before_timestamp = before_manifest_data.get("timestamp")
        except (OSError, json.JSONDecodeError):
            # If we can't read the manifest, skip the check (test environment issue)
            before_manifest_exists = False

    # Yield to allow tests to run
    yield

    # Check state at session end
    after_exists = real_index_dir.exists()
    after_manifest_exists = manifest_path.exists() if after_exists else False

    if before_exists and not after_exists:
        raise AssertionError(
            f"Index directory {real_index_dir} was deleted during test session. "
            f"Tests must use isolated index directories via indexer.index(..., index_dir=...) "
            f"to avoid modifying the repository's .ssgrep/ directory."
        )

    if before_manifest_exists and after_manifest_exists:
        try:
            after_mtime = manifest_path.stat().st_mtime
            manifest_text = manifest_path.read_text()
            after_hash = hashlib.md5(manifest_text.encode()).hexdigest()
            after_manifest_data = json.loads(manifest_text)
            after_generation = after_manifest_data.get("generation")
            after_row_count = after_manifest_data.get("vector_row_count")
            after_timestamp = after_manifest_data.get("timestamp")

            if before_hash != after_hash or before_mtime != after_mtime:
                if before_generation == after_generation:
                    diagnosis = (
                        "generation UNCHANGED: this is a bounded tail repair, not a "
                        "rebuild — search()'s incremental self-heal ran against an "
                        "appended session file. Likely cause: an in-process "
                        "search()/api.search() call, or an MCP search_sessions/"
                        "get_mcp_server() call, resolved project_dir to this repo's "
                        "real path (e.g. via Path.cwd()) with the real corpus "
                        "available, instead of an isolated project_dir."
                    )
                else:
                    diagnosis = (
                        "generation CHANGED: this is a full index build or rebuild "
                        "against the real path, not a repair. Likely cause: an "
                        "api.index()/indexer.index() call (or `ssgrep index` "
                        "subprocess) is missing its index_dir/--project-dir "
                        "isolation."
                    )
                raise AssertionError(
                    f"Repository index was modified during test session — {diagnosis} "
                    f"Generation: {before_generation} -> {after_generation}, "
                    f"vector_row_count: {before_row_count} -> {after_row_count}, "
                    f"timestamp: {before_timestamp} -> {after_timestamp}, "
                    f"Manifest hash: {before_hash} -> {after_hash}. "
                    f"Tests must use isolated index directories via "
                    f"indexer.index(..., index_dir=tmp_path / 'idx') "
                    f"to avoid writing to {real_index_dir}."
                )
        except (OSError, json.JSONDecodeError) as e:
            raise AssertionError(
                f"Failed to validate repository index integrity: {e}. "
                f"This may indicate an index corruption or file access error."
            ) from e
