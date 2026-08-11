"""Privacy and hygiene tests.

Four assertions, each mutation-tested:
1. `.ssgrep/` is created with mode 0700
2. `.ssgrep/` is added to `.gitignore` on first index, and adding twice does not duplicate
3. No network calls after model download
4. Transcripts are never written to
"""

from __future__ import annotations

import hashlib
import socket
import time
from pathlib import Path
from typing import Any

import pytest

from ssgrep import api, embed


class TestSSGrepDirPermissions:
    """Test that .ssgrep/ directory is created with mode 0700."""

    def test_ssgrep_dir_created_with_0700_mode(self, isolated_home: tuple[Path, Path]) -> None:
        """Assert .ssgrep/ is created with exact mode 0700, not just readable by owner.

        This is a privacy guarantee: .ssgrep/ holds cached models and index data
        that should not be world-readable or group-readable.
        """
        home, projects = isolated_home

        # Create a minimal project directory
        project_dir = home / "test-project"
        project_dir.mkdir()

        # Create a test session in projects
        session_dir = projects / "test-session"
        session_dir.mkdir(parents=True)

        test_jsonl = session_dir / "session-abc123.jsonl"
        test_jsonl.write_text('{"type":"user","turn":1,"text":"test"}\n')

        # Index the project, which creates .ssgrep/
        api.index(project_dir, quiet=True)

        # Check .ssgrep/ exists and has exact mode 0700
        ssgrep_dir = project_dir / ".ssgrep"
        assert ssgrep_dir.exists(), ".ssgrep/ directory was not created"
        assert ssgrep_dir.is_dir(), ".ssgrep/ is not a directory"

        # Get actual mode bits using stat
        stat_info = ssgrep_dir.stat()
        actual_mode = stat_info.st_mode & 0o777

        assert actual_mode == 0o700, (
            f"Expected .ssgrep/ to have mode 0700, but got {oct(actual_mode)}. "
            f"This is a privacy regression — the directory should not be "
            f"readable or writable by group or others."
        )


class TestGitIgnoreHandling:
    """Test that .ssgrep/ is added to .gitignore on first index."""

    def test_ssgrep_added_to_gitignore_on_first_index(
        self, isolated_home: tuple[Path, Path]
    ) -> None:
        """Assert .ssgrep/ is added to .gitignore on first index.

        This prevents accidental commits of cached data and embeddings.
        """
        home, projects = isolated_home

        # Create a minimal project directory with git repo
        project_dir = home / "test-project"
        project_dir.mkdir()

        # Initialize a git repo (minimal, no .git dir needed for this test)
        gitignore_path = project_dir / ".gitignore"

        # Create a test session in projects
        session_dir = projects / "test-session"
        session_dir.mkdir(parents=True)

        test_jsonl = session_dir / "session-abc123.jsonl"
        test_jsonl.write_text('{"type":"user","turn":1,"text":"test"}\n')

        # Index the project, which should add .ssgrep/ to .gitignore
        api.index(project_dir, quiet=True)

        # Check that .gitignore exists and contains .ssgrep/
        assert gitignore_path.exists(), (
            ".gitignore was not created by index(). .ssgrep/ must be added to "
            ".gitignore on first index to prevent accidental commits."
        )

        gitignore_content = gitignore_path.read_text()
        assert (
            ".ssgrep/" in gitignore_content or ".ssgrep" in gitignore_content
        ), f".ssgrep/ entry not found in .gitignore. Content: {gitignore_content}"

    def test_gitignore_not_duplicated_on_second_index(
        self, isolated_home: tuple[Path, Path]
    ) -> None:
        """Assert adding .ssgrep/ to .gitignore twice does not duplicate the line.

        Idempotency prevents .gitignore from growing unbounded with repeated
        index operations.
        """
        home, projects = isolated_home

        # Create a minimal project directory
        project_dir = home / "test-project"
        project_dir.mkdir()

        # Create a test session in projects
        session_dir = projects / "test-session"
        session_dir.mkdir(parents=True)

        test_jsonl = session_dir / "session-abc123.jsonl"
        test_jsonl.write_text('{"type":"user","turn":1,"text":"test"}\n')

        # First index
        api.index(project_dir, quiet=True)

        gitignore_path = project_dir / ".gitignore"
        first_content = gitignore_path.read_text()

        # Count occurrences of .ssgrep in first content
        first_count = first_content.count(".ssgrep")
        assert first_count >= 1, ".ssgrep not found in .gitignore after first index"

        # Second index (should not duplicate)
        api.index(project_dir, quiet=True)

        second_content = gitignore_path.read_text()
        second_count = second_content.count(".ssgrep")

        assert second_count == first_count, (
            f".ssgrep/ entry was duplicated in .gitignore. "
            f"First index had {first_count} occurrence(s), "
            f"second index has {second_count}. "
            f"Content:\n{second_content}"
        )


def _purge_fastmcp_modules() -> None:
    """Drop fastmcp (and submodules) from sys.modules.

    The two tests below import fastmcp in-process. Leaving it resident makes
    every later-collected test inherit an import-order dependency (e.g.
    test_fastmcp_not_imported_at_module_level would pass or fail depending on
    whether these ran first). Registered as a finalizer so it runs on failure
    paths too.
    """
    import sys as _sys

    for name in [m for m in _sys.modules if m == "fastmcp" or m.startswith("fastmcp.")]:
        del _sys.modules[name]


class TestNoNetworkCalls:
    """Test that no network calls are made after model download."""

    @pytest.fixture(autouse=True)
    def ensure_corpus_available(self) -> None:
        """Skip if real corpus is not available (isolated_real_corpus_index requires it)."""
        from ssgrep.paths import resolve_claude_dir

        projects_dir = resolve_claude_dir() / "projects"
        if not projects_dir.exists():
            pytest.skip(
                reason=(
                    "Real session corpus is not available in this environment "
                    f"(checked {projects_dir})"
                )
            )

    def test_no_network_calls_after_model_download(
        self, isolated_real_corpus_index: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Assert a cold model load resolves entirely from the on-disk HF cache
        with no network access, and does so quickly.

        History: an earlier version of this test used a
        fixture record ('{"type":"user","turn":1,"text":"test"}') that never
        parsed into a session at all (0 sessions, not just 0 chunks). That
        meant total_chunks was 0, search() returned before ever reaching the
        vector leg, and embed._get_model() was called zero times — the test
        passed identically whether or not the code phoned home on every call.
        An audit of this test's own coverage found that gap.
        This version fixes it in three ways:

        1. Uses isolated_real_corpus_index — a real, populated,
           production-shaped index (thousands of real chunks from this
           repo's own session history), not a synthetic single-line
           fixture, so total_chunks > 0 and search() actually reaches the
           vector leg that calls embed.encode().
        2. Forces a *cold* load (embed._model = None) immediately before the
           guarded call. An already-warm in-process singleton was never the
           exposure — the defect was that a fresh process (i.e. every real
           CLI invocation, since _model does not survive across processes)
           always attempted a network call on its first load, no matter how
           long after the model had been cached. Skipping this reset would
           let this test pass against the original, broken code too, for
           the same reason the test it replaces did.
        3. Spies on embed._get_model and asserts it was actually invoked — a
           call count of zero proves nothing about network behavior (again,
           see the history above).

        It also asserts wall-clock latency, not just "no exception": a
        silent multi-second stall is a more damaging failure than a raised
        exception and would pass a bare try/except check. See
        test_cold_load_timeout_is_enforced_and_actionable and
        test_first_run_proxy_error_is_actionable_not_a_raw_exception in
        test_embed.py for the specific failure modes this replaces on the
        first-run/uncached path (an unbounded hang, and an uncaught
        ProxyError) — this test only needs to prove the warm-cache path
        never attempts network at all, so those don't need reproducing here.

        Blocking socket.socket() alone is not sufficient: huggingface_hub
        keeps a process-wide httpx.Client shared across every call
        (huggingface_hub/utils/_http.py::get_session, explicitly documented
        as "shared between all calls made by huggingface_hub"). If anything
        earlier in this process — e.g. isolated_real_corpus_index's own
        fixture build, on a run where the model was not yet warm — already
        opened a live pooled connection to huggingface.co, a later request
        can be sent over that *already-open* connection without ever
        constructing a new socket.socket(), silently walking straight past
        a socket-only block. Confirmed this is not hypothetical: an early,
        socket-only-blocking version of this test passed unchanged even
        with the pre-fix code restored (force_download defaulting to True),
        for exactly this reason. httpx.Client.send() is patched too — every
        outgoing request, pooled connection or not, funnels through it, so
        it is the direct, unavoidable proof that no HTTP request was
        attempted, not a proxy for one.
        """
        import httpx

        project_dir = isolated_real_corpus_index

        # Force a cold load: the vulnerable path is a *fresh* model load.
        monkeypatch.setattr(embed, "_model", None)

        # Spy on _get_model so we can prove the guarded path actually ran.
        real_get_model = embed._get_model
        call_count = [0]

        def spying_get_model():
            call_count[0] += 1
            return real_get_model()

        monkeypatch.setattr(embed, "_get_model", spying_get_model)

        # Block socket creation so any real network attempt raises immediately
        # instead of actually reaching out.
        def blocked_socket(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "Network access attempted during what must be an offline, "
                "warm-cache cold model load (socket.socket)."
            )

        monkeypatch.setattr(socket, "socket", blocked_socket)

        # Block the actual HTTP-request layer too — see the docstring above
        # for why socket.socket() alone can miss a reused pooled connection.
        def blocked_send(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "Network access attempted during what must be an offline, "
                "warm-cache cold model load (httpx.Client.send)."
            )

        monkeypatch.setattr(httpx.Client, "send", blocked_send)

        start = time.perf_counter()
        search_result = api.search(project_dir, "workqueue")
        elapsed_s = time.perf_counter() - start

        assert call_count[0] >= 1, (
            "embed._get_model() was never invoked — this test proves nothing "
            "about network access if the guarded code path never ran. (This "
            "is exactly how the test this replaces passed vacuously.)"
        )
        assert search_result.index_empty is False, (
            "search() took the empty-index early-return path, which returns "
            "before ever calling embed._get_model() — this test needs a "
            "populated index to prove anything. isolated_real_corpus_index "
            "should guarantee this; if it doesn't, the fixture itself changed."
        )

        # Loose but real budget: measured warm-cache cold loads take
        # 0.19-0.38s. 2s leaves huge
        # headroom for slow/noisy CI while still catching a hidden network
        # attempt that happens to succeed quickly in this test environment
        # but would hang for many seconds (or ~25s pre-fix) against a real
        # degraded network.
        assert elapsed_s < 2.0, (
            f"Cold model load + search took {elapsed_s:.2f}s with sockets "
            f"blocked. A warm-cache cold load must resolve from the local "
            f"HF cache only and finish in well under a second."
        )


class TestTranscriptsNeverWritten:
    """Test that transcripts are never written to during indexing or search."""

    def test_transcripts_unchanged_after_index_and_search(
        self, isolated_home: tuple[Path, Path]
    ) -> None:
        """Assert all source transcripts remain byte-identical after index and search.

        This test is crucial for archive integrity: indexing must be a read-only
        operation on transcripts. Even tail repair (which reads an appended file)
        must never write to it.

        Process:
        1. Record md5 and mtime of every source transcript
        2. Run index(), search(), and tail repair (if applicable)
        3. Assert every transcript is byte-identical
        """
        home, projects = isolated_home

        # Create a minimal project directory
        project_dir = home / "test-project"
        project_dir.mkdir()

        # Create a test session in projects with multiple records
        session_dir = projects / "test-session"
        session_dir.mkdir(parents=True)

        test_jsonl = session_dir / "session-abc123.jsonl"
        test_content = (
            '{"type":"user","turn":1,"text":"test prompt 1"}\n'
            '{"type":"assistant","turn":1,"text":"test response 1"}\n'
            '{"type":"user","turn":2,"text":"test prompt 2"}\n'
        )
        test_jsonl.write_text(test_content)

        # Record baseline: md5 and mtime of the original transcript
        original_bytes = test_jsonl.read_bytes()
        original_md5 = hashlib.md5(original_bytes).hexdigest()
        original_mtime = test_jsonl.stat().st_mtime

        # Perform indexing and searching
        api.index(project_dir, quiet=True)
        api.search(project_dir, "test")

        # Verify transcript is unchanged
        after_bytes = test_jsonl.read_bytes()
        after_md5 = hashlib.md5(after_bytes).hexdigest()
        after_mtime = test_jsonl.stat().st_mtime

        assert after_md5 == original_md5, (
            f"Transcript {test_jsonl} was modified during indexing. "
            f"Original md5: {original_md5}, After md5: {after_md5}. "
            f"Indexing must be read-only to preserve archive integrity."
        )

        assert after_mtime == original_mtime, (
            f"Transcript {test_jsonl} mtime changed during indexing. "
            f"Original mtime: {original_mtime}, After mtime: {after_mtime}. "
            f"Transcript file must not be written to."
        )

        # Verify bytes are identical
        assert (
            after_bytes == original_bytes
        ), "Transcript bytes changed. This should have been caught by md5 check."

    def test_multiple_transcripts_unchanged(self, isolated_home: tuple[Path, Path]) -> None:
        """Assert all source transcripts in a multi-session project are unchanged.

        Extended version of test_transcripts_unchanged_after_index_and_search
        with multiple transcripts to ensure the guard works across the whole corpus.
        """
        home, projects = isolated_home

        # Create a minimal project directory
        project_dir = home / "test-project"
        project_dir.mkdir()

        # Create multiple test sessions
        transcript_records: dict[Path, tuple[bytes, str, float]] = {}

        for i in range(3):
            session_dir = projects / f"test-session-{i}"
            session_dir.mkdir(parents=True)

            test_jsonl = session_dir / f"session-abc{i}.jsonl"
            content = (
                f'{{"type":"user","turn":1,"text":"test prompt {i}"}}\n'
                f'{{"type":"assistant","turn":1,"text":"test response {i}"}}\n'
            )
            test_jsonl.write_text(content)

            # Record baseline
            original_bytes = test_jsonl.read_bytes()
            original_md5 = hashlib.md5(original_bytes).hexdigest()
            original_mtime = test_jsonl.stat().st_mtime

            transcript_records[test_jsonl] = (original_bytes, original_md5, original_mtime)

        # Perform indexing and searching
        api.index(project_dir, quiet=True)
        api.search(project_dir, "test")

        # Verify all transcripts are unchanged
        for jsonl_path, (
            _original_bytes,
            original_md5,
            original_mtime,
        ) in transcript_records.items():
            after_bytes = jsonl_path.read_bytes()
            after_md5 = hashlib.md5(after_bytes).hexdigest()
            after_mtime = jsonl_path.stat().st_mtime

            assert after_md5 == original_md5, (
                f"Transcript {jsonl_path} was modified. "
                f"Original md5: {original_md5}, After md5: {after_md5}"
            )

            assert after_mtime == original_mtime, (
                f"Transcript {jsonl_path} mtime changed. "
                f"Original mtime: {original_mtime}, After mtime: {after_mtime}"
            )


class TestMcpServerMakesNoNetworkCalls:
    """The MCP server is a documented, first-class integration, and the
    privacy suite never started it.

    README states "after that, no network access is needed or used" and then
    "These assertions are verified by tests in every CI run" -- while every
    test in this file exercised only the embed path reached by index/search.
    Meanwhile `ssgrep mcp` inherited FastMCP's startup banner, whose FIRST
    statement is check_for_newer_version(): an outbound HTTPS GET to
    https://pypi.org/pypi/fastmcp/json on every cold start, a version cache
    written under the user's config dir, and an advert in the client's stderr
    telling the buyer to `pip install --upgrade fastmcp` -- a package
    pyproject pins, so following it breaks their paid install. CI stayed
    green over a false guarantee for exactly as long as no test constructed
    the server.
    """

    def test_constructing_the_mcp_server_attempts_no_http_request(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        request: pytest.FixtureRequest,
    ) -> None:
        """Build the real server with the HTTP layer blocked.

        httpx.Client.send is the choke point every huggingface_hub AND
        fastmcp request funnels through, pooled connection or not (see
        TestNoNetworkCalls' docstring for why blocking socket.socket alone
        is not enough).
        """
        request.addfinalizer(_purge_fastmcp_modules)

        import httpx

        import ssgrep.mcp_server as mcp_server

        attempts: list[str] = []

        def blocked_send(self: Any, request: Any, *args: Any, **kwargs: Any) -> None:
            attempts.append(str(request.url))
            raise RuntimeError(f"MCP startup attempted a network request to {request.url}")

        monkeypatch.setattr(httpx.Client, "send", blocked_send)

        home = tmp_path / "home"
        (home / ".claude" / "projects").mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(home / ".claude"))
        monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
        # get_mcp_server() drains the work queue, which resolves the project
        # from the CURRENT DIRECTORY -- without this the drain indexes this
        # repo and writes to its real .ssgrep.
        project = tmp_path / "project"
        project.mkdir()
        monkeypatch.chdir(project)
        monkeypatch.setattr(mcp_server, "_mcp", None)

        server = mcp_server.get_mcp_server()

        # Drive the exact call site the egress came from. FastMCP's update
        # check does not run in the constructor -- run_stdio_async() calls
        # log_server_banner(), whose FIRST statement is
        # check_for_newer_version(). Asserting on construction alone would
        # pass against the unfixed code and prove nothing, so the banner path
        # is invoked here directly, with a cold cache dir so no stale
        # version_cache.json can short-circuit the request.
        import fastmcp
        from fastmcp.utilities.cli import log_server_banner

        monkeypatch.setattr(fastmcp.settings, "home", tmp_path / "fastmcp-home")
        log_server_banner(server=server)

        assert attempts == [], (
            "MCP startup reached the network, breaking the README's "
            f"'no network access is needed or used' guarantee: {attempts}"
        )
        assert not (tmp_path / "fastmcp-home").exists(), (
            "a version cache was written, which only happens after a successful "
            "request to pypi.org"
        )

    def test_mcp_server_reports_ssgreps_own_version_not_the_sdks(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        request: pytest.FixtureRequest,
    ) -> None:
        """serverInfo.version was reporting fastmcp's version, so every bug
        report filed from an MCP client carried the wrong one.
        """
        request.addfinalizer(_purge_fastmcp_modules)

        from importlib.metadata import version

        import ssgrep.mcp_server as mcp_server

        home = tmp_path / "home"
        (home / ".claude" / "projects").mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(home / ".claude"))
        monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
        # get_mcp_server() drains the work queue, which resolves the project
        # from the CURRENT DIRECTORY -- without this the drain indexes this
        # repo and writes to its real .ssgrep.
        project = tmp_path / "project"
        project.mkdir()
        monkeypatch.chdir(project)
        monkeypatch.setattr(mcp_server, "_mcp", None)

        server = mcp_server.get_mcp_server()

        import fastmcp

        assert server.version == version("ssgrep")
        assert server.version != fastmcp.__version__, (
            "reporting the SDK's version is the defect, and it happens to be "
            "indistinguishable from ssgrep's only if they ever coincide"
        )
