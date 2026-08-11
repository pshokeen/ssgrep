"""Tests for lazy loading of the MCP server module."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_import_ssgrep_mcp_server_succeeds():
    """Test that importing ssgrep.mcp_server succeeds in the default environment.

    This test confirms that fastmcp is available as a main dependency.
    """
    import ssgrep.mcp_server  # noqa: F401


def test_fastmcp_not_imported_at_module_level():
    """Test that fastmcp is not imported when ssgrep.mcp_server is imported.

    This test must run in a subprocess with a clean interpreter, since
    importing fastmcp in any test will pollute sys.modules for other tests.
    """
    code = (
        "import ssgrep.mcp_server; import sys; "
        "assert 'fastmcp' not in sys.modules, 'fastmcp was imported at module level'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    error_msg = f"Subprocess failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    assert result.returncode == 0, error_msg
    assert "fastmcp was imported at module level" not in result.stderr


def test_get_mcp_server_imports_fastmcp():
    """Test that calling get_mcp_server() does import fastmcp.

    This confirms that laziness is deferral (to when the server is
    constructed) rather than removal.

    Isolated via chdir + reload, same pattern as the no-index tests below.
    get_mcp_server() is a process-wide singleton (module-level `_mcp`) whose
    first-ever construction unconditionally calls _drain_startup_workqueue(),
    which runs indexer.index(_get_project_dir(), quiet=True) with no
    index_dir override. _get_project_dir() is Path.cwd(), which during
    `uv run poe test` is this repo's real root — so an unisolated first call
    here would write straight into the real .ssgrep/ (see
    tests/conftest.py's guard_real_index_unchanged). Reloading the module in
    `finally` resets `_mcp` to None so this test's cached instance can never
    leak into a later, non-isolated caller either.
    """
    import importlib
    import os
    import tempfile

    import ssgrep.mcp_server

    # Before calling get_mcp_server, fastmcp should not be imported
    assert "fastmcp" not in sys.modules, "fastmcp was imported before get_mcp_server() was called"

    with tempfile.TemporaryDirectory(prefix="mcp-import-") as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            importlib.reload(ssgrep.mcp_server)

            # Call the server-construction function
            server = ssgrep.mcp_server.get_mcp_server()

            # After calling get_mcp_server, fastmcp should be imported
            assert "fastmcp" in sys.modules, "fastmcp was not imported by get_mcp_server()"

            # The server should be a valid FastMCP instance
            from fastmcp import FastMCP

            assert isinstance(server, FastMCP), f"expected a FastMCP server, got {type(server)}"
        finally:
            os.chdir(original_cwd)
            importlib.reload(ssgrep.mcp_server)


def test_mcp_server_lists_exactly_three_tools():
    """Test that the MCP server exposes exactly three tools.

    This is a hard requirement: search_sessions, show_session, index_status.
    Verifies the tool count and names match exactly.

    Isolated via chdir + reload — see test_get_mcp_server_imports_fastmcp's
    docstring for why get_mcp_server() must never construct against the real
    repo's cwd.
    """
    import asyncio
    import importlib
    import os
    import tempfile

    import ssgrep.mcp_server

    async def check_tools():
        server = ssgrep.mcp_server.get_mcp_server()
        tools = await server.list_tools()

        # Must be exactly 3 tools
        assert len(tools) == 3, f"Expected 3 tools, got {len(tools)}: {[t.name for t in tools]}"

        # Extract tool names
        tool_names = {tool.name for tool in tools}

        # Must be the exact tools
        expected = {"search_sessions", "show_session", "index_status"}
        assert tool_names == expected, f"Expected {expected}, got {tool_names}"

        # Verify each tool has required metadata (convert to MCP format for schema)
        for tool in tools:
            assert tool.name in expected, f"Unexpected tool: {tool.name}"
            assert tool.description, f"Tool {tool.name} missing description"
            # Get MCP-formatted tool (what clients actually receive)
            mcp_tool = tool.to_mcp_tool()
            assert mcp_tool.inputSchema is not None, f"Tool {tool.name} missing inputSchema"

    with tempfile.TemporaryDirectory(prefix="mcp-tools-") as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            importlib.reload(ssgrep.mcp_server)
            asyncio.run(check_tools())
        finally:
            os.chdir(original_cwd)
            importlib.reload(ssgrep.mcp_server)


def test_mcp_search_round_trip():
    """Test that search_sessions can be called through the MCP tool layer.

    This verifies the end-to-end MCP tool invocation, not just direct API calls.
    Requires an index to exist; will skip gracefully if not.

    Isolated via chdir + reload into an empty tmpdir — see
    test_get_mcp_server_imports_fastmcp's docstring for why get_mcp_server()
    must never construct against the real repo's cwd. With no .ssgrep/ at the
    isolated project dir, this deterministically exercises the "no index"
    branch below, which the test already handles as a pass (this file's other
    real-corpus round trip is tests/test_e2e_parity.py's
    test_cli_and_mcp_search_parity, via isolated_real_corpus_index).
    """
    import asyncio
    import importlib
    import json
    import os
    import tempfile

    import ssgrep.mcp_server
    from ssgrep.types import IndexNotFoundError, IndexNotReadyError

    async def search_via_mcp():
        server = ssgrep.mcp_server.get_mcp_server()

        # Try to search for something generic
        query = "test"

        try:
            # Call the tool via the server's call_tool method
            result = await server.call_tool("search_sessions", {"query": query, "limit": 5})

            # FastMCP returns a ToolResult object; extract the content
            if hasattr(result, "content"):
                # ToolResult with content list
                content = result.content
                if isinstance(content, list) and len(content) > 0:
                    # Extract text from the first content item
                    text_content = content[0]
                    if hasattr(text_content, "text"):
                        result_json = text_content.text
                    else:
                        result_json = str(text_content)
                else:
                    result_json = str(content)
            else:
                result_json = str(result)

            # Parse the JSON response
            if isinstance(result_json, str) and result_json.startswith("{"):
                result_data = json.loads(result_json)
            else:
                result_data = result_json

            # Either error (index not found) or results dict
            if isinstance(result_data, dict):
                # Either it's an error dict or a results dict
                if "error" in result_data:
                    # Index not found is OK for this test
                    error_msg = result_data["error"]
                    has_index_msg = "Run `ssgrep index` first" in error_msg
                    has_not_found = "not found" in error_msg.lower()
                    assert has_index_msg or has_not_found
                else:
                    # Should have results structure
                    assert (
                        "results" in result_data or "total_matches" in result_data
                    ), f"Expected results or total_matches in response, got: {result_data.keys()}"
        except (IndexNotFoundError, IndexNotReadyError):
            # Expected if no index exists
            pass

    with tempfile.TemporaryDirectory(prefix="mcp-roundtrip-") as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            importlib.reload(ssgrep.mcp_server)
            asyncio.run(search_via_mcp())
        finally:
            os.chdir(original_cwd)
            importlib.reload(ssgrep.mcp_server)


def _tools_list_via_real_stdio(timeout: float = 20.0) -> tuple[str, list[dict]]:
    """Speak real newline-delimited JSON-RPC 2.0 to the actual installed
    `ssgrep mcp` binary over its real stdin/stdout, exactly as a real MCP
    client would, and return the raw `tools/list` response line plus its
    parsed `tools` array.

    Every read goes through a background-thread queue with an explicit
    timeout (never a blocking read on the pipe), per this suite's standing
    rule that the MCP server is a stdio process that blocks on stdin and
    must never be allowed to hang a test. HOME and cwd are both isolated
    temp directories so this never touches the real ~/.claude/projects,
    ~/.claude/settings.json, or this repo's own .ssgrep/.

    This exists because the tool-definition sizes MCP clients actually pay
    for can only be observed on the wire: fastmcp's real tools/list
    response carries outputSchema, annotations, and _meta on every tool,
    none of which appear if you hand-build a {name, description,
    inputSchema} dict from the in-process FastMCP object (what this test
    used to do). See test_mcp_tool_definitions_token_count's docstring.
    """
    import queue
    import tempfile
    import threading

    from tests.conftest import get_ssgrep_binary

    binary = get_ssgrep_binary()

    with (
        tempfile.TemporaryDirectory(prefix="mcp-tokencount-home-") as home,
        tempfile.TemporaryDirectory(prefix="mcp-tokencount-project-") as project,
    ):
        (Path(home) / ".claude" / "projects").mkdir(parents=True)
        env = os.environ.copy()
        env["HOME"] = home

        proc = subprocess.Popen(
            [str(binary), "mcp"],
            cwd=project,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        out_q: queue.Queue[str | None] = queue.Queue()

        def reader() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                out_q.put(line)
            out_q.put(None)

        threading.Thread(target=reader, daemon=True).start()

        def send(obj: dict) -> None:
            assert proc.stdin is not None
            proc.stdin.write(json.dumps(obj) + "\n")
            proc.stdin.flush()

        def recv() -> str | None:
            try:
                return out_q.get(timeout=timeout)
            except queue.Empty:
                return None

        try:
            send(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-11-25",
                        "capabilities": {},
                        "clientInfo": {"name": "ssgrep-test-suite", "version": "0.0.1"},
                    },
                }
            )
            init_line = recv()
            assert (
                init_line is not None
            ), "timed out waiting for the real MCP server's initialize response"
            init_resp = json.loads(init_line)
            assert (
                "error" not in init_resp
            ), f"initialize returned a JSON-RPC error: {init_resp['error']}"

            send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
            send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})

            tools_line = recv()
            assert (
                tools_line is not None
            ), "timed out waiting for the real MCP server's tools/list response"
            data = json.loads(tools_line)
            assert "error" not in data, f"tools/list returned a JSON-RPC error: {data['error']}"
            return tools_line, data["result"]["tools"]
        finally:
            try:
                if proc.stdin:
                    proc.stdin.close()
            except Exception:
                pass
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


def test_mcp_tool_definitions_token_count():
    """Test that the three tool definitions stay under 1000 tokens, measured
    against the actual bytes a real MCP client receives over the wire.

    This used to measure a hand-built {name, description, inputSchema}
    dict reconstructed from the in-process FastMCP object -- close to, but
    not the same as, what fastmcp==3.4.4 actually serializes: the real
    tools/list response also carries outputSchema, annotations, and a
    per-tool _meta.fastmcp block, none of which the old reconstruction
    counted. That gap was real: the old method measured ~233 tokens
    (chars/4) against this same live server, matching the design comment's
    "~222 tokens" claim closely -- but the true wire payload (confirmed via
    tiktoken cl100k_base against a real stdio round-trip with the actual
    installed binary, independent of this test) is closer to ~288-303
    tokens, about 30% more. Both numbers are comfortably under this test's
    1000-token ceiling, so this was never a product defect -- but a test
    claiming to measure "tool definitions ... as they would be sent to a
    client" should measure bytes that were actually sent to a client, which
    is what this version does: a real stdio JSON-RPC round-trip against the
    real installed binary (see _tools_list_via_real_stdio), not an
    in-process reconstruction.

    Uses character count ÷ 4 as the approximation, same convention this
    file already used (tiktoken is not a project dependency, so it isn't
    used in the shipped assertion here — cross-checked once against
    tiktoken cl100k_base out of band; the two heuristics agree within
    ~10%, consistent with this file's own prior comment about the margin).
    """
    tools_line, tools = _tools_list_via_real_stdio()

    # Self-check that this test is actually measuring the real wire shape
    # and not quietly regressing back to the narrow reconstruction: every
    # tool must carry at least one of the fields the old measurement
    # missed, or this test would once again be measuring the wrong object.
    extra_fields_seen = set()
    for tool in tools:
        extra_fields_seen |= set(tool.keys()) & {"outputSchema", "annotations", "_meta"}
    assert extra_fields_seen, (
        f"none of outputSchema/annotations/_meta were present on the real wire tools -- "
        f"this test would be back to under-measuring the payload. Fields seen: "
        f"{sorted({k for t in tools for k in t.keys()})}"
    )

    # Compact separators throughout: matches the actual wire's density (no
    # pretty-printing spaces), so re-serializing a sub-structure for
    # measurement doesn't inflate its size relative to the real bytes sent.
    full_line_chars = len(tools_line.strip())
    tools_only_chars = len(json.dumps(tools, separators=(",", ":")))

    approx_tokens_full_line = full_line_chars / 4
    approx_tokens_tools_only = tools_only_chars / 4

    print(
        f"\nReal tools/list response line (full JSON-RPC envelope): {full_line_chars} chars "
        f"≈ {approx_tokens_full_line:.0f} tokens"
    )
    print(
        f"Real 'tools' array only: {tools_only_chars} chars ≈ {approx_tokens_tools_only:.0f} tokens"
    )
    for tool in tools:
        tool_chars = len(json.dumps(tool, separators=(",", ":")))
        print(
            f"  {tool['name']}: {tool_chars} chars ≈ {tool_chars / 4:.0f} tokens "
            f"(fields: {sorted(tool.keys())})"
        )

    # The tools array itself -- not the outer jsonrpc/id/result envelope --
    # is the fairer measure of "the tool definitions" the requirement
    # names; the envelope is fixed per-call overhead, not a cost that
    # grows with what these three tools declare. Gated on this, not the
    # full line, so the assertion tracks the requirement's actual subject.
    assert approx_tokens_tools_only < 1000, (
        f"Tool definitions too large: {approx_tokens_tools_only:.0f} tokens (limit ~1000). "
        f"Per-tool breakdown:\n"
        + "\n".join(
            f"  {tool['name']}: {len(json.dumps(tool, separators=(',', ':'))) / 4:.0f} tokens"
            for tool in tools
        )
    )


def test_mcp_search_no_index_returns_error_not_empty_results():
    """Test that search_sessions with no index returns error, not empty results.

    Defect 1: The MCP server must never return empty results when no index exists.
    It must return an error dict with 'error' key mentioning 'ssgrep index'.
    This test ensures the guard against empty result sets is in place.

    Mutation test: if the branch is changed to return {'results': []}, this test fails.
    """
    import tempfile

    import ssgrep.mcp_server

    # Set up MCP server with no index
    with tempfile.TemporaryDirectory(prefix="mcp-noindex-") as tmpdir:
        import os

        # Change to tmpdir so _get_project_dir() returns it
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            # Import mcp_server to get a fresh server instance for this test
            import importlib

            importlib.reload(ssgrep.mcp_server)
            server_module = ssgrep.mcp_server

            # Call search_sessions directly (not through MCP layer)
            result = server_module.search_sessions("test query")

            # GUARD: Must have error key, never empty results
            assert isinstance(result, dict), f"Expected dict response, got {type(result)}"
            assert "error" in result, f"Expected 'error' key in response, got keys: {result.keys()}"
            assert (
                "ssgrep index" in result["error"]
            ), f"Error message must mention 'ssgrep index', got: {result['error']}"

            # CRITICAL GUARD: Must NOT have empty results
            if "results" in result:
                assert result["results"] is not None, (
                    "DEFECT: results key present with None value (should be absent). "
                    "Empty result list is indistinguishable from 'no matches' for agents."
                )
                assert len(result["results"]) > 0, (
                    "DEFECT: results key contains empty list. "
                    "This violates the never-empty-results requirement. "
                    "An empty list looks like 'your query matched nothing' not 'no index exists'."
                )
        finally:
            os.chdir(original_cwd)
            # Reload to restore state
            import importlib

            importlib.reload(ssgrep.mcp_server)


def test_mcp_show_no_index_returns_error_not_missing_field():
    """Test that show_session with no index returns error, distinguishable from missing episode.

    Defect 1b: show_session must return error dict with 'error' key when no index exists,
    never return None or other form that could be confused with 'episode not found'.
    """
    import tempfile

    import ssgrep.mcp_server

    with tempfile.TemporaryDirectory(prefix="mcp-noindex-show-") as tmpdir:
        import os

        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            import importlib

            importlib.reload(ssgrep.mcp_server)
            server_module = ssgrep.mcp_server

            result = server_module.show_session("session:ep:0")

            assert isinstance(result, dict), f"Expected dict response, got {type(result)}"
            assert (
                "error" in result
            ), f"Expected 'error' key in response when no index, got keys: {result.keys()}"
            error_msg = result["error"]
            assert (
                "ssgrep index" in error_msg or "No index" in error_msg
            ), f"Error message must indicate missing index, got: {error_msg}"
        finally:
            os.chdir(original_cwd)
            import importlib

            importlib.reload(ssgrep.mcp_server)


def test_mcp_index_status_no_index_returns_gracefully():
    """Test that index_status with no index returns gracefully with index_exists=false.

    Defect 1c: index_status must not crash when no index exists. It should return
    a response indicating no index, not raise or return empty data.
    """
    import tempfile

    import ssgrep.mcp_server

    with tempfile.TemporaryDirectory(prefix="mcp-noindex-status-") as tmpdir:
        import os

        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            import importlib

            importlib.reload(ssgrep.mcp_server)
            server_module = ssgrep.mcp_server

            result = server_module.index_status()

            assert isinstance(result, dict), f"Expected dict response, got {type(result)}"
            # Should have either explicit 'message' field or 'index_exists' field set to false
            assert (
                "message" in result or result.get("index_exists") is False
            ), f"Expected graceful response for missing index, got: {result}"
        finally:
            os.chdir(original_cwd)
            import importlib

            importlib.reload(ssgrep.mcp_server)


def test_search_sessions_includes_stale_count_when_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """search_sessions must surface stale_count, not just the stale flag.

    Regression test: unlike the CLI's JSON path (render.render_search_response
    and cli.commands.to_jsonable, both asdict()-based and so pick up every
    SearchResponse field automatically), this dict is hand-built field by
    field, and had "stale" but silently dropped "stale_count" -- leaving an
    agent knowing the index is stale but not by how much.
    """
    from ssgrep import api, mcp_server
    from ssgrep.types import ResultCard, SearchResponse

    card = ResultCard(
        ref="ep-1",
        title="Test",
        timestamp=None,
        score=0.9,
        excerpt="hello",
        files_touched=(),
        is_subagent=False,
    )
    fake_response = SearchResponse(results=[card], stale=True, stale_count=9)

    def fake_search(project_dir, query, limit=10, **kwargs):
        return fake_response

    monkeypatch.setattr(api, "search", fake_search)

    result = mcp_server.search_sessions("test query")

    assert result["stale"] is True
    assert result["stale_count"] == 9


def test_show_session_truncated_folds_in_detail_level_truncation(monkeypatch, tmp_path):
    """show_session must report truncated=True when detail.py already capped the text.

    An episode whose stored prompt/response was truncated at the detail layer
    is truncated for the MCP caller even when show_session's own max_chars
    budget is generous — the flag OR-folds prompt_truncated/response_truncated.
    """
    from ssgrep import mcp_server
    from tests.conftest import build_episode_detail

    monkeypatch.setattr(mcp_server, "_get_project_dir", lambda: tmp_path)

    truncated_detail = build_episode_detail(
        prompt_text="short prompt",
        response_text="short response",
        response_truncated=True,
    )
    monkeypatch.setattr(mcp_server.api, "show", lambda project, ref: truncated_detail)
    result = mcp_server.show_session("s:ep:0", max_chars=10_000_000)
    assert "error" not in result, result
    assert (
        result["truncated"] is True
    ), "detail-level truncation must surface even under a generous max_chars"

    # Positive control: same episode with no detail-level truncation and the
    # same generous budget reports untruncated.
    clean_detail = build_episode_detail(
        prompt_text="short prompt",
        response_text="short response",
    )
    monkeypatch.setattr(mcp_server.api, "show", lambda project, ref: clean_detail)
    result = mcp_server.show_session("s:ep:0", max_chars=10_000_000)
    assert result["truncated"] is False


def test_set_project_dir_pins_scope_regardless_of_cwd(tmp_path):
    """--project-dir's mechanism: an explicit scope set via set_project_dir()
    must win over the process cwd, and clearing it (None) must restore the
    dynamic cwd-resolution behavior.

    Positive assertion first (the pinned path is actually returned), then the
    restore path -- not just absence-of-effect. Module state is restored in a
    finally so no other test inherits a pinned scope.
    """
    import os

    import ssgrep.mcp_server as mcp_server

    pinned = tmp_path / "pinned-project"
    pinned.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    original_cwd = os.getcwd()
    try:
        os.chdir(elsewhere)
        mcp_server.set_project_dir(pinned)
        # Positive: explicit scope wins over cwd
        assert mcp_server._get_project_dir() == pinned
        # cwd changes must NOT leak through while pinned
        os.chdir(original_cwd)
        assert mcp_server._get_project_dir() == pinned
        # None restores dynamic cwd resolution
        mcp_server.set_project_dir(None)
        os.chdir(elsewhere)
        assert mcp_server._get_project_dir() == Path(elsewhere).resolve() or (
            mcp_server._get_project_dir() == Path(elsewhere)
        )
    finally:
        mcp_server.set_project_dir(None)
        os.chdir(original_cwd)


def test_mcp_cli_command_accepts_project_dir_parameter():
    """The CLI surface: McpCommand.handle must accept a project_dir keyword
    (the field-deployment gap: MCP client configs support command/args but
    not always cwd, so scope needs an argument-shaped spelling).

    Signature-level check plus the default value -- the full server startup
    is exercised by the existing MCP handshake tests, which this must not
    duplicate (server.run() blocks on stdio).
    """
    import inspect

    from ssgrep.cli.commands.mcp import McpCommand

    sig = inspect.signature(McpCommand.handle)
    assert "project_dir" in sig.parameters, (
        "McpCommand.handle() must accept project_dir so MCP client configs "
        "can pass --project-dir as an argument instead of needing a cwd"
    )
    assert sig.parameters["project_dir"].default == ".", (
        "project_dir must default to '.' to preserve the original "
        "cwd-resolution behavior for existing registrations"
    )
