# MCP Server Registration

`ssgrep` exposes an MCP server over stdio that allows AI coding assistants to search and browse session transcripts. The server requires an index to exist first — run `ssgrep index` in your project directory to build one.

> **No public registry.** `ssgrep` is sold as a direct download and is not on PyPI, TestPyPI, or any other package registry — a `"command": "uvx", "args": ["ssgrep", ...]` config on this page will always fail with a "not found in the package registry" error, verified directly. Every config below instead uses `"args": ["--from", "/path/to/ssgrep-<version>-py3-none-any.whl", "ssgrep", ...]`, pointing at the wheel file you downloaded after purchase. Verified directly for Claude Code's config (see that section); the same substitution applies to every other client's config on this page. See [../README.md](../README.md#installation) for the equivalent CLI substitution.

**Client coverage.** The Claude Code setup below is verified end-to-end against this
release. The Cursor, Zed, Codex CLI, and opencode configurations follow each client's
documented MCP format but are not part of ssgrep's test matrix — treat them as starting
points. The server itself is identical in every case: one stdio process, three tools.

## Protocol Reference

The server exposes exactly three tools. Every message below is a real, captured exchange.

### `tools/list` Response

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "tools": [
      {
        "name": "search_sessions",
        "description": "Search session transcripts. Returns ranked cards (title, date, score, files, excerpt, ref) plus total_matches and truncation flags.",
        "inputSchema": {
          "additionalProperties": false,
          "properties": {
            "query": {"type": "string"},
            "limit": {"default": 10, "type": "integer"}
          },
          "required": ["query"],
          "type": "object"
        }
      },
      {
        "name": "show_session",
        "description": "Full prompt and response for one episode; ref comes from search_sessions. Content beyond max_chars is truncated.",
        "inputSchema": {
          "additionalProperties": false,
          "properties": {
            "ref": {"type": "string"},
            "max_chars": {"default": 20000, "type": "integer"}
          },
          "required": ["ref"],
          "type": "object"
        }
      },
      {
        "name": "index_status",
        "description": "Index health: existence, session/episode/chunk counts, last-indexed time, model, schema version, staleness.",
        "inputSchema": {
          "additionalProperties": false,
          "properties": {},
          "type": "object"
        }
      }
    ]
  }
}
```

### `initialize` Message

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2024-11-05",
    "capabilities": {},
    "clientInfo": {
      "name": "test-client",
      "version": "1.0"
    }
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2024-11-05",
    "serverInfo": {
      "name": "ssgrep",
      "version": "0.2.0"
    },
    "instructions": "Search and browse AI coding session transcripts. Use search_sessions to find relevant past work, show_session for full context on one episode, index_status to check index health."
  }
}
```

> **`serverInfo.version` is ssgrep's own version** — the same value `ssgrep --version` reports, not the MCP SDK's — so a bug report filed from an MCP client carries the number you need.

### `search_sessions` Round-Trip

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "search_sessions",
    "arguments": {
      "query": "test fixture",
      "limit": 5
    }
  }
}
```

**Response (partial):**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"results\":[...], \"total_matches\":533, \"omitted_count\":0, \"excerpts_truncated\":false, \"clamped\":false, \"index_empty\":false, \"stale\":true}"
      }
    ]
  }
}
```

## Prerequisites

Before configuring the MCP server in any client, you must build an index:

```sh
ssgrep index
```

This creates a `.ssgrep/` directory in your project (mode 0700, .gitignored) containing:
- `index.db` — FTS5-backed SQLite database of chunks
- `vectors.f32` — memory-mapped float32 embeddings
- `.manifest` — index metadata and staleness tracking
- `workqueue.db` — session-end hints awaiting indexing

If the index does not exist, all search operations return an actionable error message:

```json
{
  "error": "No index found at <path>/.ssgrep/index.db. Run `ssgrep index` first."
}
```

## Claude Code

Claude Code supports MCP servers via project-scoped `.mcp.json` files or user-scoped registration.

### Configuration

**Recommended:** Use Claude Code's `mcp` command to register ssgrep at user scope (works from any directory, no manual JSON editing):

```bash
claude mcp add --scope user ssgrep -- uvx --from /path/to/ssgrep-<version>-py3-none-any.whl ssgrep mcp
```

**Alternative:** Create a `.mcp.json` file in your project root with:

```json
{
  "mcpServers": {
    "ssgrep": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--from", "/path/to/ssgrep-<version>-py3-none-any.whl", "ssgrep", "mcp"]
    }
  }
}
```

Verified directly: `uvx --from /path/to/<wheel file> ssgrep mcp` starts the server correctly. The server resolves its project scope from the current working directory at startup — a project directory passed to Claude Code will be the working directory when the server starts.

**If your MCP client can't set a working directory** (many client configs support only
`command`/`args`), pass the project explicitly instead — no shell wrapper needed:

```bash
claude mcp add --scope user ssgrep -- uvx --from /path/to/ssgrep-<version>-py3-none-any.whl ssgrep mcp --project-dir /path/to/your/project
```

`--project-dir` accepts relative, absolute, or `~`-prefixed paths and fixes the server's
scope for its lifetime; without it, the working-directory behavior above is unchanged.

One more operational note, reported from a real deployment: servers registered via a
project's `.mcp.json` file start out as **"⏸ Pending approval"** in Claude Code and need an
interactive session (`claude` in that project, then approve the server) before tools work —
registration alone is not activation.

### Startup Behavior

The server reconciles the index **before** it answers the `initialize` handshake, not lazily
on first tool call. Startup drains any pending session-end hints synchronously, so:

- **Warm index:** a couple of seconds before the handshake completes.
- **Cold start** (no index yet, or a large stale backlog): the full indexing cost is paid
  inside that same window — potentially many seconds on a large corpus.

Once startup finishes there is no further warmup; all three tools are immediately ready.
If your client has a short startup timeout, run `ssgrep index` once beforehand.

---

## Cursor

Cursor uses `~/.cursor/mcp.json` with the same `mcpServers` structure as Claude Code.

**Configuration:**
```json
{
  "mcpServers": {
    "ssgrep": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--from", "/path/to/ssgrep-<version>-py3-none-any.whl", "ssgrep", "mcp"]
    }
  }
}
```


---

## Zed

Zed uses a different key in `~/.config/zed/settings.json`.

**Configuration:**
```json
{
  "context_servers": {
    "ssgrep": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--from", "/path/to/ssgrep-<version>-py3-none-any.whl", "ssgrep", "mcp"]
    }
  }
}
```

Note the key is `context_servers` instead of `mcpServers`.

---

## Codex CLI

Codex CLI uses `~/.codex/config.toml` with a different format.

**Configuration:**
```toml
[mcp_servers.ssgrep]
type = "local"
command = "uvx"
args = ["--from", "/path/to/ssgrep-<version>-py3-none-any.whl", "ssgrep", "mcp"]
```


---

## opencode

opencode uses `opencode.json` in the current project directory.

**Configuration:**
```json
{
  "mcp": [
    {
      "type": "local",
      "name": "ssgrep",
      "command": "uvx",
      "args": ["--from", "/path/to/ssgrep-<version>-py3-none-any.whl", "ssgrep", "mcp"]
    }
  ]
}
```


---

## Error Handling

The MCP server provides actionable error messages for common issues:

**No index:**
```json
{
  "error": "No index found at <path>/.ssgrep/index.db. Run `ssgrep index` first."
}
```

**Empty query:**
```json
{
  "error": "Query must not be empty."
}
```

**Corrupt index:**
```json
{
  "error": "Index at <path>/.ssgrep/index.db is corrupt or unreadable. Run `ssgrep index --rebuild` to recreate it."
}
```

**Index built by a different ssgrep version:**
```json
{
  "error": "Index schema version '2' does not match the expected '3'; run `ssgrep index --rebuild`."
}
```

All error messages are surfaced in the MCP response and displayed to the user — never silently returning empty results.

## About `uvx`

The `uvx` launcher is provided by the `uv` package manager and is an alternative to `npx` or `pip install --break-system-packages`. It normally resolves a package from a registry, but `--from /path/to/a/wheel-or-sdist` (as used throughout this page) runs a local file the same way — either form runs the package without polluting the system environment or requiring a persistent venv.

Users who prefer not to use `uvx` can use `claude mcp add` with their venv binary path:

```bash
claude mcp add --scope user ssgrep -- /path/to/.venv/bin/ssgrep mcp
```

Or create a `.mcp.json` file in your project root:

```json
{
  "mcpServers": {
    "ssgrep": {
      "type": "stdio",
      "command": "/Users/<username>/.venv/bin/ssgrep",
      "args": ["mcp"]
    }
  }
}
```

However, `uvx` is recommended for ad-hoc usage where the user doesn't have `ssgrep` installed locally.
