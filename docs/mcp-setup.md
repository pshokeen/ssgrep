# MCP Server Setup

`ssgrep mcp` starts a Model Context Protocol server over stdio. It gives MCP-capable coding assistants read-only search and detail access to the same global LanceDB database used by the CLI.

There is one database for every project and every client registration. The MCP process's working directory does not select a project. Use the `where` argument on an individual search when project-specific results are required.

> **Agent guidance:** the MCP tools are the *transport*; the rules for using
> ssgrep well ship separately as a skill installed by `ssgrep init` and printable
> with `ssgrep rules`. See [`docs/agent-guidance.md`](agent-guidance.md). There is
> intentionally no MCP tool for the rules, and none that writes.

## One-Command Setup

`ssgrep mcp install` writes the ssgrep registration into every supported client's configuration automatically (`ssgrep init` performs the same registration as part of full first-run setup):

```bash
ssgrep mcp install            # every supported client
ssgrep mcp install cursor zed # a subset: claude, cursor, zed, codex, opencode, omp
```

The command is idempotent: an exact existing entry is left untouched, a stale entry owned by ssgrep is replaced, and unrelated settings are preserved. Malformed files are reported with an `error: ...` status instead of being modified. Each client reports `installed`, `updated`, `already_installed`, `skipped: ...` (for example, when the `claude` CLI is not on `PATH`), or `error: ...`; with `--json`, each client returns `{status, config}` naming the file that was written or checked.

The sections below show the exact registrations the command writes, for manual setup or review.


## Prepare the Index

No manual step is required for MCP clients: server startup reconciles every discovered runtime in a background thread, including the very first index build. A pre-existing index is not a protocol requirement.

Running `ssgrep index` manually is still useful because it exposes indexing errors directly instead of leaving them in the MCP server's stderr, and it lets a terminal user warm the model cache before a client connects.

The database is created beneath `platformdirs.user_data_path("ssgrep", appauthor=False)/lancedb` by default. `SSGREP_DATA_DIR` overrides the application-data root.

## Startup Behavior

When the server is constructed, it:

1. Disables FastMCP's banner and dependency update check.
2. Registers the three tools described below.
3. Runs the global index service once in a background thread.

Step 3 is **best-effort and asynchronous**: reconciliation happens in a daemon thread, so it never blocks the stdio handshake or adds process startup time, and a failure does not prevent the MCP server from starting. The tools use any usable database left on disk. If no usable index exists yet (for example, during a very first start before the background reconciliation finishes), search and detail calls return an actionable error; `index_status` still returns `index_exists: false` and a message.

The background reconciliation is also what keeps the index current: it discovers every supported coding agent and ingests new or changed transcripts on each server start, so an MCP-only user never runs `ssgrep index` manually. If startup reconciliation failed and you need the full CLI diagnostic, run `ssgrep index` from a terminal.

## Tools

The server exposes exactly three tools.

### `search_sessions`

Searches every indexed project unless `where` narrows it.

Input:

```json
{
  "query": "how did I handle retry backoff",
  "limit": 10,
  "where": "project = '/absolute/path/to/app'"
}
```

Schema:

- `query` (string, required)
- `limit` (integer, default `10`; the search service caps results at `50`)
- `where` (string or null, default `null`) — a raw Lance SQL predicate applied before hybrid ranking

The result object contains:

- `results` — ranked episode cards
- `total_matches` — distinct episodes represented in the bounded retrieval candidate set
- `omitted_count` — cards removed by response budgeting
- `excerpts_truncated` — whether an excerpt was shortened
- `clamped` — whether the requested limit exceeded the service maximum
- `index_empty` — whether a usable database exists but has no searchable chunks

A result card includes its episode ref, title, timestamp, backend score, excerpt, file and agent metadata, project and source paths, content type, and `source_absent`. The score is a ranking value, not a normalized probability.

Useful predicates:

```text
project = '/Users/me/code/app'
source_status = 'available' AND content_type = 'response'
is_subagent = true AND agent_model = 'claude-opus-4-6'
timestamp >= timestamp '2026-01-01T00:00:00'
```

Predicate fields are scalar columns from the Lance `chunks` table:

```text
chunk_id, episode_id, session_id, runtime,
project, source_path, source_project, source_status,
content_type, title, timestamp, git_branch, cwd,
files_touched, tool_names,
is_subagent, parent_session_id,
agent_type, agent_name, agent_description, agent_model,
claude_version, entrypoint, permission_mode, user_type
```

When embedding a predicate in JSON, remember to escape it according to JSON string rules.

### `show_session`

Returns the stored detail for one episode. The ref comes from `search_sessions`.

Input:

```json
{
  "ref": "session-id:ep:12",
  "max_chars": 20000
}
```

Schema:

- `ref` (string, required)
- `max_chars` (integer, default `20000`) — combined prompt/response character budget for the MCP result

The underlying detail layer first bounds prompts at 50,000 characters and responses at 150,000 characters. `show_session` then applies `max_chars` across the two fields and returns `truncated: true` if either layer shortened content. The result also includes `prompt_truncated` and `response_truncated` from the detail layer.

### `index_status`

Takes no arguments and returns the global `IndexStats` fields:

```text
session_count, episode_count, chunk_count,
index_size_bytes, last_index_time, last_optimize_time,
model_id, vector_dimension, schema_version,
skipped_records, malformed_records,
tombstoned_source_count, tombstoned_chunk_count,
index_exists, data_dir, runtime_counts
```

The total session, episode, and chunk counts include archived rows; the tombstone fields identify the archived subset. When the index is missing, all corpus counts are zero, `index_exists` is false, and the response adds:

```json
{"message": "No index found. Run `ssgrep index` to build one."}
```

`index_status` reads database metadata. It does not scan the transcript corpus; use `ssgrep index` to reconcile source changes.

## Claude Code

Claude Code supports user-scoped registration and project `.mcp.json` files.

### User-scoped registration

```bash
claude mcp add --scope user ssgrep -- ssgrep mcp
```

### Project configuration

Create `.mcp.json` in the project root:

```json
{
  "mcpServers": {
    "ssgrep": {
      "type": "stdio",
      "command": "ssgrep",
      "args": ["mcp"]
    }
  }
}
```

Although the registration file is project-scoped, the server still searches the global database. Ask the assistant to supply a `where` predicate when you want that project only.

Claude Code may require approval before enabling a server declared by a project's `.mcp.json`. Open an interactive session in that project and approve the registration if it remains pending.

## Cursor

Cursor uses `~/.cursor/mcp.json` with an `mcpServers` object:

```json
{
  "mcpServers": {
    "ssgrep": {
      "command": "ssgrep",
      "args": ["mcp"]
    }
  }
}
```

## Zed

Zed uses the `context_servers` key in `~/.config/zed/settings.json`:

```json
{
  "context_servers": {
    "ssgrep": {
      "command": "ssgrep",
      "args": ["mcp"]
    }
  }
}
```

## Codex CLI

Codex CLI uses `~/.codex/config.toml`:

```toml
[mcp_servers.ssgrep]
command = "ssgrep"
args = ["mcp"]
```

## opencode

opencode's global configuration (`~/.config/opencode/opencode.json`, honoring `XDG_CONFIG_HOME` and `OPENCODE_CONFIG_DIR`) registers local servers under `mcp.servers` with an argument-list command:

```json
{
  "mcp": {
    "servers": {
      "ssgrep": {
        "type": "local",
        "command": ["ssgrep", "mcp"]
      }
    }
  }
}
```

## omp

omp's user-scope MCP config (`~/.omp/agent/mcp.json`, honoring `OMP_AGENT_DIR`) lists servers under `mcpServers`:

```json
{
  "mcpServers": {
    "ssgrep": {
      "type": "stdio",
      "command": "ssgrep",
      "args": ["mcp"]
    }
  }
}
```

Client configuration formats can change independently of ssgrep. If a client rejects one of these starting points, compare it with that client's current stdio-MCP documentation; the ssgrep command and arguments remain the same.

## Use an Installed Binary Instead

If `ssgrep` is already installed globally and available on the MCP client's `PATH`:

```bash
claude mcp add --scope user ssgrep -- ssgrep mcp
```

For a virtual environment, use the absolute binary path:

```bash
claude mcp add --scope user ssgrep -- /path/to/.venv/bin/ssgrep mcp
```

Equivalent JSON:

```json
{
  "mcpServers": {
    "ssgrep": {
      "type": "stdio",
      "command": "/path/to/.venv/bin/ssgrep",
      "args": ["mcp"]
    }
  }
}
```

Absolute paths are more reliable because GUI applications often inherit a smaller `PATH` than an interactive shell.

## Error Handling

Tool failures are returned as objects with an `error` field rather than as empty successful results.

Missing index:

```json
{"error": "No global index found. Run `ssgrep index` first."}
```

Empty query:

```json
{"error": "Query must not be empty."}
```

Unknown episode ref:

```json
{
  "error": "Episode not found: bad-ref. Run search_sessions to find valid refs."
}
```

An incompatible schema instructs the user to run `ssgrep index --rebuild`. Unexpected tool failures include the tool name in the returned error and are also written to stderr, never to the MCP stdout transport.

## Offline and Privacy Behavior

After the pinned embedding model is cached, the MCP server does not need network access. FastMCP's update check is disabled before the server is constructed. Transcript content, queries, embeddings, and results remain local.

The MCP tools cannot write notes or mutate Claude transcript files. The startup reconciliation can update the global ssgrep database, and the database itself contains copies of indexed transcript text, so protect the application-data directory accordingly.
