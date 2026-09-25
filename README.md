<a name="readme-top"></a>

<div align="center">
  <h3 align="center">ssgrep</h3>

  <p align="center">
    Search AI coding-session transcripts without an LLM, a hosted service, or a daemon.
    <br />
    <a href="#table-of-contents"><strong>Explore the Documentation »</strong></a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->

<a name="table-of-contents"></a>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about">About</a></li>
    <li><a href="#quick-start">Quick Start</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#supported-agents">Supported Agents</a></li>
    <li><a href="#mcp">MCP</a></li>
    <li><a href="#privacy">Privacy</a></li>
    <li><a href="#docs">Docs</a></li>
    <li><a href="#credits">Credits</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

<!-- ABOUT -->

<a name="about"></a>

## About

Your coding-agent transcripts contain problems you already solved, but ordinary text search is poor at finding a solution when you remember the idea rather than the exact words. **ssgrep** turns the prompt/response episodes in those transcripts into a local search index.

- **One global index** — Discovers transcripts from every supported coding agent on the machine and reconciles them into a single local LanceDB database
- **Semantic search** — Late-interaction ColBERT embeddings with native MaxSim scoring; searches work on paraphrases, not just exact words
- **Fully local and offline** — After a one-time model download, indexing, search, and the MCP server run without network access
- **Agent-ready** — `ssgrep init` installs an ssgrep skill into each agent harness, and an MCP server exposes read-only search to any MCP client

Requires **macOS or Linux** (Windows is not supported) and **Python 3.11+**.

```bash
ssgrep init
ssgrep search "how did I handle async migration failures"
ssgrep show <ref>
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- QUICK START -->

<a name="quick-start"></a>

## Quick Start

### Install

ssgrep isn't on a package registry yet, so install it from a clone of this repository.

Install globally with `uv` (recommended):

```bash
git clone https://github.com/pshokeen/ssgrep.git
cd ssgrep
uv tool install .
ssgrep --version
```

Or with pip:

```bash
git clone https://github.com/pshokeen/ssgrep.git
cd ssgrep
pip install .
```

Upgrade by pulling the latest commit and reinstalling (`uv tool install --reinstall .`); remove with `uv tool uninstall ssgrep`. To run from the checkout without installing, use `uv run ssgrep ...`.

### Use with Your Coding Agent

One command sets everything up — it registers the `ssgrep mcp` server with every supported client (Claude Code, Cursor, Zed, Codex CLI, opencode, omp), installs an ssgrep skill into each agent harness, and builds the global index:

```bash
ssgrep init
```

Prefer to register only the MCP clients, or skip one?

```bash
ssgrep mcp install            # every supported client
ssgrep mcp install cursor zed # a subset
```

Or register a client manually (per-client snippets in [`docs/mcp-setup.md`](docs/mcp-setup.md)):

```bash
claude mcp add --scope user ssgrep -- ssgrep mcp
```

That's all an MCP user has to do. On startup the server discovers every supported coding agent on the machine and builds the global index itself (the first run downloads the ColBERT embedding model from Hugging Face; after that everything is offline). It reconciles new transcripts on every start, so the index stays current without manual commands. Then just ask your agent to search — e.g. *"search my sessions for how I fixed the flaky migration test"*.

The installed ssgrep skill teaches the agent to search before solving, open hits with `show`, and capture durable lessons with `note` (see [`docs/agent-guidance.md`](docs/agent-guidance.md)).

### Use from the Terminal

The CLI searches the same index the MCP server maintains:

```bash
ssgrep search "how do I handle async errors"

# Inspect one result using the ref printed by search
ssgrep show <ref>

# Inspect index counts, archive state, and runtime census
ssgrep status
```

To keep the index fresh when working only from the terminal, run `ssgrep index` to reconcile new or changed transcripts.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE -->

<a name="usage"></a>

## Usage

Run `ssgrep --help` or `ssgrep <command> --help` for the installed CLI's authoritative option list.

| Command | Purpose |
|---|---|
| `ssgrep init` | One-time setup: install agent skills, then index |
| `ssgrep index` | Build or update the global index |
| `ssgrep search <QUERY>` | Find relevant episodes |
| `ssgrep show <REF>` | Inspect one episode's prompt and response |
| `ssgrep status` | Inspect index counts and database state |
| `ssgrep note` | Add a durable searchable note |
| `ssgrep prune` | Permanently delete archived content |
| `ssgrep mcp` | Start the MCP stdio server |
| `ssgrep rules` | Print the operating rules installed by `init` |

Search is global by default; narrow it with `--where` predicates:

```bash
ssgrep search "retry policy" --where "project = '/absolute/path/to/app'"
ssgrep search "tool failure" --where "is_subagent = true AND content_type = 'response'"
```

Data-oriented commands accept a global `--json` flag. A transcript that is no longer discovered stays searchable but is marked `source_status = 'absent'`; restrict to live sources with `--where "source_status = 'available'"` and clean up archived content with `ssgrep prune`.

After the first `ssgrep init`, run `ssgrep index` any time to reconcile new or changed transcripts into the global index.

See [`docs/usage.md`](docs/usage.md) for the full command reference, `--where` predicate fields, exit codes, JSON envelopes, and archive semantics.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- SUPPORTED AGENTS -->

<a name="supported-agents"></a>

## Supported Agents

ssgrep ingests sessions from every coding agent it can find on the machine, through one adapter per runtime. Each adapter reads only the runtime's own on-disk transcript data and normalizes it into one shared episode schema.

| Runtime | Transcript source | Default location |
|---|---|---|
| **Claude Code** | native record-pair JSONL | `~/.claude/projects` |
| **OpenCode** | local SQLite store | `~/.local/share/opencode/opencode.db` |
| **Codex** | rollout session JSONL | `~/.codex/sessions` |
| **Pi** | session JSONL | `~/.pi/agent/sessions` |
| **Prime Agent** | session JSONL + `session-artifacts/` | `~/.prime/agent/sessions` |

`ssgrep status` reports the runtime census (`Runtimes: claude=12, opencode=3, ...`), and every search can be narrowed with `--where "runtime = 'pi'"`.

`ssgrep init` installs an idempotent ssgrep skill into each runtime's own global skills directory. The installed rules tell agents to search before solving, read hits with `show`, and capture durable lessons with `note` — see [`docs/agent-guidance.md`](docs/agent-guidance.md).

See [`docs/runtimes.md`](docs/runtimes.md) for per-runtime ingestion details, environment overrides, and model/device configuration.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MCP -->

<a name="mcp"></a>

## MCP

`ssgrep mcp` starts a read-only MCP stdio server (`search_sessions`, `show_session`, `index_status`) over the same global database the CLI uses. It builds and refreshes the index automatically on startup, for every client — see [Quick Start](#quick-start) for registration.

`ssgrep mcp install` registers every supported client (Claude Code, Cursor, Zed, Codex CLI, opencode) in one step; manual snippets and full tool details are in [`docs/mcp-setup.md`](docs/mcp-setup.md).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- PRIVACY -->

<a name="privacy"></a>

## Privacy

1. **Transcripts stay local.** Indexing and search run on your machine; transcript content is not sent to an LLM or hosted retrieval service.
2. **Runtime is offline after the model is cached.** The first model download uses Hugging Face; warm loads are cache-first and update checks are disabled.
3. **Transcript history is read-only.** ssgrep never writes to any agent's transcript files; `note` writes only to ssgrep's own application-data directory.
4. **No network listener or daemon.** CLI commands are ordinary local processes; MCP uses the client's stdio transport.

The index contains copies of transcript text in the application-data root (created with mode `0700`). Protect and back up that directory accordingly.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- DOCS -->

<a name="docs"></a>

## Docs

- [`docs/usage.md`](docs/usage.md) — full command reference, predicate fields, exit codes, JSON output, archive semantics
- [`docs/runtimes.md`](docs/runtimes.md) — per-runtime ingestion, environment overrides, model and device configuration
- [`docs/retrieval.md`](docs/retrieval.md) — chunking, embedding, scoring pipeline, measured benchmark results
- [`docs/agent-guidance.md`](docs/agent-guidance.md) — the ssgrep operating rules installed into agent harnesses
- [`docs/mcp-setup.md`](docs/mcp-setup.md) — MCP client configuration and tool details
- [`docs/architecture.md`](docs/architecture.md) — discovery, reconciliation, schemas, and service-layer details

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CREDITS -->

<a name="credits"></a>

## 🚀 Credits

- Multi-vector search was significantly improved by [@thememium](https://github.com/thememium).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->

<a name="license"></a>

## License

MIT License. See [LICENSE](LICENSE).
