# Changelog

## [2.0.1] - 2026-09-26

### Added

- **Published to PyPI.** `uvx ssgrep`, `uv tool install ssgrep`, and
  `pip install ssgrep` now work; releases publish automatically from `v*` tags
  via PyPI trusted publishing (no stored token).

- **omp runtime.** ssgrep now discovers and indexes omp sessions
  (`~/.omp/agent/sessions`, Pi-format JSONL); `ssgrep init` installs the
  skill to `~/.omp/agent/skills/ssgrep`. See [docs/runtimes.md](docs/runtimes.md).
- **omp as an MCP client.** `ssgrep mcp install` and `ssgrep init` register
  the ssgrep MCP server in omp's `~/.omp/agent/mcp.json` (honoring
  `OMP_AGENT_DIR`). See [docs/mcp-setup.md](docs/mcp-setup.md).

### Changed

- **`ssgrep mcp install` prefers `uvx`.** When `uvx` is on `PATH`, client
  registrations are written as `uvx ssgrep mcp` (portable across machines and
  virtual-environment moves); otherwise the absolute binary path is written as
  before. Re-running upgrades an existing absolute-path entry.
- CI caps the eval harness's worker pool in the emulated Linux leg
  (`SSGREP_EVAL_WORKERS`), fixing a recurring out-of-memory flake.

## [2.0.0] - 2026-09-24

ssgrep is open source. Headline changes since 0.2.0:

### Changed

- **BREAKING: licence changed from PolyForm Internal Use License 1.0.0 to
  the MIT License** (see [LICENSE](LICENSE)). Not retroactive:
  releases before 2.0.0 were obtained under PolyForm terms and remain governed
  by them; only 2.0.0 and later carry MIT terms, including permission
  to redistribute. Version bumped to 2.0.0 to mark this.
- **Search rewritten as multivector late interaction.** The six-leg
  SQLite/FTS5 + model2vec pipeline was replaced by single-stage MaxSim over
  per-token ColBERT vectors in LanceDB (IVF-PQ), fused with a BM25 lexical
  signal via Reciprocal Rank Fusion when the index carries lexical stats. See
  [docs/architecture.md](docs/architecture.md) and
  [docs/retrieval.md](docs/retrieval.md).
- **`ssgrep mcp install`** registers the ssgrep MCP server with Claude Code,
  Cursor, Zed, Codex, and OpenCode; `ssgrep init` now registers MCP clients
  and reports their status. See [docs/mcp-setup.md](docs/mcp-setup.md).
- **Operating rules ship with the product.** The skill `ssgrep init` installs
  is ssgrep's guidance document, installed into all supported runtimes from
  one shared body; **`ssgrep rules`** prints it (full, `--short`, or `--json`)
  without reading an index or loading a model.
- MCP startup reconciliation runs in a background thread so it no longer
  blocks the client handshake.

### Removed

- **Checked-in eval snapshots.** `eval/results/` carried 341 MB of
  near-duplicate benchmark payloads. Benchmark output is no longer committed at all — `.gitignore`
  excludes `eval/results/`; the figures in
  [docs/retrieval.md](docs/retrieval.md) come from a local run whose payload
  is not in the repository.
- Project, organisation, and personal identifiers scrubbed from shipped
  artifacts.

## [0.2.0] - 2026-08-10

Initial snapshot: offline search over AI coding-session transcripts —
retrieval reranking with an n=66 benchmark, `--scope`, `note`, external
transcript roots (`SSGREP_TRANSCRIPT_DIRS`), and CI on a self-hosted runner.
Development history before 2.0.0 is retained privately.
