# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2026-08-10

> Squashed release: everything from `v0.1.0` to this point lands as one commit
> representing the 0.2.0 feature set (retrieval reranking + n=66 benchmark,
> `--scope`, `note`, external transcript roots, CI on a local self-hosted
> runner). `ssgrep --version` reports `0.2.0`.

### Added

- `SSGREP_TRANSCRIPT_DIRS`: opt-in external transcript roots (colon-separated,
  optional `format=` adapter tags — `native` today, the seam for future runtime
  adapters). External roots ride the real discovery→staleness→index pipeline,
  bypass `--scope` (explicit configuration is always in scope), warn-and-skip
  when unmounted, and are shrink-guarded on removal. Unknown format tags fail
  loudly. No env var → zero behavior change.
- **`ssgrep index --scope <path>`.** Decouples WHICH transcripts are discovered
  (cwd containment against `--scope`) from WHERE the index lives
  (`--project-dir`). The moved-repo / org-rename remedy: history recorded under
  an old path can be indexed into the current project's index and searched from
  here. The scope is persisted in the index and is STICKY: a plain
  `ssgrep index` (or MCP startup reconciliation, or `ssgrep init`) keeps the
  persisted scope — only an explicit, different `--scope` changes it, and
  that change forces a full rebuild, like a model change. Search-time
  staleness compares against the persisted scope (so a scope-built index
  never mis-reads as vanished), the SessionEnd hook enqueues at the
  persisted scope, and the work-queue drain drops hints it cannot attribute
  to the scope (fail-closed, via the same cwd matcher discovery uses) — so
  no path, routine or stale, can silently mix another scope's sessions into
  a scoped index. When the persisted scope differs from the project
  directory, every index run prints a notice that locally-recorded sessions
  are not included. The zero/suspicious-discovery remedies now suggest
  `--scope` instead of the old build-a-second-index workaround.
- **`ssgrep note --title <question> --body <text|->`.** Writes a durable,
  searchable note as a native transcript record pair under
  `<project>/.ssgrep/notes/` — never under `~/.claude` — and reindexes
  quietly so it is immediately searchable. Phrase the title as the question
  you will later search for. Notes ride the exact same
  records → episodes → chunking → embedding pipeline as real transcripts,
  are always in scope for their own index, and follow normal staleness and
  tombstone semantics. CLI-only by design: no MCP write tool (a write
  capability reachable by a prompt-injected agent is a different trust class).
- **Retrieval: MaxSim late-interaction rerank leg (sixth leg).** The top 250
  chunks from the five-leg fused ranking are reranked by ColBERT-style MaxSim
  over per-token static embeddings from the already-loaded model2vec model
  (`search/rerank.py`, `embed.encode_sequence()`) and fused back in at weight
  1.0. No new dependency; measured leg cost ~33 ms/query inside the existing
  budget (p50 58 ms post-model-load for the full leg sequence).
- **`ssgrep mcp --project-dir <path>`.** Fixes the server's project scope
  explicitly, for MCP client configs that support `command`/`args` but not a
  working directory. Default behavior (scope from the server's cwd) unchanged.
- **Suspiciously-small-discovery warning.** `ssgrep index` reporting exactly
  1 session now checks the rejection census for a rejected `cwd` whose final
  path component matches the project's (the moved-repo / org-rename
  signature) and, only on that positive evidence, warns on stderr with the
  sibling path, its transcript count, and the exact `--project-dir` command
  to index it. `--json` callers get a `suspicious_discovery` payload;
  the zero-discovery payload gains a `same_basename_rejected` field.
- **Score-interpretability guidance.** README documents that the score is a
  weighted-RRF sum, not 0–1 relevance (max observed top score ≈0.10), with a
  measured noise-floor calibration recipe
  (`eval/results/score_calibration_2026-08-07.json`) and a rank-interleaving
  rule for multi-index merges.
- **`uv tool install` documented as the recommended install method** (wheel
  and checkout forms, both verified), with a PATH-shadowing caution.

### Changed
- **CI now executes on a local self-hosted runner** (`m5-local-runner`).
  GitHub-hosted runners were billing-blocked on this account, so CI had never
  actually run. A self-hosted macOS runner (concurrency 1) plus a Linux
  ubuntu-smoke leg (run inside a Docker container) make the full pipeline —
  lint, format, typecheck, sizecheck, 847 tests, and CLI-identity verification —
  run and pass on every push, with GitHub Actions pinned to latest versions
  (checkout v7, setup-uv v9, setup-python v7, upload-artifact v7) by commit SHA.


- **Retrieval: chunk window retuned for the six-leg pipeline.**
  `CHUNK_TARGET_SIZE` 1200 → 1750, `CHUNK_OVERLAP` 200 → 450 (≈21% fewer
  chunks, smaller index), and `ROLLUP_TAIL_WEIGHT` 0.2 → 0.05 (fixes
  episode-level near-misses where tail-vote accumulators outranked episodes
  holding a top-12 fused chunk; validated on the live corpus and a
  never-before-evaluated blind holdout before shipping). Honest effect
  accounting (full detail in `eval/README.md`, "Update 2026-08-07"): against
  the pre-session config these changes measure +16.8% MRR on the 48 tuning
  queries and +15.2% on the same queries over the live corpus, but ≈0% MRR on
  18 never-tuned queries — while recall@10 was equal-or-better in every
  comparison on every query set, which is the claim that generalizes.
  Published headline is now recall@10 **87.9%** live / **89.4%** frozen on the
  n=66 set, MRR ≈0.63.
- **Eval label set extended 48 → 66** by merging the 18 holdout queries that
  had never been used in any tuning decision (6 from the 2026-08-06 label
  repair, 12 blind-composed after all 2026-08-07 tuning froze). Guarded
  artifact: `eval/results/baseline_n66_2026-08-07b.json`. A fresh 8-query
  blind holdout (`.auto/holdout_v4_queries.jsonl`, never evaluated) is
  reserved for the next tuning session.

- **Retrieval: three new fusion legs.** Search now fuses five legs by weighted RRF:
  the existing AND-BM25 and vector legs (weight 1.0) plus an OR-BM25 leg (0.9), a
  trigram-subword leg (1.0, new `chunks_fts_tri` FTS5 table, SCHEMA_VERSION 3 → 4,
  one-time index rebuild on first use), and a phrase-proximity leg (0.5). Episode
  roll-up adds a bounded non-best-chunk tail (`max + 0.2 × next two chunks`).
  Measured on the regenerated-and-extended 48-query labelled set (2026-08-06 corpus):
  overall recall@10 from a measured 44.4% (silently decayed from the published 77.8%
  after Claude Code's transcript cleanup deleted query targets) to **87.5%**, MRR
  ≈0.54 → **≈0.59**; exact-identifier, error-string, and subagent-only classes all
  at 100%.
- **Eval label set regenerated.** Claude Code's transcript cleanup deleted the target
  episodes of 13 of the 36 committed queries (measured recall had silently decayed to
  44.4%). `eval/build_labels.py` replaced the 9 dead anchor specs with new
  independently-sourced anchors per the documented methodology and `eval/queries.jsonl`
  was regenerated, then extended from 36 to 48 queries (12 per class) with twelve
  new queries composed blind after the five-leg ranking shipped, as a
  generalization check (10/12 hit). `README.md`'s table now cites the n=48
  artifact `eval/results/baseline_boost_0.0_2026-08-06d.json` (overall recall@10
  87.5%, MRR ≈0.59).

## [0.1.0] - 2026-08-06

### Upgrade notes

- **One-time full index rebuild.** SCHEMA_VERSION moved to 3 (episodes now
  persist canonical `prompt_text`/`response_text`). The first command after
  upgrading detects the version change and rebuilds the index from your
  transcripts through the shrink-guarded path.
- **Tombstoned history cannot be reproduced by a rebuild.** Content whose
  source transcript has been deleted exists only inside `.ssgrep/`. If
  `ssgrep status` shows non-zero tombstone counts and you care about that
  history, back up `.ssgrep/` before upgrading.

### Added

- **Rebuild shrink guard.** A rebuild that would leave fewer than half the indexed
  sessions *or* fewer than half the indexed chunks is now refused before the manifest
  is touched, naming old-vs-new counts. Previously the rebuilt generation was
  committed unconditionally, so a scope mismatch (ssgrep run from the wrong
  directory, or a moved/renamed project) swapped an empty index over a healthy one,
  printed "Indexed 0 sessions, 0 episodes, 0 chunks." and exited 0. Not
  `--rebuild`-only: the first `ssgrep index` after any release that changes the
  schema version or embedding model takes the same path with no flag.
- **`--allow-shrink`** on `ssgrep index`, the explicit opt-in past that guard.
- **Zero-discovery diagnostic.** "Indexed 0 sessions" now explains itself: the scope
  matched against (after canonicalization), the transcript root actually scanned
  (after `CLAUDE_CONFIG_DIR` resolution), how many transcripts exist there, how many
  scope rejected, and the working directories those transcripts record — plus the
  exact command that indexes the path they were recorded under. Also emitted to
  `--json` callers and to the MCP server's `search_sessions` (as `zero_discovery`),
  and not suppressed by `--quiet`.

### Fixed

- **`init --json` always emits a structured error document.** Indexing failures
  other than the already-structured cases (e.g. `IndexNotFoundError` on a
  machine with no transcript root) previously escaped to the CLI framework,
  producing a generic envelope. All indexing-failure paths now write one
  `{ok, condition, message, command}` document to stdout with the same exit
  code as before; text-mode behavior is unchanged. (Also the reason the CLI
  framework was upgraded: usecli 0.1.76 → 0.1.78, whose 0.1.77 release
  narrowed its JSON-mode catch-all — ssgrep no longer relies on the
  framework envelope on any path. No other locked package moved.)
- **`CLAUDE_CONFIG_DIR` is honoured everywhere.** Transcript discovery, work-queue
  session classification, and SessionEnd hook installation all resolve Claude Code's
  config root through one resolver. Hook installation in particular used to write to
  `~/.claude/settings.json` and report success while Claude Code read a different
  file, so the hook never fired and produced no error.
- **One path canonicalization for both identities.** The scope typed on the command
  line and the `cwd` recorded in a transcript are now reduced the same way
  (expanduser → normpath → case-fold iff the filesystem folds case, probed at
  runtime rather than assumed from the platform). Previously `--project-dir
  ~/code/app` and `~/Code/app` shared one physical `.ssgrep` but matched different
  transcripts, and a quoted `--project-dir "~/code/x"` resolved to `<cwd>/~/code/x`
  and matched nothing.
- Staging a rebuild no longer trusts a cached generation number, and both
  `discard_generation()` and `commit_generation()` now take the same advisory lock,
  so a run that stages while another run commits can no longer unlink the live index.
- Deleted (tombstoned) transcripts and `--no-subagents` no longer count against the
  shrink guard, which used to hard-block rebuilds for buyers who had done nothing
  wrong.
- Suggested commands in the zero-discovery diagnostic are shell-quoted, are never
  built from the "(no cwd recorded)" placeholder, and are only offered when one
  recorded directory actually dominates.
- `ssgrep init` now reports a refused rebuild the same way `ssgrep index` does
  (exit 2, same JSON envelope) instead of failing with a remedy that hits the same
  guard.

## [0.0.1] - 2026-07-27

### Added

Initial MVP release of ssgrep: a local CLI and MCP server for searching AI coding-session transcripts without an LLM, server, or network dependency.

**What it does:**
- Indexes Claude Code session transcripts from `~/.claude/projects/` using hybrid BM25 + semantic search
- Answers "how did I solve this before?" with 77.8% recall@10 across diverse queries
- Runs offline after a one-time embedding-model download, with no server, daemon, or infrastructure setup
- Supports three surfaces: CLI (`ssgrep search`), MCP server for Claude Code integration, and a skill wrapper

**How it works:**

- **Contracts** — Frozen dataclasses (`SessionFile`, `Episode`, `Chunk`, `ContentType`, `ResultCard`, `EpisodeDetail`, `IndexStats`) and API signatures that every module codes against, with sanitized test fixtures built from the real corpus covering all 15 record types

- **Discovery and parsing** — Path-encoded lookup of session transcripts, subagent classification, streaming JSONL record parsing, episode segmentation (treating `compact_boundary` as a hard boundary), signal/noise classification (~4% of transcript bytes is indexable prose), and metadata harvesting

- **Indexing** — Chunking at ~1200 characters with 200-character overlap, embedding with `model2vec/potion-base-8M` (256 dims), a SQLite schema with FTS5, and memory-mapped vector storage. Incremental, driven by append-offset cursors and rewrite guards, and crash-atomic via staged generations and a manifest swap

- **Search** — Hybrid BM25 + cosine similarity fused by RRF, with shared result rendering

- **Surfaces** — CLI commands (`index`, `search`, `show`, `status`, `mcp`), an MCP server exposing three tools, and a Claude Code skill wrapper, plus first-run and error UX; `init` handles one-time setup and hook installation

- **Quality gates** — An evaluation harness measuring retrieval quality across 36 labelled queries (recall@10 77.8%, MRR ≈0.54, ±0.012 cross-run drift on this live corpus); performance regression tests asserting a loose, deliberate ≤1500ms end-to-end search budget (measured median 523ms as a subprocess, corrected from an earlier in-process-only 195ms figure — see `tests/test_perf.py`); incremental index caching that brings session discovery under 100ms; and format-drift and mutation-resilience tests

- **Tooling and packaging** — `uv`-managed, with pytest, ruff, mypy, and a CI matrix covering Python 3.11–3.13 on macOS and Linux. MCP registration is verified on Claude Code (Cursor config is documented but unverified — Cursor is not installed on the reference machine). Ships with a README quickstart and performance table, an architecture document, and tag-triggered build automation; see "Release Automation" below

### Performance

- **Search latency**: ~500ms end-to-end as a subprocess (median 523ms, min 481ms, max 557ms, n=12). Breakdown: CLI boot ~136ms (`ssgrep status`, project + index resolved), model load ~324ms (cold process), discovery ~44ms (cached), query ~30ms
- **Indexing**: No-change re-index <1s (~82ms measured, 2026-07-28); full index of 242 sessions (7,561 chunks) ~2.4s (an earlier ~15s figure was measured against a differently-scoped corpus; see `tests/test_perf.py`)
- **Storage**: SQLite + FTS5 + memory-mapped float32 matrix; ~1 KB per chunk in the vector matrix alone (256 floats × 4 bytes, measured exact), ~4 KB per chunk total including the SQLite/FTS5 text index; hybrid retrieval uses BM25 + brute-force cosine fused by RRF
- **Embedding throughput**: 23,951 docs/sec on real chunk text (mean 603 characters — the shape indexing actually embeds); short, query-length strings measure far higher (~132k docs/sec, the figure originally published here) but that isn't the shape a real index build embeds. Indexing stays parse-bound, not embed-bound, at either rate

### Retrieval Quality

Measured over 36 labelled queries derived from the real corpus:

- **Overall**: recall@10 77.8%, MRR ≈0.54 (±0.012 cross-run drift on this live corpus)
- **By query class**:
  - Exact identifier: 100%
  - Error string: 88.9%
  - Multi-hop: 66.7%
  - Subagent-only: 69.2%
  - Paraphrase: 55.6% (known weakness)

### Known Limitations

1. **Paraphrase ranking**: static embeddings (`potion-base-8M`, MTEB Retrieval 31.11 vs
   all-MiniLM's 42.92) rank vague, natural-language queries worst — paraphrase recall@10 is
   55.6%. This is the trade that keeps search offline and sub-millisecond. The larger
   `potion-retrieval-32M` reliably wins paraphrase but reliably *loses* multi-hop (36/36 runs)
   and usually loses subagent-only, while regressing exact-identifier MRR and forcing a
   breaking 512-dimension re-index. Deferred pending a larger labelled query set — it is a
   measured trade between query classes, not a margin too small to bother with. Full
   evidence and reproduction: [`eval/README.md`](eval/README.md).
2. **Latency**: Subprocess search is ~500ms end-to-end, dominated by Python cold-process startup (~136ms) and embedding model load (~324ms). An MCP server trades startup cost for latency once warm. A hand-rolled static encoder could recover ~280ms but adds code to maintain.

3. **Scope**: MVP is folder-scoped (single `~/.claude/projects/` directory). Cross-project search is deferred to V2.

4. **Harnesses**: Claude Code is the only integrated harness. Cursor MCP registration is untested; Codex and OpenCode support (requiring a source adapter interface to read from SQLite) are deferred to V2.

5. **Subagent transcripts**: Indexed by default (they hold ~72% of all indexed chunks). Attribution and rank preference mitigate the risk that exploratory work surfaces as authoritative, but this is a ranking problem, not a reason to exclude them. The evaluation harness tuned the main-session boost based on data.

### Privacy

- `.ssgrep/` is created with mode 0700 and added to `.gitignore` on first index
- No network calls after one-time model download
- Transcripts are never written to; only read
- All claims above are asserted in the test suite

### Installation

> **Not yet published.** `ssgrep` is not on PyPI or TestPyPI yet (see "Release Automation" below); `uvx ssgrep` and `pip install ssgrep` fail today with a "not found in the package registry" error — verified directly. Until it ships, run from a local checkout instead:

```bash
# From a local checkout (works today)
cd /path/to/ssgrep && uv sync
uv run ssgrep init
uv run ssgrep search "your query here"

# Or from anywhere, without a persistent checkout:
uvx --from /path/to/ssgrep ssgrep search "your query here"
```

Once published, the intended quick start is `uvx ssgrep search "your query here"`, or `pip install ssgrep` followed by `ssgrep init` and `ssgrep search "your query"` to skip `uvx`'s one-time dependency-resolution cost (model load still applies either way — see README's Performance section).

### MCP Integration

`ssgrep mcp` starts the stdio MCP server; Claude Code launches it automatically once registered, so you don't normally run this command by hand. The server exposes three tools:
- `search_sessions`: Search transcripts by query
- `show_session`: Retrieve full context for a result
- `index_status`: Check index health and staleness

Register the server with Claude Code using:
```bash
claude mcp add --scope user ssgrep -- uvx --from /path/to/ssgrep ssgrep mcp
```

Or add a `.mcp.json` file in your project root:
```json
{
  "mcpServers": {
    "ssgrep": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--from", "/path/to/ssgrep", "ssgrep", "mcp"]
    }
  }
}
```

See [docs/mcp-setup.md](docs/mcp-setup.md) for additional configuration options and editor support.

### Release Automation (Not Yet Exercised)

`.github/workflows/release.yml` exists and is tag-triggered (`v*`): it builds and tests the package, verifies CLI identity in a clean venv, and uploads the wheel and sdist as a workflow artifact. **There is no publish job** — ssgrep is sold as a paid direct download, with no public package registry, so the workflow has no credentials for and no step calling any index. The `"Private :: Do Not Upload"` classifier in `pyproject.toml` is a second, independent interlock: it makes PyPI reject an upload server-side even if someone runs `uv publish` by hand.

The workflow has never been run end-to-end, and a workflow that has never run is a plan, not verified automation. What remains is decisions and accounts that live outside this codebase — a lawyer's review of the license before taking money, and a storefront account with product-listing rights — so no amount of local testing settles it. See [docs/release-checklist.md](docs/release-checklist.md) for the exact remaining steps.

### Thanks

Built on `model2vec` (static embeddings), `fastmcp` (MCP server framework), `usecli` (CLI framework), and Claude Code's transcript format.
