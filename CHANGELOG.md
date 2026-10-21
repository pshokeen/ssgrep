# Changelog

## [Unreleased]

### Added

- **Saved investigation metadata, compatible with older notes.** The new MCP
  `save_note` tool and CLI `note --work FILE` attach optional provider/model,
  elapsed time and usage to one shared work ID. Existing refs can be enriched
  without rewriting transcripts; corrections retain revisions. `show` exposes
  original investment separately from savings. Legacy notes and index schemas
  remain unchanged, and unknown historical values are not fabricated.

- **Opt-in local savings ledger.** One `savings` command group provides setup,
  collect, status, pause, resume, scoped purge, summary, explain, revision-aware
  history, bounded lookups, and paged status diagnostics. Required values are
  positional where appropriate; `savings --help` groups commands by user task.
  Observed usage, modeled credit, complete nets, known subtotals, finite bounds
  and missing evidence remain distinct. See [the guide](docs/savings.md).
- **Optional savings service and organization export.** `savings export serve`
  exposes authenticated loopback snapshots/OpenMetrics; separate
  per-destination/signal export consent supports sanitized metrics and permitted
  pseudonymous logs, bounded offline queues and exact acknowledged views.
  No service is automatically installed, and collector acceptance is not backend
  durability or erasure.
- **Versioned accounting methodology.** Native-domain identity v2, compact evidence
  references by default, separately consented bounded plaintext cache, per-lookup
  frozen rarity inputs and explicit replacement revisions. Lexical rarity is a
  conservative lower bound, not exact candidate frequency. Observed service latency
  stays separate from modeled work; neither measures human time saved.
  Child task ownership follows proven native launch-call linkage, not whichever
  prompt preceded a delayed result; missing or contradictory launch context stays unknown.
  Native action/verifier outcomes also use verified call/result links. Recognized
  native timing footers no longer hide JSON cards or remove their visible context cost.
  Conflicted action, verifier and launch records cannot establish trusted reuse or
  ancestry; disputed prior context remains unavailable rather than falsely novel.
  Daily and receipt allocations preserve supported lower-tier ownership rather than
  borrowing an optimistic-only day. `honest_bounds_ns_v4` replaces current calculation;
  accepted v3 values remain read-only history without relabeling or double counting.
  Historical cwd aliases cannot transfer old evidence between organization profiles.
  Administrative rewrites cannot renew optional plaintext-cache retention.
  Lookup and receipt commands preserve opaque IDs and quoted scopes at narrow widths.
  See [methodology](docs/accounting-methodology.md) and revision-specific
  [benchmark results](docs/accounting-benchmarks.md); full release acceptance is pending.
- **Affected-closure collection.** Stable native/journal descriptors skip unchanged
  raw inputs; compact identity/session indexes and per-root state reuse limit
  projection work to dependencies. Legacy conversion, retention and compaction
  expose resumable backlog instead of declaring unfinished work complete.
  Missing or oversized committed legacy evidence reports typed corruption without
  replacing the accepted checkpoint or presenting lost evidence as empty usage.
  Previously accepted journal disappearance likewise fails without replacing committed
  values; restoring the owned journal permits recovery.
  Frozen frequency inputs are reclaimed only after a committed, resumable
  reference proof; unknown legacy indexes conservatively protect replay inputs.
  One connected root remains an atomic reduction unit; arbitrary history size
  does not have constant CPU, serialization cost or RSS. Resource revalidation
  is pending and does not infer CPU duty from wall latency.
  Genuine previously published session-scoped state enters native-proof migration
  before strict current decoding. Original published history/receipts are retained
  byte-for-byte under their original namespace, without revaluing old work units
  or extending their original retention dates.
  Original-publication retention advances by one immutable child or at most
  32 receipts, with explicit backlog for a child beyond the available byte budget.
  Original capture shares its before-read budget across history and all receipt
  reads, including foreign-method documents; blocked cursors never advance.
  Exact-child date indexes avoid rereading clean future children on each day;
  empty phases advance within a bounded metadata quantum. Legacy upgrades also
  recover staged snapshot generations without decoding old work units; unique
  uncommitted delivery pins block rather than lose their only pinned bytes.
  Already accepted compact usage is sealed before evidence expiry can discard
  its projection; this preserves lifetime counts without inflating live counters
  or reviving observations removed by aggregate retention or explicit purge.
  Archived counter deduplication linkage follows the original evidence horizon;
  durable numeric counts, histograms and epochs remain unchanged.
- **Scope-bound export eligibility.** Collection-time scope provenance prevents
  exporting stale or unstamped snapshots even on first send. Scope changes
  require recollection into a new scope/counter epoch rather than relabeling
  frozen old totals; consent-only and credential changes do not reset scope.

### Fixed

- **`search --limit 0` is a usage error, not a corrupt index.** `--limit` (and the
  service/MCP `limit`) must be at least 1; zero or negative values exit 2 instead of
  surfacing a raw LanceDB message and advising `--rebuild`.
- **`show` with no index exits 4 like `search`.** A malformed ref no longer masks a
  missing or incomplete index as "Episode not found" (exit 3).
- **Damaged index files exit 4 from `status` and `index` too.** A lance storage error
  (for example deleted table data files) is reported as `corrupt_index` with the
  `--rebuild` advice, in text and JSON, instead of a raw error with exit 1.
- **An unfinished build is resumed by plain `ssgrep index`.** An interrupted or failed
  first build, an interrupted `--rebuild`, or an empty `lancedb/` directory no longer
  demands `--rebuild` (contradicting `search`), and sources the memo journal believed
  done are re-ingested instead of staying missing. A genuinely incompatible schema or
  model still requires `--rebuild`.
- **`index --live` survives cycles that ingest data.** One App now serves every poll
  cycle (the engine kept the name of an App that had processed a source, so the next
  cycle failed with "already registered"), `--rebuild` and `--full-reprocess` apply to
  the first cycle only instead of wiping the index every poll, and Ctrl-C lets the
  running cycle finish before one final catch-up (a second Ctrl-C stops at once).
- **Interrupting a one-shot `index` exits 130 with a message** ("run `ssgrep index` to
  resume it") instead of exiting 0, printing a traceback or "Task was destroyed".
- **`ssgrep mcp` stops on SIGINT and exits cleanly.** The stdio server previously
  ignored Ctrl-C after startup (only SIGTERM worked). SIGINT and SIGTERM now give an
  in-flight startup reconcile a 3-second grace period, then end the process by that
  signal (status 130/143 in a shell) with no traceback. When stdin closes during a
  reconcile that outlasts the grace period, the process exits directly instead of
  finalizing Python under a live LanceDB worker, which intermittently crashed.
- **`mcp install` and `init` fail when a client registration fails.** A client ending
  in `error: ...` (unwritable config, invalid or commented JSON, non-object JSON,
  failed `claude mcp add`, unwritable skill directory) now makes the command exit 1
  and report `"ok": false` under `--json`, while still reporting every client.
  `skipped: ...` (client not installed) still exits 0. A Zed `settings.json` with
  comments is reported as "cannot safely edit commented or malformed settings"
  instead of a raw JSON parser error, and is left untouched. `--json` output no longer
  nests a second `ok` inside `data`.
- **Claude registration is idempotent.** `mcp install claude` and `init` read
  Claude's user-scope entry and report `already_installed` without invoking `claude`
  when it already launches the right command. A stale entry is replaced, and if the
  re-add fails the previous entry is restored instead of being lost.
- **MCP tools say when a background update is in progress.** While the server's
  startup reconcile is running, `search_sessions` and `show_session` now answer
  "being updated by this server's background reconcile; retry in a moment" instead of
  advising `ssgrep index`.
- **Codex config keeps the blank line before the next section** when the ssgrep
  section is rewritten, and a section followed by a blank line is recognized as
  already installed.
- **Savings export preview honors a zero-byte limit.** A zero payload cap now
  hides every candidate body and reports the omitted payload in the summary.
- **Paused sending now points to `savings resume`.** Delivery status for a paused
  profile with an unknown or unreachable destination previously suggested a
  connectivity check that exits 1 while paused; it now explains that nothing is
  retried until resume and names the resume command.
- **A damaged committed ledger document fails closed.** A non-object ledger blob
  during an incremental save now reports `state_corrupt` without replacing the
  checkpoint instead of raising an unhandled error.
- **Export guidance names the real commands and the real unblock step.** Messages
  that said "run `--check`" (enable notices, blocked sends) now name
  `ssgrep savings export check DESTINATION`, and a 401/403 block now says it
  clears only when the credential value, endpoint or proxy changes (a
  successful `check` does not clear it), then `send`.
- **`savings export disable` reports the unacknowledged batch count.** Pending
  batches now counts only queued or in-flight spool batches, matching `status`,
  `/v1/export-status` and `/metrics`, not acknowledged batches kept in the spool.
- **`/v1/export-status` reports `enabled`.** Each destination/signal row now
  shows whether the current policy has it enabled instead of always `null`.

- **`ssgrep index --live` survives a failing source.** A component error
  (an unreadable transcript, a locked OpenCode database) is now logged and the
  poller retries on the next cycle instead of exiting. One-shot `index` still
  fails loudly. (#18)
- **Index-update failures name their cause.** The error now reads
  `Index update failed: N component errors (first: <source>: <ExceptionType>: <message>)`,
  with the cause bounded in length.
- Read-only CLI startup defers indexing-engine initialization, keeping offline
  commands and machine-readable output free of import-time telemetry logging.
- **Savings status keeps warning text private.** The summary shows stable diagnostic codes and counts; raw warning text requires `--details --include-text`.
- **`savings status` hints are runnable.** `--details` and `--sources` print their
  diagnostics and source-inventory blocks before the closing `# Next:` command, and a
  rejected `--details --page` suggests `ssgrep savings status --details` instead of a
  non-command. The inventory is a titled `SOURCE INVENTORY` block that wraps to the
  terminal, as do section headings and failure messages at narrow widths.
- **`savings purge` names `BEFORE_DATE`.** An invalid or missing date is reported
  against the positional argument rather than a non-existent `--before` option.
- **Usage errors name the real command.** `ssgrep savings --bogus`,
  `savings explain`, `savings purge` and `savings export preview` print their full
  command path in the usage block, and a missing argument says
  `Missing argument 'receipt_id'.` instead of an empty `Invalid value` line.
- **Closed output pipes exit quietly.** `ssgrep ... | head` (text or `--json`) no
  longer prints `BrokenPipeError`; the command exits without a traceback (status 1,
  or the command's own failure status).

## [2.0.1] - 2026-09-28

### Added

- **Published to PyPI.** `uvx ssgrep`, `uv tool install ssgrep`, and
  `pip install ssgrep` now work; releases publish automatically from `v*` tags.
- **omp runtime.** ssgrep now discovers and indexes omp sessions
  (`~/.omp/agent/sessions`, Pi-format JSONL); `ssgrep init` installs the
  skill to `~/.omp/agent/skills/ssgrep`. See [docs/runtimes.md](docs/runtimes.md).
- **omp as an MCP client.** `ssgrep mcp install` and `ssgrep init` register
  the ssgrep MCP server in omp's `~/.omp/agent/mcp.json` (honoring
  `OMP_AGENT_DIR`). See [docs/mcp-setup.md](docs/mcp-setup.md).

### Changed

- **`ssgrep mcp install` writes `uvx` only when it can resolve ssgrep.** By
  default the registration is `uvx ssgrep@<version> mcp` (pinned to the
  installed version) only when `uvx` is on `PATH` *and* ssgrep was installed
  from a package index; otherwise the absolute binary path. Override with
  `SSGREP_MCP_LAUNCHER=auto|uvx|path`. Claude Code registrations are
  refreshed on every run. (#4)
- `ssgrep status` reports `archived_source_count`.

### Fixed

- **Deleted transcripts no longer flood stderr or lose history.** After an
  upgrade invalidates the pipeline's memo, sources whose files are gone are
  served from their indexed snapshot (rows kept, source stays archived)
  instead of printing a traceback per source; `ssgrep prune` now also drops a
  pruned source's registry entry. (#5)
- **OpenCode sessions deleted from a database that still exists** keep their
  indexed history instead of having it reconciled away. (#8)
- **No telemetry.** cocoindex's usage tracking (a request to scarf.sh on every
  index, note and MCP start) is disabled by default, honouring the "fully
  local and offline" promise; set `COCOINDEX_DISABLE_USAGE_TRACKING=0` to opt
  back in. (#10)
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
