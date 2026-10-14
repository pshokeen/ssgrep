# Changelog

## [Unreleased]

### Added

- **Opt-in local savings ledger.** `metrics setup`, `collect`, `status`, `pause`,
  `resume`, and scoped `purge`; `savings`, `savings explain`, and revision-aware
  `savings history`. Observed usage remains distinct from modeled scenario credit,
  known debit, missing evidence and negative net. See [the guide](docs/savings.md).
- **Optional metrics service and organization export.** Explicit foreground
  `metrics serve` exposes authenticated loopback snapshots/OpenMetrics; separate
  per-destination/signal export consent supports sanitized metrics and permitted
  pseudonymous logs, bounded offline queues and exact acknowledged views.
  No service is automatically installed, and collector acceptance is not backend
  durability or erasure.
- **Versioned accounting methodology.** Frozen evidence, deduplicated source atoms,
  explicit frequency repinning into a new methodology namespace, and reproducible
  receipt explanations. Exact visible partial output retains its debit; pure
  ssgrep shell aggregates are charged once while executions remain individually
  counted. See [methodology and validation limits](docs/accounting-methodology.md).
- **Validation observer trust boundaries.** Native evidence uses supervisor-selected
  actor/task/run associations and bounded, digest-checked private file reads; a
  solver record or hash chain is not authentication. Unmatched completion stays
  unknown. This documents the boundary, not a claim that live/security release
  gates have passed.
- **Scope-bound export eligibility.** Collection-time scope provenance prevents
  exporting stale or unstamped snapshots even on first send. Scope changes
  require recollection into a new scope/counter epoch rather than relabeling
  frozen old totals; consent-only and credential changes do not reset scope.

### Fixed

- **`ssgrep index --live` survives a failing source.** A component error
  (an unreadable transcript, a locked OpenCode database) is now logged and the
  poller retries on the next cycle instead of exiting. One-shot `index` still
  fails loudly. (#18)
- **Index-update failures name their cause.** The error now reads
  `Index update failed: N component errors (first: <source>: <ExceptionType>: <message>)`,
  with the cause bounded in length.
- Read-only CLI startup defers indexing-engine initialization, keeping offline
  commands and machine-readable output free of import-time telemetry logging.
- Piped savings/metrics output stays plain even with `FORCE_COLOR`; matched CLI
  project totals use the captured canonical workspace rather than splitting
  execution counts from transcript costs across path aliases.
- Delivery status distinguishes discarded policy-contraction data from expiry
  and no longer describes an empty queue as an active network backlog.
- Corrupt policy/state files and pending purge markers fail closed without
  silently repairing read-only storage or resetting retained accounting.
  Bounded reads also reject files that grow past their limit during reading.
- Retention resumes loss-committed batch deletion without double-counting,
  expires receipts across interrupted cleanup, and prevents aged observations
  from reappearing in lifetime views. Missing registered spool data blocks
  delivery rather than falsely acknowledging an incomplete generation.
- Export lock failures close their directory descriptors. Invalid delivery
  dates and oversized numeric HTTP headers no longer break metrics projection.
- Malformed native message shapes and nonfinite JSON numbers no longer abort
  reading subsequent valid runtime evidence.
- Interactive setup treats EOF as cancellation; failed worker startup does not
  announce readiness. Wrapped next-action commands preserve quoted arguments,
  spaces and shell metacharacters without exceeding the terminal width.
- Airline checkpoint tests and live isolation probes no longer require unpublished
  OpenSpec planning files. The probe uses shipped fixture/oracle sources and retains
  both denied-read checks and writable-scratch positive controls.
- Collection resume suggestions quote interval arguments explicitly. The organization
  guide distinguishes owner-configured personal destinations from managed destination
  ceilings; the methodology guide maps module families to their public entry points.
- The copied/mutated-history sensitivity case now recollects its existing arithmetic
  checkpoint, preserving the original accepted variant instead of comparing it with
  a fresh store's independently admitted history.
- Python-based live-sandbox tests now name an unavailable execution prerequisite
  when their checkout or launcher is in deliberately sealed temporary storage.
  Supported layouts still exercise the protected-read and allowed-write controls.
- Ingest credential separation compares exact binary key bytes before text
  normalization, so hexadecimal copies of local keys with whitespace edge bytes
  are refused. Deterministic regressions cover both local binary-key types.
- CI retains the full test matrix with execution bounds sized for the expanded
  suite, and the Linux test step preserves its complete diagnostic output.
- Suggested-command labels are shell comments, so the complete displayed block can
  be pasted without removing its label. Command wrapping uses terminal-cell widths
  to preserve wide Unicode arguments through actual console rendering.

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
