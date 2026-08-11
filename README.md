# ssgrep

Search AI coding-session transcripts without an LLM, a server, or a daemon.

You have months of Claude coding sessions. Every time you solve something, you record it — but you never remember where. **ssgrep** is `grep` for your past work: point it at your project folder, and it answers "how did I solve that before?" with no setup and no infrastructure. After a one-time embedding-model download on first run, it needs no network access at all.

```bash
$ ssgrep search "how do I fix a TypeError in async code"
```

## The Problem

An AI coding session is a transcript: your prompts, the assistant's responses, tools it ran, output it received. You have hundreds of these files in `~/.claude/projects/`. Each one answers a question or solves a problem. But finding the right one requires:

- Remembering which project you worked in
- Remembering what you were solving
- Scrolling through the transcript to find the exact pattern you need

**ssgrep** inverts this: you describe what you're looking for, and it searches every session you have. It finds the relevant episodes — not just a text match, but the exact user prompt and assistant response that solved your problem.

## Why It Exists

Most retrieval tools assume you have access to an LLM. They send your queries to an API, get embeddings back, and retrieve from a central database. **ssgrep** assumes you don't:

- **No network, after first run.** A one-time ~30 MB embedding-model download happens on first use; after that, indexing, search, and the MCP server make no network calls. Transcripts are never sent anywhere, at any point.
- **No LLM.** Static embeddings run in 3.3 ms and download once.
- **No server.** It runs as a subprocess or MCP server — no daemon, no listening port.
- **No infrastructure.** `ssgrep` is sold as a direct download, not published to any package registry — after purchase, `uvx --from` your downloaded wheel runs on a clean machine with nothing else installed beyond `uv` — no daemon, no server process, a single command (see [Installation](#installation)).
- **Privacy first.** `.ssgrep/` is created 0700 and gitignored, and your transcripts are never written to. See [Privacy](#privacy) for the full list of guarantees and the tests that hold them.

The retrieval quality is honest: **87.9% recall@10** across diverse queries. Exact identifiers and error strings are at 100%. For paraphrases it is 85% and for multi-hop questions 73.7% — a known weakness documented plainly so you know when a vague query might not find anything.

## Installation

**ssgrep is sold as a direct download — not published to PyPI, TestPyPI, Homebrew, or any other package registry, and it will not be.** `pyproject.toml` carries a `Private :: Do Not Upload` classifier, which makes PyPI itself reject an accidental publish attempt server-side — verified directly. After purchase you receive a wheel file, `ssgrep-<version>-py3-none-any.whl`; every command below substitutes that file's path for a bare `ssgrep`. Read `ssgrep` everywhere else in this README as shorthand for whichever of the three forms below you used.

**Note:** ssgrep supports macOS and Linux only; Windows is not supported.

### Global Install (with `uv tool` — recommended)

`uv tool install` (from [`uv`](https://docs.astral.sh/uv/)) installs `ssgrep` once, globally, into
its own isolated environment and puts the `ssgrep` command on your `PATH` — every project and shell
gets the same binary with nothing else to remember:

```bash
uv tool install /path/to/ssgrep-<version>-py3-none-any.whl
ssgrep search "your query here"
```

Working from a source checkout instead of a wheel? The same form works from the repo root:

```bash
uv tool install .
```

Upgrades are `uv tool install --reinstall <new-wheel>`; removal is `uv tool uninstall ssgrep`.
Both forms above were verified directly (isolated tool dir, `--version` reporting the right build,
clean uninstall). One caution for developers of ssgrep itself: a global `ssgrep` on `PATH` can
shadow a project venv's binary — this repo's own tests invoke the venv binary by absolute path for
exactly that reason.

### Run Without Installing

`uvx` can run the downloaded wheel directly, with no persistent install — useful for one-off invocations or an MCP client config:

```bash
uvx --from /path/to/ssgrep-<version>-py3-none-any.whl ssgrep search "your query here"
```

### Install Into a Project (with `uv`)

To skip `uvx`'s one-time dependency-resolution cost, add the downloaded wheel to a project (model load still applies either way — see [Performance](#performance) below):

```bash
uv add /path/to/ssgrep-<version>-py3-none-any.whl
uv run ssgrep search "your query here"
```

### Install Into a Venv (with `pip`)

```bash
python3 -m venv .venv
.venv/bin/pip install /path/to/ssgrep-<version>-py3-none-any.whl
.venv/bin/ssgrep search "your query here"
```

The `uvx`, `uv add`, and `pip` forms were verified directly against a wheel built from this checkout, for `--help`, `status`, `search`, `show`, `index`, and `mcp`; the `uv tool install` forms were verified for install, `--version`, and uninstall.

### For Development

To work on `ssgrep` itself from a checkout of this repository (not the buyer flow above):

```bash
cd /path/to/ssgrep   # your local checkout
uv sync              # one-time: creates .venv, installs dependencies
uv run ssgrep search "your query here"
```

### One-Time Setup

Before searching, build an index of your sessions:

```bash
ssgrep init
```

This command:
- Creates `.ssgrep/` in the current directory (mode 0700, gitignored)
- Downloads the embedding model (~30 MB; allow a few minutes on a slow link, and set `SSGREP_MODEL_LOAD_TIMEOUT` in seconds if the default deadline is too short)
- Indexes all AI sessions in this project
- Installs the SessionEnd hook to keep the index fresh

If you prefer to skip hook installation, run:

```bash
ssgrep index
```

This indexes sessions without installing the hook. You can still use the MCP server or run searches manually.

## Quickstart

```bash
# Index your current project (one-time)
ssgrep init

# Search for a past solution
ssgrep search "how do I handle async errors"

# See full context for one episode
ssgrep show <ref>

# Check index health
ssgrep status

# Keep the index up to date
ssgrep index
```

## Commands

### `init` — One-Time Setup

```bash
ssgrep init [--project-dir PATH]
```

Builds the index, downloads the embedding model, and installs the SessionEnd hook (optional). Idempotent — safe to run multiple times.

**Output:**
```
Indexed 216 sessions, 3161 episodes, 5634 chunks.
```

### `index` — Build or Update the Index

```bash
ssgrep index [--project-dir PATH] [--scope PATH] [--rebuild] [--no-subagents] [--quiet] [--allow-shrink]
```

Index session transcripts in the current or specified project directory.

**Options:**
- `--scope PATH` — Discover transcripts recorded under PATH instead of the project
  directory, while the index still lives at `--project-dir`. The moved-repo /
  org-rename remedy: if your project used to live at `/old/path`, `ssgrep index
  --scope /old/path` indexes that stranded history *into this project's index*,
  searchable from here. The path is matched against the `cwd` strings recorded
  inside transcripts, so it need not exist on disk. The scope is persisted in the
  index and sticky: plain `ssgrep index` runs keep it (staleness, the SessionEnd
  hook, and the work-queue drain all honor it too), and only an explicit
  different `--scope` changes it, forcing a full rebuild.
- `--rebuild` — Delete and recreate the entire index (slow; use only to fix corruption). Refused if it would drop the indexed session or chunk count below half of what you already have — see below
- `--no-subagents` — Exclude subagent transcripts. They are indexed by default because they hold roughly **72% of all indexed chunks** (0.718–0.724 across the committed eval artifacts): main sessions record what a subagent concluded, while the investigation itself lives in the subagent transcript
- `--quiet` — Suppress progress output. Does **not** suppress the diagnostic printed when nothing is discovered
- `--allow-shrink` — Proceed with a rebuild that the shrink guard would otherwise refuse

**The shrink guard.** A rebuild replaces your whole index with whatever this run
discovers. If ssgrep is run from the wrong directory, or the project was moved or
renamed, discovery finds nothing — and an unguarded rebuild would commit that empty
result over your real history and exit 0. So a rebuild that would leave fewer than
half your sessions *or* fewer than half your chunks is refused, with the old and new
counts and a census of the working directories your transcripts actually record.
Nothing is written when it is refused; your existing index is untouched.

This is not a `--rebuild`-only concern: the first `ssgrep index` after an upgrade
that changes the schema or the embedding model rebuilds automatically.

Two legitimate shrinks are excluded from the comparison rather than left to the
threshold, so neither is ever blocked: transcripts you deleted from disk (a rebuild
cannot recover those; `ssgrep prune` drops them without one), and `--no-subagents`,
which is compared main-session-to-main-session. If the smaller index really is what
you want, re-run with `--allow-shrink`.

**Output:**
```
Indexed 216 sessions, 3161 episodes, 5634 chunks.
```


### `note` — Write a Durable, Searchable Note

```bash
ssgrep note --title "how do we handle retry backoff" --body -   # body from stdin
ssgrep note --title "..." --body "the answer text" [--project-dir PATH]
```

Writes a note into the project's index as a native transcript record pair and
reindexes quietly, so it is searchable immediately. **Phrase the title as the
question you will later search for** — that is the measured-best shape (authored
notes retrieve at rank 1 with near-ceiling scores). Notes live under
`<project>/.ssgrep/notes/` — never under `~/.claude`; your Claude Code history is
read-only to ssgrep, unconditionally. They follow normal staleness and tombstone
semantics, and `note` is deliberately CLI-only: there is no MCP tool that can
write to your index.

### External Transcript Roots (`SSGREP_TRANSCRIPT_DIRS`)

Index transcripts that live *outside* the Claude Code corpus — team knowledge
bases, exported histories, or hand-authored record files (the same schema
`ssgrep note` writes):

```bash
export SSGREP_TRANSCRIPT_DIRS="/team/knowledge:/exports/claude"
ssgrep index
```

Colon-separated directories; every `*.jsonl` beneath each root is indexed as a
native-format transcript through the exact same pipeline as your corpus. An
optional format tag (`native=/path`) selects the adapter — `native` is the only
format today; the tag syntax is the seam future runtime adapters plug into, and
an unknown tag fails loudly rather than silently skipping a root. External
roots are explicit configuration, so they are always in scope regardless of
`--scope`, and they participate in staleness like any transcript. A root that
is temporarily missing (unmounted volume) is skipped with a warning; removing
one permanently triggers the same shrink protection as any disappearing
content — nothing is dropped silently.

### `search` — Find Episodes

```bash
ssgrep search <QUERY> [--limit N] [--token-budget T] [--project-dir PATH]
```

Search session transcripts and return ranked episode cards. Each card shows:
- **Title** — session title or derived from first prompt
- **Date** — when the session happened
- **Score** — relevance (0–1, hybrid BM25 + semantic)
- **Files** — source file paths edited or created
- **Excerpt** — truncated context from the episode

**Options:**
- `--limit` (default 10) — return top N matches
- `--token-budget` (default 1500) — truncate excerpts to stay within token budget

**Example output:**
```
[e787f78c-c93e-46e3-ac00-8b112fa6001e:ep:159] Episode 159
  Score: 0.876
  Fix verified both directions — valid overrides apply, bogus kwargs raise TypeError...

[e787f78c-c93e-46e3-ac00-8b112fa6001e:ep:154] Episode 154
  Score: 0.654
  ...
```

**Exit codes:**
- `0` — Found results
- `1` — Unexpected internal failure
- `2` — Usage error (empty query, bad flags)
- `3` — No results found (index exists but no matches)
- `4` — Index not found or corrupt (run `ssgrep index` to rebuild)

> **On exit code 3:** you get this when the index exists but holds no sessions for this
> project — in `--json` mode with a structured `index_empty` document that includes a
> zero-discovery census naming the directory scanned and what it found. On a *populated*
> index, exit 3 is effectively unreachable: the vector leg is a brute-force cosine search
> with no relevance floor, so it always returns its top-k candidates and even a weak query
> matches *something*. Filter on `score` if you need a quality threshold.

### `show` — Full Episode Context

```bash
ssgrep show <REF> [--project-dir PATH]
```

Display the complete, untruncated user prompt and assistant response for one episode. The `<REF>` comes from `search` output (format: `session-id:ep:episode-num`).

### `status` — Index Health

```bash
ssgrep status [--project-dir PATH]
```

Check index existence, session/episode/chunk counts, freshness, embedding model, and staleness.

**Example output:**
```
Sessions:  216
Episodes:  3161
Chunks:    5634
Size:      22816768 bytes (21.8 MiB)
Indexed:   2026-07-28T02:18:17.887500+00:00
Model:     minishlab/potion-base-8M (dim 256)
Records:   0 skipped, 0 malformed
Stale:     yes (8 transcripts changed)
```

**Interpreting staleness:**
- `Stale: no` — index matches your sessions exactly; no re-index needed
- `Stale: yes (N transcripts changed)` — N sessions have new content; run `ssgrep index` to catch up
- The hook (installed by `init`) enqueues work when a session ends. `status` reports index
  STALENESS, not hook health: if the hook never fires, `Stale: yes` is the signal, and
  `ssgrep index` is the authoritative way to catch up either way.

### `prune` — Delete Archived Content

```bash
ssgrep prune [--older-than DAYS] [--dry-run] [--yes] [--project-dir PATH]
```

Permanently delete tombstoned (archived) content from the index.

**Options:**
- `--older-than` (default 0) — only delete sessions not seen on disk for at least N days
- `--dry-run` — show what would be deleted without actually deleting
- `--yes` — skip the confirmation prompt (required in JSON mode)

**Why this exists:** When you delete a session file from `~/.claude/projects/`, ssgrep keeps it searchable (see "Archive Semantics" below). Running `prune` permanently deletes archived content that has been missing from disk.

### `version` — Show Version

```bash
ssgrep version
```

Display installed version.

### `mcp` — Start MCP Server

```bash
ssgrep mcp
```

Start the Model Context Protocol server over stdio. See [docs/mcp-setup.md](docs/mcp-setup.md) for integration with Claude Code, Cursor, and other editors.

### `hooks` — Manage the SessionEnd Hook

```bash
ssgrep hooks [install|uninstall|enqueue] [--project-dir PATH]
```

Install or uninstall the Claude Code SessionEnd hook that keeps the index fresh. `enqueue` is the action the installed hook itself invokes when a session ends — it queues the session for reconciliation without indexing directly; it's exposed for manual/debugging use rather than typical end-user use. Called automatically by `init`; exposed for manual control.

## JSON Output

Every command accepts `--json` and emits **exactly one** JSON document on stdout. Human-readable
progress and diagnostics always go to stderr, so stdout stays machine-clean and safe to pipe.

**Success** wraps the payload in an envelope:

```bash
ssgrep search "flaky migration" --limit 1 --json
```
```json
{
  "ok": true,
  "data": {
    "results": [
      {
        "ref": "11111111-2222-4333-8444-555555555555:ep:0",
        "title": "how do I retry a flaky database migration",
        "timestamp": "2026-08-01T10:00:00+00:00",
        "score": 0.0328,
        "excerpt": "how do I retry a flaky database migration",
        "files_touched": [],
        "is_subagent": false,
        "agent_name": null,
        "agent_description": null,
        "parent_session_id": null,
        "content_type": "prompt",
        "source_absent": false
      }
    ],
    "omitted_count": 0,
    "index_exists": true,
    "index_empty": false,
    "total_matches": 1,
    "excerpts_truncated": false,
    "clamped": false,
    "stale": false,
    "stale_count": 0
  }
}
```

**Errors** are structured documents, not stack traces — `condition` is a stable machine-readable
tag and `command` is the remedy to run:

```json
{
  "ok": false,
  "condition": "unknown_ref",
  "message": "Episode not found: nope:ep:0. Run `ssgrep search <query>` to find valid refs.",
  "command": "ssgrep search <query>"
}
```

The exit code carries the same signal as `condition`, so a script can branch on either
(see [`search` exit codes](#search--find-episodes)).

| Condition | Exit | Meaning |
|-----------|------|---------|
| `empty_query` | 2 | Query was empty or whitespace |
| `confirmation_required` | 2 | `prune` needs `--yes` — it never prompts in JSON mode |
| `rebuild_would_shrink` | 2 | Rebuild refused by the shrink guard; re-run with `--allow-shrink` if intended |
| `unknown_ref` | 3 | `show` ref does not resolve |
| `index_empty` | 3 | Index exists but holds no sessions for this project |
| `missing_index` | 4 | No index yet — run `ssgrep index` |
| `corrupt_index` | 4 | Index unreadable or built by another schema version; rebuild it |
| `index_error` | 1 | Indexing failed for another reason |
| `generation_contention` | 1 | `prune` could not get a stable lock against concurrent indexing; re-run |

`index_empty` additionally carries a `zero_discovery` census (the directory scanned, how many
transcripts it holds, how many the project scope rejected, and a remedy command) so an agent can
diagnose a scope mismatch without a terminal to read.

Notable fields: `stale`/`stale_count` appear on `search`, `show`, and `status` so a consumer can
tell it is reading a lagging index; `source_absent` marks a result whose transcript has been
deleted from disk but is still searchable (see [Archive Semantics](#archive-semantics)); and
`prompt_truncated`/`response_truncated` on `show` mark output the detail layer bounded.

## MCP Server Integration

**ssgrep** exposes a Model Context Protocol (MCP) server that allows AI assistants to search sessions from within your editor.

### Setup

`ssgrep` has no public registry entry, so register it using Claude Code's `mcp` command:

```bash
claude mcp add --scope user ssgrep -- uvx --from /path/to/ssgrep-<version>-py3-none-any.whl ssgrep mcp
```

Alternatively, create a `.mcp.json` file in your project root with:

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

Verified directly: `uvx --from /path/to/<wheel file> ssgrep mcp` starts the server correctly. If you installed into a venv instead (see [Installation](#installation)), use `claude mcp add` with the path to that venv's `ssgrep` binary and `mcp` as the single argument, or create a `.mcp.json` with `"command": "/path/to/.venv/bin/ssgrep"` and `"args": ["mcp"]`.

Then from Claude Code or another MCP-aware editor, you can ask questions like:

> "Search my session history: how did I handle database migration errors before?"

For detailed setup instructions and support for other editors (Cursor, Zed, Codex CLI, opencode), see [docs/mcp-setup.md](docs/mcp-setup.md).

## Performance

ssgrep is measured for latency and retrieval quality on real workloads. All numbers below are from a reference M-series macOS machine with ~327 MB of real transcripts across multiple projects.

### Search Latency

| Component | Time |
|-----------|------|
| CLI startup | ~136 ms |
| Model load (cold) | ~324 ms |
| BM25 search | ~0.2 ms |
| Vector cosine search | ~0.9 ms |
| **Total (fresh process)** | **~523 ms** (median) |

CLI startup is measured as `ssgrep status --project-dir <dir>`, which resolves the project directory and reads the index — the same work `search` does before it ever loads the model. A bare `ssgrep version` (no project_dir, no index) is faster (~84 ms) but isn't representative of what a real search pays. BM25 and vector cosine are measured in-process against a 7,582-chunk index (36 real queries × 3 repeats each); they're internal per-leg costs rather than independently-timed subprocess stages, so they won't sum exactly to the total above. That total is its own direct subprocess measurement — min 481 ms, median 523 ms, max 557 ms (n=12), the figures `tests/test_perf.py` asserts against — not an arithmetic sum of the rows.

The model-load overhead dominates every invocation and applies whether you run via `uvx` or a local install — installing locally does **not** avoid it; a fresh process always pays ~324 ms for a cold model load. What local install avoids is `uvx`'s one-time dependency-resolution cost: measured on this machine, a cold first `uvx --from <wheel file> ssgrep ...` call took ~1.6 s (resolving and caching the environment) versus ~0.45–0.50 s for every call after that, once `uv`'s cache is warm — at that point `uvx` and a local install cost the same. If you run `ssgrep` often enough that even that first resolution matters, install into a project instead:

```bash
uv add /path/to/ssgrep-<version>-py3-none-any.whl
ssgrep search "your query"  # still ~523 ms — model load dominates either way
```

### Retrieval Quality

Measured against 66 independently-derived queries:

| Query Class | Recall@10 | MRR |
|-------------|-----------|-----|
| Exact identifier (e.g., `TypeError`) | **100%** | ≈0.81 |
| Error string (e.g., `ENOENT: no such file`) | **100%** | ≈0.94 |
| Multi-hop (connect two unrelated concepts) | **73.7%** | ≈0.51 |
| Subagent-only (requires looking in subagent transcripts) | **100%** | ≈0.80 |
| Paraphrase (vague, natural-language query) | **85.0%** | **≈0.42** |
| **Overall** | **87.9%** | **≈0.63** |

Source artifact: `eval/results/baseline_n66_2026-08-07b.json` (158 sessions, 7,348 chunks
at measurement time). The label set grew 36 → 48 → 66: the 2026-08-06 regeneration replaced
13 queries whose targets Claude Code's transcript cleanup had deleted and added 12 blind-composed
queries; the 2026-08-07 extension merged 18 more queries that had been held out of — and were
never used in — any tuning decision (6 from the 2026-08-06 label repair's holdout, 12 blind-composed
after all 2026-08-07 tuning was frozen; see `eval/build_labels.py` for the anchor methodology).
`tests/test_published_numbers.py` asserts the **Overall** row against that
file on every CI run, so this table cannot drift from the evidence without failing the build.

**How to read these numbers.** They are a snapshot of a live, growing corpus, not fixed constants:

- **Re-runs are deterministic.** Rebuilding against an *unchanged* corpus reproduces every metric
  bit-identically (60 such rebuilds in `eval/results/noise_floor_2026-07-27.json`). Movement between
  measurements is the corpus changing, never measurement noise.
- **MRR drifts, without direction.** Across 12 consecutive corpus states in
  `eval/results/matched_pair_gap_study_2026-07-28.json`, overall MRR moves non-monotonically
  between 0.5369 and 0.5582 — it rises about as often as it falls. Do not read a trend into a
  single pair of measurements. CI allows ±0.012.
- **Per-class figures are n=13–20.** One query changing rank moves a per-class recall number by
  5.0–7.7 points. Treat the per-class rows as indicative and the Overall row as the load-bearing one.
- **Recall@10 is a step function.** A hit at rank 8 scores the same as rank 3, so recall only moves
  when a query crosses the top-10 boundary — rarer than the rank churn MRR registers every run.

**Interpreting the `score` field.** The score attached to each result is a weighted
reciprocal-rank-fusion sum (six legs of `w/(60 + rank)`), **not** a 0–1 relevance
probability — its magnitude is set by the fusion constants, and the highest score a
top result reaches on this corpus is ≈0.10 (`eval/results/score_calibration_2026-08-07.json`).
Do not filter on an absolute threshold like 0.3: it can never fire, and genuinely
decision-changing results routinely score within 1.5–2× of the noise floor. If you need a
quality bar, calibrate it on *your* index: run two or three nonsense queries
(e.g. `zzqqxx blorptastic wibblefrotz`), take the highest score they return as your noise
floor (measured here: ≈0.034), and treat ~1.5× that as the lowest plausibly-relevant score.
The floor falls as the corpus grows, so raw scores are not comparable across indexes of
very different sizes — when merging results from multiple indexes, interleave by rank,
never by raw score.

For a stable scale, every result also carries **`score_normalized`**: the raw score divided
by the ranking's theoretical ceiling (a result ranked first in every fusion leg with a
maximal roll-up tail — derived from the live fusion constants, currently ≈0.097, a ceiling
real results genuinely approach). It lands in [0, 1], survives config retunes, and is the
field to use for thresholds or cross-index comparison; the noise-floor calibration above
still applies (divide your measured floor by the same ceiling to place your bar).

**What these numbers mean:**

- **87.9% recall@10:** The right episode appears in the top 10 results for roughly 7 out of 8 queries.
- **Weaker paraphrase and multi-hop ranking:** Vague queries like "How do I handle retries?" usually
  surface a relevant episode somewhere in the top 10 (85%), though often not at rank 1 (MRR ≈0.42).
  If your search returns nothing useful near the top, try a more specific query (include a library
  name, error message, or function signature).
- **MRR ≈0.59:** The median query finds a relevant hit in the top 2 results.

The weak paraphrase performance is a known trade-off: static embeddings trade ~28% relative quality for three orders of magnitude in latency. For more details, see "Paraphrase Ranking" in [docs/architecture.md](docs/architecture.md#paraphrase-ranking-mteb-3111-vs-4292).

### Index Size and Growth

- **Index size:** ~22.8 MB for 5,634 chunks across 216 sessions
- **Disk footprint:** Scales linearly with transcript volume (roughly 0.4 MB index per 100 chunks, ~4 KB/chunk — measured directly from the live index: 24,462,336 bytes for 5,885 chunks)
- **Re-index time:** ~82 ms for no-change re-index (single-session fixture, median of 10 runs — matches the 84 ms `tests/test_perf.py` already asserts); ~2.4 seconds for a full index of 242 sessions from scratch (7,561 chunks)

Growth is unlikely to threaten performance at the scales measured here: at 7,582 chunks, cosine
search is under 1 ms, while cold model load (~324 ms) dominates end-to-end latency regardless of
corpus size. Disk grows linearly; search cost is dwarfed by the fixed model-load floor. See
[docs/architecture.md](docs/architecture.md#scalability) for the scaling analysis.

## Privacy

All five guarantees below are asserted by tests that run in every CI build:

1. **Everything is local.** No transcripts are sent anywhere. All indexing and searching happens on your machine.
2. **No network after model download.** The embedding model downloads once (~30 MB) from the model2vec registry; after that, no network access is needed or used — by `index`, `search`, or `ssgrep mcp`.
3. **`.ssgrep/` is created mode 0700 and gitignored.** The index directory is not world-readable and is automatically excluded from version control.
4. **No LLM sees your data.** Queries and transcripts are never sent to any LLM API. Embeddings are computed locally with a static model.
5. **Transcripts are never written to.** The index is a one-way read. Your session files remain exactly as they are.

A regression in any of these fails the build.

## Archive Semantics

**ssgrep stores chunks, not offsets.** When you delete a session file from `~/.claude/projects/`, ssgrep keeps its content searchable. The deleted session is marked as "tombstoned" — archived but still searchable.

**Why?** Deleting a transcript file does not mean you no longer want to search it. Sessions often get garbage-collected by Claude Code, but the knowledge in them is permanent. By storing chunk text directly (rather than offsets into files), ssgrep outlives the transcripts.

**Cleanup:** There is no automatic grace period. `ssgrep prune`'s default `--older-than` is **0**, so confirmed-tombstoned content is deleted the moment you run `prune` and confirm — not after 30 days or any other waiting period. Search remains available until you explicitly run `prune`; if you want a grace period, opt into one with `--older-than N` (only prunes sessions not seen on disk for at least N days).

```bash
ssgrep status                  # Shows tombstone count
ssgrep prune --dry-run         # Preview what would be deleted, without deleting
ssgrep prune                   # Permanently removes ALL tombstoned content now (after confirmation)
ssgrep prune --older-than 30   # Opt in to a 30-day grace period before deletion
```

## Architecture and Implementation

For the technical design — how it indexes, how it searches, why it uses SQLite + memory-mapped vectors, and how episode segmentation works — see [docs/architecture.md](docs/architecture.md).

## Feedback and Support

This is Alpha software. Expect changes to the CLI, index format, and retrieval algorithm. If you find a retrieval failure (a query that should work but doesn't), please report it with the query and what you were looking for, through the support channel listed on the purchase page.

`ssgrep` is closed-source, paid software — this is not a public project accepting outside code contributions.

## License

PolyForm Internal Use License 1.0.0 — not open source. See [LICENSE](LICENSE) for the full text.
