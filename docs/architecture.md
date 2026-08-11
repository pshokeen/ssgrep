# Architecture of ssgrep

**What this project is.** `ssgrep` indexes AI coding-session transcripts from `~/.claude/projects/**/*.jsonl` and answers "how was this problem solved before?" without an LLM, a server, or a daemon. It ships as a wheel you run directly (`uvx --from ssgrep-<version>-py3-none-any.whl ssgrep ...`) — see the [README's Installation section](../README.md#installation). Python 3.11+, macOS and Linux.

**Measurement baseline.** Every number below was measured, not estimated, on the reference machine (Apple-silicon macOS) against the live real transcript corpus. That corpus grows and prunes itself continuously, so each figure states the scale and date it was taken at; re-measure rather than assume a figure is a constant.

## Storage: SQLite + Memory-Mapped Vectors

State lives in `.ssgrep/`, a hidden directory holding two files:

- **`index.db`** — SQLite database (`src/ssgrep/store/schema.py`, `SCHEMA_VERSION = 4`; the 3→4 bump added the trigram table and forces a one-time rebuild) with tables for `chunks`, `episodes`, `sessions`, `session_files`, and `meta`, plus `chunks_fts` (word-level FTS5, for the BM25 and phrase legs) and `chunks_fts_tri` (trigram-subword FTS5, for the trigram leg)
- **`vectors.f32`** — raw memory-mapped float32 matrix, shape `(n_chunks, 256)`, keyed by the `vec_row` column in `chunks`

`episodes` stores canonical `prompt_text` and `response_text` columns alongside its metadata. `ssgrep show` reads those columns directly (`src/ssgrep/detail.py`) rather than reconstructing episode text from overlapping search chunks, so what `show` prints is the text as parsed, with no de-overlap step to get wrong.

### Why not a vector database?

At project scale (7,582 chunks, 2026-07-28), brute-force cosine search is **0.94 ms** — sub-millisecond, and negligible next to the ~324 ms of cold model load that dominates every search regardless of corpus size. Any specialized vector index spends more time *loading* than brute force *scans*. Memory-mapping means we touch only the pages we read, and the index stays as simple as possible. See [Scalability](#scalability) for what is and isn't safe to project from that number.

Alternatives evaluated:

- **Postgres+pgvector**: a server process; disqualifying for a CLI.
- **sqlite-vec**: alpha, and its macOS wheel dependency (`pysqlite3-binary`) does not exist.
- **DuckDB+VSS**: HNSW persistence is experimental with explicit corruption warnings.
- **LanceDB**: solid runner-up, but at under 1 ms of brute-force cosine there is nothing to buy.

The narrow interface (`append(vectors) → row_ids`, `cosine_topk(query, k) → hits`) is deliberately swappable — if brute force ever saturates, an ANN index replaces one function.

## Indexing: Append-Offset Cursor

Transcripts are append-only. The `session_files` table stores `(path, size, mtime, byte_offset, first_line_hash)` for each source.

**On re-index:**
1. For each session file, check `size >= stored_size AND hash(first_line) == stored_hash`
2. If both hold, seek to `byte_offset` and parse only the new bytes
3. Otherwise, delete the file's rows and re-parse it whole

This is what makes "re-index a 33 MB session that grew by 4 KB" cost **4.66 ms** (measured for a 3,870-byte append; the reference behind `src/ssgrep/repair.py`'s bounds) instead of seconds. That append reads 12,062 bytes with the cursor, versus 67,264,584 bytes for a whole-file re-read.

The guard is crude but correct. Transcripts are append-only by design and never truncated, so `size >= stored` is reliable. The hash protects against rewrites that preserve size (rare, but possible).

### Why not content hashing?

General-purpose incremental engines like CocoIndex use content fingerprinting because they must handle arbitrary sources with arbitrary mutation patterns. We handle exactly one pattern (append-only JSONL), so append-offset exploits a specific property and wins in practice.

### Reconciliation: Hints and Authoritative Checking

Hooks enqueue durable work items (session id, path, reason) and exit **without indexing**. Hooks are bypassed by SIGKILL, OOM, and `disableAllHooks`; they cannot be a source of truth.

Authoritative reconciliation runs on the explicit `index` command, on MCP server startup, and when the hook work queue is drained. It never trusts a directory listing. It:
1. Detects staleness with a cheap stat-and-cursor check (22.69 ms measured across 1,209 files; 1.57 ms for this project's 88)
2. Reports it to the caller
3. Performs bounded tail repair within strict byte and time caps (`MAX_APPEND_BYTES = 256_000` in `src/ssgrep/repair.py`; 4.66 ms for a 3,870-byte append)

Unbounded backlog is refused and reported rather than blocking search.

## Retrieval: Six-Leg Hybrid Fused by Weighted RRF

Developers search for three kinds of things:

- **Exact identifiers**: `TypeError: cannot read property 'x' of undefined`, `useDeferredValue`, `retry_backoff_ms`
- **Rare tokens**: specific library names, config keys, error codes
- **Paraphrases**: "How do I handle retries?", "What's the pattern for async setup?"

Static embeddings (`potion-base-8M`, MTEB Retrieval 31.11) are weakest on exact identifiers — precisely where BM25 shines. Two precision legs carry both ends, three cheap FTS5-based support legs (all measured in with `eval/harness.py`, 2026-08-06) close the gap on partial-overlap, morphology, and word-order signals the first two cannot see, and a late-interaction rerank leg (added 2026-08-07) re-scores the fused shortlist with per-token detail the pooled vectors discard:

1. **BM25 AND leg** (FTS5, weight 1.0): every literal token must match — high precision on pasted identifiers and error strings, correctly empty on paraphrases
2. **Vector leg** (cosine similarity, weight 1.0): semantic queries and paraphrases
3. **BM25 OR leg** (weight `OR_LEG_WEIGHT = 0.9`): partial token overlap — a paraphrase rarely contains *every* literal token of its target, but often one rare one; skipped for single-token queries where it would duplicate the AND leg
4. **Trigram leg** (`chunks_fts_tri`, weight `TRIGRAM_LEG_WEIGHT = 1.0`): shared 3-character substrings over tokens ≥4 chars — bridges morphology and compounding ("undercounted" reaches "under-report") that word-boundary tokenizing treats as unrelated terms
5. **Phrase leg** (weight `PHRASE_LEG_WEIGHT = 0.5`): adjacent bigrams/trigrams of informative query tokens as quoted FTS5 phrases — word order as a precision signal; a pasted literal like `MTEB Retrieval 35.06` names its source episode almost uniquely as a phrase
6. **MaxSim rerank leg** (`src/ssgrep/search/rerank.py`, weight `RERANK_LEG_WEIGHT = 1.0`): the top `RERANK_CANDIDATE_POOL = 250` chunks from the five-leg fused ranking, re-scored by ColBERT-style MaxSim over per-token embeddings from the *same* already-loaded model2vec model (`embed.encode_sequence()`) and fused back in as a sixth ranked list — a prefetch-then-rerank stage (~33 ms/query, no extra model or dependency), whose measured value on never-tuned queries is recall rescue: surfacing evidence the other five legs rank too deep

**Fusion via weighted Reciprocal Rank Fusion (RRF):** `score(doc) = Σ w_i/(k + rank_i(doc))` with `k = 60` (`RRF_K`, `src/ssgrep/search/__init__.py`). RRF needs no score calibration between incomparable scales and is robust to missing ranks.

Trade-off: static embeddings cost ~28% relative retrieval quality on the MTEB Retrieval benchmark (31.11 vs all-MiniLM's 42.92) but gain three orders of magnitude in latency (3.3 ms to encode a query versus 2–5 s just to import a transformer stack). The shipped performance guard is a loose 1500 ms end-to-end budget (`tests/test_perf.py`) against a measured median of 523 ms as a subprocess — of which ~324 ms is cold model load, the single largest component.

### Retrieval Unit: Episodes, Not Turns

A **chunk** is scored independently, but results are **rolled up to episodes** — one user prompt plus every assistant message it triggered. That matches the question being asked; a bare assistant paragraph is unusable out of context. The episode score is its best chunk's fused score plus `ROLLUP_TAIL_WEIGHT = 0.05` × each of the next two best chunks (a bounded tail, retuned 0.2 → 0.05 on 2026-08-07 after tail-vote accumulators were measured outranking episodes holding a top-12 fused chunk), so split evidence still counts without letting long episodes win on bulk.

**Edge case: compaction.** Sessions are compacted — a `system` record with `subtype: "compact_boundary"` marks a summarization point. These carry `parentUuid: null` (like a session start) but are **not** new sessions; they are hard episode boundaries. The real 33 MB session has 9 null-parent records: 1 session start + 8 compaction events. A naive parser reads that as 9 sessions.

## Indexing Pipeline

```
Discovery → Records → Episodes → Signal → Metadata → Chunker → Embed → Store + Vectors
```

Module map: `discovery.py`, `records.py`, `episodes.py`, `signal.py`, `metadata.py`, `chunker.py`, `embed.py`, `vectors.py`, and the `store/` package (`schema`, `generations`, `lifecycle`, `compaction`, `cwd_cache`). `indexer.py` drives the run; `indexer_support.py` holds its two self-contained side quests (draining the SessionEnd hook queue, keeping `.ssgrep/` out of the working tree) so `indexer.py` stays under the repo's file-size gate. `repair.py` and `staleness.py` implement bounded tail repair and stale detection; `search/` implements the six retrieval legs and their weighted RRF fusion; `detail.py` and `render.py` back `show`; `textsafe.py` escapes transcript-derived text at every text-mode interpolation point (transcript content is attacker-influenced, and raw ESC bytes could otherwise forge terminal output — JSON and MCP output are never escaped, so machine consumers get the original bytes).

### Discovery: Forward Encoding with Full `cwd` Projection

Claude Code encodes `/Users/p/code/github.com/x/y` as `-Users-p-code-github-com-x-y`. This is not reversible — the encoded name cannot distinguish a literal `-` from a separator.

**Naive approach fails:** encode the target path and match directory names. This misses real sessions, because work happens across multiple project directories (renamed repos, mid-session `cd`). Measured: one project's encoded directory exists but holds zero transcripts, while 213 files / 5,662 records with that `cwd` value live elsewhere.

**Naive approach #2 fails:** scan only the first record's `cwd`. That under-reports for 12 of 16 `cwd` values in the reference corpus.

**Correct method:** full streaming projection of `cwd` across all transcripts, building a `cwd → files` index — measured ~234 ms over 1,209 files / 321 MB with a fast JSON path (~790 ms with stdlib `json`). It is a bootstrap cost, cached (`store/cwd_cache.py`) and updated incrementally, never recomputed per command; caching it is what brought subprocess search from ~1,044 ms down to today's ~523 ms. A session is in scope if **any** of its records has a `cwd` at or beneath the requested folder.

### Records: Streaming, Tolerant, Line-at-a-Time

The JSONL format is undocumented and evolving, so parsing is **allowlist-driven**. `records.RecordType` enumerates exactly what is recognized: `user`, `assistant`, `system`, `queue-operation`, `permission-mode`, `mode`, `last-prompt`, `attachment`, `agent-name`, `custom-title`, `ai-title`, `file-history-snapshot`, `bridge-session`, `agent-setting`, `pr-link`. Anything else classifies as `unknown` and is counted, not parsed. Recognized content blocks are `thinking`, `text`, `tool_use`, `tool_result`, and `image`; unknown block types are skipped. Malformed lines increment a counter instead of aborting, and a per-line cap (`MAX_LINE_BYTES = 500_000`) skips pathological records — the corpus contains one 552 KB base64-encoded image on a single line.

**Notable measurement:** 2,135 of 2,135 `thinking` blocks have `"thinking": ""` — 100% empty on disk. The 19.9% of corpus bytes they occupy is opaque base64 signature payload for API replay, containing zero readable text. There is therefore no reasoning-trace indexing.

### Signal: A Few Percent of Bytes are Prose

Over a 54 MB sample:

| Content | % of Bytes | Indexed? |
|---|---|---|
| `assistant/thinking` | 19.9% | No (empty) |
| `toolUseResult` + `tool_result` | 27.3% | No |
| `assistant/tool_use` | 7.0% | No (args dropped) |
| `user/text` + `assistant/text` | 5.4% | **Yes** |
| `system/away_summary` | <0.1% | **Yes** |
| Everything else | ~39% | No |

The ratio moves with corpus composition rather than being a fixed constant: measured independently across three sessions (83 KB, 4.5 MB, 33 MB) it was 3.6%, 1.7%, and 5.5%. A full-corpus re-measurement — running `records.read_records` plus `signal.extract_signal_text` over every session under `~/.claude/projects` — put it at **3.48%** (24,721,510 of 710,593,607 bytes; 1,813 files, 230,269 records, 2026-08-06). To reproduce: parse each `*.jsonl` with the production parser, extract signal text per record, and divide total extracted UTF-8 bytes by total file bytes. Whatever the exact figure on a given corpus, the conclusion is unchanged: prose is a small enough fraction of raw bytes that every downstream component can stay simple.

### Metadata: No LLM, Pure Harvesting

Title precedence: `custom-title` → `ai-title` → `last-prompt` → first user prompt, each normalized (wrapper tags, ANSI sequences, and control characters stripped) and skipped when normalization leaves it empty.

Files touched are harvested from `tool_use` blocks named `Read`, `Edit`, or `Write`, all of which carry the path under the **`file_path`** key. Tools without a path key (such as `Bash`) contribute tool names only.

For subagent transcripts, the sibling `agent-<hash>.meta.json` carries `agentType`, `model`, and `description` (a human-authored task statement). That description is both attribution metadata **and** indexable prose — it is exactly the kind of problem statement queries match against.

### Chunking: ~1750 Characters with Overlap

Prose is split into ~1,750-character chunks with ~450-character overlap (retuned from 1,200/200 on 2026-08-07 — the smaller window predated the support legs and rerank leg, and the re-sweep found larger context windows lift paraphrase and multi-hop ranking while producing ~21% fewer chunks), preferring paragraph boundaries. Chunk IDs are content-derived (`sha256` of offset + text) and stable across runs.

An A/B against per-turn chunking (`eval/results/chunking_ab_2026-07-27.json`) found recall@10 identical between the strategies, verified query-by-query: the same 28 of 36 queries hit in both arms, not merely equal aggregates. A dedicated 36-run matched-pair study (`eval/results/matched_pair_gap_study_2026-07-28.json`) found the MRR gap crossing zero repeatedly — 7 sign changes across 15 distinct corpus states — with the gap's own spread (0.036) larger than either arm's individual MRR spread. There is nothing stable to resolve here, so uniform chunking ships: it is what is built and tested, not a measured winner.

### Embedding: Static, Fast, Swappable

`model2vec` + `potion-base-8M` (`minishlab/potion-base-8M`, pinned to a fixed Hugging Face revision): 256 dimensions, float32, L2-normalized. Measured:

- Import: 51 ms
- Load (warm): 88 ms; cold, in a fresh process: ~324 ms
- Per-query encoding: 3.3 ms cold — the first call in a fresh process, which is what a real `search` subprocess pays
- Batch throughput: 23,951 docs/sec on real chunk text (mean 603 characters — the shape `index()` actually embeds). Short, query-length strings (~44 characters) measure 132,000–170,000 docs/sec; that is a real number for the wrong workload, and it must not be used to estimate indexing time.
- Full-corpus embedding at the realistic rate: ~0.3 s for a 7,582-chunk corpus, small next to the ~1.7–2.0 s full-index time in `tests/test_perf.py`. Indexing is parse-bound, never embed-bound.

The interface is deliberately narrow (`encode(list[str]) → np.ndarray`) so the model is swappable, and `embed.DIMENSION` is the single place the vector width is defined. Model ID and dimension are recorded in the index; a mismatch forces a rebuild.

## Key Architectural Decisions

### Subagents Indexed by Default

Subagent and workflow transcripts hold **~72% of all indexed chunks** (`subagent_chunk_share` 0.718–0.724 across the committed eval artifacts). Excluding them by default would discard most of what the tool exists to find: main sessions hold only a summary of what subagents concluded, while the investigation itself happens in the subagent transcripts.

Mitigation: **attribution and rank preference.** Subagent hits are labeled with the agent task description and parent session, and the main-session rank boost is a tuned constant (`MAIN_SESSION_BOOST`, currently `0.0` — the value the evaluation harness selected; see `eval/results/boost_sweep_2026-07-27.json`).

### Two-Store Crash Atomicity

`index.db` and `vectors.f32` cannot commit atomically. Solution: staged generations with an atomic manifest swap (`store/generations.py`). On a crash between commits the store is left consistent — no orphaned vector rows, no chunk referencing a missing row.

### Archive, Not Cache

The index stores chunk text rather than offsets into transcripts. This costs roughly double the SQLite footprint but buys the core property: **the index outlives the transcripts**. When Claude Code garbage-collects old sessions, `ssgrep` retains them.

The corpus prunes itself — over a single ~20-minute window it went from 10 project directories / 1,181 files / 326 MB to 6 / 1,130 / 298 MB. Storing offsets would create dangling pointers to exactly the sessions users most want to search.

**Tombstoning, not deletion:** vanished sources are marked absent but remain searchable. Only the explicit `prune` command deletes, and it requires confirmation. This disqualifies CocoIndex, whose contract removes target states that cease to be declared.

## Known Limitations

### Paraphrase Ranking (MTEB 31.11 vs 42.92)

**Update, 2026-08-06:** this trade-off was re-measured directly. Swapping in or adding transformer embedders (all-MiniLM-L6-v2, MTEB 42.92; bge-small-en-v1.5, MTEB ~51.7) adds **zero recall@10** over the shipped five-leg hybrid on this corpus (39–41/48 versus the hybrid's 41–42/48). The remaining paraphrase misses are world-knowledge vocabulary gaps — project dialect like "cwd" for "working directory" — that no offline embedder bridges. The latency argument is therefore no longer the only justification for static embeddings: the quality upside of a transformer swap measured ≈zero. The July 2026 study below is retained verbatim as dated history.

Static embeddings trade ~28% relative retrieval quality on MTEB Retrieval for three orders of magnitude in latency, and the gap is most visible on paraphrase queries ("How do I handle retries?"). BM25 covers exact tokens, and the evaluation harness measures the result before release.

The alternative was measured rather than assumed (July 2026 study; recall figures below reflect the n=36 label set of that date). A 36-run matched-pair study of `potion-retrieval-32M` (MTEB Retrieval 35.06) against the shipped `potion-base-8M`, at the shipped `MAIN_SESSION_BOOST = 0.0` (`eval/results/matched_pair_gap_study_2026-07-28.json`, backed by `eval/results/model_ab_2026-07-27.json` and `eval/results/noise_floor_2026-07-27.json`):

- **Overall recall@10** is 77.8% in every run of the shipped arm — zero spread. `potion-retrieval-32M` ties that in 19 of 36 runs and beats it by one query (+2.8 points) in the other 17.
- **Per class, this is a trade, not a thin margin.** Paraphrase wins 36/36 (55.6% → 66.7%). Multi-hop *loses* 36/36 (66.7% → 55.6%, zero variance). Subagent-only loses in 19 of 36 runs (69.2% → 61.5%) and ties in the rest — and subagent transcripts hold ~72% of this corpus's indexed chunks. Exact-identifier ties recall 36/36 (both arms at 100%) but regresses MRR in all 36 runs (≈0.86 → ≈0.83). Error-string and overall are mixed: 17 wins, 19 ties, no losses.
- **MRR direction is real; magnitude is not.** The gap favors `potion-retrieval-32M` in 36/36 runs (mean +0.026 overall MRR) but ranges from +0.002 to +0.045 across corpus states measured minutes apart — a wider spread (0.043) than either arm's own MRR spread (0.021 base-8M, 0.038 retrieval-32M). Matched-pairing does not cancel corpus drift here, so no fixed effect size can be quoted.

These counts are not measurement noise: 60 independent rebuilds at an unchanged corpus are bit-identical on every metric. They are the live corpus differing between runs and the two models responding to that difference differently.

The shipped default stays `potion-base-8M` because the trade is real, not because the margin is thin: the alternative reliably wins paraphrase, reliably loses multi-hop, more often than not loses subagent-only, and buys nothing on exact-identifier recall while regressing its ranking. Switching would also be a breaking index-format change — `potion-retrieval-32M` is 512-dimensional against the current 256, doubling vector storage and forcing a re-index for every existing user. The change itself is cheap in code (`embed.DIMENSION` is the single definition of the vector width). It is deferred pending a larger labelled query set, which is what would tell us whether the trade is worth taking.

### Latency Dominated by Model Load

Real CLI latency to a human is ~500 ms: **523 ms median**, measured as a subprocess (n=12, 2026-07-27, populated real-corpus index; `tests/test_perf.py`). The dominant component is cold model load at ~324 ms, which a fresh process pays on every invocation; CLI startup (command-tree resolution, argv parsing, config load, index open) accounts for ~136 ms, and the retrieval legs are sub-millisecond in-process (BM25 ~0.2 ms, cosine ~0.9 ms, both against a 7,582-chunk index).

Escape hatches exist, none deployed:

- Hand-rolled static embedder (43.6 ms measured, behind the `encode()` interface)
- Resident daemon (disqualifying for a one-shot CLI; viable for V2)
- Smaller static model (a one-line change after measurement)

The shipped guard is a loose 1500 ms regression budget (`tests/test_perf.py`) — ~2.9× the median, deliberately generous so CI variance never trips it. It catches a change of *kind* (a blocking network call, an N+1, a re-download loop), not a change of *degree*. ~500 ms is the real wall a human at a terminal hits, and it is embedded in model load, not retrieval.

### Episode Segmentation Must Be One Pass

Segmentation and extraction must produce episodes and their underlying records **together, in one loop**. Two independently maintained implementations of "where does an episode start and end" will eventually disagree — for example, on a `user` turn whose extracted text is empty — and a count mismatch is not a safe condition to recover from silently: chunking every episode in a batch from the *entire* batch multiplies the index by the episode count (measured 733× on this repo's own main session; ~254× corpus-wide). `episodes.segment_episode_groups()` is the one and only boundary-detection pass, answering both questions at the same decision points. If an output count is implausible, fail visibly rather than falling back.

## Scalability

**Disk footprint scales linearly and predictably. Search time at large scale is an open measurement, not a projected constant.**

At the measured scale — 7,582 chunks, 2026-07-28 — brute-force cosine search is **0.94 ms**, negligible next to the ~324 ms of cold model load that dominates end-to-end latency at any corpus size. Extrapolating that to hundreds of thousands of chunks is not defensible: a naive linear rescale lands an order of magnitude away from the projection an earlier, smaller-scale baseline produced, which is evidence that small-scale linear extrapolation does not hold over this range. What search costs at year-5 scale needs a measurement built at that scale.

Disk, unlike latency, genuinely is linear in chunk count: `vectors.f32` is exactly 1,024 bytes/chunk by construction, and the SQLite/FTS5 side scales with it. Measured total is **3.97 KB/chunk** (30,119,936 bytes for 7,582 chunks in the live index, 2026-07-28 — 1,024 B/chunk `vectors.f32` plus ~2,948.6 B/chunk `index.db`; same constant as the README's Index Size and Growth section). Against the measured growth rate of ~7.9 MB/day per machine (1,140 transcript files over 38.6 days) → ~2.9 GB/year of transcripts → ~141 MB/year of indexable prose → ~141k chunks/year:

- Year 1: 141k chunks, ~560 MB index
- Year 3: 422k chunks, ~1.68 GB index
- Year 5: 703k chunks, ~2.79 GB index

The growth-rate inputs (~7.9 MB/day, ~141k chunks/year) come from that single-machine observation and carry its uncertainty; the per-chunk constant is direct measurement.

Disk growth is also bounded by scope — folder scoping divides the global figure across projects. If size becomes a constraint, the levers in descending value are int8 quantization (4× smaller; measured slower only for implementation reasons), explicit retention policies, and dimension truncation (requires re-embedding, supported by the existing `revectorize` path).

## Three Surfaces, One Library

CLI, MCP server, and skill wrapper all reach `ssgrep.api` — the only entry point — so behavior cannot drift between surfaces. The skill wrapper ships as a template at `src/ssgrep/templates/claude/skills/ssgrep/SKILL.md`, with a matching slash command at `src/ssgrep/templates/claude/commands/ssgrep/search.md`. `ssgrep init` installs both into the user's Claude Code config (`$CLAUDE_CONFIG_DIR` or `~/.claude/`), so they are real deliverables shipped in the wheel. The markdown drives `ssgrep search --json`, inheriting the CLI's contract rather than defining a second one. `ssgrep init` also installs the size-guard hook wrapper `scripts/check-file-size.py` (paired with the checker `scripts/check_file_size.py`), and `cli/commands/hooks_settings.py` owns the settings-file surgery that registers hooks.

**MCP deliberately small:** exactly three tools — `search_sessions`, `show_session`, `index_status` — because every tool definition is resident in the caller's context for the whole conversation. Measured cost: ~324 tokens (chars/4) for the three combined, comfortably under the ~1,000-token ceiling. That figure is measured against the real stdio `tools/list` response — the full payload a client receives, including `outputSchema`, `annotations`, and `_meta` — not a hand-built subset of `name`/`description`/`inputSchema`, which understates the true cost. Input schemas are flat; framing is stated once at server level, not per tool.

**CLI adapter boundary:** no framework types (`Argument`, `Option`, `Menu`, `Prompt`, `Confirm`) appear outside `cli/`. Swapping to Typer or Click would mean rewriting a handful of thin files.

## Constraints That Shaped the Design

- **Never invoke `ssgrep` bare.** A stray `/opt/homebrew/bin/ssgrep` left by a previous install shadows the venv. Use an absolute path, or the `get_ssgrep_binary()` helper in tests.
- **Package naming must be exact.** Distribution name, importable package, and console script must all be `ssgrep`; otherwise `usecli`'s command discovery fails silently and the commands vanish.
- **Transcripts are never written to.** Indexing is a one-way read; nothing is transformed in place.
- **`.ssgrep/` is created 0700 and gitignored.** No network after the one-time model download. Privacy is asserted by tests.

## For the Next Maintainer

- Index content is a pure derivative of transcripts — always rebuildable — **for any session whose source transcript still exists on disk**. `rm -rf .ssgrep/` recovers everything from those transcripts. But tombstoned sessions (see "Archive, Not Cache") no longer exist on disk, and for them the index is the **only** copy. Before deleting `.ssgrep/`, run `ssgrep status` and check `tombstoned_source_count` and `tombstoned_chunk_count`; if either is nonzero, back up `.ssgrep/` first or that history is gone permanently.
- The hard parts are mutation handling (format changes, file deletions) and two-store commit atomicity. Everything else is straightforward.
- The evaluation harness is the source of truth for retrieval quality. The measured numbers (87.9% recall@10, MRR ≈0.63 as of 2026-08-07, n=66 queries, artifact `eval/results/baseline_n66_2026-08-07b.json`, with a ±0.012 corpus-drift band checked by `tests/test_published_numbers.py`) are a contract, not tuning knobs. MRR is a snapshot against a corpus of a given size and it moves as the corpus changes — but not monotonically: 152 same-config re-measurements found overall MRR bouncing between 0.5396 and 0.5605 with no reliable directional relationship to chunk count. A fresh run producing a different MRR, higher or lower, is expected; re-run a few times on the same corpus state before concluding anything moved. Recall@10 moves less often because it is a coarse step function (only moving when a rank crosses the top-10 boundary), not because it is protected from corpus growth — per-class recall (n=12 per class) moves in 8.3-point steps, so a single query crossing the top-10 boundary shifts a class figure by a full step between runs.
- If you change the embedding model, run `revectorize` (not re-index) to re-embed from existing parsed chunks without re-reading transcripts.
- Watch the `cwd` projection and episode segmentation closely if you touch discovery or parsing. Both are easy to break silently.
