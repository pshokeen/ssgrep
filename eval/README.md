# Retrieval evaluation harness

Everything in this document is measured against the real corpus at
`~/.claude/projects`, never against fixtures. It backs the retrieval numbers
`README.md` publishes and `tests/test_published_numbers.py` guards.

Read the methodology section before trusting any number here: the failure mode
this harness exists to avoid — labelling targets by running the current search
and grading it against itself — produces excellent numbers that mean nothing.

## Files

| File | What it does |
|---|---|
| `queries.jsonl` | The committed, labelled query set (66 queries; ≥30 required). |
| `build_labels.py` | Builds `queries.jsonl` from independently-sourced anchors. Re-running it regenerates the file against *today's* corpus. |
| `harness.py` | Runs the query set against a private index and reports recall@10 / MRR, overall, per class, and subagent-only. |
| `model_ab.py` | Open question 1: `potion-base-8M` vs `potion-retrieval-32M`. |
| `chunking_ab.py` | Open question 2: uniform vs per-turn chunking. |
| `results/*.json` | Committed run outputs backing every number below. Each carries its own `provenance` (git commit, tuning constants, `index_stats`, `duplication_check`) *except* the four `*_2026-08-06*.json` files, written before `harness.py`'s `main()` wired up `provenance.build_provenance()` (now fixed for future runs; these four were not retroactively backfilled — see their absent `provenance` key). |

## Methodology — how circularity is avoided

The label set is the whole experiment. `build_labels.py` derives every target
**two steps removed** from ranking:

1. **Segmentation, not scoring.** The corpus is walked with ssgrep's own
   parser (`discovery.discover_sessions`, `records.read_records`,
   `episodes.segment_episodes`) to turn bytes into `(session, episode)` units.
   That is deterministic structural work. The thing under test is the *scored*
   retriever — `chunker.py` → five legs (AND-BM25, vector cosine, OR-BM25,
   trigram-subword BM25, phrase-proximity) → weighted RRF → main-session
   boost, in `search/` — which is never used to build labels.

2. **Anchors, not queries, decide the target.** Every anchor is a literal
   string copied by hand from the project's own design and handoff notes, and
   is located by plain Python substring containment over
   `episode.prompt_text + episode.response_text`. Containment does not rank,
   chunk, embed, or touch FTS5 or model2vec, so it cannot agree with the
   retriever by construction.

3. **Query text is composed independently of the anchor.** `exact-identifier`
   and `error-string` queries deliberately reuse verbatim tokens — that *is*
   the property those classes test. `paraphrase` and `multi-hop` queries are
   phrased as a person would ask and are never copied from the anchor.
   `tests/test_eval_harness.py:413`
   (`test_paraphrase_and_multihop_queries_never_contain_their_anchor_verbatim`)
   fails if any paraphrase/multi-hop query contains its own anchor text.

4. **Multiple correct answers are allowed, and disclosed.** Later sessions
   routinely recap earlier findings, so `target_episode_ids` is a *set*:
   recall@10 counts a hit if any member appears in the top 10, and MRR uses
   the best (lowest) rank among them — the standard treatment for
   multi-relevant-document evaluation.

5. **Anchors that match too much are rejected.** `build_labels.py:423` caps
   every anchor at `ANCHOR_MATCH_RATIO * corpus_size` matching episodes
   (`ANCHOR_MATCH_RATIO = 0.01`, i.e. 1% of the corpus). An anchor spanning a
   large fraction of episodes tests retrieval of a *topic*, not an episode.
   The proportional cap scales with corpus growth.

## How to read these numbers

**The corpus is live.** The project being evaluated and the corpus it is
evaluated against are the same body of sessions, and that corpus is actively
written to while runs execute. Episode counts differ between artifacts
(3,360 in `boost_sweep_2026-07-27.json`, 3,530 in
`chunking_ab_2026-07-27.json`, 3,692 in `baseline_boost_0.0_2026-07-27.json`,
3,977 in `model_ab_2026-07-27.json`). Absolute values are snapshots; A/B
comparisons whose two arms were built minutes apart are more trustworthy in
their *relative* direction than in their absolute values.

**The pipeline itself is deterministic.** Indexing, embedding, BM25, and
ranking contain no randomness: a 152-run study across 11 corpus states
included 60 independent full rebuilds at one unchanged corpus state
(`chunk_count` 6,644) that were bit-identical on every metric, overall and
per class (`noise_floor_2026-07-27.json`,
`provenance.corpus_determinism_proof`). Every difference between two runs
therefore comes from the corpus differing between them, not from measurement
noise. The same property is visible directly in
`matched_pair_gap_study_2026-07-28.json`'s raw records: each distinct
`chunk_count` maps to exactly one MRR value across all 36 runs.

**Resolution bands to judge any comparison against:**

| metric | band | source |
|---|---:|---|
| overall recall@10 (n=66) | 1.52 pts (one query) | step function, 1/66 |
| per-class recall@10 (n=12) | **8.3 pts (one query)** | step function, 1/12 |
| subagent-only recall@10 (n=8) | 12.5 pts (one query) | step function, 1/8 |
| overall MRR | **±0.012** | `noise_floor_2026-07-27.json`, `floors.mrr.recommended_bound` (measured drift between two same-config runs ~9 minutes apart; 0.0162 was observed over ~30 minutes) |

Two consequences worth stating plainly. First, **n=12 per class means one
query is 8.3 points** (and one subagent-only query is 12.5 points): a
per-class recall figure, or a per-class "tie" between two
configurations, is only as reliable as that resolution supports, and can move
between regenerations with no code change at all. No per-class conclusion here
rests on a single snapshot. Second, **MRR is cited to two decimals as `≈X`**;
more decimals would name a value more precisely than "which corpus state this
happened to be" can be pinned down.

**MRR varies with which content is present, not how much.** In
`matched_pair_gap_study_2026-07-28.json`, the shipped-config arm's overall MRR
across 12 consecutive corpus states (6,774–6,787 chunks) moves up on 5 steps,
down on 5, and is flat on 1, spanning 0.5369–0.5582. Do not read a trend from
corpus size.

**Duplication is ordinary, not defective.** The `duplication_check` block in
every committed artifact measures ~1.08× (e.g. 6,536 chunks / 6,059 distinct
texts in `baseline_boost_0.0_2026-07-27.json`) — restated content, chiefly
subagent spawn context. Subagent-authored chunks are ~72% of indexed chunks
(`duplication_check.subagent_chunk_share`, 0.718–0.724 across the committed
artifacts).

## Query set composition

66 queries across the four required classes, 11 of them (17%) answerable
**only** from a subagent transcript.

```
exact-identifier: 14   (verbatim code tokens — e.g. `reciprocal_rank_fusion RRF_K`)
error-string:     13   (verbatim error/log text)
paraphrase:       20   (natural-language asks, phrased differently than any anchor)
multi-hop:        19   (requires connecting two facts stated together in one episode)
subagent-only:    11   (target lives only in a subagent transcript; cuts across the four classes)
```

The 2026-08-07 extension (48 → 66) merged the 18 queries that had been held
out of every tuning decision ever made against this harness: the 6-query
holdout written during the 2026-08-06 label repair, plus 12 queries
blind-composed on 2026-08-07 *after* that day's retrieval tuning was frozen
(systematic every-7th sampling of substantial-prose episodes; queries
authored before any retrieval run; anchors located by substring containment
under the 1% cap — the same `build_labels.py` methodology). Merging them
deliberately skews the set toward the two hardest classes
(paraphrase/multi-hop now 39 of 66) and burns them as holdout — there is
currently **no** untouched holdout; the next blind extension should recreate
one.

This set is a repair-and-extension of the original 36-query set the July
studies below were run on. Claude Code's transcript cleanup deleted the target
episodes of 13 of the original 36 queries (all subagent-only), so those labels
were repaired with new independently-sourced anchors (same methodology), and
the set was then extended 36 → 48 with **12 blind-composed queries** written
*after* the five-leg ranking shipped, as a generalization check — they hit
10/12. See "Update 2026-08-06" below for the full account.

## Result 1 — the main-session boost, and the committed baseline

`MAIN_SESSION_BOOST` is measured, not estimated.
**`eval/results/boost_sweep_2026-07-27.json`** — recall@10 / MRR at seventeen
boost values (0.0 through 1.0), same 36 queries, one index built once and
swept in-process, never rebuilt between values:

| `main_session_boost` | overall recall@10 | overall MRR | subagent-only recall@10 |
|---:|---:|---:|---:|
| **0.0 (shipped)** | **77.8%** | **≈0.56** | **69.2%** |
| 0.001 | 75.0% | ≈0.51 | 61.5% |
| 0.005 | 66.7% | ≈0.48 | 46.2% |
| 0.01 | 58.3% | ≈0.46 | 30.8% |
| 0.02 | 50.0% | ≈0.38 | 7.7% |
| 0.03 – 1.0 | 44.4% | ≈0.36 | 0.0% |

Because every row comes from one index swept in-process, the differences
*between* rows are exact and not subject to the ±0.012 cross-run band; only
the absolute values would move on a fresh rebuild.

recall@10 and MRR fall monotonically (non-strictly) as the boost rises, across
all 17 swept values, with zero exceptions — no positive value, however small
(0.001 tested), beats disabling the preference outright. This is a scale
mismatch, not gentle tie-breaking: measured fused/rolled-up episode scores in
this corpus span `0.0164`–`0.0328` (see the relevance-floor section), so a
`0.01` boost is close to an entire rank-1 vote and pushes irrelevant
main-session episodes above relevant subagent ones. The largest damage lands
on subagent-only queries — the class the boost exists to *not* penalize.

`search/__init__.py:93` ships `MAIN_SESSION_BOOST = 0.0`, pinned by
`tests/test_search.py::TestMainSessionBoost::test_shipped_default_matches_measured_optimum`,
which fails if the shipped constant and the measured optimum disagree. A
change to this constant should come with a fresh harness run justifying the
new value.

**The committed baseline** (`eval/results/baseline_boost_0.0_2026-07-27.json`,
shipped uniform chunking, `main_session_boost=0.0`, 228 sessions / 3,692
episodes / 6,536 chunks at measurement time):

| | value |
|---|---:|
| overall recall@10 | 77.8% (28/36) |
| overall MRR | ≈0.54 |
| exact-identifier recall@10 / MRR | 100% / ≈0.86 |
| error-string recall@10 / MRR | 88.9% / ≈0.83 |
| multi-hop recall@10 / MRR | 66.7% / ≈0.34 |
| paraphrase recall@10 / MRR | 55.6% / ≈0.13 |
| subagent-only recall@10 / MRR | 69.2% / ≈0.36 |

`tests/test_published_numbers.py` checks `README.md`'s published Overall row
against this exact artifact: recall@10 within 1/n (n read from the artifact,
currently 1/36 = 0.0278) and MRR within ±0.012, plus exact equality between
`search.MAIN_SESSION_BOOST` and the artifact's recorded boost.
(2026-08-06: the guarded artifact became
`results/baseline_boost_0.0_2026-08-06d.json`, n=48, tolerance 1/48 ≈ 0.0208;
2026-08-07: it is now `results/baseline_n66_2026-08-07b.json`, n=66,
tolerance 1/66 ≈ 0.0152 — see the Update sections below.)

For contrast, `eval/results/baseline_boost_0.01_2026-07-27.json` (same index,
boost 0.01) measures 58.3% recall@10 / ≈0.46 MRR overall, with subagent-only
recall@10 at 30.8%, multi-hop at 22.2%, and paraphrase at 22.2%.

## Result 2 — open question 1: does `potion-retrieval-32M` justify itself?

`eval/results/model_ab_2026-07-27.json`, `main_session_boost=0.0`, same 36
queries, two private indexes built from the same corpus scope minutes apart
(3,977 episodes / 7,020 chunks each). `model_ab.py` monkeypatches
`embed.MODEL_ID` / `embed.DIMENSION` (and a dimension-aware
`vectors.validate_alignment`) for the duration of one build and reverts them,
never editing `embed.py`.

**One snapshot, shown as a worked example — not read on its own.** Per-class
cells move between regenerations at 11.1 points per query; the distribution
table below is the actual evidence.

| | `potion-base-8M` (shipped, MTEB Retrieval 31.11) | `potion-retrieval-32M` (MTEB Retrieval 35.06) |
|---|---:|---:|
| dimension | 256 | 512 (measured, not assumed) |
| overall recall@10 / MRR | 77.8% / ≈0.54 | 77.8% / ≈0.55 |
| error-string recall@10 / MRR | 88.9% / ≈0.83 | 88.9% / ≈0.81 |
| exact-identifier recall@10 / MRR | 100% / ≈0.86 | 100% / ≈0.82 |
| multi-hop recall@10 / MRR | 66.7% / ≈0.34 | 55.6% / ≈0.44 |
| paraphrase recall@10 / MRR | 55.6% / ≈0.14 | 66.7% / ≈0.14 |
| subagent-only recall@10 / MRR | 69.2% / ≈0.36 | 61.5% / ≈0.36 |

**The per-class evidence this document stands behind is a distribution.**
`eval/results/matched_pair_gap_study_2026-07-28.json` re-ran this comparison
36 independent times across 12 distinct corpus states (fresh temp indexes each
run, `main_session_boost` verified clean before and after). Counts below are
computed from that file's raw per-run records:

| class | recall@10 (W / T / L, n=36) | MRR (W / T / L, n=36) |
|---|---|---|
| overall | 17 / 19 / 0 | 36 / 0 / 0 |
| error-string | 17 / 19 / 0 | 20 / 8 / 8 |
| exact-identifier | 0 / 36 / 0 | 0 / 0 / **36** |
| multi-hop | 0 / 0 / **36** | 36 / 0 / 0 |
| paraphrase | **36** / 0 / 0 | 35 / 0 / 1 |
| subagent-only | 0 / 17 / **19** | 33 / 0 / 3 |

(W / T / L = `potion-retrieval-32M` ahead of / tied with / behind
`potion-base-8M`.)

**Read this as a trade, not a near-sweep.** Multi-hop recall never ties:
`potion-retrieval-32M` sits at exactly 55.6% and `potion-base-8M` at exactly
66.7% in all 36 runs. Subagent-only's modal outcome is a loss (19 losses, 17
ties, 0 wins). Paraphrase is a clean win (36/36) and exact-identifier MRR a
clean loss (36/36) for zero recall benefit. Error-string and overall recall
split between a win and a tie, never a loss — so any single snapshot showing
either as a win or a tie is an ordinary draw.

Overall-level numbers are the least informative part, because they average
over that trade. Overall MRR is unanimous in direction (36/36 positive,
`gap_all_positive: true`) but its size is not stable: the gap's own
run-to-run spread is 0.043, larger than either arm's individual spread (0.021
and 0.038), so the direction is real and the magnitude is not claimable.

**Cost of the larger model.** Doubling the vector dimension to 512 doubles
per-chunk vector storage. The dimension itself is centralized: `embed.py:25`
defines `DIMENSION = 256` and downstream sites derive from it
(`vectors.py:153` `vec_size // (embed.DIMENSION * 4)`,
`store/generations.py:83` `VECTOR_ROW_BYTES = embed.DIMENSION * 4`,
`store/compaction.py:68` `vector_dimension = gen_store.VECTOR_ROW_BYTES // 4`),
so a swap is `MODEL_ID` + `DIMENSION` plus a re-vectorize of existing indexes.

**The shipped default does not change: `embed.MODEL_ID` stays
`potion-base-8M`.** This is a trade-off call, not a "too thin to tell" one:
`potion-retrieval-32M` reliably wins paraphrase and reliably loses multi-hop,
and more often than not loses subagent-only recall — and subagent transcripts
hold most of this corpus's indexable prose. Revisit with a larger, more
diverse labelled query set (9 per class is coarse), or on a corpus with a
different class mix.

## Result 3 — open question 2: does per-turn indexing beat uniform chunking?

`eval/results/chunking_ab_2026-07-27.json`, `main_session_boost=0.0`, same 36
queries, two private indexes built from the same corpus scope minutes apart
(3,530 episodes each).

| | uniform (shipped) | per-turn (reference-design rule) |
|---|---:|---:|
| chunk count | 6,259 | 6,250 (~0.14% fewer) |
| overall recall@10 | 77.8% | 77.8% |
| overall MRR | ≈0.54 | ≈0.55 |
| exact-identifier recall@10 / MRR | 100% / ≈0.86 | 100% / ≈0.86 |
| error-string recall@10 / MRR | 88.9% / ≈0.83 | 88.9% / ≈0.83 |
| multi-hop recall@10 / MRR | 66.7% / ≈0.34 | 66.7% / ≈0.34 |
| paraphrase recall@10 / MRR | 55.6% / ≈0.13 | 55.6% / ≈0.15 |
| subagent-only recall@10 / MRR | 69.2% / ≈0.36 | 69.2% / ≈0.36 |

The committed overall MRR gap is 0.0032 — inside the ±0.012 band, and 0.09× the
gap's own measured spread of 0.036 (`noise_floor_2026-07-27.json`,
`corpus_sensitivity.committed_chunking_ab_gap`).

**Methodology for this comparison.** `episodes.Episode` carries only flattened
`prompt_text` / `response_text`, so turn boundaries are gone by the time
`chunker.py` sees an episode. `chunking_ab.py` re-walks the same raw records
with a mirror of `episodes.segment_episodes`'s boundary logic (compaction
boundary, user record, EOF flush), preserving each turn. Before building any
index, `_verify_alignment()` asserts — for every episode in the corpus (3,530
in the committed run, recorded as `episodes_verified`) — that rejoining the
mirrored turns reproduces byte-identical `prompt_text` / `response_text`
against real `episodes.segment_episodes` output and that episode numbering
lines up 1:1. That mirror is also covered by
`tests/test_eval_harness.py:463`
(`test_per_turn_segmentation_mirror_matches_real_segmentation`). Per-turn
chunking is wired in by monkeypatching `ssgrep.chunker.chunk_episode` for one
private-index build, reverted in a `finally`.

**Answer: no.** `matched_pair_gap_study_2026-07-28.json` re-ran this
comparison 36 times across 15 distinct corpus states. Overall recall@10 and
all five per-class recall@10 comparisons came back a zero-exception tie
(0 wins / 36 ties / 0 losses each), and both arms sat at exactly 77.8% overall
in every run. The overall MRR gap's sign flips across those runs
(`gap_sign_flips: true`, range −0.015 to +0.021), so no stable effect size
exists.

The one durable sub-signal: **paraphrase MRR, where per-turn is ahead in 34 of
36 runs** (2 ties, 0 losses). It is small and does not change the headline.

**Recommendation: no chunking change.** Uniform stays shipped. Adopting
per-turn chunking would add the reference design's drop-short-turns rule
(`MIN_TURN_CHARS`) and this experiment's monkeypatch-based wiring for no
measured recall benefit.

## Relevance floor — evidence gathered, not implemented

No relevance floor is implemented. What the harness contributes is the score
distribution for whoever picks one, from
`eval/results/baseline_boost_0.0_2026-07-27.json`'s `per_query` array at
`main_session_boost=0.0`:

- Genuine hit scores (`top_result_score` where `rank` is not `None`, n=28)
  span `0.0164`–`0.0328`.
- The 8 queries that miss entirely (`error-07`, `para-03`, `para-05`,
  `para-06`, `para-08`, `multi-02`, `multi-03`, `multi-07`) all top out at
  exactly `0.0164` — the same value as the *bottom* of the genuine-hit range,
  not below it.

**There is no gap in the score distribution a fixed floor could occupy.** A
floor at or below 0.0164 lets every miss through; a floor above it also
suppresses genuine hits sitting at that same value. Picking one now would be
guesswork. Consequently, exit code 3 and the MCP no-match branch — both of
which only execute once a floor exists — remain unimplemented.

One caveat on the label set itself: `error-05`'s query is the gibberish probe
`zzz_nomatch_zzz_qqq`, which now scores `0.0328` at rank 1 with a genuine hit,
because the literal probe string appears verbatim in this self-referential
corpus's own discussion of the eval methodology. It is not a usable
noise/miss example on this corpus; the 8 genuine misses above are.

## Reproducing these numbers

```bash
# baseline at the shipped boost
uv run python -m eval.harness --main-session-boost 0.0 --json-out /tmp/out.json

# sweep across boost values — build the index ONCE via a shared --index-dir,
# then reuse it. --no-rebuild WITHOUT a shared --index-dir mints a fresh empty
# temp directory every iteration and silently pays a full rebuild each time
# (indexer.index()'s _needs_rebuild sees no existing index and builds
# regardless of rebuild=False).
IDXDIR=$(mktemp -d)
first=1
for b in 0.0 0.001 0.005 0.01 0.02 0.05 0.1 0.3 0.5 1.0; do
  if [ "$first" = "1" ]; then
    uv run python -m eval.harness --main-session-boost "$b" --index-dir "$IDXDIR"
    first=0
  else
    uv run python -m eval.harness --main-session-boost "$b" --no-rebuild --index-dir "$IDXDIR"
  fi
done

# model A/B (potion-base-8M vs potion-retrieval-32M)
uv run python -m eval.model_ab --main-session-boost 0.0 --json-out /tmp/model_ab.json

# chunking A/B (uniform vs per-turn)
uv run python -m eval.chunking_ab --main-session-boost 0.0 --json-out /tmp/chunking_ab.json
```

Every run builds a **private** index (via `indexer.index()`'s `index_dir`
override) scoped to this project's real corpus. None write to
`<project_dir>/.ssgrep`, the shared live index other agents and tests depend
on. At the current corpus size a build takes a few seconds; each artifact
records its own corpus size in `provenance.index_stats`, which is the number
to compare against, not another section's snapshot.

## Update 2026-08-06 — label regeneration and the five-leg pipeline

Two things changed after the studies above were committed, so read their
absolute numbers as history:

1. **Claude Code's transcript cleanup deleted the target episodes of 13 of the
   36 committed queries** (all subagent-only), silently decaying the measured
   baseline to 44.4%. The 9 anchor specs whose anchors no longer appeared
   anywhere in the corpus were replaced with new independently-sourced anchors
   (same methodology, composed before any retrieval run) and
   `queries.jsonl` was regenerated against the 2026-08-06 corpus
   (139 sessions / 2,762 episodes / 6,142 chunks).
2. **The retriever grew three support legs**, each measured in with this
   harness: OR-BM25 (weight 0.9), trigram-subword BM25 over the new
   `chunks_fts_tri` table (`TRIGRAM_LEG_WEIGHT = 1.0`, tokens ≥4 chars), and
   a phrase-proximity leg (0.5), fused by weighted RRF (k=60) alongside the
   original AND-BM25 (1.0) and `potion-base-8M` vector (1.0) legs — five legs
   total; episode roll-up gained a bounded non-best-chunk tail (max + 0.2 ×
   next two). The new table bumped `SCHEMA_VERSION` 3 → 4 (one-time rebuild).

Result on the regenerated 36-query set
(`results/baseline_boost_0.0_2026-08-06.json`): overall recall@10 **88.9%**,
MRR **≈0.61**. The set was then extended to **48 queries** (12 per class) —
the twelve new queries composed blind *after* the five-leg ranking shipped,
as a generalization check: they hit 10/12 (exact 3/3, error 3/3, paraphrase
2/3, multi-hop 2/3), consistent with the six-query holdout (5/6) that was
never tuned against. Caveat: `TRIGRAM_LEG_WEIGHT`'s final value (1.0, not
the initial 0.9) was itself re-verified against this same enlarged 48-query
set — see the comment above `TRIGRAM_LEG_WEIGHT` in `search/__init__.py` —
so the 10/12 score is not a fully untouched holdout measurement for that
constant. The six-query holdout above is the only figure here never used in
any tuning decision. The n=48 artifact
(`results/baseline_boost_0.0_2026-08-06d.json`, the artifact
`tests/test_published_numbers.py` guarded until the 2026-08-07 n=66
extension; `...06b.json` predates the
trigram-weight plateau re-centering to 1.0, and `...06c.json` predates a
favorable corpus-growth drift re-measurement the same day) measures overall
recall@10 **87.5%**, MRR **≈0.59**. Per class (n=12 each): exact-identifier
100% / ≈0.79; error-string 100% / ≈0.86; multi-hop 75.0% / ≈0.44; paraphrase
75.0% / ≈0.26; subagent-only (n=8) 100% / ≈0.69. Every miss is a paraphrase
or multi-hop vocabulary-gap query ("cwd" vs "working directories") —
paraphrase ranking remains the open question, unchanged in kind from the
bottom line below. Also measured 2026-08-06: swapping the vector leg for
transformer embedders (`all-MiniLM-L6-v2`, MTEB Retrieval 42.92;
`bge-small-en-v1.5`, ~51.7) adds **zero** recall over the shipped hybrid on
this corpus (39–41/48 vs 41–42/48) — the remaining misses are
world-knowledge vocabulary gaps no offline mechanism bridges.

## Update 2026-08-07 — six-leg pipeline, chunk-window retune, n=66

The 2026-08-07 autoresearch session added a sixth leg (MaxSim late-interaction
rerank over the prior fused ranking's top 250 chunks, weight 1.0 —
`search/rerank.py`) and retuned the chunk window (`CHUNK_TARGET_SIZE`
1200 → 1750, `CHUNK_OVERLAP` 200 → 450; 21% fewer chunks). The label set then
grew 48 → 66 by merging the 18 never-tuned holdout queries (see "Query set
composition" above). The same day, `ROLLUP_TAIL_WEIGHT` was retuned
0.2 → 0.05 after a mechanism diagnosis showed three labelled queries'
target episodes carrying a top-12 fused chunk yet losing the episode-level
ranking to tail-vote accumulators — validated on the live corpus (+3
queries, zero new misses) and on a fresh 10-query blind holdout evaluated
exactly once (recall 10/10 held, MRR improved). Current guarded artifact:
`results/baseline_n66_2026-08-07b.json` — overall recall@10 **87.9%**, MRR
**≈0.63** on the live corpus (158 sessions / 4,331 episodes / 7,348 chunks).

Honest effect-size accounting for that session (full data in
`.auto/log.jsonl` on the session branch): against the pre-session config the
changes measure **+16.8% MRR on the 48 tuning queries** (frozen snapshot),
**+15.2%** on the same queries against the larger live corpus, **+10.6%** on
the pooled n=66 set — but **≈0%** on the 18 never-tuned queries alone.
Recall@10 was identical in *every* one of those comparisons, on every query
set, tuned or untuned: the changes are recall-safe and shrink the index, and
the rerank leg's untuned-data value is one rescued recall query, but the
headline MRR improvement is substantially a property of the tuning set — the
per-query rank churn on fresh queries nets to zero. That is the empirical
case for growing this label set further (and for recreating a fresh blind
holdout, which the n=66 merge burned).

## Bottom line (2026-07-28 — superseded)

For 2026-08-06 numbers, see "Update 2026-08-06" above: the five-leg
hybrid measured overall recall@10 **87.5%**, MRR **≈0.59** on 48 queries
(`results/baseline_boost_0.0_2026-08-06d.json`). The text below is the
2026-07-28 bottom line for the two-leg pipeline and 36-query set, kept as
history.

The shipped hybrid (BM25 + static `potion-base-8M` embeddings, uniform
chunking, `MAIN_SESSION_BOOST = 0.0`) measures overall recall@10 **77.8%**,
MRR **≈0.54** on 36 independently-labelled queries.

The class shape is the expected one for BM25 plus static embeddings:
exact-identifier (100%) and error-string (88.9%) are close to solved;
multi-hop (66.7%) and paraphrase (55.6%) — queries needing genuine semantic
matching or connecting two separately-stated facts — are the weak classes,
and paraphrase is the weakest.

Neither open question changes what ships. `potion-retrieval-32M` is a real
trade (wins paraphrase 36/36, loses multi-hop 36/36, loses subagent-only in
19/36) rather than an upgrade, and this corpus's class mix does not favour it.
Per-turn chunking ties uniform on recall@10 in all 36 matched-pair runs with a
sign-flipping MRR gap. Neither result implicates the storage format or argues
for abandoning static embeddings for multi-vector / late-interaction
retrieval; paraphrase remains the open quality question, and answering it well
needs a larger labelled set than 9 queries per class.
