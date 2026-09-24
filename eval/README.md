# Retrieval evaluation benchmark

The standardized way to measure ssgrep retrieval quality. A frozen synthetic
benchmark (`eval/dataset/v2/`) exercises the real production pipeline end to
end, and a pinned metric suite reports nDCG, Recall, MRR, and Precision with
full provenance on every run.

## Quick start

The `poe` tasks wrap every runner entry point. They live in `pyproject.toml`
(`[tool.poe.tasks]`; `uv run poe eval-TAB` lists them). `eval-run`,
`eval-baseline`, and `eval-reference` each write their own timestamped file
under `eval/results/` per run:

```bash
uv run poe eval-run          # full default-arm eval -> eval/results/current_<UTC-ts>.json
uv run poe eval-smoke        # first-5-queries smoke -> eval/results/current_<UTC-ts>.json
uv run poe eval-compare eval/results/baseline_v2.json eval/results/current_<latest>.json --gates
uv run poe eval-verify       # verify frozen v2 manifest
uv run poe eval-test         # eval test suite
```

Run output is never committed: `.gitignore` excludes everything under
`eval/results/`. The reference figures in this file and in `docs/retrieval.md`
come from a local default-arm run (`current_20260823T212203Z.json`); rerun
`poe eval-run` to reproduce them.

The raw runner also accepts `--limit N` to set the metric-run depth (default
100; the summary `r@50`/`r@100` are computed at this depth). `--limit 10`
reproduces the legacy truncated behavior where the summary metrics see only
the 10-doc page:

```bash
# Evaluate with a 100-deep metric run (default)
uv run python -m eval.run_eval --dataset v2 --arm default

# Reproduce the legacy truncated summary metrics (10-doc page only)
uv run python -m eval.run_eval --dataset v2 --arm default --limit 10
```

Dataset generation is a one-time, paid pipeline: `poe eval-gen-preview`
(free, 8 records) to sanity-check the config, then `poe eval-gen-sessions`
(paid; needs `OPENROUTER_API_KEY` or `OPENAI_API_KEY`), then
`poe eval-gen-queries`, `poe eval-judge` (paid, $50 cap), then
`poe eval-freeze` → `eval/dataset/v2/`, `poe eval-verify`, and commit.
`poe eval-ingest` pre-builds the private index for judge/baseline/
`--no-rebuild` runs.

Exit codes: `0` success, `1` preflight failure (tampered dataset, missing
qrel targets, unreadable baseline), `2` usage error (unknown arm, missing
`--dataset`).

## Pinned metric definitions

All metrics are computed by `ir_measures` (0.4.3) through `eval/metrics.py`;
nothing re-implements metric math. The formulations are pinned in
`METRIC_DEFINITIONS` and stamped into every payload under
`metric_definitions`, so numbers stay comparable forever.

The unit of retrieval is the **episode**. Relevance is graded 0-3 per episode
in the dataset's qrels. Chunk-level recall is reported as a diagnostic only.

| Metric | Cutoffs | When to Use | Key Insight | Formula | Notes |
|---|---|---|---|---|---|
| nDCG | @5, @10 | Final ranking (k=5, 10) | Are most relevant results ranked highest? | `gain = 2**rel - 1`, `discount = log2(rank + 1)`, normalized by ideal DCG capped at k | exp-log2 formulation, pinned explicitly (`dcg='exp-log2'`), never the provider default |
| Recall@k | @10, @50, @100 | Prefetch stage (k=50, 100) | Did we capture relevant candidates? | `\|grade>0 episodes in top-k\| / \|all grade>0 episodes\|` | TREC convention: fraction of relevant retrieved, not Success@k |
| MRR | RR@10 primary, RR@50 also emitted | Single-answer retrieval | How quickly do we find THE right answer? | reciprocal rank of the first grade>0 episode | Only the first relevant episode counts; a second relevant hit does not improve the score. Treat it as "how fast is the best answer," not overall ranking quality — use nDCG when multiple relevant episodes matter |
| Precision@k | @5, @10 | Result page quality | How many of the shown results are relevant? | `grade>0 episodes in top-k / k` | trec_eval semantics: divides by k even when the run depth is less than k |

Edge-case rules (all verified against ir_measures 0.4.3):

- Queries with **no grade>0 qrel** are excluded from the nDCG and RR means
  (trec_eval reports them as 0.0, which would pollute the mean). They stay in
  the P/R means (0.0 is correct there). The count is surfaced as
  `excluded_query_count`.
- A run whose scores are all zero is flagged `empty_run: true` and metrics
  short-circuit to 0.0 (feeding an all-zero run to ir_measures would count
  the zero-scored docs as retrieved).
- k beyond the returned depth is fine: metrics compute over the available
  depth, and P@k still divides by k.
- The nDCG provider (gdeval) requires numeric topic ids; the engine maps
  query ids to sorted integers internally and maps them back in the report.

## Current results

Reference run: **arm `default`**, frozen v2 dataset, evaluated 2026-08-23
at `metric_limit=100`, 602 labelled queries (86 per class),
483 train / 119 holdout. Payload: `current_20260823T212203Z.json` (local run output; not kept in the repository).

### Overall

The summary `r@10`/`r@50`/`r@100` are computed over the metric run extended
to `metric_limit=100` (production top-10 page plus the deep-prefetch pool;
see [README.details.md](README.details.md) for the mechanics), so `r@50`
and `r@100` are real deep-recall numbers, not truncated equals of `r@10`.

| Slice | n | nDCG@10 | nDCG@5 | r@10 | r@50 | r@100 | P@10 | P@5 | RR@10 |
|---|---|---|---|---|---|---|---|---|---|
| **overall** | 602 | 0.880 | 0.866 | 0.932 | 0.982 | 0.985 | 0.119 | 0.225 | 0.903 |
| train | 483 | 0.873 | 0.857 | 0.930 | 0.977 | 0.981 | 0.121 | 0.227 | 0.895 |
| holdout | 119 | 0.912 | 0.905 | 0.941 | 1.000 | 1.000 | 0.110 | 0.213 | 0.935 |

The holdout slice tracks or slightly exceeds train (nDCG@10 0.912 vs 0.873,
r@10 0.941 vs 0.930), so the tuned pipeline does not overfit the train
queries.

### By query class

| Class | nDCG@10 | r@10 | r@50 | r@100 | RR@10 |
|---|---|---|---|---|---|
| exact-identifier | 0.989 | 1.000 | 1.000 | 1.000 | 0.985 |
| decision-rationale | 0.958 | 0.977 | 0.977 | 0.977 | 0.952 |
| paraphrase | 0.927 | 0.965 | 0.977 | 0.977 | 0.914 |
| multi-hop | 0.886 | 0.924 | 0.965 | 0.977 | 0.952 |
| error-string | 0.887 | 0.953 | 0.988 | 0.988 | 0.863 |
| tool-failure-recovery | 0.827 | 0.942 | 0.977 | 0.977 | 0.790 |
| cross-runtime/project-scoped | 0.690 | 0.762 | 0.988 | 1.000 | 0.866 |

Reading the class rows:

- **exact-identifier** leads nDCG@10 (0.989) and reaches perfect recall at
  every depth (r@10 = r@50 = r@100 = 1.000), as expected for verbatim
  identifier matches; its RR@10 (0.985) is also the class best.
- **multi-hop** has strong deep recall (r@100 = 0.977) but the softest r@10
  among the mid classes (0.924) — the connecting episodes are retrieved but
  not always surfaced early; nDCG@10 (0.886) still holds up because the right
  episode ranks well.
- **error-string** is the weakest on final-page quality (RR@10 0.863, nDCG@10
  0.887) even though its deep recall is strong (r@100 = 0.988): error-text
  chunks are hard to match, so the relevant chunks reach the top-10 less
  consistently and are recovered mostly by the deeper pool.
- **cross-runtime/project-scoped** has the lowest r@10 (0.762) and nDCG@10
  (0.690), but its r@50/r@100 (0.988/1.000) show the deep pool recovers the
  targets almost completely — the class hardest to pin down from the final
  page.
- **tool-failure-recovery** has the weakest RR@10 (0.790): the first relevant
  episode is found less reliably even though chunk recall stays strong.

### Operational

| Measure | Value |
|---|---|
| Warm latency p50 / p95 | 669.1 ms / 741.0 ms |
| Index size | 34.2 MB (32.6 MiB) |
| Index build | 13.8 s |
| Token vectors | 68,169 |
| Corpus | 404 sessions / 888 episodes / 1,918 chunks |

Provenance pinned: `answerai-colbert-small-v1` (96-d), ir-measures 0.4.3,
lancedb 0.37.1, pylate 1.6.0. The payload's `dataset_version` field reads
`v1` (the frozen dataset's manifest `version` was never bumped) even though
this run used the v2 dataset directory.

## Deep dive

Runner internals (preflight/ingest/rank/measure/report), the metric-run
depth and prefetch-recall mechanics, the exp-log2 rationale, dataset
versioning, the arm registry, gate methodology, provenance fields, and
cost notes live in [README.details.md](README.details.md).
