# Retrieval evaluation benchmark — detailed notes

The operational depth behind [README.md](README.md). The quick start, uv
command list, and pinned metric definitions live there; this file holds the
nuance: the runner pipeline, the exp-log2 rationale, dataset versioning, the
arm registry, gate methodology, provenance fields, and cost notes.

## What the runner does

1. **Preflight.** Loads `manifest.json`, verifies the sha256 of every listed
   file, refuses on any mismatch or any unlisted file (tampered artifact),
   and after ingestion verifies every graded qrel episode survived.
2. **Ingest.** Points the five runtime env overrides at the dataset's
   `transcripts/` and `opencode.db`, runs the real indexer under a private
   `SSGREP_DATA_DIR`, and builds the vector index. With `--no-rebuild` the
   index is trusted as-is and nothing is written.
3. **Rank.** Per query: production final ranking (limit 10), prefetch episode
   ranking (depth 100, pool 800), and prefetch chunk ranking (depth 100).
   The brute-force reference arm substitutes an exhaustive MaxSim scan.
4. **Measure.** The pinned metric suite over the final ranking, plus prefetch
   recall at both levels (episode + chunk, each at @10/@50/@100), plus
   operational numbers (index size, warm latency p50/p95, build seconds,
   token vector count).
5. **Report.** A versioned JSON payload (schema `1.0`) with summary slices,
   per-query records, metric definitions, gates, reference arms, and
   provenance.

## Metric-run depth and prefetch recall

The summary's `r@10`/`r@50`/`r@100` are computed over the **metric run**,
which is the production top-10 page (`FINAL_LIMIT=10`) extended with the
deep-prefetch pool to the configured `--limit` depth (default 100,
`METRIC_LIMIT_DEFAULT`). The extension appends prefetch episodes not already
on the page (deduped by episode id, prefetch order) until the run reaches
`--limit` entries or the pool is exhausted, and reassigns order-preserving
scores so the metric engine sorts exactly by that sequence. With the default
depth, `r@50` and `r@100` are therefore **real deep-recall numbers**, not
equal to `r@10` by construction. Passing `--limit 10` reproduces the legacy
truncated behavior (the summary metrics see only the 10-doc page).

The deep signal is additionally reported by the `prefetch_*` summary keys,
computed from the per-query prefetch records rather than from the metric run:

- `prefetch_r@10` / `prefetch_r@50` / `prefetch_r@100` — episode-level recall
  over the 800-chunk prefetch pool (`POOL_DEPTH=800`) rolled up to episodes,
  at depths 10/50/100.
- `prefetch_chunk_r@100` — chunk-level recall at depth 100
  (`CHUNK_DEPTH=100`): a chunk is relevant iff its episode is grade>0.

All four are group means over the slice's queries. Queries with no relevant
episodes (`None`) are excluded from the mean, matching the engine's None
convention (same as the nDCG/RR means); `0.0` is a legitimate value and is
kept. Per-query detail lives under `per_query[].prefetch_episode` and
`per_query[].prefetch_chunk` in every payload, and the `metric_definitions`
block documents the `prefetch` entry alongside the pinned ir_measures
formulations. The payload's top-level `metric_limit` records the effective
clamped depth used for the summary metrics.

When comparing runs, read `prefetch_r@100` / `prefetch_chunk_r@100` for the
raw deep-pool recall; the plain `r@50`/`r@100` are only truncated to the
final 10 when `--limit` is below 50.

## Why exp-log2

The exp-log2 formulation (`2^rel - 1` gain, `log2(rank + 1)` discount) is the
standard graded-relevance nDCG. The exponential gain means a grade-3 episode
at rank 1 is worth far more than three grade-1 episodes scattered through the
list, which matches how retrieval quality is actually consumed: the top of
the final ranking is what a user sees. The linear-gain variant (`rel` gain)
treats a grade-3 hit as only three times a grade-1 hit, which under-rewards
getting the best episodes to the top. The choice is recorded in every payload
so a reader never has to guess which variant produced the number.

## Dataset versioning

`eval/dataset/v1/` is a **frozen, immutable artifact**. It contains:

- `manifest.json` — sha256 of every file, per-runtime session/episode counts,
  generation-config digest, profile-census digest, split sizes, version.
- `queries.jsonl` — 602 labelled queries across the 7 classes, each with
  targets, hard negatives, and a frozen 80/20 train/holdout split.
- `qrels.tsv` (+ `qrels/train.tsv`, `qrels/holdout.tsv`) — graded 0-3
  relevance, BEIR format.
- `corpus.jsonl` — BEIR interchange export.
- `transcripts/<runtime>/` — the five emitters' native-format outputs.
- `opencode.db` — the opencode SQLite emitter output.

**Never edit a frozen dataset.** A fix or regeneration means a new directory:

1. Run the generation pipeline: `eval/datasetgen/workflow.py` (sessions),
   the five emitters, `generate_queries.py`, `judge_qrels.py`.
2. Freeze the result as `eval/dataset/v2/` with a fresh `manifest.json`
   (sha256 of every file, counts, digests, split sizes, `version: "v2"`).
3. Verify: `python -c "from eval.datasetgen.freeze import verify_manifest; verify_manifest('eval/dataset/v2')"`.
4. Commit. The runner refuses any dataset whose manifest does not match its
   files byte-for-byte.

The runner resolves `--dataset v1` to `eval/dataset/v1`; `--dataset-dir PATH`
points at an explicit directory instead.

## Arm registry

An arm is a named retrieval configuration (`eval/arms.py`). Env-knob arms
pin `SSGREP_*` overrides verbatim around the whole run (ingestion AND
ranking); the production modules own truthy/clamp semantics, so the runner
and the measured code can never disagree about what was measured.

| Arm | Configuration | Purpose |
|---|---|---|
| `default` | every knob at its shipped default | the production baseline |
| `two_stage_on` | `SSGREP_TWO_STAGE=on` | two-stage prefiltered search |
| `oversample_1` | `SSGREP_OVERSAMPLE=1` | candidate pool at its minimum |
| `nprobes_full` | `SSGREP_NPROBES=512` | every IVF partition probed |
| `brute_force_reference` | computed, not env | exhaustive MaxSim scan per query; numbers land in `reference_arms`, not the summary |

`brute_force_reference` is the recall ceiling and index-sanity check, not an
NDCG oracle. Its `ann_recall100_ratio` (mean engine prefetch r@100 divided by
brute-force r@100) should be at or above ~0.9 on the v1 corpus; a drop below
that means the ANN index is losing recall and needs investigation.

## Gate methodology

Gates derive from a baseline payload when `--baseline` is given:

| Gate | Formula |
|---|---|
| `ndcg@10_floor` | baseline `ndcg@10` - 0.01 |
| `mrr_floor` | baseline `rr@10` - 0.01 |
| `r@50_floor` | baseline `r@50` - 0.01 |
| `p95_ceiling` | 1.2 x baseline `latency_p95_ms` |
| `size_ceiling` | 0.5 x baseline `index_size_bytes` |

The payload's `gates.check` records every comparison with its value, floor or
ceiling, and a `pass` of `true`/`false`/`null` (null means not comparable,
e.g. the baseline lacked the field). `all_pass` is true unless a comparison
is explicitly false. The reference payload is a local default-arm run
(`current_20260823T212203Z.json`; not kept in the repository);
`eval/compare.py` renders a delta table between two payloads.

## Provenance fields

Every payload carries a `provenance` block (`eval/provenance.py`), read at
runtime so the file records what the code actually held, not what its last
commit said:

- `date`, `task`, `label` — what this run was and when.
- `index_stats` — session/episode/chunk counts, embedding model id, vector
  dimension. Pins the corpus.
- `duplication_check` — total vs distinct chunk texts, duplication factor,
  subagent share. This is the measurement that first surfaced the
  254x-duplicate-chunk indexer defect; every result file is checkable
  against whether that defect, or a regression of it, was present.
- `git` — commit sha plus a dirty-tree flag. A dirty tree means the sha alone
  cannot reproduce the run; the flag makes that explicit.
- `tuning_constants` — every retrieval-affecting value at its effective
  clamped/truthy value: `MULTI_CHUNK_EVIDENCE_*`, `CHUNK_TOKEN_BUDGET`,
  `CHUNK_TOKEN_OVERLAP`, embed model/revision/dimension, and all `SSGREP_*`
  knobs (`POOL_FACTOR`, `CHUNK_OVERLAP`, `PQ_BITS`, `REFINE_FACTOR`,
  `NPROBES`, `OVERSAMPLE`, `TWO_STAGE`, `TWO_STAGE_CANDIDATES`).
- `library_versions` — installed `lancedb`, `pylate`, `ir-measures`, and
  `pytrec-eval-terrier` versions. Size and latency numbers are only
  comparable across runs with the same engine versions.

## Cost notes

The one-time dataset generation and judge stages are the only paid steps;
running the benchmark itself is local and free.

- **Generation** (`eval/datasetgen/workflow.py`): NeMo Data Designer with a
  cost guardrail that logs token usage per stage and aborts before a full run
  if the projected cost exceeds the cap.
- **Judge** (`eval/datasetgen/judge_qrels.py`): LLM-as-judge grades each
  (query, episode-excerpt) pair 0-3 with a fixed rubric. The default cap is
  $50 one-time; token accounting runs before any batch submission and the
  projected cost is recorded in `judge_stats.json`. The judge never sees
  grounding labels (bias guard), and disagreements with grounding seeds are
  logged, never silently merged.

## Results

`eval/results/` is not committed; every run's output is local. The
reference figures cited in `eval/README.md` and `docs/retrieval.md` come from
a local default-arm run (`current_20260823T212203Z.json`). Historical sweep outputs are not kept in the repository; the figures
`docs/architecture.md` quotes from them are recorded there as design rationale.