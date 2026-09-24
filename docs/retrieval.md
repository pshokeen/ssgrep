# Retrieval Pipeline and Results

## Pipeline

Chunks embed through the pinned `lightonai/answerai-colbert-small-v1` late-interaction ColBERT model (PyLate, 96-d per token) and search is a multivector MaxSim pipeline whose episode ranking is fused with a BM25 lexical signal via Reciprocal Rank Fusion whenever the index carries persisted lexical stats:

1. **Chunk.** Prompt and response streams are split independently with Chonkie using the embedding model's tokenizer at a ~235-token budget, sized to the model's context window; a small prefix overlap keeps each chunk inside that limit, and the title prefix added to the embedded text is trimmed to fit.
2. **Embed.** Each chunk becomes a `(num_tokens, 96)` token-level matrix written into a LanceDB multivector `vector` column (`list<list<float16,96>>`). Document embeddings are token-pooled by default: ssgrep merges each chunk's token vectors into ~`1/pool_factor` cluster means (`SSGREP_POOL_FACTOR`, default 2; `1` disables, `3` pools harder). Queries are never pooled. The multivector column carries a cosine IVF-PQ index tuned for 96-d tokens: `num_sub_vectors=12`, `num_bits=8`, and `num_partitions = max(1, min(64, rows // 4096))` computed from the live row count at build time.
3. **Score.** The query is embedded into its own `(num_tokens, 96)` matrix and LanceDB scores every chunk with native MaxSim. Queries carry `refine_factor(2)` and `nprobes(16)`, oversampled by `SSGREP_OVERSAMPLE` (default 8) so episode rollup sees candidates from more than one episode; returned results still honor the requested limit.
4. **Roll up.** The top chunks are rolled up deterministically per episode (an episode's score is its best chunk's score; a bounded-evidence term for the next two chunks exists in the rollup but currently carries zero weight) and reduced to the requested top-k.
5. **Fuse.** When the index carries persisted lexical stats, a BM25 score over the candidate pool is fused with the MaxSim ranking via Reciprocal Rank Fusion and the fused value becomes the episode score; older indexes keep pure MaxSim ordering. See [Hybrid lexical fusion](architecture.md#hybrid-lexical-fusion).

Each chunk stores its raw text separately from a short title-prefixed `search_text` used as the embedding source, so result excerpts stay verbatim transcript content. Excerpts are windowed around the most query-relevant section — by default token-level late-interaction similarity to the query, identical across terminal, JSON, and MCP output. Prompt and response text are chunked independently so `content_type` predicates remain exact.

Index format is versioned. An index built by an older ssgrep is refused with an actionable error; rebuild it once with:

```bash
ssgrep index --rebuild
```

## Measured Results

Retrieval quality is measured on a frozen synthetic benchmark that exercises the real production pipeline end to end, with a pinned metric suite reporting nDCG, Recall, MRR, and Precision and full provenance on every run. See [eval/README.md](../eval/README.md) for the benchmark definition and how to reproduce it.

Reference run: **arm `default`**, frozen v2 dataset, evaluated 2026-08-23 at `metric_limit=100`, 602 labelled queries (86 per class), 483 train / 119 holdout. Payload: `current_20260823T212203Z.json` (local run output; not kept in the repository).

The summary `r@10` / `r@50` / `r@100` are computed over the metric run extended to `metric_limit=100` (production top-10 page plus the deep-prefetch pool), so `r@50` and `r@100` are real deep-recall numbers, not truncated equals of `r@10`.

| Slice | n | nDCG@10 | nDCG@5 | r@10 | r@50 | r@100 | P@10 | P@5 | RR@10 |
|---|---|---|---|---|---|---|---|---|---|
| **overall** | 602 | 0.880 | 0.866 | 0.932 | 0.982 | 0.985 | 0.119 | 0.225 | 0.903 |
| train | 483 | 0.873 | 0.857 | 0.930 | 0.977 | 0.981 | 0.121 | 0.227 | 0.895 |
| holdout | 119 | 0.912 | 0.905 | 0.941 | 1.000 | 1.000 | 0.110 | 0.213 | 0.935 |

The holdout slice tracks or slightly exceeds train (nDCG@10 0.912 vs 0.873, r@10 0.941 vs 0.930), so the tuned pipeline does not overfit the train queries.

By query class:

| Class | nDCG@10 | r@10 | r@50 | r@100 | RR@10 |
|---|---|---|---|---|---|
| exact-identifier | 0.989 | 1.000 | 1.000 | 1.000 | 0.985 |
| decision-rationale | 0.958 | 0.977 | 0.977 | 0.977 | 0.952 |
| paraphrase | 0.927 | 0.965 | 0.977 | 0.977 | 0.914 |
| multi-hop | 0.886 | 0.924 | 0.965 | 0.977 | 0.952 |
| error-string | 0.887 | 0.953 | 0.988 | 0.988 | 0.863 |
| tool-failure-recovery | 0.827 | 0.942 | 0.977 | 0.977 | 0.790 |
| cross-runtime/project-scoped | 0.690 | 0.762 | 0.988 | 1.000 | 0.866 |

Operational:

| Measure | Value |
|---|---|
| Warm latency p50 / p95 | 669.1 ms / 741.0 ms |
| Index size | 34.2 MB (32.6 MiB) |
| Index build | 13.8 s |
| Token vectors | 68,169 |
| Corpus | 404 sessions / 888 episodes / 1,918 chunks |

Provenance pinned: `answerai-colbert-small-v1` (96-d), ir-measures 0.4.3, lancedb 0.37.1, pylate 1.6.0.

## Environment Variables

| Variable | Default | Accepted values | Controls |
|---|---|---|---|
| `SSGREP_POOL_FACTOR` | 2 | 1, 2, 3 | Document-side token pooling factor. Queries are never pooled. |
| `SSGREP_CHUNK_OVERLAP` | 25 | integer, clamped to [0, 117] | Prefix overlap tokens carried between consecutive chunks. |
| `SSGREP_PQ_BITS` | 8 | 4, 8 | IVF-PQ code width. Any other value fails indexing with a validation error. |
| `SSGREP_REFINE_FACTOR` | 2 | integer, clamped to [1, 20] | ANN rescore depth. |
| `SSGREP_NPROBES` | 16 | integer, clamped to [1, 512] | IVF partitions probed per query. |
| `SSGREP_OVERSAMPLE` | 8 | integer, clamped to [1, 20] | Candidate-pool multiplier applied before episode rollup. |
| `SSGREP_TWO_STAGE` | off | `1`, `true`, `yes`, `on` | Enables the experimental two-stage search path. |
| `SSGREP_TWO_STAGE_CANDIDATES` | 200 | integer, clamped to [200, 10000] | Stage-1 candidate floor for two-stage search. |
| `SSGREP_RRF_K` | 60 | integer, clamped to [1, 500] | Reciprocal Rank Fusion rank constant `k`. |
| `SSGREP_DENSE_WEIGHT` | 1.0 | float, clamped to [0.25, 4.0] | Weight of the MaxSim signal in fusion; the lexical signal always weighs 1.0. |
| `SSGREP_SEMANTIC_SNIPPET` | on | `0`, `false`, `no`, `off` | When on, excerpts are windowed to the most query-relevant section (display-only). |

Model and device overrides (`SSGREP_EMBED_MODEL`, `SSGREP_EMBED_DEVICE`, and friends) are described in [runtimes.md](runtimes.md#model-and-device-configuration).
