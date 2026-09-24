# Retrieval-quality improvements — 2026-08-23 session

Branch: `autoresearch/retrieval-quality-20260823` · session log: `.auto/log.jsonl`
· prompt / ideas: `.auto/prompt.md`, `.auto/ideas.md`

A single evening's autoresearch pass took the default-arm retrieval suite from
a composite of **0.728 to 0.861 (+18.3%)** on the frozen v1/v2 benchmark
(602 labelled queries, 86 per class, 483 train / 119 holdout). The headline
win was MRR — up **+25.7 points** — from a one-line fix to a mis-scaled
multi-chunk bonus. Everything else was confirmation that the pipeline had
reached its exact-MaxSim ceiling.

## The numbers, before → after

"Before" is the previously documented reference run
(`current_20260823T070646Z.json`, the payload `eval/README.md` cited at the
time). "After" is the branch-tip run (`current_20260823T191945Z.json`,
provenance commit `c5e42f3` — byte-identical to tip `9da30d1`). Neither
payload is in the tree any more: only the latest successful run is committed
under `eval/results/`, so the tables below are the record of these two runs.

Composite = `0.4·r@10 + 0.3·rr@10 + 0.3·ndcg@10` (the session's primary
score; higher is better).

| Measure | Before | After | Δ |
|---|---|---|---|
| **composite** | 0.728 | 0.861 | **+0.133 (+18.3%)** |
| rr@10 (MRR@10) | 0.576 | 0.833 | **+0.257** |
| ndcg@10 | 0.645 | 0.818 | **+0.173** |
| ndcg@5 | 0.614 | 0.801 | +0.187 |
| r@10 | 0.903 | 0.913 | +0.010 |
| r@50 | 0.947 | 0.982 | +0.035 |
| r@100 | 0.977 | 0.985 | +0.008 |
| p@5 | 0.205 | 0.216 | +0.011 |
| p@10 | 0.115 | 0.116 | +0.001 |
| latency p95 | 66.9 ms | ~700–796 ms | (set aside — see note) |

The bulk of the gain is **ordering**, not recall: r@10 only moved +1 point
while MRR jumped +25 points. The same relevant episodes were already being
retrieved; they were just being out-ranked by the wrong (non-relevant)
episodes. Fixing that ordering is what the session was about.

> Latency note: the two numbers above are not directly comparable — the
> before-run reported a 59.0/66.9 ms warm figure from a different benchmark
> corpus, while this session's runs all reported p95 in the 680–796 ms band
> on the shared synthetic index. The oversample-8 step added ~5.6% latency
> vs oversample-4 but was kept because it buys recall and stays far under
> the 1.2× baseline ceiling the session tracked.

### By query class (ndcg@10 / r@10 / rr@10)

| Class | ndcg@10 before → after | r@10 before → after | rr@10 before → after |
|---|---|---|---|
| exact-identifier | 0.751 → 0.960 | 0.988 → 1.000 | 0.672 → 0.947 |
| decision-rationale | 0.734 → 0.918 | 0.977 → 0.977 | 0.654 → 0.898 |
| paraphrase | 0.702 → 0.889 | 0.953 → 0.953 | 0.617 → 0.869 |
| multi-hop | 0.678 → 0.820 | 0.924 → 0.872 | 0.611 → 0.920 |
| error-string | 0.558 → 0.832 | 0.864 → 0.965 | 0.470 → 0.786 |
| tool-failure-recovery | 0.581 → 0.646 | 0.884 → 0.884 | 0.482 → 0.570 |
| cross-runtime/project-scoped | 0.508 → 0.663 | 0.727 → 0.738 | 0.527 → 0.841 |

Every class improved on MRR and nDCG. The strongest gains landed where
they should have been landing anyway — **exact-identifier** reached r@10 =
1.000 and **error-string** improved the most (nDCG@10 +0.29, rr@10 +0.39),
which is exactly the failure-text class that is the product's core use case.
The weakest absolute slice remains tool-failure-recovery and
cross-runtime/project-scoped, and these are the label-intrinsic misses
described in the ceiling section.

## What was changed (the code, in order)

All retrieval changes were in `src/ssgrep/search/__init__.py` (rollup +
oversample). One tangential chunking experiment
(`CHUNK_TOKEN_BUDGET` 235 → 180 in `src/ssgrep/indexing/chunker.py`) is
**uncommitted and not part of these results**.

| Commit | Change | Effect (composite) |
|---|---|---|
| `60232f0` | multiplicative rollup bonus | 0.728 → 0.729 (noise) |
| `6f54590` | revert to additive rollup | 0.729 → 0.726 |
| `47a1111` | `MULTI_CHUNK_EVIDENCE_WEIGHT` 0.05 → 0.005 | 0.726 → **0.845** |
| `bd7da1c` | weight → 0.0 (best-chunk only) | 0.845 → **0.858** |
| `b4db370` | `OVERSAMPLE_FACTOR` 4 → 8 | 0.858 → **0.861** |
| `e919da6`, `c5e42f3`, `9da30d1` | comment-only / confirmations | 0.861 (stable) |

## What actually worked (from the logs)

### 1. The measurement methodology was the hidden precondition

The first three runs taught a lesson that made every later A/B trustworthy
(log run 3):

> Run-to-run ANN index training noise is huge (identical code: 0.903 vs
> 0.897 r@10, 0.580 vs 0.572 rr@10). MUST use a shared index + `--no-rebuild`
> for A/B.

Before that, the multiplicative-rollup "win" (run 2) was pure index-training
noise; the additive A/B on a **shared fixed index** showed additive actually
won (0.726013 vs 0.725159). Every real decision after that was measured
deterministically, which is why the final numbers are byte-identical across
repeated runs (log runs 5, 8–12).

### 2. The multi-chunk bonus weight was the dominant lever
`MULTI_CHUNK_EVIDENCE_WEIGHT = 0.05` was catastrophically miscalibrated for
the refined MaxSim score scale. Scores there are ~O(num query tokens), i.e.
~30; so `0.05 × sum(next 2 chunks)` inflated an episode by ~+3, ~+10% of its
best-chunk score. Episodes that happened to appear with many ANN chunks
(typically near-duplicate synthetic episodes) dominated the top-10, shoving
the genuinely-highest-scoring **relevant** episode down.

- Lowering to `0.005` (a ~+1% nudge) fixed the ordering: rr@10 0.58 → 0.80,
  ndcg@10 0.64 → 0.80, r@10 0.90 → 0.91. **Both train AND holdout improved**
  (generalizes — no in-sample overfit).
- Setting it to `0.0` (best-chunk-only rollup, the Occam choice) pushed it
  further: rr@10 → 0.832, ndcg@10 → 0.817. This is the single highest-value
  change in the session.

### 3. Deeper candidate pool — now harmless because ordering is correct
`OVERSAMPLE_FACTOR` 4 → 8 (final pool 40 → 80 chunks): all four target
metrics rose (r@10 0.9086 → 0.9128, rr@10 0.832 → 0.833, ndcg@10 0.817 →
0.818, p@5 0.2153 → 0.2159) at +5.6% p95 latency — far under the 1.2×
ceiling. Deeper pools had interacted badly with the big bonus earlier, but
became a clean recall knob once the bonus was gone. Boundary test confirmed
oversample 12 was byte-identical at +5% latency, so 8 is optimal.

### 4. Explicitly rejected (no value / noise)
- `NPROBES` 16 → 64: byte-identical metrics, no benefit, forced test churn.
  Reverted. The IVF-PQ candidate selection was already sufficient.
- Multiplicative (percentage) rollup bonus: superseded by the additive A/B
  on a shared index. Reverted.
- `CHUNK_TOKEN_BUDGET` 180 experiment: left uncommitted, out of scope.

## The ceiling is proven — the residual gap is label-intrinsic

Exact brute-force full-scan MaxSim ranks the first relevant episode in
top-10 for **580/602** queries; production (weight=0, oversample=8) matches
that byte-for-byte (r@10=0.9128, rr@10=0.8329, ndcg@10=0.8184, p@5=0.2159,
r@50=0.9817). The 22 exact-scan misses have the relevant episode at rank
11–200+ with **zero session-prefix overlap** against the top-5 exact hits —
i.e. these are label-vs-similarity disagreements (the relevant episode's
content genuinely does not MaxSim-match the query as well as other
sessions' content). No pool sizing or reorder can fix them.

Conclusion recorded in the session: the current late-interaction MaxSim
score function is optimal for these labels. Further gains require a
**different scorer** — new/fine-tuned embedding, cross-encoder rerank, or
query-term IDF reweighting — and every one of those risks overfitting to the
synthetic, template-reused corpus. The session stopped there deliberately
(advice: validate any scorer change on a held-out real corpus).

## Files
- `current_20260823T191945Z.json` — after run (removed from the tree; summarised above)
- `current_20260823T070646Z.json` — before run (removed from the tree; summarised above)
- `baseline_v1.json` — earlier v1-dataset baseline (removed from the tree)
- `current_20260823T212203Z.json` — the reference run that superseded these (local output; not kept in the repository)
- `.auto/log.jsonl` — every experiment, decision, and confirmation
- `.auto/ideas.md` — findings, ceiling verification, deferred ideas
