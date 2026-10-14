# Accounting benchmarks and validation results

Empirical results and verification limits for the savings ledger; methodology lives in accounting-methodology.md.

## Observed native airline experiments

Three separately declared trials used the same 12-table, 82-row problem family. Each ran four
fresh native turns: A/L1–L2, then B/L3–L4. Every attempt remains retained; there were no
within-trial model retries or replacements of inconvenient outcomes.

| Trial | Model / effort | Conversation | Native cumulative total tokens | Catalog estimate (USD) |
|---|---|---|---:|---:|
| 1 | Luna / low | A | 54,676 | 0.00390576 |
| 1 | Luna / low | B | 197,132 | 0.00692152 |
| 2 | Astra / high | A | 77,828 | 0.459552 |
| 2 | Astra / high | B | 223,854 | 0.820276 |
| 3 | Astra / high | A | 81,315 | 0.446702 |
| 3 | Astra / high | B | 275,908 | 1.013296 |

The total catalog estimate was **$2.75065328**, not provider billing. Per-turn usage/cost deltas
prevent double-counting cumulative session captures. Input, output, cache and reasoning
categories remain separate in the private reports. Do not subtract A from B as savings—their
tasks differ—or attribute differences between changed prompts/models to ssgrep.

**Trial 1:** A's diagnostic `raw_free` formula was ambiguous in the original prompt; its
sellable inventory, occupancy and live holds matched. B inverted the stated priority order.
Compact JSON show output was truncated, hiding answer tails. A later zero-provider-call frozen
replay preserved native/journal bytes and matched the final snapshot at **0 credit, 1,313 debit,
−1,313 net** context tokens. It did not retroactively pass the historical checkpoints.

**Trial 2:** Clearer business rules, stronger reasoning and full human-readable shows produced
correct L1–L4 answers and passing accounting comparisons. B nevertheless recomputed the rules
relationally: no retrieved policy identifier entered a later tool action. The declared reuse
gate correctly stayed failed. Production and fixture estimates were **0 / 9,464 / −9,464**.

**Trial 3:** The preregistered business task explicitly required auditing historical policy keys
against current database rows before planning. B independently extracted five identifiers from
A and generated a `LEFT JOIN` audit checking missing, changed and current values. Source
witness → visible exposure → successful SQL action was traceable for all five. **All 18 gates
and every C0–C10 domain/accounting checkpoint passed**, including genuine native sessions,
persistence/restart, fresh-B isolation, L1–L4 correctness, reuse, replay, scrape/export and cleanup.

The canonical runner now uses separate retrieval calls and ordinary readable `show` output,
with 120-column/no-color rendering in both environments. All answer text remained visible;
maximum observed line length was 360 bytes. Temporary-directory isolation also seals prior
and future neighboring reports, with own-scratch and SQL positive controls.

Production remains unchanged and uncalibrated: all scenarios report **0 credit, 9,555 known
visible debit, −9,555 net context tokens**. The separately reviewed arithmetic fixture checks
allocation math, not production eligibility:

| Fixture scenario | Credit | Debit | Net context tokens |
|---|---:|---:|---:|
| Conservative | 521 | 9,555 | −9,034 |
| Central | 1,327 | 9,555 | −8,228 |
| Optimistic | 2,169 | 9,555 | −7,386 |

Fixture-policy values are never exported or mixed into production totals. The successful trial
proves **explicitly prompted historical-key validation**, not spontaneous retrieval utility,
positive ROI, representative precision or causal savings. Product evidence remains conservatively
`action_reference`; the actual database audit supplies the independent domain validation.

## Calibration limits

A preliminary, agent-reviewed evidence exercise examined **60 real receipt rows (59 distinct
receipt IDs) across 30 tasks/roots**, plus **15 error/no-reuse controls**. A second agent
independently inspected ten evidence cards without opening the product predictions. This is
not human ground truth: 48 rows were from omp and 12 from one Claude root, other consumer
runtime and explicit-relay strata were absent, and 36 of the 60 source histories could not be
resolved. The reviews found prior-known names misclassified as novel shorter tokens, generic
symbol coincidences, report-only references, and source-witness binding limitations.

**Thresholds remain uncalibrated.** The exercise does not establish representative precision,
recall, causal savings, or a human-calibrated central estimate. Private review cards and
transcript content are not shipped. The native experiments above are separate from this calibration
exercise; full software verification is recorded by the release gates, not inferred from these labels.
Organization dashboard examples retain their own [verification limits](#what-was-not-verified).

An isolated empty-root CLI smoke captured all 12 new command/group help pages with exit 0.
It exercised disabled summary → explicit setup → published collection → fresh summary/status,
empty history, pause, refused paused collection (exit 1, `collection_paused`), resume and a
side-effect-free purge preview. All other smoke commands exited 0. This checks first-use
lifecycle/CLI wiring only, not nonempty receipt valuation, live-agent behavior or remote export.

## Software verification

A recorded review-fix gate at `f6d97e9` ran **4,902 passing tests** and reached **100%
source-statement coverage** (24,437 statements). Its normally excluded network-marked embedding
test also passed in a separate real-model run. Lint, formatting, typing, source-size and
dependency checks passed. The shipped-source live isolation preflight and the standalone
PostgreSQL/control journeys were rerun without private planning files or additional model calls.
The suite still reports the existing LanceDB `create_index` deprecation warnings; these were
not suppressed. Coverage is an execution measure, not proof of estimator calibration or security.
For current-head verification and subsequent findings, use [PR #22](https://github.com/pshokeen/ssgrep/pull/22)
and its CI records rather than treating this historical result as approval of later changes.

## Real collector run

**What a real collector run showed** (`tests/accounting/test_otel_collector.py` and
`tests/accounting/test_otel_delivery.py` against the image
`otel/opentelemetry-collector-contrib:0.137.0`; each module skips with a named reason when docker
or that image is absent):

- The collector accepted the client's real OTLP/HTTP JSON requests for both signals, and the
  file exporter's contents equaled the projected data points. A wrong or missing token was
  rejected with 401, which the client treats as blocked.
- With the reference configuration, payload-asserted `tenant`/`device` were replaced by identity
  from the credential. `bearertokenauth` does not expose an identity to the pipeline, so the
  reference maps the token through an OTTL table; prefer an authenticator that exposes claims in
  production.
- After a collector stop and restart, the client resent the same spooled bytes and the file
  gained exactly one copy. A logs deletion batch was delivered as a separate document.
- Vanilla logs ingestion kept all three revisions of a record (new, revised, deleted) as separate
  arrivals; only revision-aware logic (`receiver_contract.apply`) produced the current totals.
  The collector run does not demonstrate partial success, `429`/`Retry-After`, gzip, or a
  non-loopback network.

## What was not verified

- **No real backend dashboard was run.** No Grafana, Prometheus, or SQL dashboard was built or
  rendered from this document; panel layout, variables, and the OTLP-to-backend name translation
  are unexercised.
- **The PromQL is a specification.** `tests/accounting/test_org_docs.py` checks only that every
  metric and label name used appears in the catalog, that no block fabricates zero, clamps
  negatives, restates estimates over time, mixes methodologies, or ranks. It does not execute any
  PromQL. A separate test, `tests/accounting/test_metrics_scrape.py`, runs a real Prometheus
  against the local endpoint for simple selectors, `histogram_quantile`, and `time() - gauge`;
  it skips without docker and this document's joins and ratios are outside what it exercises.
- **The SQL is validated only as far as it ran on SQLite.** The test executes every SQL block on
  synthetic rows with stdlib `sqlite3` and asserts the totals, current-revision selection under
  shuffled arrival, tombstone exclusion, quarantine, `NULL` for unknown, and the freshness
  numbers. It was not run on PostgreSQL or any other engine, and not under concurrent ingest
  (the reference `apply` is also single-threaded).
- Collector acceptance is the strongest state observable; backend durability, retention, and
  erasure after a purge are the operator's to verify.
- Real network conditions, TLS to a remote host, gzip, rate limiting, and `Retry-After` against a
  real collector were not exercised.
- Real user data was never used: every identifier, token, and number here is fabricated.
