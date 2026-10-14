# Accounting methodology

The ledger reports **observations plus modeled scenarios**, not causal proof, provider billing
savings, human waiting time, or a productivity score. Start with [the local guide](savings.md).

## Implementation map

The accounting package uses module families to separate authority and state transitions. Start
at the entry points below rather than reading every helper. Thin CLI modules invoke these
families; the indexing/retrieval implementation remains outside the ledger.

| Concern | Start here | Responsibility boundary |
|---|---|---|
| Capture and private persistence | `accounting/capture.py`, `journal.py`, `storage.py` | Fail-open hot path; accepted events and private file authority |
| Raw evidence and task identity | `accounting/readers/`, `reconcile.py`, `lineage.py` | Native input parsing, attribution and originating tasks |
| Reuse and valuation | `accounting/reduce.py`, `evidence.py`, `valuation.py` | Supported source/action evidence, atom unions and compatible debit |
| Collection and revisions | `accounting/collector.py`, `collector_cycle.py`, `replay.py` | One bounded cycle, durable checkpoints and current/superseded history |
| Immutable reads and endpoints | `accounting/service.py`, `snapshots.py`, `http.py` | Read-only views, generation identity and authenticated local access |
| Remote delivery | `accounting/delivery.py`, `export.py`, `otlp.py`, `acks.py` | Current-policy projection, queued bytes, transport and acknowledgment |
| Retention and deletion | `accounting/retention.py`, `purge.py` | Forward-only expiry, scoped deletion and remote obligations |
| Human and machine CLI output | `cli/commands/metrics_*`, `savings_*`, `cli/*_render.py`, `*_json.py` | Argument handling, commands safe to copy, text/JSON representation |
| Validation-only airline lab | `eval/accounting_airline/__main__.py`, `lab_run.py`, `live_run.py` | Independent synthetic database/oracles and explicitly authorized native runs |

Paths under `accounting/` and `cli/` are relative to `src/ssgrep/`. Helpers named
`collector_*`, `delivery_*`, `export_*`, `service_*` and `purge_*` belong to those entry-point
families; they are not separate subsystems. Keep cross-family contracts explicit rather than
merging privacy, persistence and transport authority merely to reduce the file count. Source
size checks count nonblank, noncomment, non-docstring code: warning above 300, failure above 400.


## What is counted

An invocation is a runtime tool call; an execution is an actual ssgrep API operation; an exposure
is output actually visible to the consuming agent. They are different denominators. A CLI
validation error before entering the API can be an invocation without an execution; its visible
response still costs context. Repeated, empty and error responses are not free.

Default `chars_div4_v1` estimates context volume from observed characters with deterministic
rounding after aggregation. It is not a model tokenizer or a provider input-token count.
Provider input/output/cache/reasoning usage stays separate; streaming totals and overlapping
categories must not be added repeatedly. Exact offline tokenization requires an explicitly
selected installed tokenizer with pinned identity; missing tokenizers are unavailable, not
silently downloaded or mixed with the character estimator.

**Partial visible output can have an exact supported debit.** If the native transcript records
exactly 120 visible characters after truncation, debit is 30 `chars_div4_v1` tokens. A fact hidden
by truncation earns no credit. Partial exposure alone does not make an otherwise supported net
unknown: with no eligible credit, that example has a known net of −30. Unknown visible length
or inseparable output from unrelated commands remains unknown/partial, never assumed zero.

Execution success and exposure completeness also remain separate in the validation observer.
An append-only supervisor-broker attestation can verify raw execution even when OMP renders
only a first-line prefix and artifact pointer. Debit uses the actual native-visible text,
including its real footer—not hidden raw JSON. Hidden facts earn no credit, artifact pointers
are not opened by the observer, and solver-writable records cannot create supervisor
attestations. This distinction alone establishes no live-validation pass.

**Pure aggregates are charged once.** One shell invocation containing only two ssgrep searches
can produce two journal executions but one aggregate exposure. Its observed combined output is
debited once, and only actually visible source atoms can earn union credit. Mixed shell output
is not automatically assigned wholly to retrieval. Explicit child relay requires supported
lineage and matching payload; an unrelated sibling's activity is not evidence of reuse.

## Credit and net

Immutable source atoms bind origin-event identity and content digest. A ref is a locator, not
proof that its current text is the original source. Cycles and out-of-scope source evidence are
excluded. Organization scope applies to both consumers and source contributions; a personal
copy does not authorize organization use.

Each task has nested estimated scenarios:

- **Conservative:** immediate discovery steps for eligible novel facts with interpreted
  successful action outcomes.
- **Central:** adds eligible prompt-to-witness discovery spans.
- **Optimistic:** adds eligible enclosing episodes, explicit relays or report references.

These are a modeled range, **not statistical confidence intervals**. User-supplied facts and
contradicted candidates earn no discovery credit; failed-use/remediation costs remain costs.
Eligible overlapping source atoms are unioned before summing, not multiplied by result cards,
facts, references or parent/child rollups. Original-discovery leverage is separate, not recursive
net benefit.

Every scenario subtracts the same compatible known consumer debit set. Negative values remain
negative. Missing costs retain reasons and coverage; incomplete nets must not be presented as
complete totals. Never mix units, tokenizer identities, methods or currencies. Provider-paid
source usage is not itself counterfactual credit. Dollars require an explicit versioned offline
tariff; missing pricing is not zero, and zero reported subscription cost is not proof of free use.

Native runtime cost fields are not automatically provider billing records. For the airline
lab's `openai-codex` model, omp calculates those fields from its model-price catalog; the
report labels them as runtime-reported catalog estimates, separately from reported token
usage. No invoice or subscription charge is inferred from them.

Operation durations use monotonic clocks. Transcript `capped_gap_v1` work intervals cap each
positive adjacent gap at 300 seconds; invalid clock intervals are unknown. Summed parallel
actor work-seconds differ from unioned elapsed time. Neither establishes human time saved.
Observable note creation, indexing and collector overhead remain separate investments rather
than hidden offsets that improve retrieval benefit.

## Reproducibility and retention

Receipts expose IDs/revisions, lineage, execution/exposure links, atom allocations, exclusions,
known and missing costs, and pinned methodology identities. Candidate plaintext is local and
hidden by default. Canonical ledger identity excludes collection-clock timestamps.

Candidate-frequency calibration is pinned; `ssgrep metrics collect --repin-frequency` creates
a new methodology namespace. Recompute uses retained frozen evidence. Select one namespace
when comparing or aggregating; old and new estimates are not additive savings. Explicit
`--new-epoch` resets live operational counters, not the methodology's retained ledger history.

A changed canonical collection scope is a separate boundary: recollection archives the old
private derived namespace, starts new scope and cumulative-counter epochs, and rederives
permitted retained evidence. It must not relabel frozen historical totals as newly in-scope.
Snapshot coverage identifies the scope epoch and unavailable scope history; consent-only or
credential changes do not create this reset. See [scope-bound export eligibility](organization-metrics.md#enable-only-what-you-intend-to-send).

UTC daily periods remain provisional for seven days after period end. Late evidence creates
superseding revisions; only current leaf contributions enter current rollups. History preserves
the distinction. Evidence expiry can retain closed totals as non-recomputable rather than
silently shrink them; privacy purge takes precedence over reproducibility.
The committed aggregate-expiry floor is forward-only. Widening retention later does not
resurrect old sealed days, receipts or lifetime observations from still-retained raw evidence;
recomputation respects the same floor. An event exactly at the exclusive cutoff remains eligible.

## Validation boundaries

Runtime accounting is deterministic/local and does not invoke an LLM. Synthetic fixtures,
independent oracles and isolated endpoint/collector checks test particular contracts; they do
not prove causal benefit on real work. An explicitly authorized live-agent validation is a
separate optional experiment, not a runtime dependency. A deterministic stand-in run is not a
live-agent experiment, and automated receipt labels are not human calibration.

The validation observer's hash chain detects inconsistent recorded bytes; it is **not
authentication**. Solver-supplied records cannot grant themselves actor/task/run authority.
Native completion needs a supervisor-registered association; unmatched results remain unknown.
Observer reads are bounded and check ownership, file type, links and digests. Private refreshed
native copies, rather than shared hardlinks, keep collection views separate from solver files.
These boundaries do not turn local evidence into tamper-proof attestation.

Control-lane commands:

```sh
uv run python -m eval.accounting_airline --mode deterministic
uv run python -m eval.accounting_airline --mode standin
```

The deterministic lane needs Docker and the pinned PostgreSQL image. The integrated stand-in
also needs the cached embedding model and the supported filesystem sandbox; it performs real
index/search, CLI, HTTP and export work but uses scripted solver conversations. The current live
adapter additionally requires macOS `sandbox-exec`, Bun-installed omp with its native SDK,
and an existing authorized `openai-codex` credential. Authentication is read-only and credential
refresh is disabled. Missing prerequisites produce named blocked gates, not substitute models
or fabricated conversations.

Run the checkout and its executable environment outside `/tmp`, `/var/tmp` and the user
temporary directory. The live sandbox deliberately seals those trees to prevent reads of
neighboring or future experiment artifacts; it must not broadly allow them merely to launch a
virtualenv kept there. Run-owned temporary project/runtime directories receive narrow access.
Validation probes use committed oracle/fixture source files, not unpublished planning documents.
Python-based live-sandbox tests report an explicit unavailable-prerequisite skip when their
checkout or launcher is inside a sealed tree; they still execute on a supported layout.
Other parser, policy, accounting and ordinary sandbox tests are not disabled by this check.

The executed deterministic lane passed **23/23** database/domain/concurrency/rollback/cleanup
gates. The executed integrated stand-in passed domain and accounting assertions at **all
C0–C10 checkpoints**, including replay, scrape/export and owned cleanup. Its live gate correctly
remained `NOT_RUN`; these are control results, not the native experiment's result.

For a deliberately authorized, potentially billable native experiment, the implemented command
is:

```sh
uv run python -m eval.accounting_airline --mode live --runtime omp --authorize-live --model openai-codex/gpt-6-astra --thinking high
```

The current live authentication boundary supports explicit `openai-codex` models. Choose one
for which you have authorization. Omitting model/effort keeps the cheaper Luna/low defaults;
there is no automatic escalation. Resume keeps the manifest's pinned effort and rejects a
conflicting request. Preflight pins exact prompts, model/catalog prices, baseline, sandbox and
source hashes before conversation A's two turns and distinct B's two turns, ten minutes per turn.
Missing prerequisites or unmet gates remain blocked, failed or not run—not success. The command
prints the private artifact location. Native histories, SQL and source/action evidence stay
private; do not publish them without review.

### Observed native airline experiments

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

### Calibration limits

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
Organization dashboard examples retain their own [verification limits](organization-metrics.md#what-was-not-verified).

An isolated empty-root CLI smoke captured all 12 new command/group help pages with exit 0.
It exercised disabled summary → explicit setup → published collection → fresh summary/status,
empty history, pause, refused paused collection (exit 1, `collection_paused`), resume and a
side-effect-free purge preview. All other smoke commands exited 0. This checks first-use
lifecycle/CLI wiring only, not nonempty receipt valuation, live-agent behavior or remote export.

## Optional matched with/without-history experiment

This separate, explicitly authorized experiment compares **identical B tasks** on independent,
identical database clones with versus without access to verified A history. Comparing A's
discovery tasks directly with B's different tasks is not a savings estimate.

The implemented CLI below uses illustrative paths: replace `RETAINED` with the retained live
lab root whose A persistence/retrieval gates passed, and `NEW` with a new private output
directory that does not already exist. The model must match that source run.

```sh
uv run python -m eval.accounting_airline.matched --source-root RETAINED --output NEW --model openai-codex/gpt-6-luna --repetitions 1 --order-seed matched-v1 --authorize-live
```

`--repetitions` accepts 1–3 (default 1); `--order-seed` defaults to `matched-v1`. Each pair
authorizes at most four native model turns, ten minutes per turn, with no retries. Before
model calls, an immutable `declaration.json` pins order, runtime/model, prompt hashes,
baseline/seed/schema and A-source identity. All attempts stay in the private output, including
failed or interrupted attempts; an unfinished started attempt remains unknown. Do not reuse
an output directory or discard inconvenient attempts.

Correctness is reported before cost; only completed, correct paired arms are comparable.
Results are descriptive, not causal proof, statistical confidence intervals or human
calibration. Runtime catalog cost estimates are not provider invoices. Repeated clones of
one task family do not satisfy the human-receipt calibration gate.

**The actual provider-backed matched empirical run is UNRUN.** An implemented command or a
provider-free control smoke does not establish empirical savings.

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
