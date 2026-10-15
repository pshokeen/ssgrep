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

Child task ownership follows the proven native result-to-launch-call link and its initiating
user turn. A delayed launch result after a newer prompt cannot move the child to that newer
task; a missing, ambiguous or chronologically contradictory launch remains task-unknown.

Action and verifier outcomes use that same proven native call/result association, not
matching entry-ID suffixes. Missing, ambiguous, contradictory or failed results cannot
grant interpreted action success or verifier credit.

A complete JSON result followed only by a recognized native timing footer remains
structured exposure. Its decoded cards preserve escaped newlines and identifiers;
the entire visible footer still costs context. Arbitrary trailing output, truncation
and error evidence are never discarded to manufacture a successful structured result.

Lookup-page continuations use shell-safe command rendering rather than paragraph
wrapping. Quoted scope values, profiles and opaque cursors remain executable at
40/80/120 columns; following a continuation does not collect or change accounting.

**OpenCode cursor continuity is bounded evidence, not a universal replacement detector.**
Device/inode and available birth metadata are supplemented by primary-key timestamp anchors
inside the same read-only SQLite snapshot. A missing or regressed anchor invalidates the
cursor and triggers deduplicated replay; normal advancing updates and WAL checkpoints do not.
Anchor loss can also follow ordinary deletion, so a reset is not proof of physical replacement.
A replacement preserving every anchor while reusing filesystem identity is indistinguishable
from ordinary updates with these cursors.

## Credit and net

Immutable source atoms bind origin-event identity and content digest. A ref is a locator, not
proof that its current text is the original source. Cycles and out-of-scope source evidence are
excluded. Organization scope applies to both consumers and source contributions; a personal
copy does not authorize organization use.

`native_identity_v2` identifies supported native domains, not equal output text. Claude
record UUIDs require supported RFC UUID namespaces and semantic sub-identities; Pi, omp and
Prime entry IDs remain native-session-scoped. Missing native identities fall back to
source-local evidence, not cross-path copy equivalence. A same-domain identity with changed
content is a conflict. Copy deduplication never grants source ownership or export permission.
Migration replaces legacy derived totals; legacy and corrected namespaces must not be added.

Conflicted action, result, report, verifier or launch records cannot establish trusted
reuse or ancestry. Rejecting that authority is not empirical proof a fact was unused.
Conflicted earlier context makes novelty unavailable; absence from the selected primary
variant is not proof of novelty. Every observed variant must be provably later before
excluding a future-only conflict, and ancestor cutoffs use the proven child launch.
Independent trusted user/query context can still prove a known rejection. Lookup costs
survive; a disputed launch retains supported root-level cost with unknown task ownership.

Each task has nested estimated scenarios:

- **Conservative:** immediate discovery steps for eligible novel facts with interpreted
  successful action outcomes.
- **Central:** adds eligible prompt-to-witness discovery spans.
- **Optimistic:** adds eligible enclosing episodes, explicit relays or report references.

Per-receipt allocation preserves each lower scenario's already owned source ranges
before broader scenarios allocate new coverage. Another candidate's enclosing episode
cannot take that receipt's conservative share. Task-level source unions and common
debits are unchanged; inherited ranges must still be supported by the selected source,
and conflicting exact counts remain invariant errors rather than invented credit.

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

`known_net` is a compatible known subtotal, **not** a complete net or automatically a lower
bound. `credit_missing` and `debit_missing` preserve gaps; `complete` distinguishes a fully
supported zero from unavailable evidence. Explicit `bounds` carry separately supported
`credit_lower`, `credit_upper`, `debit_lower`, `debit_upper`, `net_lower` and `net_upper`.
Missing credit can leave its upper endpoint unknown; missing debit can leave net unbounded
below. Never replace an unavailable endpoint with zero or infer bounds from scenario names.

Bounds enclose the underlying raw interval: lower component endpoints round down, upper
endpoints round up, after compatible raw sums. Net bounds are
`floor(Cmin) - ceil(Dmax)` and `ceil(Cmax) - floor(Dmin)` in displayed units.
The point/known subtotal remains `floor(known_credit) - floor(known_debit)`, a whole-unit
model, not an exact raw difference. For zero credit and 1,000,200,000 ns debit, the point
is −1 s while the enclosing integer bounds are [−2 s, −1 s].

Native runtime cost fields are not automatically provider billing records. For the airline
lab's `openai-codex` model, omp calculates those fields from its model-price catalog; the
report labels them as runtime-reported catalog estimates, separately from reported token
usage. No invoice or subscription charge is inferred from them.

Raw monotonic lookup durations retain nanoseconds. `journal_active_proxy_ns_v2` caps each
unique modeled operation at 300,000,000,000 ns before summing; work-seconds divide the
compatible raw total by 1,000,000,000 only once. Observed `monotonic_v1` service milliseconds
are a separate presentation derived from summed raw ns, not sums of individually rounded ms.
Transcript `capped_gap_v1` observations retain native millisecond precision and their actor
interval unions; exact ms-to-ns scaling for composition does not invent finer observation
precision. The current model namespace is `honest_bounds_ns_v4`, which also corrects daily
tier ownership and historical scope eligibility. Accepted `honest_bounds_ns_v3` leaves,
receipts and publications stay separately labeled read-only history; new arithmetic
cannot run under that old identity. Unsupported partial historical filters report
`methodology_retired`, not a known zero. Missing required original evaluation authority
blocks retirement before replacement. Previously accepted bytes are never relabeled,
and current/historical namespaces cannot add usage twice. Missing legacy raw ns remains
unavailable; rounded stored milliseconds/seconds are never silently promoted to exact ns.
Active occupancy is an explicit proxy, not observed actor work or human waiting time.
Summed parallel actor work differs from unioned elapsed time; all time estimates are
**not human time saved**. Note creation, indexing and collection overhead remain separate
investments, not hidden offsets that improve retrieval benefit.

## Reproducibility and retention

Receipts expose IDs/revisions, lineage, execution/exposure links, atom allocations, exclusions,
known and missing costs, and methodology identities. Default retained evidence is compact
references and fingerprints, not copied prompts, queries, tool arguments or outputs. Replay
requires bounded verification of authorized original native records against their frozen
identity/digest, or an explicitly consented bounded plaintext cache. Changed or unavailable
originals limit replay; current index text is not a substitute. Accepted measurements remain
frozen with coverage reasons when replay authority is unavailable. Privacy deletion takes
precedence. Canonical ledger identity excludes collection-clock timestamps.

Snapshot identity also includes coverage diagnostics. Deliberate-exclusion counters
count admission observations, including database census revisits; they are not
unique lookup counts or a universal measure of lost capture. Different collection
schedules can therefore produce identical canonical accounting rows but different
coverage envelopes and snapshot IDs. Retrying one published snapshot does not
change its identity.

Native source verification is append-aware: JSONL witnesses pin the observed byte-prefix
length and SHA, not the entire subsequently appended file. OpenCode witnesses pin the native
row set and identity/content-digest/timestamp manifest; later rows or session updates do not
revoke earlier supported evidence or import future work. Initial reads through an existing
v6 index establish **observed-as-of provenance**, not historical cryptographic attestation.
Approved, verified older source history can support a lookup independently of the consumer's
30-day evidence-retention horizon. Missing, expired, changed, unapproved, identity-mismatched
and read-bound sources retain distinct coverage reasons. Compact record seeks/primary-key
reads remain bounded; unavailable authority does not authorize a whole-source scan fallback.

Each lookup freezes its accepted rarity generation. New supported generations apply
prospectively; later evidence for an old lookup uses its original assignment. One
exception is an explicitly provisional `frequency_build_pending` assignment: it may
be promoted when that same immutable build completes under the same profile, scope,
authority and key. Absent, refused or incompatible inputs are not silently rerated.
The input generation is not the valuation-model identity. There is no permanent
200-document corpus pin.

Production index frequency uses profile-keyed HMAC lexical metadata over indexed chunks.
For supported ASCII candidates, guaranteed lexical components form a **superset**: their
document frequency supplies an upper bound on candidate frequency and thus a **lower bound**
on rarity, not exact whole-path/flag/identifier frequency. Only proven threshold bands support
credit; absent components and inconclusive bounds remain unknown, not frequency zero.
Thresholds remain uncalibrated. No retained-plaintext corpus fallback is authorized.

Large compatible corpora build incrementally against pinned metadata/chunk table
versions, not a moving live view. Each step visits at most 4,096 lexical terms and
4,096 chunk-identity/scope projections, with 16 MiB metadata, 1 MiB page/head and
64 MiB aggregate keyed-page limits. Pending work remains unavailable, not zero.
The 250 ms visit-phase deadline is cooperative; parsing, uniqueness-proof reads,
final materialization and atomic I/O are not a hard wall-time bound.
Authority/key changes revoke partial pages. A missing head triggers bounded orphan
cleanup before another build; inventory overflow stays explicitly unavailable.

`ssgrep metrics collect --repin-frequency` or `--recompute` explicitly requests a named
replacement revision. Do not add replacement/legacy namespaces or generations as independent
savings. `--new-epoch` resets live operational counters, not retained ledger history.

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

The deterministic lane needs a reachable Docker daemon and the pinned PostgreSQL image, not
merely an installed Docker CLI. The integrated stand-in also needs host `psql`, the complete
pinned embedding snapshot and the supported filesystem sandbox; it performs real index/search,
CLI, HTTP and export work but uses scripted solver conversations. The current live
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

Integration positives report explicit prerequisite skips only when their required capability
is unavailable; independent missing-daemon/image/runtime negative controls still run. Linux CI
prepares the default embedding snapshot at its pinned revision before the suite. Lab execution
then remains offline: another cached model, or another revision, is not a substitute.

Executed control-lane results and their revision limits are recorded only in
[accounting benchmarks](accounting-benchmarks.md). They are not a current native-validation
pass or evidence of positive ROI.

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

Results: see [accounting benchmarks](accounting-benchmarks.md#observed-native-airline-experiments).

### Calibration limits

Results: see [accounting benchmarks](accounting-benchmarks.md#calibration-limits).

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


## Software verification

Results: see [accounting benchmarks](accounting-benchmarks.md#software-verification).
