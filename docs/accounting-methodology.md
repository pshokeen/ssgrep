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

## Airline validation lab (historical)

The isolated PostgreSQL airline lab that produced the native and control results in
[accounting benchmarks](accounting-benchmarks.md#airline-validation-lab-historical) is not part
of this tree. Its code is kept on branch `feat/accounting-airline-lab`. The method is fixed: a
fresh native session A solves the reconciliation and disruption tasks against an isolated
database; a fresh session B in the same canonical project must issue real search/show requests
before reading the rules it reuses, then solves the rebooking and commit tasks. A verifier
compares SQL results and final database state with sealed answers held outside solver-readable
roots.

Live native runs bill against a provider and require explicit, per-run authorization of model
and effort. Missing prerequisites are reported as blocked or not run, never as success. Control
lane results do not establish a current native pass or a positive return on investment.

### Calibration limits

Results: see [accounting benchmarks](accounting-benchmarks.md#calibration-limits).



## Software verification

Results: see [accounting benchmarks](accounting-benchmarks.md#software-verification).
