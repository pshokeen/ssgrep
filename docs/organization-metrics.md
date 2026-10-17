# Organization metrics and ledger dashboards

> **Status: normative integration reference with synthetic examples.** This defines the
> supported privacy, aggregation and receiver contracts, not evidence of a deployed dashboard.
> Example values, identifiers and payloads are fabricated. The catalog follows
> `src/ssgrep/accounting/metrics.py`, `metric_estimates.py` and `export.py`; observed checks
> and their limits live only in [accounting benchmarks](accounting-benchmarks.md).

## Purpose and honest limits

This is the dashboard and query specification for the two organization export channels:
aggregate **metrics** (OTLP `/v1/metrics`) and, only under separate permission, pseudonymous
**ledger logs** (OTLP `/v1/logs`). It defines panels for coverage, errors, output cost, scenario
estimates, and collection-versus-acknowledgment freshness.

What these panels can and cannot say:

- **Collector acceptance is not backend durability.** A fully acknowledged batch means the
  organization's collector returned success for every request of that snapshot. It says nothing
  about whether the collector's exporter persisted, replicated, or retained the data.
  `GET /v1/export-status` therefore reports `acknowledgment:
  collector_accepted_backend_durability_not_claimed`, and no panel may be titled "synced",
  "delivered", or "stored".
- **Vanilla OTLP log acceptance is not exactly-once.** Retries, restarts, and reordering deliver
  the same record more than once and revisions out of order; a stock collector stores every
  arrival as a separate row (see the real-collector run below). Current totals need a
  **revision-aware** keyed upsert in the receiver. The reference semantics are
  `src/ssgrep/accounting/receiver_contract.py`; the SQL form below implements the same rules.
- **Aggregate metrics may be non-reversible.** Once a metrics batch is accepted, cumulative
  counters and estimate gauges cannot be recalled by ssgrep. Purge sends deletion tombstones for
  ledger log records only; the operator owns erasure of anything already stored.
- **Estimates are modeled, not measured.** Credit is "modeled, not causal proof"; net can be
  negative; scenarios are a range, not a forecast.
- **There is no productivity ranking.** No panel here ranks, scores, or compares people,
  devices, or tasks, and none may be added (see [Access control](#access-control)).
- **Organization-operated infrastructure only.** The integration needs an OTLP collector and a
  backend the organization already runs. ssgrep introduces no hosted control plane, no hosted
  receiver, and no identity service.

## Enable only what you intend to send

First enable local collection using the [local guide](savings.md). A managed profile is
explicit: `ssgrep savings setup --profile work --policy /absolute/path/to/policy.json` uses a
user/operator-supplied JSON policy (that path is illustrative, not a shipped file). Review
approved roots, destinations, credential references and retention before consenting. Local
overrides can narrow managed permission, not broaden it. Invalid updates preserve the last
valid collection constraints and pause sending rather than fall back to permissive defaults.

For an **unmanaged personal profile**, the local owner chooses the destination in the private
profile policy and explicitly enables each destination/signal. That configuration is the
owner's approval; ssgrep has no vendor-maintained host allowlist. Any syntactically valid HTTPS
destination can be configured, so inspect the exact endpoint before consenting.

For a **managed profile**, the supplied policy ceiling additionally constrains the destination
ID, exact endpoint URL, permitted signals, logs permission and proxy. Local settings cannot
widen that ceiling. This is application policy within the OS-account trust boundary, not a
defense against someone who can replace the owner's private configuration.

Export eligibility also requires a collection-time scope stamp, `coverage.collection_scope`
(schema, profile and digest of canonical effective roots, assignments and ownership). A missing,
corrupt or outdated stamp blocks even a first send with `ineligible_scope`; changing credentials
or enabling consent cannot authorize an old snapshot. Recollect under the current scope.
Scope changes start a new scope/counter epoch and rederive only permitted retained evidence,
archiving the previous private derived namespace rather than relabeling frozen old history.
`coverage.scope_epoch` and `coverage.scope_history_unavailable` identify that transition and
its unavailable history. Consent-only and credential changes do not reset scope.

Company export additionally requires completed `native_identity_v2` migration and explicit
approval of the current per-signal fields (`metrics-v2` / `logs-v2`). Old permission alone
does not approve new fields: `consent_reapproval_required` requires reviewing the preview
and explicitly enabling that destination/signal again. Preview/check do not grant approval.
Pending or unavailable migration blocks sending rather than mixing legacy and corrected totals.
Unknown native ownership may be locally observable but does not authorize organization
contributions; both consumer and source must have supported scope authority.

The following commands exist; replace `work` and `DESTINATION_ID` with the configured profile
and destination. No example implicitly enrolls a real organization:

```sh
ssgrep savings export preview DESTINATION_ID --profile work
ssgrep savings export enable DESTINATION_ID --profile work
ssgrep savings export check DESTINATION_ID --profile work
ssgrep savings export send DESTINATION_ID --profile work
ssgrep savings export disable DESTINATION_ID --profile work
```

`preview` is offline: exact sanitized candidate request bytes, no authentication secrets,
network or acknowledgment changes. `enable` requires confirmation (`--yes` when
non-interactive), and enables without sending. `check` sends an empty OTLP request to check
connectivity without advancing delivery state. `send` runs one send cycle. Do not combine modes.
Use `--logs` to select the separate ledger-log signal; the default is metrics. Log export requires
both policy permission and explicit enable consent. Metrics permission never authorizes task logs.
A fresh profile with no collector checkpoint and no snapshot files previews as `no_snapshot`;
this read-only state does not establish migration authority or grant consent. Existing legacy
state still requires migration, and corrupt or unsafe private state is not treated as empty setup.

Destinations use separately enrolled **ingest-only** credentials, referenced through supported
environment/file references rather than pasted into previews or commands. Credential files
must be owner-only. Never reuse the local HTTP endpoint bearer token as a remote credential.
Remote sends use explicitly configured TLS destinations and reject redirects and ambient proxies.
An explicitly configured proxy must be `http://host[:port]` and may tunnel only an HTTPS
destination; HTTPS proxy URLs and proxied loopback HTTP destinations are unsupported.
Managed profiles additionally enforce their destination ceiling; unmanaged profiles use the local
owner's configured endpoint. Plain HTTP is limited to direct loopback development collectors.

Offline sends retain bounded local spool (seven days or 100 MiB per destination); expiry/loss,
partial rejection and unknown outcomes stay visible. Retry is not proof of exactly-once ingest.
Current policy and consent are rechecked before dispatch. `ssgrep savings pause` stops capture,
collection and sending; `ssgrep savings export disable DESTINATION` revokes one destination and
signal while preserving bounded queues/history. Neither cancels a request already in flight.
Reenable explicitly; resume alone does not override revoked signal consent.

A 401/403 from the collector blocks sending for that destination and signal. The block clears only
when the credential value, endpoint or proxy changes; a successful `check` does not clear it, so
change the credential the policy's credential reference names, then run `send`. A logs signal with
no ledger records for the generation records a collector acknowledgment without sending any
request (nothing was transmitted; "accepted" is vacuous for that signal). `disable` reports the
same unacknowledged `pending_batches` (queued or in-flight spool batches) that `status`,
`/v1/export-status` and `/metrics` show; acknowledged batches kept in the spool are not counted.
`/v1/export-status` reports `enabled` per destination/signal from the current policy.

If registered queued data is missing or corrupt, sending stops with `spool_corrupt`; an incomplete
generation is never rebuilt and called fully acknowledged by silently dropping its missing records.
Restore the original consistent spool/registry state before retrying. Retention persists a pending
deletion together with its loss count and finishes that deletion after a crash, even if retention
limits have since widened; restart neither double-counts the loss nor resends that batch.

Inspect `ssgrep savings status` and the authenticated `/v1/export-status` endpoint. The ordinary
`/v1/snapshot` is the latest **collected** generation; the `view=acknowledged` destination/signal
view is the exact last fully accepted generation, not a relabeling of newer local data. Its
content can expire independently. See [endpoint authentication](savings.md#optional-foreground-service).
Purge cannot promise remote erasure: log tombstones require permission and a revision-aware
receiver, and previously accepted aggregate metrics can be non-reversible.


## Metric catalog

Counters are cumulative with persisted start times and a counter epoch
(`ssgrep.counter.epoch` resource attribute); estimates, coverage, and info series are gauges.
The first column is the name on the local `GET /metrics` endpoint, the second the OTLP name
(sums drop `_total`; a Prometheus exporter re-adds it). Names and label sets follow `_TABLE` in
`src/ssgrep/accounting/export.py` and `FAMILIES` in `src/ssgrep/accounting/metrics.py`.

| Prometheus family | OTLP metric | kind | unit | labels |
| --- | --- | --- | --- | --- |
| `ssgrep_executions_total` | `ssgrep_executions` | sum (cumulative, monotonic) | `{execution}` | operation, transport, outcome, runtime |
| `ssgrep_rendered_bytes_total` | `ssgrep_rendered_bytes` | sum (cumulative, monotonic) | `By` | operation, transport |
| `ssgrep_operation_duration_seconds` | `ssgrep_operation_duration_seconds` | histogram (cumulative) | `s` | operation, transport |
| `ssgrep_reconciliation_records` | `ssgrep_reconciliation_records` | gauge | `{record}` | attribution |
| `ssgrep_reuse_evidence_records` | `ssgrep_reuse_evidence_records` | gauge | `{record}` | evidence |
| `ssgrep_estimated_credit_tokens` | `ssgrep_estimated_credit_tokens` | gauge | `{token}` | scenario, window, method |
| `ssgrep_debit_tokens` | `ssgrep_debit_tokens` | gauge | `{token}` | window, method |
| `ssgrep_estimated_net_tokens` | `ssgrep_estimated_net_tokens` | gauge | `{token}` | scenario, window, method |
| `ssgrep_estimated_net_work_seconds` | `ssgrep_estimated_net_work_seconds` | gauge | `s` | scenario, window, method |
| `ssgrep_coverage_records` | `ssgrep_coverage_records` | gauge | `{record}` | kind, status |
| `ssgrep_snapshot_collected_timestamp_seconds` | `ssgrep_snapshot_collected_timestamp_seconds` | gauge | `s` | - |
| `ssgrep_snapshot_available` | `ssgrep_snapshot_available` | gauge | `{state}` | - |
| `ssgrep_accounting_info` | `ssgrep_accounting_info` | gauge | `{info}` | schema_version, methodology_id |

The additional v2 gauges below have identical Prometheus and OTLP names. These are the full
additional families, not open-ended templates; all dimensions retain the closed catalog.

| Family | unit | labels |
| --- | --- | --- |
| `ssgrep_estimated_credit_work_seconds` | `s` | scenario, window, method |
| `ssgrep_debit_work_seconds` | `s` | window, method |
| `ssgrep_known_credit_tokens` | `{token}` | scenario, window, method |
| `ssgrep_known_debit_tokens` | `{token}` | window, method |
| `ssgrep_known_net_tokens` | `{token}` | scenario, window, method |
| `ssgrep_known_credit_work_seconds` | `s` | scenario, window, method |
| `ssgrep_known_debit_work_seconds` | `s` | window, method |
| `ssgrep_known_net_work_seconds` | `s` | scenario, window, method |
| `ssgrep_estimate_bound_tokens` | `{token}` | scenario, window, method, quantity, bound |
| `ssgrep_estimate_bound_work_seconds` | `s` | scenario, window, method, quantity, bound |
| `ssgrep_estimate_missing_records` | `{record}` | scenario, window, method, quantity |
| `ssgrep_measured_service_latency_milliseconds` | `ms` | window, method |
| `ssgrep_known_service_latency_milliseconds` | `ms` | window, method |
| `ssgrep_measured_service_missing_records` | `{record}` | window |

For bound families, `quantity` is exactly `credit`, `debit` or `net`, and `bound` is
`lower` or `upper`. Missing-record families emit `quantity=credit` or `quantity=debit`;
an unavailable net is counted by omitted-sample coverage, not an additional missing-record label.
Scenario is `conservative`, `central` or `optimistic`; window is `24h`, `7d`, `30d` or
`lifetime`. Supported method labels are `chars_div4_v1`, `capped_gap_v1`,
`transcript_intervals_v1`, `monotonic_v1`, `provider_usage_v1` and `unknown`.
No query, source ID, path, task, receipt, generation, digest, timestamp or employee label
may be added. Destination labels are bounded configured IDs, not endpoint URLs.

Four exporter-health families exist only on the local endpoint (client-side series about the
exporter itself; they are not part of any OTLP request):

| Family | kind | labels |
| --- | --- | --- |
| `ssgrep_export_rejected_points_total` | counter | destination, signal |
| `ssgrep_export_lost_records_total` | counter | destination, reason |
| `ssgrep_export_last_full_ack_timestamp_seconds` | gauge | destination, signal |
| `ssgrep_export_pending_batches` | gauge | destination, signal |

The two `ssgrep_export_*_timestamp`/`_pending` gauges are scraped from the profile's loopback
endpoint by the organization's own scraper (bearer token, never a remote credential). A
Prometheus-compatible backend fed *only* by OTLP will not have them. Pseudonymous resource
attributes (`service.name`, `service.version`, `ssgrep.installation.id`, `ssgrep.policy.version`,
`ssgrep.counter.epoch`) travel with every OTLP metrics request; `tenant` and `device` are NEVER
sent by ssgrep. The collector derives them from the enrolled credential and overwrites anything
the payload claims.

Names above are the names ssgrep emits. An OTLP-to-Prometheus translation in your pipeline may
add unit or `_total` suffixes or turn resource attributes into `target_info`; adapt the queries
below to what your backend actually stores.

## Methodology, units, and missing values

**Fixed methodology.** Every estimate panel is pinned to exactly one methodology. The
`ssgrep_accounting_info{schema_version, methodology_id}` series (value 1, one per snapshot)
carries the methodology hash that also appears as `methodology` in each ledger log record. The
dashboard variable `$methodology` is a single-select list (Grafana:
`label_values(ssgrep_accounting_info, methodology_id)`; no "All", no regex matcher). Estimate
selectors join the info series so a device on another methodology is *dropped*, never added in:

For the OTLP-fed queries below, the backend must attach credential-derived `tenant`
and `device` labels and promote resource `ssgrep.installation.id` to
`ssgrep_installation_id` on both estimate and info series. Installation partitions
independent local profiles; it is not authentication. Expose the current info for
each exporter, not a historical union of methodology IDs. A set intersection filters
the cohort without multiplying values or requiring a unique right-hand info row.

```
<estimate gauge>{window="$window", method="$method"} and on (tenant, device, ssgrep_installation_id) ssgrep_accounting_info{methodology_id="$methodology"}
```

`method` is the measurement method (for example `chars_div4_v1`) and is also single-select;
`window` is one of `24h`, `7d`, `30d`, `lifetime` and is never summed across windows.

For direct local scrapes, use `instance, job` as the matching key instead, and count
distinct scrape targets rather than credential-derived devices. Do not use that
identity for an OTLP-fed fleet: `service.name` can be identical for every client and
ssgrep does not emit `service.instance.id`. Resource promotion and label normalization
are backend prerequisites, not additional client metric labels.

**Units are preserved, never converted.** Tokens (`{token}`), work seconds (`s`), and rendered
bytes (`By`) are separate panels with their own axes and are never added together. Ledger
rows additionally carry `unit`, `unit_method`, `unit_tokenizer`, and `unit_currency`; the SQL
totals select all four.

**Missing is omitted, and counted as coverage, never zero.** When a quantity is unknown,
its sample is absent. `ssgrep_coverage_records{kind="estimates",status="missing"}`
counts individual omitted quantity samples, not scenarios or devices. In the synthetic
`docs/examples/otlp-metrics-request.json`, the missing count is 2: optimistic credit
and optimistic net are both omitted. Queries never use `or vector(0)`; no samples means
"no data". An unknown ledger net is SQL `NULL`, counted in `unknown_*`; a `SUM` over
only unknowns remains `NULL`.

The v2 distinction is explicit: `ssgrep_estimated_net_*` is a complete compatible net;
`ssgrep_known_net_*` is only a known subtotal, not a net lower bound. Finite endpoints use
`ssgrep_estimate_bound_*`; an unbounded endpoint has no sample. Show
`ssgrep_estimate_missing_records` alongside these series. A complete zero is valid;
unknown or unsupported evidence is not zero. Observed service latency uses `monotonic_v1`
milliseconds, not transcript work seconds. The active-occupancy conversion
`journal_active_proxy_ns_v2` is an explicit model assumption capped at 300 seconds per operation;
neither it nor `capped_gap_v1` is human time saved.

**Negative values are real.** Net is credit minus debit; it is negative whenever lookups cost
more than they are modeled to have saved. The example shows conservative net tokens of -12000.
No query clamps (`clamp_min`, `abs`) and panel axes must extend below zero.

**Estimates are restated, not accumulated.** Each snapshot restates the whole window, so the
latest value is the value. Never apply `sum_over_time`, `rate`, or `increase` to an estimate
gauge; that would add replacement revisions together.

Namespace mixture is not an additive opportunity. Select one compatible methodology and unit
identity; quarantine conflicting namespace/revision inputs rather than merging them.
Current and archived replacements, overlapping rolling windows, duplicate native copies,
parent/child contributions and repeated exposure of the same source atoms must not double-count.
Corpus input generations can differ across frozen lookups without becoming new model identities.
These are normative aggregation rules; synthetic queries below are illustrative receiver forms,
not evidence that a real backend enforces them.

## Panels

PromQL below is a specification for a Prometheus-compatible backend. Variables:
`$window`, `$method`, `$methodology`, `$destination`; `$__range` and `$__rate_interval` are
Grafana macros. `tenant`/`device` are labels your backend derives from the collector-attached
resource attributes; `instance`/`job` identify the scraped endpoint. Every query is
checked only for metric and label names against the catalog (see the limits section).

### Coverage

Say how much of the picture exists before showing any number. Shares are `NaN`/no data when the
denominator is empty; they are not turned into 0%.

```promql coverage_attribution_share
sum(ssgrep_coverage_records{kind="lookups",status="attributed"}) / sum(ssgrep_coverage_records{kind="lookups",status="recorded"})
```

```promql coverage_source_completeness
sum(ssgrep_coverage_records{kind="sources",status="complete"}) / sum(ssgrep_coverage_records{kind="sources",status="total"})
```

```promql coverage_source_costs_available
sum(ssgrep_coverage_records{kind="source_costs",status="available"}) / (sum(ssgrep_coverage_records{kind="source_costs",status="available"}) + sum(ssgrep_coverage_records{kind="source_costs",status="unavailable"}))
```

```promql coverage_estimates_missing
sum by (tenant) (ssgrep_coverage_records{kind="estimates",status="missing"})
```

```promql coverage_unknown_output_size
sum(ssgrep_coverage_records{kind="rendered_bytes",status="unknown"})
```

```promql snapshot_unavailable
ssgrep_snapshot_available == 0
```

A device whose `ssgrep_snapshot_available` is 0 has no usage or savings samples at all; show it
as "unavailable", not as zero usage.

### Errors

```promql errors_by_operation
sum by (operation, transport) (increase(ssgrep_executions_total{outcome="error"}[$__range]))
```

```promql error_share
sum(increase(ssgrep_executions_total{outcome="error"}[$__range])) / sum(increase(ssgrep_executions_total[$__range]))
```

```promql latency_p95
histogram_quantile(0.95, sum by (le, operation, transport) (rate(ssgrep_operation_duration_seconds_bucket[$__rate_interval])))
```

Export errors come from the exporter-health series (local endpoint only):

```promql export_rejected_points
sum by (destination, signal) (increase(ssgrep_export_rejected_points_total[$__range]))
```

```promql export_lost_records
sum by (destination, reason) (increase(ssgrep_export_lost_records_total[$__range]))
```

Counters are cumulative with a persisted start time; `increase` handles a reset (a new counter
epoch) as a restart. `runtime` is frozen at capture and is often `unknown`; do not read
`runtime="unknown"` as a defect.

### Output cost

Output cost is what the tool printed into agent context, and the known lookup debit.

```promql output_bytes_by_operation
sum by (operation, transport) (increase(ssgrep_rendered_bytes_total[$__range]))
```

```promql output_bytes_per_execution
sum by (operation, transport) (increase(ssgrep_rendered_bytes_total[$__range])) / sum by (operation, transport) (increase(ssgrep_executions_total[$__range]))
```

Rendered bytes are recorded only where observable. Show `coverage_unknown_output_size` beside
`output_bytes_per_execution`: while it is above zero the ratio is a lower bound.

```promql debit_tokens
sum(ssgrep_debit_tokens{window="$window", method="$method"} and on (tenant, device, ssgrep_installation_id) ssgrep_accounting_info{methodology_id="$methodology"})
```

### Scenario estimates

Three scenarios (`conservative`, `central`, `optimistic`) are shown **together as three series**
on one panel, labeled "modeled range, not causal proof". Each query takes the latest value for one
window, one method, and one methodology.

```promql net_tokens_by_scenario
sum by (scenario) (ssgrep_estimated_net_tokens{window="$window", method="$method"} and on (tenant, device, ssgrep_installation_id) ssgrep_accounting_info{methodology_id="$methodology"})
```

```promql net_work_seconds_by_scenario
sum by (scenario) (ssgrep_estimated_net_work_seconds{window="$window", method="$method"} and on (tenant, device, ssgrep_installation_id) ssgrep_accounting_info{methodology_id="$methodology"})
```

```promql credit_tokens_by_scenario
sum by (scenario) (ssgrep_estimated_credit_tokens{window="$window", method="$method"} and on (tenant, device, ssgrep_installation_id) ssgrep_accounting_info{methodology_id="$methodology"})
```

A sum over devices hides that some devices had no sample for a scenario (unknown). Always show
how many devices contributed to each scenario next to the total:

```promql scenario_reporting_devices
count by (scenario) (count by (tenant, device, scenario) (ssgrep_estimated_net_tokens{window="$window", method="$method"} and on (tenant, device, ssgrep_installation_id) ssgrep_accounting_info{methodology_id="$methodology"}))
```

If `count` for `optimistic` is lower than for `central`, the optimistic total omits devices with
an unknown value; label it so. These are **metrics-channel** estimates. Per-task, current-revision
numbers come only from the ledger-log channel (below).

### Freshness: collection versus acknowledgment

Two different clocks, and neither says the data is stored:

- **Collection age**: how old the newest local snapshot is.
- **Acknowledgment age**: how long since the collector last accepted *every* batch of a snapshot
  for this destination and signal.

```promql collection_age
time() - ssgrep_snapshot_collected_timestamp_seconds
```

```promql acknowledgment_age
time() - ssgrep_export_last_full_ack_timestamp_seconds{destination="$destination", signal="metrics"}
```

```promql collected_after_last_ack
ssgrep_snapshot_collected_timestamp_seconds - on (instance, job) ssgrep_export_last_full_ack_timestamp_seconds{destination="$destination", signal="metrics"}
```

```promql never_acknowledged
ssgrep_snapshot_collected_timestamp_seconds unless on (instance, job) ssgrep_export_last_full_ack_timestamp_seconds{destination="$destination", signal="metrics"}
```

```promql pending_batches
ssgrep_export_pending_batches{destination="$destination", signal="metrics"}
```

Reading them: `collected_after_last_ack` above zero means a snapshot was collected after the
last full acknowledgment, so newer data exists that the collector has not fully accepted; show it
as "collected newer than acknowledged". **A value at or below zero does not prove the destination
is caught up**: the newest snapshot can be collected before the previous acknowledgment lands, and
batches are enqueued only at the export cadence, so `pending_batches` can be 0 while a newer
snapshot waits. The authoritative generation comparison is `last_collected_generation` versus
`last_full_ack.generation` in `GET /v1/export-status`. `never_acknowledged` returns devices that
have collected but never had a full acknowledgment (the ack gauge is omitted until the first one,
not 0). Do not label any of these panels "fully synced".

The six series above are **client-side**: they are written by ssgrep and scraped from its local
endpoint. They describe what the client believes. Collector-side freshness (when did the
organization's receiver last hear from this device, and how old is the newest data it holds) must
come from the receiver's own ingest time; the SQL `freshness` query below does that for a
receiver database.

## Receiver database form

A receiver that keeps ledger logs in a database keys every record by the authenticated
`(tenant, device, record_id)`. **`tenant` and `device` come from the enrolled credential, never
from the payload.** Before inserting, the receiver validates the body (the strict rules in
`src/ssgrep/accounting/receiver_contract.py`), recomputes `digest` as the SHA-256 of the canonical
JSON (sorted keys, compact separators, UTF-8, no floats, `ssgrep.accounting.contracts.digest_of`)
and compares it to the `ssgrep.record.digest` attribute, and takes `collected_at` from the log
record's `timeUnixNano` (a client claim; `ingested_at` is the receiver's own clock).

**Current revision** of a key is its maximum stored revision, and a key whose maximum revision is
a `deletion` is **tombstoned** and excluded. This reproduces the reference contract: duplicates
are no-ops, a lower revision never replaces a higher one, an older contribution replayed after a
tombstone stays below the tombstone revision, and a revision above the tombstone is a deliberate
re-creation. The log keeps every distinct revision; only the view collapses them.

The SQL below is an illustrative SQLite storage projection, not the v2 wire schema or a
complete receiver implementation. Validate current records with `receiver_contract.py`
before projecting them: populate `net_*` only from complete compatible nets, use SQL `NULL`
for incomplete nets, and never substitute `known_net` or a bound endpoint. Keep missing,
bounds and provenance metadata separately if displaying them; the minimal tables below do
not store that metadata. `INSERT OR IGNORE` and `ON CONFLICT` have equivalents in other
databases, but those engines and a deployed dashboard have not been exercised here.

```sql schema
CREATE TABLE ledger_log (
  tenant TEXT NOT NULL,
  device TEXT NOT NULL,
  record_id TEXT NOT NULL,
  revision INTEGER NOT NULL,
  digest TEXT NOT NULL,
  record_type TEXT NOT NULL CHECK (record_type IN ('contribution', 'adjustment', 'deletion')),
  task_id TEXT NOT NULL,
  methodology TEXT NOT NULL,
  unit TEXT NOT NULL,
  unit_method TEXT NOT NULL,
  unit_tokenizer TEXT NOT NULL,
  unit_currency TEXT NOT NULL,
  period TEXT NOT NULL,
  net_conservative INTEGER,
  net_central INTEGER,
  net_optimistic INTEGER,
  collected_at INTEGER NOT NULL,
  ingested_at INTEGER NOT NULL,
  PRIMARY KEY (tenant, device, record_id, revision)
);
CREATE TABLE ledger_quarantine (
  tenant TEXT NOT NULL,
  device TEXT NOT NULL,
  record_id TEXT NOT NULL,
  revision INTEGER NOT NULL,
  digest TEXT NOT NULL,
  held_digest TEXT NOT NULL,
  ingested_at INTEGER NOT NULL,
  PRIMARY KEY (tenant, device, record_id, revision, digest)
);
CREATE TABLE device_seen (
  tenant TEXT NOT NULL,
  device TEXT NOT NULL,
  last_ingested_at INTEGER NOT NULL,
  PRIMARY KEY (tenant, device)
);
CREATE VIEW ledger_current AS
SELECT l.* FROM ledger_log AS l
WHERE l.record_type <> 'deletion'
  AND l.revision = (
    SELECT MAX(m.revision) FROM ledger_log AS m
    WHERE m.tenant = l.tenant AND m.device = l.device AND m.record_id = l.record_id
  );
```

`unit_tokenizer` and `unit_currency` store the empty string for "none". Credit and debit columns
follow the same pattern as `net_*` if you need them.

Ingest runs three statements per record, in this order (bind the named parameters from the
validated body):

```sql ingest-seen
INSERT INTO device_seen (tenant, device, last_ingested_at)
VALUES (:tenant, :device, :ingested_at)
ON CONFLICT (tenant, device) DO UPDATE
SET last_ingested_at = MAX(last_ingested_at, excluded.last_ingested_at);
```

```sql ingest-quarantine
INSERT OR IGNORE INTO ledger_quarantine
  (tenant, device, record_id, revision, digest, held_digest, ingested_at)
SELECT :tenant, :device, :record_id, :revision, :digest, l.digest, :ingested_at
FROM ledger_log AS l
WHERE l.tenant = :tenant AND l.device = :device AND l.record_id = :record_id
  AND l.revision = :revision AND l.digest <> :digest;
```

```sql ingest-insert
INSERT OR IGNORE INTO ledger_log
  (tenant, device, record_id, revision, digest, record_type, task_id, methodology,
   unit, unit_method, unit_tokenizer, unit_currency, period,
   net_conservative, net_central, net_optimistic, collected_at, ingested_at)
VALUES
  (:tenant, :device, :record_id, :revision, :digest, :record_type, :task_id, :methodology,
   :unit, :unit_method, :unit_tokenizer, :unit_currency, :period,
   :net_conservative, :net_central, :net_optimistic, :collected_at, :ingested_at);
```

The primary key makes an identical retry a no-op and keeps the first-held row when the same
revision arrives with a different digest; the quarantine statement records that conflict first.
One difference from the reference: a same-revision, different-digest arrival of a different
type (for example a contribution at the revision already held by a deletion) is quarantined here
but `REJECTED_TOMBSTONED` in the reference; neither changes current totals.

Totals pin one tenant, one methodology, and one unit; a different methodology or unit is simply
not selected, never added in. `NULL` means unknown or no data:

```sql totals
SELECT COUNT(*) AS records,
       SUM(net_conservative) AS net_conservative,
       COUNT(net_conservative) AS known_conservative,
       COUNT(*) - COUNT(net_conservative) AS unknown_conservative,
       SUM(net_central) AS net_central,
       COUNT(net_central) AS known_central,
       COUNT(*) - COUNT(net_central) AS unknown_central,
       SUM(net_optimistic) AS net_optimistic,
       COUNT(net_optimistic) AS known_optimistic,
       COUNT(*) - COUNT(net_optimistic) AS unknown_optimistic
FROM ledger_current
WHERE tenant = :tenant AND methodology = :methodology
  AND unit = :unit AND unit_method = :unit_method
  AND unit_tokenizer = :unit_tokenizer AND unit_currency = :unit_currency;
```

The selector for the methodology and unit dropdowns lists what exists; the dashboard never
offers "all":

```sql methodologies
SELECT methodology, unit, unit_method, COUNT(*) AS records
FROM ledger_current
WHERE tenant = :tenant
GROUP BY methodology, unit, unit_method;
```

Per-subject view for ONE device pseudonym (see [Access control](#access-control)); the device is a
required parameter, so the query cannot list or compare people:

```sql subject
SELECT period,
       COUNT(*) AS records,
       SUM(net_conservative) AS net_conservative,
       SUM(net_central) AS net_central,
       SUM(net_optimistic) AS net_optimistic,
       COUNT(*) - COUNT(net_central) AS unknown_central
FROM ledger_current
WHERE tenant = :tenant AND device = :device AND methodology = :methodology
  AND unit = :unit AND unit_method = :unit_method
  AND unit_tokenizer = :unit_tokenizer AND unit_currency = :unit_currency
GROUP BY period
ORDER BY period;
```

Collector-side freshness from the receiver's own clock: `silent_seconds` is how long since the
receiver last heard from the device (any record, including no-op duplicates);
`data_age_seconds` is how old the newest stored record's collection time is. Old data with recent
contact is a normal catch-up signal:

```sql freshness
SELECT s.tenant, s.device,
       :now - s.last_ingested_at AS silent_seconds,
       :now - (SELECT MAX(l.collected_at) FROM ledger_log AS l
               WHERE l.tenant = s.tenant AND l.device = s.device) AS data_age_seconds
FROM device_seen AS s
WHERE s.tenant = :tenant;
```

## Worked example: current revisions

Synthetic key `(tenant-a, device-1, R)` for three records, applied in this arrival order through
`receiver_contract.apply` (the table's outcome column is what the code returns). Net values are
`conservative / central / optimistic` tokens; `R1` and `R3` are fabricated 32-character ids.

| # | record | revision | type | net | outcome |
| --- | --- | --- | --- | --- | --- |
| 1 | R1 | 1 | contribution | -50 / 100 / 200 | `APPLIED_NEW` |
| 2 | R1 | 3 | adjustment | -200 / 300 / 400 | `APPLIED_UPSERT` |
| 3 | R1 | 3 | adjustment | -200 / 300 / 400 | `DUPLICATE_NOOP` |
| 4 | R1 | 2 | adjustment | -80 / 150 / 250 | `REJECTED_OLD_REVISION` |
| 5 | R1 | 3 | adjustment | -200 / 999 / 400 | `QUARANTINED` |
| 6 | R2 | 1 | contribution | -10 / -20 / unknown | `APPLIED_NEW` |
| 7 | R3 | 1 | contribution | 5 / 10 / 15 | `APPLIED_NEW` |
| 8 | R3 | 2 | deletion | - | `APPLIED_TOMBSTONE` |
| 9 | R3 | 1 | contribution | 5 / 10 / 15 | `REJECTED_TOMBSTONED` |

Arrival 2 supersedes arrival 1; 3 is a retry of 2; 4 is a reordered older revision; 5 reuses
revision 3 with a different value (999) and is quarantined, not substituted; 8 retracts `R3`, and
9 is the older contribution replayed afterwards. `current_totals` over the receiver's state
(methodology `synthetic-methodology-1`, unit `tokens`/`estimate`):

```json worked-totals
{
  "current_rows": 2,
  "quarantined": 1,
  "tombstones": 1,
  "net": {
    "conservative": {"total": -210, "known": 2, "unknown": 0},
    "central": {"total": 280, "known": 2, "unknown": 0},
    "optimistic": {"total": 400, "known": 1, "unknown": 1}
  }
}
```

Current rows are `R1` revision 3 and `R2` revision 1. The central total is 300 + (-20) = **280**.
Summing every received row, as a stock collector's stored log would, gives **1849** for central
net: that is the double counting this contract prevents. Conservative is negative (-200 + -10);
the optimistic total of 400 covers one known row and reports one unknown, which is not 0.

The same nine arrivals run through the SQL form above (the test does this on synthetic rows with
stdlib `sqlite3`) give the same totals, one quarantined conflict, and six stored log rows (the
revision-3 duplicate, the conflict, and the identical replay of `R3` revision 1 are not stored).
The conformance fixture `docs/examples/receiver-conformance.json` covers the same rules and more
(`retry_and_reorder`, `tombstone_blocks_replay`, `unknown_net_not_zero`, tenant isolation); an
organization's own receiver should replay it.

## Access control

- **Per-subject views require authorized backend access.** Pseudonyms (`record_id`, `task_id`,
  `ssgrep.installation.id`) are HMAC outputs under a key that stays on the user's machine; they
  are not names. The `subject` query is parameterized by one device and returns that device's
  periods only. Gate it with row-level security or an equivalent role check in the backend, and
  log access. ssgrep does not provide the mapping from device to person and the dashboard must
  not join one in.
- **Aggregate dashboards need no per-subject access**: the metrics channel carries no subject,
  task, query, ref, path, or fact labels.
- **No productivity ranking.** Specifically: no leaderboards, no "top"/"bottom" devices or
  people, no `topk`/`sort`/`ORDER BY` on a savings or usage value, no per-person score or
  percentile, and no cross-person comparison panel. Scenario panels show a modeled range for one
  fixed methodology, not a performance measure. The test enforces that no query block in this
  document ranks.
- The collector derives `tenant` and `device` from the enrolled credential (see the collector
  reference); a dashboard must key on those, never on payload attributes.

## Reference collector and payload examples

- [`otel-collector-reference.yaml`](examples/otel-collector-reference.yaml): a runnable OpenTelemetry Collector
  configuration with synthetic tokens. It authenticates a bearer token, overwrites payload
  `tenant`/`device` with identity from a token table, and writes metrics and logs to separate
  file exporters. It is a reference to replace with your backend exporters and identity provider,
  not a production deployment.
- [`otlp-metrics-request.json`](examples/otlp-metrics-request.json): a synthetic OTLP/HTTP JSON metrics request built by
  `export_build.build_metrics`; it contains a negative estimate, an unknown (omitted) scenario,
  and cumulative counters and histograms with start times.
- [`otlp-logs-request.json`](examples/otlp-logs-request.json): a synthetic logs request built by
  `export_build.build_logs`: revision 2 (an adjustment superseding revision 1) of one record and
  revision 1 of another.
- [`otlp-logs-tombstone-request.json`](examples/otlp-logs-tombstone-request.json): the deletion revision 3 of the first record,
  built by `export_build.build_tombstones`.
- [`receiver-conformance.json`](examples/receiver-conformance.json): the receiver conformance fixture.

The three request examples are generated from the real builders by
`tests/accounting/test_org_docs.py` on fabricated ids and fixed timestamps, and the test fails if
a checked-in file differs from what the builders produce now. Regenerate after an intentional
serializer change with `uv run python -m tests.accounting.test_org_docs`.

Results: see [accounting benchmarks](accounting-benchmarks.md#real-collector-run).

## What was not verified

Results: see [accounting benchmarks](accounting-benchmarks.md#what-was-not-verified).
