# Exporting data

Three ways to get the savings ledger out of ssgrep, from least to most shared. All of them
read the last committed snapshot; none collects or indexes implicitly, and none includes
queries, transcript text, file paths or account identity. Run `ssgrep savings collect`
first when you want fresh numbers.

| Surface | Command | Who reads it | Format |
|---|---|---|---|
| Local report | `ssgrep savings export --output savings.json` | you | one JSON document |
| Loopback service | `ssgrep savings serve` → `GET /metrics`, `GET /v1/snapshot` | Prometheus, scripts on this machine | OpenMetrics text, JSON |
| Organization export | `ssgrep savings export DESTINATION_ID` | your organization's OTLP collector | OTLP/HTTP JSON (`/v1/metrics`, optional `/v1/logs`) |

## 1. Local JSON report

```sh
ssgrep savings export --output savings.json
ssgrep savings export --output - --from 2026-10-01 --to 2026-11-01   # UTC days, end exclusive
```

Written atomically with mode `0600`. The document separates metadata, usage, modeled
estimates (every scenario with bounds and provenance), every retained lookup/history page and
coverage. Unavailable values stay unavailable rather than becoming zero. No date filter means
the retained lifetime view. Details: [local savings ledger](savings.md#save-or-share-the-latest-numbers).

## 2. Loopback service (Prometheus-style scrape)

```sh
ssgrep savings serve            # foreground; prints URL, port and the token file path
curl -s http://127.0.0.1:PORT/healthz                                  # "ok", unauthenticated
curl -s -H "Authorization: Bearer $(cat TOKEN_FILE)" http://127.0.0.1:PORT/metrics
curl -s -H "Authorization: Bearer $(cat TOKEN_FILE)" http://127.0.0.1:PORT/v1/snapshot
```

`/metrics` without the token answers `401`. The body is
`application/openmetrics-text; version=1.0.0`; an excerpt from a synthetic five-session
sandbox (values are illustrative, label grammar is not):

```
ssgrep_executions_total{operation="search",transport="cli",outcome="ok",runtime="unknown"} 6
ssgrep_operation_duration_seconds_count{operation="search",transport="cli"} 6
ssgrep_reconciliation_records{attribution="inferred_unique"} 5
ssgrep_reuse_evidence_records{evidence="action_reference"} 4
ssgrep_known_credit_tokens{scenario="central",window="7d",method="chars_div4_v1"} 2440
ssgrep_known_debit_tokens{window="7d",method="chars_div4_v1"} 3340
ssgrep_known_net_tokens{scenario="central",window="7d",method="chars_div4_v1"} -900
ssgrep_estimate_bound_tokens{scenario="central",window="7d",method="chars_div4_v1",quantity="net",bound="lower"} -901
ssgrep_estimate_missing_records{scenario="optimistic",window="7d",method="capped_gap_v1",quantity="credit"} 3
ssgrep_known_net_work_seconds{scenario="central",window="7d",method="capped_gap_v1"} 161
ssgrep_coverage_records{kind="sources",status="complete"} 944
ssgrep_snapshot_collected_timestamp_seconds 1792815488
ssgrep_accounting_info{schema_version="1",methodology_id="3b3377ce291450b6e12bf50d4f92ed3b"} 1
```

How to read it: `scenario` is conservative/central/optimistic, `window` is 24h/7d/30d/lifetime,
`method` names the unit conversion, `quantity` is credit/debit/net and `bound` lower/upper.
`known_*` sums only records with evidence; `estimated_*` adds the modeled fill and equals
`known_*` whenever `estimate_missing_records` is 0. Net can be negative; it is reported, not
clamped. Series meanings and units: [metric catalog](organization-metrics.md#metric-catalog).

`/v1/snapshot` returns the same generation as JSON with an `ETag` equal to `snapshot_id`:
top-level `snapshot_id`, `generation`, `collected_at`, `coverage_through`, `counter_epoch`,
`methodology_id`, `policy_version`, `schema_version`, `source_manifest_digest`, `warnings`
(for example `thresholds_uncalibrated`), `observations`, `windows`, `coverage` and `estimates`.
Each `windows.<window>.estimates.<unit>.<scenario>` carries `credit`/`debit`/`net` values with
`basis` and `method`, `bounds` with their stated `assumptions`, per-candidate `eligibility`
and the `provenance` pairs (normalization and frequency versions) that make it reproducible.
Service details, token rotation and the acknowledged view: [optional foreground service](savings.md#optional-foreground-service).

## 3. Organization export (OTLP)

```sh
ssgrep savings export DESTINATION_ID --dry-run          # preview candidates; no network, no state change
ssgrep savings setup --export-to DESTINATION_ID --yes    # consent screen lists every field that leaves
ssgrep savings export DESTINATION_ID                     # send every approved signal
```

The destination comes from the profile policy (`destination_id`, `endpoint`, `signals`,
`logs_permitted`, `credential_ref` such as `env:ACME_OTEL_TOKEN`). The output of a send is
explicit about what "sent" means:

```
metrics: Sent. Sent 1, acknowledged 1, partial 0, quarantined 0.
logs: Sent. Sent 1, acknowledged 1, partial 0, quarantined 0.
Acknowledged means collector acceptance, not backend durability.
```

What the collector receives, per request: `POST /v1/metrics` and (only if permitted and
approved) `POST /v1/logs`, `User-Agent: ssgrep-export`, `Content-Type: application/json`,
`Authorization: Bearer …` from the credential reference. Resource attributes are
`service.name=ssgrep`, `service.version`, `ssgrep.installation.id` (a per-installation
pseudonym), `ssgrep.counter.epoch` and `ssgrep.policy.version`. The metrics payload holds the
same families as `/metrics`; `ssgrep_executions` is a cumulative monotonic sum with a start
time so collectors can rate it, the rest are gauges plus one histogram. Each log record's body
is one per-day aggregate `contribution` record (`record_id`, `period`, `revision`,
`methodology`, scenarios with bounds, raw known costs, coverage) — aggregates, never receipts.
Full request bodies: [`docs/examples/otlp-metrics-request.json`](examples/otlp-metrics-request.json)
and [`docs/examples/otlp-logs-request.json`](examples/otlp-logs-request.json); receiver
schema, dashboards and acknowledgment semantics: [organization metrics](organization-metrics.md).

## What never leaves

Queries, transcript text, tool arguments, file paths, hostnames, user or account names, and the
plaintext evidence cache. Pseudonymous ids are keyed per installation (a `--dry-run` preview
uses a throwaway key, so its ids differ from a real send). Revoke with
`ssgrep savings setup --no-export`; the local ledger stays. Stop everything with the steps in
[turning it off and rolling back](savings.md#turning-it-off-and-rolling-back).
