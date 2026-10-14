# Local savings ledger

Savings is an **opt-in local ledger**, not a claim that a lookup caused a productivity gain.
It separates observed executions and visible output from modeled reuse credit. Collection and
remote export both start disabled. The default profile is `personal-local`, even inside an
employer's repository; every command below accepts `--profile`.

## First use

```sh
ssgrep savings
ssgrep metrics setup
ssgrep metrics collect
ssgrep savings --by task
ssgrep metrics status
```

The first summary explains the disabled state without enabling anything. Setup previews roots,
retained fields and retention, then asks for consent. Decline to cancel without changes; in
non-interactive use, `ssgrep metrics setup --yes` explicitly authorizes the displayed defaults.
Setup does not backfill history, send data, start a worker, or install a service. Use ordinary
ssgrep lookups after setup, then collect again. An empty ledger is not proof of zero benefit.

Summaries read the latest completed snapshot; they do not index or collect. `--since` and
`--until` accept UTC dates or RFC3339 timestamps (end exclusive). Filters include `--project`,
`--runtime`, `--session`, `--task`; `--include-children` accompanies `--session`. Group by
`task`, `runtime`, `project`, or `day`; page with `--limit` (1–100) and the returned `--cursor`.
Use `--json` for machine-readable output. Do not publish private local JSON unreviewed.

Suggested commands use a shell-comment label (for example `# Next:`) above the command,
with the explanation before that block. Copy the complete block, including its label and
backslash continuations, into a POSIX shell without deleting the label. Wrapped arguments
retain their exact value, including quotes, shell metacharacters and wide Unicode characters.

For a uniquely matched CLI lookup, collection uses the journal's captured canonical working
directory for both execution counts and exposure estimates, so aliases such as `/var` and
`/private/var` do not split one project into separate rows. Savings reads use those committed
identities without resolving historical symlinks again. An MCP server's working directory is
not substituted for its caller's project.

```sh
ssgrep savings --since 2026-10-01 --until 2026-10-08 --by day --json
ssgrep savings history --since 2026-10-01 --until 2026-10-08
```

To inspect a receipt, replace `RECEIPT_ID` below with an ID from the summary:

```sh
ssgrep savings explain RECEIPT_ID
ssgrep savings explain RECEIPT_ID --include-content
```

The default explanation shows allocations, exclusions, costs and missing evidence.
`--include-content` requests any retained private local content; normal production receipts
do not retain plaintext, so content can remain unavailable. It does not fetch missing sources,
collect or index. History exposes revisions, not additive copies of replaced totals. Read the
[methodology](accounting-methodology.md) before interpreting nets.

## Collection and lifecycle

```sh
ssgrep metrics collect --backfill --since 2026-10-01 --until 2026-10-08
ssgrep metrics collect --recompute
ssgrep metrics collect --repin-frequency
ssgrep metrics collect --new-epoch
ssgrep metrics pause
ssgrep metrics resume
```

- Ordinary collection is one bounded pass. Backfill requires both bounds and is separate from
  live counters; it cannot be combined with recompute. Recompute uses retained frozen evidence,
  not newly substituted source text.
- `--repin-frequency` explicitly pins a new candidate-frequency view and creates a **new
  methodology namespace**. Do not add its estimates to the previous namespace.
- `--new-epoch` declares a reset/restored backup and restarts live counters, not ledger history.
  Ordinary restart preserves counters; an undeclared whole-state restore cannot be detected.
- Pause stops capture, collection and sending, preserving readable history. Resume does not
  silently import the paused interval; use explicit backfill if wanted. There is no
  `metrics disable` command: use pause to stop local measurement, and export `--disable` for
  each authorized remote signal.
- Ctrl-C stops a foreground worker; it is not a permanent opt-out. Use pause for that.

`metrics status` distinguishes disabled, no snapshot, fresh, stale, paused and error conditions.
Missing or corrupt data is not zero savings. A previous completed snapshot stays readable during
collection; freshness ages while paused. Default collection cadence is 60 seconds and stale
means three missed intervals (180 seconds). Local collection freshness is **not** remote
acknowledgment freshness.

## Optional foreground service

When a destination/signal is separately authorized, the worker also schedules export every
300 seconds by default; serving does not itself grant export consent. Offline/failed delivery
does not make a fresh local snapshot remotely acknowledged.

```sh
ssgrep metrics serve --port 9469
```

This explicitly starts collection plus an authenticated server on `127.0.0.1`; it is independent
of MCP stdio. One worker owns a profile. A port conflict fails rather than choosing another port
(`--port 0` explicitly requests an ephemeral port). Stop with Ctrl-C or SIGTERM.

Only `GET /healthz` (and HEAD) is unauthenticated and content-free. `/metrics`, `/v1/snapshot`,
and `/v1/export-status` require `Authorization: Bearer TOKEN` using the **owner-only local token
file printed by serve**. Never put the token in a URL, expose the service remotely, or use that
token as an organization credential. Host is restricted to approved loopback names, CORS is
not enabled, and endpoints are read-only. No endpoint exposes raw receipts/transcripts.

Before a snapshot exists, `/v1/snapshot` returns 503 `snapshot_unavailable`; `/metrics` reports
availability/health rather than invented zeros. Snapshot ETags support conditional reads.
`/v1/snapshot?view=acknowledged&destination=DESTINATION_ID&signal=metrics` selects the exact
last fully acknowledged snapshot for that destination/signal; `signal=logs` selects its separate
acknowledgment. These placeholders must be replaced with a configured ID. Missing acknowledgment
is 404 `not_acknowledged`; expired content is reported rather than replaced with newer data.
Acknowledged means collector acceptance, **not backend durability**.

Service-manager setup is **optional and manual**, never installed by setup. Illustrative
invocation fragments (replace the executable path and use the same user/profile/environment):

```ini
# systemd user service fragment, not a complete unit
ExecStart=/absolute/path/to/ssgrep metrics serve --profile personal-local --port 9469
```

```xml
<!-- launchd ProgramArguments fragment, not a complete plist -->
<key>ProgramArguments</key>
<array><string>/absolute/path/to/ssgrep</string><string>metrics</string><string>serve</string><string>--profile</string><string>personal-local</string><string>--port</string><string>9469</string></array>
```

## Privacy, retention and deletion

Accounting storage is OS-account-protected (directories 0700, files 0600), not encrypted or
employee attestation. Defaults retain private evidence for 30 days, compact receipts/aggregates
for 400 days, and outbound spool for seven days or 100 MiB per destination. Managed policy can
narrow permissions/retention. Evidence expiry may leave a retained total **non-recomputable**;
privacy purge can remove the total itself. Setup `--policy` accepts a managed JSON policy;
organization roots and destinations must be explicitly approved, not inferred from cwd.

Reads never repair permissions or damaged state. Unsafe group/world access, corrupt policy,
or a malformed pending-purge marker produces an explicit failure without changing the files.
After checking ownership and the intended data root, repair permissions deliberately or restore
a consistent backup; do not delete recovery markers to make the warning disappear. Owner-only
read-only files/directories remain readable without being made writable. A declared whole-state
restore requires `metrics collect --new-epoch`.

```sh
ssgrep metrics purge --before 2026-10-01 --dry-run
ssgrep metrics purge --before 2026-10-01 --yes
```

Add `--workspace /absolute/workspace/path` to narrow the scope. Purge touches owned accounting
data only, not original transcripts, notes, search index or other profiles. Preview first;
confirmed deletion is permanent. Remote log tombstones, when permitted, are deletion requests,
not proof of backend erasure. Aggregate metrics may be non-reversible; the organization must
handle backend retention and erasure. In-flight requests cannot be recalled.

Local measurement uses no LLM and needs no network. Search/index need a cached embedding model;
optional organization export is a separately authorized network action. Optional live-agent
validation can call an LLM only with explicit authorization; it is not part of runtime accounting.
See [organization export](organization-metrics.md) for offline preview, independent signal
consent and collector examples. Never use this ledger to rank people's productivity.
