# Local savings ledger

Savings is an **opt-in local ledger**, not a claim that a lookup caused a productivity gain.
It separates observed executions and visible output from modeled reuse credit. Collection and
remote export both start disabled. The default profile is `personal-local`, even inside an
employer's repository; every command below accepts `--profile`.

## First use

```sh
ssgrep savings
ssgrep savings setup
ssgrep savings collect
ssgrep savings --group-by task
ssgrep savings status
```

The first summary explains the disabled state without enabling anything. Setup previews roots,
retained fields and retention, then asks for consent. Decline to cancel without changes; in
non-interactive use, `ssgrep savings setup --yes` explicitly authorizes the displayed defaults.
Setup does not backfill history, send data, start a worker, or install a service. Use ordinary
`ssgrep search` and `ssgrep show` after setup, then collect again. An empty ledger is not proof
of zero benefit.

The summary reads the latest completed snapshot and refreshes it once when missing or stale;
use `--no-refresh` to use the existing snapshot without collecting. `--from` and `--to` accept UTC
dates or RFC3339 timestamps (end exclusive). Summary filters include `--project`, `--runtime`,
and `--session`; group with `--group-by` using `task`, `runtime`, `project`, or `day`.
The `lookups` view also accepts `--task` and `--children` with `--session`. Use `--json` for
machine-readable output. Do not publish private local JSON unreviewed.

Grouped summaries show up to 20 rows. If more exist, the summary says so but has no page switch.
Narrow the project, runtime or session selection to focus the grouped view.

Suggested commands use a shell-comment label (for example `# Next:`) above the command,
with the explanation before that block. Copy the complete block, including its label and
backslash continuations, into a POSIX shell without deleting the label. Wrapped arguments
retain their exact value, including quotes, shell metacharacters and wide Unicode characters.
Accounting output is preformatted and is not reflowed a second time. Below the 30-column
minimum layout width, a terminal may visually soft-wrap lines; ssgrep does not insert extra
newlines into the command's arguments.

For a uniquely matched CLI lookup, collection uses the journal's captured canonical working
directory for both execution counts and exposure estimates, so aliases such as `/var` and
`/private/var` do not split one project into separate rows. Savings reads use those committed
identities without resolving historical symlinks again. An MCP server's working directory is
not substituted for its caller's project.

```sh
ssgrep savings --from 2026-10-01 --to 2026-10-08 --group-by day --json
ssgrep savings history --from 2026-10-01 --to 2026-10-08
```

Inspect every lookup, including calls without a reuse receipt, and page collection warnings:

```sh
ssgrep savings lookups --json
ssgrep savings status --details --json
```

Both views read completed snapshots only. Lookups continue with the returned `--page` token and
unchanged selection; the token binds its snapshot and generation. Status details use `--page`
to continue the warning list. Expired generations fail explicitly rather than silently paging
newer data. Lookup filters intersect; exact intervals exclude untimed native calls, otherwise
explicitly unplaced. Lookup rows separate execution, exposure, attribution, reuse, eligibility
and outcome; their scenarios have `valuation_scope=lookup_nonadditive`, not additive receipt
totals. They expose metadata, never raw query/output content, and causal usefulness stays unknown.
`reuse_evaluated` distinguishes an actual no-reuse observation from skipped evaluation; skipped
reuse is unavailable and `useful_outcome=not_evaluated`, not a negative finding.
Status details can show private warning text only with `--details --include-text`; it is hidden by default.
Native source warnings are snapshot-wide; selection filters also select ledger notes, not make
source warnings project-specific. `total` counts unique selected warning identities, while
`alias_duplicates` is separate. Page tokens are continuation handles, not authorization. Do not
publish local IDs, page tokens or diagnostic JSON unreviewed.

Read `complete`, missing reasons and bounds beside each estimate. A known subtotal is not
necessarily a complete net or a lower bound; unavailable evidence is not a complete zero.
Raw journal nanoseconds are capped per modeled operation and summed before deriving service
milliseconds or work-seconds. Native source-gap observations keep millisecond precision.
The active-occupancy mapping is `journal_active_proxy_ns_v2`; it is not human time saved.
Bounds round lower endpoints down and upper endpoints up: a −1 s whole-unit point can have
raw-enclosing bounds [−2 s, −1 s]. Missing raw duration has no finite lower net bound; old
rounded timing is not silently treated as exact nanoseconds.

To inspect a receipt, replace `RECEIPT_ID` below with an ID from the summary:

```sh
ssgrep savings explain RECEIPT_ID
ssgrep savings explain RECEIPT_ID --include-content
```

The default explanation shows allocations, exclusions, costs, exposure links and missing
evidence. Receipts for different facts are **not additive**: shared exposure costs can repeat.
Use summary/task rollups for unioned totals, not a sum of receipt values.
`--include-content` requests retained private local content; it can remain unavailable when
the optional cache is off, expired or incomplete. It does not fetch missing sources, collect
or index. History distinguishes current and archived replacement revisions; never sum them.
Read the [methodology](accounting-methodology.md) before interpreting nets.

## Collection and lifecycle

```sh
ssgrep savings collect 2026-10-01 2026-10-08 --backfill
ssgrep savings collect --recompute
ssgrep savings collect --repin-frequency
ssgrep savings collect --new-epoch
ssgrep savings pause
ssgrep savings resume
```

- Ordinary collection is one bounded pass. Backfill requires both bounds and is separate from
  live counters; it cannot be combined with recompute. Recompute uses retained frozen evidence,
  not newly substituted source text.
- New lookups automatically use the current supported frequency generation; old lookup inputs
  remain frozen. `--repin-frequency` is an explicit replacement/recomputation operation, not
  routine index maintenance. Never add replacement estimates to the previous namespace.
- If collection started before the index existed, run `ssgrep index`, then
  `ssgrep savings collect --recompute --repin-frequency`. Indexing alone does not replace
  previously pinned unavailable rarity evidence.
- Bounded backfill/recompute passes persist classifier, dependency and root progress.
  Follow the reported resume command; completed roots and replacement archives are not repeated.
  A pass may finish its already-started atomic work after its cooperative deadline.
- `replay_unavailable` is terminal partial evidence, not unfinished work: restore original
  transcripts, then run ordinary `ssgrep savings collect`. Repeating unchanged input cannot
  recover it. Accepted measured values remain readable; missing benefit is not invented.
  If a historical backfill is still pending after recovery, the next command preserves its
  original interval; a completed live pass does not falsely complete that historical job.
- `backfill_blocked` means a missing, unreadable or unsupported input needs attention.
  Inspect `savings status --sources`, resolve it, then use the printed historical retry.
  Unchanged retries are explicitly nonresumable.
- A `reduction_blocked` report names an observed minimum `--budget-seconds` override.
  This is not a prediction of total runtime. Use the printed command rather than an endless
  same-budget retry.
- After purge invalidates rarity authority, first establish a fresh approved narrowed index
  scope, then explicitly reapprove it with
  `ssgrep savings collect --repin-frequency --reset-frequency-authority`.
  Reset without repinning fails with `reset_frequency_requires_repin`. Mixed, stale or
  unapproved index authority remains unavailable; this command does not rewrite the index
  or restore purged evidence.
- `--new-epoch` declares a reset/restored backup and restarts live counters, not ledger history.
  Ordinary restart preserves counters; an undeclared whole-state restore cannot be detected.
- Pause stops capture, collection and sending, preserving readable history. Resume does not
  silently import the paused interval; use explicit backfill if wanted. Use pause to stop local
  measurement, and `ssgrep savings setup --no-export` to revoke the profile's remote exports.
- Ctrl-C stops a foreground worker; it is not a permanent opt-out. Use pause for that.

`ssgrep savings status` distinguishes disabled, no snapshot, fresh, stale, paused and error
conditions. Missing or corrupt data is not zero savings. A previous completed snapshot stays
readable during collection; freshness ages while paused. Default collection cadence is 60 seconds
and stale means three missed intervals (180 seconds). Local collection freshness is **not** remote
acknowledgment freshness.

## Save or share the latest numbers

```sh
ssgrep savings export --output savings.json
ssgrep savings export --output - --from 2026-10-01 --to 2026-11-01
ssgrep savings export --dry-run
```

Local JSON includes retained usage, modeled estimates, every lookup/history page and coverage.
Unknown remains unavailable; original receipt values and archived revisions must not be added
to current totals. Dates select UTC days, end exclusive. No collection or indexing occurs.
Use `ssgrep savings collect` first when you want a refreshed report.

To send to an organization, configure its policy, approve it separately with
`ssgrep savings setup --export-to DESTINATION`, then run `ssgrep savings export DESTINATION`.
Export automatically includes all permitted **and approved** categories. Omit the destination
when only one is approved; use `--dry-run` to inspect without writes/network.
The five export options are `--output/-o`, `--from/-f`, `--to/-t`, `--profile/-p` and
`--dry-run/-n`. Date filters apply only to local reports; remote counters stay cumulative.
See [organization metrics](organization-metrics.md#enable-only-what-you-intend-to-send)
for privacy, approval and receiver semantics.

## Optional foreground service

When a destination/signal is separately authorized, the worker also schedules export every
300 seconds by default; serving does not itself grant export consent. Offline/failed delivery
does not make a fresh local snapshot remotely acknowledged.

```sh
ssgrep savings serve --port 9469
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
ExecStart=/absolute/path/to/ssgrep savings serve --profile personal-local --port 9469
```

```xml
<!-- launchd ProgramArguments fragment, not a complete plist -->
<key>ProgramArguments</key>
<array><string>/absolute/path/to/ssgrep</string><string>savings</string><string>serve</string><string>--profile</string><string>personal-local</string><string>--port</string><string>9469</string></array>
```

## Privacy, retention and deletion

Accounting storage is OS-account-protected (directories 0700, files 0600), not encrypted or
employee attestation. Defaults retain private evidence for 30 days, compact receipts/aggregates
for 400 days, and outbound spool for seven days or 100 MiB per destination. Managed policy can
narrow permissions/retention. Evidence expiry may leave a retained total **non-recomputable**;
privacy purge can remove the total itself. Setup `--policy` accepts a managed JSON policy;
organization roots and destinations must be explicitly approved, not inferred from cwd.

By default accounting retains compact references/fingerprints, not copied prompts, queries,
tool arguments or outputs. Replay verifies bounded authorized original native evidence;
unavailable or changed originals limit recomputability without inventing replacement evidence.
The optional plaintext cache requires separate consent and age/byte bounds:

```sh
ssgrep savings setup --evidence-cache --cache-days 1 --cache-bytes 4194304
ssgrep savings setup --no-evidence-cache --yes
```

The cache may contain prompts, queries, tool arguments and outputs locally; it is not exported
or accounting authority. Limits are 1–30 days and at most 33,554,432 bytes; explicit opt-in
defaults to one day and 4,194,304 bytes. Managed policy may narrow or prohibit it. Revocation
erases cached copies. Accounting purge does not delete native transcripts or the retrieval index.

Legacy journals may still have old processes writing them. Stop **all pre-v2 capture
processes** before explicitly acknowledging retirement:

```sh
ssgrep savings collect --journal-legacy-writers-stopped
```

Do not use that acknowledgment while an old writer can still append. Normal cooperative
journals need no legacy acknowledgment. Expiry runs with collection and a durable accepted
checkpoint, not a promise of immediate wall-clock deletion. Busy writers, partial tails,
changed sources or maintenance-budget limits can defer physical cleanup; status/warnings
report those holds. Do not delete recovery files or keys to bypass them.

Reads never repair permissions or damaged state. Unsafe group/world access, corrupt policy,
or a malformed pending-purge marker produces an explicit failure without changing the files.
After checking ownership and the intended data root, repair permissions deliberately or restore
a consistent backup; do not delete recovery markers to make the warning disappear. Owner-only
read-only files/directories remain readable without being made writable. A declared whole-state
restore requires `ssgrep savings collect --new-epoch`.

```sh
ssgrep savings purge 2026-10-01 --dry-run
ssgrep savings purge 2026-10-01 --yes
```

Add `--workspace /absolute/workspace/path` to narrow the scope. Purge touches owned accounting
data only, not original transcripts, notes, search index or other profiles. Preview first;
confirmed deletion is permanent. Remote log tombstones, when permitted, are deletion requests,
not proof of backend erasure. Aggregate metrics may be non-reversible; the organization must
handle backend retention and erasure. In-flight requests cannot be recalled.

## Turning it off and rolling back

Measurement is opt-in; every step below is reversible except confirmed purge.

```sh
ssgrep savings pause                       # stop capture, collection and sending; keeps data
ssgrep savings setup --no-export --yes     # revoke organization exports; keep local history
ssgrep savings purge 2099-01-01 --dry-run  # preview deleting every owned accounting record
ssgrep savings purge 2099-01-01 --yes      # delete them (spool, receipts, history, snapshots)
```

Pausing is immediate for new lookups and for the next collection/export tick; an export
already sent cannot be recalled. To return a profile to the never-enabled default, remove the
`accounting/` directory under the ssgrep data root (`ssgrep --json status` reports it as
`data_dir`; `SSGREP_DATA_DIR` overrides it):
the policy file lives there, so the profile reads as `disabled` again and search, show, notes
and the retrieval index are unaffected. Downgrading to an ssgrep release without the ledger
leaves that directory unread; it can be deleted at any time. Managed policy files are applied
only through `savings setup --policy`, so removing the directory also discards any approved
destinations and roots.

Local measurement uses no LLM and needs no network. Search/index need a cached embedding model;
optional organization export is a separately authorized network action. Optional live-agent
validation can call an LLM only with explicit authorization; it is not part of runtime accounting.
See [organization export](organization-metrics.md) for offline preview, independent signal
consent and collector examples. Never use this ledger to rank people's productivity.
