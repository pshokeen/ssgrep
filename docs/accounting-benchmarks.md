# Accounting benchmarks and validation results

This is the single home for empirical accounting results and validation learnings.
Contracts live in [methodology](accounting-methodology.md); commands live in
[the local guide](savings.md). Results are revision-specific, not release approval.

## Release-readiness validation (2026-10-22)

Scope: candidate source based on `b5a60ae` plus the native-evidence, bounded-progress and
simplified-export repairs. These controlled results do not calibrate real-world productivity
or billing savings. Earlier rows below keep their original revision and failure labels.

### Controlled native evidence and accounting

Method/environment: macOS arm64, Python 3.11.15, cached
`lightonai/answerai-colbert-small-v1`, real CLI search/show/index/collect/export, isolated
HOME/data/runtime roots. No paid model or autonomous solver was used. The retained private
generator produced 943 sessions, 1,019 episodes and 1,962 indexed chunks, including rarity
filler. Four selected regression consumers executed actual lookups; a warm-up adds the fifth
lookup. Source-plan expectations were calculated independently of the product reducer.

Values below are conservative / central / optimistic. Context tokens use `chars_div4_v1`,
not a provider tokenizer or billed-token measure.

| Controlled case | Token credit | Token debit | Token net | Modeled source-work credit (s) |
| --- | ---: | ---: | ---: | ---: |
| A17: pre-setup Pi source | 232 / 366 / 448 | 720 | −488 / −354 / −272 | 13 / 35 / 37 |
| A21: post-setup Pi source | 46 / 63 / 63 | 729 | −683 / −666 / −666 | 2 / 7 / 7 |
| A24: legacy-wrapped Codex source | 43 / 60 / 60 | 700 | −657 / −640 / −640 | 2 / 7 / 7 |
| C05: fact already visible before lookup | 0 / 0 / 0 | 772 | −772 / −772 / −772 | 0 / 0 / 0 |
| Modeled aggregate | 321 / 491 / 573 | 2,922 | −2,601 / −2,431 / −2,349 | 17 / 49 / 51 |

- All nine positive token-credit values and nine modeled source-work-credit values match
  the independent source-plan oracle exactly. C05 receives no discovery credit.
  The aggregate also matches raw-character sums rounded once; adding separately rounded
  project values can lose fractional tokens and is not the aggregation rule.
- The first collection read 1,655,124 bytes and completed in one pass (1,846 ms reported,
  zero lag). This is one controlled observation, not a performance distribution or SLA.
- A no-change collection advanced generation 1 → 2 without changing usage, estimates or
  lookup rows. Repin/recompute advanced to generations 3 and 4; only the first declared
  frequency-revision provenance changed, not quantities. Overlapping October backfill
  read the same 1,655,124 bytes into generation 5 without doubling usage or estimates.
- All selected token nets are negative. Some modeled work nets remain unavailable where
  their evidence/bounds are incomplete; known work subtotals are not converted into complete
  time-saved claims.
- Actual local export produced all five retained lookup rows, preserved unknowns and
  nonadditive semantics, used mode `0600`, and matched stdout output.
- A foreground `savings serve` instance was exercised directly: health returned 200,
  unauthenticated metrics 401, authenticated snapshot/metrics 200, and conditional snapshot
  retrieval 304. Scraped executions were five and lifetime known credits matched the aggregate
  above. The endpoint token file was `0600`, responses did not contain the token, and SIGTERM
  stopped the owned service with exit 0.

### Persistence metadata and older versions

Actual CLI and real MCP stdio exercised save, enrichment, correction, search/show and server
restart. One shared work ID retained two immutable revisions across three note bindings:
137 seconds corrected to 149 seconds, rather than three duplicated investments. A native
fixture inferred explicitly recorded provider/model and 23 seconds. Missing historical
metadata remained absent; no token counts or prices were invented.

The published PyPI `ssgrep==2.0.1` was installed separately and confirmed not to contain the
new work-metadata module. It read newly authored notes and rebuilt their index; the candidate
then still read revision 2 / 149 seconds from the separate metadata. Conversely, the published
version authored a note that the candidate read without adding nonexistent work metadata.
This is an exercised old/new round trip, not just a schema-shape assertion.

The built wheel was installed into a separate target and its import origin verified outside
the editable tree. Export help, structured status and private report output all exited 0.
The candidate retains the development version until the normal release process; this is not
a publication of a new package version.

Isolation limit: guarded worktree settings/index stayed unchanged. The separate real global
index stayed unchanged through the MCP restart check, but changed during a later guard
interval; attribution is unknown. All smoke command environments explicitly selected the
sandbox data roots, and no globally-unchanged claim is made for that later interval.

### Verification boundary

The required offline inventory contains 6,229 cases, plus one separately passing real-model
network case. The complete core/heavy runs and affected reruns exercised that inventory;
obsolete CLI assertions and the source-recovery handoff failures were corrected. Final local
combined source-statement coverage is **100.00% of 34,488 statements**; stale coordinates
were cleared and affected modules rerun after source movement. Three pre-existing cases
were skipped in the full core lane. Counts from overlapping reruns must not be added.

Lint, formatting, typing, source-size, dependency, package and repository-hygiene checks
passed. Independent review findings were resolved, including repeated Codex decoding,
preview approval disclosure, output/cursor error classification and blocked/restored
historical-scan recovery. The final candidate wheel was installed separately and exercised.
Final-head GitHub CI remains the delivery gate, not an inference from local coverage.

The first pushed-head CI run found an ANSI-sensitive option-substring assertion in the
cold-start help test. Its import-isolation checks remain; the incidental marker and duplicate
status case were removed. All 32 cold-start/export-help cases then passed with both
`GITHUB_ACTIONS=true` and `FORCE_COLOR=1`. This was a test-formatting failure, not a product
option or import regression.

The Linux/root full-suite leg then found an overbroad whole-HOME fingerprint assertion:
amd64 translation creates `.cache/rosetta` independently of ssgrep. Invalid-output tests now
fingerprint the requested output directory and accounting state, preserving their actual
no-mutation contract. The correction passed all four macOS cases and a focused Linux/root
container run: ten passed, with the permission-denial case skipped because root bypasses
directory permissions. No production behavior, CI gate or test inventory was reduced.

Private retained evidence: `runtime-results.json`, `oracle-checks.json`,
`replacement-results.json`, `final-index-status.json`, `legacy-compatibility-results.json`,
`final-package-results.json`, `http-service-results.json`, and the MCP/inference result
documents. No new checked-in lab or temporary validation driver is part of the shipped tree.


## Consolidation baseline (`consolidate-savings-ledger`, task 1.1)

Measured at `78eeff0` against base `6bf085e`, before any consolidation step:

| Measure | Value |
| --- | ---: |
| Files changed vs base | 588 |
| Added lines | 169,221 |
| `src/ssgrep/accounting` Python modules (recursive, incl. `readers/`) | 233 |
| `tests/accounting` Python modules (recursive) | 235 (173 top-level, 22 `readers/`, 40 `airline/`) |
| `eval/accounting_airline` Python modules | 66 |
| Original split CLI baseline | 9 metrics-group + 4 savings-group subcommands (13 total); 4–18 options per command |
| Earlier consolidated public accounting CLI | One `savings` group; 16 leaf invocation forms before the later single-command export amendment |

Counts are reproducible with `find src/ssgrep/accounting -name '*.py' | wc -l` and the equivalent `tests/accounting` and `eval/accounting_airline` finds. Remediation fixes listed under "Remediation revalidation status" below are not reflected in these counts.


## Historical remediation revalidation status (pre-consolidation)

**Measured uncommitted performance freeze, now superseded by repairs:** source SHA-256
`45e0ba2df6382c3e12b96fd62a51cd99b69e6d25ef8805c7dc0fd598c89eb833`
over 332 source files, atop `b147016d2322768d9aee4ead49004a5b068078c9`.
The comparison baseline remains `6bf085e8269abbd414df16311d3b4f8a3233b53c`.
This is not a merged revision or release approval. Previous full-suite and focused
results have different boundaries; overlapping test counts must not be added.
The failed performance results below remain immutable; later startup/transport repairs
require their own frozen measurements and do not inherit a passing gate.

The newer v4 repair freeze is atop published `2f0f620`. Its focused checks and real
CLI/receiver smokes below do not clear the still-running complete suite, coverage,
representative-history, resource, native A/B or corrected-commit review gates.

| Gate or actual surface | Observed result | Evidence boundary |
| --- | --- | --- |
| Types, lint, formatting, source-size, dependencies and strict OpenSpec | Passed on the v4 repair freeze | 766 Python files format-clean; complete suite and unchanged 100% coverage gate remain separate |
| Previous complete offline suite and 100% branch-coverage requirement | **5,725 passed; 97 failed; 9 setup errors; 95.44% coverage** | Earlier source freeze; 6 skipped and one network deselected; 51m56s; failed gate, not release approval |
| **Consolidated head `908adaf`: complete offline suite and 100% statement-coverage gate** | **5,909 passed; 3 skipped; 1 network deselected; 100.00% (33,432 statements); 40m19s** | `uv run poe test-cov`; ruff format/check, `ty check`, sizecheck (no failures), deptry and strict validation of both OpenSpec changes also pass. Five operating-system race lines carry `# pragma: no cover` with reasons; this is a statement-coverage result, not proof of estimator calibration |
| Consolidated head `908adaf`: three isolated drivers | 57 passed in 1m45s | Real-binary CLI journey at 40/80/120 columns and first-use journey; company consent/dry-run/send/revoke journey against a lab receiver (the pre-amendment export subcommands, since replaced by `savings export` and `savings setup --export-to`); reader CLI census cases; collect/serve/stop restart journey. Synthetic corpus and loopback receiver only |
| Single-pass independent review of `908adaf` (`syncade --max-rounds 1`, no inline diff: the full branch diff exceeds the argument-size limit) | **NO-SHIP from the adversarial reviewer, 1 blocker, fixed; the second reviewer timed out at 1800s with no output** | Blocker: `.github/workflows/ci.yml` imported `tests.accounting.prom_lab` after it moved to `tests.accounting.serve.prom_lab`, breaking the required-integration pull and preflight steps. Reproduced with `ModuleNotFoundError`; fixed in the two import lines; both CI snippets' imports and every `tests/` path in the workflows and `pyproject.toml` lanes now resolve. No other findings. The synthesizer did not run because one reviewer failed, so this is one adversarial review, not a blind multi-judge clearance, and no second round was run per the final-gate rule. The reviewer also could not run the full coverage gate or `openspec validate` in its snapshot (`openspec/` is gitignored) and did not exhaustively diff every renamed test |
| Prior focused migration/purge/recovery suite | 178 passed | Predates final typing and phase extractions; not a substitute for the full run |
| Prior focused native identity/outcome/exposure suite | 85 passed | Real native-file/API controls, not provider conversations |
| Prior independent raw-input oracle suite | 103 passed | Synthetic independent arithmetic; later typing changes still need integrated clearance |
| Published conflict authority and replay suite | 71 passed | Thirty new authority cases; real native/compact claims, variant orders, future/ancestor cutoffs and strict canonical accounting |
| Published conflict valuation API smoke | Passed four cases | Disputed prior context stays unavailable; supported lookup cost survives both primary orders; no provider/native conversations |
| Large-corpus real-Lance API smoke | Passed | Capacity, immutable-generation and privacy observations are recorded once below; not representative calibration |
| Evaluated real CLI journey | 13 successful steps; 69/36/36 human-output lines at 40/80/120 columns | Setup, actual lookup, collect, read views and pipes; not positive-ROI evidence |
| Evaluated HTTP and real OTLP Collector | Authorization statuses 401/401/200; acknowledged generations 1 then 2 | Unavailable net omitted, known subtotal retained; transport acceptance is not backend durability |
| v4 render/command/partition/publication regression suite | 103 passed | Actual CLI/MCP render faults, copied shell commands at supported widths, strict manifest authority and restoration controls; overlaps earlier focused tests |
| Accepted-journal loss and genuine v3 retirement CLI smoke | Both passed | Missing journal rejected without checkpoint replacement, restored journal recovered; actual collect preserved original accepted v3 bytes and selected v4 |
| Exact published-code receipt upgrade | Three original receipt UTF8/SHA pairs and six native-file hashes preserved | Actual old writer, checkpoint-before-publication crash, restart and original history retained |
| Copied versus distinct lookup migration | Copied 78 became 39 tokens; two distinct lookups stayed 78 | Original namespaces retained separately; scoped erasure preserved the other workspace |
| Actual-worker profiler mechanics | Eight cases completed | Counts 0/2, new observations 0/1, both root shapes; **not performance evidence** |
| Paired startup/capture and full collector resource distributions | Original startup targets missed; documented trade-off explicitly accepted. Full collector matrix interrupted for source repairs | No p95 startup guarantee; direct warm accounting phases and original paired tails remain separate; partial resource samples are not a passed matrix |
| Blind full-PR review of `2f0f620` | **Not ship-ready** | One parsed review and eleven recovered findings from a schema-failed review; raw responses retained, no automatic model retry |
| Fresh authorized native A/B | Not run | Historical trials and component reviews cannot clear the final delta |

Earlier failures remain evidence: the first 325-target run timed out with failures;
the latest pre-correction airline integration had 213 passes and seven failures.
Native result pairing and timing-footer parsing were repaired; changed same-origin
output now requires conflict/unknown-cost assertions rather than a fabricated unchanged net.
The complete run exposed further airline, collection/replay, organization-fixture and
UX contract failures. Focused repairs and API controls do not supersede the failed
integrated gate or clear its missing coverage; a new complete run remains required.

Twelve bounded policy CLI/MCP cases also ran; these are not complete policy/runtime coverage.
- A frozen real-model index contained eight sessions, eight episodes and sixteen chunks.
  Preflight checked output equivalence, **not bytecode identity**. Real MCP stdio with the
  stock cached model exercised four operations each on baseline-disabled, current-disabled
  and current-enabled configurations: known search (eight hits), known-reference show,
  unknown-reference error and empty-query error. Payloads had identical SHA-256
  `b5cda2d3a2840f8bdcf958c4d696121c931110dbb49b0f703a35378a52001cf0`.
  The capture proof recorded four unique completed MCP executions, trusted launch-workspace
  attribution rather than server-cwd attribution, and zero damaged records in that exercise.
  This does not establish complete production index coverage or final-revision validation.
- Extended real-index fault exercises recorded 48 CLI responses and 48 real MCP protocol
  responses: three normal modes and nine fault modes, each with four operations. Payloads
  and exit behavior matched the baseline throughout that exercise; every intended fault
  was positively observed, including kernel `EFBIG` and injected `ENOSPC`. Successful search
  returned eight ranked hits and show returned the expected reference. These bounded
  observations do not establish universal fail-open behavior or final acceptance.
- A policy-cache authority exercise changed policy permissions to `0644` without changing
  mtime: capture stopped, then recovered after permissions returned to `0600`. This is one
  observed permission-transition case, not comprehensive policy-cache security evidence.
- An earlier fresh CLI journey failed with `stateCorrupt`; the current actual CLI
  journey in the table supersedes that narrow surface check, not the pending full suite.
- A stale private benchmark baseline had lost its canonical source files. It was not
  accepted: a separately extracted archive matched all 559 Git files and modes of the
  same pinned baseline, including the original 72-file source hash. The failed archive
  and provenance remain retained; the responsible process is unverified.
- An incomplete private model cache was likewise retained. Its complete replacement
  matched the original 30-file model hash and used the same pinned cached model, without
  downloading weights or substituting an embedding implementation.

Method/environment: isolated controlled CLI/MCP, identity and purge exercises, frozen
real-model-index preflight and real MCP stdio using a stock cached embedding model.
Exact platform/dependency versions were not supplied in this summary.
Private evidence is retained under artifact-relative `apply-revision/`:
`authority-lookup-smokes.json`, `named-replacement-smoke.json`,
`scoped-archive-purge-smoke.json`, `disabled-cli-integration-smoke.json`,
`task-19.4/manifest.json`, `mcp-real-index-observations.json`,
`mcp-real-index-capture-proof.json`, `cli-real-index-faults.json`,
`mcp-real-index-faults.json`, `policy-cache-authority-smoke.json` and
`empty-local-journey.json`.
These paths identify private evidence, not shipped fixtures or current passing gates.
No private transcripts or temporary absolute paths are published. Central verification after
implementation freeze must establish the final revision's result.

## Measured release-overhead gate

The `45e0ba…` source above completed randomized base/current pairs against the same
owned frozen index/model and canonical baseline. Every command returned its expected
exit and identical normalized output; source and index hashes were unchanged. The
loaded lane used 18 owned CPU-contention workers. “Unloaded” means no owned contention,
not a globally quiescent host. Quantiles are nearest-rank; all raw samples and bootstrap
intervals remain private.

**Cold-process added wall time:** proposed median <=10 ms and p95 <=20 ms were not met.

| Lane | Operation | Paired trials | Delta p50 (ms) | Delta p95 (ms) |
| --- | --- | ---: | ---: | ---: |
| Unloaded | Empty search | 30 | 31.114 | 50.519 |
| Unloaded | Valid show | 30 | 28.988 | 89.015 |
| Unloaded | Status | 30 | 25.950 | 62.756 |
| Unloaded | Embedded search | 30 | 33.591 | 115.332 |
| Loaded | Empty search | 30 | 32.113 | 62.182 |
| Loaded | Valid show | 30 | 42.072 | 164.997 |
| Loaded | Status | 30 | 32.375 | 204.938 |
| Loaded | Embedded search | 30 | -10.390 | 140.125 |

**Warm API leaf/wrapper paired differences:** these include independent leaf-execution
variance, not just time inside accounting. Proposed capture p95 <=2 ms / p99 <=5 ms
cannot be cleared from the failed real-search rows.

| Lane | Operation | Capture | Trials | Delta p95 (ms) | Delta p99 (ms) |
| --- | --- | --- | ---: | ---: | ---: |
| Unloaded | Show | Disabled | 200 | 1.259 | 4.503 |
| Unloaded | Show | Enabled | 200 | 1.647 | 7.267 |
| Unloaded | Empty search | Disabled | 200 | 0.104 | 0.178 |
| Unloaded | Empty search | Enabled | 200 | 0.435 | 0.530 |
| Unloaded | Real search | Disabled | 200 | 8.150 | 15.053 |
| Unloaded | Real search | Enabled | 200 | 4.671 | 12.288 |
| Loaded | Show | Disabled | 200 | 0.962 | 2.449 |
| Loaded | Show | Enabled | 200 | 1.944 | 4.253 |
| Loaded | Empty search | Disabled | 200 | 0.063 | 0.087 |
| Loaded | Empty search | Enabled | 200 | 0.515 | 0.564 |
| Loaded | Real search | Disabled | 200 | 13.667 | 22.100 |
| Loaded | Real search | Enabled | 200 | 11.909 | 20.554 |

Every enabled operation produced the expected journal events, and disabled controls
produced none. The disabled real-search tails also miss the proposed limits: this
requires separate accounting-phase attribution, not subtracting marginal quantiles
or declaring the end-to-end gate passed. Cold first-enabled calls and complete
distributions remain in `task-19.4/results-20261015T135852Z.json`.
An import-time diagnostic subsequently identified accounting read-view/policy imports
triggered by unrelated command discovery. Those imports were deferred without removing
commands; new-source distributions remain pending.

### Direct accounting-phase attribution

Source `2d2a1576e833…` measured the actual nonoverlapping outer
`capture.begin/fail/finish_search/finish_show` spans around real API calls, excluding
the lookup leaf. Warmups were separate. The same cached model/index and 18-worker
loaded condition were retained; source, model, index, driver and lockfile stayed
unchanged. Bare/instrumented no-op controls, cold calls, per-phase CPU and whole-call
timings remain in the private artifact.

| Lane | Operation | Capture | Measured calls | Phase p95 (ms) | Phase p99 (ms) |
| --- | --- | --- | ---: | ---: | ---: |
| Unloaded | Show | Disabled | 200 | 0.188 | 0.244 |
| Unloaded | Show | Enabled | 200 | 0.945 | 1.289 |
| Unloaded | Empty search | Disabled | 200 | 0.067 | 0.088 |
| Unloaded | Empty search | Enabled | 200 | 0.472 | 0.555 |
| Unloaded | Real search | Disabled | 200 | 0.199 | 0.265 |
| Unloaded | Real search | Enabled | 200 | 1.086 | 1.298 |
| Loaded | Show | Disabled | 200 | 0.216 | 0.280 |
| Loaded | Show | Enabled | 200 | 1.081 | 1.291 |
| Loaded | Empty search | Disabled | 200 | 0.104 | 0.138 |
| Loaded | Empty search | Enabled | 200 | 0.572 | 0.618 |
| Loaded | Real search | Disabled | 200 | 0.230 | 0.328 |
| Loaded | Real search | Enabled | 200 | 1.118 | 1.348 |

Each enabled operation accepted 440 journal events across measured calls plus
warmups; disabled accepted none. These measured accounting spans fall within the
proposed capture-only reference limits. They **do not relabel the original failed
end-to-end paired tails**, establish a hard latency guarantee, clear cold startup,
or measure full-collector duty. Private evidence:
`capture-phase-profile-20261015T144732Z/report.json`.

### Approved startup release decision

The user explicitly accepted the documented cold-start cost rather than requiring
further dispatcher changes solely to meet the proposed numeric startup targets.
The final measured source was
`94f0cc3456415265b27e54b644cddcc309d6e04019f2a4fc43a9646f09be5b9b`.
All paired commands retained their expected exits and normalized output; source
and frozen-index hashes stayed unchanged. The loaded lane retained 18 owned workers.

| Lane | Operation | Pairs | Sample median delta (ms) | Delta p95 (ms) |
| --- | --- | ---: | ---: | ---: |
| Unloaded | Empty search | 30 | 15.108 | 26.252 |
| Unloaded | Valid show | 30 | 17.377 | 58.056 |
| Unloaded | Status | 30 | 10.364 | 73.978 |
| Unloaded | Embedded search | 30 | 5.632 | 105.144 |
| Loaded | Empty search | 30 | 12.671 | 34.871 |
| Loaded | Valid show | 30 | 18.018 | 56.589 |
| Loaded | Status | 30 | 13.021 | 65.498 |
| Loaded | Embedded search | 30 | 3.154 | 83.817 |

Sample median averages the two middle observations; it is not the nearest-rank p50
reported in the earlier table. The original <=10 ms median / <=20 ms p95 targets
remain **not met**. Acceptance is a documented trade-off, **not a p95 guarantee**;
correctness, privacy, coverage, native acceptance and independent review were not waived.
Private evidence: `task-19.4/results-20261015T152600Z.json`.

## Read-only OpenCode component measurements

These are measurements of an uncommitted component snapshot, not a final revision or full-collector performance gate. The measured files were `opencode_file.py` (`49b4ad63484e…`), `opencode_vfs.py` (`e7492ea0245b…`) and `opencode_snapshot.py` (`2c27e5db2482…`); exact hashes and every sample remain in the private component manifest. SQLite's inherited Unix VFS rejected descriptor symlinks on Linux; those failed attempts are retained. The corrected backend uses APSW's public descriptor-backed file protocol on both platforms, without reopening source paths.

Method: APSW 3.54.0.0 / SQLite 3.54.0; 30 fresh-engine processes per WAL stratum. Linux used Python 3.12 in an amd64 container on the Darwin arm64 host. Snapshot setup, query and close were timed; imports were excluded. Integrity hashing warmed an uncontrolled OS cache, and ambient/concurrent validation load was not isolated. Quantiles use nearest rank; these sample sizes do not establish tail guarantees.

| Platform | WAL target (MiB) | n | Startup CPU p95 (ms) | Startup wall p95 (ms) | Max RSS (MiB) | Max logical VFS bytes served (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Darwin arm64 | 0 | 30 | 0.586 | 0.583 | 27.33 | 0.008 |
| Darwin arm64 | 4 | 30 | 2.392 | 2.485 | 27.44 | 4.102 |
| Darwin arm64 | 32 | 30 | 13.184 | 13.600 | 27.69 | 32.281 |
| Darwin arm64 | 256 | 30 | 90.747 | 94.723 | 30.05 | 257.788 |
| Linux amd64 container | 0 | 30 | 19.821 | 20.394 | 41.97 | 0.008 |
| Linux amd64 container | 4 | 30 | 23.583 | 25.009 | 41.96 | 4.102 |
| Linux amd64 container | 32 | 30 | 49.486 | 50.974 | 42.08 | 32.281 |
| Linux amd64 container | 256 | 30 | 232.130 | 242.051 | 42.09 | 257.788 |

Main/WAL/SHM inventories and bytes matched before/after each owned fixture stratum. Logical VFS bytes are not physical disk I/O. Native WAL-index startup can exceed the decoded-input byte budget; it remains cooperative, not a hard byte/time bound. Neither these numbers nor the security fixtures establish the overall capture/startup or idle-collector SLO.

A bounded original-store pass then ran under kernel file-write/network denial, persisting only numeric aggregates and fixed coverage codes:

| Measurement | Observed |
| --- | ---: |
| Native main-file size | 3,143,557,120 bytes |
| Decoded row data | 447,156 bytes |
| Logical VFS bytes served | 146,362,692 bytes |
| Logical VFS read calls | 35,735 |
| Session / message / part row visits | 128 / 128 / 128 |
| Tool-call / tool-result evidence | 46 / 46 |
| CPU / wall | 343.959 / 2470.738 ms |
| Coverage | Checked snapshot; partial / lag |

The supplied row budget was per scan lane, not a 64-row global limit. Pending totals remained unknown and no complete-corpus claim was made. The active original store was not a frozen benchmark corpus; concurrent writer activity is not attributed to this reader. This proves bounded nonzero source visitation in that pass, not useful reuse, calibrated savings or a globally unchanged index. Private numeric artifacts: `wal-linux-resource-summary.json`, `wal-darwin-resource-summary.json`, `wal-descriptor-component-freeze.json` and `original-opencode-readonly-probe/observations.json`.

## Airline validation lab (historical)

The lab code is not part of the published tree; it is kept on branch `feat/accounting-airline-lab`.
The method is summarized in [methodology](accounting-methodology.md#airline-validation-lab-historical).
The results below keep their original method and revision labels and are not restated under the
current methodology.

### Control lanes (historical)

The earlier deterministic lane recorded 23/23 database/domain/concurrency/rollback/cleanup
gates; the integrated stand-in recorded domain/accounting assertions at C0–C10, with its live
gate `NOT_RUN`. Method: isolated PostgreSQL and scripted-agent controls, not provider-backed
conversations. Environment requirements are documented in the methodology; the exact source
revision for these control artifacts was not supplied here. These historical observations
do not establish a current-remediation pass.

### Observed native airline experiments (historical)

**Historical, pre-remediation method.** The following retained trials used native omp
`openai-codex` conversations against isolated PostgreSQL airline fixtures. Exact source
revision/environment manifests remain in the private trial artifacts; no current revision
equivalence is claimed. Their failures and zero-credit results remain visible deliberately.

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
witness → visible exposure → successful SQL action was traceable for all five. The historical
trial reported **all 18 gates and every C0–C10 domain/accounting checkpoint passed**, including
native sessions, restart, fresh-B isolation, correctness, reuse, replay, scrape/export and cleanup.
That report is scoped to its old method and retained revision, not current remediation.

That historical canonical runner used separate retrieval calls and ordinary readable `show` output,
with 120-column/no-color rendering in both environments. All answer text remained visible;
maximum observed line length was 360 bytes. Temporary-directory isolation also seals prior
and future neighboring reports, with own-scratch and SQL positive controls.

Under that historical production method, all scenarios reported **0 credit, 9,555 known
visible debit, −9,555 net context tokens**. These values are not a current-method recomputation.
The separately reviewed arithmetic fixture checked allocation math, not production eligibility:

| Fixture scenario | Credit | Debit | Net context tokens |
|---|---:|---:|---:|
| Conservative | 521 | 9,555 | −9,034 |
| Central | 1,327 | 9,555 | −8,228 |
| Optimistic | 2,169 | 9,555 | −7,386 |

Fixture-policy values are never exported or mixed into production totals. The successful trial
proves **explicitly prompted historical-key validation**, not spontaneous retrieval utility,
positive ROI, representative precision or causal savings. Product evidence remains conservatively
`action_reference`; the actual database audit supplies the independent domain validation.

## Representative-history audits

The earlier audit's implementation proof (`b838307c87a3…`, not a Git revision) was
exercised against authorized original native prefixes and
actual, read-only Lance table versions. No transcripts were mirrored, notes read,
index built/repaired, provider called or native agent launched. The original index
was not declared globally unchanged: local snapshot handles pinned real table versions
while independent writers could continue updating the live index.

| Observation | Result |
| --- | ---: |
| Selected original JSONL sources | 36 |
| Collector-reported bytes across 128 cycles | 355,634,138 |
| Readable unique lookup/invocation identities | 178 |
| Linked, readable receipts | 113 |
| Frozen independently labeled native groups | 181 |
| Labels joined to product identities | 163 |
| Execution-certainty agreement / disagreement / unknown | 113 / 0 / 68 |
| Transport agreement / disagreement / unknown | 79 / 0 / 102 |
| Reuse agreement / disagreement / unknown | 1 / 1 / 179 |
| Actual index documents / lexical terms | 253,526 / 112,169 |
| Existing lexical-stat metadata bytes | 2,068,504 |
| Supported context debit subtotal, each band | 107,200 tokens |
| Supported context credit subtotal, each band | 0 tokens |
| Complete token or work-time net | Unavailable |

**Result: partial, not estimator acceptance or a savings claim.** The supported zero-credit
subtotal is not known total credit. One root retained `replay_unavailable`; the collector
did not claim complete coverage. Historical import did not increment live usage rates.
The cumulative byte count excludes proof hashing and is not physical disk I/O.

Rarity was explicitly unavailable: both document and term counts exceeded that
freeze's 100,000-element limits, although the serialized statistics fit its separate
byte ceiling. The replacement builder uses bounded resumable scope validation; this
older audit does not establish that the replacement rates the representative corpus.
No single gate is claimed to explain every rejected or unknown lookup.

The independent labels are model-assisted, not human ground truth. They contain no
useful-positive stratum; the reuse disagreement and missing runtime/positive strata remain
open. Unknown eligibility was not converted into a zero-valued complete net or calibration
precision. This audit therefore does not clear representative calibration.

Validation-specific limits: committed prefixes retained their byte/hash authority; live
tail growth remained lag. A Darwin address-space limit was unsupported, so the private
worker reported an 8-GiB sampled peak-RSS watchdog with possible overshoot, not a hard
kernel memory bound. The source/runtime and index guards remain distinct.
Private aggregate evidence: `history-acceptance-20261015T110130Z-25222/report.json`.

A later attempt at source-file SHA-256 `13eb7fe64f90…` retained actual Lance versions
but captured the index in `building` state. The worker correctly reported
`frequency_index_not_ready`, not measured rarity. Its collection remained partial;
the outer audit was **blocked** when the post-run live-index inventory exceeded its
fixed ceiling. The retained worker summary is diagnostic evidence, not a passing
acceptance run. No new savings or calibration claim follows from it.
The private guard now refuses a non-ready initial index before collection; it does
not repair, rebuild or wait indefinitely on the real index.
Private evidence: `history-acceptance-20261015T130404Z-35409/report.json` and its
`worker-summary.json`.

### Large-corpus capacity control

A separate owned **real Lance** database used synthetic lexical statistics and chunk
scope rows at the observed corpus scale. Public frequency reads reopened the repository
on each call. After the first pending result, a newer live generation gained a foreign
chunk; the accepted immutable generation still finished against its original authorized
rows. No embeddings, provider calls, original histories or exports were involved.

| Observation | Result |
| --- | ---: |
| Original chunks / lexical terms | 253,526 / 112,169 |
| Public API calls, including reopened repositories | 89 |
| Pending calls before completion | 88 |
| Live chunks after the foreign append | 253,527 |
| Rarity lower bound / independent formula | 16,951 / 16,951 milli-bits |
| Serialized lexical metadata | 1,794,756 bytes |
| Builder CPU / wall time | 5.276 / 5.389 seconds |
| Whole-process lifetime peak RSS | 395.484 MiB |

Raw candidate vocabulary and approved cwd bytes were absent from retained accounting
files. This is one capacity/immutability/privacy smoke, **not original-corpus calibration
or a collector SLO**. CPU/wall timing excludes imports and database seeding; lifetime RSS
includes them, and ambient load was uncontrolled.
Component source SHA-256 prefixes: `index_frequency.py` `ace26322938c`,
`index_frequency_build.py` `d47897390cc7`, and
`index_frequency_repository.py` `d688a6d294da`.
Private evidence: `large-lance-frequency-api-smoke.json`.

## Calibration limits

Method/environment for the historical review below: local agent-reviewed receipt cards, not
human annotation or a representative sampled study; source revision was not supplied here.

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

A historical isolated empty-root CLI smoke captured all 12 then-new command/group help pages
with exit 0 and exercised disabled summary, setup, collection, fresh summary/status, empty
history, pause/refused collection, resume and purge preview. Its revision was not supplied;
this is first-use wiring evidence only, not the separate twelve policy CLI/MCP cases above,
nonempty receipt valuation, live-agent behavior, remote export or current-head approval.

## Software verification

Method: historical automated suite and source-statement instrumentation plus separate
real-model/control checks. Exact host, OS and dependency versions were not supplied in this
summary; consult that revision's retained gate artifacts before reproducing the numbers.

A recorded review-fix gate at `f6d97e9` ran **4,902 passing tests** and reached **100%
source-statement coverage** (24,437 statements). Its normally excluded network-marked embedding
test also passed in a separate real-model run. Lint, formatting, typing, source-size and
dependency checks passed. The shipped-source live isolation preflight and the standalone
PostgreSQL/control journeys were rerun without private planning files or additional model calls.
That suite reported the existing LanceDB `create_index` deprecation warnings; these were
not suppressed. Coverage is an execution measure, not proof of estimator calibration or security.
This recorded result predates the current remediation. It is not evidence that the partial
uncommitted revision or its eventual frozen successor passes any of those gates.

## Real collector run

Historical integration-test method/environment: local Docker collector
`otel/opentelemetry-collector-contrib:0.137.0` with loopback OTLP/HTTP JSON and synthetic
payloads. The artifact's exact source revision was not supplied here; results below are
retained observations, not current-remediation verification.

**What a real collector run showed** (`tests/accounting/export/test_otel_collector.py` and
`tests/accounting/export/test_otel_delivery.py` against the image
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
  arrivals; only revision-aware logic (`ssgrep.accounting.export.receiver.apply`) produced the current totals.
  The collector run does not demonstrate partial success, `429`/`Retry-After`, gzip, or a
  non-loopback network.

## What was not verified

- **No real backend dashboard was run.** No Grafana, Prometheus, or SQL dashboard was built or
  rendered from this document; panel layout, variables, and the OTLP-to-backend name translation
  are unexercised.
- **PromQL engine validation is partial.** Five named methodology-filtered debit,
  token/work net, credit and reporting-device queries now execute in real `promtool`
  against independent multi-device/install/tenant literals, duplicate info and
  omitted unknown samples. The old `instance, job` joins failed before correction.
  This validates matching/arithmetic, not credential enrollment, every ratio,
  vendor-specific resource-label promotion, Grafana variables or a deployed dashboard.
- **The SQL is validated only as far as it ran on SQLite.** The test executes every SQL block on
  synthetic rows with stdlib `sqlite3` and asserts the totals, current-revision selection under
  shuffled arrival, tombstone exclusion, quarantine, `NULL` for unknown, and the freshness
  numbers. It was not run on PostgreSQL or any other engine, and not under concurrent ingest
  (the reference `apply` is also single-threaded).
- Collector acceptance is the strongest state observable; backend durability, retention, and
  erasure after a purge are the operator's to verify.
- Real network conditions, TLS to a remote host, gzip, rate limiting, and `Retry-After` against a
  real collector were not exercised.
- Real user data was not used in the synthetic collector/dashboard examples. Native trials
  and the receipt review above have their separately stated private-data boundaries.
- The provider-backed matched with/without-history experiment is **NOT RUN**. Implemented
  commands and provider-free controls do not establish empirical savings.
