# Test Fixtures Manifest

Sanitized fixtures for wave-2 testing, derived from the real corpus at
`~/.claude/projects/` (structures sampled 2026-07-25). Every fixture preserves
the structural keys and record-type mix of the real format; all absolute
paths, secrets, tokens, and personal prose are replaced with deterministic
placeholders.

## Redaction rules applied

| Real value | Replacement |
|---|---|
| Absolute paths (`/Users/<user>/code/...`) | `/Users/redacted/project/...` |
| API keys, tokens, request ids | `REDACTED_SECRET` / `req_REDACTED` / `cse_REDACTED` |
| Thinking-block `signature` payloads | deterministic synthetic base64 (no real replay data) |
| Image-block base64 data | deterministic synthetic base64 |
| Personal prose | generic placeholder text about a fictional widget-cache/allocator codebase |
| Repo/org names, PR urls | `example-org/example-repo`, `https://github.com/example-org/example-repo/pull/1` |
| Session/agent identifiers | fixed placeholder UUIDs and `aexample-*` agent ids |

No real secrets, paths, or personal content appear in any fixture.

## Fixtures

### `main-session.jsonl`
Exercises **all 15 observed record types** in one coherent session:
`user`, `assistant`, `queue-operation`, `permission-mode`, `mode`,
`last-prompt`, `attachment`, `agent-name`, `custom-title`, `ai-title`,
`system`, `file-history-snapshot`, `bridge-session`, `agent-setting`,
`pr-link` — plus all five required `system` subtypes: `away_summary`,
`compact_boundary`, `local_command`, `turn_duration`, `api_error`.

Also covers: title-precedence records (`custom-title` present → wins over
`ai-title`/`last-prompt`); `thinking` + `text` + `tool_use` assistant blocks;
`tool_result` user blocks with sibling `toolUseResult` payloads; `Read`,
`Edit`, `Write` (with `file_path`) and `Bash` (with `command`, no path) tool
calls for `files_touched` derivation; a `local_command` bare `/model`
invocation plus its `<local-command-stdout>` reply; a user message mixing
`<command-name>` wrapper markup with genuine prose (markup stripped, prose
kept); one mid-session compaction boundary followed by a compacted-head
assistant record.

Note on `agent-setting`: this record type is one of the 15 observed types in
the spec, but no instance exists in the *current* local corpus (the corpus
mutates and prunes itself). Its shape here follows the flat sidecar
convention shared by `agent-name`/`mode`/`permission-mode`:
`{type, <camelCase payload key>, sessionId}`.

### `compaction-session.jsonl`
One real session start (a `user` record with `parentUuid: null` and no
subtype) followed by **3 `compact_boundary` records**, each carrying
`parentUuid: null`, `logicalParentUuid`, and a full `compactMetadata` object
(`trigger`, `preTokens`, `durationMs`, `preservedSegment` with
`headUuid`/`anchorUuid`/`tailUuid`, `preservedMessages`). Exercises the rule
that null-parent records are *episode* boundaries, not new sessions: a naive
parser reads 4 sessions, a correct one reads 1 session with hard episode
boundaries.

### `thinking-signatures.jsonl`
Assistant records whose `thinking` content blocks all carry an **empty
`thinking` string** (`""`) together with non-empty base64 `signature`
payloads (800–4200 raw bytes, matching the observed ~4 KB average). Tests
must assert these blocks are skipped unconditionally and their signature
bytes never counted as prose.

### `truncated-file.jsonl`
Three valid records followed by a final line cut **mid-record** (incomplete
JSON, no closing brace, no trailing newline). Exercises EOF-truncation
handling: the partial line counts as malformed/skipped and parsing of prior
records is unaffected.

### `rewritten-file.jsonl` + `rewritten-file.original-first-line.txt`
A valid small transcript that was **rewritten in place**: its first line
(a `custom-title` for session `44444444-…`) differs from the pre-rewrite
first line stored in the `.original-first-line.txt` sidecar (a
`custom-title` for session `33333333-…`). Tests pair this with a stored
cursor (`first_line_hash` of the original line) to prove an in-place rewrite
is detected and triggers a full re-parse.

### `zero-prose-session.jsonl`
`user` and `assistant` records containing **only `tool_use`, `tool_result`,
`image`, and `thinking` blocks — zero `text` blocks**. Per the spec's
degenerate-input requirement this must yield zero episodes without error.

### `malformed-line.jsonl`
Five lines where **line 3 is malformed JSON** (`{…[INVALID JSON ,,,`) between
valid records. Parsing must increment the malformed counter, skip the line,
and continue.

### `oversized-block.jsonl`
Three lines; line 2 is a single JSON line of **534 KB (>500 KB)** — a `user`
record carrying a base64 `image` block, mirroring the real 552 KB image
observed in the corpus. Exercises the per-line size cap: the oversized line
is skipped unparsed, parsing continues with line 3.

### `agent-aexample-agent-deadbeef01.jsonl` + `agent-aexample-agent-deadbeef01.meta.json`
A subagent transcript: records carry `isSidechain: true`, `agentId`, `slug`,
and a `parentUuid: null` first `user` record whose content is a
`<teammate-message>` wrapper — note its `sessionId` equals the *parent's*,
so identity must come from parent id + agent file identity. The sibling
`.meta.json` supplies `agentType`, `description`, `name`, `spawnDepth`,
`model`, `taskKind`, `teamName`, `color`, `planModeRequired`,
`permissionMode` for attribution. Both files share the `agent-aexample-agent-deadbeef01`
basename, following the spec's sibling-association convention.

### `ismeta-record.jsonl`
Two `type: "user"` records with **`isMeta: true`**, matching the *only* shape
`isMeta: true` is ever observed in the real corpus (verified against the live
`~/.claude/projects` corpus 2026-07-28: 64/64 `isMeta: true` records are
`type: "user"`; it never appears on `assistant` records). Every other fixture
in this manifest has `isMeta: false` on every record, so this is the only
fixture that exercises the `isMeta` guards in `signal.classify_signal`,
`metadata.harvest_metadata`, and `metadata.derive_files_touched`. Line 1 mirrors the real
`<local-command-caveat>` shape (string `message.content`); line 2 mirrors the
real skill-reinjection shape (list `message.content` with one `text` block).
Both carry deliberately out-of-place `gitBranch`/`cwd`/text values prefixed
`ISMETA_POISON_*` / `should-not-surface-*` so a test can assert their absence
from any derived output, not just that some output happens to be empty.

### `old-schema-session.jsonl`
Older-schema subagent records carrying **`teamName` and `agentName`** fields
(and **no** `agentId`, **no** `slug`). Field-level drift must not break
extraction.

### `new-schema-session.jsonl`
Newer-schema records carrying **`agentId`** and **`slug`** (and no
`teamName`/`agentName`). Pair with `old-schema-session.jsonl` to test both
directions of the `teamName`/`agentName` → `agentId` drift.

## Structural invariants preserved

- Envelope keys on content records: `parentUuid`, `isSidechain`, `uuid`,
  `timestamp`, `userType`, `entrypoint`, `cwd`, `sessionId`, `version`,
  `gitBranch` (plus `slug` on newer-schema records).
- Flat sidecar records (`custom-title`, `agent-name`, `mode`,
  `permission-mode`, `last-prompt`, `ai-title`, `agent-setting`,
  `bridge-session`, `pr-link`, `queue-operation`, `file-history-snapshot`)
  carry only `type`, payload keys, and `sessionId`/`timestamp`.
- `assistant.message` keeps `model`, `id`, `type`, `role`, `content`,
  `stop_reason`, `stop_sequence`, `stop_details`, `usage`, `diagnostics`.
- `tool_use` blocks keep `id`, `name`, `input`, `caller`; `tool_result`
  blocks keep `tool_use_id`, `type`, `content`; image blocks keep
  `source.type`/`media_type`/`data`.
- `user` records may carry sibling `toolUseResult`, `sourceToolAssistantUUID`,
  and `session_id` keys alongside `message`.
