# Adapter on-disk format spike: what the five runtime adapters actually expect

Status: derived from parser source (2026-08-22), not from upstream app documentation.
Every claim below cites the parser line that enforces it. Where the README or the plan
disagrees with the parser, the parser wins and the discrepancy is flagged.

Source of truth files:

- Registry: `src/ssgrep/sessions/adapters/registry.py`
- Shared contracts: `src/ssgrep/sessions/adapters/base.py`
- Claude/native: `src/ssgrep/sessions/adapters/native.py`, `src/ssgrep/sessions/discovery.py`, `src/ssgrep/sessions/discovery_roots.py`
- OpenCode: `src/ssgrep/sessions/adapters/opencode.py`
- Codex: `src/ssgrep/sessions/adapters/codex.py`
- Pi and Prime Agent: `src/ssgrep/sessions/adapters/pi.py`
- Shared record rules: `src/ssgrep/sessions/records.py`, `src/ssgrep/sessions/signal.py`, `src/ssgrep/sessions/metadata.py`
- Episode segmentation: `src/ssgrep/sessions/episodes.py`, `src/ssgrep/pipeline/episodes.py`
- Path resolution: `src/ssgrep/utilities/paths.py`

## 1. Purpose and scope

Task 4 of the retrieval-eval overhaul. The five per-runtime emitters (T7-T11) must
write files that the real adapters ingest unchanged, because the benchmark corpus is
consumed by pointing the documented env overrides at emitted directories and running
the real indexer. A format guess on our side quietly skews ingestion, so this spec
converts parser behavior into an emitted contract before any emitter exists.

The five adapters in discovery order are `native`, `opencode`, `codex`, `pi`,
`prime-agent` (`src/ssgrep/sessions/adapters/registry.py:14-21`). Every adapter
implements `discover(...) -> list[TranscriptSource]` and
`read(source) -> ReadResult` (`base.py:44-53`), and `ReadResult.records` is always a
tuple of Claude-shaped normalized dicts (`base.py:35-41`). Each runtime file on disk
is a source format; the shared episode segmentation below sees only the normalized
form. Segmentation rules are therefore identical for all five runtimes and are
documented once, in section 2.

## 2. Shared ingestion pipeline (all runtimes)

### 2.1 Record shape after normalization

Fields the pipeline consults on a normalized record:

- `type`: one of `user`, `assistant`, `system`, `custom-title`, plus the other known
  types in `src/ssgrep/sessions/records.py:9-25`. Unknown types are skipped and
  counted as skipped records (`base.py:113-114`).
- `message: {role, content, model}` where `content` is a string or a list of blocks.
- `cwd`, `timestamp`, `gitBranch`, `sessionId`, `uuid`, `version`, `entrypoint`,
  `permissionMode`, `userType`, `isMeta`, `subtype`, `_files_touched` (harvested by
  `metadata.harvest_metadata`, `metadata.py:136-201`).

The line size limit for every JSONL parser is 500,000 bytes
(`records.py:7`; enforced in `base.py:102`, `codex.py:60`, `pi.py:70`). Lines
without a trailing newline are treated as a concurrent append and skipped
(`base.py:96-98`, `codex.py:53-58`, `pi.py:63-68`). Emitters must write files whose
final line ends with a newline.

### 2.2 Episode splitting: what forces a split

`segment_episode_groups` (`src/ssgrep/sessions/episodes.py:12-98`) is the single
boundary-detection pass:

- A `user` record with non-empty extracted text flushes the open episode and starts
  a new one (`episodes.py:81-87`).
- A `system` record with `subtype == "compact_boundary"` flushes without joining a
  group (`episodes.py:76-79`).
- An `assistant` record contributes its extracted text to the current response list
  (`episodes.py:90-93`).
- Everything else (tool echoes, thinking blocks, unknown records) lands in whichever
  episode is open; it never opens one.

Critical rule: a `user` record whose extracted text is empty does NOT split. It folds
into whatever episode follows (`episodes.py:81-87` and docstring `episodes.py:29-45`).
Episodes form around text-bearing prompts, not around raw `user` records.

Episode id format: `<session_id>:ep:<index>` (`episodes.py:8-9`), 0-based within the
session. This is the id scheme T12's query generator references as `<session>:ep:<n>`.

Signal text that lands in an episode (`extract_episode_text`,
`pipeline/episodes.py:72-101`, delegating to `signal.classify_signal` at
`src/ssgrep/sessions/signal.py:18-53`):

- prompt: `user` records with `text` blocks (`signal.py:26-29`),
- response: `assistant` records with `text` blocks (`signal.py:35-40`) plus
  `system`/`away_summary` records (`signal.py:44-49`),
- excluded blocks: `tool_result`, `image`, `thinking`, `tool_use`
  (`signal.py:30-33`, `signal.py:36-37`, `signal.py:41-42`).

Titles: `custom-title` > `ai-title` > `last-prompt` > first user text (truncated to
100 chars) > `Episode N` (`metadata.py:203-209`). `files_touched` comes only from
`tool_use` blocks whose name (lowercased) is in
`{read, edit, write, patch, apply_patch, multiedit}` (`metadata.py:103-133`).
`tool_names` is every `tool_use` name on assistant records (`metadata.py:190-199`).

Emitter consequence: an assistant message that should surface file metadata must carry
`tool_use` blocks (or the raw `_files_touched` field, `metadata.py:111-115`);
text-only assistant blocks contribute no file paths.

## 3. Claude / native adapter (`native`)

### 3.1 File layout and naming

Claude Code corpus: `$CLAUDE_CONFIG_DIR/projects` (default `~/.claude/projects`),
via `paths.resolve_claude_dir() / "projects"` (`discovery.py:248`; `CLAUDE_CONFIG_DIR`
honored at `utilities/paths.py:58-62`). Layout inside the tree:

- Main session: `<projects>/<projectDir>/<sessionId>.jsonl` (exactly 2 path parts
  under `projects/`), `discovery.py:192-213`.
- Subagent: `<projects>/<projectDir>/<sessionId>/subagents/**/*.jsonl` (at least 4
  parts, path must contain `subagents`), `discovery.py:215-233`.
- Excluded subtrees: `<projectDir>/memory/`, `<projectDir>/<sessionId>/tool-results/`,
  and `<projectDir>/<sessionId>/workflows/` (but NOT `subagents/workflows/`),
  `discovery.py:43-84`.

session_id for a main session is the file stem (`discovery.py:261`); for a subagent
it is `<parentSessionId>:agent:<agent-hash>` (`discovery.py:262-263`).

External roots (`SSGREP_TRANSCRIPT_DIRS`): every `*.jsonl` under the root recursively
(`discovery_roots.py:105`). Files whose first 5 non-empty lines contain no dict with a
`type` key are skipped with a warning (`discovery_roots.py:58-92, 106-111`), so a
minimal file must have a recognized record early. session_id of an external file is
`<stem>~<8-hex-sha1-of-absolute-path>` (`discovery_roots.py:113-117`).

### 3.2 Record schema (per JSONL line)

One JSON object per line. Known `type` values (`records.py:9-25`): `user`, `assistant`,
`system`, `queue-operation`, `permission-mode`, `mode`, `last-prompt`, `attachment`,
`agent-name`, `custom-title`, `ai-title`, `file-history-snapshot`, `bridge-session`,
`agent-setting`, `pr-link`. Only these survive `read_jsonl`; a record of any other
type is skipped and counted, while the rest of the file still indexes (`base.py:113-114`).

User/assistant records:

```json
{"type": "user",
 "message": {"role": "user", "content": [{"type": "text", "text": "..."}]},
 "cwd": "/abs/path/project", "timestamp": "2026-01-02T03:04:05+00:00",
 "sessionId": "<id>", "uuid": "<uuid>"}
```

Optional per-record metadata: `version`, `entrypoint`, `permissionMode`, `userType`
(consumed via `pipeline/episodes.py:19-43`), `gitBranch` (`metadata.py:185-186`),
`isMeta` (skipped by `signal.py:19-21`), `_files_touched` (`metadata.py:111-115`).
A `custom-title` record:

```json
{"type": "custom-title", "custom-title": "How we fixed the retry loop",
 "sessionId": "<id>", "cwd": "/abs/path/project", "timestamp": "2026-01-02T00:00:00+00:00"}
```

The `native` adapter stamps three different runtimes depending on source
(`native.py:19-28`): `claude` for the projects tree, `ssgrep` for authored notes,
`native` for `SSGREP_TRANSCRIPT_DIRS` roots. See section 9 for the consequence on T7.

### 3.3 Minimal valid file example

```jsonl
{"type": "custom-title", "custom-title": "Diagnose the retry backoff hang", "sessionId": "sess-0001", "cwd": "/fictional/checkout/payments", "timestamp": "2026-01-02T09:00:00+00:00"}
{"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": "The retry loop for card payments hangs after the third attempt. Where is the backoff policy?"}]}, "cwd": "/fictional/checkout/payments", "timestamp": "2026-01-02T09:00:05+00:00", "sessionId": "sess-0001", "uuid": "m-0001", "gitBranch": "fix/retry-backoff"}
{"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "The retry is bounded by `bounded_exponential_backoff` in `retry.py`, capped at 60 seconds after attempt 3."}, {"type": "tool_use", "name": "Read", "input": {"file_path": "/fictional/checkout/payments/retry.py"}}]}, "cwd": "/fictional/checkout/payments", "timestamp": "2026-01-02T09:00:12+00:00", "sessionId": "sess-0001", "uuid": "u-0001"}
```

One episode with id `sess-0001:ep:0`. Add another text-bearing `user` record to get a
second episode. The `tool_use` block makes `files_touched == (".../retry.py",)` and
`tool_names == ("Read",)`.

## 4. OpenCode adapter (`opencode`)

### 4.1 Database path and snapshot semantics

DB path: `SSGREP_OPENCODE_DB` if set, else `$XDG_DATA_HOME/opencode/opencode.db`,
else `~/.local/share/opencode/opencode.db` (`src/ssgrep/sessions/adapters/opencode.py:51-57`).
The adapter opens via `sqlite3.connect(uri, uri=True, timeout=1.0)` on a `?mode=ro`
URI, then `PRAGMA query_only = ON` and `BEGIN`, holding one read-only snapshot until
it closes (`opencode.py:60-73`). `opencode.db` must be a valid SQLite file: a missing
file yields zero sources (`opencode.py:298-300`), and missing required columns also
yield zero sources (`opencode.py:303-305`, `opencode.py:81-90`).

### 4.2 Required schema and discovery query

Three tables with minimum columns (`opencode.py:81-90`):

| Table | Required columns | Optional columns consulted |
|---|---|---|
| `session` | `id` | `parent_id`, `project_id`, `directory`, `title`, `version`, `time_created`, `model`, `agent`, `time_updated` |
| `message` | `id`, `session_id`, `data` | `time_created`, `time_updated` |
| `part` | `id`, `message_id`, `session_id`, `data` | `time_created`, `time_updated` |

The discovery SELECT (`opencode.py:115-152`) always asks for the optional session
columns but substitutes `NULL` when a column is absent (`_selected`,
`opencode.py:93-94`), so an emitter may omit them. Timestamps prefer `time_updated`
then `time_created` (`_time_value`, `opencode.py:97-103`); providing `time_created`
alone is enough. Message ordering uses `time_created` then `id`
(`opencode.py:357-369`); part ordering the same (`opencode.py:372-384`). Set
monotonically increasing `time_created` values within a session to fix ordering.

The fingerprint uses `message_count`, `part_count`, `session_updated`,
`message_updated`, `part_updated` (`opencode.py:213-228`), so a deterministic emitter
(stable counts and timestamps) gives stable incremental re-ingestion.

### 4.3 Record mapping and exclusions

The `data` columns hold JSON strings. Per message (`_normalize`, `opencode.py:570-637`):

- `message.data` dict must carry `role` equal to the strings `user` or `assistant`,
  else the message is skipped (`opencode.py:591-594`),
- model is read from `model`/`modelID`/`model_id` keys (`opencode.py:195-200`),
- timestamp from `time.created` (milliseconds since epoch; ISO 8601 strings also
  accepted, `opencode.py:387-409`),
- cwd from `path.cwd` (`opencode.py:412-418`),
- `parentID`/`parent_id` sets `parentUuid` (`opencode.py:560-562`).

Per part (`_part_blocks`, `opencode.py:475-500`):

- `{"type": "text", "text": "..."}` becomes a `text` block; a text part with
  `ignored: true` is dropped (`opencode.py:477-484`),
- `{"type": "tool", "tool": "<name>", "state": {"input": {...}}}` becomes a
  `tool_use`; the name goes through `_TOOL_NAMES` (`opencode.py:34-47`), and one
  block is emitted per `file_path`/`filePath`/`path`/`paths`/`filename` input key
  (`opencode.py:432-443`),
- `{"type": "file", "source": {"path": ...}}` becomes `tool_use`/`Read`
  (`opencode.py:486-489`),
- `{"type": "patch", "files": ["a", "b"]}` becomes one `tool_use`/`Edit` per file
  (`opencode.py:490-499`),
- any part `type` in `{agent, compaction, reasoning, retry, snapshot, step-finish,
  step-start, subtask}` is ignored and never indexed (`opencode.py:24-33`),
- any other part type is counted as skipped (`opencode.py:500-502`).

If a message has no surviving parts, the adapter falls back to `message.data.content`
(a string or a list of `{"type": "text", "text": ...}` blocks) (`opencode.py:505-518, 624`).

Session `title` becomes a `custom-title` record (`opencode.py:616-621`); the session
`directory` is the fallback cwd (`opencode.py:614, 412-418`).

Normalized ids: session `opencode:<raw>`, parent `opencode:<parent-raw>`; `is_main`
is true exactly when `parent_id` is NULL at discovery (`opencode.py:263-273`).

### 4.4 Minimal valid example (SQLite)

```sql
CREATE TABLE session (id TEXT PRIMARY KEY, directory TEXT, title TEXT, time_created INTEGER, model TEXT);
CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT, data TEXT, time_created INTEGER);
CREATE TABLE part (id TEXT PRIMARY KEY, message_id TEXT, session_id TEXT, data TEXT, time_created INTEGER);

INSERT INTO session VALUES ('sess-00001', '/fictional/checkout/payments', 'Diagnose the retry backoff hang', 1760000000000, 'gpt-oss-120b');
INSERT INTO message VALUES ('m-001', 'sess-00001', '{"role":"user","time":{"created":1760000001000},"path":{"cwd":"/fictional/checkout/payments"}}', 1760000001000);
INSERT INTO part VALUES ('p-001', 'm-001', 'sess-00001', '{"type":"text","text":"The retry loop hangs after attempt 3. Where is the backoff policy?"}', 1760000001001);
INSERT INTO message VALUES ('m-002', 'sess-00001', '{"role":"assistant","model":"gpt-oss-120b","time":{"created":1760000002000},"path":{"cwd":"/fictional/checkout/payments"}}', 1760000002000);
INSERT INTO part VALUES ('p-002', 'm-002', 'sess-00001', '{"type":"text","text":"Backoff lives in retry.py, capped at 60s."}', 1760000002001);
```

`message.data` can also carry `content` directly when no parts exist. Extra tables or
columns are safe: the `PRAGMA table_info` probe only inspects the three named tables
(`opencode.py:78-79`, `opencode.py:81-90`).

## 5. Codex adapter (`codex`)

### 5.1 Layout and env overrides

Root resolution (`codex.py:264-286`): `SSGREP_CODEX_SESSIONS_DIR` first, then
`CODEX_SESSIONS_DIR`, then `${CODEX_HOME}/sessions` with `~/.codex` as the default
(`paths.resolve_codex_dir()`, `utilities/paths.py:103-110`). Discovery takes every
`*.jsonl` recursively (`codex.py:296-299`), one file per session. Real codex organizes
by dated subdirectories but the parser imposes no subdir layout: any `.jsonl` deep
under the root is a candidate. There is no subagent concept: `is_main` is always true
(`codex.py:314-315`; `no_subagents` is deliberately ignored, `codex.py:291`).

### 5.2 Envelope and record schema

Every line is `{"timestamp", "type", "payload"}` (`codex.py:5-19`). The gates live in
`_inspect` (`codex.py:86-111`) and `_normalize` (`codex.py:196-260`):

| `type` / `payload.type` | effect |
|---|---|
| `session_meta` | sets session `id` (required here, else no source, `codex.py:104-105`), starting `cwd`, `model_provider` |
| `turn_context` | per-turn `cwd`, `model` (`codex.py:101-103`) |
| `response_item` + `payload.type=message` + role `user` | flush pending, emit user record (`codex.py:216-224`) |
| `response_item` + `payload.type=message` + role `assistant` | flush pending, emit assistant as `pending` (`codex.py:225-237`) |
| `response_item` + `payload.type=message` + role `developer` | intentionally omitted (`codex.py:238-239`) |
| `response_item` + `payload.type=function_call`/`custom_tool_call` | tool block; appended to the pending assistant or opening a fresh one (`codex.py:240-256`) |
| `response_item` with any other `payload.type` (reasoning, `*_output`, token counts, item bookkeeping, agent_message, world-state) | omitted (`codex.py:257-258`) |
| any other record `type` | ignored (`codex.py:196-197`) |

Accepted message content block types are `text`, `input_text`, `output_text`; other
block types are dropped (`codex.py:131-147`). `function_call.arguments` may be a JSON
string that is parsed into a dict, with a raw-string fallback (`codex.py:150-166`).

Normalized ids: session is `codex:<raw-id>` (`codex.py:82-83, 107`), per-record uuid
is `codex:<seq>:<timestamp>` (`codex.py:212`).

### 5.3 Minimal valid file example

```jsonl
{"timestamp": "2026-01-01T09:00:00Z", "type": "session_meta", "payload": {"id": "codex-sess-001", "cwd": "/fictional/checkout/payments", "model_provider": "openai"}}
{"timestamp": "2026-01-01T09:00:01Z", "type": "turn_context", "payload": {"cwd": "/fictional/checkout/payments", "model": "gpt-5"}}
{"timestamp": "2026-01-01T09:00:02Z", "type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "The retry loop hangs after attempt three. Where is the backoff policy?"}]}}
{"timestamp": "2026-01-01T09:00:03Z", "type": "response_item", "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "The backoff lives in retry.py deep maybe, capped at 60 seconds."}]}}
{"timestamp": "2026-01-01T09:00:04Z", "type": "response_item", "payload": {"type": "function_call", "name": "shell", "arguments": "{\"command\": \"sed -n 1,40p retry.py\"}", "call_id": "call-0001"}}
```

One episode (`codex-sess-001:ep:0`). Any `reasoning` or `function_call_output` line
would be silently excluded, which is why emitters must not emit them.

## 6. Pi adapter (`pi`)

### 6.1 Root layout and env overrides

Shared base class `_PiRuntimeAdapter` (`src/ssgrep/sessions/adapters/pi.py:264-289`)
with Pi configuration at `pi.py:358-365`:

- `SSGREP_PI_SESSIONS_DIR`, then `PI_SESSION_DIR`, then `PI_CODING_AGENT_SESSION_DIR`,
  then `${PI_CODING_AGENT_DIR}/sessions`, then `~/.pi/agent/sessions`.

Discovery: every `*.jsonl` recursively (`pi.py:304-309`). The first record must be
`{"type": "session", "id": ...}` or the file is skipped (`pi.py:106-110`). session id
on normalized records is `pi:<raw>` (`pi.py:99-100, 128-134`). A child session
(`rlmDepth` is an int greater than 0) is a separate source with `is_main=False`
(`pi.py:114-115`).

### 6.2 Record schema and exclusions

Entry types (`pi.py:38-53`; handling in `_normalize`, `pi.py:200-260`):

| `type` | handling |
|---|---|
| `session` | sets `cwd`, branch (via `git.branch`), `modelId` (`pi.py:202-206`) |
| `model_change` | updates `model`/`provider` (`pi.py:207-210`) |
| `git` / `git_state` | branch update (`pi.py:211-214`) |
| `custom-title` / `session_info` / `session_title` / `title` | title record from `name`/`title`/`custom-title` (`pi.py:214-226`) |
| `message` | the only conversation-bearing type (`pi.py:229-259`) |
| `message` with role `toolResult` | dropped entirely (`pi.py:238-239`) |
| `message` with any other role | skipped and counted (`pi.py:240-242`) |

Ignored entry types (no index output): `agent_status`, `branch_summary`,
`child_usage_attributed`, `compaction`, `custom`, `custom_message`, `label`,
`service_tier_change`, `session_state`, `thinking_level_change` (`pi.py:38-51`).

Content blocks (`_content_blocks`, `pi.py:158-188`):

- `{"type": "text", "text": ...}` -> kept as a text block,
- `{"type": "toolCall", name, arguments, id}` on Assistant messages -> `tool_use`
  (`pi.py:163-184`),
- `thinking`, `image`, tool results, and extension-specific blocks -> dropped
  (`pi.py:185-187`).

### 6.3 Minimal valid file example

```jsonl
{"type": "session", "id": "pi-sess-001", "cwd": "/fictional/checkout/payments", "modelId": "claude-sonnet-4-5", "timestamp": "2026-01-02T10:00:00Z"}
{"type": "custom-title", "name": "Retry loop never sleeps", "timestamp": "2026-01-02T10:00:00Z"}
{"type": "message", "id": "m1", "timestamp": "2026-01-02T10:00:01Z", "message": {"role": "user", "content": [{"type": "text", "text": "Where does the loop hang after attempt 3?"}]}}
{"type": "message", "id": "m2", "timestamp": "2026-01-02T10:00:02Z", "message": {"role": "assistant", "content": [{"type": "text", "text": "In retry.py, the backoff caps at 60 seconds."}, {"type": "toolCall", "name": "Read", "arguments": {"file_path": "/fictional/checkout/payments/retry.py"}, "id": "t-001"}]}}
```

Session id on the index is `pi:pi-sess-001`; episode id `pi:pi-sess-001:ep:0`. A
`toolResult` message placed after `m2` would add no episode text.

## 7. Prime Agent adapter (`prime-agent`)

### 7.1 Root layout and session-artifacts children

Same shared base class as Pi (`pi.py:368-390`):

- `SSGREP_PRIME_AGENT_SESSIONS_DIR`, then `PRIME_AGENT_SESSION_DIR`, then
  `PRIME_AGENT_CODING_AGENT_SESSION_DIR`, then
  `${PRIME_AGENT_CODING_AGENT_DIR}/sessions`, then `~/.prime/agent/sessions`
  (`pi.py:374-379`, `pi.py:277-289`).

Two discovery roots (`_discovery_roots`, `pi.py:381-382`):

1. the sessions root itself,
2. its sibling `session-artifacts/`, computed as `root.parent / "session-artifacts"`
   (`pi.py:381-390`). With defaults that is `~/.prime/agent/sessions` plus
   `~/.prime/agent/session-artifacts`.

Every `*.jsonl` under `session-artifacts/` becomes a source with `is_main=False`
(subagent) no matter what the `rlmDepth` header says (`pi.py:384-390`). Content placed
there becomes a separate subagent session with runtime `prime-agent`.

Record schema is identical to Pi (same `_PiRuntimeAdapter` normalization;
`pi.py:264-266`). Normalized session id is `prime-agent:<raw>` (`pi.py:99-100, 331-338`).

### 7.2 Minimal valid files

`<root>/prime-sess-001.jsonl`: the Pi minimal example with id `prime-sess-001`.

`<root.parent>/session-artifacts/fusion-snap.jsonl`:

```jsonl
{"type": "session", "id": "artsnap-01", "cwd": "/fictional/checkout/payments", "rlmDepth": 1, "parentSessionId": "prime-sess-001", "timestamp": "2026-01-02T10:00:00Z"}
{"type": "message", "id": "a1", "timestamp": "2026-01-02T10:00:01Z", "message": {"role": "user", "content": [{"type": "text", "text": "Summarize the retry findings."}]}}
{"type": "message", "id": "a2", "timestamp": "2026-01-02T10:00:02Z", "message": {"role": "assistant", "content": [{"type": "text", "text": "Backoff caps at 60s in retry.py."}]}}
```

Because the path is under `session-artifacts/`, the episodes index with
`is_subagent=true` (`pipeline/episodes.py:125`).

## 8. Env override reference

| Override | Adapter | Behavior | Parser citation |
|---|---|---|---|
| `SSGREP_TRANSCRIPT_DIRS` | `native` external roots | colon-separated; bare path means `native`; `native=/dir` is the tagged form; unknown tags raise; missing dirs warn and skip | `discovery_roots.py:38, 129-158` |
| `CLAUDE_CONFIG_DIR` | `native` Claude corpus | `projects/` resolves beneath it | `paths.py:58-62`, `discovery.py:248` |
| `SSGREP_OPENCODE_DB` | `opencode` | path to the SQLite file | `opencode.py:52-54` |
| `XDG_DATA_HOME` | `opencode` | `opencode.db` under it, else `~/.local/share` | `opencode.py:55-57` |
| `SSGREP_CODEX_SESSIONS_DIR` | `codex` | first | `codex.py:270` |
| `CODEX_SESSIONS_DIR` | `codex` | second | `codex.py:271` |
| `CODEX_HOME` | `codex` | `${CODEX_HOME}/sessions` | `paths.py:108-110` |
| `SSGREP_PI_SESSIONS_DIR` | `pi` | first | `pi.py:364` |
| `PI_SESSION_DIR` / `PI_CODING_AGENT_SESSION_DIR` / `PI_CODING_AGENT_DIR` | `pi` | second / third / sessions-root | `pi.py:364-365` |
| `SSGREP_PRIME_AGENT_SESSIONS_DIR` | `prime-agent` | first | `pi.py:375` |
| `PRIME_AGENT_SESSION_DIR` / `PRIME_AGENT_CODING_AGENT_SESSION_DIR` / `PRIME_AGENT_CODING_AGENT_DIR` | `prime-agent` | second / third / sessions-root | `pi.py:377-379` |

Note for the emitter harness: `session-artifacts/` always sits as the sibling of the
sessions root, so when `SSGREP_PRIME_AGENT_SESSIONS_DIR=/data/sessions` the artifacts
root is `/data/session-artifacts`.

## 9. README cross-check ("Supported coding agents")

The README section largely matches the parser. Four findings:

1. **External-root runtime is `native`, not `claude`.** README frames
   `SSGREP_TRANSCRIPT_DIRS` roots as "Claude Code ... external roots". The parser
   stamps those sessions `runtime="native"` (`native.py:26-27`); only the
   `~/.claude/projects` tree gets `runtime="claude"` (`native.py:20-22`) and notes get
   `runtime="ssgrep"` (`native.py:23-25`). The plan's T7 round-trip criteria
   ("runtime == 'claude' on all chunks") can only pass if emitted files sit inside a
   synthetic `CLAUDE_CONFIG_DIR/projects/...` tree. Ingesting via
   `SSGREP_TRANSCRIPT_DIRS` produces `runtime == 'native'` chunks. T7 and T13 must
   agree on one path. Recommendation: accept `runtime == 'native'` for the claude
   emitter output and treat the `claude` census bucket as satisfied by the file
   format; if a true `claude` runtime label is wanted for stratification, emit a
   small number of projects-tree files instead.
2. **Pi env overrides under-documented in README.** README lists only
   `SSGREP_PI_SESSIONS_DIR` and `PI_SESSION_DIR`; the parser also honors
   `PI_CODING_AGENT_SESSION_DIR` and `PI_CODING_AGENT_DIR` (`pi.py:364-365`).
3. **Prime Agent env overrides under-documented in README.** README lists only
   `SSGREP_PRIME_AGENT_SESSIONS_DIR` and `PRIME_AGENT_SESSION_DIR`; the parser also
   honors `PRIME_AGENT_CODING_AGENT_SESSION_DIR` and `PRIME_AGENT_CODING_AGENT_DIR`
   (`pi.py:374-379`).
4. **"Child artifacts" for Pi is imprecise.** Pi has no artifacts directory. Child
   Pi sessions (`rlmDepth > 0`) are discovered and indexed as subagents, not
   excluded (`pi.py:114-115, 323`). The always-excluded-children claim applies only
   to Prime Agent's `session-artifacts/` (`pi.py:381-390`).

Confirmed as matching the parser (no action): Codex reasoning / tool-result echoes /
bookkeeping are excluded and assistant tool calls indexed as tool use
(`codex.py:238-258`); Pi thinking/tool-result noise is excluded (`pi.py:158-186`);
OpenCode's `XDG_DATA_HOME`/`SSGREP_OPENCODE_DB` discovery and read-only snapshot
semantics (`opencode.py:51-73`).

## 10. Emitter contract (T7-T11)

Each emitter writes one output directory per runtime that, when selected by the
documented overrides, makes the adapter discover exactly the intended sessions with
`malformed_records == 0` and no skips. Emitters write fresh fully synthetic content
(fictional projects and personas only), deterministically (fixed seed, stable field
order), and files whose final line ends with a newline.

### 10.1 T7 - Claude/native emitter

- [ ] Layout: choose one of two shapes. Option A (runtime `claude`): a synthetic
  `CLAUDE_CONFIG_DIR/projects/<projectDir>/<sessionId>.jsonl` tree with exactly 2
  parts under `projects/` (`discovery.py:192-213`) and no `memory/`,
  `tool-results/`, or `workflows/` subpaths (`discovery.py:43-84`). Option B
  (runtime `native`): any flat directory; at least one of the first 5 non-empty
  lines must carry a `type` key (`discovery_roots.py:58-92`), and ingestion goes
  through `SSGREP_TRANSCRIPT_DIRS=<path>:native=<dir>`.
- [ ] One JSONL line per record, types restricted to `records.py:9-25`, every line
  under 500,000 bytes and newline-terminated.
- [ ] Records: `user`/`assistant` pairs with `message.role`, `message.content` as a
  list of `{type: text}` blocks, `timestamp` ISO-8601 UTC, `sessionId`, `uuid`,
  `cwd`; session title as a `custom-title` record; optional `gitBranch`.
- [ ] Block hygiene: assistant content uses `text` and `tool_use` only. No
  `thinking`, `tool_result`, or `image` blocks, which add no episode signal
  (`signal.py:30-42`). For `files_touched`, include a `tool_use` block with
  `file_path` (`metadata.py:103-133`).
- [ ] Segmentation: one planned episode per text-bearing `user` record
  (`episodes.py:81-87`). Keep `user`/`assistant` pairing aligned so each prompt has
  its intended response in the same episode.
- [ ] Episode ids: `f"{session_id}:ep:{n}"` (`episodes.py:8-12`).

### 10.2 T8 - Codex emitter

- [ ] Output to a directory rooted by `SSGREP_CODEX_SESSIONS_DIR`
  (`codex.py:270`); any layout under it works because discovery is recursive
  `*.jsonl` (`codex.py:296-299`).
- [ ] Every file starts with a `session_meta` record whose `payload.id` is non-empty
  (`codex.py:97-106`), and carries `cwd`.
- [ ] Envelope lines `{"timestamp", "type", "payload"}` per section 5.2. Repeat
  `turn_context` at turn boundaries so cwd/model changes are tracked
  (`codex.py:203-207`).
- [ ] Conversation: `response_item` messages with roles `user` and `assistant`,
  content blocks `input_text`/`output_text`/`text`; no `developer` role
  (`codex.py:238-239`). Tool events `function_call`/`custom_tool_call` placed
  immediately after the assistant message they belong to so they append to its
  blocks (`codex.py:251-256`).
- [ ] Emit no `reasoning`, no `function_call_output`, no `custom_tool_call_output`,
  and no bookkeeping record types (`codex.py:257-258`). They are silent drops, so
  emitting them only wastes corpus bytes.
- [ ] Uuids are computed by the adapter as `codex:<seq>:<timestamp>`
  (`codex.py:212`); emitter must keep timestamps strictly increasing so seqs are
  stable. Lines newline-terminated, under 500,000 bytes.

### 10.3 T9 - Pi emitter

- [ ] Output directory rooted by `SSGREP_PI_SESSIONS_DIR` (`pi.py:364`). One file
  per session.
- [ ] First record must be `{"type": "session", "id": ...}` (`pi.py:107-110`).
- [ ] Emit only the kept entry types: `session`, `message`, `model_change`,
  `git`/`git_state`, and one title type (`pi.py:38-53, 200-226`). Never emit
  `toolResult` role messages (`pi.py:238-239`).
- [ ] Content blocks: `text`, plus `toolCall` on assistant messages for tool use
  (`pi.py:163-184`). No `thinking`, `image`, or extension blocks.
- [ ] Title as `custom-title`/`name` before the first user message so the title is
  set before every episode (`pi.py:214-226`, `metadata.py:203-209`).
- [ ] For subagent strata: set `parentSessionId` on the child `session` record and
  `rlmDepth` to a positive int (`pi.py:114-120`).

### 10.4 T10 - Prime Agent emitter

- [ ] Main sessions: identical record rules and layout to T9.
- [ ] Additional root: a sibling `session-artifacts/` directory of the sessions root
  (`pi.py:381-382`). When the harness sets
  `SSGREP_PRIME_AGENT_SESSIONS_DIR=/data/prime/sessions`, plan for
  `/data/prime/session-artifacts`.
- [ ] Subagent content: any `*.jsonl` under `session-artifacts/`; the record set
  still needs a `type = "session"` header first (`pi.py:107-112`), but `is_main`
  is forced to false by location (`pi.py:384-390`).
- [ ] Session ids on normalized records will be `prime-agent:<id>` (`pi.py:99-100, 128-134`);
  the benchmark should treat that prefix as the runtime-session identifier.

### 10.5 T11 - OpenCode SQLite emitter

- [ ] One SQLite file at the `SSGREP_OPENCODE_DB` path (or the `opencode/opencode.db`
  default location). Tables `session`, `message`, `part` with the required columns
  (`opencode.py:82-90`).
- [ ] `message.data`/`part.data` hold JSON strings. Roles are only `user` and
  `assistant`, or the message is skipped (`opencode.py:591-594`); parts are only of
  types `text`, `tool`, `file`, `patch` (`opencode.py:475-501`).
- [ ] Per session, `time_created` increases monotonically across messages and parts
  so ordering matches intent (`opencode.py:357-384`).
- [ ] Excluded part types never appear in emitted data (`opencode.py:24-33`); do not
  rely on `ignored: true` to hide text, those parts are dropped silently
  (`opencode.py:477-479`).
- [ ] Determinism: identical rows, identical data JSON, identical timestamps produce
  a stable fingerprint and stable incremental re-ingestion (`opencode.py:213-228`).

### 10.6 Benchmark invariants (T13 ingest + all emitters)

- [ ] Every emitted file/line is newline-terminated (`base.py:96-98`; `codex.py:50-58`;
  `pi.py:61-68`).
- [ ] Dry-run each emitter's output through its real adapter in the T-test: expect
  `malformed_records == 0` and `skipped_records == 0`.
- [ ] The runtime labels visible in the index match the dataset's intended labels:
  `claude`/`native`, `opencode`, `codex`, `pi`, `prime-agent`, and the census in
  the v1 manifest.
- [ ] Episode ids are always `<session_id>:ep:<n>` (`episodes.py:8-12`) so qrel and
  target ids from T12 resolve against the index.

## Appendix: line-references for the sources read

- `src/ssgrep/sessions/discovery.py`: `:22-84`, `:191-279`
- `src/ssgrep/sessions/discovery_roots.py`: `:38`, `:58-121`, `:129-158`
- `src/ssgrep/sessions/adapters/native.py`: `:14-28`
- `src/ssgrep/sessions/adapters/opencode.py`: `:24-103`, `:203-228`, `:346-637`
- `src/ssgrep/sessions/adapters/codex.py`: `:38-111`, `:190-258`, `:264-346`
- `src/ssgrep/sessions/adapters/pi.py`: `:38-55`, `:200-260`, `:264-390`
- `src/ssgrep/sessions/episodes.py`: `:8-132`
- `src/ssgrep/pipeline/episodes.py`: `:72-144`
- `src/ssgrep/sessions/signal.py`: `:18-53`
- `src/ssgrep/sessions/metadata.py`: `:103-209`
- `src/ssgrep/sessions/records.py`: `:7-25`
- `src/ssgrep/sessions/adapters/base.py`: `:26-116`
- `src/ssgrep/utilities/paths.py`: `:47-119`
- `src/ssgrep/sessions/adapters/registry.py`: `:14-41`