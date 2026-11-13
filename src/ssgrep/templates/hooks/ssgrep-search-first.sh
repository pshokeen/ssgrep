#!/bin/sh
# ssgrep search-first hook (PreToolUse on Edit|Write)
#
# Fires before the FIRST write action of a Claude Code session. Runs a local
# ssgrep search seeded from the file being edited and injects the top hits as
# context so the agent reads prior work before writing new code. Never blocks
# the tool call: search results are advice, not a gate.
#
# Installed and managed by `ssgrep init --install-hooks`. Offline by design:
# ssgrep runs locally; no network, no transcript writes.
set -u
PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
command -v jq >/dev/null 2>&1 || exit 0

INPUT=$(cat)

TOOL_NAME=$(printf '%s' "$INPUT" | jq -r '.tool_name // empty' 2>/dev/null)
case "$TOOL_NAME" in
  Edit|Write|NotebookEdit) ;;
  *) exit 0 ;;
esac

SESSION_ID=$(printf '%s' "$INPUT" | jq -r '.session_id // empty' 2>/dev/null)
[ -n "$SESSION_ID" ] || exit 0

# once per session: a marker file under the ssgrep data dir
STATE_DIR="${TMPDIR:-/tmp}/ssgrep-hooks/$SESSION_ID"
if [ -e "$STATE_DIR/searched" ]; then
  exit 0
fi
mkdir -p "$STATE_DIR" 2>/dev/null || exit 0

FILE_PATH=$(printf '%s' "$INPUT" | jq -r '.tool_input.file_path // empty' 2>/dev/null)
QUERY=$(basename "$FILE_PATH" 2>/dev/null)
[ -n "$QUERY" ] || QUERY="$TOOL_NAME"

SS=SSGREP_BIN_PLACEHOLDER
SEARCH_OUT=$("$SS" search "$QUERY" --limit 3 --json 2>/dev/null) || true

# mark searched regardless of result: one shot per session, empty results included
: > "$STATE_DIR/searched" 2>/dev/null || true

[ -n "$SEARCH_OUT" ] || exit 0
COUNT=$(printf '%s' "$SEARCH_OUT" | jq -r '.data.results | length' 2>/dev/null)
[ "$COUNT" -gt 0 ] 2>/dev/null || exit 0

REFS=$(printf '%s' "$SEARCH_OUT" | jq -r '.data.results[0:3] | map("`" + .ref + "` — " + ((.title // "")[0:80])) | join("\n")' 2>/dev/null)
[ -n "$REFS" ] || exit 0

jq -nc --arg msg "ssgrep found $COUNT past episode(s) matching \"$QUERY\". Read them before writing new code (rule 1: search before solving):
$REFS
Use \`ssgrep show <ref>\` to open one." '{hookSpecificOutput: {hookEventName: "PreToolUse", additionalContext: $msg}}'
