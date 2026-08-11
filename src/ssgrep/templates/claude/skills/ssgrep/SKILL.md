---
name: ssgrep
description: Search AI coding session transcripts for past solutions
triggers:
  - "how was X solved before"
  - "find past session about Y"
  - "search sessions for Z"
  - "ssgrep"
---

# ssgrep — Session Search

Search AI coding session transcripts to find how problems were solved before.

## Usage

```bash
SSGREP_BIN_PLACEHOLDER search "query text" --json
```

## Commands

- `SSGREP_BIN_PLACEHOLDER search <query>` — Search for relevant episodes
- `SSGREP_BIN_PLACEHOLDER show <ref>` — Show full context for a search result
- `SSGREP_BIN_PLACEHOLDER status` — Check index health and freshness
- `SSGREP_BIN_PLACEHOLDER index` — Build or update the search index

## Examples

```bash
# Find how a bug was fixed
SSGREP_BIN_PLACEHOLDER search "TypeError cannot read property" --json

# Show full context for a result
SSGREP_BIN_PLACEHOLDER show "session-id:ep:0" --json

# Check if index is up to date
SSGREP_BIN_PLACEHOLDER status --json
```
