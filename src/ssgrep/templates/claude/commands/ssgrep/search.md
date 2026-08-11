---
name: "ssgrep search"
description: Search AI coding session transcripts for past solutions
category: Workflow
tags: [search, sessions, transcripts]
---

Search AI coding session transcripts to find how problems were solved before.

**Input**: A search query string (required).

**Steps**

1. **Parse the query**
   Accept a natural language search query from the user context.

2. **Run the search**
   ```bash
   SSGREP_BIN_PLACEHOLDER search "<query>" --json
   ```

3. **Parse the results**
   The JSON response contains:
   - `ok: true/false` — success or error
   - `data.results` — array of matching episodes
   - Each result has: `ref`, `title`, `timestamp`, `score`, `excerpt`, `files_touched`

4. **Display results**
   - If successful: show the matching episodes with scores and excerpts
   - If error: display the condition and suggested recovery command
   - If no results: suggest refining the query

**Examples**

```bash
SSGREP_BIN_PLACEHOLDER search "TypeError cannot read property" --json
SSGREP_BIN_PLACEHOLDER search "how to fix git merge conflicts" --json
SSGREP_BIN_PLACEHOLDER search "authentication middleware" --json
```

**Exit codes**
- 0: Success with results
- 2: Usage error (empty query)
- 3: No results found
- 4: Missing or corrupt index
