---
name: ssgrep
description: Search local coding-agent transcripts for past solutions, decisions, and errors before solving a problem again
---

# ssgrep — search past sessions before solving

ssgrep searches the transcripts of every coding agent installed on this machine —
Claude Code, OpenCode, Codex, Pi, and Prime Agent — plus notes captured with
`note`. It runs locally: no LLM call, no network after the one-time model
download, no daemon. Transcript content never leaves the machine, and ssgrep
never writes to the transcripts it reads.

## Operating rules

<!-- ssgrep-rules:short -->
1. **Search before solving.** Run at least one `search` before the first edit or
   debug action on a task. Past sessions have already answered more questions
   than you expect, including ones never written down anywhere else.
2. **Open the hit before writing code.** When a result looks relevant, `show` it.
   An excerpt is a pointer, not the answer.
3. **Calibrate with a control query; never quote a fixed threshold.** Scores
   are relative to the corpus and shift when the corpus or the engine changes,
   so run a nonsense query first and note its top score: that is the floor. A
   query whose top score sits at the floor is not necessarily wrong -- it means
   nothing matched literally and the ranking is semantic only, so judge those
   results by reading them, not by the number. Re-run the control when the
   corpus changes.
4. **Read every returned hit, not just the first.** The answer is often below the
   top line. Never pipe results to `head -1`.
5. **Capture durable lessons with `note`.** Title the note as the question you
   would later search for; put the answer in the body.
6. **Verify your own content came back.** After writing a note, search a
   distinctive token from that note — never a topic word — and confirm that
   specific note is in the results. "Something returned" is not "my note
   returned"; an older episode on the same topic will answer a topic query and
   read as success.
7. **Say whether a capability was verified or merely used.** They are different
   claims, and only one of them is evidence.
<!-- /ssgrep-rules:short -->

## Commands

Reconcile newly written sessions into the index (incremental, safe to repeat):

```bash
SSGREP_BIN_PLACEHOLDER index --quiet
```

Search across every project and runtime:

```bash
SSGREP_BIN_PLACEHOLDER search "how did we fix the flaky webhook test" --json
SSGREP_BIN_PLACEHOLDER search "TypeError cannot read property" --limit 20 --json
SSGREP_BIN_PLACEHOLDER search "auth middleware" --where "runtime = 'codex'" --json
```

Retrieve one episode in full, using a `ref` from a search result:

```bash
SSGREP_BIN_PLACEHOLDER show "<ref>" --json
```

Check index health, counts, model binding, and the per-runtime census:

```bash
SSGREP_BIN_PLACEHOLDER status --json
```

Capture a durable lesson, then verify it is retrievable:

```bash
SSGREP_BIN_PLACEHOLDER note --title "why does the flaky webhook test fail on CI" --body "-"
SSGREP_BIN_PLACEHOLDER search "<distinctive token from the note>" --json
```

`note` indexes the note as it writes it: it is searchable immediately, with no
separate index run and no rebuild. Notes are written to ssgrep's own data
directory, never into your agents' transcript files.

Reprint these rules, in full or as a block to paste into a project's `AGENTS.md`
or `CLAUDE.md`:

```bash
SSGREP_BIN_PLACEHOLDER rules
SSGREP_BIN_PLACEHOLDER rules --short
```

## Reading results

Each result carries a `ref`, `title`, `timestamp`, `score`, `excerpt`,
`files_touched`, and the `runtime` and agent that produced it. Rank order is what
matters; the absolute score is meaningful only against the control query from
rule 3. Expect a wide spread: a query carrying a literal identifier or error
string scores orders of magnitude above a paraphrase of the same thing, and a
paraphrase often lands at the control floor while still returning the right
episode. When results are broad, narrow with a `--where` metadata predicate
rather than by trusting a score cutoff.

Exit codes: `0` success, `2` usage error, `3` no results, `4` missing or corrupt
index. On code `4`, run `SSGREP_BIN_PLACEHOLDER index` and retry.
