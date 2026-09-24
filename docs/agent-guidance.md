# Agent Guidance

`ssgrep init` installs one shared guidance document — ssgrep's operating rules — into every supported runtime's global skills directory. It is the same text `ssgrep rules` prints, so an agent reading its skill and a human reading the CLI never see different rules. Only the frontmatter varies per runtime; the body is byte-identical everywhere, and the rules it carries are:

1. Search before solving — run a search before the first edit or debug action.
2. Open a promising hit with `show` before writing new code.
3. Calibrate with a nonsense control query; never trust a fixed score threshold.
4. Read every returned hit, not just the first.
5. Capture durable lessons with `note`, titled as the question you'd search for.
6. Verify your own content came back, using a distinctive token from it.
7. Say whether a capability was verified or merely used.

To put them in a project's instructions file, paste the short block:

```bash
ssgrep rules --short >> AGENTS.md    # or CLAUDE.md
```

## Upgrades

Re-run `ssgrep init` after upgrading; each runtime reports `installed`, `updated`, `already_installed`, `adopted`, or `user_modified`. A copy you have edited is reported `user_modified` and never overwritten — diff it against `ssgrep rules` and re-apply your edits if you want the newer rules.

## Accuracy

Every command, flag, and runtime named in the guidance is pinned by tests against the live code, so it cannot drift into promising behaviour ssgrep no longer has.
