# Contributing to ssgrep

Thanks for your interest in ssgrep. This document covers environment setup,
how to run the checks that gate a merge, and what a pull request needs
before it can be accepted.

## Environment setup

ssgrep uses [`uv`](https://docs.astral.sh/uv/) for dependency management and
task running. Install `uv` (version `>=0.11.32`, per `pyproject.toml`), then
from a checkout of this repository:

```sh
uv sync
```

This installs the runtime and dev dependency groups into a local `.venv/`
(fast on a re-run — `uv` resolves and reuses its cache). You do not need to
activate the virtualenv; every command below is run through `uv run`, which
uses it automatically.

## Running the tests

Always run the test suite as:

```sh
uv run pytest
```

**Do not** run `.venv/bin/pytest` or a bare `pytest` directly. This repo has
been bitten by exactly this before: a stray `ssgrep` binary earlier on
`PATH` than `.venv/bin/ssgrep` (historically `/opt/homebrew/bin/ssgrep`) and
an unrelated `pytest` install shadowing this project's `.venv/bin/pytest`
have both caused tests and CLI invocations to silently run against the
wrong binary. `uv run` always resolves to *this checkout's* environment, so
it's the only form that's guaranteed correct. The same applies to the CLI
itself — invoke it as `uv run ssgrep ...` (or `.venv/bin/ssgrep` if you need
the resolved path directly), never a bare `ssgrep`.

A verbose run of just the test suite is also available as a `poe` task (see
below): `uv run poe test`.

## The pre-PR check chain

This repository uses [`poe`](https://poethepoet.natn.io/) (`poethepoet`,
installed via the dev dependency group) to run its checks. Before opening a
pull request, run:

```sh
uv run poe lint        # ruff check . (no auto-fix)
uv run poe fmt          # ruff format --check . (formatting, check-only)
uv run poe typecheck   # ty check
uv run poe sizecheck   # src/**/*.py file-size gate (see below)
uv run poe hygiene     # repo-hygiene test: no agent-artifact/build-junk paths tracked
uv run poe test         # uv run pytest tests/ -v
```

or run the whole chain in one shot:

```sh
uv run poe ci
```

All of these are exactly what CI runs, so a clean `uv run poe ci` locally is
the strongest signal your PR will pass CI. `pyproject.toml`'s
`[tool.poe.tasks]` table is the source of truth for the current full task
list (including any hygiene or other checks added after this document was
written) — run `uv run poe --help` to see everything available. A few of
the listed tasks (for example, ones with "clean" or "format" in the name)
auto-fix files in place rather than just checking them; read a task's
definition in `pyproject.toml` before running it if you want to know
whether it will write to your working tree.

### File-size limits

`src/**/*.py` warns above 300 non-blank, non-comment lines and fails above
400 (docstrings are excluded from the count — this repo's long docstrings
document crash-recovery and data-loss invariants that mutation-tested tests
assert against). `uv run poe sizecheck` enforces this. If you're adding to a
file that's already close to the ceiling, prefer splitting the addition into
a new module over pushing the file over the limit.

### Test quality

Test assertions must verify the actual property under test, not a proxy or
an algebraic tautology (for example, asserting `hasattr(x, "field")` instead
of `x.field == <expected value>`, or asserting a spy was called `0` times
with no accompanying positive assertion that the code path is reachable at
all). This repo has a documented history of exactly this defect, so reviewers
will push back on it. Use the builder functions in `tests/conftest.py` (for
example `build_episode()`, `build_chunk()`) to construct complete, valid
instances for real assertions rather than hand-rolling partial fixtures.

## Pull request expectations

- Keep PRs focused — one logical change per PR is easier to review and to
  revert if something goes wrong.
- Include or update tests that would actually catch a regression in the
  behavior you're changing, not tests that merely restate the implementation.
- Update relevant documentation (`README.md`, `docs/`) when you change
  user-facing behavior.
- Make sure `uv run poe ci` passes locally before requesting review; CI runs
  the same chain and will not merge a red PR.
- Fill out the pull request template.

## Licensing of contributions

By contributing, you agree that your contributions are licensed under the
project's [MIT License](LICENSE). That's it — no agreement to sign.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By
participating, you're expected to uphold it.

## Reporting security issues

Please **do not** open a public issue for a security vulnerability — see
[`SECURITY.md`](SECURITY.md) for the private reporting channel.
