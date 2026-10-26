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

CI runs the same Poe quality and test commands natively on macOS with Python
3.11 and 3.13, and runs the full non-network gate in a Dagger Linux container
on the engine's native architecture. The Dagger graph also validates the
installed wheel's CLI identity. Use `uv run poe ci` for the host-native gate
and `dagger check` for Linux parity.

### Dagger checks and feedback loop

The Linux gate is a Dagger module (`.dagger/modules/ssgrep-ci`). Install the
Dagger CLI at version `1.0.0-beta.21` (`sh scripts/install_dagger.sh DIR`
installs it with a checksum check) and have a container runtime available;
Dagger starts its own engine there. Set `DAGGER_ENGINE` only to use an engine
elsewhere (see `dagger help engine`).

The module's `sdk/` and `clients/` directories are generated, not committed.
Run `dagger generate -y` after cloning and after a Dagger upgrade; CI runs it
before every check. They appear as untracked files because Dagger cannot load
the module if they are gitignored. Never commit them:
`tests/test_repo_hygiene.py` fails if they are tracked.

```sh
dagger generate -y
dagger check -l --all
dagger check --ssgrep-ci --check ssgrep-ci/lint --progress plain
dagger check --parallel 3 --progress plain
```

Checks: `lint`, `format`, `typecheck`, `sizecheck`, `module-quality` (the
Dagger module itself), `package` (wheel installed with runtime dependencies
only, then `scripts/check_cli_identity.sh`), and the test suite in four
shards:

- `tests-views`: `tests/accounting/views` and `tests/accounting/readers`.
- `tests-collect`: `tests/accounting/collect` and `tests/accounting/capture`.
- `tests-eval`: the five slowest eval tests (`_HEAVY_EVAL_TESTS` in the
  module). It first proves each ID matches exactly one collected test, since
  `--deselect` matches by prefix.
- `tests`: everything else, including the hygiene test. New test directories
  land here automatically.

The shards partition the suite: every test runs exactly once. The tests run as
root on Linux (asserted), on the engine's native architecture. On Linux/arm64
the locked `fast-plaid` package is omitted because it publishes no wheel for
that platform; every other platform installs the complete lockfile. The
pinned embedding model is prefetched into a cache volume before the tests.

GitHub Actions runs CI on pushes to `main`, pull requests, and manual dispatch.
Where jobs run is repository configuration, not workflow code. Each variable is
a JSON `runs-on` value; unset, jobs use GitHub-hosted runners:

| Variable | Jobs | Default |
|---|---|---|
| `CI_RUNS_ON` | `dagger-check`, network tests, PyPI publish | `"ubuntu-latest"` |
| `MACOS_RUNS_ON` | `fast-core`, `test-macos` (needs Docker) | `"macos-latest"` |
| `DOCKER_RUNS_ON` | release build (needs Docker) | `"ubuntu-latest"` |

`DAGGER_ENGINE` (optional) points `dagger-check` at an existing engine.
PyPI publishing waits for approval in the `pypi` environment (`v*` tags only).

`dagger-check` does not run for pull requests from forks; `fast-core` and
`test-macos` do. A skipped required check counts as passing, so never merge a
fork pull request on its checks alone: review the source, push the reviewed
commit to a branch in this repository, and merge after CI passes there. Do not
use `pull_request_target` to check out fork code.

Measured on a 6-CPU arm64 host (Dagger `v1.0.0-beta.21`, `--parallel 3`,
about 6,220 tests, warm caches): the whole `dagger-check` job took about
8.5 min; each test shard about 6.5 min. One container running the whole suite
took about 16 min, and the previous emulated amd64 Linux job about 53 min.

Dagger reuses operation results when inputs and the engine match, and the
uv/pip download caches avoid refetching packages. Lint, format, typecheck and
sizecheck are reused when their inputs are unchanged ("CACHED" in
`--progress plain`); the test shards rerun for every commit because they
receive fresh Git metadata. A `pyproject.toml` or `uv.lock` change rebuilds
the dependency layer. Elapsed time alone does not prove cache hits; look for
CACHED steps. Git metadata in Dagger reflects `HEAD`, not the staging index,
so run `uv run poe hygiene` before committing.

Dagger operation caching is separate from LLM prompt/token caching. Ordinary
checks make no LLM calls; paid dataset generation, judging, and network tests
stay outside the push gate. For agent-assisted work, run the check affected by
the change first and report its status and failure excerpt, not full logs.
Run the full graph before pushing.

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
