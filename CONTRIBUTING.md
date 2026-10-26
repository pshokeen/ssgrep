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
3.11 and 3.13, and runs the full non-network gate in Dagger's Linux/arm64
environment. The Dagger graph also validates the installed wheel's CLI
identity. Use `uv run poe ci` for the host-native gate and `dagger check` for
Linux parity.

### Dagger checks and feedback loop

Install the Dagger CLI at version `1.0.0-beta.21` and make a local Docker
engine available (Docker Desktop or Colima). The CI workflow uses that same
CLI version and runs on the dedicated Mac mini engine; it does not use Dagger
Cloud Engines or GitHub-hosted runners.

```sh
dagger check -l --all
dagger check --ssgrep-ci --check ssgrep-ci/lint --parallel 3 --progress plain
dagger check --fail-fast --parallel 3 --progress plain
dagger check --parallel 3 --progress plain
```

Select other checks by name from `dagger check -l --all`, or pass
`--ssgrep-ci` to select the ssgrep module. GitHub Actions runs on pushes,
trusted same-repository pull requests, and manual dispatch. The Dagger job
uses the mini's loopback engine; the network-test and release publish jobs use
the same mini-only runner label (the macOS matrix, `fast-core` and the release
build/test job need Docker and stay on the existing runner). Fork pull requests do not execute automatically on
this shared host. If the mini is offline or occupied, jobs queue rather than
falling back elsewhere.

For a fork pull request, review the source first, then validate the reviewed
commit from an internal branch with manual dispatch. Do not run unreviewed fork
code on the shared runner or use `pull_request_target` to check it out.

Dagger reuses immutable operation results when the operation inputs and engine
match. uv/pip download caches avoid fetching the same packages again when a
dependency-install step must rerun. Source changes invalidate affected
checks; dependency-manifest or lock changes also invalidate dependency
installation and downstream work. Cache reuse depends on the same Dagger
engine and its persistent cache volume; different engines do not share these
results by default. The single mini has finite capacity, so use a selected
check while iterating, then the full gate before pushing. It serializes runner
jobs and allows at most three Dagger checks at once (only `tests` and
`tests-eval` run pytest; the rest are light). GitHub cancels superseded
ordinary CI runs on the same branch, but does not cancel release publication.

Measured on the mini's ARM64 engine (Dagger `v1.0.0-beta.21`, full gate,
`--parallel 3`, 926 tests) for a new commit with a warm dependency cache:
about 52–54 s (about 58–61 s with `--parallel 2`; down from about 124 s). Three changes got there: dependency
bytecode is compiled once into the cached dependency layer (the test step no
longer recompiles torch and transformers on every run), the wheel identity
check runs as its own `package` check, and the five slowest eval tests run in
a parallel `tests-eval` check. `tests` deselects exactly those tests and
`tests-eval` selects exactly those tests, so every test runs once (926 in
total); renaming one of them makes `tests-eval` fail rather than skip it, and
`_HEAVY_EVAL_TESTS` in the Dagger module lists them. `tests` and `tests-eval`
(about 40 s each) are now the critical path; every other check finishes in
under 20 s. The first run after
a dependency change (`pyproject.toml` or `uv.lock`) is slower, about 68 s, because it
rebuilds and byte-compiles the dependency layer; the system packages
(`build-essential`, `git`) are installed in an earlier layer that does not depend on
the lockfile, so they stay cached. Lint,
format, typecheck and sizecheck results are reused when their inputs are
unchanged ("CACHED" in `--progress plain`); the tests re-run for every commit
because they receive fresh Git metadata. A source-only edit reruns the
project install and checks but keeps the dependency stage cached; a
`pyproject.toml` or `uv.lock` edit rebuilds it. A checkout without Git
metadata fails to load the workspace. Elapsed time alone does not prove cache
hits; look for CACHED steps.

Dagger operation caching is separate from LLM prompt/token caching. Ordinary
checks make no LLM calls; paid dataset generation, judging, and network tests
remain outside the ordinary push gate. Git metadata in Dagger reflects
`HEAD`, not the local staging index, so run `uv run poe hygiene` before
committing to check staged repository state.

For agent-assisted work, start with the named check affected by the change.
Report its status and the relevant failure excerpt; avoid sending full
successful logs back into the model context. Run the full graph before push.

Dagger Cloud tracing is not enabled until an authorized scoped token is
available. Managed Cloud Checks remain disabled so computation stays on the
mini. GitHub Actions logs and check statuses remain visible in the repository
while Cloud access is pending.

### Mac mini runner operations

Check runner online status and its `ssgrep-mac-mini` label under repository
Settings → Actions → Runners. If a job waits, inspect that runner first; never
change the selector to an off-host runner. SSH access may require completing
the Tailscale browser reauthentication prompt.

The repository-scoped runner is `ssgrep-mac-mini` (GitHub runner ID `22`).
It runs as `_ssgrep_runner`, with installation and work directories under
`/Users/_ssgrep_runner/actions-runner`. Its system LaunchDaemon is
`actions.runner.pshokeen-ssgrep.ssgrep-mac-mini`; it does not require a GUI
login. The plist lives in `/Library/LaunchDaemons/`. Logs are in
`/Users/_ssgrep_runner/Library/Logs/actions.runner.pshokeen-ssgrep.ssgrep-mac-mini/`.

Inspect or restart only this service, and restart only while its GitHub
`busy` field is false:

```sh
sudo launchctl print system/actions.runner.pshokeen-ssgrep.ssgrep-mac-mini
sudo launchctl kickstart -k system/actions.runner.pshokeen-ssgrep.ssgrep-mac-mini
```

For automated administration, never feed credentials into a TTY or interactive
password prompt. Use `ssh -T -o BatchMode=yes`, deliver a required sudo
password through a subprocess stdin pipe, and capture/redact output before
displaying it. Never put password or token values in command arguments,
shell tracing, files, or transcripts. Supply short-lived registration tokens
through `ACTIONS_RUNNER_INPUT_TOKEN`, not `--token`; remove the variable after
configuration. These precautions prevent terminal echo but do not replace
rotation of previously exposed credentials.

On the mini, inspect the native ARM64 Dagger Engine in the default Colima
profile. It shares the Docker daemon with Wedvilla, but uses its own
`ssgrep-dagger-engine` container and `ssgrep-dagger-cache` volume:

```sh
colima status --profile default
docker --context colima ps
docker --context colima logs --tail 100 ssgrep-dagger-engine
docker --context colima system df -v
```

Restart only the ssgrep engine when needed:

```sh
docker --context colima restart ssgrep-dagger-engine
```

The `ssgrep-dagger-cache` volume survives that restart. Do not run global
Docker prune commands or stop the default Colima profile to recover ssgrep;
those actions can remove or interrupt Wedvilla data and services. Cache
eviction is handled by the engine's configured garbage collector. The engine
API binds to loopback port `1234` only. A job that cannot reach the mini
remains queued until the runner or engine recovers.

The Linux/arm64 Dagger environment omits only `fast-plaid`, which has no
Linux/aarch64 wheel; the root lockfile and native macOS jobs retain the
complete dependency graph. Ordinary checks do not run dataset generation.

After an authorized `DAGGER_CLOUD_TOKEN` is stored as a scoped GitHub Actions
secret, Dagger can attach a trace to the local-engine run. The token enables
observability only; `DAGGER_ENGINE=tcp://127.0.0.1:1234` must remain set so
execution stays on the mini. Review current Cloud plan limits before enabling
traces. Do not enable managed Cloud Checks.

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
