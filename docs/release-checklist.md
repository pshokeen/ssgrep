# Release Checklist

Tracks what remains for release automation. The original done-when was "a
dry-run publish to TestPyPI succeeds"; **that done-when is moot.** ssgrep is
sold as a paid, direct download — a buyer downloads the wheel and installs it
themselves — with **no public package registry**: no PyPI, no TestPyPI, no
Homebrew. This checklist replaces the old PyPI dry-run criterion with what
actually remains before a real sale can happen.

## What already exists

- `.github/workflows/release.yml` — triggers on `v*` tags, builds and tests
  the package, verifies CLI identity in a clean venv (per 4.1a), builds a
  wheel and sdist, and uploads them as a workflow artifact (90-day retention).
  **There is no publish job.** The workflow cannot publish anywhere even if
  triggered — it has no PyPI credentials, no OIDC trust configuration, and no
  step that calls any package index. A human downloads the built artifact
  from the Actions run and uploads it to the storefront by hand.
- `pyproject.toml` carries a `"Private :: Do Not Upload"` classifier —
  verified directly (see the relicensing report) to make PyPI itself reject
  an upload server-side. This is a second, independent interlock against
  accidental publication, on top of the workflow simply not trying.
- None of this has been exercised end-to-end. A workflow file that has never
  run is not verified automation — it's a plan, same as before.

## What remains

### 1. Commercial license — decided and applied

`LICENSE` now holds the **PolyForm Internal Use License 1.0.0**. `pyproject.toml`
line 29 declares `license = {text = "PolyForm Internal Use License 1.0.0"}`.

**What this permits:** Buyers may use ssgrep — including modifying it — for the
internal business operations of themselves and their company. Commercial use and
modification are allowed.

**What this forbids:** Distribution to anyone else, sublicensing, and transferring
the license. A buyer gets ssgrep for themselves or their company; they cannot
resell it. (This solves the original relicensing problem: MIT would have let them
do exactly that.)

**One open item: lawyer review before taking money.** The license's permitted-purpose
clause reads "internal business operations of you and your company." The Definitions
section explicitly includes "sole proprietorship" under "your company," which covers
freelancers and independent contractors. The residual ambiguity is narrower than
originally framed: *non-business personal use* by someone with no business operations
at all. If ssgrep's buyer base skews toward individuals filing as sole proprietors —
which is likely for a developer tool — this definition covers the majority case and
the risk is confined. Worth a lawyer's eyes before you take the first payment, though.

### 2. Decide whether you need the GitHub Actions build at all

Two independent paths get you a wheel; you don't need both.

**Manual, no GitHub remote required** (this repo doesn't have one yet —
`git remote -v` is empty): build directly from a clean checkout.

```bash
rm -rf dist/
uv build
unzip -p dist/*.whl '*.dist-info/entry_points.txt'
```

The output must read `ssgrep = ssgrep.cli:main`, matching `[project.scripts]`
in `pyproject.toml`. If it says anything else (e.g. `ssgrep = usecli:main`) —
stop, do not upload, and find out why the built artifact disagrees with
source. `dist/` is gitignored, so a stale build can sit there indefinitely
without showing up in `git status` or a code review to warn you. Delete
`dist/` before building, and re-run this check immediately before every
upload — not just the first one.

**CI-built, requires pushing this repo to GitHub:** push a `v*` tag, let
`release.yml` run lint, format check, typecheck, test, and the CLI-identity
check in a clean environment, then download the `build-artifacts` artifact
from the Actions run. This gets you the same wheel with more independent
verification than a local build — see 5.6's CI note if you go this route.
The corpus-coupled tests skip cleanly on a corpus-less runner (the
session-scoped real-corpus fixture guards itself), so a bare CI runner
passes the suite.

**Branch protection:** `main` is protected on GitHub and requires all six
CI matrix legs (`test (ubuntu-latest|macos-latest, 3.11/3.12/3.13)`) to be
green before anything merges. Set via
`gh api -X PUT repos/pshokeen/ssgrep/branches/main/protection`; if the CI
job matrix changes, update the required-status-check contexts to match.

### 3. Create the storefront listing

Register the product with whichever of Lemon Squeezy, Paddle, or Gumroad you
choose, upload the wheel (the sdist is optional — it's a source-availability
artifact, not something most buyers need to install), set the price, and
attach the finalized license text from step 1 wherever the platform lets you
attach product terms (a custom-terms field, an uploaded EULA document, or
equivalent — the exact mechanism differs per platform and wasn't verified as
part of this pass; check the platform's own seller docs).

### 4. Do a real buyer-side test

Buy the product yourself (or have someone else do it), download the
delivered wheel, and confirm the three install forms documented in
`README.md`'s Installation section against that *actually-downloaded* file,
not just a locally-built one — `uvx --from`, `uv add`, and `pip install` into
a venv. All three were verified against a locally-built wheel while writing
this checklist; they have not been verified against a file that went through
an actual storefront delivery pipeline, which can differ (e.g., if the
platform repackages or renames the file).

### 5. Only after step 4 succeeds, call the release path verified

Until a real purchase has been completed end-to-end, treat the release path
as *built but unverified against a real sale*.

## Why this is a checklist and not just "run it"

Steps 1 and 3 are decisions and accounts that exist outside this codebase and
outside any automated check's reach: a legal choice the owner has to make
(step 1), and a storefront account with product-listing rights (step 3). No
amount of local testing substitutes for actually completing a purchase
through the chosen platform and confirming what a real buyer receives.
