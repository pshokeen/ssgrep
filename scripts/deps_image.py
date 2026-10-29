#!/usr/bin/env python3
"""Policy and plumbing for `.github/workflows/deps-image.yml`.

Pure functions decide what to delete and which pull requests to retire; thin commands call `gh`
and `git`. Tests drive the same functions with a fake `gh` and a local bare Git remote.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

MODULE_DIR = Path(".dagger/modules/ssgrep-ci/src/ssgrep_ci")
PIN_PATH = MODULE_DIR / "pin.py"
MODULE_PATH = MODULE_DIR / "__init__.py"
PACKAGE = "ssgrep-ci-deps"
KEEP_BUILDS = 8
BOT_LOGINS = frozenset({"app/github-actions", "github-actions", "github-actions[bot]"})
LINE_LIMIT = 100

# <UTC YYYYMMDDHHMMSS>-r<run id>-<lock hash>, optionally suffixed with an architecture.
_STEM = r"(\d{14})-r(\d+)-([0-9a-f]{8})"
_TAG = re.compile(rf"^{_STEM}(?:-(?:arm64|amd64))?$")
_STEM_ONLY = re.compile(rf"^{_STEM}$")
_BRANCH = re.compile(rf"^ci/deps-image-({_STEM})-a(\d+)$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")

Gh = Callable[[Sequence[str]], str]
_PR_FIELDS = "number,headRefName,author,isCrossRepository,baseRefName"
BuildKey = tuple[str, int]


class RetentionError(Exception):
    """The package state does not support a safe deletion plan."""


def parse_tag(tag: str) -> tuple[str, BuildKey] | None:
    """Split a build tag into its stem and chronological key, or None for foreign tags."""
    found = _TAG.match(tag)
    if not found:
        return None
    utc, run, lock = found.groups()
    return f"{utc}-r{run}-{lock}", (utc, int(run))


def proposal_branch(stem: str, attempt: int) -> str:
    if not _STEM_ONLY.match(stem):
        raise ValueError(f"not a build stem: {stem!r}")
    return f"ci/deps-image-{stem}-a{attempt}"


def _tags(version: Mapping[str, Any]) -> list[str]:
    return list(version.get("metadata", {}).get("container", {}).get("tags", []))


def plan_retention(
    versions: Sequence[Mapping[str, Any]],
    pinned_digest: str,
    candidate_stem: str,
    keep: int = KEEP_BUILDS,
) -> list[int]:
    """Return the package-version ids that are safe to delete.

    A build is the index tag plus its per-architecture tags, but a registry version (one digest)
    can carry tags of several builds, so protection is computed on version ids, not stem names.
    Retained = the `keep` newest builds by (UTC time, run id), the build holding the pinned
    digest, and this run's candidate. Never deleted: any retained version, untagged versions, and
    versions that also carry a tag this tooling did not create.
    """
    if not pinned_digest:
        # Nothing is pinned on main yet (first publish, or a deliberate rollback), so there is no
        # validated build to protect: delete nothing rather than guess.
        return []
    if not _DIGEST.match(pinned_digest):
        raise RetentionError(f"pinned digest is malformed: {pinned_digest!r}")
    if not any(v["name"] == pinned_digest for v in versions):
        raise RetentionError(f"pinned digest {pinned_digest} is not in the package")

    ids_by_stem: dict[str, set[int]] = {}
    keys: dict[str, BuildKey] = {}
    foreign: set[int] = set()
    for version in versions:
        for tag in _tags(version):
            parsed = parse_tag(tag)
            if parsed is None:
                foreign.add(version["id"])
                continue
            stem, key = parsed
            ids_by_stem.setdefault(stem, set()).add(version["id"])
            keys[stem] = key

    pinned_ids = {v["id"] for v in versions if pinned_digest and v["name"] == pinned_digest}
    protected = {candidate_stem} if candidate_stem else set()
    protected |= {stem for stem, ids in ids_by_stem.items() if ids & pinned_ids}
    retained = set(sorted(keys, key=lambda stem: keys[stem], reverse=True)[:keep]) | protected

    keep_ids = set(pinned_ids)
    obsolete_ids: set[int] = set()
    for stem, ids in ids_by_stem.items():
        (keep_ids if stem in retained else obsolete_ids).update(ids)
    return sorted(obsolete_ids - keep_ids - foreign)


def parse_branch(branch: str) -> tuple[str, int, int] | None:
    """(UTC build time, numeric run id, attempt) of a generated proposal branch, else None."""
    found = _BRANCH.match(branch)
    if not found:
        return None
    return found.group(2), int(found.group(3)), int(found.group(5))


def is_owned_proposal(pr: Mapping[str, Any], branch: str | None = None) -> bool:
    """A pull request this tooling generated: same repository, into `main`, authored by the
    Actions bot, from a branch in the exact generated format (and exactly `branch`, if given)."""
    name = pr.get("headRefName", "")
    return (
        parse_branch(name) is not None
        and (branch is None or name == branch)
        and not pr.get("isCrossRepository", True)
        and pr.get("baseRefName") == "main"
        and (pr.get("author") or {}).get("login") in BOT_LOGINS
    )


def select_superseded(prs: Sequence[Mapping[str, Any]], new_branch: str) -> list[int]:
    """Owned proposals that are strictly older than the replacement, in (UTC, run id, attempt)
    order. A retried job for an older build must not retire a newer build's proposal, nor a
    higher attempt of its own."""
    replacement = parse_branch(new_branch)
    if replacement is None:
        raise ValueError(f"not a generated proposal branch: {new_branch!r}")
    numbers = []
    for pr in prs:
        key = parse_branch(pr.get("headRefName", ""))
        if key is not None and key < replacement and is_owned_proposal(pr):
            numbers.append(int(pr["number"]))
    return sorted(numbers)


def read_string(path: Path, name: str) -> str:
    """The string literal assigned to module-level `name` in `path`, without importing it."""
    for node in ast.parse(path.read_text()).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            value = ast.literal_eval(node.value)
            if not isinstance(value, str):
                raise TypeError(f"{name} in {path} is not a string")
            return value
    raise KeyError(f"{name} is not assigned in {path}")


def read_pin(path: Path) -> tuple[str, str]:
    return read_string(path, "DEPS_IMAGE"), read_string(path, "DEPS_IMAGE_BASE")


def digest_of(image: str) -> str:
    """The digest in a pinned reference; "" for the empty (nothing published yet) pin."""
    if not image:
        return ""
    _, separator, digest = image.partition("@")
    if not separator or not _DIGEST.match(digest):
        raise RetentionError(f"pinned image is not a digest reference: {image!r}")
    return digest


def _pieces(value: str, width: int = 80) -> list[str]:
    """Split a reference at its `@` (then by width) so each piece fits a source line."""
    head, at, tail = value.partition("@")
    parts = [head, at + tail] if at else [value]
    return [part[i : i + width] for part in parts for i in range(0, len(part), width)]


def _assignment(name: str, value: str) -> str:
    """Spell `name = value` the way `ruff format` leaves it while keeping every line within the
    limit `ruff check` enforces (E501 exempts only single-token lines)."""
    inline = f'{name} = "{value}"'
    if len(inline) <= LINE_LIMIT:
        return inline
    wrapped = f'    "{value}"'
    if len(wrapped) <= LINE_LIMIT:
        return f"{name} = (\n{wrapped}\n)"
    return f"{name} = (\n" + "\n".join(f'    "{piece}"' for piece in _pieces(value)) + "\n)"


def render_pin(image: str, base: str) -> str:
    """The whole `pin.py`, formatted the way `ruff format` leaves it."""
    return (
        '"""Published dependency-image seed (see `recipe.admitted_seed`).\n\n'
        "Rewritten as a whole by `scripts/deps_image.py` from `.github/workflows/deps-image.yml`; "
        "do not\nedit by hand. An empty image builds the environment in the engine.\n"
        '"""\n\n'
        f"{_assignment('DEPS_IMAGE', image)}\n{_assignment('DEPS_IMAGE_BASE', base)}\n"
    )


def run_gh(args: Sequence[str]) -> str:
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True)


def proposal_body(ref: str, stem: str, run_url: str) -> str:
    return (
        f"Pins `{ref}` (build `{stem}`).\n\n"
        f"- amd64 pull proof: the `verify` job of {run_url}\n"
        "- arm64 pull proof: this pull request's `dagger-check` on the configured host\n\n"
        "The image is a cache seed: apt, `uv sync --locked`, the model prefetch and an interpreter "
        "check still run on top of it, and it is ignored if the checkout's base image moves."
    )


def propose(
    repo: Path,
    *,
    ref: str,
    base: str,
    stem: str,
    attempt: int,
    remote: str,
    body: str,
    gh: Gh = run_gh,
) -> str:
    """Open a pull request that pins `ref`, built from `base`; return what happened.

    The branch name carries the attempt, so a retry after a push that succeeded never collides
    with the earlier branch, and nothing is force-pushed or deleted. A create that errors after
    the server accepted it is detected by looking the head up before giving up. Older generated
    proposals are retired only after the replacement exists.
    """
    pin_file = repo / PIN_PATH
    if read_pin(pin_file) == (ref, base):
        return "unchanged"

    branch = proposal_branch(stem, attempt)
    _git(repo, "switch", "-c", branch)
    pin_file.write_text(render_pin(ref, base))
    _git(repo, "add", str(PIN_PATH))
    _git(
        repo,
        "-c",
        "user.name=github-actions[bot]",
        "-c",
        "user.email=41898282+github-actions[bot]@users.noreply.github.com",
        "commit",
        "-m",
        f"ci: pin dependency image {stem}",
    )
    _git(repo, "push", remote, f"HEAD:refs/heads/{branch}")

    try:
        gh(
            [
                "pr",
                "create",
                "--base",
                "main",
                "--head",
                branch,
                "--title",
                f"ci: pin dependency image {stem}",
                "--body",
                body,
            ]
        )
    except subprocess.CalledProcessError:
        # Accept a failed create only if the server really holds *our* proposal for this exact
        # head: a fork or human pull request with the same branch name is not a replacement.
        found = json.loads(
            gh(["pr", "list", "--head", branch, "--state", "open", "--json", _PR_FIELDS])
        )
        if not any(is_owned_proposal(pr, branch) for pr in found):
            raise

    # Pull requests opened with GITHUB_TOKEN do not start pull_request workflows; a dispatch does.
    gh(["workflow", "run", "ci.yml", "--ref", branch])

    listing = gh(["pr", "list", "--state", "open", "--limit", "200", "--json", _PR_FIELDS])
    for number in select_superseded(json.loads(listing), branch):
        gh(["pr", "close", str(number), "--comment", f"Superseded by the pin for {stem}."])
    return "proposed"


def retain(
    owner: str, pin_file: Path, candidate: str, *, dry_run: bool, gh: Gh = run_gh
) -> list[int]:
    image, _ = read_pin(pin_file)
    pinned = digest_of(image)
    endpoint = f"/users/{owner.lower()}/packages/container/{PACKAGE}/versions"
    lines = gh(["api", "--paginate", "--jq", ".[]", f"{endpoint}?per_page=100"]).splitlines()
    versions = [json.loads(line) for line in lines if line.strip()]
    doomed = plan_retention(versions, pinned, candidate)
    for version_id in doomed:
        print(f"deleting version {version_id}")
        if not dry_run:
            gh(["api", "-X", "DELETE", f"{endpoint}/{version_id}"])
    return doomed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    base = commands.add_parser("base", help="print _BASE_IMAGE of the checked-out module")
    base.add_argument("--module", type=Path, default=MODULE_PATH)

    write = commands.add_parser("write-pin", help="rewrite pin.py")
    write.add_argument("--image", required=True)
    write.add_argument("--base", required=True)
    write.add_argument("--pin", type=Path, default=PIN_PATH)

    keep = commands.add_parser("retain", help="delete obsolete package versions")
    keep.add_argument("--owner", required=True)
    keep.add_argument("--candidate", required=True)
    keep.add_argument("--pin", type=Path, default=PIN_PATH)
    keep.add_argument("--dry-run", action="store_true")

    open_pr = commands.add_parser("propose", help="open the pin pull request")
    open_pr.add_argument("--ref", required=True)
    open_pr.add_argument("--base", required=True)
    open_pr.add_argument("--stem", required=True)
    open_pr.add_argument("--attempt", type=int, required=True)
    open_pr.add_argument("--remote", help="push URL; defaults to the token-authenticated origin")

    args = parser.parse_args(argv)
    if args.command == "base":
        print(read_string(args.module, "_BASE_IMAGE"))
    elif args.command == "write-pin":
        args.pin.write_text(render_pin(args.image, args.base))
    elif args.command == "retain":
        retain(args.owner, args.pin, args.candidate, dry_run=args.dry_run)
    else:
        repository = os.environ["GITHUB_REPOSITORY"]
        server, run = os.environ["GITHUB_SERVER_URL"], os.environ["GITHUB_RUN_ID"]
        run_url = f"{server}/{repository}/actions/runs/{run}"
        remote = args.remote or (
            f"https://x-access-token:{os.environ['GH_TOKEN']}@github.com/{repository}.git"
        )
        outcome = propose(
            Path.cwd(),
            ref=args.ref,
            base=args.base,
            stem=args.stem,
            attempt=args.attempt,
            remote=remote,
            body=proposal_body(args.ref, args.stem, run_url),
        )
        print(outcome)
    return 0


if __name__ == "__main__":
    sys.exit(main())
