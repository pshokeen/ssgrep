"""Behavior of the dependency-image publisher policy (`scripts/deps_image.py`).

Retention and supersession are exercised on package/PR listings shaped like the real API output;
the pin pull request lifecycle runs real `git` against a local bare remote with a fake `gh`.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "deps_image.py"
_spec = importlib.util.spec_from_file_location("deps_image_script", _SCRIPT)
assert _spec is not None and _spec.loader is not None
deps_image = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = deps_image
_spec.loader.exec_module(deps_image)

BASE = "python:3.13-slim@sha256:" + "a" * 64
OWNER_IMAGE = "ghcr.io/example/ssgrep-ci-deps"


def digest(n: int) -> str:
    return f"sha256:{n:064x}"


def stem(utc: str, run: int, lock: str = "0123abcd") -> str:
    return f"{utc}-r{run}-{lock}"


def version(vid: int, *tags: str) -> dict[str, Any]:
    return {"id": vid, "name": digest(vid), "metadata": {"container": {"tags": list(tags)}}}


def build(first_id: int, name: str) -> list[dict[str, Any]]:
    """One build: the index version plus its two per-architecture versions."""
    return [
        version(first_id, name),
        version(first_id + 1, f"{name}-arm64"),
        version(first_id + 2, f"{name}-amd64"),
    ]


def ids_of(*builds: list[dict[str, Any]]) -> set[int]:
    return {v["id"] for b in builds for v in b}


# --- retention -------------------------------------------------------------------------------


def test_a_version_shared_with_a_protected_build_is_never_deleted() -> None:
    obsolete, pinned, newest = (
        stem("20261010000000", 1),
        stem("20261020000000", 2),
        stem("20261029000000", 3),
    )
    old_build, pin_build, new_build = build(10, obsolete), build(20, pinned), build(30, newest)
    # One digest carries tags of an obsolete build, the pinned build and the retained newest one.
    shared = version(7777, f"{obsolete}-arm64", f"{pinned}-arm64", f"{newest}-arm64")
    versions = [*old_build, *pin_build, *new_build, shared]

    doomed = deps_image.plan_retention(versions, digest(20), newest, keep=1)

    assert 7777 not in doomed
    assert set(doomed) == ids_of(old_build)  # only the shared version survives, with its builds
    assert set(doomed).isdisjoint(ids_of(pin_build, new_build))


def test_newest_build_wins_over_hash_order_on_the_same_day() -> None:
    """Ordering alone must retain the latest build: no candidate protection is in play here."""
    hashes = [f"{n:08x}" for n in range(1, 9)]
    older = [build(100 + 10 * i, stem("20261029010000", 50 + i, h)) for i, h in enumerate(hashes)]
    newest = build(500, stem("20261029120000", 99, "00000000"))  # newest by clock, lowest by hash
    pinned = build(900, stem("20261001000000", 1))
    versions = [v for b in (*older, newest, pinned) for v in b]

    doomed = deps_image.plan_retention(versions, digest(900), "", keep=8)

    assert set(doomed).isdisjoint(ids_of(newest))
    assert set(doomed).isdisjoint(ids_of(pinned))
    assert set(doomed) == ids_of(older[0])  # the single oldest of the nine same-day builds


def test_same_second_builds_order_by_numeric_run_id() -> None:
    """`r10` is newer than `r9`; a lexical comparison of the run id would keep the wrong one."""
    same_second = "20261029120000"
    older, newer = build(10, stem(same_second, 9)), build(20, stem(same_second, 10))
    pinned = build(900, stem("20261001000000", 1))
    versions = [v for b in (older, newer, pinned) for v in b]

    doomed = deps_image.plan_retention(versions, digest(900), "", keep=1)

    assert set(doomed).isdisjoint(ids_of(newer))
    assert set(doomed) == ids_of(older)


def test_the_workflows_own_candidate_is_protected_even_when_it_is_not_among_the_newest() -> None:
    candidate_stem = stem("20260101000000", 1)
    candidate = build(10, candidate_stem)
    newer = [build(100 + 10 * i, stem(f"2026020{i}000000", 2 + i)) for i in range(1, 4)]
    pinned = build(900, stem("20260301000000", 9))
    versions = [v for b in (candidate, *newer, pinned) for v in b]

    doomed = deps_image.plan_retention(versions, digest(900), candidate_stem, keep=3)

    assert set(doomed).isdisjoint(ids_of(candidate))
    assert set(doomed) == ids_of(newer[0])


def test_shared_ids_are_deleted_once() -> None:
    old, kept = stem("20260101000000", 1), stem("20260301000000", 2)
    versions = [version(1, old), version(2, f"{old}-arm64"), *build(10, kept)]
    versions.append(version(3, f"{old}-amd64", f"{old}-arm64"))

    doomed = deps_image.plan_retention(versions, digest(10), kept, keep=1)

    assert doomed == sorted(set(doomed))
    assert set(doomed) == {1, 2, 3}


def test_untagged_and_foreign_tagged_versions_are_left_alone() -> None:
    old, kept = stem("20260101000000", 1), stem("20260301000000", 2)
    versions = [
        *build(10, kept),
        version(50),  # untagged: possibly a child of a retained index
        version(51, f"{old}-arm64", "latest"),  # carries a tag this tooling did not create
        version(52, f"{old}-amd64"),
    ]

    assert deps_image.plan_retention(versions, digest(10), kept, keep=1) == [52]


def test_nothing_is_deleted_while_main_has_no_pin() -> None:
    versions = [v for i in range(20) for v in build(10 * i, stem(f"202601{i + 1:02d}000000", i))]
    assert deps_image.plan_retention(versions, "", stem("20260120000000", 19)) == []


@pytest.mark.parametrize("pinned", [digest(404), "sha256:short", "latest"])
def test_a_pin_that_is_malformed_or_missing_from_the_package_refuses(pinned: str) -> None:
    versions = build(10, stem("20260101000000", 1))
    with pytest.raises(deps_image.RetentionError):
        deps_image.plan_retention(versions, pinned, "")


def test_pin_reference_without_a_digest_refuses_instead_of_reading_as_empty() -> None:
    with pytest.raises(deps_image.RetentionError):
        deps_image.digest_of(f"{OWNER_IMAGE}:latest")
    assert deps_image.digest_of("") == ""
    assert deps_image.digest_of(f"{OWNER_IMAGE}@{digest(7)}") == digest(7)


class RecordingGh:
    """Fake `gh`: serves a package version listing and records every call."""

    def __init__(self, versions: list[dict[str, Any]]) -> None:
        self.versions = versions
        self.calls: list[list[str]] = []

    def __call__(self, args: Any) -> str:
        args = list(args)
        self.calls.append(args)
        if args[:2] == ["api", "--paginate"]:
            return "\n".join(json.dumps(v) for v in self.versions) + "\n"
        return ""

    @property
    def deletes(self) -> list[str]:
        return [c[-1] for c in self.calls if "DELETE" in c]


def _write_pin(tmp_path: Path, image: str) -> Path:
    pin = tmp_path / "pin.py"
    pin.write_text(deps_image.render_pin(image, BASE))
    return pin


def test_retain_deletes_only_what_the_plan_names(tmp_path: Path) -> None:
    old, pinned = stem("20260101000000", 1), stem("20260201000000", 2)
    gh = RecordingGh([*build(10, old), *build(20, pinned)])
    pin = _write_pin(tmp_path, f"{OWNER_IMAGE}@{digest(20)}")

    deps_image.retain("Example", pin, pinned, dry_run=False, gh=gh)

    # keep=8 retains both builds here, so nothing is obsolete.
    assert gh.deletes == []


def test_retain_refuses_before_any_delete_when_the_pin_is_not_in_the_package(
    tmp_path: Path,
) -> None:
    gh = RecordingGh(build(10, stem("20260101000000", 1)))
    pin = _write_pin(tmp_path, f"{OWNER_IMAGE}@{digest(404)}")

    with pytest.raises(deps_image.RetentionError):
        deps_image.retain("Example", pin, "", dry_run=False, gh=gh)

    assert gh.deletes == []


def test_retain_with_an_empty_pin_makes_no_delete_calls(tmp_path: Path) -> None:
    gh = RecordingGh(
        [v for i in range(12) for v in build(10 * i, stem(f"202601{i + 1:02d}000000", i))]
    )

    deps_image.retain("Example", _write_pin(tmp_path, ""), "", dry_run=False, gh=gh)

    assert gh.deletes == []


def test_retain_deletes_obsolete_versions_through_the_api(tmp_path: Path) -> None:
    builds = [build(10 * i, stem(f"202601{i:02d}000000", i)) for i in range(1, 11)]
    gh = RecordingGh([v for b in builds for v in b])
    pin = _write_pin(tmp_path, f"{OWNER_IMAGE}@{digest(100)}")  # the newest build's index

    deps_image.retain(
        "Example", pin, builds[-1][0]["metadata"]["container"]["tags"][0], dry_run=False, gh=gh
    )

    prefix = "/users/example/packages/container/ssgrep-ci-deps/versions/"
    assert gh.deletes == [prefix + str(i) for i in sorted(ids_of(builds[0], builds[1]))]


# --- supersession ---------------------------------------------------------------------------


def pr(
    number: int,
    branch: str,
    *,
    login: str = "app/github-actions",
    cross: bool = False,
    base: str = "main",
):
    return {
        "number": number,
        "headRefName": branch,
        "author": {"login": login, "is_bot": login != "someone"},
        "isCrossRepository": cross,
        "baseRefName": base,
    }


def test_only_older_generated_same_repo_bot_proposals_are_retired() -> None:
    new = deps_image.proposal_branch(stem("20261029120000", 9), 1)
    older = deps_image.proposal_branch(stem("20261022120000", 8), 1)
    prs = [
        pr(1, older),  # an earlier generated proposal: retire
        pr(2, new),  # the replacement itself
        pr(3, "ci/deps-image-fix-auth", login="someone"),  # human branch with the same prefix
        pr(4, "ci/deps-image-fix-auth"),  # bot-authored but not the generated format
        pr(5, older, cross=True),  # fork with a spoofed branch name
        pr(6, older, login="someone"),  # generated-looking branch authored by a human
        pr(7, older, base="release"),  # wrong base
        pr(8, deps_image.proposal_branch(stem("20261015120000", 7), 2), login="github-actions"),
    ]

    assert deps_image.select_superseded(prs, new) == [1, 8]


# --- pin file ------------------------------------------------------------------------------


@pytest.mark.parametrize("image", ["", f"{OWNER_IMAGE}@{digest(1)}"])
def test_rendered_pin_round_trips_and_is_valid_python(tmp_path: Path, image: str) -> None:
    pin = _write_pin(tmp_path, image)
    assert deps_image.read_pin(pin) == (image, BASE)
    assert all(len(line) <= 100 for line in pin.read_text().splitlines())


@pytest.mark.parametrize(
    ("image", "base"),
    [
        ("", ""),
        ("ghcr.io/e/d@" + digest(1), "python:3.13-slim"),
        ("ghcr.io/pshokeen/ssgrep-ci-deps@" + digest(1), BASE),
        ("x" * 120, "y" * 70),
    ],
)
def test_rendered_pin_is_what_ruff_format_leaves_alone(
    tmp_path: Path, image: str, base: str
) -> None:
    """The pin pull request must not fail `module-quality` on formatting of its own file."""
    pin = tmp_path / "pin.py"
    pin.write_text(deps_image.render_pin(image, base))
    ruff = str(Path(sys.executable).parent / "ruff")
    common = ["--isolated", "--line-length", "100"]
    for command in (
        [ruff, "format", "--check", *common, str(pin)],
        [ruff, "check", *common, "--select", "E,F,I,UP,B", str(pin)],
    ):
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr


def test_base_image_is_read_from_a_parenthesised_literal(tmp_path: Path) -> None:
    module = tmp_path / "__init__.py"
    module.write_text(f'_UV = "1"\n_BASE_IMAGE = (\n    "{BASE}"\n)\n')
    assert deps_image.read_string(module, "_BASE_IMAGE") == BASE
    with pytest.raises(KeyError):
        deps_image.read_string(module, "_MISSING")


# --- pin pull request lifecycle ----------------------------------------------------------------


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


class FakeGh:
    """Fake `gh` for the proposal flow: PR creation can be made to fail, listings are scripted."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.fail_create = False
        self.create_lands_anyway = False
        self.open_prs: list[dict[str, Any]] = []

    def __call__(self, args: Any) -> str:
        args = list(args)
        self.calls.append(args)
        if args[:2] == ["pr", "create"]:
            branch = args[args.index("--head") + 1]
            if self.fail_create:
                if self.create_lands_anyway:
                    self.open_prs.append(pr(100, branch))
                raise subprocess.CalledProcessError(1, "gh pr create")
            self.open_prs.append(pr(100 + len(self.open_prs), branch))
        if args[:2] == ["pr", "list"] and "--head" in args:
            head = args[args.index("--head") + 1]
            return json.dumps(
                [{"number": p["number"]} for p in self.open_prs if p["headRefName"] == head]
            )
        if args[:2] == ["pr", "list"]:
            return json.dumps(self.open_prs)
        return ""

    def verbs(self, *prefix: str) -> list[list[str]]:
        return [c for c in self.calls if c[: len(prefix)] == list(prefix)]


@pytest.fixture
def remote(tmp_path: Path) -> Path:
    bare = tmp_path / "origin.git"
    seed = tmp_path / "seed"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(bare)], check=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(seed)], check=True)
    pin = seed / deps_image.PIN_PATH
    pin.parent.mkdir(parents=True)
    pin.write_text(deps_image.render_pin("", ""))
    _git(seed, "add", ".")
    _git(seed, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "init")
    _git(seed, "push", "-q", str(bare), "main")
    return bare


def _fresh_checkout(tmp_path: Path, remote: Path, name: str) -> Path:
    """A new job starts from a clean checkout of main."""
    checkout = tmp_path / name
    subprocess.run(["git", "clone", "-q", str(remote), str(checkout)], check=True)
    return checkout


def _propose(checkout: Path, remote: Path, gh: FakeGh, attempt: int, run: int = 9) -> str:
    return deps_image.propose(
        checkout,
        ref=f"{OWNER_IMAGE}@{digest(1)}",
        base=BASE,
        stem=stem("20261029120000", run),
        attempt=attempt,
        remote=str(remote),
        body="body",
        gh=gh,
    )


def _branches(remote: Path) -> set[str]:
    return set(_git(remote, "for-each-ref", "--format=%(refname:short)", "refs/heads").split())


def test_failed_pr_creation_after_a_successful_push_recovers_on_retry(
    tmp_path: Path, remote: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Literal names: computing them with the code under test would not catch a regression to one
    # branch per build, and identical commits (same second) would hide the non-fast-forward.
    first_branch = "ci/deps-image-20261029120000-r9-0123abcd-a1"
    second_branch = "ci/deps-image-20261029120000-r9-0123abcd-a2"
    gh = FakeGh()
    gh.fail_create = True  # e.g. the repository still forbids Actions from creating pull requests
    first = _fresh_checkout(tmp_path, remote, "attempt1")
    monkeypatch.setenv("GIT_COMMITTER_DATE", "2026-10-29T12:00:00+0000")
    monkeypatch.setenv("GIT_AUTHOR_DATE", "2026-10-29T12:00:00+0000")
    with pytest.raises(subprocess.CalledProcessError):
        _propose(first, remote, gh, attempt=1)
    assert first_branch in _branches(remote)
    assert gh.verbs("workflow", "run") == []  # no checks dispatched for a PR that does not exist

    gh.fail_create = False  # the owner enabled the setting; the job is re-run later
    monkeypatch.setenv("GIT_COMMITTER_DATE", "2026-10-29T12:30:00+0000")
    monkeypatch.setenv("GIT_AUTHOR_DATE", "2026-10-29T12:30:00+0000")
    second = _fresh_checkout(tmp_path, remote, "attempt2")
    assert _propose(second, remote, gh, attempt=2) == "proposed"

    assert {first_branch, second_branch} <= _branches(remote)  # nothing force-pushed or deleted
    assert [p["headRefName"] for p in gh.open_prs] == [second_branch]
    assert gh.verbs("workflow", "run") == [["workflow", "run", "ci.yml", "--ref", second_branch]]
    committed = tmp_path / "committed_pin.py"
    committed.write_text(_git(remote, "show", f"{second_branch}:{deps_image.PIN_PATH}") + "\n")
    assert deps_image.read_pin(committed) == (f"{OWNER_IMAGE}@{digest(1)}", BASE)


def test_a_create_that_errors_after_the_server_accepted_it_is_treated_as_created(
    tmp_path: Path, remote: Path
) -> None:
    gh = FakeGh()
    gh.fail_create = gh.create_lands_anyway = True

    outcome = _propose(_fresh_checkout(tmp_path, remote, "job"), remote, gh, attempt=1)

    branch = deps_image.proposal_branch(stem("20261029120000", 9), 1)
    assert outcome == "proposed"
    assert branch in _branches(remote)  # the possibly live branch is never deleted
    assert gh.verbs("workflow", "run") == [["workflow", "run", "ci.yml", "--ref", branch]]


def test_replacement_is_created_before_older_owned_proposals_are_closed(
    tmp_path: Path, remote: Path
) -> None:
    gh = FakeGh()
    older = deps_image.proposal_branch(stem("20261022120000", 8), 1)
    gh.open_prs = [pr(5, older), pr(6, "ci/deps-image-fix-auth", login="someone")]

    _propose(_fresh_checkout(tmp_path, remote, "job"), remote, gh, attempt=1)

    order = [c[:2] for c in gh.calls]
    assert order.index(["pr", "create"]) < order.index(["pr", "close"])
    assert [c[2] for c in gh.verbs("pr", "close")] == ["5"]


def test_an_unchanged_pin_pushes_and_opens_nothing(tmp_path: Path, remote: Path) -> None:
    checkout = _fresh_checkout(tmp_path, remote, "job")
    (checkout / deps_image.PIN_PATH).write_text(
        deps_image.render_pin(f"{OWNER_IMAGE}@{digest(1)}", BASE)
    )
    gh = FakeGh()

    assert _propose(checkout, remote, gh, attempt=1) == "unchanged"
    assert gh.calls == []
    assert _branches(remote) == {"main"}


# --- the real command line, with a fake `gh` executable on PATH --------------------------------

_FAKE_GH = """#!{python}
import json, os, sys

args = sys.argv[1:]
path = os.environ["FAKE_GH_STATE"]
state = json.load(open(path))
state["calls"].append(args)
out = ""
if args[:2] == ["pr", "create"]:
    if state["fail_create"]:
        json.dump(state, open(path, "w"))
        sys.exit(1)
    branch = args[args.index("--head") + 1]
    state["prs"].append({{
        "number": 100 + len(state["prs"]), "headRefName": branch,
        "author": {{"login": "app/github-actions", "is_bot": True}},
        "isCrossRepository": False, "baseRefName": "main",
    }})
elif args[:2] == ["pr", "list"] and "--head" in args:
    head = args[args.index("--head") + 1]
    out = json.dumps([{{"number": p["number"]}} for p in state["prs"] if p["headRefName"] == head])
elif args[:2] == ["pr", "list"]:
    out = json.dumps(state["prs"])
elif args[:2] == ["api", "--paginate"]:
    out = "\\n".join(json.dumps(v) for v in state["versions"])
json.dump(state, open(path, "w"))
print(out)
"""


class FakeGhCli:
    """A scripted `gh` binary whose state and call log live in a JSON file."""

    def __init__(self, tmp_path: Path) -> None:
        self.bin = tmp_path / "fakebin"
        self.bin.mkdir()
        self.state_path = tmp_path / "gh_state.json"
        executable = self.bin / "gh"
        executable.write_text(_FAKE_GH.format(python=sys.executable))
        executable.chmod(0o755)
        self.set(fail_create=False, prs=[], versions=[], calls=[])

    def set(self, **changes: Any) -> None:
        state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.state_path.write_text(json.dumps({**state, **changes}))

    @property
    def state(self) -> dict[str, Any]:
        return json.loads(self.state_path.read_text())

    def env(self, **extra: str) -> dict[str, str]:
        base = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        return {
            **base,
            "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            "FAKE_GH_STATE": str(self.state_path),
            "GITHUB_REPOSITORY": "example/ssgrep",
            "GITHUB_SERVER_URL": "https://github.example",
            "GITHUB_RUN_ID": "9",
            **extra,
        }


def _cli(args: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args], cwd=cwd, env=env, capture_output=True, text=True
    )


def _propose_cli(checkout: Path, remote: Path, attempt: int, env: dict[str, str]):
    return _cli(
        [
            "propose",
            "--ref",
            f"{OWNER_IMAGE}@{digest(1)}",
            "--base",
            BASE,
            "--stem",
            stem("20261029120000", 9),
            "--attempt",
            str(attempt),
            "--remote",
            str(remote),
        ],
        checkout,
        env,
    )


def test_cli_recovers_on_rerun_after_a_failed_pr_creation(tmp_path: Path, remote: Path) -> None:
    gh = FakeGhCli(tmp_path)
    gh.set(fail_create=True)
    dates = {"GIT_AUTHOR_DATE": "2026-10-29T12:00:00+0000"}
    dates["GIT_COMMITTER_DATE"] = dates["GIT_AUTHOR_DATE"]

    first = _propose_cli(_fresh_checkout(tmp_path, remote, "a1"), remote, 1, gh.env(**dates))
    assert first.returncode != 0
    assert "ci/deps-image-20261029120000-r9-0123abcd-a1" in _branches(remote)
    assert not [c for c in gh.state["calls"] if c[:2] == ["workflow", "run"]]

    gh.set(fail_create=False)
    later = {"GIT_AUTHOR_DATE": "2026-10-29T12:30:00+0000"}
    later["GIT_COMMITTER_DATE"] = later["GIT_AUTHOR_DATE"]
    second = _propose_cli(_fresh_checkout(tmp_path, remote, "a2"), remote, 2, gh.env(**later))

    branch = "ci/deps-image-20261029120000-r9-0123abcd-a2"
    assert second.returncode == 0, second.stderr
    assert second.stdout.splitlines()[-1] == "proposed"
    assert [p["headRefName"] for p in gh.state["prs"]] == [branch]
    assert ["workflow", "run", "ci.yml", "--ref", branch] in gh.state["calls"]
    committed = tmp_path / "committed_pin.py"
    committed.write_text(_git(remote, "show", f"{branch}:{deps_image.PIN_PATH}") + "\n")
    assert deps_image.read_pin(committed) == (f"{OWNER_IMAGE}@{digest(1)}", BASE)


def _deletes(gh: FakeGhCli) -> list[str]:
    return [c[-1] for c in gh.state["calls"] if "DELETE" in c]


def test_cli_retain_deletes_through_the_api_and_refuses_on_a_bad_pin(tmp_path: Path) -> None:
    builds = [build(10 * i, stem(f"202601{i:02d}000000", i)) for i in range(1, 11)]
    gh = FakeGhCli(tmp_path)
    gh.set(versions=[v for b in builds for v in b])
    candidate = builds[-1][0]["metadata"]["container"]["tags"][0]
    pin = _write_pin(tmp_path, f"{OWNER_IMAGE}@{digest(100)}")
    args = ["retain", "--owner", "Example", "--candidate", candidate, "--pin", str(pin)]

    ok = _cli(args, tmp_path, gh.env())

    prefix = "/users/example/packages/container/ssgrep-ci-deps/versions/"
    assert ok.returncode == 0, ok.stderr
    assert _deletes(gh) == [prefix + str(i) for i in sorted(ids_of(builds[0], builds[1]))]

    gh.set(calls=[])
    pin.write_text(deps_image.render_pin(f"{OWNER_IMAGE}@{digest(404)}", BASE))
    refused = _cli(args, tmp_path, gh.env())
    assert refused.returncode != 0
    assert _deletes(gh) == []

    pin.write_text(deps_image.render_pin("", ""))
    nothing = _cli(args, tmp_path, gh.env())
    assert nothing.returncode == 0
    assert _deletes(gh) == []


def test_cli_base_and_write_pin_round_trip(tmp_path: Path) -> None:
    module = tmp_path / "__init__.py"
    module.write_text(f'_BASE_IMAGE = (\n    "{BASE}"\n)\n')
    pin = tmp_path / "pin.py"
    env = {k: v for k, v in os.environ.items()}

    base = _cli(["base", "--module", str(module)], tmp_path, env)
    assert base.stdout.strip() == BASE

    ref = f"{OWNER_IMAGE}@{digest(5)}"
    written = _cli(
        ["write-pin", "--image", ref, "--base", base.stdout.strip(), "--pin", str(pin)],
        tmp_path,
        env,
    )
    assert written.returncode == 0, written.stderr
    assert deps_image.read_pin(pin) == (ref, BASE)
