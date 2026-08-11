"""One canonicalization, applied to BOTH sides of scope matching.

The failures these pin are all silent: the scope typed on the command line
went through Path.resolve() while the cwd read out of a transcript went
through nothing, so a buyer could point ssgrep at their own project and be
told there was nothing there -- or worse, index an empty generation over
the physically identical .ssgrep belonging to another spelling of the same
directory.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from ssgrep import discovery, paths
from ssgrep.cli.commands import resolve_project_dir
from ssgrep.discovery import discover_sessions


def observed_case_insensitive(directory: Path) -> bool:
    """Ground truth for `directory`, obtained without asking ssgrep.

    Tests must not use the probe under test to decide whether to run the
    tests that exercise the probe.
    """
    marker = directory / "SsgrepCaseGroundTruth"
    marker.write_text("x")
    try:
        return (directory / "ssgrepcasegroundtruth").exists()
    finally:
        marker.unlink()


def write_transcript(claude_dir: Path, name: str, cwd: str) -> Path:
    """Write a one-record transcript under a fake ~/.claude/projects."""
    session_dir = claude_dir / name
    session_dir.mkdir(parents=True, exist_ok=True)
    transcript = session_dir / f"{name}-session.jsonl"
    transcript.write_text(json.dumps({"cwd": cwd, "type": "user"}) + "\n")
    return transcript


# ---------------------------------------------------------------------------
# The runtime probe
# ---------------------------------------------------------------------------


def test_case_probe_reports_what_the_filesystem_actually_does(tmp_path):
    """The probe must observe, not infer from sys.platform.

    macOS ships case-sensitive APFS as a supported option and Linux mounts
    case-insensitive volumes, so a platform table is wrong on real buyer
    machines. Compared against ground truth measured independently, in the
    same temp filesystem the probe uses.
    """
    paths.reset_case_probe()
    assert paths.filesystem_is_case_insensitive() is observed_case_insensitive(tmp_path)


def test_case_probe_ignores_sys_platform_when_it_disagrees(tmp_path, monkeypatch):
    """The observe-vs-infer distinction, made falsifiable.

    The test above cannot see it: on a default macOS or Linux machine the
    platform table happens to give the same answer as the probe, so replacing
    the probe with ``sys.platform in ("darwin", "win32")`` -- the exact
    implementation the module docstring forbids -- passes it. Here sys.platform
    is forced to the value that would produce the WRONG answer for this
    machine, so only a real filesystem measurement can still be right.
    """
    truth = observed_case_insensitive(tmp_path)
    # Whichever platform string the table would map to the opposite answer.
    monkeypatch.setattr(sys, "platform", "linux" if truth else "darwin")

    paths.reset_case_probe()
    assert paths.filesystem_is_case_insensitive() is truth, (
        "filesystem_is_case_insensitive() followed sys.platform instead of "
        "measuring the filesystem"
    )
    assert paths._run_probe(str(tmp_path)) is truth, (
        "_run_probe must itself report ground truth for the directory it is "
        "handed -- it is the only thing standing between a buyer on "
        "case-sensitive APFS and silently merged projects"
    )


def test_fold_case_folds_exactly_when_the_filesystem_does(tmp_path):
    """fold_case must agree with the filesystem, in both directions."""
    paths.reset_case_probe()
    folded = paths.fold_case("/Users/Someone/Code/App")
    if observed_case_insensitive(tmp_path):
        assert folded == "/users/someone/code/app"
    else:
        assert folded == "/Users/Someone/Code/App"


# ---------------------------------------------------------------------------
# Quoted tilde
# ---------------------------------------------------------------------------


def test_quoted_tilde_project_dir_is_home_relative(tmp_path):
    """`--project-dir "~/code/x"` must mean the home directory.

    A quoted tilde is never expanded by the shell, so it arrived here as a
    literal "~" and Path.resolve() anchored it under the *current* working
    directory: `<cwd>/~/code/x`, which matches no transcript ever recorded.
    """
    fake_home = tmp_path / "home"
    (fake_home / "code" / "x").mkdir(parents=True)

    with patch.dict(os.environ, {"HOME": str(fake_home)}):
        resolved = resolve_project_dir("~/code/x")

    assert resolved == (fake_home / "code" / "x").resolve()
    assert "~" not in str(resolved)


def test_quoted_tilde_scope_discovers_its_sessions(tmp_path):
    """End to end: a quoted tilde must find the transcripts it names."""
    fake_home = tmp_path / "home"
    project = fake_home / "code" / "x"
    project.mkdir(parents=True)
    claude_dir = fake_home / ".claude" / "projects"
    transcript = write_transcript(claude_dir, "tilde-proj", str(project.resolve()))

    with patch.dict(os.environ, {"HOME": str(fake_home)}):
        discovery._cwd_index_cache.clear()
        found = discover_sessions(resolve_project_dir("~/code/x"), no_subagents=False)
        discovery._cwd_index_cache.clear()

    assert [s.path for s in found] == [transcript]


# ---------------------------------------------------------------------------
# Case collision on a case-insensitive filesystem
# ---------------------------------------------------------------------------


def test_differently_cased_scope_matches_recorded_cwd(tmp_path):
    """`.../Code/app` recorded, `.../code/app` typed -- one physical index.

    Both spellings name the same directory on a case-insensitive volume, so
    both share one .ssgrep. Matching only one of them meant indexing under
    the other spelling committed an empty generation over the buyer's real
    index.
    """
    if not observed_case_insensitive(tmp_path):
        pytest.skip("filesystem is case-sensitive; the two spellings are different dirs")

    fake_home = tmp_path / "home"
    project = fake_home / "Code" / "app"
    project.mkdir(parents=True)
    claude_dir = fake_home / ".claude" / "projects"
    transcript = write_transcript(claude_dir, "cased-proj", str(project.resolve()))

    typed = str(project.resolve()).replace("/Code/app", "/code/app")
    assert typed != str(project.resolve())

    with patch.dict(os.environ, {"HOME": str(fake_home)}):
        discovery._cwd_index_cache.clear()
        found = discover_sessions(resolve_project_dir(typed), no_subagents=False)
        discovery._cwd_index_cache.clear()

    assert [s.path for s in found] == [transcript]


def test_case_collision_spellings_share_one_index_location(tmp_path):
    """The two spellings really do collide on one .ssgrep.

    This is what makes the mismatch destructive rather than merely
    annoying, and it is why matching has to fold case exactly when the
    filesystem does.
    """
    if not observed_case_insensitive(tmp_path):
        pytest.skip("filesystem is case-sensitive; the two spellings are different dirs")

    project = tmp_path / "Code" / "app"
    project.mkdir(parents=True)
    typed = str(project).replace("/Code/app", "/code/app")

    (resolve_project_dir(str(project)) / ".ssgrep").mkdir(parents=True, exist_ok=True)
    (resolve_project_dir(str(project)) / ".ssgrep" / "witness").write_text("data")

    assert (resolve_project_dir(typed) / ".ssgrep" / "witness").read_text() == "data"


def test_case_differences_do_not_match_on_a_case_sensitive_filesystem(tmp_path):
    """The fold is conditional, not unconditional.

    On a case-sensitive volume `.../Code/app` and `.../code/app` are two
    different directories with two different indexes, and folding them
    together would merge one buyer's project into another's results.
    """
    if observed_case_insensitive(tmp_path):
        pytest.skip("filesystem is case-insensitive; the two spellings are one dir")

    assert not paths.is_at_or_beneath("/w/Code/app", "/w/code/app")


def test_matching_folds_case_if_and_only_if_the_probe_says_to(monkeypatch):
    """Both arms of the conditional fold, on whatever filesystem runs this.

    The skipping test above can only ever exercise one arm on a given
    machine, so an unconditional fold (or an unconditional refusal to
    fold) would ship untested. Here the probe result is the only thing
    that varies.
    """
    monkeypatch.setattr(paths, "_probe_result", False)
    assert not paths.is_at_or_beneath("/w/Code/app", "/w/code/app")

    monkeypatch.setattr(paths, "_probe_result", True)
    assert paths.is_at_or_beneath("/w/Code/app", "/w/code/app")


# ---------------------------------------------------------------------------
# resolve() on live paths only
# ---------------------------------------------------------------------------


def test_historical_cwd_is_not_resolved_through_a_later_symlink(tmp_path):
    """A recorded cwd is history and must be read literally.

    If the cwd side ever gains resolve(), a directory that was replaced by
    a symlink after the session was recorded starts matching whatever the
    symlink points at today -- silently pulling another project's
    transcripts into these results.
    """
    real = tmp_path / "other-project"
    real.mkdir()
    recorded = tmp_path / "gone-project"
    recorded.symlink_to(real, target_is_directory=True)

    assert Path(str(recorded)).resolve() == real.resolve()
    assert not paths.is_at_or_beneath(str(recorded), str(real))


def test_symlinked_project_dir_still_resolves_to_its_target(tmp_path):
    """resolve() stays on the live side -- removing it is not the fix.

    A user who types a symlinked --project-dir must land on the same index
    as one who types the real path.
    """
    real = tmp_path / "real-project"
    real.mkdir()
    link = tmp_path / "link-to-project"
    link.symlink_to(real, target_is_directory=True)

    assert resolve_project_dir(str(link)) == resolve_project_dir(str(real))


# ---------------------------------------------------------------------------
# Round-trip property
# ---------------------------------------------------------------------------


SPELLINGS = [
    "plain",
    "trailing slash",
    "trailing double slash",
    "dot segment",
    "dot-dot segment",
    "interior dot segments",
    "tilde",
    "tilde trailing slash",
    "tilde dot-dot",
    "mixed case",
    "upper case",
]


def spell(directory: Path, home: Path, kind: str) -> str:
    """Render `directory` in one of the equivalent user-typed spellings."""
    text = str(directory)
    if kind == "plain":
        return text
    if kind == "trailing slash":
        return text + "/"
    if kind == "trailing double slash":
        return text + "//"
    if kind == "dot segment":
        return text + "/."
    if kind == "dot-dot segment":
        return f"{directory.parent}/{directory.name}/../{directory.name}"
    if kind == "interior dot segments":
        return f"{directory.parent}/./{directory.name}"
    relative = directory.relative_to(home)
    if kind == "tilde":
        return f"~/{relative}"
    if kind == "tilde trailing slash":
        return f"~/{relative}/"
    if kind == "tilde dot-dot":
        return f"~/{relative}/../{directory.name}"
    if kind == "mixed case":
        return str(directory.parent) + "/" + directory.name.capitalize()
    if kind == "upper case":
        return str(directory.parent) + "/" + directory.name.upper()
    raise AssertionError(f"unhandled spelling {kind!r}")


@pytest.mark.parametrize("kind", SPELLINGS)
def test_every_spelling_of_a_directory_has_one_canonical_identity(tmp_path, kind):
    """canonical(RAW typed spelling) == canonical(D), for every spelling.

    This is the invariant that keeps the identity used to LOCATE the index
    and the identity used to MATCH scope from ever disagreeing: whatever
    the user types, it reduces to the same key the recorded cwd reduces to.

    Note what is fed to canonical() here: the raw string as typed, NOT
    ``resolve_project_dir(typed)``. Going through resolve_project_dir first
    made this matrix prove nothing about canonical() at all -- resolve_live()
    calls Path.resolve(), which has already collapsed every one of these
    eleven spellings before canonical() is handed anything, so both of
    canonical()'s own lexical steps (expanduser, normpath) could be deleted
    with all 22 parametrized cases still green. canonical() is the function
    that must handle a raw historical ``cwd``, which never goes near
    resolve(), so a raw string is the only input that tests it.
    """
    fake_home = tmp_path / "home"
    project = fake_home / "work" / "myproject"
    project.mkdir(parents=True)

    if kind in ("mixed case", "upper case") and not observed_case_insensitive(tmp_path):
        pytest.skip("filesystem is case-sensitive; these spellings are different dirs")

    with patch.dict(os.environ, {"HOME": str(fake_home)}):
        typed = spell(project, fake_home, kind)
        assert paths.canonical(typed) == paths.canonical(str(project)), (
            f"raw spelling {kind!r} ({typed}) did not reduce to the same identity "
            "as the directory it names"
        )
        # And the live side must agree with it, or the index would be located
        # by one identity and matched by another.
        assert paths.canonical(resolve_project_dir(typed)) == paths.canonical(str(project))


def test_canonical_reduces_the_lexical_forms_it_claims_to(tmp_path):
    """canonical()'s two lexical steps, asserted as reductions.

    Stated as concrete before/after pairs rather than as a property some
    other function has already established. ``canonical`` is documented as
    expanduser -> normpath -> conditional fold, and it is the ONLY thing
    applied to a recorded ``cwd``: a transcript whose directory no longer
    exists cannot be resolve()'d, so if these steps stop happening the
    history side of every comparison silently stops matching.
    """
    fake_home = tmp_path / "home"
    fake_home.mkdir()

    with patch.dict(os.environ, {"HOME": str(fake_home)}):
        # expanduser: "~" must become the home directory, not a literal path
        # component under the current working directory.
        assert paths.canonical("~/a/b") == paths.canonical(str(fake_home / "a" / "b"))
        assert "~" not in str(paths.canonical("~/a/b"))
        # expanduser and normpath compose, in that order.
        assert paths.canonical("~/a/../b") == paths.canonical(str(fake_home / "b"))

    # normpath: duplicate separators and dot segments collapse.
    assert paths.canonical("/x//y/./z") == Path(paths.fold_case("/x/y/z"))
    assert paths.canonical("/x/y/w/../z") == Path(paths.fold_case("/x/y/z"))
    assert paths.canonical("/x/y/z/") == Path(paths.fold_case("/x/y/z"))


@pytest.mark.parametrize("kind", SPELLINGS)
def test_every_spelling_discovers_the_same_sessions(tmp_path, kind):
    """The identity property, exercised through real discovery."""
    fake_home = tmp_path / "home"
    project = fake_home / "work" / "myproject"
    project.mkdir(parents=True)
    claude_dir = fake_home / ".claude" / "projects"
    transcript = write_transcript(claude_dir, "roundtrip", str(project.resolve()))

    if kind in ("mixed case", "upper case") and not observed_case_insensitive(tmp_path):
        pytest.skip("filesystem is case-sensitive; these spellings are different dirs")

    with patch.dict(os.environ, {"HOME": str(fake_home)}):
        typed = spell(project, fake_home, kind)
        discovery._cwd_index_cache.clear()
        found = discover_sessions(resolve_project_dir(typed), no_subagents=False)
        discovery._cwd_index_cache.clear()

    assert [s.path for s in found] == [transcript], f"spelling {kind!r} ({typed}) found nothing"


def test_canonical_is_idempotent(tmp_path):
    """canonical(canonical(p)) == canonical(p).

    Load-bearing: the scope arrives already canonicalized from the CLI and
    is canonicalized again inside matching, so a non-idempotent step would
    corrupt it on the second pass.
    """
    for raw in ("~/a/../b/", "/x//y/./z", str(tmp_path) + "/.", "/W/Code//App/"):
        once = paths.canonical(raw)
        assert paths.canonical(once) == once


def test_sibling_prefix_is_not_a_match():
    """Canonicalization must not weaken the path-boundary check."""
    assert not paths.is_at_or_beneath("/w/project-extended", "/w/project")
    assert paths.is_at_or_beneath("/w/project/sub", "/w/project")
    assert paths.is_at_or_beneath("/w/project", "/w/project")
