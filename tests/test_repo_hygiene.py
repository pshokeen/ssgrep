"""Repo hygiene checks for the open-source release.

Guards against agent-tool scratch state (.claude/, .codex/, .pi/,
.opencode/, .auto/, .cursor/, .windsurf/), internal planning docs
(openspec/, CLAUDE.md, AGENTS.md), and build junk (__pycache__, *.pyc,
.DS_Store, *.egg-info, .pytest_cache, .ruff_cache) ever landing in a
tracked file, regardless of what .gitignore currently says.

Per CLAUDE.md's "Assert the Property, Not a Proxy" rule, a test that only
ever asserts "no violations found in the real tree" would pass vacuously
if `matches_hygiene_violation` were dead code (e.g. `return False`, or a
patched-out check that never runs). `test_matcher_flags_planted_bad_paths`
plants known-bad paths in an in-memory fixture list and asserts the
matcher actually flags them, proving the detector is reachable and
discriminating -- not just a spy that always reports zero.
"""

from __future__ import annotations

import fnmatch
import subprocess
from pathlib import Path

import pytest

# Patterns are matched against each "/"-separated segment of a tracked-file
# path via fnmatch.fnmatchcase (case-sensitive, platform-independent). A
# plain pattern (no wildcard) is therefore an exact-segment match -- e.g.
# "CLAUDE.md" matches the segment "CLAUDE.md" but not "NOT_CLAUDE.md" or
# "docs" -- while "*.pyc" / "*.egg-info" match by suffix.
_AGENT_ARTIFACT_PATTERNS: tuple[str, ...] = (
    ".claude",
    ".codex",
    ".pi",
    ".opencode",
    ".auto",
    ".cursor",
    ".windsurf",
    "openspec",
    "CLAUDE.md",
    "AGENTS.md",
)

_BUILD_JUNK_PATTERNS: tuple[str, ...] = (
    "__pycache__",
    "*.pyc",
    ".DS_Store",
    "*.egg-info",
    ".pytest_cache",
    ".ruff_cache",
)

_HYGIENE_PATTERNS: tuple[str, ...] = _AGENT_ARTIFACT_PATTERNS + _BUILD_JUNK_PATTERNS


def matches_hygiene_violation(path: str) -> bool:
    """True if `path` (as `git ls-files` prints it, forward-slash separated)
    has a path segment matching a forbidden agent-artifact or build-junk
    pattern.

    Matching is per-segment, not substring, so "src/ssgrep/openspec_helper.py"
    (segment "openspec_helper.py") does not falsely match the "openspec"
    pattern, and "docs/NOT_CLAUDE.md" does not falsely match "CLAUDE.md".
    """
    segments = path.split("/")
    return any(
        fnmatch.fnmatchcase(segment, pattern)
        for segment in segments
        for pattern in _HYGIENE_PATTERNS
    )


# ---- (b) positive-detection fixtures: proves the matcher is reachable ----
#
# These paths are never written to disk or added to the repo -- they only
# exist as in-memory strings passed to matches_hygiene_violation().

_KNOWN_BAD_PATHS: tuple[str, ...] = (
    ".claude/settings.json",
    ".codex/config.toml",
    ".pi/state.json",
    ".opencode/session.db",
    ".auto/log.jsonl",
    ".cursor/rules.json",
    ".windsurf/config.json",
    "openspec/changes/opensource-release/tasks.md",
    "CLAUDE.md",
    "docs/CLAUDE.md",
    "AGENTS.md",
    "src/ssgrep/__pycache__/indexer.cpython-313.pyc",
    "some/nested/module.pyc",
    ".DS_Store",
    "docs/.DS_Store",
    "ssgrep.egg-info/PKG-INFO",
    ".pytest_cache/v/cache/lastfailed",
    ".ruff_cache/0.16.2/12345/foo",
)

_KNOWN_GOOD_PATHS: tuple[str, ...] = (
    "src/ssgrep/indexer.py",
    "tests/test_repo_hygiene.py",
    "docs/release-checklist.md",
    "pyproject.toml",
    "README.md",
    ".github/workflows/ci.yml",
    # Substring near-misses: must not be flagged by segment-exact matching.
    "src/ssgrep/openspec_helper.py",
    "docs/NOT_CLAUDE.md",
    ".claudelike/file.txt",
    "src/ssgrep/autocomplete.py",
)


@pytest.mark.parametrize("bad_path", _KNOWN_BAD_PATHS)
def test_matcher_flags_planted_bad_paths(bad_path: str) -> None:
    assert matches_hygiene_violation(bad_path) is True


@pytest.mark.parametrize("good_path", _KNOWN_GOOD_PATHS)
def test_matcher_passes_known_good_paths(good_path: str) -> None:
    assert matches_hygiene_violation(good_path) is False


# ---- (a) the real gate: no tracked file in the actual repo violates ----


def _repo_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


def test_no_agent_artifact_or_build_junk_files_tracked() -> None:
    repo_root = _repo_root()
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    tracked = [line for line in result.stdout.splitlines() if line]

    # Sanity check: fail loudly rather than passing vacuously if `git
    # ls-files` ever returned nothing (wrong cwd, detached worktree, etc.).
    assert len(tracked) > 100, (
        f"git ls-files returned suspiciously few files ({len(tracked)}); "
        "the hygiene check may not be scanning the real tree"
    )

    violations = [path for path in tracked if matches_hygiene_violation(path)]
    assert violations == [], f"tracked files match forbidden hygiene patterns: {violations}"
