"""Read-only repository inventory beneath a scan root.

Symlinks are counted and never followed, so a link cannot pull in (or duplicate)
another tree. Dependency/cache directories are pruned. Only directory metadata is
read: no file is opened.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ssgrep.accounting.census_model import RepositoryInventory

PRUNE = frozenset(
    {
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".cache",
    }
)
FIXTURE_DIRS = frozenset(
    {"fixtures", "fixture", "snapshots", "__snapshots__", "testdata", "golden", "samples"}
)


@dataclass
class Repository:
    """One inventoried repository; ``name`` is relative to the scan root."""

    name: str
    path: str
    nested: bool


class RepositoryIndex:
    """Repository lookup by working directory using longest-path containment."""

    def __init__(self, repos: list[Repository]) -> None:
        self.repos = repos
        self._by_path: dict[str, Repository] = {}
        for repo in repos:
            self._by_path[repo.path] = repo
            self._by_path.setdefault(os.path.realpath(repo.path), repo)

    def match(self, cwd: str) -> Repository | None:
        current = os.path.normpath(cwd)
        while True:
            found = self._by_path.get(current)
            if found is not None:
                return found
            parent = os.path.dirname(current)
            if parent == current:
                return None
            current = parent


def inventory(root: Path) -> tuple[RepositoryInventory, RepositoryIndex]:
    summary = RepositoryInventory()
    base = os.path.abspath(root)
    if not os.path.lexists(base):
        summary.root_status = "missing"
        return summary, RepositoryIndex([])
    if not os.path.isdir(base):
        summary.root_status = "not_a_directory"
        return summary, RepositoryIndex([])
    repos: list[Repository] = []
    found: dict[str, Repository] = {}

    def unreadable(_error: OSError) -> None:
        summary.unreadable_dirs += 1

    for here, dirs, files in os.walk(base, followlinks=False, onerror=unreadable):
        summary.directories_walked += 1
        dirs.sort()
        in_fixtures = _in_fixtures(base, here)
        if ".git" in dirs or ".git" in files:
            summary.found += 1
            if ".git" in files:
                summary.worktree_files += 1
            repo = Repository(os.path.relpath(here, base), here, _has_repo_ancestor(here, found))
            summary.nested += repo.nested
            repos.append(repo)
            found[here] = repo
        dirs[:] = [d for d in dirs if d != ".git"]
        kept = []
        for name in dirs:
            if name in PRUNE:
                summary.pruned_dirs += 1
            elif os.path.islink(os.path.join(here, name)):
                summary.symlinks_skipped += 1
            else:
                kept.append(name)
        dirs[:] = kept
        for name in files:
            if os.path.islink(os.path.join(here, name)):
                summary.symlinks_skipped += 1
            elif name.endswith(".jsonl"):
                summary.jsonl_files += 1
                summary.jsonl_in_fixtures += in_fixtures
    return summary, RepositoryIndex(repos)


def _has_repo_ancestor(path: str, found: dict[str, Repository]) -> bool:
    parent = os.path.dirname(path)
    while parent and parent != os.path.dirname(parent):
        if parent in found:
            return True
        parent = os.path.dirname(parent)
    return False


def _in_fixtures(base: str, here: str) -> bool:
    relative = os.path.relpath(here, base)
    return any(part in FIXTURE_DIRS for part in relative.split(os.sep))
