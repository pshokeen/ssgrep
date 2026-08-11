"""Staleness detection for indexed transcripts.

Cheap stat-and-cursor check across in-scope transcripts. For each discovered
session file, compare filesystem st_size and st_mtime against the FileCursor
stored in the index. Report how many files have moved.

No parsing, no repair — this is purely observation. Repair is a separate task.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from ssgrep.types import FileCursor


class FileStatus(Enum):
    """Classification of a file's staleness status."""

    UNCHANGED = "unchanged"
    APPENDED = "appended"
    REWRITTEN = "rewritten"
    VANISHED = "vanished"


@dataclass(frozen=True)
class FileStalenessInfo:
    """Staleness status for a single file."""

    path: Path
    status: FileStatus
    cursor_size: int | None
    cursor_mtime: float | None
    disk_size: int | None
    disk_mtime: float | None


@dataclass(frozen=True)
class StalenessReport:
    """Summary of staleness across all discovered files."""

    unchanged_count: int
    appended_count: int
    rewritten_count: int
    vanished_count: int
    stale_files: list[FileStalenessInfo]


def classify_file_staleness(
    path: Path, cursor: FileCursor | None, disk_size: int, disk_mtime: float
) -> FileStatus:
    """Classify a file's staleness based on cursor and disk stat.

    Args:
        path: The file path (used only for error context)
        cursor: Stored FileCursor if one exists, None if file is new
        disk_size: Current filesystem size from stat().st_size
        disk_mtime: Current filesystem mtime from stat().st_mtime

    Returns:
        FileStatus classification
    """
    if cursor is None:
        # New file not in index yet — this shouldn't happen in normal
        # staleness detection (we only check indexed files), but handle it
        return FileStatus.APPENDED

    if cursor.size == disk_size and cursor.mtime == disk_mtime:
        return FileStatus.UNCHANGED

    if disk_size >= cursor.size:
        # File grew or stayed same size
        if disk_mtime == cursor.mtime:
            # Size changed but mtime didn't — treat as appended anyway
            return FileStatus.APPENDED
        # Size grew and mtime changed (or stayed same but size grew)
        return FileStatus.APPENDED

    # disk_size < cursor.size — file shrunk, so it was truncated or rewritten
    return FileStatus.REWRITTEN


def detect_staleness(
    discovered_files: list[tuple[Path, int, float]],
    cursors: dict[Path, FileCursor],
) -> StalenessReport:
    """Detect staleness across discovered session files.

    Args:
        discovered_files: List of (path, size, mtime) tuples from discovery
        cursors: Dict mapping file path to stored FileCursor

    Returns:
        StalenessReport with counts and detailed staleness per file
    """
    stale_files: list[FileStalenessInfo] = []

    unchanged_count = 0
    appended_count = 0
    rewritten_count = 0
    vanished_count = 0

    # Build a set of discovered paths for fast lookup
    discovered_paths = {path for path, _, _ in discovered_files}

    # Check discovered files for staleness
    for path, disk_size, disk_mtime in discovered_files:
        cursor = cursors.get(path)
        status = classify_file_staleness(path, cursor, disk_size, disk_mtime)

        if status == FileStatus.UNCHANGED:
            unchanged_count += 1
        elif status == FileStatus.APPENDED:
            appended_count += 1
            stale_files.append(
                FileStalenessInfo(
                    path=path,
                    status=status,
                    cursor_size=cursor.size if cursor else None,
                    cursor_mtime=cursor.mtime if cursor else None,
                    disk_size=disk_size,
                    disk_mtime=disk_mtime,
                )
            )
        elif status == FileStatus.REWRITTEN:
            rewritten_count += 1
            stale_files.append(
                FileStalenessInfo(
                    path=path,
                    status=status,
                    cursor_size=cursor.size if cursor else None,
                    cursor_mtime=cursor.mtime if cursor else None,
                    disk_size=disk_size,
                    disk_mtime=disk_mtime,
                )
            )

    # Check for vanished files: cursors that have no corresponding discovered file
    for path, cursor in cursors.items():
        if path not in discovered_paths:
            vanished_count += 1
            stale_files.append(
                FileStalenessInfo(
                    path=path,
                    status=FileStatus.VANISHED,
                    cursor_size=cursor.size,
                    cursor_mtime=cursor.mtime,
                    disk_size=None,
                    disk_mtime=None,
                )
            )

    return StalenessReport(
        unchanged_count=unchanged_count,
        appended_count=appended_count,
        rewritten_count=rewritten_count,
        vanished_count=vanished_count,
        stale_files=stale_files,
    )


def is_index_stale(report: StalenessReport) -> bool:
    """Check if the index is stale (has any changed or vanished files).

    Args:
        report: StalenessReport to evaluate

    Returns:
        True if any file has changed or vanished, False if all unchanged
    """
    return report.appended_count > 0 or report.rewritten_count > 0 or report.vanished_count > 0


def stale_count(report: StalenessReport) -> int:
    """Get the total count of stale files (changed or vanished).

    Args:
        report: StalenessReport to evaluate

    Returns:
        Total count of appended + rewritten + vanished files
    """
    return report.appended_count + report.rewritten_count + report.vanished_count
