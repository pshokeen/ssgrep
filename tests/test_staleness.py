"""Tests for staleness detection."""

from __future__ import annotations

import time
from pathlib import Path

from ssgrep.staleness import (
    FileStatus,
    StalenessReport,
    classify_file_staleness,
    detect_staleness,
    is_index_stale,
    stale_count,
)
from ssgrep.types import FileCursor


class TestClassifyFileStaleness:
    """Test the core classification logic."""

    def test_unchanged_file(self):
        """File with identical size and mtime is unchanged."""
        path = Path("/tmp/test.jsonl")
        cursor = FileCursor(
            path=path, size=1000, mtime=123.45, byte_offset=0, first_line_hash="abc123"
        )
        status = classify_file_staleness(path, cursor, disk_size=1000, disk_mtime=123.45)
        assert status == FileStatus.UNCHANGED

    def test_appended_file_same_mtime(self):
        """File that grew but kept same mtime is appended."""
        path = Path("/tmp/test.jsonl")
        cursor = FileCursor(
            path=path, size=1000, mtime=123.45, byte_offset=0, first_line_hash="abc123"
        )
        status = classify_file_staleness(path, cursor, disk_size=2000, disk_mtime=123.45)
        assert status == FileStatus.APPENDED

    def test_appended_file_newer_mtime(self):
        """File that grew and has newer mtime is appended."""
        path = Path("/tmp/test.jsonl")
        cursor = FileCursor(
            path=path, size=1000, mtime=123.45, byte_offset=0, first_line_hash="abc123"
        )
        status = classify_file_staleness(path, cursor, disk_size=2000, disk_mtime=200.0)
        assert status == FileStatus.APPENDED

    def test_rewritten_file_smaller(self):
        """File that shrank is rewritten (truncated)."""
        path = Path("/tmp/test.jsonl")
        cursor = FileCursor(
            path=path, size=1000, mtime=123.45, byte_offset=0, first_line_hash="abc123"
        )
        status = classify_file_staleness(path, cursor, disk_size=500, disk_mtime=200.0)
        assert status == FileStatus.REWRITTEN

    def test_rewritten_file_same_size_different_mtime(self):
        """File with same size but different mtime could be rewritten."""
        path = Path("/tmp/test.jsonl")
        cursor = FileCursor(
            path=path, size=1000, mtime=123.45, byte_offset=0, first_line_hash="abc123"
        )
        status = classify_file_staleness(path, cursor, disk_size=1000, disk_mtime=200.0)
        # Same size, newer mtime — treat as appended (mtime changed but size didn't grow)
        assert status == FileStatus.APPENDED

    def test_new_file_no_cursor(self):
        """File with no cursor is treated as appended."""
        path = Path("/tmp/test.jsonl")
        status = classify_file_staleness(path, None, disk_size=1000, disk_mtime=123.45)
        assert status == FileStatus.APPENDED


class TestDetectStaleness:
    """Test staleness detection across multiple files."""

    def test_all_unchanged(self):
        """All files unchanged yields no stale files."""
        discovered = [
            (Path("/tmp/f1.jsonl"), 1000, 100.0),
            (Path("/tmp/f2.jsonl"), 2000, 200.0),
        ]
        cursors = {
            Path("/tmp/f1.jsonl"): FileCursor(
                path=Path("/tmp/f1.jsonl"),
                size=1000,
                mtime=100.0,
                byte_offset=0,
                first_line_hash="h1",
            ),
            Path("/tmp/f2.jsonl"): FileCursor(
                path=Path("/tmp/f2.jsonl"),
                size=2000,
                mtime=200.0,
                byte_offset=0,
                first_line_hash="h2",
            ),
        }

        report = detect_staleness(discovered, cursors)
        assert report.unchanged_count == 2
        assert report.appended_count == 0
        assert report.rewritten_count == 0
        assert report.vanished_count == 0
        assert len(report.stale_files) == 0

    def test_appended_file(self):
        """Appended file appears in stale_files."""
        discovered = [
            (Path("/tmp/f1.jsonl"), 2000, 100.0),  # Grew from 1000 to 2000
        ]
        cursors = {
            Path("/tmp/f1.jsonl"): FileCursor(
                path=Path("/tmp/f1.jsonl"),
                size=1000,
                mtime=100.0,
                byte_offset=0,
                first_line_hash="h1",
            ),
        }

        report = detect_staleness(discovered, cursors)
        assert report.appended_count == 1
        assert len(report.stale_files) == 1
        assert report.stale_files[0].status == FileStatus.APPENDED

    def test_rewritten_file(self):
        """Rewritten (truncated) file appears in stale_files."""
        discovered = [
            (Path("/tmp/f1.jsonl"), 500, 200.0),  # Shrank from 1000 to 500
        ]
        cursors = {
            Path("/tmp/f1.jsonl"): FileCursor(
                path=Path("/tmp/f1.jsonl"),
                size=1000,
                mtime=100.0,
                byte_offset=0,
                first_line_hash="h1",
            ),
        }

        report = detect_staleness(discovered, cursors)
        assert report.rewritten_count == 1
        assert len(report.stale_files) == 1
        assert report.stale_files[0].status == FileStatus.REWRITTEN

    def test_vanished_file(self):
        """File in cursors but not discovered is vanished."""
        discovered = []
        cursors = {
            Path("/tmp/f1.jsonl"): FileCursor(
                path=Path("/tmp/f1.jsonl"),
                size=1000,
                mtime=100.0,
                byte_offset=0,
                first_line_hash="h1",
            ),
        }

        report = detect_staleness(discovered, cursors)
        assert report.vanished_count == 1
        assert len(report.stale_files) == 1
        assert report.stale_files[0].status == FileStatus.VANISHED
        assert report.stale_files[0].disk_size is None
        assert report.stale_files[0].disk_mtime is None

    def test_mixed_staleness(self):
        """Mixed unchanged, appended, rewritten, and vanished files."""
        discovered = [
            (Path("/tmp/unchanged.jsonl"), 1000, 100.0),
            (Path("/tmp/appended.jsonl"), 2000, 100.0),
            (Path("/tmp/rewritten.jsonl"), 500, 200.0),
        ]
        cursors = {
            Path("/tmp/unchanged.jsonl"): FileCursor(
                path=Path("/tmp/unchanged.jsonl"),
                size=1000,
                mtime=100.0,
                byte_offset=0,
                first_line_hash="h1",
            ),
            Path("/tmp/appended.jsonl"): FileCursor(
                path=Path("/tmp/appended.jsonl"),
                size=1000,
                mtime=100.0,
                byte_offset=0,
                first_line_hash="h2",
            ),
            Path("/tmp/rewritten.jsonl"): FileCursor(
                path=Path("/tmp/rewritten.jsonl"),
                size=1000,
                mtime=100.0,
                byte_offset=0,
                first_line_hash="h3",
            ),
            Path("/tmp/vanished.jsonl"): FileCursor(
                path=Path("/tmp/vanished.jsonl"),
                size=1000,
                mtime=100.0,
                byte_offset=0,
                first_line_hash="h4",
            ),
        }

        report = detect_staleness(discovered, cursors)
        assert report.unchanged_count == 1
        assert report.appended_count == 1
        assert report.rewritten_count == 1
        assert report.vanished_count == 1
        assert len(report.stale_files) == 3  # Only non-unchanged


class TestIsIndexStale:
    """Test staleness summary predicates."""

    def test_all_unchanged_not_stale(self):
        """Index with all unchanged files is not stale."""
        report = StalenessReport(
            unchanged_count=10,
            appended_count=0,
            rewritten_count=0,
            vanished_count=0,
            stale_files=[],
        )
        assert not is_index_stale(report)

    def test_appended_is_stale(self):
        """Index with appended files is stale."""
        report = StalenessReport(
            unchanged_count=9, appended_count=1, rewritten_count=0, vanished_count=0, stale_files=[]
        )
        assert is_index_stale(report)

    def test_rewritten_is_stale(self):
        """Index with rewritten files is stale."""
        report = StalenessReport(
            unchanged_count=9, appended_count=0, rewritten_count=1, vanished_count=0, stale_files=[]
        )
        assert is_index_stale(report)

    def test_vanished_is_stale(self):
        """Index with vanished files is stale."""
        report = StalenessReport(
            unchanged_count=9, appended_count=0, rewritten_count=0, vanished_count=1, stale_files=[]
        )
        assert is_index_stale(report)

    def test_stale_count_sum(self):
        """stale_count() sums appended, rewritten, and vanished."""
        report = StalenessReport(
            unchanged_count=10,
            appended_count=2,
            rewritten_count=3,
            vanished_count=1,
            stale_files=[],
        )
        assert stale_count(report) == 6


class TestStalenessPerformance:
    """Benchmark staleness detection against synthetic corpus.

    The acceptance bound is under ~25ms for 1,209 files. We build a synthetic
    tree of 1,200 tiny files with cursors and assert the bound.
    """

    def test_benchmark_1200_files(self):
        """Detect staleness over 1,200 files must be under ~25ms.

        This is a synthetic benchmark that proves the staleness detection
        is cheap enough for bounded tail repair and user-facing searches.
        """
        # Build synthetic discovered files (1,200 files)
        discovered = [
            (Path(f"/synthetic/{i:04d}.jsonl"), 1024 + i, 1000.0 + i * 0.1) for i in range(1200)
        ]

        # Build synthetic cursors (slightly stale: some files grew)
        cursors = {}
        for i, (path, disk_size, disk_mtime) in enumerate(discovered):
            # Make ~10% of files "appended" (size grew since cursor was taken)
            cursor_size = disk_size - (100 if i % 10 == 0 else 0)
            cursors[path] = FileCursor(
                path=path,
                size=cursor_size,
                mtime=disk_mtime,
                byte_offset=0,
                first_line_hash=f"hash{i}",
            )

        # Time the detection
        start = time.perf_counter()
        report = detect_staleness(discovered, cursors)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Must complete under ~25ms
        assert elapsed_ms < 25, (
            f"Staleness detection took {elapsed_ms:.2f}ms, " f"exceeds 25ms budget"
        )

        # Verify correct counts: ~10% (120) should be appended, rest unchanged
        assert report.appended_count == 120, f"Expected 120 appended, got {report.appended_count}"
        assert (
            report.unchanged_count == 1080
        ), f"Expected 1080 unchanged, got {report.unchanged_count}"
        assert is_index_stale(report)
        assert stale_count(report) == 120

    def test_benchmark_empty(self):
        """Staleness detection over empty corpus is instantaneous."""
        start = time.perf_counter()
        report = detect_staleness([], {})
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 5, f"Empty detection took {elapsed_ms:.2f}ms"
        assert report.unchanged_count == 0
        assert report.appended_count == 0
        assert report.rewritten_count == 0
        assert report.vanished_count == 0
        assert not is_index_stale(report)


class TestStalenessFileStalenessInfo:
    """Test that FileStalenessInfo carries correct metadata."""

    def test_stale_file_appended_has_cursor_info(self):
        """Stale file of status APPENDED retains cursor and disk info."""
        discovered = [(Path("/tmp/f1.jsonl"), 2000, 123.45)]
        cursors = {
            Path("/tmp/f1.jsonl"): FileCursor(
                path=Path("/tmp/f1.jsonl"),
                size=1000,
                mtime=100.0,
                byte_offset=500,
                first_line_hash="hash1",
            ),
        }

        report = detect_staleness(discovered, cursors)
        assert len(report.stale_files) == 1

        info = report.stale_files[0]
        assert info.status == FileStatus.APPENDED
        assert info.cursor_size == 1000
        assert info.cursor_mtime == 100.0
        assert info.disk_size == 2000
        assert info.disk_mtime == 123.45

    def test_stale_file_vanished_has_no_disk_info(self):
        """Stale file of status VANISHED has no disk info."""
        discovered = []
        cursors = {
            Path("/tmp/f1.jsonl"): FileCursor(
                path=Path("/tmp/f1.jsonl"),
                size=1000,
                mtime=100.0,
                byte_offset=500,
                first_line_hash="hash1",
            ),
        }

        report = detect_staleness(discovered, cursors)
        assert len(report.stale_files) == 1

        info = report.stale_files[0]
        assert info.status == FileStatus.VANISHED
        assert info.cursor_size == 1000
        assert info.cursor_mtime == 100.0
        assert info.disk_size is None
        assert info.disk_mtime is None


class TestStalenessGuard:
    """Mutation tests: prove the guard catches real changes.

    These tests verify that if the comparison logic is broken, tests go red.
    Each test breaks a specific comparison and asserts that detection fails.
    """

    def test_guard_detects_size_change(self):
        """Test fails if size comparison is removed.

        MUTATION: Comment out the size check in classify_file_staleness.
        This test MUST go red.
        """
        path = Path("/tmp/test.jsonl")
        cursor = FileCursor(path=path, size=1000, mtime=100.0, byte_offset=0, first_line_hash="abc")

        # Different size should be detected
        status = classify_file_staleness(path, cursor, disk_size=2000, disk_mtime=100.0)
        assert status == FileStatus.APPENDED

        # Another size difference
        status = classify_file_staleness(path, cursor, disk_size=500, disk_mtime=100.0)
        assert status == FileStatus.REWRITTEN

    def test_guard_detects_mtime_change(self):
        """Test fails if mtime comparison is removed.

        MUTATION: Comment out mtime in the unchanged check.
        This test MUST go red.
        """
        path = Path("/tmp/test.jsonl")
        cursor = FileCursor(path=path, size=1000, mtime=100.0, byte_offset=0, first_line_hash="abc")

        # Same size, different mtime should be detected
        status = classify_file_staleness(path, cursor, disk_size=1000, disk_mtime=200.0)
        assert status == FileStatus.APPENDED

    def test_guard_counts_appended(self):
        """Test fails if appended_count is always returned as 0.

        MUTATION: Change classify_file_staleness to always return UNCHANGED.
        This test MUST go red.
        """
        discovered = [(Path("/tmp/f1.jsonl"), 2000, 100.0)]
        cursors = {
            Path("/tmp/f1.jsonl"): FileCursor(
                path=Path("/tmp/f1.jsonl"),
                size=1000,
                mtime=100.0,
                byte_offset=0,
                first_line_hash="h1",
            ),
        }

        report = detect_staleness(discovered, cursors)
        assert report.appended_count == 1, "Should detect appended file"
        assert report.unchanged_count == 0, "Should not count as unchanged"

    def test_guard_counts_vanished(self):
        """Test fails if vanished files are not counted.

        MUTATION: Remove the vanished file check in detect_staleness.
        This test MUST go red.
        """
        discovered = []
        cursors = {
            Path("/tmp/f1.jsonl"): FileCursor(
                path=Path("/tmp/f1.jsonl"),
                size=1000,
                mtime=100.0,
                byte_offset=0,
                first_line_hash="h1",
            ),
        }

        report = detect_staleness(discovered, cursors)
        assert report.vanished_count == 1, "Should detect vanished file"
        assert is_index_stale(report), "Vanished file should make index stale"

    def test_guard_is_index_stale_detects_appended(self):
        """Test fails if is_index_stale always returns False.

        MUTATION: Make is_index_stale always return False.
        This test MUST go red.
        """
        report = StalenessReport(
            unchanged_count=99,
            appended_count=1,
            rewritten_count=0,
            vanished_count=0,
            stale_files=[],
        )
        assert is_index_stale(report), "Should detect appended files make index stale"

    def test_guard_is_index_stale_detects_rewritten(self):
        """Test fails if is_index_stale only checks appended.

        MUTATION: Remove rewritten check from is_index_stale.
        This test MUST go red.
        """
        report = StalenessReport(
            unchanged_count=99,
            appended_count=0,
            rewritten_count=1,
            vanished_count=0,
            stale_files=[],
        )
        assert is_index_stale(report), "Should detect rewritten files make index stale"

    def test_guard_is_index_stale_detects_vanished(self):
        """Test fails if is_index_stale only checks appended and rewritten.

        MUTATION: Remove vanished check from is_index_stale.
        This test MUST go red.
        """
        report = StalenessReport(
            unchanged_count=99,
            appended_count=0,
            rewritten_count=0,
            vanished_count=1,
            stale_files=[],
        )
        assert is_index_stale(report), "Should detect vanished files make index stale"

    def test_guard_stale_count_includes_all(self):
        """Test fails if stale_count doesn't sum all categories.

        MUTATION: Make stale_count return appended_count only.
        This test MUST go red.
        """
        report = StalenessReport(
            unchanged_count=10,
            appended_count=2,
            rewritten_count=3,
            vanished_count=1,
            stale_files=[],
        )
        assert stale_count(report) == 6, "Should sum appended + rewritten + vanished"
