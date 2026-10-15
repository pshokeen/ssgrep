"""Compact replay resolves known native records directly, never whole histories."""

from __future__ import annotations

from pathlib import Path

import pytest

from ssgrep.accounting.native_source import NativeSourceStatus
from ssgrep.accounting.readers.claude import ClaudeReader
from tests.accounting.test_native_source import old_source


def test_direct_record_seek_does_not_scan_large_history(tmp_path: Path) -> None:
    path, reference, adapter = old_source(tmp_path)
    source = ClaudeReader().source_for(path)
    result = ClaudeReader().read(source)
    witness = next(item for item in result.evidence if item.kind == "tool_result")
    with path.open("ab") as handle:
        handle.truncate(2 * 1024 * 1024 * 1024)
    found = adapter.read_record(
        reference, witness.native_identity, witness.digest, locator=witness.locator
    )
    assert found.status is NativeSourceStatus.OK
    assert found.evidence.native_identity == witness.native_identity
    assert found.evidence.digest == witness.digest
    assert found.bytes_read < 128 * 1024


def test_direct_record_changes_are_not_silently_hydrated(tmp_path: Path) -> None:
    path, reference, adapter = old_source(tmp_path)
    result = ClaudeReader().read(ClaudeReader().source_for(path))
    witness = next(item for item in result.evidence if item.kind == "tool_result")
    path.write_bytes(path.read_bytes().replace(b"ERR_QUOTA_EXCEEDED", b"ERR_QUOTA_MUTATED!"))
    found = adapter.read_record(
        reference, witness.native_identity, witness.digest, locator=witness.locator
    )
    assert found.status is NativeSourceStatus.CHANGED
    assert found.evidence is None


def test_legacy_unknown_locator_is_explicit_unavailable_not_full_scan(tmp_path: Path) -> None:
    _, reference, adapter = old_source(tmp_path)
    found = adapter.read_record(reference, "claude:src1:result:w1", "0" * 64, locator="unknown")
    assert found.status is NativeSourceStatus.UNRESOLVED
    assert found.evidence is None and found.bytes_read == 0


@pytest.mark.parametrize("original", ["native-a", "native-b"])
@pytest.mark.parametrize("retarget_ancestor", [False, True])
def test_approved_native_root_retarget_never_opens_another_tenant(
    tmp_path: Path, original: str, retarget_ancestor: bool
) -> None:
    from ssgrep.accounting.native_source import NativeSourceAdapter

    other = "native-b" if original == "native-a" else "native-a"
    original_dir = tmp_path.resolve() / original
    other_dir = tmp_path.resolve() / other
    native_root = original_dir / "records" if retarget_ancestor else original_dir
    foreign_root = other_dir / "records" if retarget_ancestor else other_dir
    _, reference, first = old_source(native_root)
    old_source(foreign_root)
    first.approved_roots = (native_root,)
    accepted = first.resolve(reference.ref)
    assert accepted.status is NativeSourceStatus.OK
    observed = first.observed_references[reference.ref]

    def restarted():
        return NativeSourceAdapter(
            {observed.ref: observed},
            first.readers,
            (native_root,),
            first.scoper,
            first.now_ms,
        )

    # A new process can still read the unchanged explicitly approved source.
    assert restarted().resolve(reference.ref).status is NativeSourceStatus.OK
    original_dir.rename(tmp_path / (original + "-accepted"))
    original_dir.symlink_to(other_dir, target_is_directory=True)
    result = restarted().resolve(reference.ref)
    # The foreign transcript intentionally has identical bytes/digests: content
    # equality cannot substitute for the approved directory's authorization.
    assert result.status is NativeSourceStatus.UNAPPROVED
    assert result.line is None
