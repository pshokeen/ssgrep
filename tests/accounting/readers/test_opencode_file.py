"""Public APSW descriptor I/O: real bytes, partial/EOF semantics and ownership."""

from __future__ import annotations

import os
import select
import signal

import apsw
import pytest

from ssgrep.accounting.readers.opencode_file import DescriptorFile


def open_file(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        return DescriptorFile(descriptor)
    finally:
        os.close(descriptor)


def test_owned_dup_reads_after_approval_fd_closes_and_never_pads_eof(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"abcdef")
    file = open_file(source)
    assert file.xRead(3, 2) == b"cde"
    assert file.xRead(5, 4) == b"ef"
    assert file.xRead(5, 9) == b""
    assert file.xFileSize() == 6
    file.xClose()
    file.xClose()
    with pytest.raises(apsw.IOError):
        file.xRead(1, 0)


def test_partial_and_interrupted_pread_complete_real_bytes_without_fake_padding(
    tmp_path, monkeypatch
):
    source = tmp_path / "partial"
    source.write_bytes(b"abcdefgh")
    file = open_file(source)
    pread = os.pread
    interrupted = False

    def short_read(fd, amount, offset):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise InterruptedError()
        return pread(fd, min(amount, 2), offset)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(os, "pread", short_read)
            assert file.xRead(8, 0) == b"abcdefgh"
            assert file.xRead(8, 5) == b"fgh"
        assert interrupted
    finally:
        file.xClose()


def test_readonly_methods_never_modify_bytes_or_metadata_and_controls_are_conservative(tmp_path):
    source = tmp_path / "readonly"
    source.write_bytes(b"preserved native bytes")
    file = open_file(source)
    before = source.stat()
    try:
        with pytest.raises(apsw.ReadOnlyError):
            file.xWrite(b"bad", 0)
        with pytest.raises(apsw.ReadOnlyError):
            file.xTruncate(0)
        file.xLock(apsw.SQLITE_LOCK_SHARED)
        file.xUnlock(apsw.SQLITE_LOCK_NONE)
        file.xSync(apsw.SQLITE_SYNC_NORMAL)
        assert file.xCheckReservedLock() is False
        assert file.xFileControl(1234, 0xDEADBEEF) is False
        assert file.xDeviceCharacteristics() == 0
        assert file.xSectorSize() == 512
        assert source.read_bytes() == b"preserved native bytes"
        after = source.stat()
        assert (after.st_size, after.st_mtime_ns, after.st_ctime_ns) == (
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
    finally:
        file.xClose()
    writable = os.open(source, os.O_RDWR)
    try:
        with pytest.raises(apsw.ReadOnlyError):
            DescriptorFile(writable)
    finally:
        os.close(writable)


def test_inherited_file_rejects_reads_but_child_can_close_its_duplicate(tmp_path):
    source = tmp_path / "fork"
    source.write_bytes(b"parent data")
    file = open_file(source)
    receiving, sending = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(receiving)
        try:
            try:
                file.xRead(1, 0)
            except apsw.IOError:
                file.xClose()
                os.write(sending, b"rejected")
                os._exit(0)
            os._exit(1)
        except BaseException:
            os._exit(2)
    os.close(sending)
    reaped = False
    try:
        assert select.select([receiving], [], [], 2)[0], "descriptor child exceeded watchdog"
        assert os.read(receiving, 64) == b"rejected"
        _, status = os.waitpid(child, 0)
        reaped = True
        assert os.waitstatus_to_exitcode(status) == 0
        assert file.xRead(11, 0) == b"parent data"
    finally:
        os.close(receiving)
        if not reaped:
            try:
                os.kill(child, signal.SIGKILL)
            except ProcessLookupError:
                pass
            os.waitpid(child, 0)
        file.xClose()
