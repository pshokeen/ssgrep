"""Git's racy-clean rule, pinned at its boundaries and at a caller that relies on it.

Every value is synthetic, so the outcome never depends on the host filesystem's
timestamp resolution or on how fast the test runs.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from ssgrep.accounting.capture import fingerprint
from ssgrep.accounting.privacy import storage
from ssgrep.accounting.racy import GRANULE_NS, racy

SECOND = 1_000_000_000
WHOLE_SECOND_GRANULE = 2 * SECOND
FINE = 1_700_000_000 * SECOND + 123_456_789  # a stamp with sub-second precision
WHOLE = 1_700_000_000 * SECOND  # a stamp from a filesystem that keeps whole seconds


def test_a_stamp_one_granule_old_is_settled_and_one_nanosecond_newer_is_racy() -> None:
    assert not racy(FINE + GRANULE_NS, FINE)
    assert racy(FINE + GRANULE_NS - 1, FINE)
    assert racy(FINE, FINE)


def test_the_newest_stamp_decides() -> None:
    older, newest = FINE - SECOND, FINE
    assert not racy(newest + GRANULE_NS, older, newest)
    assert racy(newest + GRANULE_NS - 1, older, newest)
    assert racy(newest + GRANULE_NS - 1, newest, older)


def test_whole_second_stamps_need_a_two_second_granule() -> None:
    newest = WHOLE + SECOND
    assert racy(newest + GRANULE_NS, WHOLE, newest)
    assert racy(newest + WHOLE_SECOND_GRANULE - 1, WHOLE, newest)
    assert not racy(newest + WHOLE_SECOND_GRANULE, WHOLE, newest)


def test_one_sub_second_stamp_keeps_the_fine_granule() -> None:
    fine = WHOLE + 1
    assert not racy(fine + GRANULE_NS, WHOLE, fine)
    assert racy(fine + GRANULE_NS - 1, WHOLE, fine)
    # The whole-second stamp may be the newest; the mixed set still uses the fine granule.
    assert not racy(WHOLE + SECOND + GRANULE_NS, WHOLE - 1, WHOLE + SECOND)


def test_no_stamps_is_never_racy() -> None:
    assert not racy(0)
    assert not racy(FINE)


PROFILE = "racy-key"
STAMP = FINE


def _rewrite_key(body: bytes) -> None:
    directory = storage.open_profile_dir(PROFILE)
    try:
        storage.atomic_write(directory, fingerprint.KEY_FILE, body)
    finally:
        os.close(directory)


@pytest.mark.parametrize(
    ("observed", "rereads"),
    [
        (STAMP, True),  # captured in the same tick as the stamps
        (STAMP + GRANULE_NS - 1, True),
        (STAMP + GRANULE_NS, False),  # settled: a later write must move a stamp
        (STAMP + 3 * SECOND, False),
    ],
)
def test_cached_key_survives_an_equal_signature_only_when_its_observation_was_settled(
    monkeypatch: pytest.MonkeyPatch, observed: int, rereads: bool
) -> None:
    """A same-size rewrite in one coarse tick leaves every stat field equal.

    ``_signature`` is pinned to one value so the rewrite is invisible to stat on every
    platform; only the time the cached signature was observed decides whether the key
    on disk is read again.
    """
    fingerprint.forget_keys()
    old = b"o" * fingerprint.KEY_BYTES
    new = b"n" * fingerprint.KEY_BYTES
    _rewrite_key(old)
    signature = (1, 2, STAMP, STAMP, fingerprint.KEY_BYTES, 0o100600, os.getuid())
    monkeypatch.setattr(fingerprint, "_signature", lambda descriptor: signature)
    clock = SimpleNamespace(time_ns=lambda: observed)
    monkeypatch.setattr(fingerprint, "time", clock)
    try:
        assert fingerprint.load_key(PROFILE, create=False) == old
        _rewrite_key(new)
        # The later lookup's own clock is far past the stamps; only the stored one counts.
        clock.time_ns = lambda: STAMP + 10 * SECOND
        assert fingerprint.load_key(PROFILE, create=False) == (new if rereads else old)
    finally:
        fingerprint.forget_keys()
