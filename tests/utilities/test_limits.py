"""Unit tests for the process file-descriptor limit raise."""

from __future__ import annotations

import resource

import pytest

from ssgrep.utilities import limits
from ssgrep.utilities.limits import TARGET_SOFT_NOFILE, raise_nofile_limit


def _patch_limits(
    monkeypatch: pytest.MonkeyPatch,
    *,
    soft: int,
    hard: int,
    getrlimit_error: type[BaseException] | None = None,
    setrlimit_error: type[BaseException] | None = None,
) -> list[tuple[int, int]]:
    calls: list[tuple[int, int]] = []

    def fake_getrlimit(res: int) -> tuple[int, int]:
        assert res == resource.RLIMIT_NOFILE
        if getrlimit_error is not None:
            raise getrlimit_error
        return (soft, hard)

    def fake_setrlimit(res: int, limits_pair: tuple[int, int]) -> None:
        assert res == resource.RLIMIT_NOFILE
        if setrlimit_error is not None:
            raise setrlimit_error
        calls.append(limits_pair)

    monkeypatch.setattr(limits.resource, "getrlimit", fake_getrlimit)
    monkeypatch.setattr(limits.resource, "setrlimit", fake_setrlimit)
    return calls


def test_raises_soft_limit_to_target_when_hard_is_unlimited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_limits(monkeypatch, soft=256, hard=resource.RLIM_INFINITY)

    assert raise_nofile_limit() is True
    assert calls == [(TARGET_SOFT_NOFILE, resource.RLIM_INFINITY)]


def test_clamps_target_to_finite_hard_limit_above_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_limits(monkeypatch, soft=256, hard=65536)

    assert raise_nofile_limit() is True
    assert calls == [(TARGET_SOFT_NOFILE, 65536)]


def test_clamps_target_to_hard_limit_below_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_limits(monkeypatch, soft=256, hard=4096)

    assert raise_nofile_limit() is True
    assert calls == [(4096, 4096)]


def test_noop_when_soft_limit_already_covers_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_limits(monkeypatch, soft=65535, hard=resource.RLIM_INFINITY)

    assert raise_nofile_limit() is False
    assert calls == []


def test_noop_when_hard_limit_does_not_exceed_soft_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_limits(monkeypatch, soft=1024, hard=1024)

    assert raise_nofile_limit() is False
    assert calls == []


def test_custom_target_overrides_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_limits(monkeypatch, soft=256, hard=resource.RLIM_INFINITY)

    assert raise_nofile_limit(512) is True
    assert calls == [(512, resource.RLIM_INFINITY)]


def test_getrlimit_failure_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_limits(
        monkeypatch, soft=256, hard=resource.RLIM_INFINITY, getrlimit_error=OSError
    )

    assert raise_nofile_limit() is False
    assert calls == []


def test_setrlimit_oserror_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_limits(
        monkeypatch, soft=256, hard=resource.RLIM_INFINITY, setrlimit_error=OSError
    )

    assert raise_nofile_limit() is False
    assert calls == []


def test_setrlimit_valueerror_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_limits(
        monkeypatch, soft=256, hard=resource.RLIM_INFINITY, setrlimit_error=ValueError
    )

    assert raise_nofile_limit() is False
    assert calls == []
