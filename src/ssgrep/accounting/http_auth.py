"""Endpoint bearer token and Host validation for the loopback metrics server.

The token lives in ``<data>/accounting/<profile>/endpoint.token`` (0600). It is a
different file from the fingerprint HMAC key and from any remote ingest credential:
nothing here reads either of those and nothing outside this module may send this
token anywhere. A token file that is a symlink, foreign-owned, not a regular file, or
readable by group/other is REFUSED (never silently repaired). Comparison is constant
time over fixed-length digests, so neither token length nor content leaks by timing.
"""

from __future__ import annotations

import errno
import hashlib
import hmac
import os
import re
import secrets
import stat
from collections.abc import Collection
from urllib.parse import parse_qsl, urlsplit

from ssgrep.accounting import storage

TOKEN_FILE = "endpoint.token"
TOKEN_BYTES = 32  # 256 random bits, hex encoded on disk
MAX_PRESENTED_TOKEN = 512
_BEARER = re.compile(r"(?i)bearer +([\x21-\x7e]+)")
_HEX_TOKEN = re.compile(rb"[0-9a-f]{64}")
URL_CREDENTIAL_NAMES = frozenset(
    {"token", "access_token", "api_key", "apikey", "key", "auth", "authorization", "bearer"}
)


class TokenFileError(OSError):
    """The endpoint token file is unsafe, damaged, or unreadable (the server must not start)."""


def _read_token(directory: int) -> bytes | None:
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    try:
        descriptor = os.open(TOKEN_FILE, flags, dir_fd=directory)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise TokenFileError(
            error.errno or errno.EIO, "token file cannot be opened safely"
        ) from error
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid():
            raise TokenFileError(errno.EPERM, "token file is not an owned regular file")
        if info.st_mode & 0o077:
            raise TokenFileError(errno.EPERM, "token file must be owner-only (0600)")
        data = os.read(descriptor, 4 * (TOKEN_BYTES * 2 + 1))
    finally:
        os.close(descriptor)
    token = data.strip()
    if not _HEX_TOKEN.fullmatch(token):
        raise TokenFileError(errno.EINVAL, "token file content is damaged")
    return token


def load_token(profile: str) -> bytes | None:
    """The profile's token; None when absent; ``TokenFileError`` when unsafe. Never creates."""
    try:
        directory = storage.open_profile_dir(profile, create=False)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise TokenFileError(error.errno or errno.EIO, "profile directory is unsafe") from error
    try:
        return _read_token(directory)
    finally:
        os.close(directory)


def load_token_or_none(profile: str) -> bytes | None:
    """Per-request token read; an unsafe or damaged file means NO credential is valid."""
    try:
        return load_token(profile)
    except TokenFileError:
        return None


def credential_in_url(target: str) -> bool:
    """True when the request target carries a credential-looking parameter or userinfo."""
    parts = urlsplit(target)
    if "@" in parts.netloc:
        return True
    names = [name.lower() for name, _ in parse_qsl(parts.query, keep_blank_values=True)]
    return any(name in URL_CREDENTIAL_NAMES for name in names)


def ensure_token(profile: str) -> bytes:
    """Load the token, creating a new 0600 random one atomically when none exists."""
    try:
        directory = storage.open_profile_dir(profile, create=True)
    except OSError as error:
        raise TokenFileError(error.errno or errno.EIO, "profile directory is unsafe") from error
    try:
        token = _read_token(directory)
        if token is None:
            candidate = secrets.token_hex(TOKEN_BYTES).encode("ascii") + b"\n"
            storage.write_once(directory, TOKEN_FILE, candidate)
            token = _read_token(directory)
        if token is None:
            raise TokenFileError(errno.EIO, "token file could not be created")
        return token
    finally:
        os.close(directory)


def presented_token(authorization: str | None) -> bytes | None:
    """The bearer value of an ``Authorization`` header; None when not a well-formed bearer."""
    if authorization is None or len(authorization) > MAX_PRESENTED_TOKEN + 16:
        return None
    match = _BEARER.fullmatch(authorization.strip())
    if match is None or len(match.group(1)) > MAX_PRESENTED_TOKEN:
        return None
    return match.group(1).encode("ascii")


def token_matches(presented: bytes | None, expected: bytes | None) -> bool:
    """Constant-time comparison; False when either side is missing."""
    if presented is None or expected is None:
        return False
    return hmac.compare_digest(
        hashlib.sha256(presented).digest(), hashlib.sha256(expected).digest()
    )


def approved_hosts(binds: Collection[str], port: int) -> frozenset[str]:
    """``Host`` header values this server accepts: names of its own loopback binds + its port."""
    names: set[str] = set()
    for bind in binds:
        names.update(("[::1]",) if bind == "::1" else ("127.0.0.1", "localhost"))
    return frozenset(f"{name}:{port}" for name in names)


def host_allowed(host: str | None, approved: frozenset[str]) -> bool:
    return host is not None and host.lower() in approved
