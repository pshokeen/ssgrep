"""Ingest-only credential references for organization export.

A reference is ``env:NAME`` (``NAME`` matches ``[A-Z_][A-Z0-9_]{0,63}``) or ``file:PATH``
(absolute after ``~`` expansion). ``check_ref`` is config-time and never reads an
environment value; ``resolve`` is the ONLY place an env reference is read (send time, never
cached) and returns a :class:`Credential` whose value is redacted in ``repr``/``str``.

File rules: the final path component must be a regular file opened with ``O_NOFOLLOW``,
owned by the current user, with no group/other permission bits, 1..4096 bytes, printable
ASCII after stripping, one line. (Non-ASCII is refused although the spec allows utf-8: an
HTTP header value must be representable on the wire and the transport only sends ASCII.)

Separation (privacy spec): a credential is refused (``credential_reuses_local_secret``) when
its file is the same inode as, or its value equals (``hmac.compare_digest``), the profile's
local endpoint token, fingerprint HMAC key, or export pseudonym key; the same value check
applies to env-resolved values. Any ``file:`` reference under ``<data-root>/accounting/`` is
refused outright (``credential_in_accounting_storage``): ingest credentials are provisioned
by the operator outside accounting storage.

The local file names are hard-coded here (not imported) to keep this module free of the
metrics-server import chain; keep in sync with ``http_auth.TOKEN_FILE`` ("endpoint.token"),
``fingerprint.KEY_FILE`` ("fingerprint.key") and the export key ``export/pseudonym.key``.

``CredentialError.condition`` is a stable machine value, ``.action`` is corrective text;
``str()`` is ``"<condition>: <action>"`` and never contains a secret.
"""

from __future__ import annotations

import hmac
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from ssgrep.accounting.ledger.ledger_contracts import valid_profile_id
from ssgrep.accounting.privacy import storage as datadir

HEADER_NAME = "Authorization"
MAX_CREDENTIAL_BYTES = 4096
ACCOUNTING_DIR = "accounting"
LOCAL_SECRET_FILES = (
    ("endpoint.token",),  # ssgrep.accounting.serve.http_auth.TOKEN_FILE
    ("fingerprint.key",),  # ssgrep.accounting.serve.fingerprint.KEY_FILE
    ("export", "pseudonym.key"),
)
_ENV_NAME = re.compile(r"[A-Z_][A-Z0-9_]{0,63}")
_PRINTABLE = re.compile(r"[\x21-\x7e]+")


class CredentialError(Exception):
    """A credential reference is unusable. Carries a stable condition and corrective action."""

    def __init__(self, condition: str, action: str) -> None:
        super().__init__(f"{condition}: {action}")
        self.condition = condition
        self.action = action


@dataclass(frozen=True)
class Credential:
    header_name: str
    value: str = field(repr=False)

    def __repr__(self) -> str:
        return f"Credential(header_name={self.header_name!r}, value=<redacted>)"

    __str__ = __repr__


def _parse(profile: str, ref: str) -> tuple[str, str]:
    if not valid_profile_id(profile):
        raise CredentialError("profile_invalid", "use a profile id such as 'acme'")
    scheme, _, rest = ref.partition(":")
    if scheme == "env" and _ENV_NAME.fullmatch(rest):
        return scheme, rest
    if scheme == "file" and rest and "\x00" not in rest:
        path = os.path.expanduser(rest)
        if os.path.isabs(path):
            return scheme, os.path.normpath(path)
    raise CredentialError(
        "credential_ref_invalid",
        "use env:NAME (upper case letters, digits, underscore) or file:/absolute/path",
    )


def _accounting_root() -> str:
    return os.path.join(str(datadir.data_root()), ACCOUNTING_DIR)


def _within(path: str, root: str) -> bool:
    return os.path.commonpath([path, root]) == root


def _check_location(path: str) -> None:
    root = _accounting_root()
    real_root = os.path.realpath(root)
    real_parent = os.path.realpath(os.path.dirname(path))
    if _within(path, os.path.normpath(root)) or _within(real_parent, real_root):
        raise CredentialError(
            "credential_in_accounting_storage",
            "store the ingest credential outside the ssgrep data directory",
        )


def _read_secret_fd(path: str) -> tuple[bytes, os.stat_result]:
    """Open ``path`` safely and return (content, stat). Raises ``CredentialError``."""
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError as error:
        raise CredentialError(
            "credential_missing", f"create the credential file {path} (chmod 600) or change the ref"
        ) from error
    except OSError as error:
        raise CredentialError(
            "credential_unsafe_file",
            f"{path} must be a regular file, not a symlink: replace it with the real file",
        ) from error
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise CredentialError(
                "credential_unsafe_file", f"{path} must be a regular file: point the ref at one"
            )
        if info.st_uid != os.geteuid():
            raise CredentialError(
                "credential_unsafe_file", f"{path} must be owned by you: chown $(id -un) {path}"
            )
        if info.st_mode & 0o077:
            raise CredentialError(
                "credential_unsafe_file", f"{path} must be owner-only: chmod 600 {path}"
            )
        if not 1 <= info.st_size <= MAX_CREDENTIAL_BYTES:
            raise CredentialError(
                "credential_malformed",
                f"{path} must hold a single token of 1 to {MAX_CREDENTIAL_BYTES} bytes",
            )
        data = os.read(descriptor, MAX_CREDENTIAL_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(data) > MAX_CREDENTIAL_BYTES:
        raise CredentialError(
            "credential_malformed", f"{path} must be at most {MAX_CREDENTIAL_BYTES} bytes"
        )
    return data, info


def _token(raw: bytes, where: str) -> str:
    """The stripped single-line printable ASCII token, else ``credential_malformed``."""
    try:
        text = raw.decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise CredentialError(
            "credential_malformed", f"{where} must contain printable ASCII only"
        ) from error
    if not _PRINTABLE.fullmatch(text):
        raise CredentialError(
            "credential_malformed",
            f"{where} must hold one non-empty line without spaces or control characters",
        )
    return text


def _local_secrets(profile: str) -> list[tuple[bytes, tuple[int, int] | None]]:
    """(content, (dev, ino)) of each local secret that exists; unreadable ones are skipped."""
    base = os.path.join(_accounting_root(), profile)
    found: list[tuple[bytes, tuple[int, int] | None]] = []
    for parts in LOCAL_SECRET_FILES:
        try:
            descriptor = os.open(
                os.path.join(base, *parts), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            )
        except OSError:
            continue
        try:
            info = os.fstat(descriptor)
            if stat.S_ISREG(info.st_mode):
                content = os.read(descriptor, 4 * MAX_CREDENTIAL_BYTES)
                found.append((content, (info.st_dev, info.st_ino)))
        except OSError:
            pass
        finally:
            os.close(descriptor)
    return found


def _check_separation(profile: str, value: str, info: os.stat_result | None) -> None:
    probe = value.encode("ascii")
    for content, identity in _local_secrets(profile):
        same_inode = info is not None and identity == (info.st_dev, info.st_ino)
        stripped = content.strip()
        # Binary key edge bytes are entropy, not text padding. Check the exact
        # material before also accepting the existing text-token normalization.
        same_value = hmac.compare_digest(probe, content.hex().encode("ascii")) or (
            hmac.compare_digest(probe, stripped)
            or (bool(stripped) and hmac.compare_digest(probe, stripped.hex().encode("ascii")))
        )
        if same_inode or same_value:
            raise CredentialError(
                "credential_reuses_local_secret",
                "configure a separate ingest-only credential issued by the organization; "
                "never reuse a local ssgrep token or key",
            )


def check_ref(profile: str, ref: str | None, *, environ: Mapping[str, str] | None = None) -> None:
    """Config-time validation. Never reads an environment value and never returns content.

    ``environ`` is accepted for call-site symmetry with ``resolve`` and deliberately unused:
    environment references are resolved only at send time.
    """
    del environ
    if ref is None:
        return
    scheme, rest = _parse(profile, ref)
    if scheme == "env":
        return
    _check_location(rest)
    raw, info = _read_secret_fd(rest)
    _check_separation(profile, _token(raw, rest), info)


def resolve(
    profile: str, ref: str | None, *, environ: Mapping[str, str] | None = None
) -> Credential | None:
    """SEND-TIME ONLY: the credential for ``ref`` (None when no auth is configured)."""
    if ref is None:
        return None
    scheme, rest = _parse(profile, ref)
    info: os.stat_result | None = None
    if scheme == "env":
        source = os.environ if environ is None else environ
        raw = source.get(rest, "")
        if not raw.strip():
            raise CredentialError(
                "credential_missing", f"set the environment variable {rest} before sending"
            )
        value = _token(raw.encode("utf-8", "replace"), f"environment variable {rest}")
    else:
        _check_location(rest)
        data, info = _read_secret_fd(rest)
        value = _token(data, rest)
    _check_separation(profile, value, info)
    return Credential(HEADER_NAME, f"Bearer {value}")


def endpoint_display(url: str) -> str:
    """``scheme://host[:port]/path``; userinfo/query are refused by policy and never shown."""
    parts = urlsplit(url)
    port = f":{parts.port}" if parts.port else ""
    return f"{parts.scheme}://{parts.hostname or ''}{port}{parts.path}"


def host_display(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.hostname or ''}{f':{parts.port}' if parts.port else ''}"


def credential_display(ref: str | None) -> str | None:
    """The reference NAME only: ``env:NAME`` or ``file:<basename>``; never a value or a path."""
    if ref is None:
        return None
    scheme, _, rest = ref.partition(":")
    if scheme == "file":
        return f"file:{rest.rstrip('/').rsplit('/', 1)[-1]}"
    return f"{scheme}:{rest}"
