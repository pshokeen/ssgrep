"""Private accounting storage: owner/symlink checks, 0700/0600, atomic writes.

Everything lives under ``<data-root>/accounting/`` (never ``notes/``, the repo
``.ssgrep/``, or LanceDB). Directory components below the data root are opened
one at a time with ``O_NOFOLLOW`` relative to the parent descriptor, so a
component replaced by a symlink (or owned by another user) is refused rather
than followed: callers get :class:`UnsafeStorageError` and must disable the
affected capture instead of writing through an untrusted destination.

Files are addressed by (directory descriptor, name) only, which also stops a
name from escaping its directory. Atomic publication is write to a unique
temporary name, fsync, rename, then fsync of the directory. Nothing here runs
on import beyond stdlib imports; the data root is resolved lazily.
"""

from __future__ import annotations

import contextlib
import dataclasses
import errno
import fcntl
import json
import os
import secrets
import stat
import types
import typing
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, StrEnum
from functools import cache
from pathlib import Path
from typing import Any, NamedTuple

from ssgrep.accounting.ledger.ledger_contracts import valid_profile_id

ACCOUNTING_DIR = "accounting"
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_DIR_MODE = 0o700
_FILE_MODE = 0o600


class UnsafeStorageError(OSError):
    """A component is linked, foreign-owned, non-plain, or exposed to other users."""


def _own(info: os.stat_result, what: str) -> None:
    if info.st_uid != os.geteuid():
        raise UnsafeStorageError(errno.EPERM, f"{what} is not owned by the current user")


def _open_component(parent_fd: int, name: str, *, create: bool) -> int:
    if not name or name in {".", ".."} or "/" in name or "\0" in name:
        raise UnsafeStorageError(errno.EINVAL, f"unsafe path component {name!r}")
    try:
        try:
            descriptor = os.open(name, _DIR_FLAGS, dir_fd=parent_fd)
        except FileNotFoundError:
            if not create:
                raise
            with contextlib.suppress(FileExistsError):  # lost a creation race: open the winner's
                os.mkdir(name, _DIR_MODE, dir_fd=parent_fd)
            descriptor = os.open(name, _DIR_FLAGS, dir_fd=parent_fd)
    except OSError as error:
        if error.errno in (errno.ELOOP, errno.ENOTDIR):
            raise UnsafeStorageError(
                error.errno, f"{name} is a symlink or not a directory"
            ) from error
        raise
    try:
        info = os.fstat(descriptor)
        _own(info, name)
        if info.st_mode & 0o077:
            if not create:
                raise UnsafeStorageError(errno.EPERM, f"{name} is accessible to other users")
            os.fchmod(descriptor, stat.S_IMODE(info.st_mode) & 0o700)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def open_dir(
    *parts: str,
    create: bool = True,
    root: SafeRootSnapshot | None = None,
) -> int:
    """Descriptor for ``<data-root>/accounting/<parts...>`` (owned, private directories).

    The data root itself is the operator's configured location and is opened
    as given (created 0700 when absent and ``create``); every component below
    it must be a plain, owned, non-symlink directory. Read-only opens refuse
    group/other access without changing permissions; creating opens may tighten
    them, never broadening owner access. ``FileNotFoundError`` when a component
    is absent and ``create`` is false.

    An explicit root binds the operation to its authorized path and device/inode;
    it is never created or reopened from mutable configuration. Child mutations
    remain relative to that pinned descriptor even if configuration changes.
    """
    if root is None:
        path = data_root()
        if create:
            path.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
        current = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    else:
        current = open_root(root)
    try:
        for part in (ACCOUNTING_DIR, *parts):
            following = _open_component(current, part, create=create)
            os.close(current)
            current = following
    except BaseException:
        os.close(current)
        raise
    return current


def open_profile_dir(
    profile: str,
    *sub: str,
    create: bool = True,
    root: SafeRootSnapshot | None = None,
) -> int:
    """Per-profile directory; a profile id is an identifier, never a path."""
    if not valid_profile_id(profile):
        raise UnsafeStorageError(errno.EINVAL, "profile ids are bounded identifiers, not paths")
    return open_dir(profile, *sub, create=create, root=root)


def create_exclusive(dir_fd: int, name: str) -> int:
    """Create a brand-new 0600 append-only file; never reuses or follows an existing name."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND | os.O_NOFOLLOW | os.O_CLOEXEC
    return os.open(name, flags, _FILE_MODE, dir_fd=dir_fd)


def open_regular(dir_fd: int, name: str) -> int:
    """Read an owned private regular file; refuse unsafe modes without changing permissions."""
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK  # a FIFO must not hang
    try:
        descriptor = os.open(name, flags, dir_fd=dir_fd)
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise UnsafeStorageError(error.errno, f"{name} is a symlink") from error
        raise
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise UnsafeStorageError(errno.EINVAL, f"{name} is not a regular file")
        _own(info, name)
        if info.st_mode & 0o177:  # group/other access or any execute bit
            raise UnsafeStorageError(errno.EPERM, f"{name} has unsafe file permissions")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def read_file(dir_fd: int, name: str, max_bytes: int) -> bytes | None:
    """Whole file, or None when absent; ``ValueError`` if larger than ``max_bytes``."""
    try:
        descriptor = open_regular(dir_fd, name)
    except FileNotFoundError:
        return None
    try:
        data = b""
        while len(data) <= max_bytes:
            chunk = os.read(descriptor, max_bytes + 1 - len(data))
            if not chunk:
                return data
            data += chunk
        raise ValueError(f"{name} exceeds {max_bytes} bytes")
    finally:
        os.close(descriptor)


def list_names(dir_fd: int) -> list[str]:
    return sorted(os.listdir(dir_fd))


def fsync_dir(dir_fd: int) -> None:
    os.fsync(dir_fd)


def write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError(errno.EIO, "write made no progress")
        view = view[written:]


def _temporary_file(dir_fd: int, name: str, data: bytes) -> str:
    temporary = f".{name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptor = os.open(temporary, flags, _FILE_MODE, dir_fd=dir_fd)
    try:
        write_all(descriptor, data)
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        os.unlink(temporary, dir_fd=dir_fd)
        raise
    os.close(descriptor)
    return temporary


def atomic_write(dir_fd: int, name: str, data: bytes) -> None:
    """Replace ``name``: write temporary, fsync, rename, fsync the directory."""
    temporary = _temporary_file(dir_fd, name, data)
    try:
        os.rename(temporary, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
    except BaseException:
        os.unlink(temporary, dir_fd=dir_fd)
        raise
    fsync_dir(dir_fd)


def write_once(dir_fd: int, name: str, data: bytes) -> bool:
    """Atomically create ``name`` iff absent (hard link); False when it already exists."""
    temporary = _temporary_file(dir_fd, name, data)
    try:
        os.link(temporary, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd, follow_symlinks=False)
    except FileExistsError:
        return False
    finally:
        os.unlink(temporary, dir_fd=dir_fd)
    fsync_dir(dir_fd)
    return True


def delete_file(dir_fd: int, name: str) -> bool:
    """Unlink an owned regular file inside ``dir_fd``; refuse symlinks, dirs, foreign files."""
    try:
        info = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(info.st_mode):
        raise UnsafeStorageError(errno.EINVAL, f"{name} is not a regular file")
    _own(info, name)
    os.unlink(name, dir_fd=dir_fd)
    return True


"""Stdlib-only owner-protected file primitives for the policy and read surface.

Independent of the capture-side ``storage`` module on purpose: policy must be
loadable and writable even when journal storage is unavailable, and a read
must never create anything. Every directory below the configured data root is
opened relative to its pinned parent without following symlinks and checked
for foreign owners or group/other access before private reads.
"""


class UnsafePathError(OSError):
    """A path component is linked, foreign-owned, or has unsafe permissions."""


def _validate_dir(info: os.stat_result, path: Path | str, *, private: bool) -> None:
    if not stat.S_ISDIR(info.st_mode):
        raise UnsafePathError(f"{path} is not a plain directory")
    if info.st_uid != os.geteuid():
        raise UnsafePathError(f"{path} is not owned by the current user")
    if info.st_mode & 0o022:
        raise UnsafePathError(f"{path} is writable by other users")
    if private and info.st_mode & 0o077:
        raise UnsafePathError(f"{path} is accessible to other users")


def _check_dir(path: Path, *, private: bool = False) -> bool:
    """True when ``path`` is a safe directory; False when absent."""
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return False
    _validate_dir(info, path, private=private)
    return True


def _open_private_directory(parent: int, name: str) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        descriptor = os.open(name, flags, dir_fd=parent)
    except FileNotFoundError:
        raise
    except OSError as error:
        raise UnsafePathError(f"{name} cannot be opened safely: {error.strerror}") from error
    try:
        _validate_dir(os.fstat(descriptor), name, private=True)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


@contextmanager
def open_private(base: Path | int, *parts: str) -> Iterator[int | None]:
    """Pin an owned private regular file; yield None when any component is absent.

    A root pathname preserves operator-selected symlink locations; a supplied
    root descriptor is duplicated, preserving its pinned identity. Every
    component below it is a single no-follow relative open.
    Namespace changes cannot redirect a read after its parent has been pinned.
    Never creates, chmods or waits for a FIFO writer.
    """
    if not parts or any(
        not part or part in {".", ".."} or "/" in part or "\0" in part for part in parts
    ):
        raise UnsafePathError("unsafe private path component")
    try:
        directory = (
            os.dup(base)
            if isinstance(base, int)
            else os.open(base, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        )
    except FileNotFoundError:
        yield None
        return
    try:
        for part in parts[:-1]:
            try:
                following = _open_private_directory(directory, part)
            except FileNotFoundError:
                yield None
                return
            os.close(directory)
            directory = following
        try:
            descriptor = os.open(
                parts[-1],
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                dir_fd=directory,
            )
        except FileNotFoundError:
            yield None
            return
        except OSError as error:
            raise UnsafePathError(
                f"{parts[-1]} cannot be opened safely: {error.strerror}"
            ) from error
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise UnsafePathError(f"{parts[-1]} is not a regular file")
            if info.st_uid != os.geteuid() or info.st_mode & 0o022:
                raise UnsafePathError(f"{parts[-1]} is foreign-owned or writable by other users")
            if info.st_mode & 0o177:
                raise UnsafePathError(f"{parts[-1]} has unsafe file permissions")
            yield descriptor
        finally:
            os.close(descriptor)
    finally:
        os.close(directory)


def read_descriptor(descriptor: int, *, max_bytes: int) -> bytes:
    """Bounded bytes from an already validated private regular-file descriptor."""
    if os.fstat(descriptor).st_size > max_bytes:
        raise ValueError(f"private file exceeds {max_bytes} bytes")
    with os.fdopen(descriptor, "rb", closefd=False) as handle:
        data = handle.read(max_bytes + 1)
    if len(data) > max_bytes:  # a writer may have appended since fstat
        raise ValueError(f"private file exceeds {max_bytes} bytes")
    return data


def read_private(base: Path, *parts: str, max_bytes: int) -> bytes | None:
    """Bounded private bytes under ``base``; None when any component is absent.

    Unsafe components are rejected without repairing permissions. Owner read-only
    directories and files remain valid, and the read never creates anything.
    """
    with open_private(base, *parts) as descriptor:
        return None if descriptor is None else read_descriptor(descriptor, max_bytes=max_bytes)


def _ensure_dirs(base: Path, parts: tuple[str, ...]) -> Path:
    if not base.exists():
        base.mkdir(parents=True, exist_ok=True, mode=0o700)
    current = base
    for part in parts:
        current = current / part
        if not _check_dir(current):
            os.mkdir(current, 0o700)
    return current


def write_private(base: Path, *parts: str, data: bytes) -> None:
    """Atomically replace ``base``/``parts`` with ``data`` (dirs 0700, file 0600)."""
    directory = _ensure_dirs(base, parts[:-1])
    target = directory / parts[-1]
    temporary = directory / f".{parts[-1]}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    directory_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


@contextmanager
def exclusive_lock(base: Path, *dir_parts: str) -> Iterator[None]:
    """Serialize read-modify-write cycles across processes (flock on ``.lock``)."""
    directory = _ensure_dirs(base, dir_parts)
    descriptor = os.open(directory / ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


"""Data-root resolution without importing the heavy ``ssgrep.store`` package.

``ssgrep.store.paths.data_dir`` lives inside a package whose ``__init__``
imports LanceDB (hundreds of milliseconds). Capture resolves the same location
without that dependency: ``SSGREP_DATA_DIR`` when set, otherwise the platform
user-data directory for ``ssgrep``. A test pins this function to ``data_dir()``
for both branches so the two cannot drift. Root snapshots are noncreating by
default and bind private capture operations to one path and device/inode.
"""


def data_root() -> Path:
    override = os.environ.get("SSGREP_DATA_DIR")
    if override:
        return Path(override).expanduser().absolute()
    from platformdirs import user_data_path

    return Path(user_data_path("ssgrep", appauthor=False))


@dataclass(frozen=True, slots=True)
class SafeRootSnapshot:
    """Private configured namespace identity; never serialized into evidence."""

    path: Path
    identity: tuple[int, int]


class RootChanged(OSError):
    """Transient configuration/namespace change, not an unsafe-profile latch."""


def snapshot(path: Path | None = None, *, create: bool = False) -> SafeRootSnapshot | None:
    """Observe the configured root inode without creating anything by default.

    Operator-selected root symlinks remain valid. Capture authorization uses the
    noncreating default; explicit standalone storage callers may create a root.
    """
    path = data_root() if path is None else path
    if create:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    except FileNotFoundError:
        return None
    try:
        info = os.fstat(descriptor)
        return SafeRootSnapshot(path, (info.st_dev, info.st_ino))
    finally:
        os.close(descriptor)


def open_root(root: SafeRootSnapshot) -> int:
    """Pin exactly the authorized inode, never a newly resolved destination.

    Configuration is only a rejection guard: the open always uses ``root.path``,
    and all child operations must be relative to the returned descriptor.
    """
    if data_root() != root.path:
        raise RootChanged(errno.ESTALE, "accounting data root configuration changed")
    try:
        descriptor = os.open(root.path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    except OSError as error:
        raise RootChanged(errno.ESTALE, "accounting data root is no longer available") from error
    try:
        info = os.fstat(descriptor)
        if (info.st_dev, info.st_ino) != root.identity:
            raise RootChanged(errno.ESTALE, "accounting data root identity changed")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def require_current(root: SafeRootSnapshot) -> None:
    """Reject changed configuration or root identity without creating anything."""
    os.close(open_root(root))


"""Generic, float-free, deterministic dataclass <-> plain-JSON codec.

``to_plain`` produces exactly the structure ``contracts.canonical_bytes`` hashes
(dataclasses -> dicts of fields, StrEnum -> value, tuples -> lists, dict keys ->
str, floats rejected). ``from_plain`` rebuilds frozen dataclasses recursively from
the declared type hints and rejects anything that does not fit them: wrong types,
``bool`` where ``int`` is declared, unknown enum values, missing required fields,
and unknown (possibly newer-schema) fields. Types are never resolved from the data:
nothing is imported or evaluated by name.
"""


_Path = tuple[Any, ...]  # (parent path, segment) chain: formatted only on error
_ROOT: _Path = ()


def to_plain(value: Any) -> Any:
    """Plain JSON-able form of ``value``; TypeError on floats or unsupported types."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, Mapping):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    if isinstance(value, frozenset):
        return sorted(to_plain(v) for v in value)
    if isinstance(value, float):
        raise TypeError("floats are not allowed in canonical accounting data")
    if value is None or isinstance(value, (bool, int, str)):
        return value
    raise TypeError(f"cannot encode {type(value).__name__} as plain JSON")


def loads(data: str | bytes) -> Any:
    """``json.loads`` that refuses float literals and NaN/Infinity (ValueError)."""

    def no_float(text: str) -> Any:
        raise ValueError("float literals are not allowed in accounting data")

    def no_constant(text: str) -> Any:
        raise ValueError("non-finite numbers are not allowed in accounting data")

    return json.loads(data, parse_float=no_float, parse_constant=no_constant)


def from_plain(cls: Any, data: Any) -> Any:
    """Rebuild a value of declared type ``cls`` from plain JSON data (ValueError if unfit)."""
    return _decode(cls, data, _ROOT)


class _Field(NamedTuple):
    name: str
    tp: Any
    required: bool
    default: Any
    factory: Any


@cache
def _fields(cls: Any) -> tuple[_Field, ...]:
    hints = typing.get_type_hints(cls)
    out: list[_Field] = []
    for f in dataclasses.fields(cls):
        if not f.init:
            continue
        required = f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
        out.append(_Field(f.name, hints[f.name], required, f.default, f.default_factory))
    return tuple(out)


def _where(path: _Path) -> str:
    parts: list[str] = []
    while path:
        path, segment = path
        parts.append(str(segment))
    return ".".join(reversed(parts)) or "<root>"


def _bad(path: _Path, expected: str, data: Any) -> ValueError:
    return ValueError(f"{_where(path)}: expected {expected}, got {type(data).__name__}")


def _decode(tp: Any, data: Any, path: _Path) -> Any:
    if tp is Any or tp is object:
        _check_plain(data, path)
        return data
    if tp is None or tp is type(None):
        if data is not None:
            raise _bad(path, "null", data)
        return None
    origin = typing.get_origin(tp)
    if origin is typing.Union or origin is types.UnionType:
        return _decode_union(typing.get_args(tp), data, path)
    if origin is not None:
        return _decode_generic(origin, typing.get_args(tp), data, path)
    if tp is bool:
        if not isinstance(data, bool):
            raise _bad(path, "bool", data)
        return data
    if tp is int:
        if isinstance(data, bool) or not isinstance(data, int):
            raise _bad(path, "int", data)
        return data
    if tp is str:
        if not isinstance(data, str):
            raise _bad(path, "str", data)
        return data
    if tp is float:
        raise ValueError(f"{_where(path)}: floats are not allowed in accounting data")
    if isinstance(tp, type):
        if issubclass(tp, Enum):
            return _decode_enum(tp, data, path)
        if dataclasses.is_dataclass(tp):
            return _decode_dataclass(tp, data, path)
    raise ValueError(f"{_where(path)}: unsupported declared type {tp!r}")


def _decode_union(members: tuple[Any, ...], data: Any, path: _Path) -> Any:
    if data is None and type(None) in members:
        return None
    for member in members:
        if member is type(None):
            continue
        try:
            return _decode(member, data, path)
        except ValueError:
            continue
    raise _bad(path, "one of the declared union members", data)


def _decode_generic(origin: Any, args: tuple[Any, ...], data: Any, path: _Path) -> Any:
    if origin is tuple:
        if not isinstance(data, list):
            raise _bad(path, "list", data)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_decode(args[0], v, (path, i)) for i, v in enumerate(data))
        if len(data) != len(args):
            raise ValueError(f"{_where(path)}: expected {len(args)} items, got {len(data)}")
        return tuple(
            _decode(a, v, (path, i)) for i, (a, v) in enumerate(zip(args, data, strict=True))
        )
    if origin is list:
        if not isinstance(data, list):
            raise _bad(path, "list", data)
        return [_decode(args[0] if args else Any, v, (path, i)) for i, v in enumerate(data)]
    if origin is frozenset:
        if not isinstance(data, list):
            raise _bad(path, "list", data)
        return frozenset(
            _decode(args[0] if args else Any, v, (path, i)) for i, v in enumerate(data)
        )
    if origin is dict or origin is Mapping:
        if not isinstance(data, dict):
            raise _bad(path, "object", data)
        value_tp = args[1] if len(args) == 2 else Any
        out: dict[str, Any] = {}
        for key, value in data.items():
            if not isinstance(key, str):
                raise _bad(path, "string key", key)
            out[key] = _decode(value_tp, value, (path, key))
        return out
    raise ValueError(f"{_where(path)}: unsupported declared type {origin!r}")


def _decode_enum(tp: type[Enum], data: Any, path: _Path) -> Any:
    if not isinstance(data, str):
        raise _bad(path, tp.__name__, data)
    try:
        return tp(data)
    except ValueError:
        raise ValueError(f"{_where(path)}: unknown {tp.__name__} value") from None


def _decode_dataclass(tp: type, data: Any, path: _Path) -> Any:
    if not isinstance(data, dict):
        raise _bad(path, "object", data)
    declared = _fields(tp)
    names = {f.name for f in declared}
    extra = sorted(str(k) for k in data if k not in names)
    if extra:
        raise ValueError(f"{_where(path)}: unknown field(s) {', '.join(extra)}")
    kwargs: dict[str, Any] = {}
    for f in declared:
        if f.name in data:
            kwargs[f.name] = _decode(f.tp, data[f.name], (path, f.name))
        elif f.required:
            raise ValueError(f"{_where(path)}: missing field {f.name}")
        elif f.factory is not dataclasses.MISSING:
            kwargs[f.name] = f.factory()
        else:
            kwargs[f.name] = f.default
    try:
        return tp(**kwargs)
    except TypeError as exc:
        raise ValueError(f"{_where(path)}: {tp.__name__} rejected its fields") from exc


def _check_plain(data: Any, path: _Path) -> None:
    """``Any`` passthrough still has to be float-free JSON data."""
    if data is None or isinstance(data, (bool, int, str)):
        return
    if isinstance(data, float):
        raise ValueError(f"{_where(path)}: floats are not allowed in accounting data")
    if isinstance(data, list):
        for i, v in enumerate(data):
            _check_plain(v, (path, i))
        return
    if isinstance(data, dict):
        for k, v in data.items():
            if not isinstance(k, str):
                raise _bad(path, "string key", k)
            _check_plain(v, (path, k))
        return
    raise _bad(path, "plain JSON value", data)


"""Bounded, float-free JSON reads of existing accounting files (never creates)."""


MAX_SNAPSHOT_BYTES = 16 * 1024 * 1024


def _reject_float(text: str) -> Any:
    raise ValueError("floats are not allowed in accounting data")


def load_json(profile: str, *parts: str, limit: int) -> Any | None:
    """Parsed JSON under ``accounting/<profile>/``; None when the file is absent."""
    raw = read_private(data_root(), "accounting", profile, *parts, max_bytes=limit)
    return None if raw is None else json.loads(raw, parse_float=_reject_float)
