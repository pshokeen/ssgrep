"""Generic, float-free, deterministic dataclass <-> plain-JSON codec.

``to_plain`` produces exactly the structure ``contracts.canonical_bytes`` hashes
(dataclasses -> dicts of fields, StrEnum -> value, tuples -> lists, dict keys ->
str, floats rejected). ``from_plain`` rebuilds frozen dataclasses recursively from
the declared type hints and rejects anything that does not fit them: wrong types,
``bool`` where ``int`` is declared, unknown enum values, missing required fields,
and unknown (possibly newer-schema) fields. Types are never resolved from the data:
nothing is imported or evaluated by name.
"""

from __future__ import annotations

import dataclasses
import json
import types
import typing
from collections.abc import Mapping
from enum import Enum, StrEnum
from functools import cache
from typing import Any, NamedTuple

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
