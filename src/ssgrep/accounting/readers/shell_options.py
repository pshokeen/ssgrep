"""Pure option parsing for supported shell command wrappers; never executes text."""

from __future__ import annotations

import re

ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
XARGS_VALUE_OPTIONS = frozenset(
    {"-n", "-I", "-P", "-L", "-d", "-E", "-s", "-a", "-l", "-i", "-j", "-N", "-S"}
)
ENV_VALUE_OPTIONS = frozenset({"-u", "--unset", "-C", "--chdir"})
DURATION = re.compile(r"^(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)[smhd]?$")
BUFFER = re.compile(r"^(?:L|0|[0-9]+(?:[KMGTPEZY](?:B)?)?)$")


def _after_options(
    words: list[str], value_options: frozenset[str], flags: frozenset[str] = frozenset()
) -> list[str] | None:
    j = 0
    while j < len(words) and words[j].startswith("-") and words[j] != "--":
        word = words[j]
        option = next(
            (
                opt
                for opt in value_options
                if word == opt
                or (word.startswith(opt + "=") if opt.startswith("--") else word.startswith(opt))
            ),
            None,
        )
        if option:
            if word == option:
                if j + 1 >= len(words):
                    return None
                value = words[j + 1]
                j += 2
            else:
                value = word[len(option) + (1 if option.startswith("--") else 0) :]
                j += 1
            if not value or (
                value_options == XARGS_VALUE_OPTIONS
                and option in {"-n", "-P", "-L", "-s", "-l", "-j", "-N", "-S"}
                and not value.isdecimal()
            ):
                return None
        elif word in flags:
            j += 1
        else:
            return None
    return words[j + 1 :] if j < len(words) and words[j] == "--" else words[j:]


def _wrapper(base: str, rest: list[str]) -> list[str] | None:
    """Consume only documented wrapper options, leaving the request argv intact."""
    if base == "env":
        tail = _after_options(rest, ENV_VALUE_OPTIONS, frozenset({"-i", "--ignore-environment"}))
        if tail is None:
            return None
        while tail and ASSIGN.match(tail[0]):
            tail = tail[1:]
        return tail or None
    j, modes = 0, 0
    while j < len(rest) and rest[j].startswith("-") and rest[j] != "--":
        word = rest[j]
        if base == "timeout" and word in {"--foreground", "--preserve-status", "--verbose", "-v"}:
            j += 1
            continue
        options = (
            {"-k": "duration", "--kill-after": "duration", "-s": "signal", "--signal": "signal"}
            if base == "timeout"
            else {
                "-i": "buffer",
                "-o": "buffer",
                "-e": "buffer",
                "--input": "buffer",
                "--output": "buffer",
                "--error": "buffer",
            }
        )
        found = next(
            (
                opt
                for opt in options
                if word == opt
                or (word.startswith(opt + "=") if opt.startswith("--") else word.startswith(opt))
            ),
            None,
        )
        if found is None:
            return None
        if word == found:
            j += 1
            if j >= len(rest):
                return None
            value = rest[j]
        else:
            value = word[len(found) + (1 if found.startswith("--") else 0) :]
        kind = options[found]
        if (
            not value
            or (kind == "duration" and not DURATION.fullmatch(value))
            or (kind == "buffer" and not BUFFER.fullmatch(value))
            or (kind == "signal" and not re.fullmatch(r"(?:[0-9]+|(?:SIG)?[A-Z]+)", value))
        ):
            return None
        modes += 1
        j += 1
    if j < len(rest) and rest[j] == "--":
        j += 1
    if base == "timeout":
        if j >= len(rest) or not DURATION.fullmatch(rest[j]):
            return None
        j += 1
        if j < len(rest) and rest[j] == "--":
            j += 1
    elif not modes:
        return None
    return rest[j:] or None
