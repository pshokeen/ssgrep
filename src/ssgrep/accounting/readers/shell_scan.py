"""Recognize ssgrep CLI invocations in shell text by parsing, never executing.

Only a word at *command position* counts: after env assignments, a small set of
wrappers (``env``, ``time``, ``uv run``, ``python -m``, ``xargs``, ``find -exec``,
``timeout``, ``stdbuf``, ``bash -c``, ``eval``) and shell keywords. Shell stdin
heredocs are scripts only for an unambiguous supported shell command. Quoted
strings, comments, other heredoc bodies and arguments are never command position.
Text shows at most that an invocation exists;
loops, conditionals, wrappers and multi-occurrence text are ``unknown`` cardinality.
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass, field

from ssgrep.accounting.readers.shell_options import (
    ASSIGN,
    XARGS_VALUE_OPTIONS,
    _after_options,
    _wrapper,
)

MAX_COMMAND_CHARS = 256 * 1024
MAX_DEPTH = 8
OPERATIONS = frozenset({"search", "show", "note"})
PUNCT = frozenset("();<>|&")
HEREDOC = re.compile(r"""<<(-?)[ \t]*(?:'([^'\n]+)'|"([^"\n]+)"|([^\s;<>|&()'"]+))""")
PYTHON = re.compile(r"^python[0-9.]*$")
SHELLS = frozenset({"bash", "sh", "zsh", "dash", "ksh"})
SKIP = frozenset(
    {
        "do",
        "then",
        "else",
        "!",
        "{",
        "}",
        "time",
        "command",
        "builtin",
        "exec",
        "nohup",
        "nice",
        "sudo",
    }
)
CONDITIONAL = frozenset({"if", "elif"})
LOOPS = frozenset({"while", "until"})
HEADERS = frozenset({"for", "select", "case"})
UV_VALUE_OPTIONS = frozenset(
    {"--project", "--directory", "--with", "--python", "-p", "--package", "--group", "--env-file"}
)
EXEC_FLAGS = frozenset({"-exec", "-execdir", "-ok", "-okdir"})
INTERPRETERS = frozenset({"ruby", "perl", "node", "fish", "pwsh", "powershell"})


class ShellParseError(ValueError):
    """The text could not be tokenized (for example an unterminated quote)."""


@dataclass
class ShellScan:
    argvs: list[tuple[str, ...]] = field(default_factory=list)
    uncertain: bool = False
    error: str | None = None
    unsupported: bool = False
    has_other_commands: bool = False

    @property
    def operations(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(argv[0] for argv in self.argvs))


def logical_lines(text: str) -> list[str]:
    """Split on newlines outside quotes; drop comments and heredoc bodies."""
    return [line for line, _ in _logical_units(text)]


def _logical_units(text: str) -> list[tuple[str, list[str]]]:
    """Retain stdin bodies separately so text is never a fallback command."""
    lines: list[tuple[str, list[str]]] = []
    buf: list[str] = []
    quote = ""
    pending: list[tuple[str, bool]] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if quote:
            buf.append(c)
            if quote == '"' and c == "\\" and i + 1 < n:
                buf.append(text[i + 1])
                i += 2
                continue
            quote = "" if c == quote else quote
            i += 1
            continue
        if c == "\\" and i + 1 < n:
            if text[i + 1] != "\n":
                buf.extend((c, text[i + 1]))
            i += 2
            continue
        if c in "'\"":
            quote = c
        elif c == "#" and (not buf or buf[-1] in " \t;|&("):
            while i < n and text[i] != "\n":
                i += 1
            continue
        elif c == "<" and text.startswith("<<", i) and not text.startswith("<<<", i):
            found = HEREDOC.match(text, i)
            if found:
                pending.append(
                    (next(g for g in found.groups()[1:] if g is not None), bool(found.group(1)))
                )
                buf.append(found.group(0))
                i = found.end()
                continue
            raise ShellParseError("unsupported heredoc delimiter")
        elif c == "\n":
            line = "".join(buf)
            buf = []
            i, bodies = _skip_heredocs(text, i + 1, pending)
            lines.append((line, bodies))
            pending = []
            continue
        buf.append(c)
        i += 1
    if quote or pending:
        raise ShellParseError("unterminated quote or heredoc")
    if buf:
        lines.append(("".join(buf), []))
    return lines


def _skip_heredocs(text: str, i: int, pending: list[tuple[str, bool]]) -> tuple[int, list[str]]:
    bodies: list[str] = []
    for delimiter, strip in pending:
        body: list[str] = []
        while i < len(text):
            end = text.find("\n", i)
            end = len(text) if end < 0 else end
            line = text[i:end]
            i = end + 1
            line = line.lstrip("\t") if strip else line
            if line == delimiter:
                break
            body.append(line)
        else:
            raise ShellParseError("unterminated heredoc")
        bodies.append("\n".join(body))
    return i, bodies


def _words(line: str) -> list[list[str]]:
    """Tokenize one logical line into command segments (redirects removed)."""
    lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    segments: list[list[str]] = [[]]
    skip_target = False
    for token in lexer:
        if token and all(ch in PUNCT for ch in token):
            if "<" in token or ">" in token:
                skip_target = True
                continue
            segments.append([])
            continue
        if skip_target:
            skip_target = False
            continue
        if token.startswith("`"):
            segments.append([])
            token = token.lstrip("`")
        closing = token.endswith("`")
        token = token.rstrip("`")
        if token:
            segments[-1].append(token)
        if closing:
            segments.append([])
    return segments


def scan_shell(text: str, *, depth: int = 0) -> ShellScan:
    result = ShellScan()
    if len(text) > MAX_COMMAND_CHARS:
        result.error = "command_too_large"
        return result
    try:
        for line, bodies in _logical_units(text):
            segments = [segment for segment in _words(line) if segment]
            for segment in segments:
                _command(segment, result, depth, bodies=bodies if len(segments) == 1 else [])
    except (ShellParseError, ValueError):
        result.argvs.clear()
        result.error = "unparseable_shell"
    return result


def scan_words(words: list[str], *, depth: int = 0) -> ShellScan:
    """Analyze an already-tokenized argv (for example ``["bash", "-lc", "..."]``)."""
    result = ShellScan()
    if sum(len(word) + 1 for word in words) > MAX_COMMAND_CHARS:
        result.error = "command_too_large"
        return result
    _command(words, result, depth)
    return result


def _script(script: str, result: ShellScan, depth: int) -> None:
    result.uncertain = True
    if depth >= MAX_DEPTH:
        result.unsupported = True
        result.has_other_commands = True
        return
    inner = scan_shell(script, depth=depth + 1)
    if inner.error is None:
        result.argvs.extend(inner.argvs)
        result.unsupported |= inner.unsupported
        result.has_other_commands |= inner.has_other_commands
    else:
        result.unsupported = True
        result.has_other_commands = True


def _command(
    words: list[str], result: ShellScan, depth: int, *, bodies: list[str] | None = None
) -> None:
    if depth > MAX_DEPTH:
        result.unsupported = True
        result.has_other_commands = True
        return
    i = 0
    while i < len(words):
        word = words[i]
        if ASSIGN.match(word) or word in SKIP:
            if word in SKIP and i + 1 < len(words) and words[i + 1].startswith("-"):
                result.unsupported = True
                result.has_other_commands = True
                return
            i += 1
        elif word in CONDITIONAL or word in LOOPS:
            result.uncertain = True
            i += 1
        elif word in HEADERS:
            result.uncertain = True
            return
        else:
            break
    if i >= len(words):
        return
    base = os.path.basename(words[i])
    rest = words[i + 1 :]
    if base == "ssgrep":
        _ssgrep(rest, result)
    elif base in {"timeout", "stdbuf", "env"}:
        result.uncertain = True
        tail = _wrapper(base, rest)
        if tail is None:
            result.unsupported = True
        else:
            if base == "timeout" and any(
                w in {"--verbose", "-v"} for w in rest[: len(rest) - len(tail)]
            ):
                result.has_other_commands = True
            _command(tail, result, depth + 1, bodies=bodies)
    elif base in {"uv", "uvx"}:
        _uv(base, rest, result, depth, bodies)
    elif PYTHON.match(base):
        if rest[:2] == ["-m", "ssgrep"]:
            _ssgrep(rest[2:], result)
        else:
            result.unsupported = True
            result.has_other_commands = True
    elif base in {"xargs", "parallel"}:
        result.uncertain = True
        tail = _after_options(rest, XARGS_VALUE_OPTIONS, frozenset({"-0", "-r", "-t", "-x"}))
        if tail is None:
            result.unsupported = True
        else:
            _command(tail, result, depth + 1)
    elif base == "find":
        _find(rest, result, depth)
    elif base in SHELLS:
        _shell_c(rest, result, depth, bodies)
    elif base == "eval" and rest:
        _script(" ".join(rest), result, depth)
    elif base in INTERPRETERS:
        result.unsupported = True
        result.has_other_commands = True
    elif base not in {"fi", "done", "esac", "in"}:
        result.has_other_commands = True


def _uv(
    base: str, rest: list[str], result: ShellScan, depth: int, bodies: list[str] | None
) -> None:
    if base == "uv":
        if not rest or rest[0] != "run":
            return
        rest = rest[1:]
    tail = _after_options(
        rest,
        UV_VALUE_OPTIONS,
        frozenset({"--quiet", "-q", "--no-sync", "--offline", "--frozen", "--locked"}),
    )
    if tail is None:
        result.unsupported = True
    else:
        _command(tail, result, depth + 1, bodies=bodies)


def _find(rest: list[str], result: ShellScan, depth: int) -> None:
    for j, word in enumerate(rest):
        if word in EXEC_FLAGS:
            result.uncertain = True
            command = []
            for item in rest[j + 1 :]:
                if item in {";", "+"}:
                    break
                command.append(item)
            _command(command, result, depth + 1)


def _shell_c(rest: list[str], result: ShellScan, depth: int, bodies: list[str] | None) -> None:
    j = 0
    while j < len(rest) and rest[j].startswith("-") and rest[j] != "--":
        word = rest[j]
        if word == "-" or word.startswith("--") or not set(word[1:]) <= set("celuvxs"):
            result.unsupported = True
            return
        if set(word[1:]) & {"x", "v"}:
            result.has_other_commands = True
        if "c" in word[1:]:
            if j + 1 < len(rest):
                _script(rest[j + 1], result, depth)
            return
        j += 1
    stdin = j == len(rest) or rest[j:] == ["--"] or any("s" in w[1:] for w in rest[:j])
    if stdin and bodies and len(bodies) == 1:
        _script(bodies[0], result, depth)
    else:
        result.has_other_commands = True


def _ssgrep(args: list[str], result: ShellScan) -> None:
    for j, word in enumerate(args):
        if word.startswith("-"):
            continue
        if word in OPERATIONS:
            result.argvs.append(tuple(args[j:]))
        else:
            result.has_other_commands = True
        return
    result.has_other_commands = True
