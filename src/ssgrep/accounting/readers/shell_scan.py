"""Recognize ssgrep CLI invocations in shell text by parsing, never executing.

Only a word at *command position* counts: after env assignments, a small set of
wrappers (``env``, ``time``, ``uv run``, ``python -m``, ``xargs``, ``find -exec``,
``bash -c``, ``eval``) and shell keywords. Quoted strings, comments, heredoc
bodies, arguments (``echo ssgrep search``, ``grep 'ssgrep search'``) and file
contents are never command position. Text can show at most *that an invocation
exists*; whether it runs zero, one or many times is not knowable from text, so
loops, conditionals, wrappers and multi-occurrence text are ``unknown`` cardinality.
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass, field

MAX_COMMAND_CHARS = 256 * 1024
MAX_DEPTH = 3
OPERATIONS = frozenset({"search", "show", "note"})
PUNCT = frozenset("();<>|&")
ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
HEREDOC = re.compile(r"<<(-?)[ \t]*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\2")
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
XARGS_VALUE_OPTIONS = frozenset(
    {"-n", "-I", "-P", "-L", "-d", "-E", "-s", "-a", "-l", "-i", "-j", "-N", "-S"}
)
ENV_VALUE_OPTIONS = frozenset({"-u", "--unset", "-C", "-S"})
EXEC_FLAGS = frozenset({"-exec", "-execdir", "-ok", "-okdir"})


class ShellParseError(ValueError):
    """The text could not be tokenized (for example an unterminated quote)."""


@dataclass
class ShellScan:
    argvs: list[tuple[str, ...]] = field(default_factory=list)
    uncertain: bool = False
    error: str | None = None

    @property
    def operations(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(argv[0] for argv in self.argvs))


def logical_lines(text: str) -> list[str]:
    """Split on newlines outside quotes; drop comments and heredoc bodies."""
    lines: list[str] = []
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
                pending.append((found.group(3), bool(found.group(1))))
                buf.append(found.group(0))
                i = found.end()
                continue
        elif c == "\n":
            lines.append("".join(buf))
            buf = []
            i = _skip_heredocs(text, i + 1, pending)
            pending = []
            continue
        buf.append(c)
        i += 1
    if quote:
        raise ShellParseError("unterminated quote")
    if buf:
        lines.append("".join(buf))
    return lines


def _skip_heredocs(text: str, i: int, pending: list[tuple[str, bool]]) -> int:
    for delimiter, strip in pending:
        while i < len(text):
            end = text.find("\n", i)
            end = len(text) if end < 0 else end
            line = text[i:end]
            i = end + 1
            if (line.lstrip("\t") if strip else line) == delimiter:
                break
    return i


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
        for line in logical_lines(text):
            for segment in _words(line):
                _command(segment, result, depth)
    except (ShellParseError, ValueError):
        result.argvs.clear()
        result.error = "unparseable_shell"
    return result


def scan_words(words: list[str], *, depth: int = 0) -> ShellScan:
    """Analyze an already-tokenized argv (for example ``["bash", "-lc", "..."]``)."""
    result = ShellScan()
    _command(words, result, depth)
    return result


def _script(script: str, result: ShellScan, depth: int) -> None:
    result.uncertain = True
    if depth >= MAX_DEPTH:
        return
    inner = scan_shell(script, depth=depth + 1)
    if inner.error is None:
        result.argvs.extend(inner.argvs)


def _command(words: list[str], result: ShellScan, depth: int) -> None:
    i = 0
    while i < len(words):
        word = words[i]
        if ASSIGN.match(word) or word in SKIP:
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
    elif base == "env":
        j = 0
        while j < len(rest) and (rest[j].startswith("-") or ASSIGN.match(rest[j])):
            j += 2 if rest[j] in ENV_VALUE_OPTIONS else 1
        _command(rest[j:], result, depth)
    elif base in {"uv", "uvx"}:
        _uv(base, rest, result, depth)
    elif PYTHON.match(base):
        if rest[:2] == ["-m", "ssgrep"]:
            _ssgrep(rest[2:], result)
    elif base in {"xargs", "parallel"}:
        result.uncertain = True
        _command(_after_options(rest, XARGS_VALUE_OPTIONS), result, depth)
    elif base == "find":
        _find(rest, result, depth)
    elif base in SHELLS:
        _shell_c(rest, result, depth)
    elif base == "eval" and rest:
        _script(" ".join(rest), result, depth)


def _after_options(words: list[str], value_options: frozenset[str]) -> list[str]:
    j = 0
    while j < len(words) and words[j].startswith("-") and words[j] != "--":
        j += 2 if words[j] in value_options else 1
    return words[j + 1 :] if j < len(words) and words[j] == "--" else words[j:]


def _uv(base: str, rest: list[str], result: ShellScan, depth: int) -> None:
    if base == "uv":
        if not rest or rest[0] != "run":
            return
        rest = rest[1:]
    _command(_after_options(rest, UV_VALUE_OPTIONS), result, depth)


def _find(rest: list[str], result: ShellScan, depth: int) -> None:
    for j, word in enumerate(rest):
        if word in EXEC_FLAGS:
            result.uncertain = True
            command = []
            for item in rest[j + 1 :]:
                if item in {";", "+"}:
                    break
                command.append(item)
            _command(command, result, depth)


def _shell_c(rest: list[str], result: ShellScan, depth: int) -> None:
    for j, word in enumerate(rest):
        if word.startswith("-") and not word.startswith("--") and "c" in word[1:]:
            if j + 1 < len(rest):
                _script(rest[j + 1], result, depth)
            return


def _ssgrep(args: list[str], result: ShellScan) -> None:
    for j, word in enumerate(args):
        if word.startswith("-"):
            continue
        if word in OPERATIONS:
            result.argvs.append(tuple(args[j:]))
        return
