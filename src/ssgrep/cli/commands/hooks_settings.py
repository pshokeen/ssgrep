"""Claude settings.json hook-entry bookkeeping for `ssgrep hooks`.

Extracted from hooks.py to keep that module near the repo's file-size warn
line. Everything here is pure dict/JSON manipulation of the SessionEnd hook
schema — reading and writing settings, recognizing our own (and legacy)
hook entries, and stripping or rewriting them. Path resolution, template
installation, and the CLI command itself stay in hooks.py.
"""

from __future__ import annotations

import json
from pathlib import Path

#: The stable fragment identifying OUR hook command in settings.json.
HOOK_COMMAND_FRAGMENT = "ssgrep hooks enqueue"

#: Legacy command from pre-D13 versions. We recognize and remove these
#: on install/uninstall to clean up old hooks that violate D13.
_LEGACY_COMMAND_FRAGMENT = "ssgrep index"

#: Legacy description from pre-D13 versions (flat schema). Used to detect
#: old flat-shape entries that need migration.
_LEGACY_DESCRIPTION = "Index session transcripts for ssgrep"
_LEGACY_FLAT_DESCRIPTION = "ssgrep-session-index"


def _read_settings(path: Path) -> dict:
    """Read settings.json, tolerating a missing file but never a corrupt one.

    A missing file yields an empty dict (install creates it). A corrupt file
    raises ValueError — we must never silently clobber the user's settings.
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"{path} is not valid JSON ({error})") from error
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object, got {type(data).__name__}")
    return data


def _write_settings(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def _session_end_hooks(settings: dict) -> list:
    """Return the SessionEnd hook entries list, without mutating settings.

    A missing or malformed 'hooks' / 'hooks.SessionEnd' shape yields an
    empty list rather than materializing placeholder keys into settings —
    merely reading must never change what a later write would persist.
    Callers that need to persist changes go through _set_session_end_hooks,
    which touches settings only on the write path.
    """
    hooks = settings.get("hooks", {})
    if not isinstance(hooks, dict):
        return []
    entries = hooks.get("SessionEnd", [])
    if not isinstance(entries, list):
        return []
    return entries


def _set_session_end_hooks(settings: dict, entries: list) -> None:
    """Materialize `entries` as settings['hooks']['SessionEnd'], in place.

    This is the only place allowed to create the 'hooks' key from scratch —
    called on the write path, only once a write has already been decided.
    """
    hooks = settings.get("hooks")
    if not isinstance(hooks, dict):
        hooks = {}
        settings["hooks"] = hooks
    hooks["SessionEnd"] = entries


def _is_current_command(command: str) -> bool:
    """True if a single hook's command string is ssgrep's current form."""
    return HOOK_COMMAND_FRAGMENT in command


def _is_legacy_command(command: str) -> bool:
    """True if a single hook's command string is ssgrep's legacy (pre-D13) form."""
    return _LEGACY_COMMAND_FRAGMENT in command


def _is_legacy_ssgrep_hook(entry: dict) -> bool:
    """Detect a legacy, flat-shape ssgrep hook entry (pre-D13 or broken form).

    Flat-shape entries (command + description at top level, no "hooks"
    array) are inherently the old, broken schema, so a match here is always
    the *whole* entry — a flat-shape entry can't have sibling hooks the way
    a matcher group can:
    1. Old description "ssgrep-session-index" or
       "Index session transcripts for ssgrep".
    2. A top-level command invoking 'ssgrep index' (D13 violation) or
       'ssgrep hooks enqueue' (right command, wrong — flat — shape).

    Legacy hooks *nested* inside a real matcher group's "hooks" array are
    handled separately, per-hook, by _strip_hooks_from_group — that path
    preserves any sibling hooks that happen to share the group, which
    dropping the whole entry here would not.
    """
    if not isinstance(entry, dict):
        return False

    # Check for flat-shape legacy descriptions (old schema)
    if entry.get("description") in (_LEGACY_DESCRIPTION, _LEGACY_FLAT_DESCRIPTION):
        return True

    # Check if this is a flat-shape entry (has command at top level)
    # This is the ssgrep's old format that needs migration
    if "command" in entry and "hooks" not in entry:
        command = entry.get("command", "")
        if isinstance(command, str) and (
            _is_current_command(command) or _is_legacy_command(command)
        ):
            return True

    return False


def _is_our_hook_matcher_group(entry: dict) -> bool:
    """Check if a matcher group contains our current-schema ssgrep hook."""
    if not isinstance(entry, dict) or "hooks" not in entry:
        return False
    hooks = entry.get("hooks", [])
    if not isinstance(hooks, list):
        return False
    for hook in hooks:
        if isinstance(hook, dict):
            command = hook.get("command", "")
            if isinstance(command, str) and _is_current_command(command):
                return True
    return False


def _strip_hooks_from_group(entry: dict, *, include_current: bool, include_legacy: bool) -> int:
    """Remove ssgrep's own hook dict(s) from a matcher group's "hooks" list, in place.

    This is the surgical primitive both install and uninstall build on: it
    only ever drops individual entries inside entry["hooks"] — every sibling
    hook and every other key on the group dict (notably "matcher") is left
    untouched. `include_current`/`include_legacy` select which of ssgrep's
    own command forms count as "ours" for this call. Returns the number
    removed; leaves entry["hooks"] as the same list object when nothing
    matches.
    """
    hooks = entry.get("hooks", [])
    if not isinstance(hooks, list):
        return 0
    remaining = []
    removed = 0
    for hook in hooks:
        command = hook.get("command") if isinstance(hook, dict) else None
        is_ours = isinstance(command, str) and (
            (include_current and _is_current_command(command))
            or (include_legacy and _is_legacy_command(command))
        )
        if is_ours:
            removed += 1
        else:
            remaining.append(hook)
    if removed:
        entry["hooks"] = remaining
    return removed


def _our_hook_command(entry: dict) -> str:
    """Return ssgrep's own current-schema command within a matcher group.

    Returns "" if the group has no current-schema ssgrep hook.
    """
    hooks = entry.get("hooks", [])
    if not isinstance(hooks, list):
        return ""
    for hook in hooks:
        if isinstance(hook, dict):
            command = hook.get("command", "")
            if isinstance(command, str) and _is_current_command(command):
                return command
    return ""


def _replace_our_hook_command(entry: dict, new_command: str) -> None:
    """Replace ssgrep's own hook command(s) within a matcher group, in place.

    Only the individual hook dict(s) whose command is ssgrep's current form
    are replaced — every sibling hook and every other key on the matcher
    group (notably "matcher") is preserved untouched.
    """
    hooks = entry.get("hooks", [])
    if not isinstance(hooks, list):
        return
    for i, hook in enumerate(hooks):
        if isinstance(hook, dict):
            command = hook.get("command", "")
            if isinstance(command, str) and _is_current_command(command):
                hooks[i] = {"type": "command", "command": new_command}
