"""Hooks command for ssgrep — install/uninstall the Claude Code SessionEnd hook.

The hook is a shell entry in Claude Code's ``settings.json`` (see
``paths.resolve_claude_dir``, which honours ``CLAUDE_CONFIG_DIR``) under
``hooks.SessionEnd`` that enqueues a work item via ``ssgrep hooks enqueue``
when a session ends. Per D13 the hook is a hint — it never blocks on heavy
work itself; reconciliation on the next explicit index is authoritative.

The hook never calls indexing directly. Instead it enqueues work items to
.ssgrep/workqueue.db, which are drained by explicit `index`, MCP startup,
and the reconciliation process.
"""

from __future__ import annotations

import shlex
import shutil
import sys
from pathlib import Path

from usecli.cli.core.base_command import BaseCommand
from usecli.cli.core.runtime import is_json_mode

from ssgrep import paths
from ssgrep.cli import exit_codes
from ssgrep.cli.commands import resolve_project_dir
from ssgrep.cli.commands.hooks_settings import (  # noqa: F401  (re-exported)
    HOOK_COMMAND_FRAGMENT,
    _is_legacy_ssgrep_hook,
    _is_our_hook_matcher_group,
    _our_hook_command,
    _read_settings,
    _replace_our_hook_command,
    _session_end_hooks,
    _set_session_end_hooks,
    _strip_hooks_from_group,
    _write_settings,
)
from ssgrep.discovery import discover_sessions
from ssgrep.workqueue import WorkQueue

_ACTIONS = ("install", "uninstall", "enqueue")


def _settings_path() -> Path:
    """The settings.json Claude Code actually reads.

    Must go through the same resolver as transcript discovery: a user who
    relocated Claude Code with CLAUDE_CONFIG_DIR and got a hook written to
    ``~/.claude/settings.json`` gets no error and no output at all -- Claude
    Code never reads that file, so the SessionEnd hook simply never fires.
    """
    return paths.resolve_claude_dir() / "settings.json"


def ssgrep_executable() -> str:
    """The path Claude Code should invoke ssgrep by, resolved at install time.

    Not the bare name ``ssgrep``. Claude Code runs a SessionEnd hook through a
    plain shell, and none of the three installs the README documents put an
    ``ssgrep`` executable on the global PATH: ``uvx --from <wheel>`` never
    does, ``uv add`` and a venv ``pip install`` only do while that venv is
    activated, and a hook does not inherit an activation. So the shell got
    ``sh: ssgrep: command not found`` (rc=127) on every session while
    ``ssgrep init`` had already printed "SessionEnd hook installed." The
    README knows this -- it tells MCP users to point ``command`` at the venv's
    binary directly -- but this writer did not.

    Resolution order, most stable first:

    1. The console script alongside the running interpreter. In every venv
       and uv install this is the buyer's own ``ssgrep``, and it keeps
       working with no PATH and no activation.
    2. Whatever ``ssgrep`` PATH resolves to right now, for an install laid
       out some other way.
    3. The bare name, which at least behaves as before rather than writing a
       path that does not exist. install() warns when it comes to this.
    """
    candidate = Path(sys.executable).parent / "ssgrep"
    if candidate.exists():
        return str(candidate)
    found = shutil.which("ssgrep")
    return found or "ssgrep"


def _hook_command(project_dir: Path) -> str:
    """Generate the hook shell command.

    The command enqueues work items but never indexes directly.
    SessionEnd passes transcript_path via environment variable if available.

    Every interpolated component is shell-quoted. This is one unquoted string
    stored in settings.json and handed to a shell, so a project under
    ``~/Documents/My Project`` -- ordinary on macOS -- word-split into
    ``Got unexpected extra argument(s) (Project)`` and rc=2 on every session,
    for a hook install that reported success. scope_report.remedy_command()
    already learned this lesson; this site had not. shlex.quote is a no-op for
    every space-free path, so existing hooks are unchanged to the byte.
    """
    return (
        f"{shlex.quote(ssgrep_executable())} hooks enqueue "
        f"--project-dir {shlex.quote(str(project_dir))}"
    )


def _build_hook_matcher_group(project_dir: Path) -> dict:
    """Build the real Claude Code hook matcher group structure.

    Schema:
    {
        "hooks": [
            {"type": "command", "command": "..."}
        ]
    }
    """
    command = _hook_command(project_dir)
    return {
        "hooks": [
            {
                "type": "command",
                "command": command,
            }
        ]
    }


def _install_hook(project_dir: Path) -> dict:
    settings_path = _settings_path()
    settings = _read_settings(settings_path)
    entries = _session_end_hooks(settings)

    # First pass: strip legacy ssgrep entries that violate D13. This
    # operates within shared matcher groups (via _strip_hooks_from_group) so
    # sibling hooks and group-level keys like "matcher" survive even when a
    # legacy hook happens to be co-located with someone else's hook. A
    # group's current-schema ssgrep hook, if any, is left alone here — it is
    # updated in place below, never dropped and rebuilt.
    kept = []
    removed_legacy = False

    for entry in entries:
        if not isinstance(entry, dict):
            kept.append(entry)
            continue

        if "hooks" not in entry:
            # Flat-shape entry: it IS a single hook, not a container.
            if _is_legacy_ssgrep_hook(entry):
                removed_legacy = True
            else:
                kept.append(entry)
            continue

        n_removed = _strip_hooks_from_group(entry, include_current=False, include_legacy=True)
        if n_removed:
            removed_legacy = True
            if not entry.get("hooks"):
                continue  # group emptied out entirely — drop it
        kept.append(entry)

    new_command = _hook_command(project_dir)
    found_current = any(
        isinstance(entry, dict) and _is_our_hook_matcher_group(entry) for entry in kept
    )

    if found_current:
        # Update ssgrep's own command in place, preserving sibling hooks and
        # every other key on the group (notably "matcher") — never replace
        # the group wholesale.
        changed = False
        for entry in kept:
            if _is_our_hook_matcher_group(entry):
                changed = _our_hook_command(entry) != new_command
                if changed:
                    _replace_our_hook_command(entry, new_command)
                break
        if changed or removed_legacy:
            _set_session_end_hooks(settings, kept)
            _write_settings(settings_path, settings)
        return {"installed": True, "changed": changed or removed_legacy}
    else:
        # No current hook found anywhere — append a brand new matcher group.
        kept.append(_build_hook_matcher_group(project_dir))
        _set_session_end_hooks(settings, kept)
        _write_settings(settings_path, settings)
        return {"installed": True, "changed": True}


def _substitute_ssgrep_placeholder(content: str) -> str:
    """Substitute SSGREP_BIN_PLACEHOLDER with the resolved executable path."""
    return content.replace("SSGREP_BIN_PLACEHOLDER", ssgrep_executable())


def _enqueue_sessions(project_dir: Path) -> dict:
    """Enqueue pending work items for all sessions in the project.

    This is called by the SessionEnd hook. It discovers all sessions for the
    project and enqueues them to the work queue. Enqueueing is crash-safe and
    deduplicating, and never raises — failures are silently handled so the
    hook does not break the user's session exit.

    Returns a dict with enqueueing results.
    """
    try:
        # Discover at the index's PERSISTED scope, not project_dir: for a
        # --scope-built index, enqueueing project_dir-scoped sessions would
        # inject out-of-scope transcripts through the drain path -- the
        # scope-re-derivation consumer class the 2026-08-07 blind review
        # told us to hunt in "hooks/MCP paths". Best-effort: any failure to
        # read the persisted scope falls back to the legacy default.
        index_dir = project_dir / ".ssgrep"
        try:
            from ssgrep import store as store_mod
            from ssgrep.indexer_support import read_persisted_scope

            persisted = read_persisted_scope(
                store_mod.GenerationalStore(index_dir).get_index_path()
            )
        except Exception:
            persisted = None
        sessions = discover_sessions(project_dir, scope=persisted)
        if not sessions:
            # No sessions found — this is normal on first run
            return {"enqueued": 0, "error": None}
        queue = WorkQueue(index_dir)
        queue.open()

        enqueued_count = 0
        try:
            for session in sessions:
                # Enqueue with the session file path and session_id
                queue.enqueue(
                    transcript_path=str(session.path),
                    session_id=session.session_id,
                    reason="SessionEnd hook",
                )
                enqueued_count += 1
        finally:
            queue.close()

        return {"enqueued": enqueued_count, "error": None}
    except Exception as error:
        # Never raise — silently handle errors so the hook does not break
        # the user's session exit. Return error info for diagnostics if needed.
        return {"enqueued": 0, "error": str(error)}


def _template_hash(presub_content: str) -> str:
    """SHA-256 of the PRE-substitution template body, newline-normalized."""
    import hashlib

    return hashlib.sha256(presub_content.rstrip("\n").encode()).hexdigest()


_TEMPLATE_MARKER_RE = None  # compiled lazily; see _template_marker_re()


def _template_marker_re():
    global _TEMPLATE_MARKER_RE
    if _TEMPLATE_MARKER_RE is None:
        import re

        _TEMPLATE_MARKER_RE = re.compile(
            r"\n?<!-- ssgrep-template: hash=([0-9a-f]{64}) bin=(.*?) -->\n?"
        )
    return _TEMPLATE_MARKER_RE


def _render_template(presub_content: str) -> str:
    """The content actually installed: substituted body plus a version marker.

    The marker records (a) the hash of the pre-substitution template this file
    was generated from and (b) the executable path substituted in, so a later
    install can tell apart the three cases that used to collapse into
    "user_modified": a shipped-template upgrade, a moved-venv path change,
    and a genuine user edit.
    """
    executable = ssgrep_executable()
    body = presub_content.replace("SSGREP_BIN_PLACEHOLDER", executable)
    marker = f"<!-- ssgrep-template: hash={_template_hash(presub_content)} bin={executable} -->"
    return body.rstrip("\n") + "\n\n" + marker + "\n"


def _install_template_file(template_path: Path, dest_path: Path) -> str:
    """Install one template, telling upgrades and path changes from user edits.

    Decision ladder:
    1. Missing dest: install → "installed".
    2. Byte-identical to the current rendering → "already_installed".
    3. Dest carries a marker: un-substitute the recorded executable path and
       hash the result. If it matches the marker's hash the file is exactly
       what some ssgrep version installed (shipped-template upgrade or
       moved-venv path change) → overwrite with the current rendering →
       "installed". A mismatch is a genuine user edit → "user_modified",
       never overwritten.
    4. No marker (pre-marker install): raw-byte fallback — if it matches the
       current template substituted with the current path, adopt the marker
       ("already_installed"); anything else is indistinguishable from a user
       edit and is protected ("user_modified").
    """
    presub = template_path.read_text()
    rendered = _render_template(presub)
    if not dest_path.exists():
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text(rendered)
        return "installed"

    existing = dest_path.read_text()
    if existing == rendered:
        return "already_installed"

    match = _template_marker_re().search(existing)
    if match:
        recorded_hash, recorded_bin = match.groups()
        body = _template_marker_re().sub("\n", existing).rstrip("\n")
        unsubstituted = body.replace(recorded_bin, "SSGREP_BIN_PLACEHOLDER")
        if _template_hash(unsubstituted) == recorded_hash:
            dest_path.write_text(rendered)
            return "installed"
        return "user_modified"

    # Pre-marker install: only an exact match against the current substituted
    # template proves it is ours; adopt the marker so future upgrades work.
    if existing == presub.replace("SSGREP_BIN_PLACEHOLDER", ssgrep_executable()):
        dest_path.write_text(rendered)
        return "already_installed"
    return "user_modified"


def _install_claude_code_templates() -> dict[str, str | None]:
    """Install Claude Code skill and command templates into the user's claude config.

    Copies from the packaged templates in src/ssgrep/templates/claude/ into
    the user's CLAUDE_CONFIG_DIR (or ~/.claude/). Idempotent and non-destructive:
    re-running does not duplicate or corrupt anything, and never overwrites
    files the user has edited (see _install_template_file for how upgrades and
    moved-venv path changes are told apart from user edits).

    Returns:
        A dict with "skill" and "command" keys, each having one of:
        - "installed": File was newly installed (or upgraded/re-pathed)
        - "already_installed": File already matched the current rendering
        - "user_modified": File was edited by the user (not overwritten)
        - None: Template file could not be found
    """
    claude_dir = paths.resolve_claude_dir()
    results: dict[str, str | None] = {"skill": None, "command": None}

    # Template sources live inside the installed package.
    templates_root = Path(__file__).parent.parent.parent / "templates" / "claude"
    installs = {
        "skill": (
            templates_root / "skills" / "ssgrep" / "SKILL.md",
            claude_dir / "skills" / "ssgrep" / "SKILL.md",
        ),
        "command": (
            templates_root / "commands" / "ssgrep" / "search.md",
            claude_dir / "commands" / "ssgrep" / "search.md",
        ),
    }
    for key, (template_path, dest_path) in installs.items():
        if template_path.exists():
            results[key] = _install_template_file(template_path, dest_path)

    return results


def _uninstall_hook(project_dir: Path) -> dict:
    settings_path = _settings_path()
    settings = _read_settings(settings_path)
    entries = _session_end_hooks(settings)

    # Strip ssgrep's own hook(s) — current schema and legacy 'ssgrep index'
    # alike — operating within shared matcher groups (via
    # _strip_hooks_from_group) so sibling hooks and group-level keys like
    # "matcher" survive even when ssgrep shares a group with someone else's
    # hook. A group is dropped entirely only once it has no hooks left.
    kept = []
    removed = 0

    for entry in entries:
        if not isinstance(entry, dict):
            kept.append(entry)
            continue

        if "hooks" not in entry:
            # Flat-shape entry: it IS a single hook, not a container.
            if _is_legacy_ssgrep_hook(entry):
                removed += 1
            else:
                kept.append(entry)
            continue

        n_removed = _strip_hooks_from_group(entry, include_current=True, include_legacy=True)
        removed += n_removed
        if not (n_removed and not entry.get("hooks")):
            kept.append(entry)

    if removed:
        # `entries` is only non-trivially populated when settings["hooks"]
        # already holds a real "SessionEnd" list (see _session_end_hooks's
        # read-only contract), so settings["hooks"] is guaranteed to exist
        # here. This whole block is the write path — the only place allowed
        # to mutate `settings`.
        if kept:
            # Keep the SessionEnd array if there are other hooks
            _set_session_end_hooks(settings, kept)
        else:
            # Remove SessionEnd key if it's now empty
            del settings["hooks"]["SessionEnd"]
            # Remove hooks dict if it's now empty
            if not settings["hooks"]:
                del settings["hooks"]

        # If settings is now empty, delete the file to restore pristine state
        if not settings:
            if settings_path.exists():
                settings_path.unlink()
        else:
            _write_settings(settings_path, settings)
    return {"uninstalled": removed > 0, "removed": removed}


class HooksCommand(BaseCommand):
    """Manage Claude Code session hooks."""

    def visible(self) -> bool:
        """Show this command in help."""
        return True

    def signature(self) -> str:
        """Command name."""
        return "hooks"

    def description(self) -> str:
        """Command description."""
        return "Install or remove the Claude Code SessionEnd hook"

    def handle(self, action: str, project_dir: str = ".") -> object:
        """Install, uninstall, or enqueue from the ssgrep SessionEnd hook."""
        project = resolve_project_dir(project_dir)
        settings_path = _settings_path()

        if action not in _ACTIONS:
            message = f"Unknown action: {action}. Use 'install', 'uninstall', or 'enqueue'."
            print(message, file=sys.stderr)
            raise SystemExit(exit_codes.USAGE_ERROR)

        try:
            if action == "install":
                result = _install_hook(project)
            elif action == "uninstall":
                result = _uninstall_hook(project)
            else:  # action == "enqueue"
                result = _enqueue_sessions(project)
        except ValueError as error:
            # Corrupt settings.json — warn, never crash, never clobber.
            # Only applicable for install/uninstall; enqueue handles this internally.
            if is_json_mode():
                raise
            print(f"Cannot {action} hook: {error}", file=sys.stderr)
            print(
                f"Fix or remove {settings_path} and re-run `ssgrep hooks {action}`.",
                file=sys.stderr,
            )
            raise SystemExit(exit_codes.INTERNAL_FAILURE) from error
        except OSError as error:
            if is_json_mode():
                raise
            print(f"Cannot {action} hook: {error}", file=sys.stderr)
            print(
                f"Check permissions on {settings_path.parent} and retry.",
                file=sys.stderr,
            )
            raise SystemExit(exit_codes.INTERNAL_FAILURE) from error

        if is_json_mode():
            if action == "enqueue":
                return {
                    "ok": True,
                    "action": action,
                    **result,
                }
            return {
                "ok": True,
                "action": action,
                "settings_path": str(settings_path),
                "hook_command": _hook_command(project),
                **result,
            }

        if action == "install":
            verb = "Updated" if result["changed"] else "Already installed"
            print(f"{verb} SessionEnd hook in {settings_path}", file=sys.stderr)
            print(f"Hook runs: {_hook_command(project)}", file=sys.stderr)
            if ssgrep_executable() == "ssgrep":
                # Neither the interpreter's own bin/ nor PATH yielded an
                # ssgrep to name, so the hook is a bare command that Claude
                # Code's shell will most likely fail to resolve. Reporting
                # "installed" for a hook that can never run is the defect
                # this warning exists to stop repeating.
                print(
                    "Warning: could not resolve an absolute path to the ssgrep "
                    "executable, so the hook uses the bare name. If `ssgrep` is not "
                    "on PATH for non-interactive shells, this hook will not run; "
                    "`ssgrep index` remains the authoritative way to update the index.",
                    file=sys.stderr,
                )
        elif action == "uninstall":
            if result["uninstalled"]:
                print(f"Removed SessionEnd hook from {settings_path}", file=sys.stderr)
            else:
                print(
                    f"No ssgrep hook found in {settings_path}; nothing to remove.",
                    file=sys.stderr,
                )
        else:  # action == "enqueue"
            if result.get("error"):
                print(f"Enqueue error: {result['error']}", file=sys.stderr)
                # Don't exit with error — the hook must never break session exit
            elif result.get("enqueued", 0) > 0:
                # Success — quiet unless verbose
                pass
        return None
