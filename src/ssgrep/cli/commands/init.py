"""Init command — first-run setup for ssgrep."""

from __future__ import annotations

import json
import os
import sys

from usecli.cli.core.base_command import BaseCommand
from usecli.cli.core.runtime import is_json_mode, is_quiet

from ssgrep import api
from ssgrep.cli import exit_codes
from ssgrep.cli.commands import resolve_project_dir, to_jsonable
from ssgrep.cli.commands.hooks import (
    _install_claude_code_templates,
    _install_hook,
    _settings_path,
)
from ssgrep.cli.commands.index import _attach_census, _fail
from ssgrep.types import IndexNotReadyError, IndexStats, RebuildWouldShrinkError


class InitCommand(BaseCommand):
    """Initialize ssgrep: create index, download model, install hooks."""

    def visible(self) -> bool:
        """Show this command in help."""
        return True

    def signature(self) -> str:
        """Command name."""
        return "init"

    def description(self) -> str:
        """Command description."""
        return "Initialize ssgrep: create index, download model, install hooks"

    def handle(self, project_dir: str = ".") -> object:
        """Set up ssgrep from scratch for the resolved project.

        Safe to re-run: indexing is incremental and hook installation
        de-duplicates on the hook description.

        Installation is resilient: template and hook setup happen before
        indexing, so a failed index does not prevent setup. All errors surface
        and the exit code reflects the worst failure, but partial success is
        always reported.
        """
        project_path = resolve_project_dir(project_dir)
        suppress = is_quiet() or is_json_mode()

        if not suppress:
            print(f"Initializing ssgrep for {project_path}...", file=sys.stderr)
            print("Downloading embedding model (first run only)...", file=sys.stderr)

        # Step 1: install Claude Code skill and command templates. Best-effort:
        # failures must not fail init. Handles idempotency and user edits.
        # Runs before indexing since it has no dependencies on the index.
        templates_result: dict | None = None
        try:
            templates_result = _install_claude_code_templates()
        except Exception as error:
            if not is_quiet():
                print(
                    f"Warning: could not install Claude Code templates: {error}",
                    file=sys.stderr,
                )

        # Step 2: install the SessionEnd hook. Best-effort: a hook failure
        # must not fail init (D13: hooks are hints, reconciliation is
        # authoritative). Runs before indexing since it has no dependencies.
        # hook_changed is None when installation failed, so the summary never
        # claims success.
        hook_changed: bool | None = None
        try:
            result = _install_hook(project_path)
            hook_changed = result.get("changed")
        except Exception as error:
            if not is_quiet():
                settings_path = _settings_path()
                print(
                    f"Warning: could not install Claude Code hook: {error}",
                    file=sys.stderr,
                )
                print(
                    "To install manually, add a SessionEnd hook running "
                    f"`ssgrep hooks enqueue --project-dir {project_path}` to "
                    f"{settings_path}.",
                    file=sys.stderr,
                )

        # Steps 3-5: api.index() creates .ssgrep/, initializes SQLite, pulls
        # the embedding model on first encode, and runs the first full index.
        # This step's failure does NOT prevent setup above; errors are reported
        # but do not stop the command from returning successfully if setup
        # succeeded. The exit code will still reflect any indexing failure.
        stats: IndexStats | None = None
        indexing_error: Exception | None = None
        try:
            stats = api.index(project_path, rebuild=False, quiet=suppress)
        except RebuildWouldShrinkError as error:
            # rebuild=False does NOT keep init off the rebuild path: indexer
            # computes `rebuild or needs_rebuild(...)`, so after any release
            # that bumps SCHEMA_VERSION or changes the embedding model, plain
            # `ssgrep init` takes the gated path and can be refused here.
            #
            # Routed through index.py's own handler so the identical condition
            # produces the identical outcome from both commands: USAGE_ERROR
            # (nothing broke, nothing was written), and the same
            # `{ok: false, condition: "rebuild_would_shrink", ...}` envelope in
            # JSON mode.
            _attach_census(error, project_path)
            _fail(error, exit_codes.USAGE_ERROR)
        except IndexNotReadyError as error:
            # This is a recoverable indexing error — report it but allow
            # setup above to have already run. The error will still exit
            # non-zero after reporting the summary.
            indexing_error = error
            if not suppress:
                print(error, file=sys.stderr)
        except Exception as error:
            # Other indexing failures — also recoverable for setup purposes.
            indexing_error = error
            if not suppress:
                print(f"Initialization failed during indexing: {error}", file=sys.stderr)

                # Suppress generic remedy suggestion when error carries its own
                carries_remedy = getattr(error, "command", None) is not None

                if not carries_remedy:
                    msg = "Run `ssgrep init` again, or `ssgrep index --rebuild` for full rebuild."
                    print(msg, file=sys.stderr)

        if is_json_mode():
            # JSON mode: indexing errors still take the exit code path below
            # after we return the stats (or None if indexing failed).
            if stats:
                return to_jsonable(stats)
            elif indexing_error:
                # Preserve the original error-handling path for JSON mode
                if isinstance(indexing_error, IndexNotReadyError):
                    error_doc = {
                        "ok": False,
                        "condition": indexing_error.condition,
                        "message": str(indexing_error),
                        "command": indexing_error.command,
                    }
                else:
                    # Any other indexing failure (e.g. IndexNotFoundError, or
                    # an exception type usecli's JSON wrapper does not catch)
                    # gets the same structured envelope instead of being
                    # allowed to escape as a raw traceback with empty stdout.
                    # SearchException subclasses carry condition/command;
                    # anything else falls back to a generic condition.
                    error_doc = {
                        "ok": False,
                        "condition": getattr(indexing_error, "condition", None) or "index_error",
                        "message": str(indexing_error),
                        "command": getattr(indexing_error, "command", None) or "ssgrep init",
                    }
                sys.__stdout__.write(json.dumps(error_doc) + "\n")  # type: ignore
                sys.__stdout__.flush()  # type: ignore
                os._exit(exit_codes.INTERNAL_FAILURE)

        if not suppress:
            if hook_changed is None:
                hook_state = "NOT installed (see warning above)"
            else:
                hook_state = "installed" if hook_changed else "already installed"

            # Build a summary of template installation
            template_states = []
            if templates_result:
                if templates_result.get("skill") == "installed":
                    template_states.append("skill installed")
                elif templates_result.get("skill") == "user_modified":
                    template_states.append("skill user-customized (not overwritten)")
                if templates_result.get("command") == "installed":
                    template_states.append("command installed")
                elif templates_result.get("command") == "user_modified":
                    template_states.append("command user-customized (not overwritten)")

            template_summary = (
                f"; Claude Code: {', '.join(template_states)}" if template_states else ""
            )

            # Build the summary, handling both successful and failed indexing
            if stats:
                print(
                    f"Done. Indexed {stats.session_count} sessions, "
                    f"{stats.episode_count} episodes, "
                    f"{stats.chunk_count} chunks. "
                    f"SessionEnd hook {hook_state}{template_summary}.",
                    file=sys.stderr,
                )
            elif indexing_error:
                # Indexing failed, but setup succeeded
                print(
                    f"Setup complete (hook {hook_state}{template_summary}), "
                    f"but indexing failed. Run `ssgrep init` again when transcripts are available.",
                    file=sys.stderr,
                )
            else:
                # Should not reach here, but defensive fallback
                print(
                    f"Setup complete (hook {hook_state}{template_summary}).",
                    file=sys.stderr,
                )

        # Exit with error code if indexing failed, even though setup succeeded
        if indexing_error and not isinstance(indexing_error, RebuildWouldShrinkError):
            raise SystemExit(exit_codes.INTERNAL_FAILURE) from indexing_error

        return None
