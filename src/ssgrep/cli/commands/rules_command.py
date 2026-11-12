"""Print ssgrep's operating rules for coding agents."""

from __future__ import annotations

from typing import Annotated

from usecli import BaseCommand, Option
from usecli.cli.core.runtime import is_json_mode

from ssgrep.services import integrations


class RulesCommand(BaseCommand):
    """Print the guidance ssgrep installs into every supported agent runtime."""

    def visible(self) -> bool:
        """Show this command in help."""
        return True

    def signature(self) -> str:
        """Command name."""
        return "rules"

    def description(self) -> str:
        """Command description."""
        return "Print ssgrep's operating rules for coding agents"

    def handle(
        self,
        short: Annotated[
            bool,
            Option("--short", help="Print only the block to paste into AGENTS.md/CLAUDE.md"),
        ] = False,
        apply: Annotated[
            bool,
            Option(
                "--apply",
                help="Install the marked rules block into this repo's CLAUDE.md/AGENTS.md",
            ),
        ] = False,
        remove: Annotated[
            bool,
            Option("--remove", help="Strip the managed rules block from this repo's instructions"),
        ] = False,
    ) -> dict[str, object] | None:
        """Print the same guidance body that `ssgrep init` installs as a skill.

        Deliberately raw markdown, not rich-rendered: this output is meant to be
        pasted into a project's agent instructions and read by agents, so the
        source markdown IS the artifact. That also keeps three contracts exactly
        equal rather than merely similar -- the installed skill body, this text,
        and `--json`'s `full` field.

        Read-only and dependency-free by design: no index, no model, no data
        directory, so it works on a fresh install and inside CI.

        ``--apply``/``--remove`` are the per-repo variant of the same block: they
        edit the current repo's instruction file idempotently and never overwrite
        user edits (status ``user_modified`` means diff and re-apply by hand).
        """
        if apply and remove:
            from ssgrep.cli import exit_codes
            from ssgrep.cli.accounting_cli import exit_with

            exit_with(
                "conflicting_rules_flags",
                "Choose --apply or --remove, not both.",
                "ssgrep rules --help",
                exit_codes.USAGE_ERROR,
            )
        if apply or remove:
            from pathlib import Path

            target = integrations.project_rule_instruction_file(Path.cwd())
            status = (
                integrations.remove_project_rule(target)
                if remove
                else integrations.apply_project_rule(target)
            )
            if is_json_mode():
                return {
                    "action": "removed" if remove else "applied",
                    "file": str(target),
                    "status": status,
                }
            label = "Removed the rules block from" if remove else "Applied the rules block to"
            suffix = (
                " (you have local edits; diff against `ssgrep rules --short` and re-apply)"
                if status == "user_modified"
                else ""
            )
            print(f"{label} {target} [{status}]{suffix}")
            return None
        if is_json_mode():
            # usecli supplies the {"ok": ..., "data": ...} envelope around what
            # handle() returns, so return the payload itself -- wrapping it here
            # would nest a second envelope inside data.
            payload: dict[str, object] = dict(integrations.guidance_document())
            return payload
        print(integrations.render_guidance(short=short), end="")
        return None
