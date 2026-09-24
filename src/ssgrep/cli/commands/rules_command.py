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
    ) -> dict[str, object] | None:
        """Print the same guidance body that `ssgrep init` installs as a skill.

        Deliberately raw markdown, not rich-rendered: this output is meant to be
        pasted into a project's agent instructions and read by agents, so the
        source markdown IS the artifact. That also keeps three contracts exactly
        equal rather than merely similar -- the installed skill body, this text,
        and `--json`'s `full` field.

        Read-only and dependency-free by design: no index, no model, no data
        directory, so it works on a fresh install and inside CI.
        """
        if is_json_mode():
            # usecli supplies the {"ok": ..., "data": ...} envelope around what
            # handle() returns, so return the payload itself -- wrapping it here
            # would nest a second envelope inside data.
            payload: dict[str, object] = dict(integrations.guidance_document())
            return payload
        print(integrations.render_guidance(short=short), end="")
        return None
