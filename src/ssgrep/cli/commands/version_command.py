"""Print the installed ssgrep version."""

from __future__ import annotations

from importlib.metadata import version

from usecli import BaseCommand
from usecli.cli.core.runtime import is_json_mode


class VersionCommand(BaseCommand):
    def visible(self) -> bool:
        return True

    def signature(self) -> str:
        return "version"

    def description(self) -> str:
        return "Display the installed ssgrep version"

    def handle(self) -> dict[str, object] | None:
        installed = version("ssgrep")
        if is_json_mode():
            return {"name": "ssgrep", "version": installed}
        print(f"ssgrep {installed}")
        return None
