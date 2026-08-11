"""Version command for ssgrep."""

from importlib.metadata import version

from usecli.cli.core.base_command import BaseCommand


class VersionCommand(BaseCommand):
    """Display ssgrep version."""

    def visible(self) -> bool:
        """Show this command in help."""
        return True

    def signature(self) -> str:
        """Command name."""
        return "version"

    def description(self) -> str:
        """Command description."""
        return "Display ssgrep version"

    def handle(self) -> None:
        """Execute the command."""
        print(f"ssgrep {version('ssgrep')}")
