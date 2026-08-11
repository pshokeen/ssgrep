"""MCP command for ssgrep."""

from __future__ import annotations

import sys

from usecli.cli.core.base_command import BaseCommand


class McpCommand(BaseCommand):
    """Start the MCP stdio server."""

    def visible(self) -> bool:
        """Show this command in help."""
        return True

    def signature(self) -> str:
        """Command name."""
        return "mcp"

    def description(self) -> str:
        """Command description."""
        return "Start the MCP stdio server"

    def handle(self, project_dir: str = ".") -> None:
        """Run the stdio server until the client disconnects.

        The fastmcp SDK is imported lazily when get_mcp_server() is called,
        so the search, show, and status code paths never pay for it. Project
        scope resolves from --project-dir when given (MCP client configs
        universally support command/args but not always cwd, so a server
        registered from a config file needs an argument-shaped way to name
        its project); the default "." preserves the original behavior of
        resolving from the server process's working directory.
        """
        # usecli 0.1.75 loads command files as top-level modules named by file
        # stem, so this file is sys.modules["mcp"] — shadowing the real `mcp`
        # package that fastmcp imports. Evict the shadow first; the command
        # instance is already constructed and no longer needs the entry.
        if sys.modules.get("mcp") is not None and not hasattr(sys.modules["mcp"], "types"):
            del sys.modules["mcp"]
        try:
            from ssgrep.mcp_server import get_mcp_server, set_project_dir
        except ImportError:
            print(
                "MCP support dependencies are not available. "
                "Ensure fastmcp is installed: pip install fastmcp",
                file=sys.stderr,
            )
            # SystemExit, not typer.Exit: usecli 0.1.75's invoke() maps every
            # click Exit to sys.exit(0), swallowing the requested code.
            raise SystemExit(1) from None
        if project_dir != ".":
            from ssgrep.cli.commands import resolve_project_dir

            set_project_dir(resolve_project_dir(project_dir))
        server = get_mcp_server()
        server.run()
