"""MCP server command: start the stdio server or install client registrations."""

from __future__ import annotations

import json
import os
import sys
from typing import Annotated, NoReturn

from usecli import Argument, BaseCommand, console
from usecli.cli.core.runtime import is_json_mode, is_quiet

from ssgrep.cli import exit_codes
from ssgrep.services import mcp_install


def _exit_failed(failed: tuple[str, ...], payload: dict[str, object]) -> NoReturn:
    """End a run in which some clients failed, still reporting every client.

    ``failed`` names the clients with an ``error: ...`` status. In JSON mode the
    full ``payload`` accompanies ``"ok": false``; the document is written
    straight to the real stdout (as ``index`` does) because usecli would
    otherwise replace it with its generic exit envelope.
    """
    message = f"registration failed for: {', '.join(failed)}"
    if is_json_mode():
        stdout = sys.__stdout__
        assert stdout is not None
        document = {
            "ok": False,
            "condition": "client_registration_failed",
            "message": message,
            "command": None,
            **payload,
        }
        stdout.write(json.dumps(document) + "\n")
        stdout.flush()
        os._exit(exit_codes.INTERNAL_FAILURE)
    print(message, file=sys.stderr)
    raise SystemExit(exit_codes.INTERNAL_FAILURE)


def _registrations_json(
    results: tuple[mcp_install.InstallResult, ...],
) -> dict[str, dict[str, str | None]]:
    return {
        name: {"status": status, "config": str(config) if config else None}
        for name, status, config in results
    }


def _display_registrations(results: tuple[mcp_install.InstallResult, ...]) -> None:
    """Print styled per-client registration results with config locations."""
    for name, status, config in results:
        styled = _status_style(status)
        location = f" ({config})" if config else ""
        console.print(f"  [cyan]{name:<10}[/cyan] {styled}{location}")


def _status_style(status: str) -> str:
    if status.startswith("error:"):
        return f"[red]{status}[/red]"
    if status.startswith("skipped:"):
        return f"[gray]{status}[/gray]"
    if status == "already_installed":
        return "[gray]already installed[/gray]"
    if status == "updated":
        return "[yellow]updated[/yellow]"
    if status in ("installed",):
        return "[green]installed[/green]"
    return status


class McpCommand(BaseCommand):
    def visible(self) -> bool:
        return True

    def signature(self) -> str:
        return "mcp"

    def description(self) -> str:
        return "Start the MCP stdio server (or install client registrations)"

    def handle(
        self,
        action: Annotated[
            str | None,
            Argument(
                help="`install` writes the ssgrep registration into every supported MCP client"
            ),
        ] = None,
        clients: Annotated[
            list[str] | None,
            Argument(help="Optional subset: claude, cursor, zed, codex, opencode, omp"),
        ] = None,
    ) -> dict[str, object] | None:
        if action is not None and action != "install":
            message = f"unknown mcp action: {action!r} (only `ssgrep mcp install` is available)"
            if is_json_mode():
                sys.stderr.write(f"{message}\n")
            else:
                print(message, file=sys.stderr)
            raise SystemExit(exit_codes.USAGE_ERROR)
        if action == "install":
            return self._install(clients)
        self._serve()

    def _install(self, clients: list[str] | None) -> dict[str, object] | None:
        """Register ssgrep with each supported MCP client, independently.

        Every client is attempted and reported; one ending in ``error: ...``
        makes the command exit nonzero (``skipped`` does not).
        """
        try:
            results = mcp_install.install_mcp_registrations(tuple(clients or ()))
        except ValueError as error:
            print(f"Usage error: {error}", file=sys.stderr)
            raise SystemExit(exit_codes.USAGE_ERROR) from error
        payload: dict[str, object] = {"clients": _registrations_json(results)}
        if not (is_json_mode() or is_quiet()):
            _display_registrations(results)
        failed = mcp_install.failed_clients(results)
        if failed:
            _exit_failed(failed, payload)
        return payload if is_json_mode() else None

    def _serve(self) -> None:
        # usecli loads this file as top-level ``mcp_command``. Defensively
        # evict any stray ``mcp`` shadow (one that is not the real package)
        # before FastMCP imports the actual ``mcp`` package.
        if sys.modules.get("mcp") is not None and not hasattr(sys.modules["mcp"], "types"):
            del sys.modules["mcp"]
        from ssgrep.services.mcp_server import serve

        serve()
