"""Corrections for usecli 0.1.78 usage errors on nested commands (pinned exactly).

usecli names a command in its usage block with ``<program> <leaf name>`` and, while
printing a usage error, rewrites the context's ``info_name`` to the program name. For
``savings explain`` that prints ``ssgrep explain`` and, for the ``savings`` group,
``ssgrep ssgrep``. usecli also rebuilds a missing-argument error from its raw message,
which is empty, so the error text ends after ``Invalid value for 'receipt_id':``.
Each patch below fixes one of those at the point usecli builds the text; nothing else
about usecli's error handling, exit codes or ``--json`` documents changes.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from click.exceptions import Exit


def _full_command_name(ctx: Any) -> str:
    """``savings explain`` for the context of ``ssgrep savings explain`` (program dropped)."""
    return " ".join(ctx.command_path.split()[1:])


def install_usecli_fixes() -> None:
    """Patch usecli's usage-error text; called once when the ssgrep binary starts."""
    from usecli.cli.core.base_command import CustomHelpCommand
    from usecli.cli.core.exceptions import UsecliBadParameter, usage

    help_for_root = usage._get_help_text_with_command_name
    leaf_help = CustomHelpCommand.format_help
    bad_parameter_message = UsecliBadParameter.format_message

    def help_text(ctx: Any) -> str:
        if ctx.parent is None:
            return help_for_root(ctx)
        try:
            return ctx.get_help()
        except Exit:
            return ""  # the command already printed its own help

    def format_help(self: Any, ctx: Any, formatter: Any) -> None:
        name = self.name
        self.name = _full_command_name(ctx)
        try:
            leaf_help(self, ctx, formatter)
        finally:
            self.name = name

    def format_message(self: Any) -> str:
        if self.message or self.param is None:
            return bad_parameter_message(self)
        missing = import_module("typer._click.exceptions").MissingParameter
        return missing(ctx=self.ctx, param=self.param).format_message()

    setattr(usage, "_get_help_text_with_command_name", help_text)  # noqa: B010
    setattr(CustomHelpCommand, "format_help", format_help)  # noqa: B010
    setattr(UsecliBadParameter, "format_message", format_message)  # noqa: B010
