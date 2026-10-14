"""``ssgrep metrics setup``: preview, then enable local measurement after consent.

Never enables export, installs a service, or starts a worker. A rejected policy,
a refused confirmation, or a cancelled prompt leaves every file untouched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any

from usecli import BaseCommand, Option
from usecli.cli.core.runtime import is_json_mode

from ssgrep.accounting import policy
from ssgrep.accounting.contracts import PERSONAL_PROFILE
from ssgrep.accounting.policy_rules import PolicyError
from ssgrep.accounting.scope import check_profile
from ssgrep.accounting.service_types import command_for
from ssgrep.cli import exit_codes
from ssgrep.cli.accounting_cli import exit_with, fail, stdin_is_tty, terminal_width
from ssgrep.cli.metrics_json import lifecycle_document
from ssgrep.cli.metrics_render import render_setup_preview, render_setup_result
from ssgrep.cli.savings_render import emit

MAX_POLICY_FILE_BYTES = 256 * 1024


def _load_policy_file(path: str) -> Any:
    try:
        target = Path(path).expanduser()
        with target.open("rb") as handle:
            raw = handle.read(MAX_POLICY_FILE_BYTES + 1)
    except OSError as error:
        raise PolicyError(
            "invalid_policy", f"Policy file cannot be read: {error.strerror}."
        ) from error
    if len(raw) > MAX_POLICY_FILE_BYTES:
        raise PolicyError("invalid_policy", "Policy file is larger than 256 KiB.")
    try:
        return json.loads(raw)
    except ValueError as error:
        raise PolicyError("invalid_policy", "Policy file is not valid JSON.") from error


class MetricsSetupCommand(BaseCommand):
    def visible(self) -> bool:
        return True

    def signature(self) -> str:
        return "metrics setup"

    def description(self) -> str:
        return "Preview approved roots, fields and retention, then enable local measurement"

    def handle(
        self,
        policy_file: Annotated[
            str | None, Option("--policy", help="Managed policy JSON file to apply and preview")
        ] = None,
        yes: Annotated[
            bool, Option("--yes", help="Confirm without prompting (required when non-interactive)")
        ] = False,
        profile: Annotated[
            str, Option("--profile", help="Accounting profile (default personal-local)")
        ] = PERSONAL_PROFILE,
    ) -> dict[str, object] | None:
        interactive = stdin_is_tty() and not is_json_mode()
        if not yes and not interactive:
            exit_with(
                "confirmation_required",
                "Setup needs explicit consent: re-run with --yes (non-interactive). "
                "Nothing was changed.",
                command_for(profile, "metrics setup --yes"),
                exit_codes.USAGE_ERROR,
            )
        try:
            check_profile(profile)
            raw = _load_policy_file(policy_file) if policy_file else None
            previewed = policy.preview_enable(profile, raw)
            already = policy.load_profile_policy(profile).collection_enabled
        except PolicyError as error:
            fail(error)
        source = previewed.policy_source if policy_file else "local defaults"
        if not is_json_mode():
            emit(
                render_setup_preview(
                    previewed, width=terminal_width(), already_enabled=already, source=source
                )
            )
        if not yes:
            print(file=sys.stdout)
            try:
                answer = input("Enable local measurement with these settings? [y/N] ")
            except EOFError:
                answer = ""
            if answer.strip().lower() not in {"y", "yes"}:
                print("Cancelled. Nothing was changed.", file=sys.stderr)
                return None
        try:
            enabled = policy.enable_collection(profile, raw)
        except PolicyError as error:
            fail(error)
        if is_json_mode():
            return lifecycle_document(
                "enabled", enabled, profile, "metrics collect", "Run the first collection."
            )
        print(file=sys.stdout)
        emit(render_setup_result(enabled, terminal_width()))
        return None
