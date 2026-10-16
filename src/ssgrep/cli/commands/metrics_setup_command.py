"""``ssgrep savings setup``: preview, then enable local measurement after consent.

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

from ssgrep.accounting.ledger.contracts import PERSONAL_PROFILE

MAX_POLICY_FILE_BYTES = 256 * 1024


def _load_policy_file(path: str) -> Any:
    from ssgrep.accounting.privacy.policy import PolicyError

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
        from ssgrep.cli.commands import accounting_command_visible

        return accounting_command_visible()

    def signature(self) -> str:
        return "savings setup"

    def description(self) -> str:
        return "Start here: turn on local-only measurement for this machine"

    def handle(
        self,
        policy_file: Annotated[
            str | None, Option("--policy", "-P", help="Managed policy JSON to apply")
        ] = None,
        yes: Annotated[
            bool, Option("--yes", "-y", help="Confirm local measurement without prompting")
        ] = False,
        profile: Annotated[
            str, Option("--profile", help="Accounting profile (default personal-local)")
        ] = PERSONAL_PROFILE,
        evidence_cache: Annotated[
            bool,
            Option(
                "--evidence-cache", "-e", help="Separately allow a bounded local plaintext cache"
            ),
        ] = False,
        no_evidence_cache: Annotated[
            bool,
            Option(
                "--no-evidence-cache", "-E", help="Revoke cache consent and erase cached copies"
            ),
        ] = False,
        cache_days: Annotated[
            int, Option("--cache-days", "-d", help="Plaintext cache retention in days (1–30)")
        ] = 1,
        cache_bytes: Annotated[
            int, Option("--cache-bytes", "-b", help="Maximum plaintext cache size in bytes")
        ] = 4194304,
    ) -> dict[str, object] | None:
        from ssgrep.accounting.privacy import policy
        from ssgrep.accounting.privacy.policy import PolicyError
        from ssgrep.accounting.views.types import check_profile, command_for
        from ssgrep.cli import exit_codes
        from ssgrep.cli.accounting_cli import exit_with, fail, stdin_is_tty, terminal_width

        interactive = stdin_is_tty() and not is_json_mode()
        if not yes and not interactive:
            exit_with(
                "confirmation_required",
                "Setup needs explicit consent: re-run with --yes (non-interactive). "
                "Nothing was changed.",
                command_for(profile, "savings setup --yes"),
                exit_codes.USAGE_ERROR,
            )

        try:
            check_profile(profile)
            if evidence_cache and no_evidence_cache:
                raise PolicyError(
                    "invalid_policy", "Choose --evidence-cache or --no-evidence-cache, not both."
                )
            cache = (
                (cache_days, cache_bytes)
                if evidence_cache
                else (0, 0)
                if no_evidence_cache
                else None
            )
            if evidence_cache and (cache_days <= 0 or cache_bytes <= 0):
                raise PolicyError(
                    "invalid_policy",
                    "Plaintext cache consent requires positive retention and byte cap.",
                )
            raw = _load_policy_file(policy_file) if policy_file else None
            previewed = policy.preview_enable(profile, raw, evidence_cache=cache)
            already = policy.load_profile_policy(profile).collection_enabled
        except PolicyError as error:
            fail(error)
        source = previewed.policy_source if policy_file else "local defaults"
        if not is_json_mode():
            from ssgrep.cli.metrics_render import render_setup_preview
            from ssgrep.cli.savings_render import emit

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
            enabled = policy.enable_collection(profile, raw, evidence_cache=cache)
        except PolicyError as error:
            fail(error)
        if is_json_mode():
            from ssgrep.cli.metrics_json import lifecycle_document

            return lifecycle_document(
                "enabled", enabled, profile, "savings collect", "Run the first collection."
            )
        from ssgrep.cli.metrics_render import render_setup_result

        print(file=sys.stdout)
        emit(render_setup_result(enabled, terminal_width()))
        return None
