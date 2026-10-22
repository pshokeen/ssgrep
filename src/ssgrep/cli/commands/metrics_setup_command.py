"""Set up local measurement, or separately approve/revoke organization export.

Default setup remains local-only. Export approval previews every permitted category and
requires explicit consent; approval never sends or starts a service.
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


def _configure_export(profile: str, destination: str | None, *, disable: bool, yes: bool):
    from ssgrep.accounting.export import consent
    from ssgrep.accounting.ledger.ledger_contracts import ProfilePolicy, Signal
    from ssgrep.accounting.privacy import policy
    from ssgrep.accounting.views.types import NextAction, check_profile
    from ssgrep.cli import exit_codes
    from ssgrep.cli.accounting_cli import envelope, exit_with, stdin_is_tty, terminal_width
    from ssgrep.cli.export_json import disable_document, enable_document
    from ssgrep.cli.savings_render import emit

    check_profile(profile)
    if disable:
        current = policy.read_policy_file().profiles.get(profile, ProfilePolicy(profile_id=profile))
        reports = [
            disable_document(
                consent.apply_disable(profile, name, Signal(signal)),
                profile,
                current.policy_version,
            )
            for name, signal in current.enabled_exports
        ]
        if is_json_mode():
            return envelope("disabled", {"profile": profile, "reports": reports})
        print("Organization export disabled. Local history and queued data were retained.")
        print("Already accepted remote data cannot be recalled by this command.")
        return None

    from ssgrep.cli.commands.metrics_export_command import _target
    from ssgrep.cli.export_render import render_enable_preview

    selected, signals = _target(profile, destination, preview=True)
    plans = [consent.plan_enable(profile, selected, signal) for signal in signals]
    if not yes and (not stdin_is_tty() or is_json_mode()):
        exit_with(
            "confirmation_required",
            "Export approval needs explicit consent. Re-run with --yes after reviewing "
            "the destination policy; nothing was changed.",
            consent.target_command(profile, selected, signals[0], "--enable --yes"),
            exit_codes.USAGE_ERROR,
        )
    if not is_json_mode():
        for plan in plans:
            emit(render_enable_preview(plan, terminal_width()))
    if not yes:
        try:
            answer = input(f"Approve all listed data for {selected}? [y/N] ")
        except EOFError:
            answer = ""
        if answer.strip().lower() not in {"y", "yes"}:
            print("Cancelled. Nothing was changed.", file=sys.stderr)
            return None
    approvals = []
    changed = False
    for plan in plans:
        enabled = consent.apply_enable(profile, selected, plan.signal)
        changed |= enabled
        approvals.append(enable_document(plan, enabled))
    command = consent.target_command(profile, selected, signals[0])
    if is_json_mode():
        return envelope(
            "enabled",
            {
                "profile": profile,
                "destination": selected,
                "signals": [signal.value for signal in signals],
                "changed": changed,
                "approvals": approvals,
            },
            next_actions=(NextAction(command, "Send approved data."),),
        )
    print(f"Approved {selected}: {', '.join(signal.value for signal in signals)}. Nothing sent.")
    print(f"Next: {command}")
    return None


class MetricsSetupCommand(BaseCommand):
    def visible(self) -> bool:
        from ssgrep.cli.commands import accounting_command_visible

        return accounting_command_visible()

    def signature(self) -> str:
        return "savings setup"

    def description(self) -> str:
        return "Set up local measurement or explicitly approve organization export"

    def handle(
        self,
        policy_file: Annotated[
            str | None, Option("--policy", "-P", help="Managed policy JSON to apply")
        ] = None,
        yes: Annotated[
            bool, Option("--yes", "-y", help="Confirm the displayed collection or export consent")
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
        export_to: Annotated[
            str | None,
            Option("--export-to", "-x", help="Approve all permitted data for this destination"),
        ] = None,
        no_export: Annotated[
            bool,
            Option("--no-export", "-X", help="Revoke organization exports; keep local history"),
        ] = False,
    ) -> dict[str, object] | None:
        from ssgrep.accounting.privacy import policy
        from ssgrep.accounting.privacy.policy import PolicyError
        from ssgrep.accounting.views.types import check_profile, command_for
        from ssgrep.cli import exit_codes
        from ssgrep.cli.accounting_cli import exit_with, fail, stdin_is_tty, terminal_width

        if export_to is not None or no_export:
            if (
                (export_to is not None and no_export)
                or policy_file is not None
                or evidence_cache
                or no_evidence_cache
                or cache_days != 1
                or cache_bytes != 4194304
            ):
                exit_with(
                    "conflicting_setup",
                    "Choose --export-to DEST or --no-export. Configure policy and cache "
                    "settings in a separate setup call.",
                    command_for(profile, "savings setup --help"),
                    exit_codes.USAGE_ERROR,
                )
            try:
                return _configure_export(profile, export_to, disable=no_export, yes=yes)
            except PolicyError as error:
                fail(error)

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
