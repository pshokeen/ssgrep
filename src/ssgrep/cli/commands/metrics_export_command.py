"""Export the latest accounting data: one destination or one private JSON report."""

from __future__ import annotations

import json
import os
import sys
from contextlib import redirect_stdout
from typing import Annotated, cast

from typer.core import TyperCommand
from usecli import Argument, BaseCommand, Option
from usecli.cli.core.runtime import is_json_mode

from ssgrep.accounting.ledger.contracts import PERSONAL_PROFILE
from ssgrep.cli import exit_codes


def _target(profile: str, destination: str | None, *, preview: bool):
    from ssgrep.accounting.export import consent
    from ssgrep.accounting.ledger.ledger_contracts import ProfilePolicy, Signal, valid_profile_id
    from ssgrep.accounting.privacy import policy
    from ssgrep.accounting.privacy.policy import PolicyError
    from ssgrep.accounting.views.types import check_profile, command_for

    check_profile(profile)
    current = policy.read_policy_file().profiles.get(profile, ProfilePolicy(profile_id=profile))
    if destination is None:
        approved = {name for name, _ in current.enabled_exports}
        choices = approved or {item.destination_id for item in current.destinations}
        if len(choices) != 1:
            raise PolicyError(
                "destination_required",
                "Choose a configured destination"
                + (f" ({', '.join(sorted(choices))})" if choices else "")
                + ", or use --output FILE for a private JSON report.",
                command_for(profile, "savings export --help"),
                exit_code=exit_codes.USAGE_ERROR,
            )
        destination = next(iter(choices))
    if not valid_profile_id(destination):
        raise PolicyError(
            "invalid_destination", "Destination must be a configured ID.", exit_code=2
        )
    target = next((d for d in current.destinations if d.destination_id == destination), None)
    if target is None:
        raise PolicyError(
            "unknown_destination",
            "This destination is not in the profile policy.",
            command_for(profile, "savings status"),
            exit_code=exit_codes.USAGE_ERROR,
        )
    if not preview:
        policy.require_not_paused(profile, "export")
    signals = tuple(
        signal
        for signal in target.signals
        if (signal is not Signal.LOGS or target.logs_permitted)
        and (preview or (destination, signal.value) in current.enabled_exports)
    )
    if not signals:
        raise PolicyError(
            "export_not_enabled",
            "Nothing was sent. Approve this destination in setup first.",
            consent.target_command(profile, destination, Signal.METRICS, "--enable"),
            exit_code=exit_codes.INTERNAL_FAILURE,
        )
    for signal in signals:
        consent.require_target(profile, destination, signal)
    return destination, signals


def _preview(profile: str, destination: str, signals: tuple) -> dict[str, object] | None:
    from ssgrep.accounting.export import consent
    from ssgrep.accounting.export.registry import RecordsUnavailable, RegistryCorrupt
    from ssgrep.accounting.views.types import command_for
    from ssgrep.cli.accounting_cli import envelope, exit_with, terminal_width
    from ssgrep.cli.export_json import preview_document
    from ssgrep.cli.export_render import preview_payload_lines, render_preview_header
    from ssgrep.cli.savings_render import emit

    try:
        views = [consent.build_preview(profile, destination, signal) for signal in signals]
    except (RecordsUnavailable, RegistryCorrupt) as error:
        exit_with(
            error.condition,
            "Local ledger state is unavailable. Nothing was changed.",
            command_for(profile, "savings status"),
            exit_codes.INTERNAL_FAILURE,
        )
    except (ValueError, OSError):
        exit_with(
            "preview_unavailable",
            "Cannot build the current candidate. Nothing was changed.",
            command_for(profile, "savings status"),
            exit_codes.INTERNAL_FAILURE,
        )
    approvals = [consent.consent_state(profile, destination, signal) for signal in signals]
    if is_json_mode():
        previews = []
        for view, approval in zip(views, approvals, strict=True):
            item = cast(dict[str, object], preview_document(view)["data"])
            item.update(
                approved=approval.enabled,
                consent_state=approval.state,
                sending_paused=approval.paused,
            )
            previews.append(item)
        return envelope(
            "preview",
            {
                "profile": profile,
                "destination": destination,
                "signals": [signal.value for signal in signals],
                "previews": previews,
                "network": False,
                "state_changed": False,
            },
        )
    for view, approval in zip(views, approvals, strict=True):
        emit(render_preview_header(view, terminal_width()))
        label = "approved" if approval.enabled else "not approved"
        print(f"Approval: {label}; current status: {approval.state}.")
        for line in preview_payload_lines(view):
            print(line)
    return None


def _send(profile: str, destination: str, signals: tuple) -> dict[str, object] | None:
    from ssgrep.accounting.export import consent, delivery
    from ssgrep.accounting.views.types import NextAction, command_for
    from ssgrep.cli.accounting_cli import envelope, exit_with
    from ssgrep.cli.commands import to_jsonable

    cycles = [delivery.run_destination(profile, destination, signal) for signal in signals]
    verdicts = [consent.classify_cycle(cycle, profile) for cycle in cycles]
    failures = [verdict for verdict in verdicts if not verdict.ok]
    data: dict[str, object] = {
        "profile": profile,
        "destination": destination,
        "signals": [signal.value for signal in signals],
        "cycles": [to_jsonable(cycle) for cycle in cycles],
    }
    warnings = tuple(warning for cycle in cycles for warning in cycle.warnings)
    actions = tuple(NextAction(v.next_command, "Next step.") for v in verdicts if v.next_command)
    result = envelope(
        "partial" if failures else "sent", data, warnings=warnings, next_actions=actions
    )
    summaries = [
        f"{cycle.signal.value}: {verdict.message} Sent {cycle.sent}, acknowledged {cycle.acked}, "
        f"partial {cycle.partial}, quarantined {cycle.quarantined}."
        for cycle, verdict in zip(cycles, verdicts, strict=True)
    ]
    if failures:
        failure = failures[0]
        command = failure.next_command or command_for(profile, "savings status")
        message = " ".join(summaries)
        if is_json_mode():
            # Preserve the successful signal results too; usecli must not wrap a partial failure
            # in {ok:true}. As in exit_with, write outside its captured stdout and exit explicitly.
            stream = sys.__stdout__ or sys.stdout
            try:
                stream.write(
                    json.dumps(
                        {
                            "ok": False,
                            "condition": failure.condition,
                            "message": message,
                            "command": command,
                            "data": result,
                        }
                    )
                    + "\n"
                )
                stream.flush()
            except BrokenPipeError:
                pass
            os._exit(failure.code)
        exit_with(failure.condition or "send_failed", message, command, failure.code)
    if is_json_mode():
        return result
    print(
        "\n".join(
            (
                *summaries,
                "Acknowledged means collector acceptance, not backend durability.",
                *warnings,
            )
        )
    )
    return None


def _local(
    profile: str, output: str, since: str | None, until: str | None, dry_run: bool
) -> dict[str, object] | None:
    from ssgrep.accounting.views.types import command_for
    from ssgrep.cli.accounting_cli import envelope, usage_error
    from ssgrep.services.export_report import UnsafeReportOutput, build_report, write_report

    document = build_report(profile, since, until)
    if dry_run:
        if is_json_mode():
            return envelope(
                "preview",
                {"output": output, "report": document, "network": False, "state_changed": False},
            )
        print("Preview only; no file was written and nothing was sent.")
        print(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False))
        return None
    try:
        if output == "-" and is_json_mode():
            # Explicit data stream: --output - emits the same document as file output,
            # even under root --json, without a framework status wrapper.
            with redirect_stdout(sys.__stdout__ or sys.stdout):
                write_report(document, output)
                sys.stdout.flush()
            os._exit(0)
        write_report(document, output)
    except (UnsafeReportOutput, NotADirectoryError, FileNotFoundError, PermissionError):
        usage_error(
            "invalid_output",
            "Cannot write this output path. Choose an owned regular file under existing, "
            "writable directories without symlinks. On macOS use /private/tmp, not /tmp. "
            "No partial report replaced the target.",
            command_for(profile, "savings export --help"),
        )
    if output == "-":
        return None
    if is_json_mode():
        return envelope("saved", {"output": output, "metadata": document["metadata"]})
    print(f"Saved private accounting report to {output}. Nothing was sent.")
    return None


class MetricsExportCommand(BaseCommand):
    def visible(self) -> bool:
        from ssgrep.cli.commands import accounting_command_visible

        return accounting_command_visible()

    def signature(self) -> str:
        return "savings export"

    def description(self) -> str:
        return "Send approved data to a destination, or save a private JSON report"

    def register(self) -> None:
        if not self.visible():
            return
        from usecli.cli.core.base_command import NestedCommandRegistry

        savings = NestedCommandRegistry().get_or_create_group(self.app, "savings")
        # Standard Typer deliberately avoids usecli's injected --json/--interactive options.
        savings.command(
            name="export",
            help=self.description(),
            cls=TyperCommand,
            context_settings={"help_option_names": ["--help", "-h"]},
        )(self.handle)

    def handle(
        self,
        destination: Annotated[
            str | None, Argument(help="Destination ID; defaults to the only approved destination")
        ] = None,
        output: Annotated[
            str | None,
            Option("--output", "-o", help="Save private JSON to FILE, or - for stdout; never send"),
        ] = None,
        since: Annotated[
            str | None,
            Option("--from", "-f", help="Local report start date, YYYY-MM-DD (inclusive)"),
        ] = None,
        until: Annotated[
            str | None, Option("--to", "-t", help="Local report end date, YYYY-MM-DD (exclusive)")
        ] = None,
        profile: Annotated[
            str, Option("--profile", "-p", help="Accounting profile")
        ] = PERSONAL_PROFILE,
        dry_run: Annotated[
            bool, Option("--dry-run", "-n", help="Preview the data; write nothing and send nothing")
        ] = False,
    ) -> dict[str, object] | None:
        from ssgrep.accounting.privacy.policy import PolicyError
        from ssgrep.accounting.views.types import command_for
        from ssgrep.cli.accounting_cli import exit_with, fail, usage_error

        if output is not None and destination is not None:
            usage_error(
                "conflicting_destination",
                "Choose DESTINATION or --output FILE, not both.",
                command_for(profile, "savings export --help"),
            )
        if output is None and (since is not None or until is not None):
            usage_error(
                "local_range_required",
                "Date ranges require --output FILE. Remote OTLP "
                "counters stay cumulative; they cannot be truncated to a date range.",
                command_for(profile, "savings export --help"),
            )
        try:
            if output is not None:
                return _local(profile, output, since, until, dry_run)
            selected, signals = _target(profile, destination, preview=dry_run)
            return (
                _preview(profile, selected, signals)
                if dry_run
                else _send(profile, selected, signals)
            )
        except PolicyError as error:
            fail(error)
        except BrokenPipeError:
            raise SystemExit(exit_codes.INTERNAL_FAILURE) from None
        except (OSError, ValueError):
            exit_with(
                "export_unavailable",
                "The accounting export could not be completed safely.",
                command_for(profile, "savings status"),
                exit_codes.INTERNAL_FAILURE,
            )
