"""``ssgrep savings export``: consent, preview, connectivity check and manual send.

Modes are validated before any file is read. ``--preview`` and ``--check`` never change
delivery state; ``--enable`` always shows the privacy preview and needs confirmation; enabling
never sends. The delivery stack is imported only after validation has passed.
"""

from __future__ import annotations

import dataclasses
import sys
from typing import TYPE_CHECKING, Annotated

from usecli import Argument, BaseCommand, Option
from usecli.cli.core.base_command import CustomHelpCommand
from usecli.cli.core.runtime import is_json_mode

from ssgrep.accounting.ledger.contracts import PERSONAL_PROFILE
from ssgrep.cli import exit_codes

if TYPE_CHECKING:
    from ssgrep.accounting.export import consent


def _enable(request: consent.ExportRequest, yes: bool) -> dict[str, object] | None:
    from ssgrep.accounting.export import consent
    from ssgrep.accounting.privacy.policy import PolicyError
    from ssgrep.cli.accounting_cli import exit_with, fail, stdin_is_tty, terminal_width

    profile, destination, signal = request.profile, request.destination, request.signal
    try:
        plan = consent.plan_enable(profile, destination, signal)
    except PolicyError as error:
        fail(error)
    interactive = stdin_is_tty() and not is_json_mode()
    if not yes and not interactive:
        exit_with(
            "confirmation_required",
            "Enabling export needs explicit consent: re-run with --yes (non-interactive). "
            "Nothing was changed.",
            consent.target_command(profile, destination, signal, "--enable --yes"),
            exit_codes.USAGE_ERROR,
        )
    if not is_json_mode():
        from ssgrep.cli.export_render import render_enable_preview
        from ssgrep.cli.savings_render import emit

        emit(render_enable_preview(plan, terminal_width()))
    if not yes:
        print(file=sys.stdout)
        try:
            answer = input(
                f"Enable {destination}/{signal.value} export with these settings? [y/N] "
            )
        except EOFError:
            answer = ""
        if answer.strip().lower() not in {"y", "yes"}:
            print("Cancelled. Nothing was changed.", file=sys.stderr)
            return None
    try:
        changed = consent.apply_enable(profile, destination, signal)
    except PolicyError as error:
        fail(error)
    if is_json_mode():
        from ssgrep.cli.export_json import enable_document

        return enable_document(plan, changed)
    from ssgrep.cli.export_render import render_enable_result
    from ssgrep.cli.savings_render import emit

    print(file=sys.stdout)
    emit(render_enable_result(plan, changed, terminal_width()))
    return None


def _disable(request: consent.ExportRequest) -> dict[str, object] | None:
    from ssgrep.accounting.export import consent
    from ssgrep.accounting.privacy.policy import PolicyError
    from ssgrep.cli.accounting_cli import fail, terminal_width

    try:
        report = consent.apply_disable(request.profile, request.destination, request.signal)
        version = consent.consent_state(
            request.profile, request.destination, request.signal
        ).policy.policy_version
    except PolicyError as error:
        fail(error)
    if is_json_mode():
        from ssgrep.cli.export_json import disable_document

        return disable_document(report, request.profile, version)
    from ssgrep.cli.export_render import render_disable
    from ssgrep.cli.savings_render import emit

    emit(render_disable(report, request.profile, terminal_width()))
    return None


def _preview(request: consent.ExportRequest) -> dict[str, object] | None:
    from ssgrep.accounting.export import consent as _consent
    from ssgrep.accounting.export.registry import RecordsUnavailable, RegistryCorrupt
    from ssgrep.accounting.privacy.policy import PolicyError
    from ssgrep.accounting.views.types import command_for
    from ssgrep.cli.accounting_cli import exit_with, fail, terminal_width

    try:
        view = _consent.build_preview(request.profile, request.destination, request.signal)
    except PolicyError as error:
        fail(error)
    except (RecordsUnavailable, RegistryCorrupt) as error:
        exit_with(
            error.condition,
            "The export candidate cannot be built because local ledger state is unavailable. "
            "Nothing was changed.",
            command_for(request.profile, "savings status"),
            exit_codes.INTERNAL_FAILURE,
        )
    except (ValueError, OSError):  # never echo text derived from untrusted snapshot content
        exit_with(
            "preview_unavailable",
            "The export candidate cannot be built from the current local data. "
            "Nothing was changed.",
            command_for(request.profile, "savings status"),
            exit_codes.INTERNAL_FAILURE,
        )
    if is_json_mode():
        from ssgrep.cli.export_json import preview_document

        return preview_document(view)
    from ssgrep.cli.export_render import preview_payload_lines, render_preview_header
    from ssgrep.cli.savings_render import emit

    emit(render_preview_header(view, terminal_width()))
    lines = preview_payload_lines(view)
    if lines:
        print(file=sys.stdout)
        for line in lines:
            print(line)
    return None


def _check(request: consent.ExportRequest) -> dict[str, object] | None:
    from ssgrep.accounting.export import consent, delivery
    from ssgrep.accounting.privacy.policy import PolicyError
    from ssgrep.accounting.views.types import command_for
    from ssgrep.cli.accounting_cli import exit_with, fail, terminal_width

    try:
        consent.require_target(request.profile, request.destination, request.signal)
        result = delivery.check(request.profile, request.destination, request.signal)
    except PolicyError as error:
        fail(error)
    except delivery.ExportPaused:
        exit_with(
            "export_paused",
            "Export is paused; no connectivity check was made. Nothing was changed.",
            command_for(request.profile, "savings resume"),
            exit_codes.INTERNAL_FAILURE,
        )
    if not result.ok:
        action = (
            " Update the credential reference, then check again."
            if result.state in ("unauthorized", "forbidden")
            else ""
        )
        exit_with(
            f"check_{result.state}",
            f"Connectivity check failed ({result.state.replace('_', ' ')}; result category "
            f"{result.category}).{action} Nothing was changed.",
            consent.target_command(request.profile, request.destination, request.signal, "--check"),
            exit_codes.INTERNAL_FAILURE,
        )
    if is_json_mode():
        from ssgrep.cli.export_json import check_document

        return check_document(
            request.profile,
            result.destination,
            result.signal.value,
            result.state,
            result.ok,
            result.status,
            result.category,
        )
    from ssgrep.cli.export_render import render_check
    from ssgrep.cli.savings_render import emit

    emit(
        render_check(
            result.destination,
            result.signal.value,
            result.category,
            result.status,
            terminal_width(),
        )
    )
    return None


def _send(request: consent.ExportRequest) -> dict[str, object] | None:
    from ssgrep.accounting.export import consent, delivery
    from ssgrep.accounting.privacy.policy import PolicyError
    from ssgrep.accounting.views.types import command_for
    from ssgrep.cli.accounting_cli import exit_with, fail, terminal_width

    try:
        consent.require_target(request.profile, request.destination, request.signal)
        cycle = delivery.run_destination(request.profile, request.destination, request.signal)
    except PolicyError as error:
        fail(error)
    verdict = consent.classify_cycle(cycle, request.profile)
    if not verdict.ok:
        exit_with(
            verdict.condition or "send_failed",
            f"{verdict.message} Nothing further was sent.",
            verdict.next_command or command_for(request.profile, "savings status"),
            verdict.code,
        )
    data = dataclasses.asdict(cycle)
    data["signal"] = cycle.signal.value
    data["warnings"] = list(cycle.warnings)
    if is_json_mode():
        from ssgrep.cli.export_json import send_document

        return send_document(request.profile, data, verdict.next_command, "Next step.", cycle.state)
    lines = (
        verdict.message,
        f"Enqueued {cycle.enqueued_batches} batch(es); sent {cycle.sent}, "
        f"acknowledged {cycle.acked}, partial {cycle.partial}, quarantined {cycle.quarantined}.",
        "Acknowledged means collector acceptance only, not backend durability.",
        *cycle.warnings,
    )
    from ssgrep.cli.export_render import render_send
    from ssgrep.cli.savings_render import emit

    emit(render_send("Export cycle finished.", lines, terminal_width()))
    return None


class MetricsExportCommand(BaseCommand):
    def visible(self) -> bool:
        from ssgrep.cli.commands import accounting_command_visible

        return accounting_command_visible()

    def signature(self) -> str:
        return "savings export"

    def description(self) -> str:
        return "Organization export: preview, check, enable, disable, send data, or run its service"

    def register(self) -> None:
        if not self.visible():
            return
        from ssgrep.cli.commands import savings_export_group

        export = savings_export_group(self.app, help_text=self.description())

        def dispatch(
            destination: str, logs: bool, yes: bool, profile: str, mode: str
        ) -> dict[str, object] | None:
            return self.handle(
                destination=destination,
                signal="logs" if logs else "metrics",
                preview=mode == "preview",
                check=mode == "check",
                enable=mode == "enable",
                disable=mode == "disable",
                yes=yes if mode == "enable" else False,
                profile=profile,
            )

        def register_mode(name: str, description: str, mode: str) -> None:
            if mode == "enable":

                def command(
                    destination: Annotated[str, Argument(..., help="Approved destination ID")],
                    logs: Annotated[
                        bool,
                        Option(
                            "--logs",
                            "-l",
                            help="Also use the separately permitted ledger-log signal",
                        ),
                    ] = False,
                    yes: Annotated[
                        bool, Option("--yes", "-y", help="Confirm consent without prompting")
                    ] = False,
                    profile: Annotated[
                        str, Option("--profile", help="Accounting profile (default personal-local)")
                    ] = PERSONAL_PROFILE,
                ) -> dict[str, object] | None:
                    return dispatch(destination, logs, yes, profile, mode)

            else:

                def command(
                    destination: Annotated[str, Argument(..., help="Approved destination ID")],
                    logs: Annotated[
                        bool,
                        Option(
                            "--logs",
                            "-l",
                            help="Also use the separately permitted ledger-log signal",
                        ),
                    ] = False,
                    profile: Annotated[
                        str, Option("--profile", help="Accounting profile (default personal-local)")
                    ] = PERSONAL_PROFILE,
                ) -> dict[str, object] | None:
                    return dispatch(destination, logs, False, profile, mode)

            export.command(name=name, help=description, cls=CustomHelpCommand)(command)

        register_mode("preview", "Inspect the exact sanitized request; send nothing.", "preview")
        register_mode("check", "Test connectivity with an empty request.", "check")
        register_mode("enable", "Approve sending this destination and signal.", "enable")
        register_mode("disable", "Stop sending this destination and signal.", "disable")
        register_mode("send", "Send the queued approved data now.", "send")

    def handle(
        self,
        destination: Annotated[
            str | None, Option("--destination", help="Destination id from the profile policy")
        ] = None,
        signal: Annotated[
            str, Option("--signal", help="Signal: metrics (default) or logs")
        ] = "metrics",
        preview: Annotated[
            bool, Option("--preview", help="Show the exact candidate request bytes; no network")
        ] = False,
        check: Annotated[
            bool, Option("--check", help="Send an empty OTLP request to test connectivity")
        ] = False,
        enable: Annotated[
            bool, Option("--enable", help="Authorize this destination/signal after a preview")
        ] = False,
        disable: Annotated[
            bool, Option("--disable", help="Stop sending this destination/signal")
        ] = False,
        yes: Annotated[
            bool, Option("--yes", help="Confirm --enable without prompting (non-interactive)")
        ] = False,
        profile: Annotated[
            str, Option("--profile", help="Accounting profile (default personal-local)")
        ] = PERSONAL_PROFILE,
    ) -> dict[str, object] | None:
        from ssgrep.accounting.export import consent
        from ssgrep.accounting.privacy.policy import PolicyError
        from ssgrep.cli.accounting_cli import fail

        try:
            request = consent.validate_modes(
                destination=destination,
                signal=signal,
                profile=profile,
                preview=preview,
                check=check,
                enable=enable,
                disable=disable,
            )
        except PolicyError as error:
            fail(error)
        handlers = {
            "enable": lambda: _enable(request, yes),
            "disable": lambda: _disable(request),
            "preview": lambda: _preview(request),
            "check": lambda: _check(request),
            consent.SEND: lambda: _send(request),
        }
        return handlers[request.mode]()
