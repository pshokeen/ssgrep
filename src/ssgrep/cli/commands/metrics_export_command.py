"""``ssgrep metrics export``: consent, preview, connectivity check and manual send.

Modes are validated before any file is read. ``--preview`` and ``--check`` never change
delivery state; ``--enable`` always shows the privacy preview and needs confirmation; enabling
never sends. The delivery stack is imported only after validation has passed.
"""

from __future__ import annotations

import dataclasses
import sys
from typing import Annotated

from usecli import BaseCommand, Option
from usecli.cli.core.runtime import is_json_mode

from ssgrep.accounting import export_consent as consent
from ssgrep.accounting.contracts import PERSONAL_PROFILE
from ssgrep.accounting.export_consent_preview import build_preview
from ssgrep.accounting.export_records_model import RecordsUnavailable, RegistryCorrupt
from ssgrep.accounting.policy_rules import PolicyError
from ssgrep.accounting.service_types import command_for
from ssgrep.cli import exit_codes
from ssgrep.cli.accounting_cli import exit_with, fail, stdin_is_tty, terminal_width
from ssgrep.cli.export_json import (
    check_document,
    disable_document,
    enable_document,
    preview_document,
    send_document,
)
from ssgrep.cli.export_render import (
    preview_payload_lines,
    render_check,
    render_disable,
    render_enable_preview,
    render_enable_result,
    render_preview_header,
    render_send,
)
from ssgrep.cli.savings_render import emit


def _enable(request: consent.ExportRequest, yes: bool) -> dict[str, object] | None:
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
        return enable_document(plan, changed)
    print(file=sys.stdout)
    emit(render_enable_result(plan, changed, terminal_width()))
    return None


def _disable(request: consent.ExportRequest) -> dict[str, object] | None:
    try:
        report = consent.apply_disable(request.profile, request.destination, request.signal)
        version = consent.consent_state(
            request.profile, request.destination, request.signal
        ).policy.policy_version
    except PolicyError as error:
        fail(error)
    if is_json_mode():
        return disable_document(report, request.profile, version)
    emit(render_disable(report, request.profile, terminal_width()))
    return None


def _preview(request: consent.ExportRequest) -> dict[str, object] | None:
    try:
        view = build_preview(request.profile, request.destination, request.signal)
    except PolicyError as error:
        fail(error)
    except (RecordsUnavailable, RegistryCorrupt) as error:
        exit_with(
            error.condition,
            "The export candidate cannot be built because local ledger state is unavailable. "
            "Nothing was changed.",
            command_for(request.profile, "metrics status"),
            exit_codes.INTERNAL_FAILURE,
        )
    except (ValueError, OSError):  # never echo text derived from untrusted snapshot content
        exit_with(
            "preview_unavailable",
            "The export candidate cannot be built from the current local data. "
            "Nothing was changed.",
            command_for(request.profile, "metrics status"),
            exit_codes.INTERNAL_FAILURE,
        )
    if is_json_mode():
        return preview_document(view)
    emit(render_preview_header(view, terminal_width()))
    lines = preview_payload_lines(view)
    if lines:
        print(file=sys.stdout)
        for line in lines:
            print(line)
    return None


def _check(request: consent.ExportRequest) -> dict[str, object] | None:
    from ssgrep.accounting import delivery

    try:
        consent.require_target(request.profile, request.destination, request.signal)
        result = delivery.check(request.profile, request.destination, request.signal)
    except PolicyError as error:
        fail(error)
    except delivery.ExportPaused:
        exit_with(
            "export_paused",
            "Export is paused; no connectivity check was made. Nothing was changed.",
            command_for(request.profile, "metrics resume"),
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
        return check_document(
            request.profile,
            result.destination,
            result.signal.value,
            result.state,
            result.ok,
            result.status,
            result.category,
        )
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
    from ssgrep.accounting import delivery

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
            verdict.next_command or command_for(request.profile, "metrics status"),
            verdict.code,
        )
    data = dataclasses.asdict(cycle)
    data["signal"] = cycle.signal.value
    data["warnings"] = list(cycle.warnings)
    if is_json_mode():
        return send_document(request.profile, data, verdict.next_command, "Next step.", cycle.state)
    lines = (
        verdict.message,
        f"Enqueued {cycle.enqueued_batches} batch(es); sent {cycle.sent}, "
        f"acknowledged {cycle.acked}, partial {cycle.partial}, quarantined {cycle.quarantined}.",
        "Acknowledged means collector acceptance only, not backend durability.",
        *cycle.warnings,
    )
    emit(render_send("Export cycle finished.", lines, terminal_width()))
    return None


class MetricsExportCommand(BaseCommand):
    def visible(self) -> bool:
        return True

    def signature(self) -> str:
        return "metrics export"

    def description(self) -> str:
        return "Preview, enable, disable, check, or manually send organization export"

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
