"""``ssgrep savings serve``: foreground collection plus authenticated loopback HTTP."""

from __future__ import annotations

import json
import os
import sys
from typing import Annotated

from usecli import BaseCommand, Option
from usecli.cli.core.runtime import is_json_mode

from ssgrep.accounting.ledger.contracts import DEFAULT_PORT, PERSONAL_PROFILE


class MetricsServeCommand(BaseCommand):
    def visible(self) -> bool:
        from ssgrep.cli.commands import accounting_command_visible

        return accounting_command_visible()

    def signature(self) -> str:
        return "savings serve"

    def description(self) -> str:
        return "Run the foreground worker and authenticated loopback metrics server"

    def handle(
        self,
        port: Annotated[
            int, Option("--port", "-p", help=f"Loopback HTTP port (default {DEFAULT_PORT})")
        ] = DEFAULT_PORT,
        profile: Annotated[
            str, Option("--profile", help="Accounting profile (default personal-local)")
        ] = PERSONAL_PROFILE,
    ) -> None:
        from ssgrep.accounting.privacy.policy import PolicyError
        from ssgrep.accounting.views.types import check_profile, command_for
        from ssgrep.cli import exit_codes
        from ssgrep.cli.accounting_cli import exit_with, fail, terminal_width, usage_error

        try:
            check_profile(profile)
        except PolicyError as error:
            fail(error)
        if not 0 <= port <= 65535:
            usage_error(
                "invalid_port",
                "Port must be between 0 and 65535.",
                command_for(profile, "savings serve"),
            )

        from ssgrep.accounting.export.delivery import export_callback
        from ssgrep.accounting.serve import http, http_auth
        from ssgrep.accounting.serve.lease import AlreadyRunning, Lease
        from ssgrep.accounting.serve.serve_loop import LoopConfig, Worker, WorkerState
        from ssgrep.cli.collect_render import (
            already_running_message,
            render_serve_final,
            render_serve_start,
            serve_data,
        )
        from ssgrep.cli.savings_render import emit
        from ssgrep.store.paths import data_dir

        config = LoopConfig()
        server: http.ServerHandle | None = None
        bound_port = port
        announced = False
        token_file = data_dir() / "accounting" / profile / http_auth.TOKEN_FILE

        last_error: str | None = None

        def announce(state: WorkerState) -> None:
            nonlocal announced, last_error
            if state.last_error != last_error and state.last_error is not None:
                # A failing cycle or export must be visible while the service keeps running.
                last_error = state.last_error
                print(f"savings serve: last cycle failed ({state.last_error})", file=sys.stderr)
            if announced or state.phase in {"starting", "stopping", "stopped"}:
                return
            announced = True
            url = f"http://127.0.0.1:{bound_port}"
            if is_json_mode():
                document = {
                    "ok": True,
                    "state": "started",
                    "url": url,
                    "port": bound_port,
                    "profile": profile,
                    "token_file": str(token_file),
                }
                stdout = sys.__stdout__ or sys.stdout
                stdout.write(json.dumps(document, separators=(",", ":")) + "\n")
                stdout.flush()
                return
            emit(
                render_serve_start(
                    profile,
                    url,
                    bound_port,
                    str(token_file),
                    config.collect_seconds,
                    config.export_seconds,
                    os.getpid(),
                    terminal_width(),
                )
            )

        worker = Worker(profile, config=config, on_event=announce, export=export_callback(profile))
        try:
            server = http.serve(profile, port=port, worker=worker, start=False)
        except http.PortInUse:
            try:
                lease = Lease.acquire(profile, "serve", retry_seconds=0)
            except AlreadyRunning as error:
                exit_with(
                    "already_running",
                    already_running_message(error),
                    command_for(profile, "savings status"),
                    exit_codes.INTERNAL_FAILURE,
                )
            except Exception as error:
                exit_with(
                    "server_start_failed",
                    f"The loopback metrics server could not start ({type(error).__name__}).",
                    command_for(profile, "savings status"),
                    exit_codes.INTERNAL_FAILURE,
                )
            else:
                lease.release()
            exit_with(
                "port_in_use",
                f"Port {port} is already in use; no alternate port was selected.",
                command_for(profile, "savings status"),
                exit_codes.INTERNAL_FAILURE,
            )
        except http_auth.TokenFileError:
            exit_with(
                "token_file_unsafe",
                "The endpoint token file is unsafe; restrict it to owner-only permissions (0600).",
                command_for(profile, "savings status"),
                exit_codes.INTERNAL_FAILURE,
            )
        except Exception as error:
            exit_with(
                "server_start_failed",
                f"The loopback metrics server could not start ({type(error).__name__}).",
                command_for(profile, "savings status"),
                exit_codes.INTERNAL_FAILURE,
            )
        bound_port = server.port

        already_running: AlreadyRunning | None = None
        run_error: BaseException | None = None
        try:
            worker.install_signal_handlers()
            server.start()
            final = worker.run()
        except AlreadyRunning as error:
            already_running = error
        except BaseException as error:
            run_error = error
        finally:
            try:
                server.stop()
            finally:
                try:
                    worker.stop()
                finally:
                    worker.join()

        if already_running is not None:
            exit_with(
                "already_running",
                already_running_message(already_running),
                command_for(profile, "savings status"),
                exit_codes.INTERNAL_FAILURE,
            )
        if run_error is not None:
            if is_json_mode() and announced:
                sys.stderr.write(
                    f"Metrics worker stopped unexpectedly ({type(run_error).__name__}).\n"
                )
                sys.stderr.flush()
                code = (
                    run_error.code
                    if isinstance(run_error, SystemExit) and type(run_error.code) is int
                    else 130
                    if isinstance(run_error, KeyboardInterrupt)
                    else 1
                )
                os._exit(code)
            if isinstance(run_error, SystemExit | KeyboardInterrupt):
                raise run_error
            exit_with(
                "serve_failed",
                f"The metrics worker stopped unexpectedly ({type(run_error).__name__}).",
                command_for(profile, "savings status"),
                exit_codes.INTERNAL_FAILURE,
            )
        if is_json_mode():
            (sys.__stdout__ or sys.stdout).flush()
            os._exit(0)
        final_data = serve_data(profile, final, config.collect_seconds, config.export_seconds)
        final_data.update(
            {
                "url": f"http://127.0.0.1:{bound_port}",
                "port": bound_port,
                "token_file": str(token_file),
            }
        )
        emit(render_serve_final(final_data, terminal_width()))
