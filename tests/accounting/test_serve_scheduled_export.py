"""The serve command must deliver permitted snapshots, not just expose a scrape port."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from ssgrep.accounting import serve_loop
from ssgrep.accounting.collector_types import CycleReport
from ssgrep.cli.commands import metrics_serve_command
from tests.accounting import delivery_fixtures as fx
from tests.accounting.lab_receiver import LabReceiver


def test_serve_sends_enabled_snapshot_on_its_export_tick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accelerate only time/collection; CLI, worker, policy, spool and HTTP are real."""
    monotonic = [0.0]
    snapshot_id = ""

    class ScheduledWorker(serve_loop.Worker):
        def __init__(self, profile: str, **kwargs: Any) -> None:
            def wait(seconds: float) -> bool:
                monotonic[0] += seconds
                if monotonic[0] > 300:
                    self.stop()
                return monotonic[0] > 300

            def cycle(_stopped: Any) -> CycleReport:
                return CycleReport("published", 1, snapshot_id, True, "complete")

            super().__init__(
                profile,
                clock=lambda: monotonic[0],
                wait=wait,
                cycle=cycle,
                **kwargs,
            )

    monkeypatch.setattr(serve_loop, "Worker", ScheduledWorker)
    monkeypatch.setattr(metrics_serve_command, "is_json_mode", lambda: False)
    monkeypatch.setenv(fx.ENV_NAME, fx.SECRET)
    with LabReceiver.start() as receiver:
        fx.configure(receiver.url)
        snapshot_id = fx.publish(1, collected_at=fx.stamp(datetime.now(UTC)))
        command = object.__new__(metrics_serve_command.MetricsServeCommand)
        command.handle(port=0, profile=fx.PROFILE)

        assert len(receiver.requests) == 1
        request = receiver.requests[0]
        assert request.path == "/v1/metrics"
        payload = json.loads(request.body)
        metrics = payload["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]
        executions = next(metric for metric in metrics if metric["name"] == "ssgrep_executions")
        points = executions["sum"]["dataPoints"]
        counts = {
            next(
                attr["value"]["stringValue"]
                for attr in point["attributes"]
                if attr["key"] == "operation"
            ): int(point["asInt"])
            for point in points
        }
        assert counts == {"search": 4, "show": 1}
        assert fx.state().last_acked_snapshot_id == snapshot_id
        assert fx.state().last_acked_generation == 1
