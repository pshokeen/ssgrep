"""Exercise the production update driver, not a parallel error check."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ssgrep.pipeline.app import _drive_update
from ssgrep.pipeline.diagnostics import current as diagnostics
from ssgrep.pipeline.update import ComponentUpdateError


@pytest.fixture(autouse=True)
def _clean_diagnostics():
    diagnostics.reset()
    yield
    diagnostics.reset()


class Handle:
    def __init__(self, errors: int, snapshot_errors: int = 0):
        self.errors = errors
        self.snapshot_errors = snapshot_errors
        self.finished = False

    def stats(self):
        return SimpleNamespace(total=SimpleNamespace(num_errors=self.errors))

    async def result(self):
        self.finished = True

    async def watch(self):
        yield SimpleNamespace(
            stats=SimpleNamespace(
                total=SimpleNamespace(num_errors=self.snapshot_errors, num_finished=1),
                by_component={},
            )
        )
        self.finished = True


@pytest.mark.parametrize("total,snapshot_errors", [(1, 0), (1, 1), (0, 0)])
def test_child_error_rejected_after_workers_finish(total, snapshot_errors):
    handle = Handle(errors=1, snapshot_errors=snapshot_errors)
    app = Mock()
    app.update.return_value = handle
    with pytest.raises(RuntimeError, match="component errors"):
        asyncio.run(_drive_update(app, total=total, full_reprocess=False, quiet=True))
    assert handle.finished, "failure was reported while workers could still write"
    app.update.assert_called_once_with(full_reprocess=False)


@pytest.mark.parametrize("total", [0, 1])
def test_successful_update_finishes(total):
    handle = Handle(errors=0)
    app = Mock()
    app.update.return_value = handle
    asyncio.run(_drive_update(app, total=total, full_reprocess=False, quiet=True))
    assert handle.finished
    app.update.assert_called_once_with(full_reprocess=False)


def _fail(errors: int = 2) -> str:
    handle = Handle(errors=errors)
    app = Mock()
    app.update.return_value = handle
    with pytest.raises(ComponentUpdateError) as failure:
        asyncio.run(_drive_update(app, total=1, full_reprocess=False, quiet=True))
    return str(failure.value)


def test_failure_message_names_first_source_and_cause():
    diagnostics.record_error(
        "/t/a.jsonl", "Traceback...\nsqlite3.OperationalError: database is locked"
    )
    diagnostics.record_error("/t/b.jsonl", "ValueError: later error")
    assert _fail() == (
        "Index update failed: 2 component errors "
        "(first: /t/a.jsonl: sqlite3.OperationalError: database is locked)"
    )


def test_failure_message_cause_is_bounded():
    diagnostics.record_error("/t/a.jsonl", "E: " + "x" * 5000)
    message = _fail(1)
    assert message.startswith("Index update failed: 1 component errors (first: /t/a.jsonl: E: x")
    assert message.endswith("...)")
    assert len(message) < 400


def test_failure_message_without_recorded_cause_is_unchanged():
    assert _fail() == "Index update failed: 2 component errors"
