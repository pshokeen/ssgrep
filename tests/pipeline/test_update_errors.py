"""Exercise the production update driver, not a parallel error check."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ssgrep.pipeline.app import _drive_update


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
