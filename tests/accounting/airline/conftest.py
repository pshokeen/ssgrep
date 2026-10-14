"""Fixtures for the airline lab tests.

PostgreSQL-backed tests carry ``pytest.mark.postgres`` (registered here so no shared
config file changes). They SKIP, with a visible reason, only when ``preflight()`` reports
that docker or the pinned image is truly unavailable; any other failure is a test error.
"""

from __future__ import annotations

import sys
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from eval.accounting_airline import faults
from eval.accounting_airline.database import BaselineInfo, LabInstance, preflight
from eval.accounting_airline.gates import PASS
from eval.accounting_airline.lab_product import REPO_ROOT
from eval.accounting_airline.sandbox import availability


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "postgres: needs real docker + the pinned PostgreSQL 17 image (skips visibly if absent)",
    )


@pytest.fixture
def live_sandbox_ready() -> None:
    """Name unavailable execution prerequisites without relaxing the live sandbox."""
    unavailable = availability()
    if unavailable is not None:
        pytest.skip(f"Live sandbox prerequisite unavailable: {unavailable.reason}")
    sealed = {
        Path(tempfile.gettempdir()).resolve(),
        Path("/tmp").resolve(),
        Path("/var/tmp").resolve(),
    }
    launcher = Path(sys.executable)
    # Resolve the directory, not the executable symlink: sandbox-exec must traverse
    # the actual launcher path even when its final target lives outside temp storage.
    launcher = launcher.parent.resolve() / launcher.name
    for path in (REPO_ROOT.resolve(), launcher):
        if any(path == root or path.is_relative_to(root) for root in sealed):
            pytest.skip(
                "Live sandbox prerequisite unavailable: checkout or Python launcher is "
                f"inside sealed temporary storage ({path}); move the checkout and its "
                "executable environment outside temporary directories."
            )


@dataclass
class Lab:
    instance: LabInstance
    base: BaselineInfo
    prep: faults.Prepared


@pytest.fixture(scope="session")
def lab() -> Iterator[Lab]:
    """One run-owned container for the whole test session, torn down in the finalizer."""
    gate = preflight()
    if gate.status != PASS:
        pytest.skip(f"PostgreSQL lab prerequisite unavailable: {gate.detail}")
    instance = LabInstance.start()
    try:
        base = instance.provision_baseline(faults.baseline_sql("correct"))
        yield Lab(instance, base, faults.Prepared(instance, base.state))
    finally:
        try:
            instance.teardown()
        finally:
            LabInstance.remove_run(instance.run_id)


@pytest.fixture
def scratch(lab: Lab) -> Iterator[str]:
    """A throwaway clone of the recorded baseline for one test."""
    database = lab.instance.clone("scratch")
    try:
        yield database
    finally:
        lab.instance.drop_clone("scratch")
