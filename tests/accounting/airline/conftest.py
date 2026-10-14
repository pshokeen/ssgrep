"""Fixtures for the airline lab tests.

PostgreSQL-backed tests carry ``pytest.mark.postgres`` (registered here so no shared
config file changes). They SKIP, with a visible reason, only when ``preflight()`` reports
that docker or the pinned image is truly unavailable; any other failure is a test error.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pytest

from eval.accounting_airline import faults
from eval.accounting_airline.database import BaselineInfo, LabInstance, preflight
from eval.accounting_airline.gates import PASS


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "postgres: needs real docker + the pinned PostgreSQL 17 image (skips visibly if absent)",
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
