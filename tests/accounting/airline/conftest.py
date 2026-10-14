"""Fixtures for the airline lab tests.

Positive integration controls request named prerequisite fixtures, rather than trusting
the ``postgres`` marker to enforce availability. Only absent prerequisites SKIP with a
visible reason; a failed lane after preflight remains a test failure.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from eval.accounting_airline import faults
from eval.accounting_airline.database import BaselineInfo, LabInstance, preflight
from eval.accounting_airline.gates import BLOCKED, PASS, Gate
from eval.accounting_airline.lab_product import BINARY, REPO_ROOT, real_model_cache
from eval.accounting_airline.manifest import _gate_model_cache, _gate_psql
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


def _require_prerequisite(gate: Gate, lane: str) -> None:
    if gate.status == BLOCKED:
        pytest.skip(f"{lane} prerequisite unavailable: {gate.name} - {gate.detail}")
    if gate.status != PASS:
        pytest.fail(f"{lane} prerequisite failed: {gate.name} - {gate.status}: {gate.detail}")


@pytest.fixture(scope="session")
def postgres_ready() -> None:
    """Real Docker CLI, reachable daemon and the exact local pinned image."""
    _require_prerequisite(preflight(), "PostgreSQL lab")


@pytest.fixture
def standin_ready(postgres_ready: None, request: pytest.FixtureRequest) -> None:
    """Prerequisites actually consumed by the complete scripted C0-C10 pipeline."""
    _require_prerequisite(_gate_psql(), "Scripted stand-in")
    _require_prerequisite(_gate_model_cache(real_model_cache()), "Scripted stand-in")
    if not BINARY.is_file() or not os.access(BINARY, os.X_OK):
        pytest.skip(
            f"Scripted stand-in prerequisite unavailable: ssgrep executable missing: {BINARY}"
        )
    # C4 proves B's filesystem boundary; later B turns run through sandbox-exec.
    request.getfixturevalue("live_sandbox_ready")


@pytest.fixture
def omp_binary_ready() -> None:
    """Only binary absence skips; bad version/help behavior must still fail the control."""
    if shutil.which("omp") is None:
        pytest.skip(
            "OMP version integration prerequisite unavailable: omp binary not found on PATH"
        )


@dataclass
class Lab:
    instance: LabInstance
    base: BaselineInfo
    prep: faults.Prepared


@pytest.fixture(scope="session")
def lab(postgres_ready: None) -> Iterator[Lab]:
    """One run-owned container for the whole test session, torn down in the finalizer."""
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
