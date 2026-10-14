"""The oracle must stay independent of production valuation/replay/reduce/collector code."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from eval.accounting_airline.accounting_oracle import import_violations

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "eval" / "accounting_airline"
ORACLE_MODULES = ("accounting_oracle", "oracle_rules", "oracle_scan")
FORBIDDEN = re.compile(r"^ssgrep\.accounting\.(valuation|replay|reduce|collector)")


@pytest.mark.parametrize("name", ORACLE_MODULES)
def test_oracle_modules_import_no_forbidden_accounting_code(name: str) -> None:
    assert import_violations(PACKAGE / f"{name}.py") == []


def test_oracle_modules_import_no_ssgrep_at_all_and_only_lab_or_stdlib_modules() -> None:
    from eval.accounting_airline.oracle_rules import imported_modules

    for name in ORACLE_MODULES:
        mods = imported_modules(PACKAGE / f"{name}.py")
        assert not [m for m in mods if m.split(".")[0] == "ssgrep"], (name, mods)


def test_import_violations_detects_a_violating_module_positive_twin(tmp_path: Path) -> None:
    cases = {
        "import ssgrep.accounting.valuation": "ssgrep.accounting.valuation",
        "from ssgrep.accounting import replay": "ssgrep.accounting.replay",
        "from ssgrep.accounting.reduce_engine import x": "ssgrep.accounting.reduce_engine",
        "from ssgrep.accounting.collector.run import go": "ssgrep.accounting.collector.run",
        "import importlib\nimportlib.import_module('ssgrep.accounting.valuation')": (
            "ssgrep.accounting.valuation"
        ),
    }
    for source, expected in cases.items():
        bad = tmp_path / "bad_oracle.py"
        bad.write_text(source + "\n")
        found = import_violations(bad)
        assert found and any(expected in f for f in found), (source, found)
    clean = tmp_path / "clean.py"
    clean.write_text("import json\nfrom ssgrep.accounting.contracts import SCHEMA_VERSION\n")
    assert import_violations(clean) == []  # an unrelated ssgrep import is not a violation


def test_import_violations_is_the_checker_applied_to_a_real_importable_temp_module(
    tmp_path: Path,
) -> None:
    (tmp_path / "violator.py").write_text("import ssgrep.accounting.valuation  # noqa\n")
    assert import_violations(tmp_path / "violator.py")
    probe = textwrap.dedent(
        """
        import sys
        sys.path.insert(0, sys.argv[1])
        import violator
        print(sorted(m for m in sys.modules if m.startswith("ssgrep.accounting.valuation")))
        """
    )
    done = subprocess.run(
        [sys.executable, "-c", probe, str(tmp_path)],
        capture_output=True, text=True, cwd=ROOT, timeout=120,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    assert "ssgrep.accounting.valuation" in done.stdout  # the runtime probe sees the violation


def test_importing_the_oracle_loads_no_forbidden_ssgrep_module_at_runtime() -> None:
    code = (
        "import eval.accounting_airline.accounting_oracle, sys; "
        'print(sorted(m for m in sys.modules if m.startswith("ssgrep")))'
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, timeout=120
    )
    assert done.returncode == 0, done.stderr
    loaded = eval(done.stdout.strip(), {})  # a printed list of module names  # noqa: S307
    assert not [m for m in loaded if FORBIDDEN.match(m)], loaded
    assert loaded == [], loaded  # stronger: nothing from ssgrep is imported at all


def test_boundary_checker_output_is_json_serialisable() -> None:
    assert json.dumps(import_violations(PACKAGE / "oracle_rules.py")) == "[]"
