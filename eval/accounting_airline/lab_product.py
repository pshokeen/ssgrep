"""Drive the REAL ssgrep binaries (collect / savings / status / index / search / show).

The lab never imports the product reducer to get numbers: it runs ``.venv/bin/ssgrep`` as a
subprocess in an explicit isolated environment and reads the JSON the binary printed. The one
product API touched in-process is ``pin_arithmetic_view``, which runs a tiny subprocess to pin
the test-only candidate-frequency view of the ``airline-lab-arithmetic-v1`` fixture profile in a
COPIED, isolated accounting root (never the root of the production-policy evaluation).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .observer import Frozen

REPO_ROOT = Path(__file__).resolve().parents[2]
BINARY = Path(sys.executable).parent / "ssgrep"
PROFILE = "personal-local"
FILLER_DOCS = 1000  # frozen test-only view: each reviewed token occurs in exactly one document
INDEX_TIMEOUT = 1800.0  # first cold index loads the embedding model; bounded, never retried
CLI_TIMEOUT = 900.0


class ProductError(RuntimeError):
    """A real product command failed in a way the lab cannot interpret."""


@dataclass(frozen=True)
class Product:
    """One explicit isolated environment in which ``ssgrep`` runs."""

    env: Mapping[str, str]
    cwd: Path
    binary: Path = BINARY

    @property
    def data_dir(self) -> Path:
        return Path(self.env["SSGREP_DATA_DIR"])

    def with_env(self, **changes: str) -> Product:
        return Product({**self.env, **changes}, self.cwd, self.binary)

    def run(
        self, *args: str, timeout: float = CLI_TIMEOUT, stdin: str = ""
    ) -> subprocess.CompletedProcess[str]:
        env = {**self.env, "COLUMNS": "100", "NO_COLOR": "1"}
        return subprocess.run(
            [str(self.binary), *args], input=stdin, capture_output=True, text=True, env=env,
            cwd=self.cwd, timeout=timeout, check=False,
        )  # fmt: skip

    def json(self, *args: str, timeout: float = CLI_TIMEOUT) -> tuple[int, dict[str, Any]]:
        done = self.run(*args, "--json", timeout=timeout)
        lines = [ln for ln in done.stdout.strip().splitlines() if ln.startswith("{")]
        if not lines:
            raise ProductError(f"ssgrep {' '.join(args)} printed no JSON: {done.stderr[-300:]}")
        return done.returncode, json.loads(lines[-1])

    def data(self, *args: str) -> dict[str, Any]:
        code, body = self.json(*args)
        if code != 0 or body.get("ok") is False:
            raise ProductError(f"ssgrep {' '.join(args)} failed ({code}): {json.dumps(body)[:300]}")
        return body.get("data", body)


@dataclass
class Collected:
    """What the real binaries said after one collection."""

    collect: dict[str, Any]
    savings: dict[str, Any]
    status: dict[str, Any]
    elapsed_ms: int
    savings_document: dict[str, Any] = field(default_factory=dict)

    @property
    def estimates(self) -> dict[str, Any]:
        return find(self.savings_document, "estimates") or {}

    @property
    def measured(self) -> dict[str, Any]:
        return find(self.savings_document, "measured") or {}

    @property
    def coverage(self) -> dict[str, Any]:
        return find(self.savings_document, "coverage") or {}


def find(doc: Any, key: str) -> dict[str, Any] | None:
    """First mapping stored under ``key`` anywhere in a nested JSON document."""
    if isinstance(doc, dict):
        if isinstance(doc.get(key), dict):
            return doc[key]
        for value in doc.values():
            found = find(value, key)
            if found is not None:
                return found
    return None


def setup_profile(product: Product) -> dict[str, Any]:
    """Enable local measurement for the default profile (export stays off)."""
    return product.data("metrics", "setup", "--yes")


def collect(product: Product, *flags: str) -> Collected:
    """``metrics collect`` + ``savings --json`` + ``metrics status --json`` (real binaries)."""
    started = time.monotonic()
    code, body = product.json("metrics", "collect", *flags)
    if code != 0:
        raise ProductError(f"metrics collect failed ({code}): {json.dumps(body)[:400]}")
    code, doc = product.json("savings")
    if code != 0:
        raise ProductError(f"savings failed ({code}): {json.dumps(doc)[:400]}")
    status = product.data("metrics", "status")
    return Collected(
        body.get("data", body), doc.get("data", doc), status,
        int((time.monotonic() - started) * 1000), doc,
    )  # fmt: skip


def savings_digest_view(collected: Collected) -> dict[str, Any]:
    """The deterministic part of a savings document (no clocks/ages) for equality checks."""
    return {
        "estimates": collected.estimates,
        "measured": {k: v for k, v in collected.measured.items()},
        "coverage": collected.coverage,
    }


def index(product: Product, *, timeout: float = INDEX_TIMEOUT) -> tuple[int, int, str]:
    """Real ``ssgrep index --quiet``; returns (returncode, elapsed_ms, last stderr line)."""
    started = time.monotonic()
    done = product.run("index", "--quiet", timeout=timeout)
    tail = (done.stderr.strip().splitlines() or [""])[-1][:200]
    return done.returncode, int((time.monotonic() - started) * 1000), tail


def index_status(product: Product) -> dict[str, Any]:
    return product.data("status")


# ------------------------------------------------------------------- isolated data copies


def copy_accounting(source: Path, destination: Path, *, journal_only: bool = False) -> Path:
    """Copy a lab accounting root (never touching the source); ``journal_only`` = replay seed.

    Everything is copied with owner-only permissions. ``journal_only`` keeps just the policy,
    the profile key and the journal, so a collector started there must rebuild everything.
    """
    src = source / "accounting"
    dst = destination / "accounting"
    if dst.exists():
        raise ProductError(f"refusing to overwrite {dst}")
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not journal_only:
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.lock", "lease.lock"))
    else:
        dst.mkdir(mode=0o700)
        shutil.copy2(src / "policy.json", dst / "policy.json")
        profile = dst / PROFILE
        profile.mkdir(mode=0o700)
        shutil.copy2(src / PROFILE / "fingerprint.key", profile / "fingerprint.key")
        shutil.copytree(src / PROFILE / "journal", profile / "journal")
    for path in [dst, *dst.rglob("*")]:
        if not path.is_symlink():
            path.chmod(0o700 if path.is_dir() else 0o600)
    return destination


def reviewed_view_documents(tokens: Sequence[str]) -> list[str]:
    """Test-only frozen corpus: one document per reviewed token plus neutral filler."""
    docs = [f"filler {i}" for i in range(FILLER_DOCS - len(tokens))]
    return docs + [f"reviewed fixture {token}" for token in tokens]


_PIN = """
import json, sys
from ssgrep.accounting import collector_frequency as cf
from ssgrep.accounting.frequency import build_view
docs = json.loads(sys.stdin.read())
view = build_view(docs)
cf.pin("personal-local", view)
print(json.dumps({"view_id": view.version_id, "docs": view.doc_count}))
"""


def pin_arithmetic_view(product: Product, tokens: Sequence[str]) -> dict[str, Any]:
    """Pin the airline-lab-arithmetic-v1 test view in ``product``'s (copied) accounting root.

    Every reviewed token occurs in exactly one of 1000 documents (rarity above all three
    production thresholds); every other candidate is absent from the view and therefore
    unknown, so the ONLY credit-eligible candidates are the reviewed list. Production
    thresholds and code are untouched: the copy is never exported or merged.
    """
    done = subprocess.run(
        [sys.executable, "-c", _PIN], input=json.dumps(reviewed_view_documents(tokens)),
        capture_output=True, text=True, env={**product.env}, cwd=product.cwd, timeout=300,
        check=False,
    )  # fmt: skip
    if done.returncode != 0:
        raise ProductError(f"pinning the arithmetic view failed: {done.stderr[-400:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


# ------------------------------------------------------------------- expected counts


def journaled_expected(frozen: Frozen) -> dict[str, int]:
    """How many observed ssgrep executions entered the service API (so the journal has them).

    Counted from OBSERVED dispatches only: a verified normal exit through the API. A CLI usage
    error (exit 2, argument validation before the API) is an invocation failure, not an
    execution; attempted/interrupted/unknown dispatches are never counted as journaled.
    """
    out = {"search": 0, "show": 0, "note": 0, "pre_api_failures": 0, "not_completed": 0}
    for call in frozen.calls:
        if call.tool != "ssgrep" or len(call.argv) < 2 or call.argv[1] not in out:
            continue
        if call.state != "verified":
            out["not_completed"] += 1
        elif call.exit_code == 2:
            out["pre_api_failures"] += 1
        else:
            out[call.argv[1]] += 1
    return out


def clean_env_names(env: Mapping[str, str]) -> list[str]:
    """Names in ``env`` that look like credentials (must be empty for every lab environment)."""
    from .roots import forbidden_env_names

    return forbidden_env_names(env)


def real_model_cache() -> Path:
    return Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
