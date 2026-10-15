"""Drive the REAL ssgrep binaries (collect / savings / status / index / search / show).

The lab never imports the product reducer to get expected numbers: the independent oracle
uses raw observer/native inputs. Production commands use the real binary unchanged.
``controlled_arithmetic`` separately executes the real collector in a private subprocess
with explicit reviewed fixture frequency and a distinct actual methodology namespace;
its published snapshot is then read by the real savings/status CLI.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .bounded_capture import run_bounded
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
        return run_bounded(
            [str(self.binary), *args], input=stdin, text=True, env=env,
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
    """Real ``ssgrep index``; returns (returncode, elapsed_ms, bounded diagnostic cause)."""
    started = time.monotonic()
    done = product.run("index", "--quiet", "--json", timeout=timeout)
    diagnostic = (done.stderr.strip() or done.stdout.strip())[-1200:]
    if done.returncode != 0:
        for line in reversed(done.stdout.splitlines()):
            if not line.startswith("{"):
                continue
            try:
                body = json.loads(line)
            except ValueError:
                continue
            if not isinstance(body, dict):
                continue
            error = body.get("error", body)
            if isinstance(error, dict) and isinstance(error.get("message"), str):
                kind = error.get("type") or body.get("condition")
                message = error["message"]
                diagnostic = (f"{kind}: {message}" if kind else message)[:1200]
                break
    return done.returncode, int((time.monotonic() - started) * 1000), diagnostic


def index_status(product: Product) -> dict[str, Any]:
    return product.data("status")


# ------------------------------------------------------------------- isolated data copies


def copy_accounting(source: Path, destination: Path, *, journal_only: bool = False) -> Path:
    """Copy isolated accounting state, retaining a read-through locator-index authority.

    ``journal_only`` keeps policy, key and journal, rebuilding derived accounting state.
    Its index alias refers to the same owned baseline index: accounting commands read it
    for native source locators, never copy its transcript corpus or rewrite its paths.
    This destination is for accounting commands only, not ``index`` or other index writers.
    An absent source index stays absent and therefore cannot manufacture source evidence.
    """
    src = source / "accounting"
    dst = destination / "accounting"
    if dst.exists():
        raise ProductError(f"refusing to overwrite {dst}")
    locator_index = None
    if journal_only:
        index = source / "lancedb"
        try:
            info = index.lstat()
        except FileNotFoundError:
            pass
        else:
            if (
                not stat.S_ISDIR(info.st_mode)
                or info.st_uid != os.geteuid()
                or info.st_mode & 0o022
            ):
                raise ProductError(f"refusing to alias an unapproved owned index: {index}")
            locator_index = index.resolve(strict=True)
        if (destination / "lancedb").exists() or (destination / "lancedb").is_symlink():
            raise ProductError(f"refusing to overwrite {destination / 'lancedb'}")
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
    if locator_index is not None:
        (destination / "lancedb").symlink_to(locator_index, target_is_directory=True)
    return destination


def native_source_proof(product: Product, ref: str, *, outside_root: Path) -> dict[str, Any]:
    """Probe actual production index/native authority even when rarity grants no credit."""
    done = run_bounded(
        [sys.executable, "-m", "eval.accounting_airline._native_source_probe"],
        input=json.dumps({"ref": ref, "outside_root": str(outside_root)}),
        text=True,
        env={**product.env, "PYTHONPATH": str(REPO_ROOT)},
        cwd=product.cwd,
        timeout=CLI_TIMEOUT,
        check=False,
    )
    if done.returncode != 0:
        raise ProductError(f"production source authority probe failed: {done.stderr[-1200:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def reviewed_view_documents(
    tokens: Sequence[str],
    *,
    common_tokens: Sequence[str] = (),
) -> list[str]:
    """Explicit closed-world fixture: reviewed once, observed non-reviewed identifiers common.

    ``common_tokens`` is supplied only from independent visible-card vocabulary. No
    implicit absent-token fallback exists: anything not supplied remains unknown.
    These synthetic documents never enter the real index or production-policy lane.
    """
    reviewed = tuple(dict.fromkeys(tokens))
    common = " ".join(sorted(set(common_tokens) - set(reviewed)))
    docs = [f"filler {i}" for i in range(FILLER_DOCS - len(reviewed))]
    return [
        f"{document} {common}"
        for document in (*docs, *(f"reviewed fixture {token}" for token in reviewed))
    ]


def controlled_arithmetic(
    product: Product,
    tokens: Sequence[str],
    *flags: str,
    source_data: Path,
    source_roots: Sequence[Path] = (),
    common_tokens: Sequence[str] = (),
) -> Collected:
    """REAL collection with private, explicitly named arithmetic-fixture inputs.

    Unlike a stale pinned view, the controlled frequency is supplied at the collector's
    actual resolution boundary. Journal reconciliation, source identity/scope/digests,
    native adoption, partition reduction, checkpoint publication and CLI readers remain
    real. This fixture is never a production-policy result or eligible for export.
    """
    started = time.monotonic()
    reviewed = tuple(dict.fromkeys(tokens))
    common = tuple(sorted(set(common_tokens) - set(reviewed)))
    from ssgrep.accounting.frequency import build_view

    from ._arithmetic_collect import fixture_methodology

    fixture_view = build_view(reviewed_view_documents(reviewed, common_tokens=common))
    method = fixture_methodology(fixture_view)
    method_info = {"label": method.label, "hash": method.hash}
    frozen_inputs = {
        "eligibility_model": "observed_identifier_whitelist_v1",
        "frequency_view_id": fixture_view.version_id,
        "reviewed_count": len(reviewed),
        "observed_common_count": len(common),
        "input_digest": hashlib.sha256(
            json.dumps(
                {"reviewed": sorted(reviewed), "observed_common": common},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
        ).hexdigest(),
    }
    request = {
        "tokens": reviewed,
        "common_tokens": common,
        "source_data": str(source_data),
        "source_roots": [str(path) for path in source_roots],
        "program": str(product.binary),
        "fixture_inputs": frozen_inputs,
        "methodology": method_info,
    }
    done = run_bounded(
        [sys.executable, "-m", "eval.accounting_airline._arithmetic_collect", *flags, "--json"],
        input=json.dumps(request),
        text=True,
        env={**product.env, "PYTHONPATH": str(REPO_ROOT)},
        cwd=product.cwd,
        timeout=CLI_TIMEOUT,
        check=False,
    )
    lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
    if done.returncode != 0 or not lines:
        raise ProductError(
            f"controlled arithmetic collection failed ({done.returncode}): "
            f"{done.stderr[-1200:] or done.stdout[-1200:]}"
        )
    body = json.loads(lines[-1])
    if body.get("ok") is False:
        raise ProductError(f"controlled arithmetic collection failed: {json.dumps(body)[:1200]}")
    code, document = product.json("savings")
    if code != 0:
        raise ProductError(
            f"controlled arithmetic savings failed ({code}): {json.dumps(document)[:1200]}"
        )
    provenance = document.get("data", document).get("provenance") or {}
    if provenance.get("methodology_id") != method.hash:
        raise ProductError(
            "controlled arithmetic published outside its fixture methodology namespace"
        )
    document["lab_valuation_profile"] = "airline-lab-arithmetic-v1"
    document["lab_valuation_methodology"] = method_info
    document["lab_fixture_inputs"] = frozen_inputs
    return Collected(
        body.get("data", body),
        document.get("data", document),
        product.data("metrics", "status"),
        int((time.monotonic() - started) * 1000),
        document,
    )


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
