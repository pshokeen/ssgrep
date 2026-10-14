"""Run manifest (pins, identities, limitations) and the live-lane preflight gates.

The manifest records WHAT ran and under which pins, never secrets: a credential is a
read-only reference whose content is never read, copied or printed (only the path and a digest
of the path string are recorded). ``preflight_live`` returns named gates and never raises,
never contacts a model/provider and never starts an agent.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import database, dataset, sandbox
from .gates import BLOCKED, NOT_RUN, PASS, Gate
from .roots import LabRoots

PRODUCTION_METHODOLOGY = "production-policy"
ARITHMETIC_METHODOLOGY = "airline-lab-arithmetic-v1"
STANDIN_LABEL = "scripted-standin"
MODES = ("standin", "live", "deterministic")
THRESHOLDS_MILLI = {"conservative": 8000, "central": 6000, "optimistic": 4000}
SUPPORTED_LIVE_RUNTIME = "omp"
OMP_REQUIRED_FLAGS = ("--session-dir", "--tools", "--print", "--cwd")
_EMBED_SOURCE = Path(__file__).resolve().parents[2] / "src" / "ssgrep" / "indexing" / "embed.py"
_VERSION_RE = re.compile(r"\d+\.\d+\.\d+[0-9A-Za-z.+-]*")
_BINARY_TIMEOUT = 30


class ManifestError(ValueError):
    """The requested run cannot be recorded honestly (for example live without authorization)."""


@dataclass(frozen=True)
class CredentialRef:
    """A read-only reference to a credential file; its content is never read by the lab."""

    name: str
    path: str
    kind: str

    @property
    def path_digest(self) -> str:
        return hashlib.sha256(self.path.encode()).hexdigest()

    def as_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "path": self.path,
            "kind": self.kind,
            "path_sha256": self.path_digest,
        }


def prompt_hashes(prompts: Mapping[str, str]) -> dict[str, str]:
    return {
        name: hashlib.sha256(text.encode("utf-8")).hexdigest() for name, text in prompts.items()
    }


def ssgrep_version() -> str:
    try:
        return importlib.metadata.version("ssgrep")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _seed_digest() -> str:
    return dataset.overall_digest(dataset.seed_digests())


def _methodology() -> dict[str, Any]:
    return {
        "production": PRODUCTION_METHODOLOGY,
        "arithmetic": ARITHMETIC_METHODOLOGY,
        "thresholds_milli": dict(THRESHOLDS_MILLI),
        "arithmetic_threshold_milli": 0,
    }


def _timeouts() -> dict[str, dict[str, str]]:
    timeouts = {role: dict(limits) for role, limits in database.ROLE_LIMITS.items()}
    timeouts[database.VERIFIER] = {"options": database.VERIFIER_OPTIONS}
    return timeouts


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    mode: str
    runtime: str
    counts_toward_live_gate: bool
    live_authorized: bool
    model: str | None
    runtime_version: str | None
    provider: str | None
    sampling: Mapping[str, str]
    ssgrep_version: str
    seed_digest: str
    schema_digest: str
    image_ref: str
    image_digest: str
    prompt_hashes: Mapping[str, str]
    methodology: Mapping[str, Any]
    timeouts: Mapping[str, Mapping[str, str]]
    credentials: tuple[CredentialRef, ...]
    roots: Mapping[str, str]
    private_artifact_dir: str
    created_utc: str
    limitations: tuple[Mapping[str, str], ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "runtime": self.runtime,
            "counts_toward_live_gate": self.counts_toward_live_gate,
            "live_authorized": self.live_authorized,
            "model": self.model,
            "runtime_version": self.runtime_version,
            "provider": self.provider,
            "sampling": dict(self.sampling),
            "ssgrep_version": self.ssgrep_version,
            "seed_digest": self.seed_digest,
            "schema_digest": self.schema_digest,
            "image_ref": self.image_ref,
            "image_digest": self.image_digest,
            "prompt_hashes": dict(self.prompt_hashes),
            "methodology": json.loads(json.dumps(self.methodology)),
            "timeouts": {role: dict(limits) for role, limits in self.timeouts.items()},
            "credentials": [ref.as_dict() for ref in self.credentials],
            "roots": dict(self.roots),
            "private_artifact_dir": self.private_artifact_dir,
            "created_utc": self.created_utc,
            "limitations": [dict(item) for item in self.limitations],
        }

    def digest(self) -> str:
        payload = self.as_dict()
        payload.pop("created_utc")
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def write(self, path: Path) -> None:
        data = json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(data)


def _limitation_dict(item: Any) -> dict[str, str]:
    if isinstance(item, sandbox.Limitation):
        return item.as_dict()
    return {str(key): str(value) for key, value in dict(item).items()}


def build_manifest(
    *,
    run_id: str,
    mode: str,
    runtime: str,
    roots: LabRoots,
    prompts: Mapping[str, str],
    authorize_live: bool = False,
    model: str | None = None,
    runtime_version: str | None = None,
    provider: str | None = None,
    sampling: Mapping[str, str] | None = None,
    credentials: Sequence[CredentialRef] = (),
    limitations: Sequence[Any] = (),
) -> RunManifest:
    if mode not in MODES:
        raise ManifestError(f"unknown mode {mode!r}; expected one of {MODES}")
    if mode == "live" and not authorize_live:
        raise ManifestError("live mode requires explicit live-agent authorization")
    if mode == "live" and runtime == STANDIN_LABEL:
        raise ManifestError(f"the {STANDIN_LABEL!r} runtime can never be recorded as a live run")
    return RunManifest(
        run_id=run_id,
        mode=mode,
        runtime=runtime,
        counts_toward_live_gate=mode == "live" and authorize_live and runtime != STANDIN_LABEL,
        live_authorized=authorize_live,
        model=model,
        runtime_version=runtime_version,
        provider=provider,
        sampling=dict(sampling or {}),
        ssgrep_version=ssgrep_version(),
        seed_digest=_seed_digest(),
        schema_digest=dataset.schema_digest(),
        image_ref=dataset.IMAGE_REF,
        image_digest=dataset.IMAGE_DIGEST,
        prompt_hashes=prompt_hashes(prompts),
        methodology=_methodology(),
        timeouts=_timeouts(),
        credentials=tuple(credentials),
        roots={
            "root": str(roots.root),
            "project": str(roots.project),
            "runtime_a": str(roots.runtime_a),
            "runtime_b": str(roots.runtime_b),
            "ssgrep_data": str(roots.ssgrep_data),
            "verifier": str(roots.verifier),
            "artifacts": str(roots.artifacts),
            "tools": str(roots.tools),
        },
        private_artifact_dir=str(roots.artifacts),
        created_utc=datetime.now(UTC).isoformat(),
        limitations=tuple(_limitation_dict(item) for item in limitations),
    )


def verify_pins(manifest: RunManifest, *, prompts: Mapping[str, str]) -> list[str]:
    """Recompute every pin; each returned string describes one drift (empty = pins hold)."""
    drift: list[str] = []
    current = {
        "seed_digest": _seed_digest(),
        "schema_digest": dataset.schema_digest(),
        "image_ref": dataset.IMAGE_REF,
        "image_digest": dataset.IMAGE_DIGEST,
        "ssgrep_version": ssgrep_version(),
    }
    for name, now in current.items():
        recorded = getattr(manifest, name)
        if recorded != now:
            drift.append(f"{name} drifted: recorded {recorded}, now {now}")
    now_hashes = prompt_hashes(prompts)
    for name in sorted(set(manifest.prompt_hashes) | set(now_hashes)):
        recorded = manifest.prompt_hashes.get(name)
        now = now_hashes.get(name)
        if recorded is None:
            drift.append(f"prompt {name!r} was not recorded in the manifest")
        elif now is None:
            drift.append(f"prompt {name!r} is recorded but no longer supplied")
        elif recorded != now:
            drift.append(f"prompt {name!r} changed: recorded {recorded}, now {now}")
    return drift


# ------------------------------------------------------------------------ live preflight


def _embed_model_pin() -> tuple[str, str]:
    """DEFAULT_MODEL_ID / DEFAULT_MODEL_REVISION parsed from source (torch is never imported)."""
    tree = ast.parse(_EMBED_SOURCE.read_text(encoding="utf-8"))
    found: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in (
                    "DEFAULT_MODEL_ID",
                    "DEFAULT_MODEL_REVISION",
                ):
                    found[target.id] = str(node.value.value)
    if set(found) != {"DEFAULT_MODEL_ID", "DEFAULT_MODEL_REVISION"}:
        raise LookupError(f"model pin constants not found in {_EMBED_SOURCE}")
    return found["DEFAULT_MODEL_ID"], found["DEFAULT_MODEL_REVISION"]


def snapshot_dir(model_cache: Path, model_id: str, revision: str) -> Path:
    return model_cache / "hub" / f"models--{model_id.replace('/', '--')}" / "snapshots" / revision


def _gate_authorization(authorize_live: bool) -> Gate:
    name = "live.authorization"
    if not authorize_live:
        return Gate(name, BLOCKED, "live-agent authorization was not given (authorize_live=False)")
    return Gate(name, PASS, "live-agent use explicitly authorized")


def _gate_runtime_binary(runtime: str) -> Gate:
    name = "live.runtime_binary"
    if runtime != SUPPORTED_LIVE_RUNTIME:
        return Gate(
            name,
            BLOCKED,
            f"unsupported runtime {runtime!r}: only {SUPPORTED_LIVE_RUNTIME!r} is verified",
        )
    binary = shutil.which("omp")
    if binary is None:
        return Gate(name, BLOCKED, "omp binary not found on PATH")
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", ""),
        "LANG": "C",
    }
    try:
        version = subprocess.run(
            [binary, "--version"],
            env=env,
            capture_output=True,
            text=True,
            timeout=_BINARY_TIMEOUT,
            check=False,
        )
        helped = subprocess.run(
            [binary, "--help"],
            env=env,
            capture_output=True,
            text=True,
            timeout=_BINARY_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return Gate(name, BLOCKED, f"omp could not be run: {type(exc).__name__}")
    match = _VERSION_RE.search(version.stdout + version.stderr)
    if version.returncode != 0 or match is None:
        return Gate(name, BLOCKED, f"omp --version unparseable (exit {version.returncode})")
    text = helped.stdout + helped.stderr
    missing = [flag for flag in OMP_REQUIRED_FLAGS if flag not in text]
    if not re.search(r"(^|\s)-p[,\s]", text):
        missing.append("-p")
    if helped.returncode != 0 or missing:
        return Gate(
            name,
            BLOCKED,
            "omp --help does not document required flag(s): " + ", ".join(missing or ["<help>"]),
            {"binary": binary, "version": match.group(0)},
        )
    return Gate(name, PASS, "", {"binary": binary, "version": match.group(0)})


def _gate_database() -> Gate:
    inner = database.preflight()
    return Gate("live.database", inner.status, inner.detail, inner.evidence)


def _gate_psql() -> Gate:
    name = "live.psql_client"
    binary = shutil.which("psql")
    if binary is None:
        return Gate(name, BLOCKED, "host psql client not found on PATH (needed by the SQL tool)")
    return Gate(name, PASS, "", {"binary": binary})


def _gate_sandbox() -> Gate:
    name = "live.sandbox"
    limitation = sandbox.availability()
    if limitation is not None:
        return Gate(
            name,
            BLOCKED,
            f"{limitation.name}: {limitation.reason}",
            {"limitation": limitation.as_dict()},
        )
    return Gate(name, PASS, "", {})


def _gate_model_cache(model_cache: Path | None) -> Gate:
    name = "live.model_cache"
    cache = model_cache if model_cache is not None else Path.home() / ".cache" / "huggingface"
    try:
        model_id, revision = _embed_model_pin()
    except (OSError, SyntaxError, LookupError) as exc:
        return Gate(name, BLOCKED, f"embedding model pin unreadable: {exc}")
    snapshot = snapshot_dir(cache, model_id, revision)
    evidence = {"model_id": model_id, "revision": revision, "snapshot": str(snapshot)}
    if not snapshot.is_dir():
        return Gate(name, BLOCKED, f"model snapshot directory missing: {snapshot}", evidence)
    absent = [
        str(snapshot / filename)
        for filename in ("model.safetensors", "modules.json")
        if not (snapshot / filename).is_file()
    ]
    if absent:
        return Gate(name, BLOCKED, "model snapshot file(s) missing: " + ", ".join(absent), evidence)
    return Gate(name, PASS, "", evidence)


def _gate_credentials(credential_paths: Sequence[Path]) -> Gate:
    name = "live.credentials"
    if not credential_paths:
        return Gate(name, NOT_RUN, "no credential references were supplied")
    problems: list[str] = []
    for path in credential_paths:
        if not path.is_file():
            problems.append(f"{path} is not an existing file")
        elif not os.access(path, os.R_OK):
            problems.append(f"{path} is not readable")
    evidence = {"paths": [str(path) for path in credential_paths]}
    if problems:
        return Gate(name, BLOCKED, "; ".join(problems), evidence)
    return Gate(name, PASS, "", evidence)


def preflight_live(
    *,
    runtime: str,
    authorize_live: bool,
    credential_paths: Sequence[Path] = (),
    model_cache: Path | None = None,
) -> list[Gate]:
    """Named live-lane prerequisite gates. Never raises, never contacts a provider."""
    return [
        _gate_authorization(authorize_live),
        _gate_runtime_binary(runtime),
        _gate_database(),
        _gate_psql(),
        _gate_sandbox(),
        _gate_model_cache(model_cache),
        _gate_credentials(credential_paths),
    ]
