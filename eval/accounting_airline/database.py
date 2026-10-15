"""Ephemeral, run-owned PostgreSQL 17 for the airline lab.

No Python database driver exists in this environment and none may be added: every SQL
statement runs through ``docker exec <container> psql -v ON_ERROR_STOP=1`` and the
output is parsed as JSON. The module accepts no caller-supplied DSN; the only database
it can reach is the container it started itself, identified by a run label.
"""

from __future__ import annotations

import json
import re
import secrets
import shutil
import subprocess
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from . import dataset
from .bounded_capture import OutputLimitExceeded, run_bounded
from .gates import BLOCKED, PASS, Gate, blocked

RUN_LABEL = "ssgrep-lab-run"
KIND_LABEL = "ssgrep-lab"
KIND_VALUE = "airline"
NAME_PREFIX = "ssgrep-airline-"
ADMIN_DB = "lab_admin"
VERIFIER = "lab_verifier"
READER = "lab_reader"
WRITER = "lab_writer"
ROLES = (VERIFIER, READER, WRITER)
INTERNAL_PORT = 5432

# Recorded in the run manifest: timeouts the lab imposes on every non-verifier role.
ROLE_LIMITS: dict[str, dict[str, str]] = {
    READER: {
        "statement_timeout": "5s",
        "lock_timeout": "2s",
        "idle_in_transaction_session_timeout": "10s",
    },
    WRITER: {
        "statement_timeout": "15s",
        "lock_timeout": "5s",
        "idle_in_transaction_session_timeout": "10s",
    },
}
VERIFIER_OPTIONS = "-c statement_timeout=30s -c lock_timeout=10s"

_IDENT = re.compile(r"^[a-z][a-z0-9_]{0,62}$")
SQLSTATE_RE = re.compile(r"ERROR:\s+([0-9A-Z]{5}):")


class LabError(RuntimeError):
    """The lab could not do what it was asked to do (never silently ignored)."""


class ForeignDatabaseRefused(LabError):
    """A caller tried to point the lab at a database it does not own."""


class OwnershipError(LabError):
    """Teardown refused: the resource does not carry this run's ownership labels."""


class SqlError(LabError):
    def __init__(self, message: str, sqlstate: str | None, stderr: str) -> None:
        super().__init__(message)
        self.sqlstate = sqlstate
        self.stderr = stderr


def refuse_external_dsn(dsn: str | None) -> None:
    """The lab never connects to an address it did not create."""
    if dsn:
        raise ForeignDatabaseRefused(
            "the airline lab only drives its own labelled container; a caller-supplied "
            "DSN/address is refused and was not contacted"
        )


def _docker_raw(
    *args: str,
    timeout: float = 60,
    input_text: str | None = None,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """The only place the lab spawns the docker CLI."""
    return run_bounded(
        ["docker", *args],
        text=True,
        timeout=timeout,
        input=input_text,
        env=None if env is None else dict(env),
        check=False,
    )


REMOVAL_VERBS = frozenset({"rm", "stop", "kill"})
_VALUE_FLAGS = frozenset({"-t", "--time", "-s", "--signal"})


@dataclass(frozen=True)
class RemovalRecord:
    """One container the lab asked docker to remove/stop/kill, identified BEFORE the call."""

    verb: str
    ref: str
    container_id: str | None  # None: the container was not inspectable beforehand
    name: str | None
    run_label: str | None
    kind_label: str | None
    succeeded: bool

    @property
    def identified(self) -> bool:
        return self.container_id is not None

    def owned_by(self, run_id: str) -> bool:
        """True only for this run's own container: label, kind label and name all match."""
        return (
            self.identified
            and self.run_label == run_id
            and self.kind_label == KIND_VALUE
            and self.name == "/" + NAME_PREFIX + run_id
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "verb": self.verb,
            "container_id": self.container_id or self.ref,
            "name": self.name,
            "run_label": self.run_label,
            "succeeded": self.succeeded,
        }


_removals: list[RemovalRecord] = []
_removals_lock = threading.Lock()


def removal_mark() -> int:
    """Position in the removal ledger; pass it to ``removals_since`` to audit one run."""
    with _removals_lock:
        return len(_removals)


def removals_since(mark: int) -> list[RemovalRecord]:
    with _removals_lock:
        return list(_removals[mark:])


def _removal_targets(args: Sequence[str]) -> tuple[str | None, list[str]]:
    """(verb, container refs) when ``args`` is a docker rm/stop/kill; else (None, [])."""
    words = list(args)
    if words[:1] == ["container"]:
        words = words[1:]
    if not words or words[0] not in REMOVAL_VERBS:
        return None, []
    targets: list[str] = []
    skip = False
    for token in words[1:]:
        if skip:
            skip = False
        elif token in _VALUE_FLAGS:
            skip = True
        elif not token.startswith("-"):
            targets.append(token)
    return words[0], targets


def _identify(ref: str) -> tuple[str | None, str | None, str | None, str | None]:
    """(id, name, run label, kind label) of a container, read without side effects."""
    try:
        probe = _docker_raw("inspect", ref, timeout=30)
        if probe.returncode != 0:
            return None, None, None, None
        info = json.loads(probe.stdout)[0]
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
        return None, None, None, None
    labels = info.get("Config", {}).get("Labels") or {}
    return info.get("Id"), info.get("Name"), labels.get(RUN_LABEL), labels.get(KIND_LABEL)


def _docker(
    *args: str,
    timeout: float = 60,
    input_text: str | None = None,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run docker for the lab. Choke point: every rm/stop/kill target is recorded in the
    removal ledger with its ownership identity captured before the call, so cleanup can be
    audited from what the lab actually removed rather than from unrelated daemon churn."""
    verb, targets = _removal_targets(args)
    identities = [(ref, _identify(ref)) for ref in targets] if verb else []
    result = _docker_raw(*args, timeout=timeout, input_text=input_text, env=env)
    if verb:
        with _removals_lock:
            for ref, (cid, name, run_label, kind) in identities:
                _removals.append(
                    RemovalRecord(verb, ref, cid, name, run_label, kind, result.returncode == 0)
                )
    return result


def list_containers(label_filter: str | None = None) -> set[str]:
    """Full ids of all containers, optionally limited by a ``label=...`` filter."""
    args = ["ps", "-aq", "--no-trunc"]
    if label_filter:
        args += ["--filter", f"label={label_filter}"]
    return set(_docker(*args).stdout.split())


def preflight() -> Gate:
    """BLOCKED (never PASS, never an exception) if docker or the pinned image is missing."""
    name = "database.prerequisites"
    if shutil.which("docker") is None:
        return blocked(name, "docker CLI not found on PATH")
    try:
        info = _docker("version", "--format", "{{.Server.Version}}", timeout=20)
    except OutputLimitExceeded as exc:
        return Gate(name, BLOCKED, str(exc), {"capture": exc.capture.metadata()})
    except (OSError, subprocess.TimeoutExpired) as exc:
        return blocked(name, f"docker daemon unreachable: {type(exc).__name__}")
    if info.returncode != 0:
        return blocked(name, "docker daemon unreachable: " + info.stderr.strip()[:200])
    try:
        image = _docker("image", "inspect", dataset.IMAGE_REF, "--format", "{{.Id}}", timeout=20)
    except OutputLimitExceeded as exc:
        return Gate(name, BLOCKED, str(exc), {"capture": exc.capture.metadata()})
    if image.returncode != 0 or image.stdout.strip() != dataset.IMAGE_DIGEST:
        return blocked(
            name,
            f"image {dataset.IMAGE_REF} is not present locally (the lab never pulls "
            "or substitutes another image)",
        )
    return Gate(name, PASS, "", {"docker_server": info.stdout.strip(), "image": dataset.IMAGE_REF})


@dataclass(frozen=True)
class PsqlResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def sqlstate(self) -> str | None:
        match = SQLSTATE_RE.search(self.stderr)
        return match.group(1) if match else None

    def last_json(self) -> Any:
        lines = [ln for ln in self.stdout.splitlines() if ln.strip()]
        if not lines:
            raise LabError("psql produced no output to parse")
        return json.loads(lines[-1], parse_float=Decimal)


def _check_ident(value: str) -> str:
    if not _IDENT.match(value):
        raise ValueError(f"unsafe identifier {value!r}")
    return value


@dataclass
class LabInstance:
    run_id: str
    container_id: str
    host_port: int
    _passwords: dict[str, str] = field(repr=False)
    sealed: set[str] = field(default_factory=set)
    removed: bool = False

    # -- lifecycle -------------------------------------------------------------------

    @classmethod
    def start(cls, run_id: str | None = None, *, ready_timeout: float = 60.0) -> LabInstance:
        run_id = run_id or secrets.token_hex(6)
        _check_ident("r" + run_id)
        name = NAME_PREFIX + run_id
        passwords = {role: secrets.token_urlsafe(18) for role in ROLES}
        env = _clean_env({"POSTGRES_PASSWORD": passwords[VERIFIER]})
        proc = _docker(
            "run", "-d", "--pull", "never",
            "--name", name,
            "--label", f"{RUN_LABEL}={run_id}",
            "--label", f"{KIND_LABEL}={KIND_VALUE}",
            "-p", f"127.0.0.1::{INTERNAL_PORT}",
            "-e", "POSTGRES_PASSWORD",
            "-e", f"POSTGRES_USER={VERIFIER}",
            "-e", f"POSTGRES_DB={ADMIN_DB}",
            "--tmpfs", "/var/lib/postgresql/data:rw,size=768m",
            "--security-opt", "no-new-privileges",
            "--memory", "1536m", "--pids-limit", "256",
            dataset.IMAGE_REF,
            "-c", "fsync=off", "-c", "max_connections=60",
            "-c", "max_wal_size=96MB", "-c", "min_wal_size=32MB",
            "-c", "password_encryption=scram-sha-256",
            env=env,
            timeout=120,
        )  # fmt: skip
        if proc.returncode != 0:
            raise LabError("docker run failed: " + proc.stderr.strip()[:300])
        container_id = proc.stdout.strip()
        port = _published_port(container_id)
        instance = cls(run_id, container_id, port, passwords)
        try:
            instance._check_ownership()
            instance._wait_ready(ready_timeout)
            instance._enforce_password_auth()
        except BaseException:
            instance.teardown()
            raise
        return instance

    @property
    def container_name(self) -> str:
        return NAME_PREFIX + self.run_id

    def inspect(self) -> dict[str, Any]:
        proc = _docker("inspect", self.container_id)
        if proc.returncode != 0:
            raise LabError("container not inspectable: " + proc.stderr.strip()[:200])
        return json.loads(proc.stdout)[0]

    def _check_ownership(self) -> dict[str, Any]:
        """Raise unless the container carries exactly this run's identity."""
        info = self.inspect()
        labels = info["Config"].get("Labels") or {}
        if (
            info["Id"] != self.container_id
            or info["Name"] != "/" + self.container_name
            or labels.get(RUN_LABEL) != self.run_id
            or labels.get(KIND_LABEL) != KIND_VALUE
        ):
            raise OwnershipError(
                f"container {self.container_id[:12]} does not carry run label {self.run_id}"
            )
        return info

    def binding(self) -> dict[str, Any]:
        info = self._check_ownership()
        bound = info["NetworkSettings"]["Ports"][f"{INTERNAL_PORT}/tcp"]
        return {"host_ips": sorted({b["HostIp"] for b in bound}), "port": self.host_port}

    def _wait_ready(self, timeout: float) -> None:
        import time

        deadline = time.monotonic() + timeout
        last = ""
        while time.monotonic() < deadline:
            res = self.psql(ADMIN_DB, "SELECT 1;", timeout=10, role=VERIFIER, check=False)
            if res.ok:
                return
            last = res.stderr.strip()[:200]
            time.sleep(0.25)
        raise LabError("postgres did not become ready: " + last)

    def _enforce_password_auth(self) -> None:
        """Replace the image's loopback ``trust`` rules so the random credentials are real."""
        fix = _docker(
            "exec", self.container_id, "sh", "-c",
            "sed -i 's/[[:space:]]trust$/ scram-sha-256/' \"$PGDATA/pg_hba.conf\"",
        )  # fmt: skip
        if fix.returncode != 0:
            raise LabError("could not harden pg_hba.conf: " + fix.stderr.strip()[:200])
        self.psql(ADMIN_DB, "SELECT pg_reload_conf();", role=VERIFIER)

    def teardown(self) -> bool:
        """Remove only this run's own container; refuse anything lacking the run label."""
        if self.removed:
            return False
        probe = _docker("inspect", self.container_id)
        if probe.returncode != 0:  # already gone: nothing of ours remains
            self.removed = True
            return False
        self._check_ownership()
        rm = _docker("rm", "-f", "-v", self.container_id, timeout=120)
        if rm.returncode != 0:
            raise LabError("docker rm failed: " + rm.stderr.strip()[:200])
        self.removed = True
        return True

    @staticmethod
    def remove_run(run_id: str) -> list[str]:
        """Last-resort cleanup by run label; every candidate is ownership-checked first."""
        listing = _docker("ps", "-aq", "--no-trunc", "--filter", f"label={RUN_LABEL}={run_id}")
        removed = []
        for cid in listing.stdout.split():
            info = json.loads(_docker("inspect", cid).stdout)[0]
            labels = info["Config"].get("Labels") or {}
            if (
                labels.get(RUN_LABEL) == run_id
                and labels.get(KIND_LABEL) == KIND_VALUE
                and info["Name"] == "/" + NAME_PREFIX + run_id
            ):
                if _docker("rm", "-f", "-v", cid, timeout=120).returncode == 0:
                    removed.append(cid)
        return removed

    # -- SQL transport ---------------------------------------------------------------

    def _command(
        self, database: str, role: str, variables: Mapping[str, str] | None, app: str
    ) -> tuple[list[str], dict[str, str]]:
        if role not in ROLES:
            raise ValueError(f"unknown lab role {role!r}")
        env = _clean_env({"PGPASSWORD": self._passwords[role], "PGAPPNAME": app})
        cmd = ["docker", "exec", "-i", "-e", "PGPASSWORD", "-e", "PGAPPNAME"]
        if role == VERIFIER:
            env["PGOPTIONS"] = VERIFIER_OPTIONS
            cmd += ["-e", "PGOPTIONS"]
        cmd += [
            self.container_id, "psql", "-X", "-q", "-At",
            "-h", "127.0.0.1", "-p", str(INTERNAL_PORT),
            "-U", role, "-d", database,
            "-v", "ON_ERROR_STOP=1", "-v", "VERBOSITY=verbose",
        ]  # fmt: skip
        for key, value in (variables or {}).items():
            cmd += ["-v", f"{_check_ident(key)}={value}"]
        return [*cmd, "-f", "-"], env

    def psql(
        self,
        database: str,
        sql: str,
        *,
        role: str = VERIFIER,
        variables: Mapping[str, str] | None = None,
        timeout: float = 60,
        app: str = "lab",
        check: bool = True,
    ) -> PsqlResult:
        cmd, env = self._command(database, role, variables, app)
        try:
            proc = run_bounded(
                cmd, input=sql, text=True, timeout=timeout, env=env,
                check=False,
            )  # fmt: skip
        except subprocess.TimeoutExpired as exc:
            raise LabError(f"psql exceeded {timeout}s client timeout") from exc
        result = PsqlResult(proc.returncode, proc.stdout, proc.stderr)
        if check and not result.ok:
            raise SqlError(
                f"psql failed ({result.returncode}): {result.stderr.strip()[:400]}",
                result.sqlstate,
                result.stderr,
            )
        return result

    def popen(
        self,
        database: str,
        *,
        role: str = VERIFIER,
        variables: Mapping[str, str] | None = None,
        app: str = "lab",
    ) -> subprocess.Popen[str]:
        """A real independent psql client; the caller writes the script to its stdin."""
        cmd, env = self._command(database, role, variables, app)
        return subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=env, start_new_session=True,
        )  # fmt: skip

    def rows(
        self,
        database: str,
        sql: str,
        *,
        role: str = VERIFIER,
        variables: Mapping[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """Run one SELECT and return its rows as dicts (JSON-parsed, no float rounding)."""
        body = sql.strip().rstrip(";")
        wrapped = (
            f"SELECT coalesce(jsonb_agg(to_jsonb(lab_row)), '[]'::jsonb) FROM ({body}) lab_row;"
        )
        return self.psql(database, wrapped, role=role, variables=variables).last_json()

    def value(self, database: str, sql: str, *, role: str = VERIFIER) -> Any:
        body = sql.strip().rstrip(";")
        return self.psql(
            database, f"SELECT to_jsonb(lab_row) FROM ({body}) lab_row;", role=role
        ).last_json()

    # -- databases ------------------------------------------------------------------

    def database_name(self, label: str) -> str:
        return _check_ident(f"airline_{self.run_id}_{label}")

    def clone(self, label: str, template_label: str = "base") -> str:
        """``CREATE DATABASE ... TEMPLATE``: scenarios never stack on one mutated state."""
        name = self.database_name(label)
        template = self.database_name(template_label)
        self.psql(
            ADMIN_DB,
            f"CREATE DATABASE {name} TEMPLATE {template};\n"
            f"REVOKE CONNECT ON DATABASE {name} FROM PUBLIC;\n"
            f"GRANT CONNECT ON DATABASE {name} TO {READER}, {WRITER};\n",
        )
        return name

    def drop_clone(self, label: str) -> None:
        name = self.database_name(label)
        self.psql(ADMIN_DB, f"DROP DATABASE IF EXISTS {name} WITH (FORCE);")

    def read_state(self, database: str) -> dict[str, list[dict[str, Any]]]:
        return self.value(database, dataset.state_select_sql())["state"]

    def provision_baseline(self, extra_sql: Sequence[str] = ()) -> BaselineInfo:
        """Create, seed, restrict, verify and seal the recorded baseline database."""
        base = self.database_name("base")
        self.psql(ADMIN_DB, self._cluster_setup_sql(base), variables=self._role_variables())
        self.psql(base, dataset.seed_sql())
        self.psql(base, _restrict_sql(base))
        for sql in extra_sql:
            self.psql(base, sql)
        state = self.read_state(base)
        counts = dataset.row_counts(state)
        digests = dataset.logical_digests(state)
        info = BaselineInfo(base, counts, digests, dataset.overall_digest(digests), state)
        self.psql(ADMIN_DB, f"ALTER DATABASE {base} WITH ALLOW_CONNECTIONS false;")
        self.sealed.add(base)
        return info

    def _role_variables(self) -> dict[str, str]:
        return {
            "reader_pw": self._passwords[READER],
            "writer_pw": self._passwords[WRITER],
        }

    def _cluster_setup_sql(self, base: str) -> str:
        lines = [
            f"CREATE ROLE {READER} LOGIN PASSWORD :'reader_pw' NOSUPERUSER NOCREATEDB "
            "NOCREATEROLE;",
            f"CREATE ROLE {WRITER} LOGIN PASSWORD :'writer_pw' NOSUPERUSER NOCREATEDB "
            "NOCREATEROLE;",
        ]
        for role, limits in ROLE_LIMITS.items():
            for key, value in limits.items():
                lines.append(f"ALTER ROLE {role} SET {key} = '{value}';")
        lines += [
            f"ALTER ROLE {READER} SET default_transaction_read_only = on;",
            f"ALTER ROLE {READER} SET search_path = {dataset.SCHEMA}, pg_catalog;",
            f"ALTER ROLE {WRITER} SET search_path = {dataset.SCHEMA}, pg_catalog;",
            f"CREATE DATABASE {base};",
            f"REVOKE ALL ON DATABASE {base} FROM PUBLIC;",
            f"GRANT CONNECT ON DATABASE {base} TO {READER}, {WRITER};",
            f"REVOKE ALL ON DATABASE {ADMIN_DB} FROM PUBLIC;",
        ]
        return "\n".join(lines) + "\n"

    def manifest(self) -> dict[str, Any]:
        """Run identity for the private report; credentials are deliberately absent."""
        return {
            "run_id": self.run_id,
            "container_name": self.container_name,
            "container_id": self.container_id,
            "image": dataset.IMAGE_REF,
            "bind": self.binding(),
            "labels": {RUN_LABEL: self.run_id, KIND_LABEL: KIND_VALUE},
            "roles": {
                role: ROLE_LIMITS.get(role, {"verifier_options": VERIFIER_OPTIONS})
                for role in ROLES
            },
            "as_of": dataset.AS_OF,
            "schema_digest": dataset.schema_digest(),
        }


@dataclass(frozen=True)
class BaselineInfo:
    database: str
    counts: dict[str, int]
    digests: dict[str, str]
    digest: str
    state: dict[str, list[dict[str, Any]]] = field(repr=False)


def _restrict_sql(database: str) -> str:
    """Least privilege: reader only reads; the L4 writer moves coupons and logs requests.

    Row locks (``FOR UPDATE``) need an UPDATE privilege, so the writer holds a
    column-level UPDATE on inventory.capacity guarded by an immutability trigger.
    """
    schema = dataset.SCHEMA
    return f"""
REVOKE ALL ON SCHEMA public FROM PUBLIC;
REVOKE TEMPORARY ON DATABASE {database} FROM PUBLIC;
GRANT USAGE ON SCHEMA {schema} TO {READER}, {WRITER};
GRANT SELECT ON ALL TABLES IN SCHEMA {schema} TO {READER}, {WRITER};
GRANT UPDATE (flight_id, status, version) ON {schema}.coupons TO {WRITER};
GRANT INSERT ON {schema}.rebooking_requests TO {WRITER};
GRANT UPDATE (capacity) ON {schema}.inventory TO {WRITER};
CREATE FUNCTION {schema}.inventory_capacity_immutable() RETURNS trigger
  LANGUAGE plpgsql AS $$
BEGIN
  IF NEW.capacity IS DISTINCT FROM OLD.capacity THEN
    RAISE EXCEPTION 'inventory capacity is immutable for lab roles' USING ERRCODE = '42501';
  END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER inventory_capacity_immutable BEFORE UPDATE ON {schema}.inventory
  FOR EACH ROW EXECUTE FUNCTION {schema}.inventory_capacity_immutable();
"""


def _clean_env(extra: Mapping[str, str]) -> dict[str, str]:
    import os

    env = {k: v for k, v in os.environ.items() if not k.startswith(("PG", "POSTGRES"))}
    env.update(extra)
    return env


def _published_port(container_id: str) -> int:
    proc = _docker("port", container_id, f"{INTERNAL_PORT}/tcp")
    match = re.search(r"127\.0\.0\.1:(\d+)", proc.stdout)
    if proc.returncode != 0 or not match:
        _docker("rm", "-f", "-v", container_id)
        raise LabError("container did not publish a loopback-only port")
    return int(match.group(1))
