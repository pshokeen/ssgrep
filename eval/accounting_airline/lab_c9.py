"""C9's isolated scrape/export acceptance over the stand-in's real collected snapshots."""

from __future__ import annotations

import hashlib
import http.client as http_client
import json
import os
import stat
import threading
import time
import traceback
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from ssgrep.accounting import export, export_ids, http, http_auth, http_views, metrics, policy
from ssgrep.accounting.ledger_contracts import DestinationPolicy, Signal, SnapshotEnvelope
from tests.accounting.lab_receiver import LabReceiver, Partial
from tests.accounting.otel_collector_lab import metric_points

from .gates import BLOCKED, FAIL, NOT_RUN, PASS, Gate
from .lab_product import Collected, Product

PROFILE = "personal-local"
DESTINATION = "airline-lab-c9"
PARTIAL_DESTINATION = "airline-lab-c9-partial"
CREDENTIAL_ENV = "SSGREP_C9_INGEST_TOKEN"
SERVER_TIMEOUT = 120.0


@dataclass(frozen=True)
class C9Outcome:
    collected: Collected
    snapshot_id: str | None
    snapshot_generation: int | None
    snapshot_before_id: str | None
    snapshot_before_generation: int | None
    scrape: Gate
    export: Gate
    evidence: dict[str, Any]

    @property
    def status(self) -> str:
        statuses = (self.scrape.status, self.export.status)
        if FAIL in statuses:
            return FAIL
        if BLOCKED in statuses:
            return BLOCKED
        return PASS if statuses == (PASS, PASS) else FAIL


@dataclass(frozen=True)
class _Reply:
    status: int
    headers: dict[str, str]
    body: bytes


class _BoundedWorker:
    """Worker composition probe; deliberately does not collect or mutate profile state."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.stopped = threading.Event()

    def run(self) -> None:
        self.started.set()
        self.stopped.wait(SERVER_TIMEOUT)

    def stop(self) -> None:
        self.stopped.set()


class _CheckFailure(RuntimeError):
    pass


def _failure_detail(error: Exception) -> str:
    if isinstance(error, _CheckFailure):
        return str(error)
    frames = traceback.extract_tb(error.__traceback__)
    if frames:
        frame = frames[-1]
        return f"integration_error:{type(error).__name__}@{frame.name}:{frame.lineno}"
    return f"integration_error:{type(error).__name__}"


@contextmanager
def _data_root(path: Path) -> Iterator[None]:
    previous = os.environ.get("SSGREP_DATA_DIR")
    os.environ["SSGREP_DATA_DIR"] = str(path)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("SSGREP_DATA_DIR", None)
        else:
            os.environ["SSGREP_DATA_DIR"] = previous


def _current(profile: str) -> SnapshotEnvelope:
    current, error = http_views.load_current(profile)
    if current is None:
        raise _CheckFailure(error or "snapshot_unavailable")
    return current.envelope


def _get(port: int, path: str, token: bytes | None, extra: dict[str, str] | None = None) -> _Reply:
    connection = http_client.HTTPConnection("127.0.0.1", port, timeout=5)
    headers = {"Host": f"127.0.0.1:{port}", **(extra or {})}
    if token is not None:
        headers["Authorization"] = f"Bearer {token.decode('ascii')}"
    try:
        connection.request("GET", path, headers=headers)
        response = connection.getresponse()
        body = response.read()
        return _Reply(response.status, {k.lower(): v for k, v in response.getheaders()}, body)
    finally:
        connection.close()


def _tree(path: Path) -> tuple[tuple[str, int, int, int, str | None], ...]:
    if not path.exists():
        return ()
    rows = []
    for entry in sorted(path.rglob("*")):
        info = entry.lstat()
        if stat.S_ISLNK(info.st_mode):
            digest = hashlib.sha256(os.fsencode(os.readlink(entry))).hexdigest()
        elif stat.S_ISREG(info.st_mode):
            with entry.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
        else:
            digest = None
        rows.append(
            (
                str(entry.relative_to(path)),
                info.st_mode & 0o777,
                info.st_size,
                info.st_mtime_ns,
                digest,
            )
        )
    return tuple(rows)


def source_digests(source_dirs: Mapping[str, Path]) -> dict[str, str]:
    """SHA-256 of every native-source file beneath each isolated runtime directory."""
    result: dict[str, str] = {}
    for actor, directory in sorted(source_dirs.items()):
        if not directory.is_dir() or directory.is_symlink():
            result[f"{actor}/<missing>"] = "missing"
            continue
        for path in sorted(directory.rglob("*")):
            info = path.lstat()
            if stat.S_ISDIR(info.st_mode):
                continue
            if stat.S_ISLNK(info.st_mode):
                digest = hashlib.sha256(os.fsencode(os.readlink(path))).hexdigest()
            elif stat.S_ISREG(info.st_mode):
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
            else:
                digest = hashlib.sha256(
                    f"special:{info.st_mode}:{info.st_size}".encode()
                ).hexdigest()
            result[f"{actor}/{path.relative_to(directory).as_posix()}"] = digest
    return result


def _document(reply: _Reply) -> dict[str, Any]:
    value = json.loads(reply.body)
    if not isinstance(value, dict):
        raise _CheckFailure("http_json_invalid")
    return value


def _preview(product: Product, destination: str, signal: str) -> tuple[dict[str, Any], str]:
    accounting = product.data_dir / "accounting"
    before = _tree(accounting)
    code, document = product.json(
        "metrics",
        "export",
        "--destination",
        destination,
        "--profile",
        PROFILE,
        "--signal",
        signal,
        "--preview",
    )
    if _tree(accounting) != before:
        raise _CheckFailure(f"{signal}_preview_changed_accounting_state")
    raw = document.get("data", document)
    data = raw.get("data", raw) if isinstance(raw, dict) else raw
    if code != 0:
        raise _CheckFailure(
            f"{signal}_preview_failed:exit_{code}:{document.get('condition', 'unknown')}"
        )
    if not isinstance(data, dict):
        raise _CheckFailure(f"{signal}_preview_invalid_document")
    if data.get("network") is not False:
        raise _CheckFailure(f"{signal}_preview_network_flag_invalid:{data.get('state', 'unknown')}")
    return data, json.dumps(document, sort_keys=True, separators=(",", ":"))


def _command(product: Product, secret: str, *args: str) -> tuple[dict[str, Any], str]:
    secured = product.with_env(**{CREDENTIAL_ENV: secret})
    code, document = secured.json(*args, "--profile", PROFILE)
    raw = document.get("data", document)
    data = raw.get("data", raw) if isinstance(raw, dict) else raw
    rendered = json.dumps(document, sort_keys=True, separators=(",", ":"))
    if code != 0 or document.get("ok") is False or not isinstance(data, dict):
        raise _CheckFailure("export_command_failed")
    return data, rendered


def _enable(product: Product, secret: str, destination: str, signal: str) -> str:
    _, output = _command(
        product,
        secret,
        "metrics",
        "export",
        "--destination",
        destination,
        "--signal",
        signal,
        "--enable",
        "--yes",
    )
    return output


def _send(
    product: Product, secret: str, destination: str, signal: str
) -> tuple[dict[str, Any], str]:
    return _command(
        product, secret, "metrics", "export", "--destination", destination, "--signal", signal
    )


def _expected_samples(
    envelope: SnapshotEnvelope,
) -> dict[tuple[str, tuple[tuple[str, str], ...]], int | Decimal]:
    configured = tuple(
        (destination.destination_id, tuple(signal.value for signal in destination.signals))
        for destination in policy.load_profile_policy(PROFILE).destinations
    )
    rows = http_views.delivery_rows(PROFILE, configured)
    samples = metrics.all_samples(envelope, http_views.export_health(rows))
    return {
        (sample.family, tuple(sorted(sample.labels))): (
            Decimal(sample.value) / Decimal(1_000_000)
            if sample.kind == "histogram_sum"
            else sample.value
        )
        for sample in samples
    }


def _sample_map(body: bytes, parse: Any) -> dict[tuple[str, tuple[tuple[str, str], ...]], Decimal]:
    actual: dict[tuple[str, tuple[tuple[str, str], ...]], Decimal] = {}
    for family in parse(body.decode("utf-8")):
        for sample in family.samples:
            if sample.timestamp is not None:
                raise _CheckFailure("scrape_has_explicit_timestamps")
            key = (sample.name, tuple(sorted(sample.labels.items())))
            if key in actual:
                raise _CheckFailure("scrape_duplicate_sample")
            actual[key] = Decimal(str(sample.value))
    return actual


class _OpenMetricsParser:
    def __init__(self, parse: Any) -> None:
        self.parse = parse

    def samples(self, body: bytes) -> dict[tuple[str, tuple[tuple[str, str], ...]], Decimal]:
        return _sample_map(body, self.parse)


class _PrometheusQuery:
    def __init__(self, prometheus: Any) -> None:
        self.prometheus = prometheus
        self.last_scrape: str | None = None

    def samples(self, _body: bytes) -> dict[tuple[str, tuple[tuple[str, str], ...]], Decimal]:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            target = self.prometheus.targets().get("c9")
            scrape = target.get("lastScrape") if target else None
            if scrape and scrape != self.last_scrape:
                self.last_scrape = scrape
                break
            time.sleep(0.1)
        else:
            raise _CheckFailure("prometheus_did_not_scrape")
        if Decimal(str(self.prometheus.scalar('up{job="c9"}'))) != 1:
            raise _CheckFailure("prometheus_target_down")
        result = self.prometheus.query('{__name__=~"ssgrep_.*"}')
        actual: dict[tuple[str, tuple[tuple[str, str], ...]], Decimal] = {}
        for row in result:
            labels = dict(row["metric"])
            if labels.pop("job", None) != "c9":
                continue
            labels.pop("instance", None)
            name = labels.pop("__name__", None)
            if name is None:
                raise _CheckFailure("prometheus_sample_name_missing")
            actual[(name, tuple(sorted((k, str(v)) for k, v in labels.items())))] = Decimal(
                str(row["value"][1])
            )
        return actual


def _scrape_validator(
    handle: http.ServerHandle, token: bytes, stack: ExitStack, evidence: dict[str, Any]
) -> tuple[_OpenMetricsParser | _PrometheusQuery, str | None]:
    try:
        from prometheus_client.openmetrics.parser import text_string_to_metric_families
    except ImportError:
        from tests.accounting import prom_lab

        reason = prom_lab.docker_available()
        if reason is not None:
            raise _CheckFailure(f"prometheus_prerequisite:{reason}") from None
        run_id = prom_lab.new_run_id()

        def verify_cleanup() -> None:
            leftovers = prom_lab.leftovers(run_id)
            evidence["prometheus_leftovers"] = len(leftovers)
            if leftovers:
                raise _CheckFailure("prometheus_container_leftover")

        stack.callback(verify_cleanup)
        prometheus = stack.enter_context(
            prom_lab.prometheus(
                run_id, [("c9", handle.port, "c9.token")], {"c9.token": token.decode("ascii")}
            )
        )
        prometheus.wait_scraped(["c9"])
        return _PrometheusQuery(prometheus), run_id
    return _OpenMetricsParser(text_string_to_metric_families), None


def _assert_samples(
    reply: _Reply,
    envelope: SnapshotEnvelope,
    validator: _OpenMetricsParser | _PrometheusQuery | None,
) -> dict[tuple[str, tuple[tuple[str, str], ...]], Decimal]:
    if reply.status != 200 or reply.headers.get("content-type") != http_views.CONTENT_TYPE:
        raise _CheckFailure("metrics_http_contract")
    if not reply.body.endswith(b"# EOF\n"):
        raise _CheckFailure("openmetrics_eof_missing")
    if validator is None:
        raise _CheckFailure("openmetrics_parser_unavailable")
    expected = _expected_samples(envelope)
    actual = validator.samples(reply.body)
    normalized = {key: Decimal(str(value)) for key, value in expected.items()}
    if actual != normalized:
        raise _CheckFailure("openmetrics_values_differ_from_snapshot")
    return actual


def _sample_evidence(
    samples: dict[tuple[str, tuple[tuple[str, str], ...]], Decimal],
) -> list[dict[str, Any]]:
    return [
        {"name": name, "labels": dict(labels), "value": str(value)}
        for (name, labels), value in sorted(samples.items())
    ]


def _accepted_metrics(bodies: list[bytes], envelope: SnapshotEnvelope) -> int:
    expected = export.project_metrics(envelope).points
    expected_by_key = {(point.name, tuple(sorted(point.attributes))): point for point in expected}
    rows = [row for body in bodies for row in metric_points(json.loads(body))]
    actual_by_key = {(row["name"], tuple(sorted(row["attributes"].items()))): row for row in rows}
    if len(rows) != len(expected) or set(actual_by_key) != set(expected_by_key):
        raise _CheckFailure("accepted_metric_set_differs_from_snapshot")
    for key, point in expected_by_key.items():
        row = actual_by_key[key]
        if row["kind"] != point.kind or row["unit"] != point.unit:
            raise _CheckFailure("accepted_metric_type_differs_from_snapshot")
        wire = row["point"]
        if point.kind in ("sum", "gauge"):
            if wire.get("asInt") != str(point.value):
                raise _CheckFailure("accepted_metric_value_differs_from_snapshot")
        else:
            histogram = point.hist
            if histogram is None or wire.get("count") != str(histogram.count):
                raise _CheckFailure("accepted_histogram_count_differs_from_snapshot")
            if wire.get("bucketCounts") != [str(value) for value in histogram.counts]:
                raise _CheckFailure("accepted_histogram_buckets_differ_from_snapshot")
            if Decimal(str(wire.get("sum"))) != Decimal(histogram.sum):
                raise _CheckFailure("accepted_histogram_sum_differs_from_snapshot")
    return len(rows)


def _acknowledged(
    port: int, token: bytes, destination: str, signal: str
) -> tuple[_Reply, dict[str, Any]]:
    path = "/v1/snapshot?" + urlencode(
        {"view": "acknowledged", "destination": destination, "signal": signal}
    )
    reply = _get(port, path, token)
    return reply, _document(reply) if reply.status == 200 else {}


def _export_status(port: int, token: bytes) -> dict[tuple[str, str], dict[str, Any]]:
    reply = _get(port, "/v1/export-status", token)
    if reply.status != 200:
        raise _CheckFailure("export_status_unavailable")
    document = _document(reply)
    return {(row["destination_id"], row["signal"]): row for row in document.get("destinations", [])}


def _payloads_match(preview: dict[str, Any], requests: list[Any]) -> bool:
    batches = preview.get("batches")
    if not isinstance(batches, list) or not batches:
        return not requests
    expected = sorted(str(batch.get("sha256")) for batch in batches)
    actual = sorted(hashlib.sha256(request.body).hexdigest() for request in requests)
    return expected == actual and all(
        json.loads(request.body) in [batch.get("payload") for batch in batches]
        for request in requests
    )


def _configure(profile: str, healthy_url: str, partial_url: str) -> None:
    current = policy.load_profile_policy(profile)
    reserved = {DESTINATION, PARTIAL_DESTINATION}
    destinations = tuple(d for d in current.destinations if d.destination_id not in reserved) + (
        DestinationPolicy(
            DESTINATION,
            healthy_url,
            (Signal.METRICS, Signal.LOGS),
            True,
            f"env:{CREDENTIAL_ENV}",
        ),
        DestinationPolicy(
            PARTIAL_DESTINATION,
            partial_url,
            (Signal.METRICS,),
            False,
            f"env:{CREDENTIAL_ENV}",
        ),
    )
    enabled = tuple((d, s) for d, s in current.enabled_exports if d not in reserved)
    policy.save_profile(replace(current, destinations=destinations, enabled_exports=enabled))


def _exercise(
    product: Product,
    previous: Collected,
    collect_latest: Callable[[], Collected],
    secrets: tuple[str, ...],
) -> C9Outcome:
    output: list[str] = []
    payload_bodies: list[bytes] = []
    evidence: dict[str, Any] = {
        "sample_count": 0,
        "scrape_samples": [],
        "metrics_ack_generation": None,
        "metrics_ack_snapshot_id": None,
        "logs_consent_was_separate": False,
        "partial_rejected_points": 0,
        "partial_snapshot_was_not_acknowledged": False,
        "privacy_canary_absent": False,
        "accepted_metric_points": 0,
        "collected_ahead_of_acknowledged": False,
        "logs_acknowledged": False,
    }
    before: SnapshotEnvelope | None = None
    latest: SnapshotEnvelope | None = None
    collected = previous
    scrape = Gate("scrape", NOT_RUN, "scrape was not attempted")
    export_gate = Gate("export_privacy", NOT_RUN, "export was not attempted")
    try:
        before = _current(PROFILE)
        if before.generation < 1:
            raise _CheckFailure("previous_snapshot_generation_invalid")
        evidence["snapshot_before_generation"] = before.generation
        evidence["snapshot_before_id"] = before.snapshot_id
        if previous.collect.get("snapshot_id") not in (None, before.snapshot_id):
            raise _CheckFailure("previous_collection_differs_from_published_snapshot")

        secret = "c9-" + os.urandom(24).hex()
        product = product.with_env(**{CREDENTIAL_ENV: secret})
        with LabReceiver.start() as healthy, LabReceiver.start() as partial:
            _configure(PROFILE, healthy.url, partial.url)
            export_ids.load_key(PROFILE, create=True)
            worker = _BoundedWorker()
            handle = http.serve(PROFILE, port=0, worker=worker, start=False)
            worker_thread = threading.Thread(target=handle.run_worker, daemon=True)
            worker_thread.start()
            if not worker.started.wait(2):
                handle.stop()
                raise _CheckFailure("bounded_worker_did_not_start")
            try:
                token = http_auth.load_token(PROFILE)
                if token is None or token == secret.encode("ascii"):
                    raise _CheckFailure("endpoint_and_ingest_credentials_not_separate")
                scrape_problems: list[str] = []
                validator: _OpenMetricsParser | _PrometheusQuery | None = None
                blocked_scrape: str | None = None
                with ExitStack() as stack:
                    try:
                        validator, _ = _scrape_validator(handle, token, stack, evidence)
                    except _CheckFailure as error:
                        if str(error).startswith("prometheus_prerequisite:"):
                            blocked_scrape = str(error).partition(":")[2]
                        else:
                            scrape_problems.append(str(error))
                    unauthorized = _get(handle.port, "/v1/snapshot", None)
                    unauthorized_metrics = _get(handle.port, "/metrics", None)
                    current = _get(handle.port, "/v1/snapshot", token)
                    if (
                        unauthorized.status != 401
                        or unauthorized_metrics.status != 401
                        or current.status != 200
                    ):
                        scrape_problems.append("authenticated_snapshot_or_metrics_read_failed")
                    if current.status == 200:
                        current_doc = _document(current)
                        if current.headers.get("etag") != f'"{before.snapshot_id}"':
                            scrape_problems.append("snapshot_etag_mismatch")
                        if current_doc.get("snapshot_id") != before.snapshot_id:
                            scrape_problems.append("snapshot_body_mismatch")
                        conditional = _get(
                            handle.port,
                            "/v1/snapshot",
                            token,
                            {"If-None-Match": f'"{before.snapshot_id}"'},
                        )
                        if conditional.status != 304 or conditional.body:
                            scrape_problems.append("conditional_snapshot_read_failed")
                    initial_ack, _ = _acknowledged(handle.port, token, DESTINATION, "metrics")
                    if initial_ack.status != 404:
                        scrape_problems.append("unexpected_prior_acknowledgment")
                    old_metrics = _get(handle.port, "/metrics", token)
                    if blocked_scrape is None:
                        try:
                            old_samples = _assert_samples(old_metrics, before, validator)
                            evidence["old_sample_count"] = len(old_samples)
                        except (ImportError, ValueError, _CheckFailure) as error:
                            scrape_problems.append(
                                f"previous_scrape_invalid:{type(error).__name__}"
                            )

                    old_generation = before.generation
                    try:
                        old_preview, rendered = _preview(product, DESTINATION, "metrics")
                        output.append(rendered)
                        if (
                            old_preview.get("snapshot_id") != before.snapshot_id
                            or healthy.requests
                            or partial.requests
                        ):
                            raise _CheckFailure("initial_preview_or_export_state_invalid")
                        before_enable_state = _tree(product.data_dir / "accounting")
                        output.append(_enable(product, secret, DESTINATION, "metrics"))
                        if (
                            _tree(product.data_dir / "accounting") == before_enable_state
                            or healthy.requests
                            or partial.requests
                        ):
                            raise _CheckFailure("metrics_consent_enable_side_effects_invalid")
                        enabled = policy.load_profile_policy(PROFILE).enabled_exports
                        if (DESTINATION, "metrics") not in enabled:
                            raise _CheckFailure("metrics_signal_consent_missing")
                        if (DESTINATION, "logs") in enabled:
                            raise _CheckFailure("logs_were_enabled_by_metrics_consent")
                        partial_preview, rendered = _preview(
                            product, PARTIAL_DESTINATION, "metrics"
                        )
                        output.append(rendered)
                        if (
                            partial_preview.get("snapshot_id") != before.snapshot_id
                            or healthy.requests
                            or partial.requests
                        ):
                            raise _CheckFailure("partial_preview_not_exact")
                        output.append(_enable(product, secret, PARTIAL_DESTINATION, "metrics"))
                        logs_preview, rendered = _preview(product, DESTINATION, "logs")
                        output.append(rendered)
                        if healthy.requests or partial.requests:
                            raise _CheckFailure("logs_preview_sent_data")
                        output.append(_enable(product, secret, DESTINATION, "logs"))
                        logs_separate = (DESTINATION, "logs") in policy.load_profile_policy(
                            PROFILE
                        ).enabled_exports
                        evidence["logs_consent_was_separate"] = logs_separate
                        if not logs_separate or not old_preview.get("batch_count", 0):
                            raise _CheckFailure("separate_logs_consent_or_metrics_preview_missing")

                        old_metric_start = len(healthy.requests)
                        old_send, rendered = _send(product, secret, DESTINATION, "metrics")
                        output.append(rendered)
                        old_metric_requests = [
                            request
                            for request in healthy.requests[old_metric_start:]
                            if request.path == "/v1/metrics"
                        ]
                        if (
                            old_send.get("sent", 0) <= 0
                            or old_send.get("acked") != old_send.get("sent")
                            or old_send.get("partial", 0)
                            or not _payloads_match(old_preview, old_metric_requests)
                        ):
                            raise _CheckFailure("previous_snapshot_not_fully_acknowledged")
                        old_ack, old_ack_doc = _acknowledged(
                            handle.port, token, DESTINATION, "metrics"
                        )
                        if (
                            old_ack.status != 200
                            or old_ack_doc.get("generation") != before.generation
                            or old_ack_doc.get("snapshot_id") != before.snapshot_id
                        ):
                            raise _CheckFailure("previous_acknowledged_view_not_exact")
                        old_logs_ack_generation = None
                        old_logs_ack_snapshot_id = None
                        if logs_preview.get("batch_count", 0):
                            old_log_start = len(healthy.requests)
                            old_logs, rendered = _send(product, secret, DESTINATION, "logs")
                            output.append(rendered)
                            old_log_requests = [
                                request
                                for request in healthy.requests[old_log_start:]
                                if request.path == "/v1/logs"
                            ]
                            if (
                                old_logs.get("partial", 0)
                                or old_logs.get("acked") != old_logs.get("sent")
                                or not _payloads_match(logs_preview, old_log_requests)
                            ):
                                raise _CheckFailure("previous_logs_snapshot_not_fully_acknowledged")
                            old_logs_ack, old_logs_doc = _acknowledged(
                                handle.port, token, DESTINATION, "logs"
                            )
                            if (
                                old_logs_ack.status != 200
                                or old_logs_doc.get("generation") != before.generation
                                or old_logs_doc.get("snapshot_id") != before.snapshot_id
                            ):
                                raise _CheckFailure("previous_logs_acknowledged_view_not_exact")
                            old_logs_ack_generation = old_logs_doc["generation"]
                            old_logs_ack_snapshot_id = old_logs_doc["snapshot_id"]
                        else:
                            old_logs_ack, _ = _acknowledged(handle.port, token, DESTINATION, "logs")
                            if (
                                old_logs_ack.status != 404
                                or _document(old_logs_ack).get("error") != "not_acknowledged"
                            ):
                                raise _CheckFailure(
                                    "empty_previous_logs_signal_claimed_acknowledgment"
                                )
                        collected = collect_latest()
                        latest = _current(PROFILE)
                        if (
                            latest.generation <= old_generation
                            or latest.snapshot_id == before.snapshot_id
                        ):
                            raise _CheckFailure("new_collection_did_not_advance_snapshot_content")
                        evidence["collected_ahead_of_acknowledged"] = True
                        current_reply = _get(handle.port, "/v1/snapshot", token)
                        latest_doc = _document(current_reply)
                        current_ack, current_ack_doc = _acknowledged(
                            handle.port, token, DESTINATION, "metrics"
                        )
                        if (
                            current_reply.status != 200
                            or current_reply.headers.get("etag") != f'"{latest.snapshot_id}"'
                            or latest_doc.get("generation") != latest.generation
                            or latest_doc.get("snapshot_id") != latest.snapshot_id
                        ):
                            raise _CheckFailure("latest_snapshot_endpoint_or_etag_mismatch")
                        if (
                            current_ack.status != 200
                            or current_ack_doc.get("generation") != before.generation
                            or current_ack_doc.get("snapshot_id") != before.snapshot_id
                        ):
                            raise _CheckFailure("older_acknowledged_snapshot_was_substituted")
                        latest_metrics = _get(handle.port, "/metrics", token)
                        if blocked_scrape is None:
                            try:
                                latest_samples = _assert_samples(latest_metrics, latest, validator)
                                evidence["sample_count"] = len(latest_samples)
                            except (ImportError, ValueError, _CheckFailure) as error:
                                scrape_problems.append(
                                    f"latest_scrape_invalid:{type(error).__name__}"
                                )

                        preview_latest, rendered = _preview(product, DESTINATION, "metrics")
                        output.append(rendered)
                        latest_send, rendered = _send(product, secret, DESTINATION, "metrics")
                        output.append(rendered)
                        if (
                            latest_send.get("sent", 0) <= 0
                            or latest_send.get("acked") != latest_send.get("sent")
                            or latest_send.get("partial", 0)
                        ):
                            raise _CheckFailure("latest_metrics_snapshot_not_fully_acknowledged")
                        latest_ack, latest_ack_doc = _acknowledged(
                            handle.port, token, DESTINATION, "metrics"
                        )
                        if (
                            latest_ack.status != 200
                            or latest_ack_doc.get("generation") != latest.generation
                            or latest_ack_doc.get("snapshot_id") != latest.snapshot_id
                        ):
                            raise _CheckFailure("latest_metrics_acknowledgment_not_exact")
                        latest_logs_preview, rendered = _preview(product, DESTINATION, "logs")
                        output.append(rendered)
                        logs_acked = False
                        if latest_logs_preview.get("batch_count", 0):
                            log_start = len(healthy.requests)
                            latest_logs, rendered = _send(product, secret, DESTINATION, "logs")
                            output.append(rendered)
                            log_requests = [
                                request
                                for request in healthy.requests[log_start:]
                                if request.path == "/v1/logs"
                            ]
                            if (
                                latest_logs.get("partial", 0)
                                or latest_logs.get("acked") != latest_logs.get("sent")
                                or not _payloads_match(latest_logs_preview, log_requests)
                            ):
                                raise _CheckFailure("latest_logs_snapshot_not_fully_acknowledged")
                            logs_ack, logs_doc = _acknowledged(
                                handle.port, token, DESTINATION, "logs"
                            )
                            logs_acked = (
                                logs_ack.status == 200
                                and logs_doc.get("generation") == latest.generation
                                and logs_doc.get("snapshot_id") == latest.snapshot_id
                            )
                            if not logs_acked:
                                raise _CheckFailure("latest_logs_acknowledgment_not_exact")
                        else:
                            logs_ack, logs_doc = _acknowledged(
                                handle.port, token, DESTINATION, "logs"
                            )
                            if old_logs_ack_snapshot_id is None:
                                if (
                                    logs_ack.status != 404
                                    or _document(logs_ack).get("error") != "not_acknowledged"
                                ):
                                    raise _CheckFailure("empty_logs_signal_claimed_acknowledgment")
                            elif (
                                logs_ack.status != 200
                                or logs_doc.get("generation") != old_logs_ack_generation
                                or logs_doc.get("snapshot_id") != old_logs_ack_snapshot_id
                            ):
                                raise _CheckFailure("older_logs_acknowledgment_was_substituted")

                        partial_latest_preview, rendered = _preview(
                            product, PARTIAL_DESTINATION, "metrics"
                        )
                        output.append(rendered)
                        if partial_latest_preview.get("snapshot_id") != latest.snapshot_id:
                            raise _CheckFailure("partial_latest_preview_not_exact")
                        partial.queue(Partial(1, kind="metrics"), path="/v1/metrics")
                        partial_send, rendered = _send(
                            product, secret, PARTIAL_DESTINATION, "metrics"
                        )
                        output.append(rendered)
                        if partial_send.get("partial", 0) <= 0:
                            raise _CheckFailure("partial_receiver_ack_not_observed")
                        partial_requests = [
                            request for request in partial.requests if request.path == "/v1/metrics"
                        ]
                        if (
                            not partial_requests
                            or partial_requests[-1].status != 200
                            or not _payloads_match(partial_latest_preview, partial_requests)
                        ):
                            raise _CheckFailure("partial_payload_or_response_not_observed")
                        partial_ack, _ = _acknowledged(
                            handle.port, token, PARTIAL_DESTINATION, "metrics"
                        )
                        if (
                            partial_ack.status != 404
                            or _document(partial_ack).get("error") != "not_acknowledged"
                        ):
                            raise _CheckFailure("partial_snapshot_claimed_fully_acknowledged")
                        rows = _export_status(handle.port, token)
                        main_status = rows.get((DESTINATION, "metrics"), {})
                        partial_status = rows.get((PARTIAL_DESTINATION, "metrics"), {})
                        if (
                            main_status.get("last_full_ack", {}).get("snapshot_id")
                            != latest.snapshot_id
                        ):
                            raise _CheckFailure("export_status_latest_metrics_ack_mismatch")
                        main_logs_status = rows.get((DESTINATION, "logs"), {})
                        expected_logs_id = (
                            latest.snapshot_id if logs_acked else old_logs_ack_snapshot_id
                        )
                        expected_logs_generation = (
                            latest.generation if logs_acked else old_logs_ack_generation
                        )
                        if (
                            main_logs_status.get("last_full_ack", {}).get("snapshot_id")
                            != expected_logs_id
                            or main_logs_status.get("last_full_ack", {}).get("generation")
                            != expected_logs_generation
                        ):
                            raise _CheckFailure("export_status_latest_logs_ack_mismatch")
                        if (
                            partial_status.get("last_full_ack", {}).get("snapshot_id") is not None
                            or partial_status.get("rejected_points", 0) <= 0
                        ):
                            raise _CheckFailure(
                                "partial_status_does_not_preserve_unacknowledged_state"
                            )
                        expected_hashes = {
                            str(item.get("sha256")) for item in preview_latest.get("batches", [])
                        }
                        current_metrics_requests = [
                            request
                            for request in healthy.requests
                            if request.path == "/v1/metrics"
                            and hashlib.sha256(request.body).hexdigest() in expected_hashes
                        ]
                        if not _payloads_match(preview_latest, current_metrics_requests):
                            raise _CheckFailure("sent_metrics_differ_from_exact_preview")
                        evidence["accepted_metric_points"] = _accepted_metrics(
                            [request.body for request in current_metrics_requests], latest
                        )
                        evidence["partial_rejected_points"] = int(
                            partial_status.get("rejected_points", 0)
                        )
                        evidence["partial_snapshot_was_not_acknowledged"] = True
                        evidence["metrics_ack_generation"] = latest.generation
                        evidence["metrics_ack_snapshot_id"] = latest.snapshot_id
                        evidence["logs_acknowledged"] = logs_acked
                        evidence["partial_preview_matched"] = _payloads_match(
                            partial_latest_preview, partial_requests
                        )
                        evidence["main_preview_matched"] = _payloads_match(
                            preview_latest, current_metrics_requests
                        )
                        remote_requests = [*healthy.requests, *partial.requests]
                        if not remote_requests or any(
                            request.headers.get("authorization") != f"Bearer {secret}"
                            for request in remote_requests
                        ):
                            raise _CheckFailure("synthetic_ingest_credential_not_used")
                        final_metrics = _get(handle.port, "/metrics", token)
                        if blocked_scrape is None:
                            try:
                                scraped_samples = _assert_samples(final_metrics, latest, validator)
                                evidence["sample_count"] = len(scraped_samples)
                                evidence["scrape_samples"] = _sample_evidence(scraped_samples)
                            except (ImportError, ValueError, _CheckFailure) as error:
                                scrape_problems.append(
                                    f"final_scrape_invalid:{type(error).__name__}"
                                )
                        payload_bodies.extend(request.body for request in remote_requests)
                        status_reply = _get(handle.port, "/v1/export-status", token)
                        public_text = (
                            current.body
                            + old_metrics.body
                            + latest_metrics.body
                            + final_metrics.body
                            + status_reply.body
                        )
                        private_values = (*secrets, secret)
                        evidence["privacy_canary_absent"] = all(
                            value.encode("utf-8") not in public_text
                            and all(value.encode("utf-8") not in body for body in payload_bodies)
                            and all(value not in text for text in output)
                            for value in private_values
                        )
                        if not evidence["privacy_canary_absent"]:
                            raise _CheckFailure("privacy_canary_leaked")
                        if (
                            not evidence["partial_preview_matched"]
                            or not evidence["main_preview_matched"]
                        ):
                            raise _CheckFailure("sent_payload_differs_from_preview")
                        if evidence["accepted_metric_points"] <= 0:
                            raise _CheckFailure("no_accepted_metric_points")
                        scrape = (
                            Gate("scrape", FAIL, "; ".join(scrape_problems))
                            if scrape_problems
                            else Gate("scrape", BLOCKED, blocked_scrape)
                            if blocked_scrape
                            else Gate(
                                "scrape",
                                PASS,
                                "authenticated HTTP scrape parsed; snapshot values exact",
                            )
                        )
                        export_gate = Gate(
                            "export_privacy",
                            PASS,
                            "sanitized preview matched accepted OTLP; full and partial "
                            "acknowledgments verified",
                        )
                    except Exception as error:
                        detail = _failure_detail(error)
                        export_gate = Gate("export_privacy", FAIL, detail)
                        if scrape.status not in (BLOCKED, FAIL):
                            scrape = Gate("scrape", FAIL, detail)
                final_rows = _export_status(handle.port, token)
                final_main = final_rows.get((DESTINATION, "metrics"), {})
                if latest is not None:
                    evidence["metrics_ack_generation"] = final_main.get("last_full_ack", {}).get(
                        "generation"
                    )
                    evidence["metrics_ack_snapshot_id"] = final_main.get("last_full_ack", {}).get(
                        "snapshot_id"
                    )
            finally:
                handle.stop()
                worker_thread.join(timeout=3)
                if worker_thread.is_alive():
                    raise _CheckFailure("bounded_worker_did_not_stop")
        evidence["sample_count"] = int(evidence.get("sample_count", 0))
        evidence["snapshot_before_generation"] = before.generation
        evidence["snapshot_before_id"] = before.snapshot_id
        evidence["snapshot_generation"] = latest.generation if latest else None
        evidence["snapshot_id"] = latest.snapshot_id if latest else None
        return C9Outcome(
            collected=collected,
            snapshot_id=latest.snapshot_id if latest else None,
            snapshot_generation=latest.generation if latest else None,
            snapshot_before_id=before.snapshot_id,
            snapshot_before_generation=before.generation,
            scrape=scrape,
            export=export_gate,
            evidence=evidence,
        )
    except Exception as error:
        detail = _failure_detail(error)
        if scrape.status in (NOT_RUN, PASS):
            scrape = Gate("scrape", FAIL, detail)
        if export_gate.status in (NOT_RUN, PASS):
            export_gate = Gate("export_privacy", FAIL, detail)
        return C9Outcome(
            collected=collected,
            snapshot_id=latest.snapshot_id if latest else None,
            snapshot_generation=latest.generation if latest else None,
            snapshot_before_id=before.snapshot_id if before else None,
            snapshot_before_generation=before.generation if before else None,
            scrape=scrape,
            export=export_gate,
            evidence=evidence,
        )


def run_c9(
    product: Product,
    previous: Collected,
    collect_latest: Callable[[], Collected],
    *,
    secrets: tuple[str, ...] = (),
) -> C9Outcome:
    """Verify the previous snapshot, publish the next one, and exercise real scrape/export paths."""
    with _data_root(product.data_dir):
        return _exercise(product, previous, collect_latest, secrets)
