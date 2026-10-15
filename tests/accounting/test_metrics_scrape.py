"""11.6 acceptance: a REAL Prometheus scrapes the live loopback server with its bearer token.

Needs docker and ``prom/prometheus``; without them the whole module reports a NAMED skip
reason (never a silent pass). Values asserted through Prometheus's own query API must equal
the snapshot's content, and a bad exposition (missing ``# EOF``, data after ``# EOF``) must
make the real scraper mark the target down.
"""

from __future__ import annotations

import json
import re
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ssgrep.accounting import http_views, metrics_render
from tests.accounting import prom_lab
from tests.accounting.http_builders import (
    PROFILE,
    make_snapshot,
    publish,
    start,
    token_for,
    window_with_unknown_optimistic,
    write_delivery,
)

_REASON = prom_lab.docker_available()
pytestmark = [pytest.mark.skipif(_REASON is not None, reason=str(_REASON))]


@pytest.fixture
def run_id():
    run = prom_lab.new_run_id()
    yield run
    assert prom_lab.leftovers(run) == []  # nothing of ours survives the test


def stamp(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def test_real_prometheus_scrapes_fresh_stale_and_unavailable_servers(run_id: str) -> None:
    now = datetime.now(UTC)
    fresh_at, stale_at = stamp(now), stamp(now - timedelta(minutes=30))

    window = window_with_unknown_optimistic()
    optimistic = window["estimates"]["context_tokens"]["optimistic"]
    assert optimistic["credit"]["value"] == 20000  # known credit subtotal, not complete credit
    assert optimistic["credit_missing"] == [["synthetic_source", "source_costs_missing"]]
    assert optimistic["net"]["value"] is None and optimistic["complete"] is False
    assert optimistic["known_net"] == 4000
    publish(PROFILE, make_snapshot(7, window=window, collected_at=fresh_at))
    write_delivery(
        PROFILE,
        "acme",
        "metrics",
        pending_batches=2,
        rejected_points=5,
        last_acked_at=fresh_at,
        loss_summary=[{"reason": "spool_expired", "batches": 1, "records": 7, "at": fresh_at}],
    )
    write_delivery(PROFILE, "acme", "logs", pending_batches=0)
    stale_profile, empty_profile = "stale-org", "empty-org"
    publish(stale_profile, make_snapshot(3, profile=stale_profile, collected_at=stale_at))

    with start(PROFILE) as fresh, start(stale_profile) as stale, start(empty_profile) as empty:
        jobs = [
            ("fresh", fresh.port, "fresh.token"),
            ("stale", stale.port, "stale.token"),
            ("empty", empty.port, "empty.token"),
            ("wrongtoken", fresh.port, "stale.token"),  # another profile's token must fail
            ("notoken", fresh.port, None),
        ]
        tokens = {
            "fresh.token": token_for(PROFILE),
            "stale.token": token_for(stale_profile),
            "empty.token": token_for(empty_profile),
        }
        with prom_lab.prometheus(run_id, jobs, tokens) as prom:
            prom.wait_scraped([job for job, _, _ in jobs])
            up = {r["metric"]["job"]: r["value"][1] for r in prom.query("up")}
            assert {j: up[j] for j in ("fresh", "stale", "empty")} == {
                "fresh": "1",
                "stale": "1",
                "empty": "1",
            }
            assert up["wrongtoken"] == "0" and up["notoken"] == "0"
            targets = prom.targets()
            assert "401" in targets["wrongtoken"]["lastError"]
            assert "401" in targets["notoken"]["lastError"]
            assert targets["fresh"]["lastError"] == ""

            # counters (exact snapshot values)
            ok = 'ssgrep_executions_total{job="fresh",operation="search",outcome="ok"}'
            assert prom.scalar(ok) == 3
            err = 'ssgrep_executions_total{job="fresh",operation="show",outcome="error"}'
            assert prom.scalar(err) == 1
            rendered = 'ssgrep_rendered_bytes_total{job="fresh",operation="search"}'
            assert prom.scalar(rendered) == 3000
            rejected = 'ssgrep_export_rejected_points_total{job="fresh",destination="acme"}'
            assert prom.scalar(rejected) == 5
            lost = 'ssgrep_export_lost_records_total{job="fresh",reason="spool_expired"}'
            assert prom.scalar(lost) == 7

            # histogram: cumulative buckets, count, sum in seconds
            bucket = (
                'ssgrep_operation_duration_seconds_bucket{job="fresh",operation="search",'
                'transport="cli",le="%s"}'
            )
            assert prom.scalar(bucket % "0.001") == 0
            assert prom.scalar(bucket % "0.005") == 3
            assert prom.scalar(bucket % "+Inf") == 3
            count = 'ssgrep_operation_duration_seconds_count{job="fresh",operation="search"}'
            assert prom.scalar(count) == 3
            total = 'ssgrep_operation_duration_seconds_sum{job="fresh",operation="search"}'
            assert prom.scalar(total) == pytest.approx(0.006)
            quantile = (
                "histogram_quantile(0.5, sum by (le) "
                '(ssgrep_operation_duration_seconds_bucket{job="fresh",operation="search"}))'
            )
            assert 0.001 < prom.scalar(quantile) <= 0.005

            # gauges: negative, unknown omitted, coverage, freshness, info
            net = 'ssgrep_estimated_net_tokens{job="fresh",window="7d",scenario="%s"}'
            assert prom.scalar(net % "central") == -7000
            assert prom.scalar(net % "conservative") == -12000
            assert prom.query(net % "optimistic") == []  # unknown: no sample, not 0
            credit = 'ssgrep_estimated_credit_tokens{job="fresh",window="7d",scenario="optimistic"}'
            assert prom.query(credit) == []  # incomplete credit is a second distinct omission
            known_credit = (
                'ssgrep_known_credit_tokens{job="fresh",window="7d",scenario="optimistic"}'
            )
            known_net = 'ssgrep_known_net_tokens{job="fresh",window="7d",scenario="optimistic"}'
            assert prom.scalar(known_credit) == 20000 and prom.scalar(known_net) == 4000
            # Two unit groups each expose credit/net per scenario plus one shared debit:
            # fourteen complete-estimate slots, with optimistic token credit AND net absent.
            missing = 'ssgrep_coverage_records{job="fresh",kind="estimates",status="missing"}'
            available = 'ssgrep_coverage_records{job="fresh",kind="estimates",status="available"}'
            assert prom.scalar(missing) == 2 and prom.scalar(available) == 12
            recorded = 'ssgrep_coverage_records{job="fresh",kind="lookups",status="recorded"}'
            assert prom.scalar(recorded) == 112
            explicit = 'ssgrep_reconciliation_records{job="fresh",attribution="explicit"}'
            assert prom.scalar(explicit) == 70
            validated = 'ssgrep_reuse_evidence_records{job="fresh",evidence="validated"}'
            assert prom.scalar(validated) == 3
            assert prom.scalar('ssgrep_export_pending_batches{job="fresh",signal="metrics"}') == 2
            assert prom.scalar('ssgrep_snapshot_available{job="fresh"}') == 1
            collected = prom.scalar('ssgrep_snapshot_collected_timestamp_seconds{job="fresh"}')
            expected = datetime.strptime(fresh_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
            assert collected == int(expected.timestamp())
            info = 'ssgrep_accounting_info{job="fresh",methodology_id="m-test-v1"}'
            assert prom.scalar(info) == 1

            # stale: still served; freshness is the gauge (30 min older), not a sample timestamp
            stale_gauge = prom.scalar('ssgrep_snapshot_collected_timestamp_seconds{job="stale"}')
            assert collected - stale_gauge == 1800
            age = 'time() - ssgrep_snapshot_collected_timestamp_seconds{job="stale"}'
            assert prom.scalar(age) > 1700
            stale_ok = 'ssgrep_executions_total{job="stale",operation="search",outcome="ok"}'
            assert prom.scalar(stale_ok) == 3

            # unavailable: up, available=0, and NO fabricated usage/savings
            assert prom.scalar('ssgrep_snapshot_available{job="empty"}') == 0
            usage = (
                '{job="empty",__name__=~"ssgrep_(executions|rendered|operation|estimated|debit|'
                'coverage|reconciliation|reuse|accounting).*"}'
            )
            assert prom.query(usage) == []


@pytest.mark.parametrize("mutation", ["drop_eof", "data_after_eof"])
def test_real_scraper_marks_a_malformed_exposition_down(
    run_id: str, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    real = metrics_render.render_openmetrics

    def mutated(samples):
        body = real(samples).decode()
        if mutation == "drop_eof":
            body = body.replace("# EOF\n", "")
        else:
            body += "ssgrep_extra 1\n"
        return body.encode()

    publish(PROFILE, make_snapshot(7))
    monkeypatch.setattr(http_views, "render_openmetrics", mutated)
    with start(PROFILE) as handle:
        tokens = {"t.token": token_for(PROFILE)}
        with prom_lab.prometheus(run_id, [("broken", handle.port, "t.token")], tokens) as prom:
            target = None
            for _ in range(40):
                target = prom.targets().get("broken")
                if target and target["lastError"]:
                    break
                time.sleep(0.5)
            assert target and target["health"] == "down" and target["lastError"]
            assert prom.scalar('up{job="broken"}') == 0
            assert prom.query('ssgrep_snapshot_available{job="broken"}') == []


_GUIDE = Path(__file__).resolve().parents[2] / "docs" / "organization-metrics.md"
_WINDOW = "7d"
_TOKEN_METHOD = "chars_div4_v1"
_WORK_METHOD = "capped_gap_v1"
_METHODOLOGY = "synthetic-current"
_OTHER_METHODOLOGY = "synthetic-other"


def _guide_query(name: str, method: str, methodology: str) -> str:
    """Treat the named guide block as the program, not an expected source string."""
    blocks = dict(re.findall(r"```promql ([a-z_]+)\n(.*?)\n```", _GUIDE.read_text(), re.DOTALL))
    return (
        blocks[name]
        .replace("$window", _WINDOW)
        .replace("$methodology", methodology)
        .replace("$method", method)
    )


def _organization_series() -> list[dict[str, str]]:
    # Backend-authenticated tenant/device; only installation came from the client.
    # Every row shares a job and has NO instance, as an OTLP backend may store it.
    identities = {
        "first": ("tenant-a", "device-1", "install-a"),
        "second": ("tenant-a", "device-2", "install-b"),
        "additional": ("tenant-a", "device-1", "install-c"),
        "old": ("tenant-a", "device-1", "install-old"),
        "other_tenant": ("tenant-b", "device-1", "install-a"),
    }
    series: list[dict[str, str]] = []

    def add(family: str, identity: str, value: int, **dimensions: str) -> None:
        tenant, device, installation = identities[identity]
        labels = {
            "tenant": tenant,
            "device": device,
            "ssgrep_installation_id": installation,
            "job": "otlp",
            **dimensions,
        }
        selector = ",".join(f"{key}={json.dumps(value)}" for key, value in labels.items())
        series.append({"series": f"{family}{{{selector}}}", "values": f"{value} {value}"})

    for identity in identities:
        methodology = _OTHER_METHODOLOGY if identity in {"old", "other_tenant"} else _METHODOLOGY
        add("ssgrep_accounting_info", identity, 1, methodology_id=methodology, replica="one")
    # A second current info row must be a membership match, not a multiplier
    # (nor a many-to-many arithmetic-join error).
    add("ssgrep_accounting_info", "first", 1, methodology_id=_METHODOLOGY, replica="two")

    for identity, value in (
        ("first", 11),
        ("second", 13),
        ("additional", 17),
        ("old", 100),
        ("other_tenant", 200),
    ):
        add("ssgrep_debit_tokens", identity, value, window=_WINDOW, method=_TOKEN_METHOD)
    add("ssgrep_debit_tokens", "first", 10000, window="24h", method=_TOKEN_METHOD)
    add("ssgrep_debit_tokens", "first", 20000, window=_WINDOW, method="provider_usage_v1")

    for family, method, values, conservative in (
        ("ssgrep_estimated_net_tokens", _TOKEN_METHOD, (-7, 20, 3, 1000, 2000), -12),
        ("ssgrep_estimated_net_work_seconds", _WORK_METHOD, (-2, 5, 1, 100, 200), -9),
        ("ssgrep_estimated_credit_tokens", _TOKEN_METHOD, (30, 40, 10, 1000, 2000), 15),
    ):
        for identity, value in zip(identities, values, strict=True):
            add(family, identity, value, window=_WINDOW, method=method, scenario="central")
        add(
            family,
            "first",
            conservative,
            window=_WINDOW,
            method=method,
            scenario="conservative",
        )
        add(family, "first", 10000, window="24h", method=method, scenario="central")
        add(
            family,
            "first",
            20000,
            window=_WINDOW,
            method="provider_usage_v1",
            scenario="central",
        )
    # Optimistic complete values are unknown and absent, despite a known subtotal.
    add(
        "ssgrep_known_net_tokens",
        "first",
        9000,
        window=_WINDOW,
        method=_TOKEN_METHOD,
        scenario="optimistic",
    )
    return series


@pytest.mark.parametrize(
    ("name", "method", "current", "other"),
    [
        ("debit_tokens", _TOKEN_METHOD, [(None, 41)], [(None, 300)]),
        (
            "net_tokens_by_scenario",
            _TOKEN_METHOD,
            [("central", 16), ("conservative", -12)],
            [("central", 3000)],
        ),
        (
            "net_work_seconds_by_scenario",
            _WORK_METHOD,
            [("central", 4), ("conservative", -9)],
            [("central", 300)],
        ),
        (
            "credit_tokens_by_scenario",
            _TOKEN_METHOD,
            [("central", 80), ("conservative", 15)],
            [("central", 3000)],
        ),
        (
            "scenario_reporting_devices",
            _TOKEN_METHOD,
            [("central", 2), ("conservative", 1)],
            [("central", 2)],
        ),
    ],
    ids=[
        "debit_tokens",
        "net_tokens_by_scenario",
        "net_work_seconds_by_scenario",
        "credit_tokens_by_scenario",
        "scenario_reporting_devices",
    ],
)
def test_real_promtool_executes_organization_guide_queries(
    run_id: str,
    tmp_path: Path,
    name: str,
    method: str,
    current: list[tuple[str | None, int]],
    other: list[tuple[str | None, int]],
) -> None:
    """Execute both methodologies, including the otherwise-excluded positive controls."""
    cases = []
    for methodology, expected in ((_METHODOLOGY, current), (_OTHER_METHODOLOGY, other)):
        cases.append(
            {
                "expr": _guide_query(name, method, methodology),
                "eval_time": "1m",
                "exp_samples": [
                    {
                        "labels": "{}" if scenario is None else f'{{scenario="{scenario}"}}',
                        "value": value,
                    }
                    for scenario, value in expected
                ],
            }
        )
    fixture = tmp_path / "organization-query-tests.json"
    fixture.write_text(
        json.dumps(
            {
                "rule_files": [],
                "evaluation_interval": "1m",
                "tests": [
                    {
                        "interval": "1m",
                        "input_series": _organization_series(),
                        "promql_expr_test": cases,
                    }
                ],
            }
        )
    )
    # Resolve the existing lab image once to an immutable local ID; never pull or
    # race a mutable tag between creating the container and executing promtool.
    image = prom_lab._docker("image", "inspect", "--format", "{{.Id}}", prom_lab.IMAGE)
    container = prom_lab._docker(
        "create",
        "--label",
        f"{prom_lab.LABEL}={run_id}",
        "--network",
        "none",
        "--entrypoint",
        "/bin/promtool",
        image,
        "test",
        "rules",
        "/tmp/organization-query-tests.json",
    )
    try:
        prom_lab._docker("cp", str(fixture), f"{container}:/tmp/organization-query-tests.json")
        output = prom_lab._docker("start", "--attach", container)
        exit_code = prom_lab._docker("inspect", "--format", "{{.State.ExitCode}}", container)
        assert exit_code == "0", output
    finally:
        prom_lab._docker("rm", "-f", container, check=False)
