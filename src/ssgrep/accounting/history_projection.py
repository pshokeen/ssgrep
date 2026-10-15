"""Canonical period read documents including explicitly inactive named archives."""

from __future__ import annotations

from ssgrep.accounting.contracts import SCHEMA_VERSION
from ssgrep.accounting.plain_codec import to_plain
from ssgrep.accounting.revisions import history, unit_key


def history_documents(rows, unit, today):
    result = []
    versions = [("", rows.namespace, rows.periods, True)]
    versions.extend((a.revision_name, a.namespace, a.periods, False) for a in rows.archives)
    for revision_name, namespace, periods, active in versions:
        by_key = {(p.period, p.revision): p for p in periods if p.namespace == namespace}
        for record in history(periods, namespace, unit, today):
            period = by_key[(record.period, record.revision)]
            estimates = next(
                (values for key, values in period.estimates if key == unit_key(unit)), ()
            )
            scenarios = [to_plain(e) for e in estimates]
            result.append(
                {
                    "format": "history_v2",
                    "schema": SCHEMA_VERSION,
                    "namespace": namespace,
                    "revision_name": revision_name,
                    "period": record.period,
                    "revision": record.revision,
                    "status": record.status if active else "superseded",
                    "current": active and record.current,
                    "supersedes": record.supersedes,
                    "net": to_plain(record.net),
                    "kind": record.kind,
                    "recomputable": record.recomputable,
                    "scenarios": scenarios,
                }
            )
    return sorted(
        result, key=lambda r: (r["period"], r["namespace"], r["revision_name"], r["revision"])
    )
