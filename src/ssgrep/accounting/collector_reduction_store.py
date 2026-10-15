"""Root/session native-row manifests in the existing immutable collector store."""

from __future__ import annotations

import hashlib
from collections import defaultdict

from ssgrep.accounting import collector_store as store
from ssgrep.accounting.collector_types import CollectError
from ssgrep.accounting.plain_codec import to_plain

FORMAT = "root-reduction-v1"
ROWS = (
    "executions",
    "invocations",
    "session_roots",
    "session_parents",
    "session_cwd",
    "invocation_sessions",
)


def _groups(document):
    sessions = {(row[0], row[1]): row[2] for row in document.get("session_roots", ())}
    grouped = defaultdict(lambda: {field: [] for field in ROWS})
    for field in ROWS:
        for row in document.get(field, ()):
            if field in {"executions", "invocations"}:
                runtime, session = row.get("runtime"), row.get("session_id")
                root = sessions.get((runtime, session))
                if root is None and runtime and session:
                    root = runtime + ":" + session
                if root is None:
                    root = (
                        "@journal:" + hashlib.sha256(row["execution_id"].encode()).hexdigest()[:2]
                    )
            elif field == "invocation_sessions":
                root = sessions.get((row[1], row[2]), row[1] + ":" + row[2])
            else:
                root = sessions.get((row[0], row[1]), row[0] + ":" + row[1])
            grouped[root][field].append(row)
    return dict(grouped)


def expand_document(profile, raw):
    if raw.get("format") is None:
        return raw
    if raw.get("format") != FORMAT:
        raise CollectError("unsupported_schema", "Collector native-row format is unsupported.")
    result = dict(raw["summary"])
    result.update((field, []) for field in ROWS)
    for filename in raw["roots"].values():
        child = store.read_blob(profile, filename)
        if not isinstance(child, dict):
            raise CollectError("state_corrupt", "Collector native-row document is invalid.")
        for field in ROWS:
            result[field].extend(child[field])
    result["executions"].sort(key=lambda row: (row["ts_utc"], row["execution_id"]))
    result["invocations"].sort(
        key=lambda row: (row["runtime"], row["session_id"], row["invocation_id"])
    )
    for field in ROWS[2:]:
        result[field].sort()
    return result


def write_document(profile, name, document, *, previous=None, prior_manifest=None):
    """Write a complete logical document; optional prior typed rows reuse unchanged roots."""
    groups = _groups(document)
    old = _groups(previous) if previous is not None else {}
    roots = {}
    files = (
        prior_manifest.get("roots", {})
        if prior_manifest and prior_manifest.get("format") == FORMAT
        else {}
    )
    for root, rows in sorted(groups.items()):
        if root in files and rows == old.get(root):
            roots[root] = files[root]
        else:
            roots[root] = store.write_blob(profile, "reduction-root", rows)
    summary = {field: value for field, value in document.items() if field not in ROWS}
    return store.write_blob(
        profile,
        name,
        {
            "format": FORMAT,
            "summary": summary,
            "roots": roots,
            "frequency_refs": sorted(
                {
                    row["frequency_view_id"]
                    for row in document.get("invocations", ())
                    if row.get("frequency_view_id")
                }
            ),
        },
    )


def save_incremental(profile, name, document, blobs, *, previous=None):
    """Reuse dataclass identity before serialization, not after serializing every root."""

    def groups(doc):
        found = defaultdict(lambda: {field: [] for field in ROWS})
        mapping = {(runtime, sid): root for runtime, sid, root in doc.session_roots}
        for field in ROWS:
            for row in getattr(doc, field):
                if field in {"executions", "invocations"}:
                    root = mapping.get((row.runtime, row.session_id))
                    if root is None and row.runtime and row.session_id:
                        root = row.runtime + ":" + row.session_id
                    if root is None:
                        root = (
                            "@journal:" + hashlib.sha256(row.execution_id.encode()).hexdigest()[:2]
                        )
                elif field == "invocation_sessions":
                    root = mapping.get((row[1], row[2]), row[1] + ":" + row[2])
                else:
                    root = mapping.get((row[0], row[1]), row[0] + ":" + row[1])
                found[root][field].append(row)
        return found

    current = groups(document)
    old = groups(previous) if previous is not None else {}
    manifest = store.read_blob(profile, blobs.get(name))
    files = manifest.get("roots", {}) if manifest and manifest.get("format") == FORMAT else {}
    roots = {}
    for root, rows in sorted(current.items()):
        if root in files and rows == old.get(root):
            roots[root] = files[root]
        else:
            roots[root] = store.write_blob(profile, "reduction-root", to_plain(rows))
    summary = {
        field: getattr(document, field)
        for field in ("coverage", "warnings", "digest", "session_last", "lookup_schema")
    }
    blobs[name] = store.write_blob(
        profile,
        name,
        {
            "format": FORMAT,
            "summary": to_plain(summary),
            "roots": roots,
            "frequency_refs": sorted(
                {row.frequency_view_id for row in document.invocations if row.frequency_view_id}
            ),
        },
    )


def references(raw):
    return set(raw.get("roots", {}).values()) if raw.get("format") == FORMAT else set()
