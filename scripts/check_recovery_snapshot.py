"""Read-only, value-level preservation check for a recovered ssgrep snapshot.

Usage: uv run python scripts/check_recovery_snapshot.py BACKUP_DATA RECOVERED_DATA
Hashes archived rows, including vectors, without printing transcript contents.
Only availability metadata may change. Missing sources are determined from the
backup's recorded paths; original transcript completeness is not inferred.

Run this after any change that could invalidate CocoIndex memos for a corpus
with tombstoned sources (an ssgrep upgrade, a pipeline code change, a manual
memo-journal reset): copy the data dir aside as BACKUP_DATA *before*
reconciling, reconcile as usual, then compare the backup against the live
data dir (``ssgrep status`` prints its path). ``preserved: true`` confirms
every row a tombstoned source owned before the reconcile -- session,
episode, and chunk rows, including embedding vectors -- is still present and
byte-identical after it (see ``ssgrep/pipeline/archive.py``).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

from ssgrep.pipeline.sources import read_registry
from ssgrep.store import LanceStore

TABLES = {"sessions": "session_id", "episodes": "episode_id", "chunks": "chunk_id"}
AVAILABILITY = {"source_status", "absent_since"}
VECTORS = {"vector": np.float16, "proxy_vector": np.float32}


def snapshot(root: str, expected: dict | None = None) -> dict:
    os.environ["SSGREP_DATA_DIR"] = root
    repo = LanceStore()
    try:
        registry = read_registry(repo)
        path_cache = {}
        output = {"rows": {}, "counts": {}, "sessions": [], "state": repo.get_meta("index_state")}
        for name, primary_key in TABLES.items():
            table = repo.table(name)
            if table is None:
                raise AssertionError(f"missing table: {name}")
            count = repo.count(name)
            assert count > 0, f"empty table: {name}"
            output["counts"][name] = count
            hashes = {}
            seen = 0
            for batch in table.search().limit(count).to_batches(batch_size=128):
                for index in range(batch.num_rows):
                    seen += 1
                    key = batch.column(primary_key)[index].as_py()
                    if name == "sessions":
                        output["sessions"].append(key)
                    if expected is not None:
                        if key not in expected["rows"][name]:
                            continue
                    else:
                        field = "path" if name == "sessions" else "source_path"
                        path = batch.column(field)[index].as_py()
                        if name == "sessions" and path in registry:
                            path = registry[path].path
                        if path not in path_cache:
                            path_cache[path] = Path(path).exists()
                        if path_cache[path]:
                            continue
                    values = {
                        field: batch.column(field)[index].as_py()
                        for field in batch.schema.names
                        if field not in AVAILABILITY and field not in VECTORS
                    }
                    digest = hashlib.sha256(
                        json.dumps(values, sort_keys=True, default=str, ensure_ascii=True).encode()
                    )
                    for field, dtype in VECTORS.items():
                        if field not in batch.schema.names:
                            continue
                        value = batch.column(field)[index]
                        digest.update(field.encode())
                        if not value.is_valid:
                            digest.update(b"null")
                        else:
                            vector = np.asarray(value.as_py(), dtype=dtype)
                            digest.update(str(vector.shape).encode())
                            digest.update(vector.tobytes())
                    assert key not in hashes, f"duplicate key in {name}: {key}"
                    hashes[key] = digest.hexdigest()
            assert seen == count, f"scan truncated in {name}: {seen}/{count}"
            assert hashes, f"no protected rows found in {name}"
            output["rows"][name] = hashes
        return output
    finally:
        repo.close()


def differences(before: dict, after: dict) -> dict:
    report = {}
    for table in TABLES:
        old, new = before["rows"][table], after["rows"][table]
        missing = sorted(old.keys() - new.keys())
        changed = sorted(key for key in old.keys() & new.keys() if old[key] != new[key])
        report[table] = {
            "protected": len(old),
            "missing": len(missing),
            "changed": len(changed),
            "examples": (missing + changed)[:5],
        }
    return report


def main() -> int:
    before = snapshot(sys.argv[1])
    after = snapshot(sys.argv[2], expected=before)
    # Instrument negative control: one changed digest must be detected.
    probe = {**before, "rows": {name: dict(values) for name, values in before["rows"].items()}}
    key = next(iter(probe["rows"]["chunks"]))
    probe["rows"]["chunks"][key] = "deliberate mutation"
    assert differences(before, probe)["chunks"]["changed"] == 1
    report = differences(before, after)
    lost_sessions = set(before["sessions"]) - set(after["sessions"])
    ok = (
        after["state"] == "ready"
        and not lost_sessions
        and all(not row["missing"] and not row["changed"] for row in report.values())
    )
    print(
        json.dumps(
            {
                "preserved": ok,
                "before_counts": before["counts"],
                "after_counts": after["counts"],
                "after_state": after["state"],
                "lost_sessions": len(lost_sessions),
                "tables": report,
            }
        ),
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
