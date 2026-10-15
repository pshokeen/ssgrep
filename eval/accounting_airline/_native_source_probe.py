"""Private read-only probe of the unchanged production native-source authority.

The positive reference is resolved only through the actual index and the configured
runtime discovery roots. The negative read uses those same roots, without adding
consent, replacing a locator or injecting source records.
"""

from __future__ import annotations

import json
import os
import sys
import time


def main() -> None:
    from pathlib import Path

    from ssgrep.accounting import policy
    from ssgrep.accounting.native_source import SourceReadError, open_source
    from ssgrep.accounting.reduce_io import native_source_adapter

    request = json.loads(sys.stdin.read())
    adapter = native_source_adapter(
        policy.load_profile_policy("personal-local"),
        os.environ,
        now_ms=time.time_ns() // 1_000_000,
    )
    result = adapter.resolve(request["ref"])
    reference = adapter.observed_references.get(request["ref"])
    negative = "unexpectedly_approved"
    try:
        descriptor = open_source(Path(request["outside_root"]), adapter.approved_roots)
    except SourceReadError as error:
        negative = error.status.value
    else:
        os.close(descriptor)
    print(
        json.dumps(
            {
                "status": result.status.value,
                "session": list(result.line.key) if result.line is not None else None,
                "path": str(reference.path) if reference is not None else None,
                "source_digest": result.source_digest,
                "source_bytes": reference.source_bytes if reference is not None else None,
                "record_digest_count": len(result.record_digests),
                "native_identities": [identity for identity, _ in result.record_digests],
                "outside_root_status": negative,
            }
        )
    )


if __name__ == "__main__":
    main()
