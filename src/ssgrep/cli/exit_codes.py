"""Exit code contract for the ssgrep CLI.

Every command uses this fixed set of process exit codes to indicate its outcome
to callers, especially machine consumers like agents and scripts. The contract
makes machine-readable output (--json) truly useful: the exit code alone
distinguishes between success, failure modes, and empty results without
requiring stdout parsing.

Exit codes:
- 0: Success. Command completed its operation as requested.
- 1: Unexpected or internal failure. Unhandled exception, crash, or
     condition not covered by the other codes.
- 2: Usage error. Invalid, missing, or unrecognized arguments, flags, or
     required confirmations. Precedes any index or filesystem access.
- 3: No matching data. Command completed but found no results. Examples:
     - search query ran but matched nothing
     - show ref not found
     - (status always succeeds, even with empty index)
- 4: Missing or unusable index. Required by the operation and either does not
     exist or fails an integrity check.
"""

from __future__ import annotations

INTERNAL_FAILURE = 1
USAGE_ERROR = 2
NO_MATCHING_DATA = 3
MISSING_INDEX = 4
