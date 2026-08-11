"""CLI module for ssgrep.

usecli 0.1.76+ has fixed all compatibility issues, so this module now simply
delegates to usecli's main entry point.
"""

from __future__ import annotations

import sys


def main() -> None:
    """Run ssgrep CLI via usecli.

    usecli 0.1.76+ has fixed:
    1. Exit codes: parse-rejection path now correctly exits 2 (USAGE_ERROR)
    2. Version resolution: checks installed distribution first, before cwd's
       pyproject.toml
    3. Packaging: setuptools package-data is declared correctly

    Previous workarounds for these issues have been removed as they are no
    longer needed.
    """
    # Guard: ssgrep does not support Windows because it depends on fcntl
    # (a POSIX-only module). Check this BEFORE importing usecli, which would
    # eventually import ssgrep.store.generations and trigger the fcntl import.
    if sys.platform.startswith("win"):
        sys.stderr.write(
            "ssgrep is not supported on Windows. "
            "It requires macOS or Linux. Please use this software on a supported platform.\n"
        )
        sys.exit(1)

    from usecli import main as usecli_main

    usecli_main()


if __name__ == "__main__":
    main()
