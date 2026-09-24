"""ssgrep CLI entry point."""

from __future__ import annotations

import sys


def main() -> None:
    """Run ssgrep through usecli."""
    # Guard: ssgrep does not support Windows because it depends on fcntl
    # (a POSIX-only module).
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
