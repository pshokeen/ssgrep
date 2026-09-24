"""Global filesystem locations for ssgrep persistence."""

from __future__ import annotations

import os
from pathlib import Path

from platformdirs import user_data_path


def data_dir() -> Path:
    """Return the application root, honoring ``SSGREP_DATA_DIR``."""
    override = os.environ.get("SSGREP_DATA_DIR")
    if override:
        return Path(override).expanduser().absolute()
    return Path(user_data_path("ssgrep", appauthor=False))


def database_dir() -> Path:
    return data_dir() / "lancedb"


def ensure_data_dir() -> Path:
    """Create the private application root for a write operation."""
    target = data_dir()
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        target.chmod(0o700)
    except OSError:
        pass
    return target
