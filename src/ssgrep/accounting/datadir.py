"""Data-root resolution without importing the heavy ``ssgrep.store`` package.

``ssgrep.store.paths.data_dir`` lives inside a package whose ``__init__``
imports LanceDB (hundreds of milliseconds). Capture must stay free when it is
disabled, so it resolves the same location itself: ``SSGREP_DATA_DIR`` when
set, otherwise the platform user-data directory for ``ssgrep``. A test pins
this function to ``data_dir()`` for both branches so the two cannot drift.
"""

from __future__ import annotations

import os
from pathlib import Path


def data_root() -> Path:
    override = os.environ.get("SSGREP_DATA_DIR")
    if override:
        return Path(override).expanduser().absolute()
    from platformdirs import user_data_path

    return Path(user_data_path("ssgrep", appauthor=False))
