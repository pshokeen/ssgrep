"""Named evaluation arms for the standardized runner (``eval.run_eval``).

An arm is a named retrieval configuration. Two kinds exist:

* **Env-knob arms** — a fixed set of ``SSGREP_*`` environment overrides
  applied (with the ``arm_env`` context manager) around the whole run:
  ingestion AND ranking. Each override is written verbatim; the production
  modules apply their own truthy/clamp semantics
  (``src/ssgrep/search/__init__.py:67-91``, ``store._env_int``), so an arm
  never re-implements knob parsing — it just pins the value the production
  code sees.

* **Computed arms** — ``brute_force_reference`` is NOT an environment
  configuration. It declares ``computed=True`` and the runner substitutes
  ``eval.ranking.brute_force_ranking`` for the prefetch/final rankings
  per query (no env override exists that turns production search into an
  exhaustive scan). Its numbers land in the payload's ``reference_arms``
  block, not in the arm's summary.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Arm:
    """One named evaluation configuration."""

    name: str
    label: str
    env_overrides: Mapping[str, str] = field(default_factory=dict)
    computed: bool = False


ARMS: dict[str, Arm] = {
    "default": Arm(
        name="default",
        label="production environment (every knob at its shipped default)",
    ),
    "two_stage_on": Arm(
        name="two_stage_on",
        label="two-stage prefiltered search enabled (SSGREP_TWO_STAGE=on)",
        env_overrides={"SSGREP_TWO_STAGE": "on"},
    ),
    "oversample_1": Arm(
        name="oversample_1",
        label="candidate pool oversample forced to its minimum (SSGREP_OVERSAMPLE=1)",
        env_overrides={"SSGREP_OVERSAMPLE": "1"},
    ),
    "nprobes_full": Arm(
        name="nprobes_full",
        label="every IVF partition probed (SSGREP_NPROBES=512)",
        env_overrides={"SSGREP_NPROBES": "512"},
    ),
    "brute_force_reference": Arm(
        name="brute_force_reference",
        label=(
            "computed reference arm: per-query exhaustive MaxSim scan "
            "(not an environment configuration)"
        ),
        computed=True,
    ),
}


def resolve(name: str) -> Arm:
    """Look up one arm by name; unknown names fail loudly with the registry."""
    try:
        return ARMS[name]
    except KeyError:
        registered = ", ".join(ARMS)
        raise ValueError(f"unknown arm {name!r}; known arms: {registered}") from None


@contextmanager
def arm_env(arm: Arm) -> Iterator[None]:
    """Apply ``arm.env_overrides`` for the block, restoring exactly after.

    An arm without overrides is a no-op. The values are written verbatim —
    the production modules decide truthiness/clamping, so the runner and the
    measured code can never disagree about what was measured.
    """
    previous: dict[str, str | None] = {name: os.environ.get(name) for name in arm.env_overrides}
    os.environ.update(arm.env_overrides)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


__all__ = ["ARMS", "Arm", "arm_env", "resolve"]
