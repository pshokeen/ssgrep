"""The one seam between the worker and the combined reducer.

The collector consumes ``reduce.reduce_profile`` and ``reduce.methodology_for`` ONLY through
:class:`Reducer`, imported lazily so importing the worker never loads the reducer stack. A
different ``Reducer`` can be injected (tests build one from the real dataclasses).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ssgrep.accounting.frequency import FrequencyView
from ssgrep.accounting.valuation_inputs import Methodology


@dataclass(frozen=True)
class Reducer:
    reduce_profile: Callable[..., Any]  # reduce.reduce_profile(profile, evidence, events, *, ...)
    methodology_for: Callable[[FrequencyView], Methodology]


def default_reducer() -> Reducer:
    from ssgrep.accounting import reduce as reduce_module

    return Reducer(reduce_module.reduce_profile, reduce_module.methodology_for)
