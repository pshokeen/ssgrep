"""Paired capture-latency measurement (not collected by pytest by default).

    uv run python tests/accounting/perf_capture.py [--iterations 4000] [--warmup 300]

Measures the overhead the capture hot path adds to ``services.api.search`` and
``services.api.show`` against a frozen, instant retrieval leaf, so the numbers
are the cost of capture itself rather than of retrieval. Each iteration is a
*pair*: the same call with capture off (no policy file) and on (enabled
profile), alternating order to cancel drift, in one process with warm
imports. The report gives p50/p95/p99 of the per-pair added milliseconds; it
only reports -- there are no assertions to tune. Also reported separately:
cold first-capture cost (key creation, segment creation, imports) and added
latency under a contending CPU-bound worker process.

Everything runs in a throwaway ``SSGREP_DATA_DIR``; the real data root, index,
and settings are never touched.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import platform
import shutil
import statistics
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _burn(stop) -> None:  # pragma: no cover - contention worker
    value = 0
    while not stop.is_set():
        value += 1


def _percentile(sorted_values: list[float], fraction: float) -> float:
    index = min(len(sorted_values) - 1, int(round(fraction * (len(sorted_values) - 1))))
    return sorted_values[index]


def _report(label: str, added_ns: list[int]) -> None:
    ms = sorted(value / 1e6 for value in added_ns)
    print(
        f"{label:<34} n={len(ms):<6} p50={_percentile(ms, 0.50):7.4f} ms  "
        f"p95={_percentile(ms, 0.95):7.4f} ms  p99={_percentile(ms, 0.99):7.4f} ms  "
        f"max={ms[-1]:7.4f} ms  mean={statistics.fmean(ms):7.4f} ms"
    )


def _install_fixtures():
    import ssgrep.search as search_module
    import ssgrep.search.detail as detail_module
    from ssgrep.utilities.types import EpisodeDetail, ResultCard, SearchResponse

    cards = [
        ResultCard(
            ref=f"claude:s{i}:ep:{i}",
            title=f"t{i}",
            timestamp=None,
            score=0.9 - i / 100,
            excerpt="x" * 200,
            files_touched=(),
            is_subagent=False,
        )
        for i in range(10)
    ]
    response = SearchResponse(results=cards, total_matches=10)
    detail = EpisodeDetail(
        episode_id="e",
        session_id="s",
        title="t",
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        git_branch=None,
        cwd="/w",
        prompt_text="p",
        response_text="r",
        files_touched=(),
        tool_names=(),
        is_subagent=False,
    )
    vars(search_module)["search"] = lambda query, **_k: response
    vars(detail_module)["show"] = lambda ref: detail


def _paired(iterations: int, warmup: int, mode: str, policy_path: Path, enabled: bytes) -> dict:
    """Paired added latency per call: ``api.X`` minus the bare retrieval leaf.

    ``api.X`` is the leaf plus capture and nothing else, so the difference is
    exactly what capture adds. Modes decide the policy state around the call:

    * ``steady``: policy file present and unchanged (production steady state);
    * ``disabled``: no policy file (the default state of every user);
    * ``policy_changed``: the file's identity changes before every captured
      call, forcing a re-read and re-parse each time (worst case bound).
    """
    import ssgrep.search as search_module
    import ssgrep.search.detail as detail_module
    from ssgrep.services import api

    leaf = {"search": vars(search_module)["search"], "show": vars(detail_module)["show"]}
    wrapped = {"search": api.search, "show": api.show}
    args = {
        "search": ("how did we fix the payments flake",),
        "show": ("claude:s1:ep:1",),
    }
    kwargs = {"search": {"limit": 10, "token_budget": 1500}, "show": {}}

    def timed(function, which: str) -> int:
        start = time.perf_counter_ns()
        function(*args[which], **kwargs[which])
        return time.perf_counter_ns() - start

    if mode == "disabled":
        policy_path.unlink(missing_ok=True)
    else:
        policy_path.write_bytes(enabled)
    results: dict[str, list[int]] = {"search": [], "show": []}
    for which in ("search", "show"):
        for i in range(warmup + iterations):
            if mode == "policy_changed":
                policy_path.unlink(missing_ok=True)
                policy_path.write_bytes(enabled)
                os.utime(policy_path, ns=(i + 10, i + 10))
            if i % 2 == 0:  # alternate order to cancel drift and cache-warmth bias
                bare = timed(leaf[which], which)
                full = timed(wrapped[which], which)
            else:
                full = timed(wrapped[which], which)
                bare = timed(leaf[which], which)
            if i >= warmup:
                results[which].append(full - bare)
    return results


COLD_CHILD = """
import os, sys, time
sys.path.insert(0, {root!r})
os.environ['SSGREP_DATA_DIR'] = {data!r}
from ssgrep.accounting import capture
from ssgrep.accounting.contracts import Operation, Outcome
start = time.perf_counter_ns()
execution = capture.begin(Operation.SEARCH, text='how did we fix the payments flake', limit=10)
capture.complete(execution, Outcome.OK)
print(time.perf_counter_ns() - start)
"""


def _cold_process(policy_path: Path, enabled: bytes, samples: int) -> tuple[list[int], list[int]]:
    """First-call cost in fresh interpreters: capture disabled vs enabled (cold caches)."""
    import subprocess

    def sample(on: bool) -> int:
        if on:
            policy_path.write_bytes(enabled)
        else:
            policy_path.unlink(missing_ok=True)
        code = COLD_CHILD.format(root=str(ROOT), data=os.environ["SSGREP_DATA_DIR"])
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        return int(done.stdout.strip())

    off = [sample(False) for _ in range(samples)]
    on = [sample(True) for _ in range(samples)]
    return off, on


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--iterations", type=int, default=4000)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--cold-samples", type=int, default=15)
    args = parser.parse_args()
    if args.iterations < 2000:
        parser.error("--iterations must be at least 2000 for a meaningful p99")

    data_root = Path(tempfile.mkdtemp(prefix="ssgrep-perf-"))
    os.environ["SSGREP_DATA_DIR"] = str(data_root)
    try:
        _install_fixtures()
        from ssgrep.accounting import policy

        policy.enable_collection("personal-local")
        policy_path = policy.policy_path()
        enabled = policy_path.read_bytes()
        policy_path.unlink()

        print(f"python {platform.python_version()} on {platform.platform()}")
        print(f"cpu_count={os.cpu_count()} iterations={args.iterations} warmup={args.warmup}")
        print("added latency per call = api.X (leaf + capture) - bare leaf, paired, alternating")

        for mode in ("steady", "disabled", "policy_changed"):
            for which, values in _paired(
                args.iterations, args.warmup, mode, policy_path, enabled
            ).items():
                _report(f"idle {mode}, {which}", values)

        stop = multiprocessing.Event()
        workers = [
            multiprocessing.Process(target=_burn, args=(stop,))
            for _ in range(max(1, (os.cpu_count() or 2)))
        ]
        [worker.start() for worker in workers]
        try:
            loaded = _paired(args.iterations, args.warmup, "steady", policy_path, enabled)
        finally:
            stop.set()
            [worker.join() for worker in workers]
        for which, values in loaded.items():
            _report(f"loaded({len(workers)} spinners) steady, {which}", values)

        off, on = _cold_process(policy_path, enabled, args.cold_samples)
        print(
            f"cold process first call (n={len(on)}): disabled median="
            f"{statistics.median(off) / 1e6:.3f} ms max={max(off) / 1e6:.3f} ms; enabled median="
            f"{statistics.median(on) / 1e6:.3f} ms max={max(on) / 1e6:.3f} ms "
            "(includes lazy imports, key + segment creation)"
        )
        print("gate under evaluation: p95 <= 2 ms, p99 <= 5 ms (reported only, not asserted)")
    finally:
        shutil.rmtree(data_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
