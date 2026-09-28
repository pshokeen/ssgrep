"""Tests for eval.run_eval: manifest preflight, arms, gates, payload determinism.

The fixture dataset is a tiny frozen-style artifact: native transcripts
ingested through the REAL indexer (with the embedding-model boundary swapped
for the deterministic offline fake used across tests/eval), plus
queries.jsonl / qrels.tsv / manifest.json whose episode targets are read
from the built index so they always exist. The runner is exercised through
its CLI entry point (``main``) and its ``run_eval`` API.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from eval import arms, harness, run_eval
from ssgrep import search as search_module
from ssgrep.indexing import indexer
from ssgrep.indexing.embed import DIMENSION
from ssgrep.pipeline import app as app_mod
from ssgrep.store import EPISODES_TABLE, LanceStore

pytestmark = pytest.mark.slow

EXACT_QUERY = "purple monkey dishwasher"

_FILLER = [
    "zebra",
    "quartz",
    "jigsaw",
    "kayak",
    "xylophone",
    "vortex",
    "nebula",
    "cactus",
    "falcon",
    "harbor",
    "mosaic",
    "tundra",
    "wombat",
    "yonder",
    "lizard",
    "octopus",
    "penguin",
    "rhino",
    "sphinx",
    "tiger",
    "umbrella",
    "viper",
    "walrus",
    "zephyr",
    "bamboo",
    "cobalt",
    "dune",
    "ember",
    "fjord",
    "gizmo",
    "hazel",
    "indigo",
    "jasmine",
    "kettle",
    "lagoon",
    "mango",
    "nectar",
    "onyx",
    "pebble",
    "quiver",
    "raven",
    "saffron",
    "tulip",
    "umber",
    "velvet",
    "willow",
    "xenon",
    "yarrow",
    "zinnia",
]

SESSION_COUNT = 2
EPISODES_PER_SESSION = 5
MATCH_SESSION = 0
MATCH_EPISODE = 0


def fake_vector(text: str) -> np.ndarray:
    """Deterministic ``(num_tokens, DIMENSION)`` matrix from character trigrams."""
    s = str(text)
    grams = [s[i : i + 3] for i in range(max(1, len(s) - 2))]
    if not grams:
        grams = [s]
    rows = []
    for gram in grams:
        seed = int.from_bytes(gram.encode("utf-8"), "little") % (2**32)
        rng = np.random.default_rng(seed)
        row = rng.standard_normal(DIMENSION).astype(np.float32)
        norm = np.linalg.norm(row)
        rows.append((row / np.maximum(norm, 1e-8)).astype(np.float32))
    return np.stack(rows).astype(np.float32)


class FakePyLateEmbedder:
    """CocoIndex provider stand-in producing ``(num_tokens, DIMENSION)`` matrices."""

    def __init__(self, model_name_or_path: str = "", *, device: str | None = None) -> None:
        self._model = model_name_or_path
        self._device = device

    def encode_many(self, texts: list[str], *, is_query: bool = False) -> list[np.ndarray]:
        return [fake_vector(text) for text in texts]

    async def encode_many_async(
        self, texts: list[str], *, is_query: bool = False
    ) -> list[np.ndarray]:
        return self.encode_many(texts, is_query=is_query)

    def __coco_memo_key__(self) -> object:
        return (self._model, self._device)


def _write_session(root: Path, session_id: str, episodes: list[tuple[str, str]]) -> Path:
    path = root / f"{session_id}.jsonl"
    records: list[dict] = []
    for index, (prompt, response) in enumerate(episodes):
        stamp = datetime(2025, 1, 1, tzinfo=UTC) + timedelta(minutes=index)
        records.append(
            {
                "type": "user",
                "cwd": "/work/app",
                "message": {"content": prompt},
                "timestamp": stamp.isoformat(),
                "sessionId": session_id,
            }
        )
        records.append(
            {
                "type": "assistant",
                "message": {"content": response},
                "timestamp": (stamp + timedelta(seconds=1)).isoformat(),
                "sessionId": session_id,
            }
        )
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    return path


def _episode_texts() -> list[tuple[str, str]]:
    episodes: list[tuple[str, str]] = []
    for session in range(SESSION_COUNT):
        for episode in range(EPISODES_PER_SESSION):
            if session == MATCH_SESSION and episode == MATCH_EPISODE:
                response = "The purple monkey dishwasher protocol is the answer."
            else:
                offset = (session * EPISODES_PER_SESSION + episode) % len(_FILLER)
                response = " ".join((_FILLER * 2)[offset : offset + 6]) + "."
            prompt = f"how do I fix the {_FILLER[(session + episode) % len(_FILLER)]} error"
            episodes.append((prompt, response))
    return episodes


def _write_transcripts(root: Path) -> None:
    for session in range(SESSION_COUNT):
        start = session * EPISODES_PER_SESSION
        episodes = _episode_texts()[start : start + EPISODES_PER_SESSION]
        _write_session(root, f"session-{session}", episodes)


def _episode_ids_by_position(index_dir: Path) -> dict[tuple[str, int], str]:
    with harness._private_data_dir(index_dir):
        rows = LanceStore().rows(EPISODES_TABLE, columns=["session_id", "episode_id"])
    # External roots get a content-hash suffix on the session id
    # (``session-0~8459a0a1``); strip it so positions map to the written ids.
    return {
        (
            str(row["session_id"]).split("~", 1)[0],
            int(str(row["episode_id"]).rsplit(":ep:", 1)[1]),
        ): str(row["episode_id"])
        for row in rows
    }


def _write_dataset_files(dataset_dir: Path, index_dir: Path) -> None:
    by_position = _episode_ids_by_position(index_dir)
    targets = {
        "q1": by_position[("session-0", 0)],
        "q2": by_position[("session-0", 1)],
        "q3": by_position[("session-1", 0)],
        "q4": by_position[("session-1", 1)],
        "q5": by_position[("session-0", 2)],
        "q6": by_position[("session-1", 2)],
    }
    queries = [
        {
            "id": "q1",
            "query": EXACT_QUERY,
            "class": "exact-identifier",
            "target_episode_ids": [targets["q1"]],
            "subagent_only": False,
            "split": "train",
        },
        {
            "id": "q2",
            "query": "jigsaw",
            "class": "exact-identifier",
            "target_episode_ids": [targets["q2"]],
            "subagent_only": False,
            "split": "train",
        },
        {
            "id": "q3",
            "query": "how do I fix the zebra error",
            "class": "paraphrase",
            "target_episode_ids": [targets["q3"]],
            "subagent_only": False,
            "split": "holdout",
        },
        {
            "id": "q4",
            "query": "kayak",
            "class": "paraphrase",
            "target_episode_ids": [targets["q4"]],
            "subagent_only": False,
            "split": "train",
        },
        {
            "id": "q5",
            "query": "vortex",
            "class": "exact-identifier",
            "target_episode_ids": [targets["q5"]],
            "subagent_only": False,
            "split": "holdout",
        },
        {
            "id": "q6",
            "query": "how do I fix the falcon error",
            "class": "paraphrase",
            "target_episode_ids": [targets["q6"]],
            "subagent_only": False,
            "split": "train",
        },
    ]
    (dataset_dir / "queries.jsonl").write_text("\n".join(json.dumps(row) for row in queries) + "\n")
    qrel_lines = []
    for query_id, target in targets.items():
        qrel_lines.append(f"{query_id}\t{target}\t3")
        secondary = (
            by_position[("session-0", 0)] if query_id == "q3" else by_position[("session-1", 0)]
        )
        qrel_lines.append(f"{query_id}\t{secondary}\t2")
    (dataset_dir / "qrels.tsv").write_text("\n".join(qrel_lines) + "\n")

    files = {}
    for path in sorted(dataset_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files[path.relative_to(dataset_dir).as_posix()] = run_eval._sha256(path)
    (dataset_dir / "manifest.json").write_text(
        json.dumps({"version": "v1", "files": files}, indent=2) + "\n"
    )


@pytest.fixture
def fixture_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A frozen-style dataset + built index, with the model boundary faked.

    Function-scoped on purpose: the repo's autouse ``isolated_environment``
    fixture (also function-scoped) sandboxes every live transcript root, and
    the index build must run inside that sandbox or discovery ingests the
    live corpus.
    """
    dataset_dir = tmp_path / "dataset"
    transcripts = dataset_dir / "transcripts"
    # Native transcripts live under transcripts/claude/ (the T15 frozen
    # layout; the runner's rebuild path requires this root).
    native_root = transcripts / "claude"
    native_root.mkdir(parents=True)
    _write_transcripts(native_root)

    monkeypatch.setattr("ssgrep.indexing.embed.ensure_model_downloaded", lambda *a, **kw: None)
    monkeypatch.setattr(app_mod, "ensure_model_downloaded", lambda *a, **kw: None)
    monkeypatch.setattr(app_mod, "ColBERTEmbedder", FakePyLateEmbedder)
    monkeypatch.setattr(search_module, "_query_matrix", lambda query: fake_vector(str(query)))
    monkeypatch.setenv("SSGREP_TRANSCRIPT_DIRS", f"native={native_root}")

    index_dir = tmp_path / "index"
    with harness._private_data_dir(index_dir):
        indexer.index(rebuild=True, allow_shrink=True, quiet=True)
        LanceStore().ensure_vector_index()
    _write_dataset_files(dataset_dir, index_dir)
    return SimpleNamespace(dataset_dir=dataset_dir, index_dir=index_dir)


# ---------------------------------------------------------------------------
# Preflight: manifest verification
# ---------------------------------------------------------------------------


def test_preflight_manifest_accepts_pristine_dataset(fixture_dataset: SimpleNamespace) -> None:
    manifest = run_eval.preflight_manifest(fixture_dataset.dataset_dir)
    assert manifest["version"] == "v1"
    assert "queries.jsonl" in manifest["files"]
    assert "transcripts/claude/session-0.jsonl" in manifest["files"]


def test_preflight_manifest_tamper_names_the_file(
    fixture_dataset: SimpleNamespace, tmp_path: Path
) -> None:
    tampered = tmp_path / "tampered"
    shutil.copytree(fixture_dataset.dataset_dir, tampered)
    target = tampered / "transcripts" / "claude" / "session-0.jsonl"
    content = bytearray(target.read_bytes())
    content[10] ^= 0xFF
    target.write_bytes(bytes(content))

    with pytest.raises(
        run_eval.ManifestError,
        match="sha256 mismatch for transcripts/claude/session-0.jsonl",
    ):
        run_eval.preflight_manifest(tampered)


def test_preflight_manifest_missing_file_named(
    fixture_dataset: SimpleNamespace, tmp_path: Path
) -> None:
    broken = tmp_path / "broken"
    shutil.copytree(fixture_dataset.dataset_dir, broken)
    (broken / "queries.jsonl").unlink()

    with pytest.raises(run_eval.ManifestError, match="queries.jsonl"):
        run_eval.preflight_manifest(broken)


def test_preflight_manifest_extra_file_named(
    fixture_dataset: SimpleNamespace, tmp_path: Path
) -> None:
    polluted = tmp_path / "polluted"
    shutil.copytree(fixture_dataset.dataset_dir, polluted)
    (polluted / "stray.bin").write_bytes(b"not in the manifest")

    with pytest.raises(run_eval.ManifestError, match="stray.bin"):
        run_eval.preflight_manifest(polluted)


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


def test_load_qrels_accepts_three_and_four_column_forms(tmp_path: Path) -> None:
    (tmp_path / "qrels.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\tep-1\t3\nq2\tQ0\tep-2\t2\n"
    )
    qrels = run_eval.load_qrels(tmp_path)
    assert qrels == {"q1": {"ep-1": 3}, "q2": {"ep-2": 2}}


def test_load_qrels_rejects_malformed_row(tmp_path: Path) -> None:
    (tmp_path / "qrels.tsv").write_text("q1\tep-1\n")
    with pytest.raises(run_eval.PreflightError, match="columns"):
        run_eval.load_qrels(tmp_path)


def test_load_queries_validates_required_fields(tmp_path: Path) -> None:
    (tmp_path / "queries.jsonl").write_text(
        json.dumps({"id": "q1", "query": "x", "class": "paraphrase"}) + "\n"
    )
    with pytest.raises(run_eval.PreflightError, match="missing fields"):
        run_eval.load_queries(tmp_path)


def test_load_queries_rejects_unknown_split(tmp_path: Path) -> None:
    (tmp_path / "queries.jsonl").write_text(
        json.dumps(
            {
                "id": "q1",
                "query": "x",
                "class": "paraphrase",
                "target_episode_ids": ["ep-1"],
                "split": "test",
            }
        )
        + "\n"
    )
    with pytest.raises(run_eval.PreflightError, match="split"):
        run_eval.load_queries(tmp_path)


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def test_derive_gates_matches_plan_formulas() -> None:
    baseline = {
        "summary": {"overall": {"ndcg@10": 0.5, "rr@10": 0.4, "r@50": 0.8}},
        "latency_p95_ms": 100.0,
        "index_size_bytes": 2000,
    }
    gates = run_eval.derive_gates(baseline)
    assert gates["ndcg@10_floor"] == pytest.approx(0.49)
    assert gates["mrr_floor"] == pytest.approx(0.39)
    assert gates["r@50_floor"] == pytest.approx(0.79)
    assert gates["p95_ceiling"] == pytest.approx(120.0)
    assert gates["size_ceiling"] == pytest.approx(1000.0)


def test_check_gates_flags_regression_and_passes_improvement() -> None:
    gates = {
        "ndcg@10_floor": 0.49,
        "mrr_floor": 0.39,
        "r@50_floor": 0.79,
        "p95_ceiling": 120.0,
        "size_ceiling": 1000.0,
    }
    regressed = {
        "summary": {"overall": {"ndcg@10": 0.47, "rr@10": 0.38, "r@50": 0.78}},
        "latency_p95_ms": 130.0,
        "index_size_bytes": 1200,
    }
    result = run_eval.check_gates(gates, regressed)
    assert result["all_pass"] is False
    assert result["checks"]["ndcg@10"]["pass"] is False
    assert result["checks"]["latency_p95_ms"]["pass"] is False

    improved = {
        "summary": {"overall": {"ndcg@10": 0.55, "rr@10": 0.45, "r@50": 0.85}},
        "latency_p95_ms": 90.0,
        "index_size_bytes": 800,
    }
    assert run_eval.check_gates(gates, improved)["all_pass"] is True


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


def test_arm_registry_and_env_restore(monkeypatch: pytest.MonkeyPatch) -> None:
    assert set(arms.ARMS) == {
        "default",
        "two_stage_on",
        "oversample_1",
        "nprobes_full",
        "brute_force_reference",
    }
    assert arms.ARMS["brute_force_reference"].computed is True
    assert arms.ARMS["default"].computed is False

    monkeypatch.delenv("SSGREP_TWO_STAGE", raising=False)
    with arms.arm_env(arms.ARMS["two_stage_on"]):
        assert arms.ARMS["two_stage_on"].env_overrides["SSGREP_TWO_STAGE"] == "on"
        assert search_module._two_stage_enabled() is True
    assert search_module._two_stage_enabled() is False


def test_resolve_dataset_dir() -> None:
    assert run_eval.resolve_dataset_dir("v1", None) == run_eval.DATASET_ROOT / "v1"
    explicit = Path("/tmp/custom")
    assert run_eval.resolve_dataset_dir("v1", explicit) == explicit
    with pytest.raises(run_eval.PreflightError):
        run_eval.resolve_dataset_dir(None, None)


# ---------------------------------------------------------------------------
# End-to-end CLI
# ---------------------------------------------------------------------------


def test_cli_smoke_writes_valid_payload(
    fixture_dataset: SimpleNamespace, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    json_out = tmp_path / "smoke.json"
    exit_code = run_eval.main(
        [
            "--dataset-dir",
            str(fixture_dataset.dataset_dir),
            "--index-dir",
            str(fixture_dataset.index_dir),
            "--no-rebuild",
            "--quick",
            "--json-out",
            str(json_out),
        ]
    )
    assert exit_code == 0
    payload = json.loads(json_out.read_text())
    assert payload["result_schema_version"] == "1.0"
    assert payload["arm"] == "default"
    assert "ndcg@10" in str(payload)
    assert "ndcg@10" in payload["summary"]["overall"]
    assert "class:exact-identifier" in payload["summary"]
    assert "class:paraphrase" in payload["summary"]
    assert "holdout" in payload["summary"]
    assert payload["per_query"]
    assert payload["provenance"]["git"]["commit"]
    assert "ir-measures" in payload["provenance"]["library_versions"]
    assert "SSGREP_OVERSAMPLE" in payload["provenance"]["tuning_constants"]
    assert payload["gates"] == {}
    assert payload["reference_arms"] == {}
    assert "token_vector_count" in payload
    assert payload["metric_limit"] == 100


def test_cli_tamper_rejected_before_scoring(
    fixture_dataset: SimpleNamespace, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tampered = tmp_path / "tampered"
    shutil.copytree(fixture_dataset.dataset_dir, tampered)
    target = tampered / "transcripts" / "claude" / "session-0.jsonl"
    content = bytearray(target.read_bytes())
    content[10] ^= 0xFF
    target.write_bytes(bytes(content))

    exit_code = run_eval.main(["--dataset-dir", str(tampered), "--quick"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "sha256 mismatch" in captured.err
    assert "session-0.jsonl" in captured.err


def test_cli_unknown_arm_is_usage_error(fixture_dataset: SimpleNamespace) -> None:
    exit_code = run_eval.main(
        [
            "--dataset-dir",
            str(fixture_dataset.dataset_dir),
            "--index-dir",
            str(fixture_dataset.index_dir),
            "--no-rebuild",
            "--arm",
            "no_such_arm",
        ]
    )
    assert exit_code == 2


def test_reference_arm_block_populated(fixture_dataset: SimpleNamespace, tmp_path: Path) -> None:
    payload = run_eval.run_eval(
        fixture_dataset.dataset_dir,
        index_dir=fixture_dataset.index_dir,
        rebuild=False,
        quick=True,
        with_reference=True,
        date="2026-08-22T00:00:00Z",
    )
    reference = payload["reference_arms"]["brute_force"]
    assert reference["arm"] == "brute_force_reference"
    assert "ann_recall100_ratio" in reference
    assert len(reference["per_query"]) == run_eval.QUICK_LIMIT


class _FrozenClock:
    """Deterministic perf_counter: ``step`` seconds per call, resettable between runs."""

    def __init__(self, step: float = 0.001) -> None:
        self.t = 0.0
        self.step = step

    def __call__(self) -> float:
        self.t += self.step
        return self.t


def _run_gated(
    fixture_dataset: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    second_step: float,
) -> tuple[dict, dict, Path]:
    """Run the eval twice on a frozen clock; the second run is gated on the first.

    Latency is the only wall-clock quantity in the gate, so a fake clock makes
    the p95 budget deterministic: run one ticks ``0.001`` s per call and run
    two ticks ``second_step`` s per call. ``parallel=False`` keeps the timing
    in this process, where the patched clock applies.
    """
    monkeypatch.setattr("time.perf_counter", clock := _FrozenClock())
    monkeypatch.setattr("eval.run_eval._perf_counter", clock)

    def run(baseline_path: Path | None = None) -> dict:
        return run_eval.run_eval(
            fixture_dataset.dataset_dir,
            index_dir=fixture_dataset.index_dir,
            rebuild=False,
            quick=True,
            date="2026-08-22T00:00:00Z",
            parallel=False,
            baseline_path=baseline_path,
        )

    payload = run()
    baseline = dict(payload)
    baseline["latency_p95_ms"] = 2.0 * float(payload["latency_p95_ms"])
    baseline["index_size_bytes"] = 2 * int(payload["index_size_bytes"])
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline))

    clock.t = 0.0
    clock.step = second_step
    return payload, run(baseline_path=baseline_path), baseline_path


def test_gates_derived_and_checked_from_baseline(
    fixture_dataset: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload, gated, baseline_path = _run_gated(
        fixture_dataset, monkeypatch, tmp_path, second_step=0.001
    )
    # The frozen clock really drove the measurement (1 ms per timed call).
    assert float(payload["latency_p95_ms"]) == pytest.approx(1.0)
    assert float(gated["latency_p95_ms"]) == pytest.approx(1.0)
    gates = gated["gates"]
    assert gates["from_baseline"] == str(baseline_path)
    ndcg10 = float(payload["summary"]["overall"]["ndcg@10"])
    assert gates["ndcg@10_floor"] == pytest.approx(ndcg10 - 0.01)
    assert gates["check"]["all_pass"] is True


def test_latency_gate_fails_when_second_run_exceeds_budget(
    fixture_dataset: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # 10x slower second run: p95 10 ms against a 2 ms baseline * 1.2 ceiling.
    _, gated, _ = _run_gated(fixture_dataset, monkeypatch, tmp_path, second_step=0.010)
    check = gated["gates"]["check"]
    latency = check["checks"]["latency_p95_ms"]
    assert latency["value"] == pytest.approx(10.0)
    assert latency["ceiling"] == pytest.approx(2.4)
    assert latency["pass"] is False
    assert check["all_pass"] is False
    # Specifically the latency check failed: every other gate still passes.
    others = {name: c["pass"] for name, c in check["checks"].items() if name != "latency_p95_ms"}
    assert others and all(passed is not False for passed in others.values())


def test_deterministic_payloads_byte_identical(
    fixture_dataset: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _FrozenClock()
    monkeypatch.setattr("time.perf_counter", clock)
    monkeypatch.setattr("eval.run_eval._perf_counter", clock)

    def run_once() -> dict:
        clock.t = 0.0
        # Each run starts from a byte-identical index copy: an incremental
        # index pass writes fragments, so reusing one live index_dir would
        # legitimately change index_size_bytes between runs.
        index_dir = fixture_dataset.index_dir.parent / "clone"
        if index_dir.exists():
            shutil.rmtree(index_dir)
        shutil.copytree(fixture_dataset.index_dir, index_dir)
        return run_eval.run_eval(
            fixture_dataset.dataset_dir,
            index_dir=index_dir,
            rebuild=False,
            quick=True,
            with_reference=True,
            date="2026-08-22T00:00:00Z",
            parallel=False,  # Sequential for deterministic timing
        )

    first = run_once()
    second = run_once()
    assert json.dumps(first, indent=2, sort_keys=True) == json.dumps(
        second, indent=2, sort_keys=True
    )


# ---------------------------------------------------------------------------
# Extended metric run
# ---------------------------------------------------------------------------


def test_extended_metric_run_preserves_final_and_dedupes() -> None:
    final = [("a", 1.0), ("b", 0.9), ("c", 0.8)]
    prefetch = [("b", 0.7), ("d", 0.6), ("e", 0.5), ("f", 0.4)]

    # (a) final preserved at front; (b) dedupes prefetch already in final;
    # (c) respects limit; (d) scores strictly descending (order preserved).
    ordered = run_eval._extended_metric_run(final, prefetch, limit=5)
    assert [eid for eid, _score in ordered] == ["a", "b", "c", "d", "e"]
    assert len(ordered) == 5
    scores = [score for _eid, score in ordered]
    assert all(scores[i] > scores[i + 1] for i in range(len(scores) - 1))

    # (e) limit larger than pool -> returns all unique episodes.
    all_unique = run_eval._extended_metric_run(final, prefetch, limit=100)
    assert [eid for eid, _score in all_unique] == ["a", "b", "c", "d", "e", "f"]
    assert len(all_unique) == 6

    # (f) limit 10 with a full 10-doc final page -> identical to final alone.
    final10 = [(f"e{i}", float(10 - i)) for i in range(10)]
    prefetch_more = [(f"p{i}", 0.0) for i in range(5)]
    truncated = run_eval._extended_metric_run(final10, prefetch_more, limit=10)
    assert [eid for eid, _score in truncated] == [f"e{i}" for i in range(10)]
    assert len(truncated) == 10


def test_metric_limit_option_plumbed_through(fixture_dataset: SimpleNamespace) -> None:
    payload = run_eval.run_eval(
        fixture_dataset.dataset_dir,
        index_dir=fixture_dataset.index_dir,
        rebuild=False,
        quick=True,
        limit=25,
        date="2026-08-22T00:00:00Z",
    )
    assert payload["metric_limit"] == 25
    assert len(payload["per_query"][0]["final"]["ranking"]) <= 10
    assert "r@50" in payload["summary"]["overall"]


# ---------------------------------------------------------------------------
# Prefetch recall aggregates
# ---------------------------------------------------------------------------


def test_prefetch_recall_aggregates_mean_math() -> None:
    """The helper aggregates per-query prefetch recall into group means."""
    per_query = [
        {
            "query_id": "q1",
            "prefetch_episode": {"r@10": 0.5, "r@50": 0.5, "r@100": 0.75},
            "prefetch_chunk": {"chunk_r@100": 0.6},
        },
        {
            "query_id": "q2",
            "prefetch_episode": {"r@10": 0.1, "r@50": 0.2, "r@100": 0.3},
            "prefetch_chunk": {"chunk_r@100": 0.4},
        },
        {
            "query_id": "q3",
            "prefetch_episode": {"r@10": 0.8, "r@50": 0.9, "r@100": 1.0},
            "prefetch_chunk": {"chunk_r@100": 0.9},
        },
    ]
    groups = {"overall": ["q1", "q2", "q3"], "classx": ["q1", "q2"]}
    agg = run_eval._prefetch_recall_aggregates(per_query, groups)

    assert agg["overall"]["prefetch_r@10"] == pytest.approx((0.5 + 0.1 + 0.8) / 3)
    assert agg["overall"]["prefetch_r@50"] == pytest.approx((0.5 + 0.2 + 0.9) / 3)
    assert agg["overall"]["prefetch_r@100"] == pytest.approx((0.75 + 0.3 + 1.0) / 3)
    assert agg["overall"]["prefetch_chunk_r@100"] == pytest.approx((0.6 + 0.4 + 0.9) / 3)
    assert agg["classx"]["prefetch_r@10"] == pytest.approx((0.5 + 0.1) / 2)
    assert agg["classx"]["prefetch_r@50"] == pytest.approx((0.5 + 0.2) / 2)
    assert agg["classx"]["prefetch_r@100"] == pytest.approx((0.75 + 0.3) / 2)
    assert agg["classx"]["prefetch_chunk_r@100"] == pytest.approx((0.6 + 0.4) / 2)

    # None (no relevant episodes) is excluded from the mean; 0.0 is kept.
    with_none = [
        {
            "query_id": "q1",
            "prefetch_episode": {"r@10": None, "r@50": 0.5, "r@100": 0.75},
            "prefetch_chunk": {"chunk_r@100": 0.6},
        },
        {
            "query_id": "q2",
            "prefetch_episode": {"r@10": 0.1, "r@50": 0.2, "r@100": 0.3},
            "prefetch_chunk": {"chunk_r@100": None},
        },
        {
            "query_id": "q3",
            "prefetch_episode": {"r@10": 0.0, "r@50": 0.9, "r@100": 1.0},
            "prefetch_chunk": {"chunk_r@100": 0.9},
        },
    ]
    agg = run_eval._prefetch_recall_aggregates(with_none, {"overall": ["q1", "q2", "q3"]})
    assert agg["overall"]["prefetch_r@10"] == pytest.approx((0.1 + 0.0) / 2)
    assert agg["overall"]["prefetch_r@50"] == pytest.approx((0.5 + 0.2 + 0.9) / 3)
    assert agg["overall"]["prefetch_r@100"] == pytest.approx((0.75 + 0.3 + 1.0) / 3)
    assert agg["overall"]["prefetch_chunk_r@100"] == pytest.approx((0.6 + 0.9) / 2)

    # All-None group -> None (nothing to average).
    all_none = [
        {
            "query_id": "q1",
            "prefetch_episode": {"r@10": None, "r@50": None, "r@100": None},
            "prefetch_chunk": {"chunk_r@100": None},
        }
    ]
    agg = run_eval._prefetch_recall_aggregates(all_none, {"overall": ["q1"]})
    assert agg["overall"]["prefetch_r@10"] is None
    assert agg["overall"]["prefetch_r@50"] is None
    assert agg["overall"]["prefetch_r@100"] is None
    assert agg["overall"]["prefetch_chunk_r@100"] is None


def test_summary_includes_prefetch_keys_after_cli_run(
    fixture_dataset: SimpleNamespace, tmp_path: Path
) -> None:
    """Every summary slice carries the deep-pool recall keys after a run."""
    payload = run_eval.run_eval(
        fixture_dataset.dataset_dir,
        index_dir=fixture_dataset.index_dir,
        rebuild=False,
        quick=True,
        date="2026-08-22T00:00:00Z",
    )
    prefetch_keys = {"prefetch_r@10", "prefetch_r@50", "prefetch_r@100", "prefetch_chunk_r@100"}
    assert payload["summary"]
    for group_name, slice_ in payload["summary"].items():
        assert prefetch_keys <= set(slice_), group_name
        for key in prefetch_keys:
            value = slice_[key]
            assert value is None or 0.0 <= float(value) <= 1.0, (group_name, key, value)
    assert "prefetch" in payload["metric_definitions"]
    assert "notes" in payload["metric_definitions"]["prefetch"]


def test_eval_worker_count_honours_override_and_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pool never exceeds the override, the CPU count, or the query count, and never hits 0."""
    monkeypatch.setattr(run_eval.os, "cpu_count", lambda: 4)
    monkeypatch.delenv("SSGREP_EVAL_WORKERS", raising=False)
    assert run_eval._eval_worker_count(100) == 4  # CPU-bound default
    assert run_eval._eval_worker_count(2) == 2  # never more workers than queries
    assert run_eval._eval_worker_count(0) == 1  # floor: ProcessPoolExecutor(0) would raise

    monkeypatch.setenv("SSGREP_EVAL_WORKERS", "2")
    assert run_eval._eval_worker_count(100) == 2  # explicit cap wins over CPU count
    monkeypatch.setenv("SSGREP_EVAL_WORKERS", "0")
    assert run_eval._eval_worker_count(100) == 4  # invalid override falls back
    monkeypatch.setenv("SSGREP_EVAL_WORKERS", "64")
    assert run_eval._eval_worker_count(8) == 8  # still bounded by the query count
