"""NeMo Data Designer synthetic session-generation workflow (Task 6).

Builds a one-time, fully fictional coding-session corpus stratified to the live
distribution measured by ``profile_live.py`` (T3, committed in
``eval/datasetgen/profile.json``). Only aggregated statistics from the live
corpus are used here -- counts, ratios, histograms, percentiles. No verbatim
transcript text ships into the generated dataset (privacy boundary D2).

Pinned external dependency: ``data-designer==0.9.1``. It declares ``pyarrow<25``
and ``rich<15`` upper bounds that conflict with ssgrep's runtime pins
(``pyarrow>=25.0.1``, ``rich>=15.0.0``); ``[tool.uv] override-dependencies``
lets both install together. Verified at implementation time against pyarrow
25.0.1 / rich 15.0.0.

Usage:

    uv run python -m eval.datasetgen.workflow --mode preview -n 8 --out /tmp/preview.parquet
    uv run python -m eval.datasetgen.workflow --mode create --out eval/datasetgen/sessions.parquet

The module also doubles as a Data Designer config source (it defines
``load_config_builder``), so the workflow is externally validatable:

    uv run data-designer validate eval/datasetgen/workflow.py

Generation requires ``OPENROUTER_API_KEY`` or ``OPENAI_API_KEY``; the CLI fails
loudly with a clear message when neither is present.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

try:
    import data_designer.config as dd
    from data_designer.interface import DataDesigner
except ImportError as exc:  # pragma: no cover - data-designer is a dev-only dep
    raise SystemExit(
        "data-designer==0.9.1 is required. Run `uv sync` (installs the dev "
        "group, which pins data-designer==0.9.1) and retry."
    ) from exc

DATA_DESIGNER_VERSION: str = "0.9.1"
DEFAULT_MODEL: str = "deepseek/deepseek-v4-flash-0731"
DEFAULT_TEMPERATURE: float = 0.9
DEFAULT_COST_CAP_USD: float = 25.0
DEFAULT_SEED: int = 0xCAFE
SESSION_BUDGET: int = 400
SMALL_RUNTIME_FLOOR: int = 40
SPIKE_RECORDS: int = 6

RUNTIMES: tuple[str, ...] = ("claude", "codex", "opencode", "pi", "prime-agent")

SCENARIO_CLASSES: tuple[str, ...] = (
    "exact-identifier",
    "error-string",
    "paraphrase",
    "multi-hop",
    "decision-rationale",
    "cross-runtime/project-scoped",
    "tool-failure-recovery",
)

DIFFICULTIES: tuple[str, ...] = ("easy", "medium", "hard")
LENGTH_BUCKETS: tuple[str, ...] = ("short", "medium", "long")

# Fully fictional project persona pool (D2: no real projects or persons).
PROJECT_POOL: tuple[str, ...] = (
    "aurora-inc/deployment-scheduler",
    "brightstar/cli-workbench",
    "cascade-config-registry",
    "daybreak-io/key-rotator",
    "ember-foundry/log-aggregator",
    "falcon-metrics/usage-billing",
    "glacier-tools/package-scan",
    "harbor-system/queue-proxy",
    "iris-works/state-snapshot",
    "jasperdev/artifact-archiver",
    "kepler-apps/uptime-monitor",
    "lumen-soft/release-trainer",
    "meridian-code/flag-service",
    "nightjar-edge/caching-layer",
    "onyx-harbor/migrate-helper",
    "pulsar-ink/oapi-wrapper",
)

# Fictional developer personas (D2 privacy boundary: not real people).
PERSONA_POOL: tuple[str, ...] = (
    "Ada Mercer",
    "David Okoro",
    "Elena Vasquez",
    "Finn Calloway",
    "Grace Lindqvist",
    "Imani Coleman",
    "Jules Ferreira",
    "Kai Nakamura",
    "Marek Novak",
    "Priya Anand",
    "Theo Marcotte",
    "Yuki Tanabe",
)

# Estimated USD per 1M tokens (input, output); used ONLY for projected-cost
# guardrails, never for billing.
PRICE_PER_MTOKENS: dict[str, tuple[float, float]] = {
    "openai/gpt-4o-mini": (0.15, 0.6),
    "openai/gpt-4o": (2.5, 10.0),
    "nvidia/nemotron-3-nano-30b-a3b": (0.05, 0.25),
    "anthropic/claude-sonnet-4": (3.0, 15.0),
    "deepseek/deepseek-v4-flash-0731": (0.09, 0.18),
}
_FALLBACK_PRICE: tuple[float, float] = (0.5, 1.5)

# Deterministic schema emitted by write_sessions_parquet for T7-T11 emitters.
SESSION_COLUMNS: tuple[str, ...] = (
    "session_id",
    "runtime",
    "scenario_class",
    "difficulty",
    "project",
    "persona",
    "episode_length_bucket",
    "num_episodes",
    "title",
    "summary",
    "files_touched",
    "tool_names",
    "episodes",
)

# Forbidden substrings that must never appear in generated text (privacy
# boundary, plus accidental real-corpus leakage guards).
#
# Operator-specific identifiers (your username, GitHub org, employer, internal
# service names) belong in SSGREP_EVAL_PRIVACY_NEEDLES as a comma-separated
# list rather than hardcoded here: this file ships with the product, so a
# hardcoded identity would leak the very thing the guard exists to catch.
_DEFAULT_PRIVACY_NEEDLES: tuple[str, ...] = (
    "/Users/",
    "/home/",
    "ssgrep",
    "ghq.github",
)


def _privacy_needles() -> tuple[str, ...]:
    extra = os.environ.get("SSGREP_EVAL_PRIVACY_NEEDLES", "")
    configured = tuple(item.strip() for item in extra.split(",") if item.strip())
    return _DEFAULT_PRIVACY_NEEDLES + configured


_PRIVACY_NEEDLES: tuple[str, ...] = _privacy_needles()


class SessionEpisode(BaseModel):
    """One prompt/response episode within a generated session."""

    prompt: str
    response: str


class SessionRecord(BaseModel):
    """Structured LLM output describing one synthetic coding session."""

    title: str
    session_id: str
    project: str
    summary: str
    files_touched: list[str]
    tool_names: list[str]
    episodes: list[SessionEpisode]


class TrackingDataDesigner(DataDesigner):
    """DataDesigner subclass that records the most recently created resource
    provider so callers can read exact per-run token usage."""

    _last_resource_provider: Any = None

    def _create_resource_provider(self, *args: Any, **kwargs: Any) -> Any:
        provider = super()._create_resource_provider(*args, **kwargs)
        self._last_resource_provider = provider
        return provider

    def get_last_usage(self) -> dict[str, dict[str, Any]]:
        """Per-model token and request usage of the most recent run."""
        if self._last_resource_provider is None:
            return {}
        return self._last_resource_provider.model_registry.get_model_usage_stats(
            total_time_elapsed=0.0
        )


def _load_profile(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def runtime_weights(budget: int, floor: int) -> list[float]:
    """Per-runtime category weights: floor the four small runtimes at ``floor``
    sessions each, give the remainder to opencode (live-dominant), normalized
    to sum 1 over the five runtimes."""
    targets: dict[str, int] = {r: floor for r in RUNTIMES if r != "opencode"}
    targets["opencode"] = max(floor, budget - floor * (len(RUNTIMES) - 1))
    return [targets[r] / budget for r in RUNTIMES]


def episode_count_weights(profile: dict[str, Any]) -> dict[int, float]:
    """Empirical weights over episodes-per-session from the live histogram,
    keyed by integer count and capped at 12 (live tail is long; capped for cost).
    """
    hist = profile["episodes_per_session"]["histogram"]
    weights: dict[int, float] = {}
    for key, count in hist.items():
        n = min(max(int(key), 1), 12)
        weights[n] = weights.get(n, 0.0) + count
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


@dd.custom_column_generator(required_columns=[])
def assign_runtime(df: pd.DataFrame) -> pd.DataFrame:
    """Full-column generator that cycles the five runtimes across rows so a
    small preview spanning several records still covers every category."""
    out = pd.DataFrame(index=df.index)
    out["runtime"] = [RUNTIMES[i % len(RUNTIMES)] for i in range(len(df))]
    return out


def build_config_builder(
    profile_path: Path,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = 4096,
    use_runtime_cycle: bool = False,
) -> dd.DataDesignerConfigBuilder:
    """Build the Data Designer config: sampler columns driven by profile.json,
    the LLM structured session column, and a local-callable validation column.

    When ``use_runtime_cycle`` is True the runtime column is a deterministic
    full-column generator that cycles the five runtimes, guaranteeing every
    category appears in a small preview. Otherwise the runtime column is a
    stratified CATEGORY sampler driven by profile.json (create mode).
    """
    profile = _load_profile(profile_path)

    builder = dd.DataDesignerConfigBuilder()

    if use_runtime_cycle:
        builder.add_column(
            dd.CustomColumnConfig(
                name="runtime",
                generation_strategy=dd.GenerationStrategy.FULL_COLUMN,
                generator_function=assign_runtime,
            )
        )
    else:
        weights = runtime_weights(SESSION_BUDGET, SMALL_RUNTIME_FLOOR)
        builder.add_column(
            dd.SamplerColumnConfig(
                name="runtime",
                sampler_type=dd.SamplerType.CATEGORY,
                params=dd.CategorySamplerParams(values=list(RUNTIMES), weights=weights),
            )
        )
    builder.add_column(
        dd.SamplerColumnConfig(
            name="scenario_class",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=list(SCENARIO_CLASSES)),
        )
    )
    builder.add_column(
        dd.SamplerColumnConfig(
            name="difficulty",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=list(DIFFICULTIES)),
        )
    )
    builder.add_column(
        dd.SamplerColumnConfig(
            name="project",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=list(PROJECT_POOL)),
        )
    )
    builder.add_column(
        dd.SamplerColumnConfig(
            name="persona",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=list(PERSONA_POOL)),
        )
    )
    builder.add_column(
        dd.SamplerColumnConfig(
            name="episode_length_bucket",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=list(LENGTH_BUCKETS)),
        )
    )
    count_weights = episode_count_weights(profile)
    counts_values = sorted(count_weights)
    builder.add_column(
        dd.SamplerColumnConfig(
            name="num_episodes",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=counts_values,
                weights=[count_weights[v] for v in counts_values],
            ),
        )
    )

    # LLM structured column: the whole session record.
    builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="session",
            model_alias="writer",
            system_prompt=(
                "You fabricate coding-agent session logs for a synthetic "
                "dataset. Invent project names, persons, file paths, commands, "
                "errors, and tool usage. Never reference a real company, "
                "project, person, or path. Return only the requested session "
                "record."
            ),
            prompt=(
                "Author a FICTIONAL coding session. Runtime: {{ runtime }}. "
                "Scenario class: {{ scenario_class }}. Difficulty: "
                "{{ difficulty }}. Project persona: {{ project }}. Developer: "
                "{{ persona }}. Episode-length bucket: "
                "{{ episode_length_bucket }}. Produce exactly "
                "{{ num_episodes }} episode prompt/response pairs; prompts must "
                "be standalone user requests (they will be searched later) and "
                "responses plausible, with code, file paths, and tool usage "
                "consistent with the runtime. Return the structured session "
                "record including unique session_id, title, project, summary, "
                "files_touched, tool_names, and episodes."
            ),
            output_format=SessionRecord,
        )
    )

    # Validation column: reject malformed / incomplete generations.
    builder.add_column(
        dd.ValidationColumnConfig(
            name="session_validity",
            target_columns=["session"],
            validator_type=dd.ValidatorType.LOCAL_CALLABLE,
            validator_params=dd.LocalCallableValidatorParams(
                validation_function=_validate_session_column,
                output_schema={
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "is_valid": {"type": ["boolean", "null"]},
                                    "error_message": {"type": "string"},
                                },
                                "required": ["is_valid"],
                            },
                        }
                    },
                },
            ),
        )
    )

    # Model config: provider selected by whichever API key is set.
    provider_name = "openrouter" if os.environ.get("OPENROUTER_API_KEY") else "openai"
    builder.add_model_config(
        dd.ModelConfig(
            alias="writer",
            model=model,
            provider=provider_name,
            skip_health_check=False,
            inference_parameters=dd.ChatCompletionInferenceParams(
                generation_type=dd.GenerationType.CHAT_COMPLETION,
                max_parallel_requests=4,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
        )
    )
    return builder


def _validate_session_column(df: pd.DataFrame) -> pd.DataFrame:
    """Local-call validator: each session is a complete structured record with
    plausible episode pairs and no privacy leaks."""

    def check(row: pd.Series) -> tuple[bool, str]:
        session = row.get("session")
        if not isinstance(session, dict):
            return False, "session missing or not an object"
        for field in ("title", "project", "summary", "files_touched", "tool_names", "episodes"):
            if not session.get(field):
                return False, f"missing or empty {field}"
        if len(session["title"]) > 200:
            return False, "title too long"
        episodes = session["episodes"]
        if not isinstance(episodes, list) or not episodes:
            return False, "no episodes"
        if len(episodes) > 12:
            return False, "too many episodes"
        if row.get("num_episodes") is not None and not pd.isna(row["num_episodes"]):
            if len(episodes) != int(row["num_episodes"]):
                return False, f"episode count mismatch {len(episodes)} vs {row['num_episodes']}"
        for ep in episodes:
            if not ep.get("prompt") or not ep.get("response"):
                return False, "episode missing prompt/response"
        blob = json.dumps(session, ensure_ascii=False)
        for needle in _PRIVACY_NEEDLES:
            if needle in blob:
                return False, f"privacy token {needle!r} leaked into generation"
        return True, ""

    verdicts = df.apply(check, axis=1)
    return pd.DataFrame(
        {
            "is_valid": [ok for ok, _ in verdicts],
            "error_message": [msg for _, msg in verdicts],
        }
    )


def write_sessions_parquet(
    df: pd.DataFrame, out_path: Path, *, drop_invalid: bool = True
) -> pd.DataFrame:
    """Flatten generated records into the emitter schema and write parquet.

    Rows whose ``session_validity`` column marks them invalid are dropped
    unless ``drop_invalid`` is False.
    """
    validity_map: dict[Any, dict[str, Any]] = {}
    if "session_validity" in df.columns:
        validity_map = df["session_validity"].to_dict()

    rows: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        validity = validity_map.get(idx, {"is_valid": True})
        is_valid = bool(validity.get("is_valid")) if isinstance(validity, dict) else False
        if drop_invalid and not is_valid:
            continue
        session = row.get("session")
        if not isinstance(session, dict):
            continue
        rows.append(
            {
                "session_id": session.get("session_id", ""),
                "runtime": row.get("runtime"),
                "scenario_class": row.get("scenario_class"),
                "difficulty": row.get("difficulty"),
                "project": row.get("project"),
                "persona": row.get("persona"),
                "episode_length_bucket": row.get("episode_length_bucket"),
                "num_episodes": int(row.get("num_episodes") or 0),
                "title": session.get("title", ""),
                "summary": session.get("summary", ""),
                "files_touched": session.get("files_touched", []),
                "tool_names": session.get("tool_names", []),
                "episodes": [
                    {"prompt": ep.get("prompt", ""), "response": ep.get("response", "")}
                    for ep in session.get("episodes", [])
                ],
            }
        )
    projected = pd.DataFrame(rows, columns=pd.Index(SESSION_COLUMNS))
    if not projected.empty:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        projected.to_parquet(out_path, index=False)
    return projected


def _token_usage_totals(usage: dict[str, dict[str, Any]]) -> tuple[int, int]:
    """Sum input/output tokens across all models in a usage snapshot."""
    in_total = out_total = 0
    for stats in usage.values():
        tokens = stats.get("token_usage") or {}
        in_total += tokens.get("input_tokens") or 0
        out_total += tokens.get("output_tokens") or 0
    return in_total, out_total


def projected_cost_usd(usage: dict[str, dict[str, Any]], model_name: str, records: int) -> float:
    """Scaled full-run cost projection from a spike run's token usage."""
    in_total, out_total = _token_usage_totals(usage)
    if in_total + out_total == 0:
        return 0.0
    rate = PRICE_PER_MTOKENS.get(model_name, _FALLBACK_PRICE)
    avg_price = (rate[0] + rate[1]) / 2
    per_record_tokens = (in_total + out_total) / SPIKE_RECORDS
    return per_record_tokens * avg_price * records / 1e6


def write_generation_config(
    *,
    model: str,
    temperature: float,
    seed: int,
    mode: str,
    projected_cost_usd: float,
) -> None:
    """Write generation_config.json capturing model id(s), seed, temperature."""
    cfg_path = Path(__file__).with_name("generation_config.json")
    payload = {
        "data-designer": DATA_DESIGNER_VERSION,
        "provider": "openrouter" if os.environ.get("OPENROUTER_API_KEY") else "openai",
        "model": model,
        "temperature": temperature,
        "seed": seed,
        "mode": mode,
        "session_budget": SESSION_BUDGET,
        "scenario_classes": list(SCENARIO_CLASSES),
        "project_personas": list(PROJECT_POOL),
        "personas_count": len(PERSONA_POOL),
        "projected_cost_usd": round(projected_cost_usd, 4),
    }
    cfg_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preview", "create"), default="preview")
    parser.add_argument("-n", "--num-records", type=int, default=None, help="Sessions requested.")
    parser.add_argument("--out", type=Path, help="Output parquet path.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--cost-cap-usd", type=float, default=DEFAULT_COST_CAP_USD)
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path(__file__).with_name("profile.json"),
        help="Profile JSON (T3) with live stratification targets.",
    )
    args = parser.parse_args(argv)

    if not os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        print(
            "FATAL: neither OPENROUTER_API_KEY nor OPENAI_API_KEY is set; "
            "synthetic session generation requires one of them.",
            file=sys.stderr,
        )
        return 2

    if not args.profile.exists():
        print(f"FATAL: profile not found: {args.profile}", file=sys.stderr)
        return 2

    np.random.seed(args.seed)
    random.seed(args.seed)

    provider_name = "openrouter" if os.environ.get("OPENROUTER_API_KEY") else "openai"
    provider_endpoint = (
        "https://openrouter.ai/api/v1"
        if provider_name == "openrouter"
        else "https://api.openai.com/v1"
    )
    api_key_env = "OPENROUTER_API_KEY" if provider_name == "openrouter" else "OPENAI_API_KEY"
    model_providers = [
        dd.ModelProvider(
            name=provider_name,
            endpoint=provider_endpoint,
            api_key=api_key_env,
        )
    ]

    artifact_dir = Path(__file__).resolve().parent / ".data-designer-artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    count = (
        args.num_records
        if args.num_records is not None
        else (SESSION_BUDGET if args.mode == "create" else 10)
    )
    if args.mode == "create" and count < 12:
        count = SESSION_BUDGET
    max_tokens = 4096 if args.mode == "create" else 2200
    cfg = build_config_builder(
        args.profile,
        model=args.model,
        temperature=args.temperature,
        max_tokens=max_tokens,
        use_runtime_cycle=args.mode == "preview",
    )

    projected_usd = 0.0
    if args.mode == "create" and count > 12:
        spike = TrackingDataDesigner(artifact_path=artifact_dir, model_providers=model_providers)
        spike.set_run_config(dd.RunConfig(disable_early_shutdown=True))
        spike.preview(cfg, num_records=SPIKE_RECORDS)
        usage = spike.get_last_usage()
        projected_usd = projected_cost_usd(usage, args.model, count)
        total_tokens = sum(_token_usage_totals(usage))
        print(
            f"[cost-guard] spike: {total_tokens} tokens / {SPIKE_RECORDS} records; "
            f"projected {projected_usd:.2f} USD for {count} sessions."
        )
        if projected_usd > args.cost_cap_usd:
            print(
                f"ABORT: projected cost ${projected_usd:.2f} exceeds --cost-cap-usd "
                f"${args.cost_cap_usd}; aborting before the full run.",
                file=sys.stderr,
            )
            return 3

    dd_instance = TrackingDataDesigner(artifact_path=artifact_dir, model_providers=model_providers)
    dd_instance.set_run_config(dd.RunConfig(disable_early_shutdown=True))

    if args.mode == "preview":
        result = dd_instance.preview(cfg, num_records=count)
        data = result.dataset
        out_path = args.out or Path(__file__).with_name("preview_samples.parquet")
        # data-designer's PreviewResults.dataset is DataFrame | None; a None
        # dataset is a hard failure that write_sessions_parquet would crash on
        # anyway (df.columns), so the None branch is unreachable in practice.
        projected = write_sessions_parquet(data, out_path)  # ty: ignore[invalid-argument-type]
        n_runtimes = projected["runtime"].nunique() if len(projected) else 0
        print(
            f"preview: {len(projected)} valid sessions across {n_runtimes} runtimes -> {out_path}"
        )
        if len(projected) < 8 or n_runtimes < 4:
            print(
                f"preview under threshold: {len(projected)} valid sessions across "
                f"{n_runtimes} runtimes (need >=8 and >=4)",
                file=sys.stderr,
            )
            return 4
    else:
        result = dd_instance.create(cfg, num_records=count)
        data = result.load_dataset()
        out_path = args.out or Path(__file__).with_name("sessions.parquet")
        projected = write_sessions_parquet(data, out_path)
        print(f"create: {len(projected)} valid sessions -> {out_path}")

    write_generation_config(
        model=args.model,
        temperature=args.temperature,
        seed=args.seed,
        mode=args.mode,
        projected_cost_usd=projected_usd,
    )
    return 0


def load_config_builder(params: Any | None = None) -> dd.DataDesignerConfigBuilder:
    """Data Designer config-source entry point: ``data-designer validate
    workflow.py`` calls this to obtain the config builder."""
    profile_path = Path(__file__).with_name("profile.json")
    model, temperature = DEFAULT_MODEL, DEFAULT_TEMPERATURE
    if params is not None and getattr(params, "argv", None):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--model", default=DEFAULT_MODEL)
        parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
        parsed, _ = parser.parse_known_args(list(params.argv))
        model = parsed.model or DEFAULT_MODEL
        temperature = parsed.temperature
    return build_config_builder(profile_path, model=model, temperature=temperature)


if __name__ == "__main__":
    sys.exit(main())
