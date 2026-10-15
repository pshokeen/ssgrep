"""Entry point behaviour (offline) and the full scripted-standin pipeline (docker + real index)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from eval.accounting_airline import __main__ as entry, database, dataset
from eval.accounting_airline.gates import BLOCKED, NOT_RUN, PASS


def test_arithmetic_replay_seed_keeps_original_locator_index_without_copying_it(
    tmp_path: Path,
) -> None:
    from eval.accounting_airline.lab_product import copy_accounting

    source, destination = tmp_path / "source", tmp_path / "arithmetic"
    profile = source / "accounting" / "personal-local"
    (profile / "journal").mkdir(parents=True)
    (source / "accounting" / "policy.json").write_text("{}")
    (profile / "fingerprint.key").write_bytes(b"k" * 32)
    (profile / "journal" / "seg-1.jsonl").write_text('{"raw":"execution"}\n')
    index = source / "lancedb"
    index.mkdir(mode=0o700)
    locator = index / "owned-index-marker"
    original = b"original native locator; no derived transcript mirror"
    locator.write_bytes(original)

    copy_accounting(source, destination, journal_only=True)

    alias = destination / "lancedb"
    assert alias.is_symlink()
    assert alias.resolve(strict=True) == index.resolve(strict=True)
    assert (alias / locator.name).stat().st_ino == locator.stat().st_ino
    assert locator.read_bytes() == original
    assert not (destination / "accounting" / "personal-local" / "collector").exists()


def test_arithmetic_replay_seed_rejects_redirected_index_authority(tmp_path: Path) -> None:
    from eval.accounting_airline.lab_product import ProductError, copy_accounting

    source = tmp_path / "source"
    profile = source / "accounting" / "personal-local"
    (profile / "journal").mkdir(parents=True)
    (source / "accounting" / "policy.json").write_text("{}")
    (profile / "fingerprint.key").write_bytes(b"k" * 32)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (source / "lancedb").symlink_to(elsewhere, target_is_directory=True)

    with pytest.raises(ProductError, match="owned index"):
        copy_accounting(source, tmp_path / "arithmetic", journal_only=True)


def test_controlled_arithmetic_method_is_not_the_production_model() -> None:
    from eval.accounting_airline._arithmetic_collect import fixture_methodology
    from eval.accounting_airline.lab_product import reviewed_view_documents
    from ssgrep.accounting.frequency import build_view
    from ssgrep.accounting.reduce import methodology_for

    view = build_view(reviewed_view_documents(["MCT-DEN-071"]))
    production, fixture = methodology_for(view), fixture_methodology(view)
    assert fixture.label.startswith("airline-lab-arithmetic-v1:")
    assert fixture.hash != production.hash
    assert fixture.units == production.units
    assert fixture.thresholds == production.thresholds
    assert fixture.frequency_view == production.frequency_view


def test_fixture_whitelist_known_common_identifiers_are_not_unknown_or_rare() -> None:
    from eval.accounting_airline.lab_product import reviewed_view_documents
    from ssgrep.accounting.evidence_tokens import Token
    from ssgrep.accounting.frequency import build_view

    view = build_view(
        reviewed_view_documents(
            ["MCT-DEN-045", "MCT-DEN-045"],
            common_tokens=["live_holds", "raw_free", "sellable_free", "MCT-DEN-045"],
        )
    )
    assert view.doc_count == 1000
    reviewed = view.rarity_evidence(Token("identifier", "MCT-DEN-045"))
    assert reviewed.value is not None
    assert reviewed.upper_df == 1 and reviewed.value >= 8000
    for token in ("live_holds", "raw_free", "sellable_free"):
        common = view.rarity_evidence(Token("identifier", token))
        assert common.kind == "exact" and common.upper_df == 1000 and common.value == 0
    assert view.rarity_evidence(Token("identifier", "unobserved_identifier")).value is None


def test_fixture_vocabulary_comes_only_from_independent_observed_visible_cards() -> None:
    from eval.accounting_airline import accounting_oracle as oracle
    from tests.accounting.airline.oracle_fixtures import (
        TOKEN,
        consumer_b,
        frozen,
        search_call,
        search_json,
        source_a,
    )

    visible = search_json(f"{TOKEN} live_holds", 200)
    world = frozen(
        [source_a(witness_tail="hidden_native_identifier"), consumer_b(visible=visible)],
        [search_call(stdout=visible)],
    )
    vocabulary = oracle.candidate_vocabulary(world)
    assert ("identifier", TOKEN) in vocabulary and ("identifier", "live_holds") in vocabulary
    assert ("identifier", "hidden_native_identifier") not in vocabulary
    assert vocabulary == tuple(dict.fromkeys(vocabulary))


def test_fixture_input_and_method_digests_are_frozen_before_product_execution(
    tmp_path, monkeypatch
) -> None:
    from eval.accounting_airline import lab_product as lp
    from eval.accounting_airline._arithmetic_collect import fixture_methodology
    from ssgrep.accounting.frequency import build_view

    def boundary(_argv, **kwargs):
        request = json.loads(kwargs["input"])
        assert request["tokens"] == ["MCT-DEN-045"]
        assert request["common_tokens"] == ["live_holds"]
        assert len(request["fixture_inputs"]["input_digest"]) == 64
        view = build_view(
            lp.reviewed_view_documents(request["tokens"], common_tokens=request["common_tokens"])
        )
        assert request["fixture_inputs"]["frequency_view_id"] == view.version_id
        assert request["methodology"]["hash"] == fixture_methodology(view).hash
        raise RuntimeError("collection boundary observed")

    monkeypatch.setattr(lp, "run_bounded", boundary)
    product = lp.Product({"SSGREP_DATA_DIR": str(tmp_path / "derived")}, tmp_path)
    with pytest.raises(RuntimeError, match="collection boundary observed"):
        lp.controlled_arithmetic(
            product,
            ["MCT-DEN-045"],
            common_tokens=["live_holds"],
            source_data=tmp_path,
        )


def test_controlled_arithmetic_injects_fixture_into_real_collection_boundary(
    tmp_path, monkeypatch
) -> None:
    from eval.accounting_airline._arithmetic_collect import fixture_frequency_resolver
    from eval.accounting_airline.lab_product import reviewed_view_documents
    from ssgrep.accounting import collector_frequency
    from ssgrep.accounting.frequency import FrequencyView, build_view

    monkeypatch.setenv("SSGREP_DATA_DIR", str(tmp_path))

    fixture = build_view(reviewed_view_documents(["MCT-DEN-071"]))
    written = []
    monkeypatch.setattr(
        collector_frequency, "pin", lambda profile, view, hook: written.append(view)
    )
    resolved = fixture_frequency_resolver(collector_frequency.resolve, fixture)(
        "personal-local",
        pinned=None,
        last_documents=None,
        evidence=lambda: (),
        scan=False,
        repin=False,
        cached=FrequencyView.unknown("frequency_index_absent"),
    )
    assert resolved.view == fixture
    assert resolved.state == "pinned" and written == [fixture]


def test_controlled_arithmetic_publishes_real_isolated_checkpoint(tmp_path: Path) -> None:
    from eval.accounting_airline import lab_product as lp
    from eval.accounting_airline.roots import LabRoots, runtime_env

    roots = LabRoots.create(tmp_path)
    try:
        original = lp.Product(runtime_env(roots, "control"), roots.project)
        lp.setup_profile(original)
        profile = roots.ssgrep_data / "accounting" / "personal-local"
        (profile / "journal").mkdir(mode=0o700, parents=True, exist_ok=True)
        key = profile / "fingerprint.key"
        if not key.exists():
            key.write_bytes(b"k" * 32)
            key.chmod(0o600)
        policy = roots.ssgrep_data / "accounting" / "policy.json"
        original_policy, original_key = policy.read_bytes(), key.read_bytes()
        destination = roots.artifacts / "arithmetic"
        lp.copy_accounting(roots.ssgrep_data, destination, journal_only=True)
        fixture = original.with_env(SSGREP_DATA_DIR=str(destination))

        collected = lp.controlled_arithmetic(
            fixture,
            ["MCT-DEN-071"],
            source_data=roots.ssgrep_data,
            common_tokens=["live_holds"],
        )

        checkpoint = json.loads(
            (destination / "accounting/personal-local/collector/checkpoint.json").read_bytes()
        )
        actual = collected.savings_document["data"]["provenance"]["methodology_id"]
        declared = collected.savings_document["lab_valuation_methodology"]
        assert checkpoint["methodology"] == actual == declared["hash"]
        assert declared["label"].startswith("airline-lab-arithmetic-v1:")
        assert collected.savings_document["lab_valuation_profile"] == "airline-lab-arithmetic-v1"
        inputs = collected.savings_document["lab_fixture_inputs"]
        pointer = json.loads(
            (destination / "accounting/personal-local/collector/frequency.json").read_bytes()
        )
        assert inputs["eligibility_model"] == "observed_identifier_whitelist_v1"
        assert inputs["reviewed_count"] == inputs["observed_common_count"] == 1
        assert inputs["frequency_view_id"] == pointer["view_id"]
        assert len(inputs["input_digest"]) == 64
        assert not (profile / "collector").exists()
        assert policy.read_bytes() == original_policy and key.read_bytes() == original_key
        assert not (destination / "lancedb").exists()  # Missing locators are not manufactured.
    finally:
        roots.remove()


def test_original_native_store_has_narrow_collector_authority_and_b_denies_a(
    tmp_path: Path,
) -> None:
    from eval.accounting_airline.roots import LabRoots, runtime_env
    from eval.accounting_airline.sandbox import profile_for_solver_b

    roots = LabRoots.create(tmp_path)
    try:
        native_a, native_b = roots.native_sessions / "a", roots.native_sessions / "b"
        assert native_a.is_dir() and native_b.is_dir()
        control = runtime_env(roots, "control", runtime_sessions=roots.native_sessions)
        solver_b = runtime_env(roots, "solver-b", runtime_sessions=native_b)
        assert Path(control["SSGREP_OMP_SESSIONS_DIR"]) == native_a.parent == native_b.parent
        assert Path(solver_b["SSGREP_OMP_SESSIONS_DIR"]) == native_b
        assert solver_b["HOME"] != str(native_b)
        profile = profile_for_solver_b(roots, real_home=tmp_path / "no-real-home")
        assert native_a in profile.denied
        assert not any(native_b.is_relative_to(path) for path in profile.denied)
        assert not roots.native_sessions.is_relative_to(roots.artifacts)
    finally:
        roots.remove()


def _live_report(tmp_path: Path) -> dict:
    return json.loads(
        next(tmp_path.glob("ssgrep-airline-run-*/artifacts/live-lane.json")).read_text()
    )


def test_live_without_authorization_is_not_run_and_names_the_missing_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = entry.main(["--mode", "live", "--runtime", "omp", "--run-root-parent", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 3 and "--authorize-live" in out and "[PASS" not in out.split("live.two")[-1]
    verdict = _live_report(tmp_path)["gates"][-1]
    assert verdict["name"] == "live.two_session_lab" and verdict["status"] == NOT_RUN
    assert _live_report(tmp_path)["authorized"] is False


def test_missing_live_authorization_never_reaches_any_external_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import subprocess

    def refused(*_args, **_kwargs):
        pytest.fail("unauthorized live request reached an external process")

    monkeypatch.setattr(subprocess, "run", refused)
    monkeypatch.setattr(subprocess, "Popen", refused)
    code = entry.main(["--mode", "live", "--runtime", "omp", "--run-root-parent", str(tmp_path)])
    assert code == 3
    assert "authorization" in capsys.readouterr().out
    verdict = _live_report(tmp_path)["gates"][-1]
    assert verdict["status"] == NOT_RUN
    assert verdict["status"] != BLOCKED  # Authorization refusal, not a failed live attempt.


def test_standin_mode_is_listed_and_live_requires_runtime_gate(tmp_path: Path) -> None:
    code = entry.main(["--mode", "live", "--run-root-parent", str(tmp_path)])
    assert code == 3
    assert _live_report(tmp_path)["gates"][-1]["status"] == NOT_RUN


@pytest.mark.parametrize(
    ("failure", "gate_name", "detail"),
    [
        ("docker", "database.prerequisites", "docker CLI not found on PATH"),
        ("daemon", "database.prerequisites", "docker daemon unreachable: socket gone"),
        (
            "image",
            "database.prerequisites",
            f"image {dataset.IMAGE_REF} is not present locally (the lab never pulls "
            "or substitutes another image)",
        ),
        ("psql", "psql_client", "host psql client not found"),
    ],
)
def test_standin_missing_prerequisite_is_blocked_and_all_checkpoints_are_not_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: str,
    gate_name: str,
    detail: str,
) -> None:
    def which(binary):
        return None if binary == failure else f"/usr/bin/{binary}"

    def docker(*args, **_kwargs):
        if args == ("version", "--format", "{{.Server.Version}}"):
            assert failure != "docker", "missing CLI must not invoke Docker"
            return subprocess.CompletedProcess(
                ["docker", *args], 1 if failure == "daemon" else 0, "27.0.0", "socket gone"
            )
        assert args == ("image", "inspect", dataset.IMAGE_REF, "--format", "{{.Id}}")
        return subprocess.CompletedProcess(
            ["docker", *args], 1 if failure == "image" else 0, dataset.IMAGE_DIGEST, ""
        )

    def refused(*_args, **_kwargs):
        pytest.fail("blocked stand-in must not start an external process")

    monkeypatch.setattr(database.shutil, "which", which)
    monkeypatch.setattr(database, "_docker", docker)
    monkeypatch.setattr(subprocess, "run", refused)
    monkeypatch.setattr(subprocess, "Popen", refused)
    code = entry.main(["--mode", "standin", "--run-root-parent", str(tmp_path)])
    out = capsys.readouterr().out
    report = json.loads(next(tmp_path.glob("*-report/report.json")).read_text())
    assert code == 3 and f"[BLOCKED ] {gate_name}: {detail}" in out
    assert report["counts_toward_live_gate"] is False
    assert report["gates"]["live_session_gate"]["status"] == NOT_RUN
    assert report["gates"]["cleanup"]["status"] == BLOCKED
    assert report["gates"]["cleanup"]["entries"] == [
        {
            "name": "prerequisite",
            "status": BLOCKED,
            "detail": f"{gate_name}: {detail}",
            "artifact": None,
        }
    ]
    assert set(report["checkpoints"]) == {f"C{i}" for i in range(11)}
    assert all(
        gate["status"] == NOT_RUN for name, gate in report["gates"].items() if name != "cleanup"
    )
    for checkpoint in report["checkpoints"].values():
        assert checkpoint["domain_status"] == NOT_RUN
        assert checkpoint["accounting_status"] == NOT_RUN
        assert checkpoint["status"] == NOT_RUN
        assert "domain" not in checkpoint and "accounting" not in checkpoint
    assert not list(tmp_path.glob("ssgrep-airline-lab-*[!t]/"))


@pytest.mark.postgres
def test_standin_pipeline_runs_c0_to_c10_with_real_binaries(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    standin_ready: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hashlib

    from eval.accounting_airline.lab_context import Run

    original_arithmetic = Run.collect_arithmetic
    checked_indexes = []

    def index_inventory(index):
        inventory = {}
        for path in sorted(index.rglob("*")):
            if path.is_file():
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                info = path.stat()
                inventory[str(path.relative_to(index))] = (info.st_size, info.st_mtime_ns, digest)
        return inventory

    def checked_arithmetic(run, frozen):
        index = run.roots.ssgrep_data / "lancedb"
        before = index_inventory(index)
        assert before, "the arithmetic copy must retain a genuine existing locator index"
        collected, tokens = original_arithmetic(run, frozen)
        assert index_inventory(index) == before, "controlled arithmetic mutated its original index"
        assert collected.savings_document["lab_valuation_methodology"]["label"].startswith(
            "airline-lab-arithmetic-v1:"
        )
        checked_indexes.append(True)
        return collected, tokens

    monkeypatch.setattr(Run, "collect_arithmetic", checked_arithmetic)
    code = entry.main(["--mode", "standin", "--run-root-parent", str(tmp_path)])
    out = capsys.readouterr().out
    report = json.loads(next(tmp_path.glob("*-report/report.json")).read_text())
    assert report["runtime"] == "scripted-standin" and report["counts_toward_live_gate"] is False
    assert report["gates"]["live_session_gate"]["status"] == NOT_RUN
    assert "scripted-standin" in out
    scrape_status = report["gates"]["scrape"]["status"]
    assert scrape_status in (PASS, BLOCKED)
    for cid in (f"C{i}" for i in range(11)):
        assert report["checkpoints"][cid]["domain_status"] == PASS, report["checkpoints"][cid]
        expected = BLOCKED if cid == "C9" and scrape_status == BLOCKED else PASS
        assert report["checkpoints"][cid]["accounting_status"] == expected, report["checkpoints"][
            cid
        ]
    assert checked_indexes, "the real C6 arithmetic collector was not exercised"
    frozen_fixture = report["checkpoints"]["C6"]["actual_delta"]["arithmetic_fixture"]
    assert "observed_identifier_whitelist_v1" in frozen_fixture["methodology"]["label"]
    assert len(frozen_fixture["methodology"]["hash"]) == 32
    assert len(frozen_fixture["frozen_inputs"]["input_digest"]) == 64
    assert frozen_fixture["frozen_inputs"]["reviewed_count"] > 0
    assert frozen_fixture["frozen_inputs"]["observed_common_count"] > 0
    source_proofs = [
        item
        for item in report["checkpoints"]["C3"]["accounting"]
        if item["name"].endswith(".production_native_source_authority")
    ]
    assert len(source_proofs) == 2
    for item in source_proofs:
        assert item["status"] == PASS
        proof = item["actual"]
        assert proof["status"] == "ok" and proof["session"] == ["omp", "sess-a"]
        assert "/native-sessions/a/" in proof["path"]
        assert proof["source_digest"] == item["expected"]["source_digest"]
        assert proof["record_digest_count"] > 0 and proof["native_identities"]
        assert proof["outside_root_status"] == "source_unapproved"
    c9 = report["checkpoints"]["C9"]
    assert c9["domain_status"] == PASS
    c9_domain = {item["name"]: item for item in c9["domain"]}
    assert c9_domain["c9.native_source_contents_unchanged"]["status"] == PASS
    assert (
        c9_domain["c9.native_source_contents_unchanged"]["expected"]
        == c9_domain["c9.native_source_contents_unchanged"]["actual"]
    )
    c9_transport = c9["actual_delta"]["c9_transport"]
    assert c9_transport["accepted_metric_points"] > 0
    assert c9_transport["partial_rejected_points"] > 0
    assert c9_transport["partial_snapshot_was_not_acknowledged"] is True
    assert c9_transport["privacy_canary_absent"] is True
    assert c9_transport["metrics_ack_generation"] == c9_transport["snapshot_generation"]
    assert c9_transport["metrics_ack_snapshot_id"] == c9_transport["snapshot_id"]
    assert report["gates"]["export_privacy"]["status"] == PASS
    if report["gates"]["scrape"]["status"] == BLOCKED:
        assert c9["accounting_status"] == BLOCKED
        assert report["gates"]["scrape"]["detail"]
    else:
        assert report["gates"]["scrape"]["status"] == PASS
        assert c9["accounting_status"] == PASS
        samples = c9_transport["scrape_samples"]
        assert len(samples) == c9_transport["sample_count"]
        assert any(
            row["name"] == "ssgrep_executions_total"
            and row["labels"].get("operation") == "search"
            and int(row["value"]) > 0
            for row in samples
        )
        assert c9_transport["sample_count"] > 0
    assert code == (3 if scrape_status == BLOCKED else 0), out
    for group in ("valuation_production", "valuation_arithmetic"):
        assert report["gates"][group]["status"] == PASS
    text = json.dumps(report)
    assert "PGPASSWORD" not in text
    assert not list(tmp_path.glob("ssgrep-airline-lab-*[!t]/"))  # lab root torn down
