#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v100_massive_source import parse_massive_archive
from v106_open_world_benchmark import (
    build_declared_training_records, character_retrieval_observations, fit_character_retrieval,
)
from run_v112_open_world_full_policy_transfer import decision_for, payload_hash, read_jsonl
from v112r1_full_policy_aggregation import evaluate_preserved_outputs


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_fixtures(lock: dict[str, Any]) -> dict[str, dict[str, Any]]:
    manifest = json.loads((PROJECT_ROOT / lock["fixture_manifest"]).read_text())
    fixtures = {}
    for entry in manifest["fixtures"]:
        path = PROJECT_ROOT / entry["path"]
        if file_sha256(path) != entry["sha256"]:
            raise RuntimeError("V112 preserved fixture drifted")
        row = json.loads(path.read_text())
        if row["name"] in fixtures:
            raise RuntimeError("duplicate preserved fixture")
        fixtures[row["name"]] = row
    if len(fixtures) != manifest["fixture_count"] or manifest["fixture_count"] != 240:
        raise RuntimeError("V112 fixture manifest count mismatch")
    return fixtures


def reconstruct(lock: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    records = read_jsonl(PROJECT_ROOT / lock["fresh_language"])
    archive_bytes = (PROJECT_ROOT / lock["source_archive"]).read_bytes()
    source_records, member = parse_massive_archive(
        archive_bytes, lock["V112_config_payload"]["extraction"]["expectedLocaleMemberSuffix"],
    )
    catalog = json.loads((PROJECT_ROOT / lock["visible_catalog"]).read_text())
    training = build_declared_training_records(source_records, catalog)
    retrieval_spec = lock["baseline_config_payload"]["deterministicBaselines"]["character_ngram_retrieval"]
    fitted = fit_character_retrieval(training, retrieval_spec["vectorizer"])
    retrieval = character_retrieval_observations(fitted, records)
    fixtures = load_fixtures(lock)
    summary = evaluate_preserved_outputs(
        records, fixtures, retrieval, lock["preserved_access"],
        lock["V112_config_payload"], lock["baseline_config_payload"],
    )
    return summary, {"record_count": len(records), "fixture_count": len(fixtures), "source_locale_member": member}


def recovery_decision(summary: dict[str, Any]) -> tuple[bool, bool, bool, str]:
    quality = all(summary["quality_gates"].values())
    novel_names = (
        "novel_evidence_precision", "novel_evidence_recall",
        "novel_evidence_non_novel_false_positive_rate", "novel_evidence_ECE",
    )
    novel = all(summary["quality_gates"][key] for key in novel_names)
    access = all(summary["access_gates"].values())
    return quality, novel, access, decision_for(quality, novel, access)


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v112r1-full-policy-aggregation-recovery-lock.json"
    output_root = PROJECT_ROOT / "outputs/v112r1-full-policy-aggregation-recovery/recovered-evaluation"
    if output_root.exists():
        raise RuntimeError("V112r1 recovery may run only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]:
        raise RuntimeError("V112r1 lock mismatch")
    dependency_keys = (
        "config", "parent_lock", "parent_failure", "fresh_language", "fixture_manifest",
        "source_archive", "visible_catalog", "fresh_population", "plan", "protocol", "tests",
        "runner", "verifier", "auditor", "design_audit",
    )
    for key in dependency_keys:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V112r1 dependency drifted: {key}")
    summary, reconstruction = reconstruct(lock)
    quality, novel, access, decision = recovery_decision(summary)
    result = {
        "schema_version": "112r1-full-policy-aggregation-recovery-result",
        "experiment": lock["config_payload"]["experiment"],
        "passed": access, "quality_gate_pass": quality, "novel_evidence_pass": novel,
        "decision": decision, "summary": summary, "reconstruction": reconstruction,
        "access": {
            "preserved_fixture_automatic_read_count": 240,
            "fresh_development_language_read_count": 1, "source_archive_read_count": 1,
            "protected_test_language_read_count": 0,
            "manual_language_or_raw_response_inspection_count": 0,
            "new_model_load_count": 0, "new_model_generation_count": 0, "LLM_API_call_count": 0,
            "adapter_training_run_count": 0, "real_service_call_count": 0,
            "external_side_effect_count": 0,
        },
        "claim_boundary": "mechanical aggregation-only recovery over 240 frozen V112 fixtures; no changed language, model output, policy, calibration, metric, gate, or decision; no protected test, new model, API, training, execution, service call, or side effect",
    }
    result_path = output_root / "result.json"
    write_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
