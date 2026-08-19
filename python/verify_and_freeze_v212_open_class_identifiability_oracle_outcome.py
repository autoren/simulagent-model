#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from run_v212_open_class_identifiability_oracle import dependency_hashes_exact
from v10_protocol import file_sha256
from v212_open_class_identifiability_oracle import (
    audit_metrics,
    build_predictions,
    materialize_cases,
    materialize_public_semantics,
    score_oracle,
)
from v22r2_grounding import PROJECT_ROOT


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v212-open-class-identifiability-oracle-lock.json"
    lock = json.loads(lock_path.read_text())
    audit_path = PROJECT_ROOT / "outputs/v212-representational-diagnosis/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v212-open-class-identifiability-oracle-outcome-lock.json"
    results_path = PROJECT_ROOT / "docs/v212-open-class-identifiability-oracle-results.md"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V212 outcome is already audited or frozen")
    config = lock["config_payload"]
    artifacts = {key: PROJECT_ROOT / value for key, value in config["artifacts"].items()}
    semantics = materialize_public_semantics(config)
    public_records, truth_records = materialize_cases(config, semantics)
    predictions = build_predictions(public_records, semantics)
    stored_semantics = json.loads(artifacts["publicSemantics"].read_text())
    stored_public = read_jsonl(artifacts["publicCases"])
    stored_truth = read_jsonl(artifacts["sealedTruth"])
    stored_predictions = read_jsonl(artifacts["predictions"])
    freeze = json.loads(artifacts["predictionFreeze"].read_text())
    stored_summary = json.loads(artifacts["summary"].read_text())
    stored_result = json.loads(artifacts["result"].read_text())
    access = stored_summary["access"]
    metrics = score_oracle(public_records, truth_records, predictions, semantics, config)
    rebuilt_audit = audit_metrics(metrics, freeze, access, config)
    checks = {
        "design_lock_and_dependencies_exact": dependency_hashes_exact(lock),
        "public_semantics_and_cases_reconstruct_exactly": stored_semantics == semantics and stored_public == public_records,
        "sealed_truth_reconstructs_exactly": stored_truth == truth_records,
        "public_only_predictions_reconstruct_exactly": stored_predictions == predictions,
        "prediction_freeze_hashes_exact_and_precedes_truth_join": bool(
            freeze["design_lock_sha256"] == file_sha256(lock_path)
            and freeze["public_semantics_sha256"] == file_sha256(artifacts["publicSemantics"])
            and freeze["public_cases_sha256"] == file_sha256(artifacts["publicCases"])
            and freeze["predictions_sha256"] == file_sha256(artifacts["predictions"])
            and freeze["predictions_frozen_before_truth_join"]
            and not freeze["truth_join_opened_before_freeze"]
            and freeze["oracle_worker_truth_path_count"] == 0
            and freeze["oracle_worker_hidden_field_count"] == 0
        ),
        "summary_metrics_access_and_audit_reconstruct_exactly": bool(
            stored_summary["metrics"] == metrics
            and stored_summary["audit"] == rebuilt_audit
            and stored_summary["claim_boundary"] == config["claimBoundary"]
        ),
        "result_reconstructs_and_scientific_audit_passes": bool(
            stored_result["passed"] == rebuilt_audit["passed"]
            and stored_result["branch"] == rebuilt_audit["branch"]
            and stored_result["decision"] == rebuilt_audit["decision"]
            and rebuilt_audit["passed"]
        ),
        "results_document_exists": results_path.is_file(),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "212-representational-diagnosis-oracle-outcome-audit",
        "experiment": lock["experiment"],
        "passed": passed,
        "branch": rebuilt_audit["branch"],
        "decision": "freeze_verified_V212" if passed else "freeze_failed_V212_verification",
        "checks": checks,
        "metrics": metrics,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "design_lock": lock_path,
        "audit": audit_path,
        "public_semantics": artifacts["publicSemantics"],
        "public_cases": artifacts["publicCases"],
        "sealed_truth": artifacts["sealedTruth"],
        "predictions": artifacts["predictions"],
        "prediction_freeze": artifacts["predictionFreeze"],
        "summary": artifacts["summary"],
        "result": artifacts["result"],
        "results_document": results_path,
        "verifier": PROJECT_ROOT / lock["verifier"],
    }
    outcome: dict[str, Any] = {
        "schema_version": "212-representational-diagnosis-oracle-outcome-lock",
        "experiment": lock["experiment"],
        "outcome": {
            "passed": True,
            "branch": rebuilt_audit["branch"],
            "decision": rebuilt_audit["decision"],
            "metrics": metrics,
        },
        "authorization": {
            "design_V213_fresh_programmatic_concept_population": rebuilt_audit["branch"] == "V213_DESIGN_ELIGIBLE",
            "generate_V213_population_without_separate_lock": False,
            "design_metadata_first_external_resource_census": rebuilt_audit["branch"] == "V213_DESIGN_ELIGIBLE",
            "read_external_ontology_payload_without_separate_lock": False,
            "run_local_or_API_model_or_training": False,
            "register_mutate_call_act_or_execute": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"outcome_lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
