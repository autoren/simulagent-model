#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from v10_protocol import file_sha256
from v218_mondo_artifact_population import load_obo
from v221_deterministic_mondo_residual import (
    audit_evaluation,
    build_catalog,
    score_record,
    summarize_observations,
)
from v22r2_grounding import PROJECT_ROOT


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value, sort_keys=True) + "\n" for value in values))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def dependency_hashes_exact(lock: dict[str, Any]) -> bool:
    keys = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    return valid_lock(lock) and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in keys)


def evaluate(lock: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    config = lock["config_payload"]
    inputs = config["inputContract"]
    role_manifest = json.loads((PROJECT_ROOT / config["artifacts"]["roleManifest"]).read_text())
    older = load_obo(PROJECT_ROOT / inputs["olderOBO"])
    newer = load_obo(PROJECT_ROOT / inputs["newerOBO"])
    public_records = read_jsonl(PROJECT_ROOT / inputs["developmentPublic"])
    truth_records = read_jsonl(PROJECT_ROOT / inputs["developmentTruth"])
    truth_by_id = {record["case_id"]: record for record in truth_records}
    if len(truth_by_id) != len(truth_records) or {record["case_id"] for record in public_records} != set(truth_by_id):
        raise RuntimeError("V220 development public/truth alignment mismatch")
    roles = {group_id: "CALIBRATION" for group_id in role_manifest["calibration_group_ids"]}
    roles.update({group_id: "EVALUATION" for group_id in role_manifest["evaluation_group_ids"]})
    if {record["group_id"] for record in public_records} != set(roles):
        raise RuntimeError("V221 role manifest does not account for development records")
    catalog = build_catalog(older, newer, config)
    truth_class_ids = {
        class_id for truth in truth_records for class_id in truth["equivalence_classes"]
    }
    catalog["manifest"]["state_class_accuracy"] = (
        len(truth_class_ids & catalog["valid_classes"]) / len(truth_class_ids) if truth_class_ids else 1.0
    )
    catalog["manifest"]["development_truth_state_class_count"] = len(truth_class_ids)
    observations: list[dict[str, Any]] = []
    for public in public_records:
        truth = truth_by_id[public["case_id"]]
        for method_id in config["metrics"]["requiredByMethod"]:
            for budget in config["candidateBudgets"]:
                observations.append(score_record(public, truth, roles[public["group_id"]], catalog, method_id, budget))
    metrics, residual = summarize_observations(observations, role_manifest, config)
    return catalog["manifest"], observations, metrics, residual


def access_ledger() -> dict[str, int]:
    return {
        "catalog_build_count": 1, "development_evaluation_run_count": 1,
        "development_public_load_count": 1, "development_truth_load_count": 1,
        "protected_public_load_count": 0, "protected_truth_load_count": 0,
        "v218_development_record_read_count": 0, "v218_protected_record_read_count": 0,
        "v216_protected_access_count": 0, "v213_protected_access_count": 0,
        "model_load_count": 0, "model_generation_count": 0, "model_api_call_count": 0,
        "training_run_count": 0, "network_request_count": 0, "ontology_registration_count": 0,
        "trusted_state_mutation_count": 0, "service_action_count": 0,
        "external_side_effect_count": 0, "actual_execution_count": 0,
    }


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v221-deterministic-mondo-residual-lock.json"
    lock = json.loads(lock_path.read_text())
    if not dependency_hashes_exact(lock):
        raise RuntimeError("V221 design lock or dependency hash mismatch")
    if not lock["authorization"]["run_one_development_only_deterministic_evaluation"]:
        raise RuntimeError("V221 evaluation is not authorized")
    config = lock["config_payload"]
    artifacts = {key: PROJECT_ROOT / value for key, value in config["artifacts"].items()}
    for key in ("catalogManifest", "observations", "residualManifest", "summary", "result"):
        if artifacts[key].exists():
            raise RuntimeError(f"V221 output already exists: {key}")
    # Protected paths are hash-checked only; their JSONL bodies are never loaded.
    inputs = config["inputContract"]
    if file_sha256(PROJECT_ROOT / inputs["protectedPublic"]) != inputs["protectedPublicSha256"] or file_sha256(PROJECT_ROOT / inputs["protectedTruth"]) != inputs["protectedTruthSha256"]:
        raise RuntimeError("sealed protected hash mismatch")
    catalog_manifest, observations, metrics, residual = evaluate(lock)
    access = access_ledger()
    audit = audit_evaluation(metrics, catalog_manifest, access, config)
    write_json(artifacts["catalogManifest"], catalog_manifest)
    write_jsonl(artifacts["observations"], observations)
    write_json(artifacts["residualManifest"], residual)
    summary = {
        "schema_version": "221-deterministic-mondo-residual-summary",
        "experiment": config["experiment"], "claim_boundary": config["claimBoundary"],
        "role_manifest_sha256": file_sha256(artifacts["roleManifest"]),
        "catalog_manifest_sha256": file_sha256(artifacts["catalogManifest"]),
        "observations_sha256": file_sha256(artifacts["observations"]),
        "residual_manifest_sha256": file_sha256(artifacts["residualManifest"]),
        "metrics": metrics, "access": access, "audit": audit,
    }
    result = {
        "schema_version": "221-deterministic-mondo-residual-result",
        "experiment": config["experiment"], "passed": audit["passed"],
        "branch": audit["branch"], "decision": audit["decision"],
        "claim_boundary": config["claimBoundary"],
        "residual_evaluation_group_count": metrics["residual_evaluation_group_count"],
        "authorization": {
            "design_one_local_model_candidate_study": bool(audit["passed"] and metrics["model_eligible_residual"]),
            "run_model_or_open_protected": False,
            "register_mutate_service_act_execute": False,
        },
    }
    write_json(artifacts["summary"], summary)
    write_json(artifacts["result"], result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({key: value for key, value in metrics.items() if key not in {"cells", "primary_evaluation_strata", "residual_evaluation_group_ids"}}, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
