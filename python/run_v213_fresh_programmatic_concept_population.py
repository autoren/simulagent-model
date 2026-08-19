#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from v10_protocol import file_sha256
from v213_fresh_programmatic_concept_population import audit_population, score_population
from v22r2_grounding import PROJECT_ROOT


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def dependency_hashes_exact(lock: dict[str, Any]) -> bool:
    keys = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    return valid_lock(lock) and all(
        file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in keys
    )


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v213-fresh-programmatic-concept-population-lock.json"
    lock = json.loads(lock_path.read_text())
    if not dependency_hashes_exact(lock):
        raise RuntimeError("V213 design lock or dependency hash mismatch")
    if not lock["authorization"]["materialize_one_role_separated_population"]:
        raise RuntimeError("V213 population materialization is not authorized")
    config = lock["config_payload"]
    artifacts = {key: PROJECT_ROOT / value for key, value in config["artifacts"].items()}
    if any(path.exists() for path in artifacts.values()):
        raise RuntimeError("V213 formal population output already exists")
    config_path = PROJECT_ROOT / lock["config"]
    semantics_path = PROJECT_ROOT / lock["parent_public_semantics"]
    blueprint_worker = PROJECT_ROOT / lock["blueprint_worker"]
    projection_worker = PROJECT_ROOT / lock["public_projection_worker"]
    subprocess.run(
        [
            sys.executable,
            str(blueprint_worker),
            "--config",
            str(config_path),
            "--semantics",
            str(semantics_path),
            "--manifest",
            str(artifacts["generatorManifest"]),
            "--public-blueprints",
            str(artifacts["publicBlueprints"]),
            "--sealed-truth",
            str(artifacts["sealedTruth"]),
            "--split",
            str(artifacts["split"]),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(projection_worker),
            "--public-blueprints",
            str(artifacts["publicBlueprints"]),
            "--public-records",
            str(artifacts["publicRecords"]),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    parent_outcome_path = PROJECT_ROOT / lock["parent_V212_outcome"]
    projection_freeze = {
        "schema_version": "213-public-projection-freeze",
        "design_lock_sha256": file_sha256(lock_path),
        "parent_outcome_lock_sha256": file_sha256(parent_outcome_path),
        "parent_public_semantics_sha256": file_sha256(semantics_path),
        "generator_manifest_sha256": file_sha256(artifacts["generatorManifest"]),
        "public_blueprints_sha256": file_sha256(artifacts["publicBlueprints"]),
        "sealed_truth_sha256": file_sha256(artifacts["sealedTruth"]),
        "split_sha256": file_sha256(artifacts["split"]),
        "public_records_sha256": file_sha256(artifacts["publicRecords"]),
        "public_projection_frozen_before_truth_join": True,
        "sealed_truth_joined_before_public_projection_freeze": False,
        "public_projection_worker_sealed_truth_path_count": projection_worker.read_text().count("sealed-truth"),
        "public_projection_worker_forbidden_token_count": sum(
            projection_worker.read_text().count(token)
            for token in config["roleSeparation"]["publicProjectionForbiddenTokens"]
        ),
    }
    write_json(artifacts["publicProjectionFreeze"], projection_freeze)

    semantics = json.loads(semantics_path.read_text())
    blueprints = read_jsonl(artifacts["publicBlueprints"])
    public_records = read_jsonl(artifacts["publicRecords"])
    truth_records = read_jsonl(artifacts["sealedTruth"])
    split = json.loads(artifacts["split"].read_text())
    parent_public_records = read_jsonl(PROJECT_ROOT / lock["parent_public_cases"])
    metrics = score_population(
        blueprints,
        public_records,
        truth_records,
        split,
        semantics,
        parent_public_records,
        projection_freeze,
        config,
    )
    access = {
        "blueprint_generation_run_count": 1,
        "public_projection_run_count": 1,
        "structural_verification_run_count": 1,
        "natural_language_surface_read_count": 0,
        "external_ontology_payload_read_count": 0,
        "protected_downstream_evaluation_count": 0,
        "model_load_count": 0,
        "model_generation_count": 0,
        "api_call_count": 0,
        "training_run_count": 0,
        "ontology_registration_count": 0,
        "trusted_state_mutation_count": 0,
        "service_call_count": 0,
        "external_side_effect_count": 0,
        "actual_execution_count": 0,
    }
    audit = audit_population(metrics, access, config)
    summary = {
        "schema_version": "213-fresh-programmatic-concept-population-summary",
        "experiment": config["experiment"],
        "claim_boundary": config["claimBoundary"],
        "metrics": metrics,
        "access": access,
        "audit": audit,
    }
    result = {
        "schema_version": "213-fresh-programmatic-concept-population-result",
        "experiment": config["experiment"],
        "passed": audit["passed"],
        "branch": audit["branch"],
        "decision": audit["decision"],
        "claim_boundary": config["claimBoundary"],
        "authorization": {
            "design_V214_deterministic_candidate_controls": audit["passed"],
            "run_protected_downstream_evaluation": False,
            "run_local_or_API_model_or_training": False,
            "read_external_payload_register_mutate_call_act_or_execute": False,
        },
    }
    write_json(artifacts["summary"], summary)
    write_json(artifacts["result"], result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
