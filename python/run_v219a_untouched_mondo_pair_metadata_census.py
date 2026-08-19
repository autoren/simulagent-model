#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from v10_protocol import file_sha256
from v219a_untouched_mondo_pair_metadata_census import audit_census, build_census
from v22r2_grounding import PROJECT_ROOT


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def dependency_hashes_exact(lock: dict[str, Any]) -> bool:
    keys = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    return valid_lock(lock) and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in keys)


def zero_boundary_access() -> dict[str, int]:
    return {
        "metadata_census_run_count": 1,
        "evidence_snapshot_read_count": 1,
        "network_request_count": 0,
        "new_payload_body_read_count": 0,
        "new_ontology_term_or_axiom_record_read_count": 0,
        "v218_development_record_read_count": 0,
        "v218_protected_record_read_count": 0,
        "v216_protected_access_count": 0,
        "v213_protected_access_count": 0,
        "deterministic_method_evaluation_count": 0,
        "model_load_count": 0,
        "model_generation_count": 0,
        "model_api_call_count": 0,
        "training_run_count": 0,
        "ontology_registration_count": 0,
        "trusted_state_mutation_count": 0,
        "service_action_count": 0,
        "external_side_effect_count": 0,
        "actual_execution_count": 0,
    }


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v219a-untouched-mondo-pair-metadata-census-lock.json"
    lock = json.loads(lock_path.read_text())
    if not dependency_hashes_exact(lock):
        raise RuntimeError("V219A design lock or dependency hash mismatch")
    if not lock["authorization"]["run_one_local_metadata_census"]:
        raise RuntimeError("V219A local metadata census is not authorized")
    config = lock["config_payload"]
    artifacts = {key: PROJECT_ROOT / value for key, value in config["artifacts"].items()}
    if any(path.exists() for path in artifacts.values()):
        raise RuntimeError("V219A census artifact already exists")
    snapshot_path = PROJECT_ROOT / config["evidenceSource"]["path"]
    snapshot_hash_accurate = file_sha256(snapshot_path) == config["evidenceSource"]["sha256"]
    releases = json.loads(snapshot_path.read_text())
    evidence, metrics = build_census(releases, config, snapshot_hash_accurate=snapshot_hash_accurate)
    access = zero_boundary_access()
    audit = audit_census(metrics, access, config)
    write_json(artifacts["evidence"], evidence)
    summary = {
        "schema_version": "219a-untouched-mondo-pair-metadata-census-summary",
        "experiment": config["experiment"],
        "claim_boundary": config["claimBoundary"],
        "evidence_snapshot_sha256": file_sha256(snapshot_path),
        "evidence_sha256": file_sha256(artifacts["evidence"]),
        "metrics": metrics,
        "access": access,
        "audit": audit,
    }
    result = {
        "schema_version": "219a-untouched-mondo-pair-metadata-census-result",
        "experiment": config["experiment"],
        "passed": audit["passed"],
        "branch": audit["branch"],
        "decision": audit["decision"],
        "claim_boundary": config["claimBoundary"],
        "selected_pair_ids": metrics["selected_pair_ids"],
        "authorization": {
            "design_one_untouched_pair_payload_protocol": audit["passed"],
            "retrieve_payload_or_evaluate_method": False,
            "open_protected_or_run_model": False,
            "register_mutate_service_act_execute": False,
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
