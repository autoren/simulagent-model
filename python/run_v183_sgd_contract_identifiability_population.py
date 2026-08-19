#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v183_sgd_contract_identifiability_population import audit_population, build_population


DEPENDENCY_KEYS = (
    "config", "parent_V182_outcome", "source_V134_outcome", "source_V134_analysis_lock", "source_archive",
    "source_catalog", "source_population", "roadmap", "plan", "protocol",
    "tests", "runner", "verifier", "auditor", "design_audit",
)


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def reconstruct(lock: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    population = build_population(
        (PROJECT_ROOT / lock["source_archive"]).read_bytes(),
        json.loads((PROJECT_ROOT / lock["source_catalog"]).read_text()),
        json.loads((PROJECT_ROOT / lock["source_population"]).read_text()),
        lock["config_payload"],
    )
    return population, audit_population(population, lock["config_payload"])


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v183-sgd-contract-identifiability-population-lock.json"
    output_root = PROJECT_ROOT / "outputs/v183-sgd-contract-identifiability-population/population"
    if output_root.exists():
        raise RuntimeError("V183 formal population may be built only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]:
        raise RuntimeError("V183 lock mismatch")
    for key in DEPENDENCY_KEYS:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V183 dependency drifted: {key}")

    population, audit = reconstruct(lock)
    config = lock["config_payload"]
    decision = (
        config["decisionRule"]["ifEveryPopulationAndIdentifiabilityGatePasses"]
        if audit["passed"]
        else config["decisionRule"]["otherwise"]
    )
    paths = {
        "contract_catalog": output_root / "contract-catalog.json",
        "hidden_identifiability": output_root / "hidden-identifiability.json",
        "development_identities": output_root / "development-identities.json",
        "protected_identities": output_root / "protected-identities.json",
        "population_summary": output_root / "population-summary.json",
    }
    payloads = {
        "contract_catalog": population["contract_catalog"],
        "hidden_identifiability": {"records": population["hidden_records"], "contains_language_or_values": False},
        "development_identities": {"records": population["public_development"], "contains_language_or_hidden_labels": False},
        "protected_identities": {"records": population["public_protected"], "contains_language_or_hidden_labels": False},
        "population_summary": population["summary"],
    }
    for key, path in paths.items():
        write_json(path, payloads[key])
    output_integrity = {
        key: {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(path)}
        for key, path in paths.items()
    }
    access = {
        "formal_population_build_count": 1,
        "source_archive_structured_parse_count": 1,
        "selected_structured_frame_count": population["summary"]["source_record_count"],
        "utterance_or_dialogue_text_emission_count": 0,
        "slot_value_or_span_emission_count": 0,
        "manual_language_inspection_count": 0,
        "deterministic_policy_score_count": 0,
        "model_load_count": 0,
        "model_generation_count": 0,
        "API_call_count": 0,
        "training_run_count": 0,
        "ontology_registration_count": 0,
        "trusted_state_mutation_count": 0,
        "protected_language_read_count": 0,
        "service_or_sensor_call_count": 0,
        "external_side_effect_count": 0,
        "actual_execution_count": 0,
    }
    result = {
        "schema_version": "183-sgd-contract-identifiability-population-result",
        "experiment": config["experiment"],
        "passed": audit["passed"],
        "decision": decision,
        "summary": population["summary"],
        "population_gates": audit["checks"],
        "access": access,
        "output_integrity": output_integrity,
        "claim_boundary": config["claimBoundary"],
    }
    write_json(output_root / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
