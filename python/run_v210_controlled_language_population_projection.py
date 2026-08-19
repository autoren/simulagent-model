#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from v10_protocol import file_sha256
from v210_controlled_language_population_projection import (
    audit_evaluation,
    canonical_jsonl,
    evaluate_population,
    generate_population,
    project_development_surfaces,
)
from v22r2_grounding import PROJECT_ROOT


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_jsonl(records))


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v210-controlled-language-population-projection-lock.json"
    lock = json.loads(lock_path.read_text())
    if not valid_lock(lock):
        raise RuntimeError("invalid V210 design lock")
    dependency_keys = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    for key in dependency_keys:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V210 locked dependency changed: {key}")

    config = lock["config_payload"]
    artifact_paths = {key: PROJECT_ROOT / value for key, value in config["artifacts"].items()}
    if any(path.exists() for path in artifact_paths.values()):
        raise RuntimeError("V210 population or evaluation artifact already exists")
    parent_outcome = json.loads((PROJECT_ROOT / lock["parent_V209r1_outcome"]).read_text())
    repair_lock = json.loads((PROJECT_ROOT / parent_outcome["repair_lock"]).read_text())
    v209_lock = json.loads((PROJECT_ROOT / repair_lock["parent_V209_design_lock"]).read_text())
    parent_config = v209_lock["config_payload"]

    population = generate_population(config, parent_config)
    predictions = project_development_surfaces(population["DEVELOPMENT"]["surfaces"], config)
    summary = evaluate_population(population, predictions, config, parent_config)
    audit = audit_evaluation(summary, config)
    scientific_pass = audit["population_projection_gates_passed"]
    decision = config["decisionRule"][
        "ifEveryIntegrityPopulationProjectionAndAccessGatePasses" if scientific_pass else "otherwise"
    ]
    result = {
        "schema_version": "210-controlled-language-population-projection-result",
        "experiment": config["experiment"],
        "passed": audit["access_gates_passed"],
        "population_projection_gates_passed": scientific_pass,
        "decision": decision,
        "claim_boundary": config["claimBoundary"],
        "checks": audit["checks"],
        "role_checks": audit["role_checks"],
        "access_checks": audit["access_checks"],
        "summary": summary,
        "authorization": {
            "preregister_separate_deterministic_development_baseline_design_only": bool(audit["access_gates_passed"] and scientific_pass),
            "open_protected_or_run_model": False,
            "API_training_registration_authority_action_or_execution": False,
        },
    }

    write_jsonl(artifact_paths["developmentSurface"], population["DEVELOPMENT"]["surfaces"])
    write_jsonl(artifact_paths["developmentTruth"], population["DEVELOPMENT"]["truth"])
    write_jsonl(artifact_paths["protectedSurface"], population["PROTECTED"]["surfaces"])
    write_jsonl(artifact_paths["protectedTruth"], population["PROTECTED"]["truth"])
    write_jsonl(artifact_paths["developmentProjection"], predictions)
    write_json(artifact_paths["summary"], summary)
    write_json(artifact_paths["result"], result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not audit["access_gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
