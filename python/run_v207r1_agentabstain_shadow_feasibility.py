#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from v10_protocol import file_sha256
from v207r1_agentabstain_shadow_feasibility import audit_feasibility, evaluate_feasibility
from v22r2_grounding import PROJECT_ROOT


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v207r1-agentabstain-shadow-feasibility-lock.json"
    lock = json.loads(lock_path.read_text())
    if not valid_lock(lock):
        raise RuntimeError("invalid V207r1 design lock")
    dependency_keys = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    for key in dependency_keys:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V207r1 locked dependency changed: {key}")
    output_root = PROJECT_ROOT / "outputs/v207r1-agentabstain-shadow-feasibility/evaluation"
    if output_root.exists():
        raise RuntimeError("V207r1 output already exists")

    summary = evaluate_feasibility(lock["scientific_config_payload"], lock["config_payload"])
    audit = audit_feasibility(summary, lock["scientific_config_payload"], lock["config_payload"])
    scientific_pass = summary["scientific_feasibility_passed"]
    config = lock["config_payload"]
    decision = config["decisionRule"][
        "ifTransportIntegrityAndOriginalV207GatesPass"
        if scientific_pass and summary["transport_integrity_passed"]
        else "ifOriginalScientificGateFails"
    ]
    result = {
        "schema_version": "207r1-agentabstain-shadow-metadata-schema-transport-repair-result",
        "experiment": config["experiment"],
        "passed": audit["passed"],
        "scientific_feasibility_passed": scientific_pass,
        "transport_integrity_passed": summary["transport_integrity_passed"],
        "decision": decision,
        "claim_boundary": config["claimBoundary"],
        "checks": audit["checks"],
        "access_checks": audit["access_checks"],
        "summary": summary,
        "authorization": {
            "preregister_separate_deterministic_text_extraction_only": bool(
                audit["passed"] and scientific_pass and summary["transport_integrity_passed"]
            ),
            "task_text_or_model_run": False,
            "tools_execution_API_training_registration_authority_or_side_effects": False,
        },
    }
    write_json(output_root / "summary.json", summary)
    write_json(output_root / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
