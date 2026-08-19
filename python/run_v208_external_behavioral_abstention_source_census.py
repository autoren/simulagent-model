#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from v10_protocol import file_sha256
from v208_external_behavioral_abstention_source_census import audit_census, evaluate_census
from v22r2_grounding import PROJECT_ROOT


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v208-external-behavioral-abstention-source-census-lock.json"
    lock = json.loads(lock_path.read_text())
    if not valid_lock(lock):
        raise RuntimeError("invalid V208 design lock")
    dependency_keys = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    if not all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependency_keys):
        raise RuntimeError("V208 locked dependency changed")
    output_root = PROJECT_ROOT / "outputs/v208-external-behavioral-abstention-source-census/evaluation"
    if output_root.exists():
        raise RuntimeError("V208 output exists")

    summary = evaluate_census(lock["config_payload"])
    audit = audit_census(summary, lock["config_payload"])
    scientific_pass = summary["scientific_feasibility_passed"]
    decision = lock["config_payload"]["decisionRule"][
        "ifAtLeastOneCandidatePassesEveryGate" if scientific_pass else "otherwise"
    ]
    result = {
        "schema_version": "208-external-behavioral-abstention-source-census-result",
        "experiment": lock["experiment"],
        "passed": audit["passed"],
        "scientific_feasibility_passed": scientific_pass,
        "decision": decision,
        "claim_boundary": lock["config_payload"]["claimBoundary"],
        "checks": audit["checks"],
        "summary": summary,
        "authorization": {
            "preregister_exact_identifier_selection_and_text_extraction_only": bool(audit["passed"] and scientific_pass),
            "task_text_or_model_run": False,
            "API_training_tool_service_registration_authority_action_or_execution": False,
        },
    }
    write_json(output_root / "summary.json", summary)
    write_json(output_root / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
