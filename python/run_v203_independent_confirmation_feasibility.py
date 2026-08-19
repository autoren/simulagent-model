#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from v10_protocol import file_sha256
from v203_independent_confirmation_feasibility import audit_feasibility, evaluate_feasibility
from v22r2_grounding import PROJECT_ROOT


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v203-independent-confirmation-feasibility-lock.json"
    lock = json.loads(lock_path.read_text())
    if not valid_lock(lock):
        raise RuntimeError("invalid V203 design lock")
    dependency_keys = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    for key in dependency_keys:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V203 locked dependency changed: {key}")
    output_root = PROJECT_ROOT / "outputs/v203-independent-confirmation-feasibility/evaluation"
    if output_root.exists():
        raise RuntimeError("V203 output already exists")
    summary = evaluate_feasibility(
        (PROJECT_ROOT / lock["source_archive"]).read_bytes(),
        *[
            json.loads((PROJECT_ROOT / lock[key]).read_text())
            for key in (
                "source_inventory",
                "contract_catalog",
                "V183_consumed_population",
                "V191_consumed_population",
                "source_V87_design_lock",
            )
        ],
        lock["config_payload"],
    )
    audit = audit_feasibility(summary, lock["config_payload"])
    config = lock["config_payload"]
    scientific_pass = summary["scientific_feasibility_passed"]
    decision = config["decisionRule"][
        "ifEligibleFamilyPassesEveryQualificationAndAccessGate" if scientific_pass else "otherwise"
    ]
    result = {
        "schema_version": "203-independent-confirmation-feasibility-result",
        "experiment": config["experiment"],
        "passed": audit["passed"],
        "scientific_feasibility_passed": scientific_pass,
        "decision": decision,
        "claim_boundary": config["claimBoundary"],
        "checks": audit["checks"],
        "summary": summary,
        "authorization": {
            "preregister_separate_text_free_population_selection_only": bool(audit["passed"] and scientific_pass),
            "preregister_separate_richer_model_free_POMDP_only": bool(audit["passed"] and not scientific_pass),
            "immediate_population_language_model_or_protected_access": False,
            "API_training_registration_authority_action_or_execution": False,
        },
    }
    write_json(output_root / "summary.json", summary)
    write_json(output_root / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
