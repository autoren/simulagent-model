#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v10_protocol import file_sha256
from v203_independent_confirmation_feasibility import audit_feasibility, evaluate_feasibility
from v22r2_grounding import PROJECT_ROOT


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v203-independent-confirmation-feasibility-lock.json"
    lock = json.loads(lock_path.read_text())
    output_root = PROJECT_ROOT / "outputs/v203-independent-confirmation-feasibility/evaluation"
    audit_path = PROJECT_ROOT / "outputs/v203-independent-confirmation-feasibility/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v203-independent-confirmation-feasibility-outcome-lock.json"
    results_path = PROJECT_ROOT / "docs/v203-independent-confirmation-feasibility-results.md"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V203 outcome already frozen")
    dependency_keys = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    dependencies_exact = valid_lock(lock) and all(
        file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependency_keys
    )
    rebuilt = evaluate_feasibility(
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
    rebuilt_audit = audit_feasibility(rebuilt, lock["config_payload"])
    summary_exact = json.loads((output_root / "summary.json").read_text()) == rebuilt
    result_path = output_root / "result.json"
    result = json.loads(result_path.read_text())
    scientific_pass = rebuilt["scientific_feasibility_passed"]
    expected_decision = lock["config_payload"]["decisionRule"][
        "ifEligibleFamilyPassesEveryQualificationAndAccessGate" if scientific_pass else "otherwise"
    ]
    result_exact = bool(
        result["passed"] == rebuilt_audit["passed"]
        and result["scientific_feasibility_passed"] == scientific_pass
        and result["checks"] == rebuilt_audit["checks"]
        and result["summary"] == rebuilt
        and result["decision"] == expected_decision
    )
    checks = {
        "design_lock_and_dependencies_are_exact": dependencies_exact,
        "summary_reconstructs_exactly": summary_exact,
        "result_reconstructs_exactly": result_exact,
        "integrity_and_access_audit_passes": rebuilt_audit["passed"],
        "results_document_exists": results_path.is_file(),
        "utterance_protected_model_API_training_authority_action_and_execution_remain_zero": bool(
            rebuilt["utterance_or_dialogue_text_read_or_emission_count"] == 0
            and rebuilt["protected_language_read_count"] == 0
            and rebuilt["model_load_count"] == 0
            and rebuilt["model_generation_count"] == 0
            and rebuilt["API_call_count"] == 0
            and rebuilt["training_run_count"] == 0
            and rebuilt["ontology_registration_count"] == 0
            and rebuilt["trusted_state_mutation_count"] == 0
            and rebuilt["service_call_count"] == 0
            and rebuilt["external_side_effect_count"] == 0
            and rebuilt["actual_execution_count"] == 0
        ),
    }
    passed = all(checks.values())
    outcome_audit = {
        "schema_version": "203-independent-confirmation-feasibility-outcome-audit",
        "experiment": lock["experiment"],
        "passed": passed,
        "scientific_feasibility_passed": scientific_pass,
        "decision": "freeze_verified_V203_feasibility_result" if passed else "freeze_failed_V203_verification",
        "checks": checks,
        "summary": rebuilt,
    }
    write_json(audit_path, outcome_audit)
    if not passed:
        print(json.dumps(outcome_audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "evaluation_lock": lock_path,
        "audit": audit_path,
        "summary": output_root / "summary.json",
        "result": result_path,
        "results_document": results_path,
        "verifier": PROJECT_ROOT / lock["verifier"],
    }
    outcome: dict[str, Any] = {
        "schema_version": "203-independent-confirmation-feasibility-outcome-lock",
        "experiment": lock["experiment"],
        "outcome": {
            "passed": True,
            "scientific_feasibility_passed": scientific_pass,
            "decision": expected_decision,
            "summary": rebuilt,
        },
        "authorization": {
            "preregister_separate_text_free_population_selection_only": scientific_pass,
            "preregister_separate_richer_model_free_POMDP_only": not scientific_pass,
            "immediate_population_language_model_or_protected_access": False,
            "API_training_registration_authority_action_or_execution": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps(outcome_audit, indent=2, sort_keys=True))
    print(json.dumps({"outcome_lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
