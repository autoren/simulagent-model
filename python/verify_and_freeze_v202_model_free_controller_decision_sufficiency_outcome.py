#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v10_protocol import file_sha256
from v202_model_free_controller_decision_sufficiency import (
    audit_evaluation,
    evaluate_controllers,
)
from v22r2_grounding import PROJECT_ROOT


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v202-model-free-controller-decision-sufficiency-lock.json"
    lock = json.loads(lock_path.read_text())
    output_root = PROJECT_ROOT / "outputs/v202-model-free-controller-decision-sufficiency/evaluation"
    audit_path = PROJECT_ROOT / "outputs/v202-model-free-controller-decision-sufficiency/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v202-model-free-controller-decision-sufficiency-outcome-lock.json"
    results_path = PROJECT_ROOT / "docs/v202-model-free-controller-decision-sufficiency-results.md"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V202 outcome already frozen")

    dependency_keys = [
        key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock
    ]
    dependencies_exact = valid_lock(lock) and all(
        file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"]
        for key in dependency_keys
    )
    input_keys = (
        "canonical_model_census",
        "transformed_model_census",
        "canonical_CHAR_LAST_predictions",
        "transformed_CHAR_LAST_scored_records",
        "hidden_targets",
        "canonical_hidden_option_map",
        "hidden_variant_maps",
        "primary_prior",
        "fixed_hierarchy_target_costs",
    )
    rebuilt = evaluate_controllers(
        *[json.loads((PROJECT_ROOT / lock[key]).read_text()) for key in input_keys],
        lock["config_payload"],
    )
    rebuilt_audit = audit_evaluation(rebuilt, lock["config_payload"])
    artifacts_exact = bool(
        json.loads((output_root / "summary.json").read_text()) == rebuilt["summary"]
        and json.loads((output_root / "scored-records.json").read_text())
        == rebuilt["scored_records"]
    )
    result_path = output_root / "result.json"
    result = json.loads(result_path.read_text())
    selected = rebuilt["summary"]["selected_policy_id"] is not None
    expected_decision = lock["config_payload"]["decisionRule"][
        "ifAtLeastOnePolicyQualifies" if selected else "otherwise"
    ]
    result_exact = bool(
        result["passed"] == rebuilt_audit["passed"]
        and result["scientific_selection_made"] == selected
        and result["checks"] == rebuilt_audit["checks"]
        and result["summary"] == rebuilt_audit["summary"]
        and result["decision"] == expected_decision
    )
    checks = {
        "design_lock_and_dependencies_are_exact": dependencies_exact,
        "evaluation_artifacts_reconstruct_exactly": artifacts_exact,
        "result_reconstructs_exactly": result_exact,
        "all_integrity_access_and_safety_checks_pass": rebuilt_audit["passed"],
        "results_document_exists": results_path.is_file(),
        "language_raw_model_model_API_training_authority_action_and_execution_remain_zero": bool(
            rebuilt["summary"]["raw_model_response_read_count"] == 0
            and rebuilt["summary"]["utterance_or_dialogue_language_read_count"] == 0
            and rebuilt["summary"]["model_load_count"] == 0
            and rebuilt["summary"]["model_generation_count"] == 0
            and rebuilt["summary"]["protected_language_read_count"] == 0
            and rebuilt["summary"]["API_call_count"] == 0
            and rebuilt["summary"]["training_run_count"] == 0
            and rebuilt["summary"]["ontology_registration_count"] == 0
            and rebuilt["summary"]["trusted_state_mutation_count"] == 0
            and rebuilt["summary"]["service_call_count"] == 0
            and rebuilt["summary"]["external_side_effect_count"] == 0
            and rebuilt["summary"]["actual_execution_count"] == 0
        ),
    }
    passed = all(checks.values())
    outcome_audit = {
        "schema_version": "202-model-free-controller-decision-sufficiency-outcome-audit",
        "experiment": lock["experiment"],
        "passed": passed,
        "scientific_selection_made": selected,
        "decision": "freeze_verified_V202_result" if passed else "freeze_failed_V202_verification",
        "checks": checks,
        "summary": rebuilt["summary"],
    }
    write_json(audit_path, outcome_audit)
    if not passed:
        print(json.dumps(outcome_audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "evaluation_lock": lock_path,
        "audit": audit_path,
        "scored_records": output_root / "scored-records.json",
        "summary": output_root / "summary.json",
        "result": result_path,
        "results_document": results_path,
        "verifier": PROJECT_ROOT / lock["verifier"],
    }
    outcome: dict[str, Any] = {
        "schema_version": "202-model-free-controller-decision-sufficiency-outcome-lock",
        "experiment": lock["experiment"],
        "outcome": {
            "passed": True,
            "scientific_selection_made": selected,
            "decision": expected_decision,
            "summary": rebuilt["summary"],
        },
        "authorization": {
            "preregister_new_fresh_confirmation_design_only": selected,
            "update_roadmap_and_preregister_next_model_free_track": True,
            "immediate_confirmation_or_protected_reuse": False,
            "API_model_generation_training_registration_authority_action_or_execution": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps(outcome_audit, indent=2, sort_keys=True))
    print(
        json.dumps(
            {"outcome_lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
