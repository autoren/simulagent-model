#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v10_protocol import file_sha256
from v200_transformed_char_last_controls import audit_evaluation, evaluate_transformed_char_last
from v22r2_grounding import PROJECT_ROOT


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v200-transformed-char-last-controls-lock.json"
    lock = json.loads(lock_path.read_text())
    output_root = PROJECT_ROOT / "outputs/v200-transformed-char-last-controls/evaluation"
    audit_path = PROJECT_ROOT / "outputs/v200-transformed-char-last-controls/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v200-transformed-char-last-controls-outcome-lock.json"
    results_path = PROJECT_ROOT / "docs/v200-transformed-char-last-controls-results.md"
    if audit_path.exists() or outcome_path.exists(): raise RuntimeError("V200 outcome already frozen")
    keys = (
        "config", "parent_V199_outcome", "source_V194_outcome", "source_V193_outcome", "source_V194_lock",
        "development_language", "hidden_targets", "visible_menu_variants", "hidden_variant_maps",
        "canonical_hidden_option_map", "canonical_CHAR_LAST_predictions", "primary_prior", "plan", "protocol",
        "tests", "V194_protocol", "runner", "verifier", "auditor", "design_audit",
    )
    dependencies_exact = valid_lock(lock) and all(
        file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in keys
    )
    rebuilt = evaluate_transformed_char_last(
        *[json.loads((PROJECT_ROOT / lock[key]).read_text()) for key in (
            "development_language", "hidden_targets", "visible_menu_variants", "hidden_variant_maps",
            "canonical_hidden_option_map", "canonical_CHAR_LAST_predictions", "primary_prior",
        )], lock["config_payload"],
    )
    audit = audit_evaluation(rebuilt, lock["config_payload"])
    artifacts_exact = bool(
        json.loads((output_root / "scored-records.json").read_text()) == rebuilt["scored_records"]
        and json.loads((output_root / "summary.json").read_text()) == rebuilt["summary"]
    )
    result_path = output_root / "result.json"; result = json.loads(result_path.read_text())
    decision = lock["config_payload"]["decisionRule"][
        "ifEveryIntegrityInvarianceSignalAndAccessGatePasses" if audit["passed"] else "otherwise"
    ]
    result_exact = result["passed"] == audit["passed"] and result["checks"] == audit["checks"] and result["summary"] == audit["summary"] and result["decision"] == decision
    checks = {
        "design_lock_and_dependencies_are_exact": dependencies_exact,
        "evaluation_artifacts_reconstruct_exactly": artifacts_exact,
        "result_reconstructs_exactly": result_exact,
        "all_deterministic_evaluation_gates_pass": bool(audit["passed"]),
        "results_document_exists": results_path.is_file(),
        "protected_model_authority_and_execution_access_remain_zero": bool(
            rebuilt["summary"]["protected_language_read_count"] == 0 and rebuilt["summary"]["model_load_count"] == 0
            and rebuilt["summary"]["API_call_count"] == 0 and rebuilt["summary"]["actual_execution_count"] == 0
        ),
    }
    passed = all(checks.values())
    outcome_audit = {
        "schema_version": "200-transformed-char-last-controls-outcome-audit", "experiment": lock["experiment"],
        "passed": passed, "evaluation_gates_passed": audit["passed"],
        "decision": "freeze_verified_V200_transformed_CHAR_LAST_controls" if passed else "freeze_failed_V200_verification",
        "checks": checks, "summary": rebuilt["summary"],
    }
    write_json(audit_path, outcome_audit)
    if not passed: print(json.dumps(outcome_audit, indent=2, sort_keys=True)); raise SystemExit(1)
    dependencies = {
        "evaluation_lock": lock_path, "audit": audit_path, "scored_records": output_root / "scored-records.json",
        "summary": output_root / "summary.json", "result": result_path, "results_document": results_path,
        "verifier": PROJECT_ROOT / lock["verifier"],
    }
    outcome: dict[str, Any] = {
        "schema_version": "200-transformed-char-last-controls-outcome-lock", "experiment": lock["experiment"],
        "outcome": {"passed": True, "evaluation_gates_passed": True, "decision": "freeze_V200_controls_and_allow_separate_unchanged_local_model_development_robustness_preregistration_only", "summary": rebuilt["summary"]},
        "authorization": {
            "preregister_separate_unchanged_local_model_development_robustness_only": True,
            "immediate_model_run_or_protected_access": False,
            "API_training_registration_authority_action_or_execution": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT)); outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome); write_json(outcome_path, outcome)
    print(json.dumps(outcome_audit, indent=2, sort_keys=True))
    print(json.dumps({"outcome_lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
