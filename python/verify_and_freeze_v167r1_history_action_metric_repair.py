#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from run_v167_exact_evidence_gathering_planner import DEPENDENCY_KEYS, reconstruct
from v167r1_history_action_metric_repair import corrected_history_count, corrected_summary


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    repair_lock_path = PROJECT_ROOT / "configs/v167r1-history-action-metric-repair-lock.json"
    audit_path = PROJECT_ROOT / "outputs/v167r1-history-action-metric-repair/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v167r1-history-action-metric-repair-outcome-lock.json"
    nominal_outcome = PROJECT_ROOT / "configs/v167-exact-evidence-gathering-planner-outcome-lock.json"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v167r1_history_action_metric_repair.py"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V167r1 outcome is already frozen")
    if nominal_outcome.exists():
        raise RuntimeError("nominal V167 outcome unexpectedly exists")

    repair = json.loads(repair_lock_path.read_text())
    config = repair["config_payload"]
    parent = json.loads((PROJECT_ROOT / repair["parent_V167_planner_lock"]).read_text())
    result = json.loads((PROJECT_ROOT / repair["V167_result"]).read_text())
    policy_artifact = json.loads((PROJECT_ROOT / repair["V167_policy_trees"]).read_text())
    evaluation = reconstruct(parent)
    repaired_summary = corrected_summary(evaluation)
    diagnosis = config["diagnosis"]
    repair_dependencies = [key for key in repair if not key.endswith("_sha256") and f"{key}_sha256" in repair]
    checks = {
        "repair_lock_and_dependencies_are_exact": bool(
            valid_lock(repair)
            and all(file_sha256(PROJECT_ROOT / repair[key]) == repair[f"{key}_sha256"] for key in repair_dependencies)
        ),
        "parent_lock_and_dependencies_are_exact": bool(
            valid_lock(parent)
            and all(file_sha256(PROJECT_ROOT / parent[key]) == parent[f"{key}_sha256"] for key in DEPENDENCY_KEYS)
        ),
        "original_run_and_artifacts_reconstruct_exactly": bool(
            result["summary"] == evaluation["summary"]
            and policy_artifact == {"cases": evaluation["cases"], "contains_language": False, "shadow_only": True}
            and result["passed"]
            and all(result["gates"].values())
        ),
        "only_history_action_projection_changes_from_48_to_28": bool(
            evaluation["summary"][diagnosis["affectedMetric"]] == diagnosis["persistedValue"]
            and corrected_history_count(evaluation["cases"]) == diagnosis["correctedValue"]
            and repaired_summary[diagnosis["affectedMetric"]] == diagnosis["correctedValue"]
            and {key for key in repaired_summary if repaired_summary[key] != evaluation["summary"][key]} == {diagnosis["affectedMetric"]}
        ),
        "corrected_gate_and_all_other_scientific_gates_pass": bool(
            diagnosis["correctedValue"] >= diagnosis["minimumFrozenGate"]
            and all(value for key, value in result["gates"].items() if key != "history_dependent_second_action")
        ),
        "zero_external_or_authority_access_is_preserved": bool(
            all(value == 0 for value in config["accessGates"].values())
            and all(result["access"][key] == 0 for key in (
                "evaluation_record_count", "manual_judgment_count", "model_load_count", "model_generation_count",
                "API_call_count", "training_run_count", "ontology_registration_count", "trusted_state_mutation_count",
                "real_service_call_count", "external_side_effect_count", "actual_execution_count",
            ))
        ),
        "nominal_V167_outcome_remains_absent": not nominal_outcome.exists(),
    }
    passed = all(checks.values())
    decision = config["decisionRule"]["ifRepairReconstructsExactlyAndAllCorrectedGatesPass"] if passed else config["decisionRule"]["otherwise"]
    audit = {
        "schema_version": "167r1-history-action-metric-repair-outcome-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "technical_metric_projection_repair_only": True,
        "decision": decision,
        "checks": checks,
        "corrected_summary": repaired_summary,
        "repair_access": {key: 0 for key in config["accessGates"]},
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "repair_lock": repair_lock_path,
        "parent_V167_planner_lock": PROJECT_ROOT / repair["parent_V167_planner_lock"],
        "V167_result": PROJECT_ROOT / repair["V167_result"],
        "V167_policy_trees": PROJECT_ROOT / repair["V167_policy_trees"],
        "verifier": verifier_path,
        "audit": audit_path,
        "results_document": PROJECT_ROOT / repair["results_document"],
    }
    outcome: dict[str, Any] = {
        "schema_version": "167r1-history-action-metric-repair-outcome-lock",
        "experiment": config["experiment"],
        "outcome": {
            "passed": True,
            "V167_scientific_planner_gates_passed": True,
            "development_informed_not_confirmatory": True,
            "technical_metric_projection_repair_only": True,
            "decision": decision,
            "corrected_summary": repaired_summary,
        },
        "authorization": {
            "modify_or_rerun_V167": False,
            "create_nominal_V167_outcome": False,
            "retain_corrected_V167_as_project_authored_development_mechanism_evidence": True,
            "claim_fresh_or_external_confirmation": False,
            "preregister_fixed_ontology_reversible_sandbox": True,
            "run_sandbox_without_separate_lock": False,
            "run_local_or_API_model": False,
            "register_provisional_primitive": False,
            "grant_candidate_or_planner_trusted_state_action_or_execution_authority": False,
            "perform_real_service_call_external_side_effect_or_execution": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
