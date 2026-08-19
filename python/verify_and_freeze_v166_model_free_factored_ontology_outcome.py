#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v166_model_free_factored_ontology_baselines import payload_hash
from run_v166_model_free_factored_ontology_baselines import DEPENDENCY_KEYS, reconstruct
from v166_model_free_factored_ontology_baselines import evaluate_gates


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v166-model-free-factored-ontology-baselines-lock.json"
    result_path = PROJECT_ROOT / "outputs/v166-model-free-factored-ontology/baselines/result.json"
    doc_path = PROJECT_ROOT / "docs/v166-model-free-factored-ontology-baselines-results.md"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v166_model_free_factored_ontology_outcome.py"
    audit_path = PROJECT_ROOT / "outputs/v166-model-free-factored-ontology/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v166-model-free-factored-ontology-baselines-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V166 outcome is already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write the V166 result document before freezing")

    lock = json.loads(lock_path.read_text())
    result = json.loads(result_path.read_text())
    artifacts = reconstruct(lock)
    config = lock["config_payload"]
    access = result["access"]
    reconstructed_gates = evaluate_gates(artifacts["evaluation"], access, config)
    expected_outputs = {
        "baseline_predictions": {"predictions": artifacts["predictions"], "contains_language": False},
        "baseline_evaluation": artifacts["evaluation"],
        "model_eligible_residual": {
            "record_ids": artifacts["evaluation"]["model_eligible_residual_record_ids"],
            "count": artifacts["evaluation"]["model_eligible_residual_count"],
            "intentionally_ambiguous_records_are_residuals": False,
            "contains_language": False,
        },
    }
    outputs_exact = all(
        json.loads((PROJECT_ROOT / result["output_integrity"][key]["path"]).read_text()) == value
        and file_sha256(PROJECT_ROOT / result["output_integrity"][key]["path"])
        == result["output_integrity"][key]["sha256"]
        for key, value in expected_outputs.items()
    )
    residual_count = artifacts["evaluation"]["model_eligible_residual_count"]
    expected_decision = (
        config["decisionRule"]["ifEveryBaselineGatePassesAndResidualIsZero"]
        if all(reconstructed_gates.values()) and residual_count == 0
        else config["decisionRule"]["ifPipelinePassesButResidualIsNonzero"]
        if all(reconstructed_gates.values())
        else config["decisionRule"]["otherwise"]
    )
    checks = {
        "benchmark_lock_and_dependencies_are_exact": bool(
            payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"})
            == lock["lock_payload_sha256"]
            and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in DEPENDENCY_KEYS)
        ),
        "all_outputs_reconstruct_exactly": outputs_exact,
        "metrics_gates_residual_and_decision_reconstruct_exactly": bool(
            artifacts["evaluation"]["baseline_metrics"] == result["baseline_metrics"]
            and reconstructed_gates == result["gates"]
            and result["passed"] == all(reconstructed_gates.values())
            and result["model_eligible_residual_count"] == residual_count
            and result["decision"] == expected_decision
        ),
        "combined_and_oracle_exactly_retain_the_frozen_contract": all(
            artifacts["evaluation"]["baseline_metrics"][name]["exact_version_space_accuracy"] == 1.0
            and artifacts["evaluation"]["baseline_metrics"][name]["evidence_status_accuracy"] == 1.0
            for name in ("exact_parser_plus_version_space", "oracle_hidden_contract")
        ),
        "ambiguity_is_retained_and_not_mislabeled_as_model_residual": bool(
            artifacts["evaluation"]["intentionally_ambiguous_record_count"] == 48
            and set(artifacts["evaluation"]["intentionally_ambiguous_candidate_counts"]) == {64}
            and residual_count == 0
        ),
        "model_registration_authority_action_and_execution_remain_zero": all(
            access[key] == 0 for key in (
                "evaluation_record_count", "manual_judgment_count", "model_load_count",
                "model_generation_count", "API_call_count", "training_run_count",
                "ontology_registration_count", "real_service_call_count",
                "external_side_effect_count", "actual_execution_count",
            )
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "166-model-free-factored-ontology-baselines-outcome-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "scientific_baselines_passed": result["passed"],
        "decision": "freeze_positive_V166_model_free_outcome" if passed and result["passed"] else "reject_V166_outcome",
        "checks": checks,
        "independent_evaluation": artifacts["evaluation"],
        "additional_access": {
            "public_record_read_count": 1,
            "hidden_truth_read_count": 1,
            "frozen_ontology_read_count": 1,
            "population_summary_read_count": 1,
            "model_load_count": 0,
            "model_generation_count": 0,
            "API_call_count": 0,
            "ontology_registration_count": 0,
            "actual_execution_count": 0,
        },
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "benchmark_lock": lock_path,
        "result": result_path,
        "verifier": verifier_path,
        "audit": audit_path,
        "results_document": doc_path,
        "parent_V165r1_outcome": PROJECT_ROOT / lock["parent_V165r1_outcome"],
        "frozen_ontology": PROJECT_ROOT / lock["frozen_ontology"],
        "public_records": PROJECT_ROOT / lock["public_records"],
        "hidden_records": PROJECT_ROOT / lock["hidden_records"],
        "population_summary": PROJECT_ROOT / lock["population_summary"],
    }
    for key, integrity in result["output_integrity"].items():
        dependencies[key] = PROJECT_ROOT / integrity["path"]
    outcome: dict[str, Any] = {
        "schema_version": "166-model-free-factored-ontology-baselines-outcome-lock",
        "experiment": config["experiment"],
        "outcome": {
            "passed": True,
            "scientific_baselines_passed": result["passed"],
            "decision": result["decision"],
            "baseline_metrics": result["baseline_metrics"],
            "model_eligible_residual_count": residual_count,
            "intentionally_ambiguous_record_count": result["intentionally_ambiguous_record_count"],
            "intentionally_ambiguous_candidate_count": 64,
        },
        "authorization": {
            "modify_or_rerun_V166": False,
            "retain_V166_as_project_authored_model_free_positive_development_evidence": True,
            "preregister_evidence_gathering_planner_on_frozen_ambiguous_states": bool(result["passed"] and residual_count == 0),
            "preregister_fixed_ontology_reversible_sandbox": bool(result["passed"] and residual_count == 0),
            "run_evidence_planner_or_sandbox_without_separate_lock": False,
            "run_local_or_API_model_on_zero_residual": False,
            "create_or_open_evaluation_population": False,
            "register_provisional_primitive": False,
            "grant_candidate_state_belief_action_or_execution_authority": False,
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
