#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, valid_lock, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v154r1_outcome_verifier_repair import canonical_json
from v160_model_free_controlled_relational_grammar_policy import evaluate


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v160-model-free-controlled-relational-grammar-policy-lock.json"
    realization_dir = PROJECT_ROOT / "outputs/v160-model-free-controlled-relational-grammar-policy/model-free-realization"
    result_path = realization_dir / "result.json"
    access_path = realization_dir / "access.json"
    results_doc_path = PROJECT_ROOT / "docs/v160-model-free-controlled-relational-grammar-policy-results.md"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v160_model_free_controlled_relational_grammar_policy_outcome.py"
    audit_path = PROJECT_ROOT / "outputs/v160-model-free-controlled-relational-grammar-policy/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v160-model-free-controlled-relational-grammar-policy-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V160 outcome already frozen")
    if not results_doc_path.is_file():
        raise RuntimeError("write V160 results document before outcome freeze")
    lock = json.loads(lock_path.read_text())
    config = lock["config_payload"]
    result = json.loads(result_path.read_text())
    access = json.loads(access_path.read_text())
    public_requests = json.loads((PROJECT_ROOT / lock["development_public_projection"]).read_text())
    metadata = json.loads((PROJECT_ROOT / lock["development_metadata_projection"]).read_text())
    state_free_catalog = json.loads((PROJECT_ROOT / lock["state_free_catalog_projection"]).read_text())
    witness_catalog = json.loads((PROJECT_ROOT / lock["witness_catalog"]).read_text())
    witness_config = json.loads((PROJECT_ROOT / lock["witness_config"]).read_text())
    expected = evaluate(
        public_requests,
        metadata,
        state_free_catalog,
        witness_catalog,
        witness_config,
        config,
    )
    expected_decision = (
        config["decisionRule"]["ifEveryGrammarSelectivityCostFirewallAndAccessGatePasses"]
        if expected["passed"]
        else config["decisionRule"]["otherwise"]
    )
    expected_result = {
        "schema_version": "160-model-free-controlled-relational-grammar-policy-result",
        "experiment": config["experiment"],
        "completed": True,
        "passed": expected["passed"],
        "checks": expected["checks"],
        "routing_metrics": expected["routing_metrics"],
        "comparator_metrics": expected["comparator_metrics"],
        "routing_records": expected["routing_records"],
        "episode_count": expected["episode_count"],
        "candidate_proposal_field_count": 0,
        "decision": expected_decision,
        "claim_boundary": config["claimBoundary"],
    }
    dependencies = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    prohibited_access = (
        "evaluation_policy_read_count",
        "model_load_count",
        "model_generation_or_score_count",
        "API_call_count",
        "training_run_count",
        "real_service_call_count",
        "external_side_effect_count",
        "actual_execution_count",
    )
    checks = {
        "analysis_lock_and_dependencies_exact": bool(
            valid_lock(lock)
            and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependencies)
        ),
        "result_exact_after_JSON_canonicalization": canonical_json(expected_result) == result,
        "completed_single_development_policy_and_comparators": bool(
            result["completed"]
            and len(public_requests) == 64
            and len(metadata) == 128
            and access["policy_request_decision_count"] == 128
            and access["policy_specific_query_score_count"] == 384
            and access["grammar_alias_lookup_request_count"] == 32
        ),
        "decision_and_gate_outcomes_exact": bool(
            result["decision"] == expected_decision and result["checks"] == expected["checks"]
        ),
        "no_language_or_state_payload_persisted_in_routing_records": all(
            not (
                {
                    "conversation",
                    "text",
                    "truth_state_id",
                    "compatible_state_ids",
                    "state_id",
                    "witness",
                    "grammar_aliases",
                }
                & set(row)
            )
            for row in result["routing_records"].values()
        ),
        "complete_final_safety_retention_failclosure_and_zero_candidate_fields": bool(
            all(
                metric["final_exact_accuracy"] == 1.0
                and metric["irrelevant_intermediate_fail_closed_rate"] == 1.0
                and metric["authoritative_hypothesis_retention"] == 1.0
                and metric["actual_execution_count"] == 0
                for name, metric in result["comparator_metrics"].items()
                if name != "NO_QUERY"
            )
            and result["candidate_proposal_field_count"] == 0
        ),
        "zero_evaluation_model_API_training_services_side_effects_execution": all(
            access[key] == 0 for key in prohibited_access
        ),
        "authorization_remains_narrow": bool(
            lock["authorization"]["run_single_model_free_development_policy"]
            and not lock["authorization"]["read_or_score_V159_evaluation"]
            and not lock["authorization"]["change_aliases_grammar_retrieval_thresholds_costs_gates_or_comparators"]
            and not lock["authorization"]["run_model_hybrid_API_training_induction_authority_action_or_execution"]
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "160-model-free-controlled-relational-grammar-policy-outcome-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "checks": checks,
        "policy_qualified": expected["passed"],
        "decision": expected_decision,
        "routing_metrics": canonical_json(expected["routing_metrics"]),
        "comparator_metrics": canonical_json(expected["comparator_metrics"]),
        "access": access,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    paths = {
        "analysis_lock": lock_path,
        "result": result_path,
        "access": access_path,
        "verifier": verifier_path,
        "audit": audit_path,
        "results_document": results_doc_path,
    }
    outcome: dict[str, Any] = {
        "schema_version": "160-model-free-controlled-relational-grammar-policy-outcome-lock",
        "experiment": config["experiment"],
        "outcome": {
            "passed": True,
            "policy_qualified": expected["passed"],
            "decision": expected_decision,
            "routing_metrics": canonical_json(expected["routing_metrics"]),
            "comparator_metrics": canonical_json(expected["comparator_metrics"]),
            "candidate_proposal_field_count": 0,
        },
        "authorization": {
            "retain_as_project_authored_synthetic_controlled_grammar_development_evidence": True,
            "design_fresh_external_style_transfer_population_if_policy_qualified": expected["passed"],
            "run_transfer_policy_before_separate_preregistration": False,
            "open_or_score_V159_evaluation": False,
            "change_aliases_grammar_retrieval_thresholds_costs_gates_or_rerun": False,
            "run_local_model_or_hybrid": False,
            "fit_calibration_or_learned_retrieval": False,
            "run_API_training_induction_authority_action_or_execution": False,
        },
    }
    for key, path in paths.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
