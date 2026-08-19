#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v202-model-free-controller-decision-sufficiency.json"
    plan_path = PROJECT_ROOT / "docs/v202-model-free-controller-decision-sufficiency-plan.md"
    protocol_path = PROJECT_ROOT / "python/v202_model_free_controller_decision_sufficiency.py"
    tests_path = PROJECT_ROOT / "python/test_v202_model_free_controller_decision_sufficiency.py"
    runner_path = PROJECT_ROOT / "python/run_v202_model_free_controller_decision_sufficiency.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v202_model_free_controller_decision_sufficiency_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v202_model_free_controller_decision_sufficiency.py"
    audit_path = PROJECT_ROOT / "outputs/v202-model-free-controller-decision-sufficiency/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v202-model-free-controller-decision-sufficiency-lock.json"
    output_root = PROJECT_ROOT / "outputs/v202-model-free-controller-decision-sufficiency/evaluation"
    outcome_path = PROJECT_ROOT / "configs/v202-model-free-controller-decision-sufficiency-outcome-lock.json"
    if any(path.exists() for path in (audit_path, lock_path, output_root, outcome_path)):
        raise RuntimeError("V202 is already preregistered, run, or frozen")

    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV201r2OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    source_outcomes = {
        "V195_outcome": PROJECT_ROOT / "configs/v195-bounded-local-language-menu-ranker-outcome-lock.json",
        "V194_outcome": PROJECT_ROOT / "configs/v194-deterministic-language-menu-rankers-outcome-lock.json",
        "V200_outcome": PROJECT_ROOT / "configs/v200-transformed-char-last-controls-outcome-lock.json",
        "V191_outcome": PROJECT_ROOT / "configs/v191-fresh-language-to-menu-population-outcome-lock.json",
        "V193_outcome": PROJECT_ROOT / "configs/v193-shadow-menu-interface-frontier-outcome-lock.json",
        "V199_outcome": PROJECT_ROOT / "configs/v199-exact-menu-presentation-robustness-outcome-lock.json",
    }
    outcomes = {key: json.loads(path.read_text()) for key, path in source_outcomes.items()}
    inputs = {
        "canonical_model_census": PROJECT_ROOT / config["canonicalModelCensus"],
        "transformed_model_census": PROJECT_ROOT / config["transformedModelCensus"],
        "canonical_CHAR_LAST_predictions": PROJECT_ROOT / config["canonicalCHARLASTPredictions"],
        "transformed_CHAR_LAST_scored_records": PROJECT_ROOT / config["transformedCHARLASTScoredRecords"],
        "hidden_targets": PROJECT_ROOT / config["hiddenTargets"],
        "canonical_hidden_option_map": PROJECT_ROOT / config["canonicalHiddenOptionMap"],
        "hidden_variant_maps": PROJECT_ROOT / config["hiddenVariantMaps"],
        "primary_prior": PROJECT_ROOT / config["primaryPrior"],
        "fixed_hierarchy_target_costs": PROJECT_ROOT / config["fixedHierarchyTargetCosts"],
    }
    exact_hashes = {
        "canonical_model_census": outcomes["V195_outcome"]["census_result_sha256"],
        "transformed_model_census": parent["source_V201_census_sha256"],
        "canonical_CHAR_LAST_predictions": outcomes["V194_outcome"]["shadow_predictions_sha256"],
        "transformed_CHAR_LAST_scored_records": outcomes["V200_outcome"]["scored_records_sha256"],
        "hidden_targets": outcomes["V191_outcome"]["hidden_targets_sha256"],
        "canonical_hidden_option_map": outcomes["V193_outcome"]["hidden_option_map_sha256"],
        "hidden_variant_maps": outcomes["V199_outcome"]["hidden_variant_maps_sha256"],
        "primary_prior": outcomes["V193_outcome"]["primary_prior_sha256"],
        "fixed_hierarchy_target_costs": outcomes["V193_outcome"]["fixed_hierarchy_target_costs_sha256"],
    }
    source_locks_valid = all(valid_lock(outcome) for outcome in outcomes.values())
    checks = {
        "V201r2_is_valid_and_authorizes_only_separate_model_free_decision_sufficiency": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and not parent["outcome"]["V201_scientific_qualification_gates_passed"]
            and parent["authorization"]["update_roadmap_and_preregister_separate_model_free_decision_sufficiency_design"]
            and not parent["authorization"]["run_paired_protected_robustness"]
        ),
        "all_frozen_source_outcomes_are_valid": source_locks_valid,
        "all_normalized_inputs_match_frozen_hashes": all(
            path.is_file() and file_sha256(path) == exact_hashes[key]
            for key, path in inputs.items()
        ),
        "fixed_policy_selection_and_cost_contract_is_complete": bool(
            [row["policyId"] for row in config["controllerPolicies"]]
            == [
                "SINGLE_PRESENTATION_TOP1_FAMILY",
                "SINGLE_PRESENTATION_TOP3_FAMILY",
                "TOP1_PLURALITY_3X",
                "TOP3_INCLUSION_CONSENSUS_3X",
            ]
            and config["presentations"]
            == ["CANONICAL", "ORDER_ONLY", "ORDER_AND_OPAQUE_ID"]
            and config["trustedController"]["fullAuthoritativeHypothesisUniverseAlwaysRetained"]
            and config["trustedController"]["trustedAnswerRequiredForEveryExactTerminal"]
        ),
        "prelock_evaluation_language_model_API_training_and_execution_access_is_zero": all(
            value == 0 for value in config["preLockExposure"].values()
        ),
        "confirmation_protected_API_training_authority_action_and_execution_are_closed": bool(
            not config["decisionRule"]["passAuthorizesImmediateConfirmationOrProtectedReuse"]
            and not config["decisionRule"]["passAuthorizesAPIModelGenerationTrainingRegistrationAuthorityActionOrExecution"]
        ),
        "required_files_exist_and_outputs_are_absent": bool(
            all(
                path.is_file()
                for path in (
                    config_path,
                    plan_path,
                    protocol_path,
                    tests_path,
                    runner_path,
                    verifier_path,
                    auditor_path,
                    parent_path,
                    *source_outcomes.values(),
                    *inputs.values(),
                )
            )
            and not output_root.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "202-model-free-controller-decision-sufficiency-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_V202_model_free_evaluation" if passed else "reject_V202_design",
        "checks": checks,
        "prelock_exposure": config["preLockExposure"],
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "config": config_path,
        "parent_V201r2_outcome": parent_path,
        **source_outcomes,
        **inputs,
        "plan": plan_path,
        "protocol": protocol_path,
        "tests": tests_path,
        "runner": runner_path,
        "verifier": verifier_path,
        "auditor": auditor_path,
        "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "202-model-free-controller-decision-sufficiency-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "modify_sources_policies_costs_metrics_gates_or_selection": False,
            "run_exact_single_model_free_development_evaluation": True,
            "read_language_raw_model_responses_or_run_model": False,
            "run_confirmation_protected_API_training_registration_authority_action_or_execution": False,
        },
    }
    for key, path in dependencies.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT))
        lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock)
    write_json(lock_path, lock)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(
        json.dumps(
            {"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
