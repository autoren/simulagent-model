#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, valid_lock, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v148_typed_witness_firewall import evaluate


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v148-typed-witness-firewall.json"
    plan_path = PROJECT_ROOT / "docs/v148-typed-witness-firewall-plan.md"
    results_path = PROJECT_ROOT / "docs/v148-typed-witness-firewall-results.md"
    protocol_path = PROJECT_ROOT / "python/v148_typed_witness_firewall.py"
    tests_path = PROJECT_ROOT / "python/test_v148_typed_witness_firewall.py"
    runner_path = PROJECT_ROOT / "python/run_and_freeze_v148_typed_witness_firewall.py"
    audit_path = PROJECT_ROOT / "outputs/v148-typed-witness-firewall/audit.json"
    outcome_path = PROJECT_ROOT / "configs/v148-typed-witness-firewall-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V148 already frozen")

    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV147OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    result = evaluate(config)
    checks = {
        "V147_outcome_valid_negative_and_branch_closed": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["outcome"]["audit_pass"]
            and not parent["outcome"]["qualified"]
            and not parent["authorization"]["preregister_new_blind_successor_population"]
            and not parent["authorization"]["score_or_use_retired_V146_test"]
            and not parent["authorization"]["modify_retry_rescore_rerun_reprompt_tune_or_mine_V147"]
        ),
        "all_model_free_oracle_mutation_invariance_gates_pass": result["passed"],
        "LLM_proposal_is_permanently_non_authoritative": bool(
            config["interface"]["LLMProposalIsOptionalAndNonAuthoritative"]
            and config["interface"]["LLMProposalMayNotSatisfyWitness"]
            and config["interface"]["knownAcceptanceRequiresExactTrustedWitness"]
        ),
        "novel_candidate_is_generic_nonregistering_and_nonexecuting": bool(
            config["interface"]["wellTypedUnknownRoutesToGenericNovelCandidateWithoutRegistration"]
            and config["interface"]["novelCandidateIsNonExecutable"]
        ),
        "complete_hypothesis_retention_and_fail_closed_A00": bool(
            config["interface"]["completeAuthoritativeHypothesisUniverseRetained"]
            and config["interface"]["missingMalformedContradictoryOrInsufficientRoutesToA00"]
        ),
        "zero_language_model_API_training_execution": all(
            result["metrics"][key] == 0
            for key in (
                "project_language_read_count",
                "model_load_count",
                "model_generation_or_score_count",
                "API_call_count",
                "training_run_count",
                "actual_execution_count",
            )
        ),
        "required_files_exist": all(
            path.is_file()
            for path in (config_path, plan_path, results_path, protocol_path, tests_path, runner_path)
        ),
    }
    passed = all(checks.values())
    decision = (
        config["decisionRule"]["ifEveryOracleMutationInvarianceAndAccessGatePasses"]
        if passed
        else config["decisionRule"]["otherwise"]
    )
    audit = {
        "schema_version": "148-typed-witness-firewall-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "checks": checks,
        "result": result,
        "decision": decision,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "config": config_path,
        "parent_outcome": parent_path,
        "plan": plan_path,
        "results_document": results_path,
        "protocol": protocol_path,
        "tests": tests_path,
        "runner": runner_path,
        "audit": audit_path,
    }
    outcome: dict[str, Any] = {
        "schema_version": "148-typed-witness-firewall-outcome-lock",
        "experiment": config["experiment"],
        "outcome": {
            "passed": True,
            "typed_witness_firewall_structurally_feasible": True,
            "language_witness_acquisition_remains_unproven": True,
            "decision": decision,
            "metrics": result["metrics"],
        },
        "authorization": {
            "design_fresh_blind_closed_interaction_population": True,
            "run_language_or_model_before_separate_preregistration": False,
            "use_retired_V146_test_or_mine_V147": False,
            "fit_calibration_threshold": False,
            "run_API_training_induction_authority_action_or_execution": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps({"passed": passed, "decision": decision, "metrics": result["metrics"], "checks": checks}, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
