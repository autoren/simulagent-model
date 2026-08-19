#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, valid_lock, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v145_finite_certificate_codebook import evaluate


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v145-finite-certificate-codebook.json"
    plan_path = PROJECT_ROOT / "docs/v145-finite-certificate-codebook-plan.md"
    results_path = PROJECT_ROOT / "docs/v145-finite-certificate-codebook-results.md"
    protocol_path = PROJECT_ROOT / "python/v145_finite_certificate_codebook.py"
    tests_path = PROJECT_ROOT / "python/test_v145_finite_certificate_codebook.py"
    runner_path = PROJECT_ROOT / "python/run_and_freeze_v145_finite_certificate_codebook.py"
    audit_path = PROJECT_ROOT / "outputs/v145-finite-certificate-codebook/audit.json"
    codebook_path = PROJECT_ROOT / "outputs/v145-finite-certificate-codebook/codebook.json"
    outcome_path = PROJECT_ROOT / "configs/v145-finite-certificate-codebook-outcome-lock.json"
    if audit_path.exists() or codebook_path.exists() or outcome_path.exists():
        raise RuntimeError("V145 already frozen")
    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV144OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    result = evaluate(config)
    parent_checks = {
        "V144_outcome_valid_and_branch_closed": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and not parent["outcome"]["qualified"]
            and not parent["authorization"]["preregister_one_separate_frozen_V142_test_realization"]
            and not parent["authorization"]["modify_retry_rerun_reprompt_tune_or_mine_V144"]
        ),
        "all_model_free_codebook_gates_pass": result["passed"],
        "future_interface_is_score_only_and_non_authoritative": bool(
            not config["futureInterface"]["freeFormReasoningOrCertificateGeneration"]
            and config["futureInterface"]["scoreVectorIsNonAuthoritative"]
            and config["futureInterface"]["semanticCorrectnessStillRequiresFreshEmpiricalEvidence"]
        ),
        "no_language_model_API_training_or_execution": all(
            result["metrics"][key] == 0
            for key in ("project_language_read_count", "model_load_count", "model_generation_count", "API_call_count", "training_run_count", "actual_execution_count")
        ),
        "required_files_exist": all(path.is_file() for path in (config_path, plan_path, results_path, protocol_path, tests_path, runner_path)),
    }
    passed = all(parent_checks.values())
    decision = config["decisionRule"]["ifEveryStructuralOracleMutationAndAccessGatePasses"] if passed else config["decisionRule"]["otherwise"]
    audit = {
        "schema_version": "145-finite-certificate-codebook-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "checks": parent_checks,
        "result": result,
        "decision": decision,
    }
    write_json(audit_path, audit)
    write_json(codebook_path, {"schema_version": "145-registered-certificate-codebook", "entries": result["codebook"]})
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
        "codebook": codebook_path,
    }
    outcome: dict[str, Any] = {
        "schema_version": "145-finite-certificate-codebook-outcome-lock",
        "experiment": config["experiment"],
        "outcome": {
            "passed": True,
            "finite_codebook_structurally_feasible": True,
            "semantic_correctness_remains_empirical": True,
            "decision": decision,
            "metrics": result["metrics"],
        },
        "authorization": {
            "design_fresh_codebook_scoring_population_and_protocol": True,
            "run_language_or_model": False,
            "open_or_generate_on_V142_test": False,
            "retry_rerun_tune_or_mine_V144": False,
            "run_API_training_induction_authority_action_or_execution": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps({"passed": passed, "decision": decision, "metrics": result["metrics"], "checks": parent_checks}, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
