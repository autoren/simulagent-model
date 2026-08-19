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
    config_path = PROJECT_ROOT / "configs/v203-independent-confirmation-feasibility.json"
    plan_path = PROJECT_ROOT / "docs/v203-independent-confirmation-feasibility-plan.md"
    protocol_path = PROJECT_ROOT / "python/v203_independent_confirmation_feasibility.py"
    tests_path = PROJECT_ROOT / "python/test_v203_independent_confirmation_feasibility.py"
    runner_path = PROJECT_ROOT / "python/run_v203_independent_confirmation_feasibility.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v203_independent_confirmation_feasibility_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v203_independent_confirmation_feasibility.py"
    audit_path = PROJECT_ROOT / "outputs/v203-independent-confirmation-feasibility/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v203-independent-confirmation-feasibility-lock.json"
    output_root = PROJECT_ROOT / "outputs/v203-independent-confirmation-feasibility/evaluation"
    outcome_path = PROJECT_ROOT / "configs/v203-independent-confirmation-feasibility-outcome-lock.json"
    if any(path.exists() for path in (audit_path, lock_path, output_root, outcome_path)):
        raise RuntimeError("V203 is already preregistered, run, or frozen")

    config = json.loads(config_path.read_text())
    outcome_paths = {
        "parent_V202_outcome": PROJECT_ROOT / config["parentV202OutcomeLock"],
        "source_V124_outcome": PROJECT_ROOT / config["sourceV124OutcomeLock"],
        "source_V183_outcome": PROJECT_ROOT / config["sourceV183OutcomeLock"],
        "source_V191_outcome": PROJECT_ROOT / config["sourceV191OutcomeLock"],
    }
    outcomes = {key: json.loads(path.read_text()) for key, path in outcome_paths.items()}
    v87_path = PROJECT_ROOT / config["sourceV87DesignLock"]
    v87_lock = json.loads(v87_path.read_text())
    inputs = {
        "source_archive": PROJECT_ROOT / config["sourceArchive"],
        "source_inventory": PROJECT_ROOT / config["sourceInventory"],
        "contract_catalog": PROJECT_ROOT / config["contractCatalog"],
        "V183_consumed_population": PROJECT_ROOT / config["V183ConsumedPopulation"],
        "V191_consumed_population": PROJECT_ROOT / config["V191ConsumedPopulation"],
        "roadmap": PROJECT_ROOT / config["roadmap"],
    }
    expected_hashes = {
        "source_archive": outcomes["source_V124_outcome"]["source_archive_sha256"],
        "source_inventory": outcomes["source_V124_outcome"]["inventory_sha256"],
        "contract_catalog": outcomes["source_V183_outcome"]["contract_catalog_sha256"],
        "V183_consumed_population": outcomes["source_V183_outcome"]["hidden_identifiability_sha256"],
        "V191_consumed_population": outcomes["source_V191_outcome"]["hidden_targets_sha256"],
    }
    parent = outcomes["parent_V202_outcome"]
    checks = {
        "V202_is_valid_positive_and_authorizes_only_fresh_confirmation_design": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["outcome"]["scientific_selection_made"]
            and parent["authorization"]["preregister_new_fresh_confirmation_design_only"]
            and not parent["authorization"]["immediate_confirmation_or_protected_reuse"]
        ),
        "all_frozen_source_locks_are_valid": bool(
            all(valid_lock(outcome) for key, outcome in outcomes.items() if key != "parent_V202_outcome")
            and valid_lock(v87_lock)
        ),
        "archive_inventory_catalog_and_consumed_populations_are_exact": all(
            inputs[key].is_file() and file_sha256(inputs[key]) == expected_hashes[key]
            for key in expected_hashes
        ),
        "independence_and_exact_contract_gates_are_noncompensatory": bool(
            config["independenceContract"]["allowedPartitions"] == ["train", "test"]
            and config["independenceContract"]["forbiddenPartitions"] == ["dev"]
            and config["independenceContract"]["requiredContractCount"] == 14
            and config["independenceContract"]["minimumUniqueDialoguesPerContract"] == 6
            and config["exactContractIdentity"]["intentNameSimilarityIsInsufficient"]
            and config["exactContractIdentity"]["allCandidateMappingsMustBeSingleton"]
        ),
        "prelock_exact_census_language_model_API_training_and_execution_access_is_zero": bool(
            config["preLockExposure"]["formalExactCrossPartitionContractCensusCount"] == 0
            and all(
                config["preLockExposure"][key] == 0
                for key in (
                    "utteranceOrDialogueTextReadOrEmissionCount",
                    "manualLanguageInspectionCount",
                    "protectedLanguageReadCount",
                    "policyScoreCount",
                    "modelLoadCount",
                    "modelGenerationCount",
                    "APICallCount",
                    "trainingRunCount",
                    "actualExecutionCount",
                )
            )
        ),
        "population_language_model_and_authority_remain_separately_locked": bool(
            not config["decisionRule"]["passAuthorizesImmediatePopulationSelectionLanguageExtractionOrModelRun"]
            and not config["decisionRule"]["passAuthorizesProtectedReuseAPITrainingRegistrationAuthorityActionOrExecution"]
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
                    v87_path,
                    *outcome_paths.values(),
                    *inputs.values(),
                )
            )
            and not output_root.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "203-independent-confirmation-feasibility-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_V203_metadata_schema_census" if passed else "reject_V203_design",
        "checks": checks,
        "prelock_exposure": config["preLockExposure"],
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "config": config_path,
        **outcome_paths,
        "source_V87_design_lock": v87_path,
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
        "schema_version": "203-independent-confirmation-feasibility-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "modify_sources_contracts_exclusions_families_thresholds_or_decision": False,
            "run_exact_single_metadata_schema_feasibility_census": True,
            "select_population_read_utterances_or_run_model": False,
            "protected_API_training_registration_authority_action_or_execution": False,
        },
    }
    for key, path in dependencies.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT))
        lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock)
    write_json(lock_path, lock)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
