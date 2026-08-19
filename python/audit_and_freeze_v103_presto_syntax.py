#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def valid_lock(payload: dict[str, Any]) -> bool:
    return payload_hash(
        {key: value for key, value in payload.items() if key != "lock_payload_sha256"}
    ) == payload.get("lock_payload_sha256")


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v103-presto-target-syntax-census.json"
    parent_path = PROJECT_ROOT / "configs/v102r1-presto-context-source-outcome-lock.json"
    plan_path = PROJECT_ROOT / "docs/v103-presto-target-syntax-census-plan.md"
    protocol_path = PROJECT_ROOT / "python/v103_presto_target_syntax_census.py"
    tests_path = PROJECT_ROOT / "python/test_v103_presto_target_syntax_census.py"
    runner_path = PROJECT_ROOT / "python/run_v103_presto_target_syntax_census.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v103_presto_syntax_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v103_presto_syntax.py"
    audit_path = PROJECT_ROOT / "outputs/v103-presto-target-syntax/census-design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v103-presto-target-syntax-census-lock.json"
    census_root = PROJECT_ROOT / "outputs/v103-presto-target-syntax/census"
    if audit_path.exists() or lock_path.exists() or census_root.exists():
        raise RuntimeError("V103 syntax census is already frozen or materialized")

    config = json.loads(config_path.read_text())
    parent = json.loads(parent_path.read_text())
    source_path = PROJECT_ROOT / config["sourceArchive"]
    scientific_path = PROJECT_ROOT / config["unchangedScientificConfig"]
    gates = config["diagnosticGates"]
    exposure = config["preLockExposure"]
    checks = {
        "V102r1_negative_source_outcome_is_exact": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and not parent["outcome"]["scientific_source_feasibility_passed"]
            and parent["outcome"]["inventory_summary"]["eligible_candidate_count"] == 0
            and not parent["authorization"]["preregister_paired_development_and_protected_test_population"]
        ),
        "persisted_archive_and_scientific_config_are_exact": bool(
            source_path.is_file()
            and file_sha256(source_path) == config["sourceArchiveSha256"]
            and file_sha256(scientific_path) == config["unchangedScientificConfigSha256"]
        ),
        "literal_families_and_diagnostic_stages_are_frozen": bool(
            config["literalFamilies"] == [
                "guillemet", "single_guillemet", "ascii_double_quote", "curly_double_quote",
                "ascii_single_quote", "square_bracket",
            ]
            and config["candidateEligibleLiteralFamilies"] == [
                "guillemet", "single_guillemet", "ascii_double_quote", "curly_double_quote",
                "ascii_single_quote",
            ]
            and len(config["diagnosticStages"]) == 5
            and "square_bracket" not in config["candidateEligibleLiteralFamilies"]
        ),
        "union_viability_gates_preserve_V102_scientific_minima": bool(
            gates["minimumUnionDevelopmentCandidateCount"] >= 64
            and gates["minimumUnionProtectedTestCandidateCount"] >= 64
            and gates["minimumUnionTotalCandidateCount"] >= 256
            and gates["minimumUnionPreviousTurnDependentCandidateCount"] >= 64
            and gates["minimumUnionSeededStateDependentCandidateCount"] >= 64
            and gates["minimumUnionDependencySourceKindCount"] >= 2
            and gates["minimumUnionSemanticRootFunctionCount"] >= 8
        ),
        "diagnostic_is_text_identifier_model_and_side_effect_free": bool(
            all(value == 0 for value in exposure.values())
            and all(
                gates[key] == 0
                for key in (
                    "maximumEmittedLanguageRecordCount", "maximumEmittedCandidateIdentifierCount",
                    "maximumManualUtteranceInspectionCount", "maximumModelLoadCount",
                    "maximumModelGenerationCount", "maximumLLMAPICallCount",
                    "maximumAdapterTrainingRunCount", "maximumRealServiceCallCount",
                    "maximumExternalSideEffectCount",
                )
            )
            and not config["decisionRule"]["diagnosticMaySelectPopulationOrEmitIdentifiers"]
            and not config["decisionRule"]["diagnosticMayEmitLanguageOrAuthorizeModelInference"]
        ),
        "plan_and_locked_code_exist": all(
            path.is_file()
            for path in (plan_path, protocol_path, tests_path, runner_path, verifier_path, auditor_path)
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "103-presto-target-syntax-census-design-audit",
        "experiment": "v103_presto_target_syntax_census_design_audit",
        "passed": passed,
        "decision": "freeze_and_authorize_one_text_free_syntax_census" if passed else "reject_V103_census",
        "checks": checks,
        "prelock_access": exposure,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "config": config_path, "parent_source_outcome": parent_path,
        "source_archive": source_path, "scientific_config": scientific_path,
        "plan": plan_path, "protocol": protocol_path, "tests": tests_path,
        "runner": runner_path, "verifier": verifier_path, "auditor": auditor_path,
        "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "103-presto-target-syntax-census-lock",
        "experiment": "v103_presto_target_syntax_census_lock",
        "config_payload": config,
        "authorization": {
            "modify_families_stages_or_gates": False,
            "read_persisted_archive_and_emit_aggregate_census_once": True,
            "emit_language_literals_identifiers_or_root_names": False,
            "select_population_or_extract_selected_language": False,
            "load_local_or_API_model": False,
            "train_adapter_or_learn_likelihood": False,
            "grant_model_capability_state_belief_action_or_execution_authority": False,
            "perform_real_service_call_or_external_side_effect": False,
        },
    }
    for key, path in dependencies.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT))
        lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
