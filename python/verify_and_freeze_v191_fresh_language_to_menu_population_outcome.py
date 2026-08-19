#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v10_protocol import file_sha256
from v191_fresh_language_to_menu_population import audit_population, build_population
from v22r2_grounding import PROJECT_ROOT


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v191-fresh-language-to-menu-population-lock.json"
    lock = json.loads(lock_path.read_text())
    output_root = PROJECT_ROOT / "outputs/v191-fresh-language-to-menu-population/population"
    audit_path = PROJECT_ROOT / "outputs/v191-fresh-language-to-menu-population/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v191-fresh-language-to-menu-population-outcome-lock.json"
    results_path = PROJECT_ROOT / "docs/v191-fresh-language-to-menu-population-results.md"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V191 outcome already verified or frozen")

    dependencies_exact = valid_lock(lock) and all(
        file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"]
        for key in (
            "config",
            "parent_V190_outcome",
            "source_V124_outcome",
            "source_V183_outcome",
            "source_inventory",
            "contract_catalog",
            "previous_hidden_population",
            "protocol",
            "tests",
            "runner",
            "verifier",
            "auditor",
            "design_audit",
        )
    )
    rebuilt = build_population(
        json.loads((PROJECT_ROOT / lock["source_inventory"]).read_text()),
        json.loads((PROJECT_ROOT / lock["contract_catalog"]).read_text()),
        json.loads((PROJECT_ROOT / lock["previous_hidden_population"]).read_text()),
        lock["config_payload"],
    )
    expected = {
        "availability-census.json": rebuilt["availability_census"],
        "fresh-development-identities.json": rebuilt["public_identities"],
        "hidden-targets.json": rebuilt["hidden_targets"],
        "population-summary.json": rebuilt["summary"],
    }
    artifacts_exact = all(
        (output_root / name).is_file()
        and json.loads((output_root / name).read_text()) == value
        for name, value in expected.items()
    )
    result_path = output_root / "result.json"
    result = json.loads(result_path.read_text())
    audit = audit_population(rebuilt, lock["config_payload"])
    result_exact = bool(
        result["passed"] == audit["passed"]
        and result["checks"] == audit["checks"]
        and result["summary"] == audit["summary"]
        and result["decision"]
        == lock["config_payload"]["decisionRule"]["ifEveryFreshnessAvailabilityPopulationAndAccessGatePasses"]
    )
    checks = {
        "design_lock_and_dependencies_are_exact": dependencies_exact,
        "all_population_artifacts_reconstruct_exactly": artifacts_exact,
        "result_reconstructs_exactly": result_exact,
        "all_population_gates_pass": bool(audit["passed"] and result["passed"]),
        "results_document_exists": results_path.is_file(),
        "language_model_authority_and_execution_remain_zero": bool(
            result["summary"]["persisted_utterance_or_dialogue_text_count"] == 0
            and result["summary"]["protected_language_read_count"] == 0
            and result["summary"]["model_load_count"] == 0
            and result["summary"]["API_call_count"] == 0
            and result["summary"]["ontology_registration_count"] == 0
            and result["summary"]["actual_execution_count"] == 0
        ),
    }
    passed = all(checks.values())
    outcome_audit = {
        "schema_version": "191-fresh-language-to-menu-population-outcome-audit",
        "experiment": lock["experiment"],
        "passed": passed,
        "population_gates_passed": bool(audit["passed"]),
        "decision": "freeze_verified_V191_population" if passed else "freeze_failed_V191_verification",
        "checks": checks,
        "summary": result["summary"],
    }
    write_json(audit_path, outcome_audit)
    if not passed:
        print(json.dumps(outcome_audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "population_lock": lock_path,
        "audit": audit_path,
        "availability_census": output_root / "availability-census.json",
        "development_identities": output_root / "fresh-development-identities.json",
        "hidden_targets": output_root / "hidden-targets.json",
        "population_summary": output_root / "population-summary.json",
        "result": result_path,
        "results_document": results_path,
        "verifier": PROJECT_ROOT / lock["verifier"],
    }
    outcome: dict[str, Any] = {
        "schema_version": "191-fresh-language-to-menu-population-outcome-lock",
        "experiment": lock["experiment"],
        "outcome": {
            "passed": True,
            "population_gates_passed": True,
            "decision": "freeze_V191_fresh_population_and_allow_extraction_preregistration_only",
            "summary": result["summary"],
        },
        "authorization": {
            "preregister_separate_exact_development_language_extraction_only": True,
            "immediate_language_extraction_interface_scoring_or_model_run": False,
            "read_protected_language_or_run_model_API_training": False,
            "register_prune_mutate_call_service_act_or_execute": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps(outcome_audit, indent=2, sort_keys=True))
    print(json.dumps({"outcome_lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
