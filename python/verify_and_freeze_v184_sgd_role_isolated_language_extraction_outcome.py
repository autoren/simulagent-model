#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from run_v184_sgd_role_isolated_language_extraction import DEPENDENCY_KEYS, reconstruct


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v184-sgd-role-isolated-language-extraction-lock.json"
    output_root = PROJECT_ROOT / "outputs/v184-sgd-role-isolated-language-extraction/extraction"
    result_path = output_root / "result.json"
    doc_path = PROJECT_ROOT / "docs/v184-sgd-role-isolated-language-extraction-results.md"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v184_sgd_role_isolated_language_extraction_outcome.py"
    audit_path = PROJECT_ROOT / "outputs/v184-sgd-role-isolated-language-extraction/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v184-sgd-role-isolated-language-extraction-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V184 outcome is already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write V184 results before freezing")
    lock = json.loads(lock_path.read_text())
    result = json.loads(result_path.read_text())
    extraction, extraction_audit = reconstruct(lock)
    expected = {
        "development_language": extraction["development_language"],
        "protected_language": extraction["protected_language"],
        "declared_catalog_language": extraction["declared_catalog_language"],
        "extraction_summary": extraction["summary"],
    }
    outputs_exact = all(
        json.loads((PROJECT_ROOT / result["output_integrity"][key]["path"]).read_text()) == payload
        and file_sha256(PROJECT_ROOT / result["output_integrity"][key]["path"])
        == result["output_integrity"][key]["sha256"]
        for key, payload in expected.items()
    )
    config = lock["config_payload"]
    expected_decision = (
        config["decisionRule"]["ifEveryExtractionIsolationAndAccessGatePasses"]
        if extraction_audit["passed"]
        else config["decisionRule"]["otherwise"]
    )
    zero_keys = (
        "unselected_language_record_emission_count", "manual_development_language_inspection_count",
        "manual_protected_language_inspection_count", "protected_language_read_during_development_count",
        "policy_score_count", "model_load_count", "model_generation_count", "API_call_count",
        "training_run_count", "ontology_registration_count", "trusted_state_mutation_count",
        "service_call_count", "external_side_effect_count", "actual_execution_count",
    )
    checks = {
        "lock_and_dependencies_are_exact": bool(
            payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"})
            == lock["lock_payload_sha256"]
            and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in DEPENDENCY_KEYS)
        ),
        "all_extraction_outputs_reconstruct_exactly": outputs_exact,
        "summary_gates_and_decision_reconstruct_exactly": bool(
            result["summary"] == extraction["summary"]
            and result["extraction_gates"] == extraction_audit["checks"]
            and result["passed"] == extraction_audit["passed"]
            and result["decision"] == expected_decision
        ),
        "role_isolation_projection_and_catalog_boundary_hold": bool(
            extraction["summary"]["role_identifier_overlap"] == 0
            and extraction["summary"]["source_identifier_overlap"] == 0
            and extraction["summary"]["forbidden_field_occurrence_count"] == 0
            and extraction["summary"]["declared_known_choice_count"] == 6
            and not extraction["declared_catalog_language"]["contains_provisional_or_unsupported_schema_language"]
        ),
        "manual_protected_model_authority_and_effect_access_is_zero": all(result["access"][key] == 0 for key in zero_keys),
    }
    verified = all(checks.values())
    outcome_audit = {
        "schema_version": "184-sgd-role-isolated-language-extraction-outcome-audit",
        "experiment": config["experiment"],
        "passed": verified,
        "scientific_extraction_gates_passed": extraction_audit["passed"],
        "decision": "freeze_verified_V184_extraction" if verified else "reject_V184_outcome",
        "checks": checks,
        "independent_summary": extraction["summary"],
        "additional_access": {
            "source_archive_parse_count": 1,
            "manual_development_language_inspection_count": 0,
            "manual_protected_language_inspection_count": 0,
            "model_load_count": 0,
            "actual_execution_count": 0,
        },
    }
    write_json(audit_path, outcome_audit)
    if not verified:
        print(json.dumps(outcome_audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "extraction_lock": lock_path,
        "result": result_path,
        "verifier": verifier_path,
        "audit": audit_path,
        "results_document": doc_path,
        "parent_V183_outcome": PROJECT_ROOT / lock["parent_V183_outcome"],
    }
    for key, item in result["output_integrity"].items():
        dependencies[key] = PROJECT_ROOT / item["path"]
    outcome: dict[str, Any] = {
        "schema_version": "184-sgd-role-isolated-language-extraction-outcome-lock",
        "experiment": config["experiment"],
        "outcome": {
            "passed": True,
            "scientific_extraction_gates_passed": extraction_audit["passed"],
            "decision": result["decision"],
            "summary": result["summary"],
        },
        "authorization": {
            "modify_reextract_or_repartition_V184": False,
            "preregister_deterministic_interface_and_development_controls": bool(extraction_audit["passed"]),
            "score_development_language_without_separate_lock": False,
            "read_protected_language_during_development": False,
            "run_model_API_training_register_mutate_call_service_act_or_execute": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps(outcome_audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
