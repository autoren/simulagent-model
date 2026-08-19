#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v100_massive_source import parse_massive_archive
from v162_fresh_massive_transfer_language_extraction import (
    build_selected_language_artifacts,
    evaluate_extraction_gates,
)


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    extraction_lock_path = (
        PROJECT_ROOT
        / "configs/v162-fresh-massive-transfer-language-extraction-lock.json"
    )
    result_path = (
        PROJECT_ROOT
        / "outputs/v162-fresh-massive-transfer-language/selected-language/result.json"
    )
    doc_path = (
        PROJECT_ROOT
        / "docs/v162-fresh-massive-transfer-language-extraction-results.md"
    )
    verifier_path = (
        PROJECT_ROOT
        / "python/verify_and_freeze_v162_fresh_massive_transfer_language_outcome.py"
    )
    audit_path = (
        PROJECT_ROOT
        / "outputs/v162-fresh-massive-transfer-language/language-outcome-audit.json"
    )
    outcome_path = (
        PROJECT_ROOT
        / "configs/v162-fresh-massive-transfer-language-extraction-outcome-lock.json"
    )
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V162 extraction outcome is already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write the V162 extraction result before freezing")

    lock = json.loads(extraction_lock_path.read_text())
    result = json.loads(result_path.read_text())
    config = lock["config_payload"]
    population = json.loads((PROJECT_ROOT / config["selectedPopulation"]).read_text())
    inventory = json.loads((PROJECT_ROOT / config["sourceInventory"]).read_text())
    archive_bytes = (PROJECT_ROOT / config["sourceArchive"]).read_bytes()
    source_records, member = parse_massive_archive(
        archive_bytes, config["expectedLocaleMemberSuffix"]
    )
    reconstructed = build_selected_language_artifacts(
        population, inventory, source_records, config
    )
    reconstructed_gates = evaluate_extraction_gates(reconstructed, config)
    reconstructed_gates[
        "zero_manual_model_API_training_service_side_effect_or_execution_access"
    ] = True

    dependency_keys = (
        "config",
        "parent_population_outcome",
        "selected_population",
        "source_inventory",
        "source_archive",
        "plan",
        "roadmap",
        "protocol",
        "tests",
        "runner",
        "verifier",
        "auditor",
        "design_audit",
    )
    output_exact = True
    for role, integrity in result["output_integrity"].items():
        path = PROJECT_ROOT / integrity["path"]
        actual_rows = read_jsonl(path)
        output_exact = output_exact and bool(
            file_sha256(path) == integrity["file_sha256"]
            and actual_rows == reconstructed["role_records"][role]
            and len(actual_rows) == integrity["record_count"]
            and reconstructed["role_payload_sha256"][role]
            == integrity["payload_sha256"]
        )

    zero_access_keys = (
        "unselected_language_record_emission_count",
        "manual_development_utterance_inspection_count",
        "manual_protected_utterance_inspection_count",
        "protected_language_read_during_development_count",
        "model_load_count",
        "model_generation_count",
        "LLM_API_call_count",
        "adapter_training_run_count",
        "real_service_call_count",
        "external_side_effect_count",
        "actual_execution_count",
    )
    checks = {
        "extraction_lock_and_dependencies_are_exact": bool(
            payload_hash(
                {
                    key: value
                    for key, value in lock.items()
                    if key != "lock_payload_sha256"
                }
            )
            == lock["lock_payload_sha256"]
            and all(
                file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"]
                for key in dependency_keys
            )
        ),
        "source_member_and_role_artifacts_reconstruct_exactly": bool(
            output_exact and member == result["source_locale_member"]
        ),
        "result_gates_and_decision_are_consistent": bool(
            reconstructed_gates == result["gates"]
            and result["passed"] == all(result["gates"].values())
            and result["decision"]
            == (
                "freeze_language_and_preregister_deterministic_development_interface"
                if result["passed"]
                else "stop_V162_before_interface_or_model_access"
            )
        ),
        "selection_structure_familiarity_slots_and_disjointness_hold": all(
            reconstructed[key]
            for key in (
                "exact_selected_identifier_set",
                "exact_structural_ground_truth_match",
                "exact_familiarity_reconstruction",
                "exact_slot_type_count_reconstruction",
                "development_protected_role_disjoint",
            )
        ),
        "manual_protected_model_and_side_effect_boundaries_hold": all(
            result["access"][key] == 0 for key in zero_access_keys
        ),
    }
    integrity_passed = all(checks.values())
    audit = {
        "schema_version": "162-fresh-massive-transfer-language-outcome-audit",
        "experiment": "v162_fresh_massive_transfer_language_outcome_audit",
        "passed": integrity_passed,
        "scientific_extraction_passed": result["passed"],
        "decision": (
            "freeze_positive_V162_role_separated_language_artifacts"
            if integrity_passed and result["passed"]
            else "reject_V162_language_outcome"
        ),
        "checks": checks,
        "independent_summary": result["extraction_summary"],
        "additional_access": {
            "automatic_source_archive_reconstruction_count": 1,
            "manual_development_utterance_inspection_count": 0,
            "manual_protected_utterance_inspection_count": 0,
            "protected_language_read_during_development_count": 0,
            "model_load_count": 0,
            "model_generation_count": 0,
            "LLM_API_call_count": 0,
            "adapter_training_run_count": 0,
            "real_service_call_count": 0,
            "external_side_effect_count": 0,
            "actual_execution_count": 0,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not integrity_passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "extraction_lock": extraction_lock_path,
        "result": result_path,
        "verifier": verifier_path,
        "audit": audit_path,
        "results_document": doc_path,
    }
    for role, integrity in result["output_integrity"].items():
        dependencies[f"{role}_language"] = PROJECT_ROOT / integrity["path"]
    outcome: dict[str, Any] = {
        "schema_version": "162-fresh-massive-transfer-language-outcome-lock",
        "experiment": "v162_fresh_massive_transfer_language_outcome_lock",
        "outcome": {
            "passed": True,
            "scientific_extraction_passed": result["passed"],
            "decision": audit["decision"],
            "extraction_summary": result["extraction_summary"],
            "output_integrity": result["output_integrity"],
        },
        "authorization": {
            "modify_or_rerun_V162_extraction": False,
            "preregister_deterministic_development_interface_controls_metrics_and_gates": result[
                "passed"
            ],
            "automatically_read_development_language_after_interface_lock": result[
                "passed"
            ],
            "read_protected_transfer_during_development": False,
            "manually_inspect_protected_transfer_language": False,
            "load_model_before_deterministic_baseline_outcome": False,
            "run_API_model_or_train_adapter": False,
            "induce_or_register_ontology": False,
            "grant_model_capability_state_belief_action_or_execution_authority": False,
            "perform_real_service_call_external_side_effect_or_execution": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    outcome_path.write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                "lock": str(outcome_path.relative_to(PROJECT_ROOT)),
                "sha256": file_sha256(outcome_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
