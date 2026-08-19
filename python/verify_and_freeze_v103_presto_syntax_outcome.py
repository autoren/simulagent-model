#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v102r1_presto_context_source import parse_presto_archive
from v103_presto_target_syntax_census import build_target_syntax_census, evaluate_target_syntax_gates


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    census_lock_path = PROJECT_ROOT / "configs/v103-presto-target-syntax-census-lock.json"
    result_path = PROJECT_ROOT / "outputs/v103-presto-target-syntax/census/result.json"
    census_path = PROJECT_ROOT / "outputs/v103-presto-target-syntax/census/target-syntax-census.json"
    doc_path = PROJECT_ROOT / "docs/v103-presto-target-syntax-census-results.md"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v103_presto_syntax_outcome.py"
    audit_path = PROJECT_ROOT / "outputs/v103-presto-target-syntax/census-outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v103-presto-target-syntax-census-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V103 census outcome is already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write the V103 census result before freezing")
    lock = json.loads(census_lock_path.read_text())
    result = json.loads(result_path.read_text())
    artifact = json.loads(census_path.read_text())
    config = lock["config_payload"]
    scientific = json.loads((PROJECT_ROOT / config["unchangedScientificConfig"]).read_text())
    source_path = PROJECT_ROOT / config["sourceArchive"]
    source_bytes = source_path.read_bytes()
    records, members = parse_presto_archive(source_bytes, scientific["archive"]["requiredMemberBasenames"])
    reconstructed = build_target_syntax_census(records, scientific, config)
    reconstructed_gates = evaluate_target_syntax_gates(reconstructed, config)
    reconstructed_gates["zero_language_identifier_manual_model_API_training_service_or_side_effect_access"] = True
    dependency_keys = (
        "config", "parent_source_outcome", "source_archive", "scientific_config", "plan",
        "protocol", "tests", "runner", "verifier", "auditor", "design_audit",
    )
    checks = {
        "census_lock_and_dependencies_are_exact": bool(
            payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"})
            == lock["lock_payload_sha256"]
            and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependency_keys)
        ),
        "source_archive_identity_is_exact": file_sha256(source_path) == config["sourceArchiveSha256"],
        "census_reconstructs_exactly": reconstructed == {key: value for key, value in artifact.items() if key != "provenance"},
        "result_gates_and_decision_are_consistent": bool(
            reconstructed_gates == result["gates"]
            and result["passed"] == all(result["gates"].values())
            and result["decision"] == (
                "preregister_new_PRESTO_literal_family_dependency_construction"
                if result["passed"] else "close_PRESTO_paired_insufficiency_branch"
            )
        ),
        "text_identifier_and_zero_access_boundary_holds": bool(
            artifact["emitted_candidate_identifier_count"] == 0
            and not artifact["contains_input_target_literal_context_tokens_identifiers_or_root_names"]
            and not artifact["provenance"]["contains_language_literals_identifiers_or_root_names"]
            and all(result["access"][key] == 0 for key in (
                "emitted_language_record_count", "emitted_candidate_identifier_count",
                "manual_utterance_inspection_count", "model_load_count", "model_generation_count",
                "LLM_API_call_count", "adapter_training_run_count", "real_service_call_count",
                "external_side_effect_count",
            ))
        ),
    }
    integrity_passed = all(checks.values())
    audit = {
        "schema_version": "103-presto-target-syntax-census-outcome-audit",
        "experiment": "v103_presto_target_syntax_census_outcome_audit",
        "passed": integrity_passed,
        "diagnostic_viability_passed": result["passed"],
        "decision": (
            "freeze_positive_V103_PRESTO_syntax_viability"
            if result["passed"] else "freeze_negative_V103_close_PRESTO_paired_branch"
        ) if integrity_passed else "reject_V103_census_outcome",
        "checks": checks,
        "independent_summary": {
            "literal_family_stage_record_counts": reconstructed["literal_family_stage_record_counts"],
            "structural_character_record_counts": reconstructed["structural_character_record_counts"],
            "union_candidate_count": reconstructed["candidate_eligible_family_union_count"],
            "union_role_counts": reconstructed["candidate_eligible_family_union_role_counts"],
            "failed_gates": sorted(key for key, value in result["gates"].items() if not value),
        },
        "additional_access": {
            "manual_utterance_inspection_count": 0, "model_load_count": 0,
            "model_generation_count": 0, "LLM_API_call_count": 0,
            "adapter_training_run_count": 0, "real_service_call_count": 0,
            "external_side_effect_count": 0,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not integrity_passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "census_lock": census_lock_path, "result": result_path, "census": census_path,
        "source_archive": source_path, "verifier": verifier_path,
        "audit": audit_path, "results_document": doc_path,
    }
    outcome: dict[str, Any] = {
        "schema_version": "103-presto-target-syntax-census-outcome-lock",
        "experiment": "v103_presto_target_syntax_census_outcome_lock",
        "outcome": {
            "passed": True, "diagnostic_viability_passed": result["passed"],
            "decision": audit["decision"], "census_summary": result["census_summary"],
        },
        "authorization": {
            "modify_or_rerun_V103_census": False,
            "preregister_new_PRESTO_dependency_construction": result["passed"],
            "close_PRESTO_paired_insufficiency_branch": not result["passed"],
            "select_population_or_extract_language": False,
            "load_local_or_API_model": False,
            "grant_model_capability_state_belief_action_or_execution_authority": False,
            "perform_real_service_call_or_external_side_effect": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    outcome_path.write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
