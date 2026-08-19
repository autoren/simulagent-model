#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v10_protocol import file_sha256
from v207r2_agentabstain_outcome_verification_repair import evaluate_repair
from v22r2_grounding import PROJECT_ROOT


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v207r2-agentabstain-outcome-verification-repair-lock.json"
    lock = json.loads(lock_path.read_text())
    result_path = PROJECT_ROOT / "outputs/v207r2-agentabstain-outcome-verification-repair/repair/result.json"
    audit_path = PROJECT_ROOT / "outputs/v207r2-agentabstain-outcome-verification-repair/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v207r2-agentabstain-outcome-verification-repair-outcome-lock.json"
    results_path = PROJECT_ROOT / "docs/v207r2-agentabstain-outcome-verification-repair-results.md"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V207r2 outcome exists")
    dependency_keys = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    exact = valid_lock(lock) and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependency_keys)
    source_lock = json.loads((PROJECT_ROOT / lock["source_V207r1_lock"]).read_text())
    repair = evaluate_repair(
        source_lock,
        json.loads((PROJECT_ROOT / lock["source_failed_outcome_audit"]).read_text()),
        json.loads((PROJECT_ROOT / lock["source_summary"]).read_text()),
        json.loads((PROJECT_ROOT / lock["source_result"]).read_text()),
        lock["config_payload"],
    )
    stored = json.loads(result_path.read_text())
    expected_decision = lock["config_payload"]["decisionRule"][
        "ifExactBookkeepingFailureAndEverySubstantiveV207r1CheckPasses" if repair["passed"] else "otherwise"
    ]
    zero_keys = (
        "source_artifact_mutation_count",
        "network_metadata_read_count",
        "scientific_evaluation_or_model_rerun_count",
        "task_language_read_count",
        "API_call_count",
        "tool_call_count",
        "actual_execution_count",
    )
    checks = {
        "repair_lock_and_dependencies_exact": exact,
        "repair_reconstructs_exactly": stored["repair"] == repair and stored["passed"] == repair["passed"] and stored["decision"] == expected_decision,
        "repair_passes": repair["passed"],
        "source_V207r1_dependencies_remain_exact": valid_lock(source_lock) and all(
            file_sha256(PROJECT_ROOT / source_lock[key]) == source_lock[f"{key}_sha256"]
            for key in source_lock
            if not key.endswith("_sha256") and f"{key}_sha256" in source_lock
        ),
        "results_document_exists": results_path.is_file(),
        "zero_mutation_network_rerun_language_API_tool_execution": all(stored[key] == 0 for key in zero_keys),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "207r2-agentabstain-outcome-verification-repair-outcome-audit",
        "experiment": lock["experiment"],
        "passed": passed,
        "decision": "freeze_verified_V207r2" if passed else "freeze_failed_V207r2",
        "checks": checks,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    source_summary = json.loads((PROJECT_ROOT / lock["source_summary"]).read_text())
    dependencies = {
        "repair_lock": lock_path,
        "audit": audit_path,
        "result": result_path,
        "source_failed_outcome_audit": PROJECT_ROOT / lock["source_failed_outcome_audit"],
        "source_summary": PROJECT_ROOT / lock["source_summary"],
        "source_result": PROJECT_ROOT / lock["source_result"],
        "source_results_document": PROJECT_ROOT / lock["source_results_document"],
        "results_document": results_path,
        "verifier": PROJECT_ROOT / lock["verifier"],
    }
    outcome: dict[str, Any] = {
        "schema_version": "207r2-agentabstain-outcome-verification-repair-outcome-lock",
        "experiment": lock["experiment"],
        "outcome": {
            "passed": True,
            "V207_scientific_outcome_available": False,
            "V207r1_scientific_feasibility_passed": False,
            "V207r1_transport_integrity_passed": True,
            "decision": "freeze_V207r1_scientific_negative_without_opening_task_payload_or_weakening_gates",
            "V207r1_summary": source_summary,
        },
        "authorization": {
            "update_roadmap_and_preregister_separate_F1_source_census_design": True,
            "AgentAbstain_task_text_or_model_run": False,
            "API_training_tool_registration_authority_action_or_execution": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"outcome_lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
