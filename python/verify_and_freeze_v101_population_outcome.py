#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v101_massive_population import evaluate_population_gates, select_massive_population


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    population_lock_path = PROJECT_ROOT / "configs/v101-massive-population-lock.json"
    result_path = PROJECT_ROOT / "outputs/v101-massive-population/population/result.json"
    population_path = PROJECT_ROOT / "outputs/v101-massive-population/population/selected-population.json"
    doc_path = PROJECT_ROOT / "docs/v101-massive-population-results.md"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v101_population_outcome.py"
    audit_path = PROJECT_ROOT / "outputs/v101-massive-population/population-outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v101-massive-population-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V101 population outcome is already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write the V101 population result before freezing")

    lock = json.loads(population_lock_path.read_text())
    result = json.loads(result_path.read_text())
    artifact = json.loads(population_path.read_text())
    config = lock["config_payload"]
    inventory_path = PROJECT_ROOT / config["sourceInventory"]
    inventory = json.loads(inventory_path.read_text())
    reconstructed = select_massive_population(inventory, config)
    reconstructed_gates = evaluate_population_gates(reconstructed, config)
    reconstructed_gates["zero_archive_language_manual_model_API_training_or_side_effect_access"] = True
    rows = artifact["selected_population"]
    role_class_counts: dict[str, Counter[str]] = defaultdict(Counter)
    role_class_scenarios: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    role_class_intents: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        role_class_counts[row["role"]][row["class_label"]] += 1
        role_class_scenarios[row["role"]][row["class_label"]].add(row["scenario"])
        role_class_intents[row["role"]][row["class_label"]].add(row["intent"])
    dependency_keys = (
        "config", "parent_source_outcome", "source_inventory", "plan", "protocol",
        "tests", "runner", "verifier", "auditor", "design_audit",
    )
    checks = {
        "population_lock_and_dependencies_are_exact": bool(
            payload_hash(
                {key: value for key, value in lock.items() if key != "lock_payload_sha256"}
            ) == lock["lock_payload_sha256"]
            and all(
                file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"]
                for key in dependency_keys
            )
        ),
        "source_inventory_identity_is_exact": bool(
            file_sha256(inventory_path) == config["sourceInventorySha256"]
            and inventory["candidate_index_sha256"] == config["sourceCandidateIndexSha256"]
        ),
        "population_reconstructs_exactly": (
            reconstructed == {key: value for key, value in artifact.items() if key != "provenance"}
        ),
        "counts_and_coverage_reconstruct": bool(
            {
                role: dict(sorted(counts.items()))
                for role, counts in sorted(role_class_counts.items())
            } == artifact["role_class_counts"]
            and {
                role: {label: len(values) for label, values in sorted(classes.items())}
                for role, classes in sorted(role_class_scenarios.items())
            } == artifact["role_class_scenario_counts"]
            and {
                role: {label: len(values) for label, values in sorted(classes.items())}
                for role, classes in sorted(role_class_intents.items())
            } == artifact["role_class_intent_counts"]
        ),
        "identifiers_are_unique_and_splits_are_disjoint": bool(
            len({row["population_id"] for row in rows}) == len(rows)
            and len({row["candidate_id"] for row in rows}) == len(rows)
            and artifact["development_test_identifiers_are_disjoint"]
        ),
        "result_gates_and_decision_are_consistent": bool(
            reconstructed_gates == result["gates"]
            and result["passed"] == all(result["gates"].values())
            and result["decision"] == (
                "freeze_population_and_preregister_selected_language_extraction"
                if result["passed"] else "stop_V101_before_language_or_model_access"
            )
        ),
        "text_free_and_zero_access_boundary_holds": bool(
            not artifact["contains_language_tokens_slot_values_or_prompts"]
            and not artifact["provenance"]["contains_source_or_derived_language"]
            and all(
                result["access"][key] == 0
                for key in (
                    "source_archive_reopen_count", "selected_language_record_extraction_count",
                    "emitted_language_record_count", "manual_utterance_inspection_count",
                    "model_load_count", "model_generation_count", "LLM_API_call_count",
                    "adapter_training_run_count", "real_service_call_count",
                    "external_side_effect_count",
                )
            )
        ),
    }
    integrity_passed = all(checks.values())
    failed_gates = sorted(key for key, value in result["gates"].items() if not value)
    audit = {
        "schema_version": "101-massive-population-outcome-audit",
        "experiment": "v101_massive_population_outcome_audit",
        "passed": integrity_passed,
        "scientific_population_feasibility_passed": result["passed"],
        "decision": (
            "freeze_positive_V101_MASSIVE_population"
            if result["passed"] else "freeze_negative_V101_MASSIVE_population"
        ) if integrity_passed else "reject_V101_population_outcome",
        "checks": checks,
        "independent_summary": {
            "selected_candidate_count": len(rows),
            "role_class_counts": {
                role: dict(sorted(counts.items()))
                for role, counts in sorted(role_class_counts.items())
            },
            "role_class_scenario_counts": artifact["role_class_scenario_counts"],
            "role_class_intent_counts": artifact["role_class_intent_counts"],
            "failed_gates": failed_gates,
        },
        "additional_access": {
            "source_archive_reopen_count": 0,
            "selected_language_record_extraction_count": 0,
            "manual_utterance_inspection_count": 0,
            "model_load_count": 0,
            "model_generation_count": 0,
            "LLM_API_call_count": 0,
            "adapter_training_run_count": 0,
            "real_service_call_count": 0,
            "external_side_effect_count": 0,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not integrity_passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "population_lock": population_lock_path,
        "result": result_path,
        "population": population_path,
        "source_inventory": inventory_path,
        "verifier": verifier_path,
        "audit": audit_path,
        "results_document": doc_path,
    }
    outcome: dict[str, Any] = {
        "schema_version": "101-massive-population-outcome-lock",
        "experiment": "v101_massive_population_outcome_lock",
        "outcome": {
            "passed": True,
            "scientific_population_feasibility_passed": result["passed"],
            "decision": audit["decision"],
            "population_summary": result["population_summary"],
        },
        "authorization": {
            "modify_or_rerun_V101_population_stage": False,
            "preregister_selected_language_extraction": result["passed"],
            "reopen_archive_or_extract_language_before_extraction_lock": False,
            "manually_inspect_selected_language": False,
            "load_model_before_prompt_controls_metrics_and_gates_lock": False,
            "run_API_model_or_train_adapter": False,
            "grant_model_capability_state_belief_action_or_execution_authority": False,
            "perform_real_service_call_or_external_side_effect": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["selected_population_payload_sha256"] = artifact["selected_population_sha256"]
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    outcome_path.write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({
        "lock": str(outcome_path.relative_to(PROJECT_ROOT)),
        "sha256": file_sha256(outcome_path),
    }, indent=2))


if __name__ == "__main__":
    main()
