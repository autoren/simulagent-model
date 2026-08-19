#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, valid_lock, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v161_fresh_massive_transfer_population import (
    evaluate_population_gates,
    select_transfer_population,
)


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v161-fresh-massive-transfer-population-lock.json"
    result_path = PROJECT_ROOT / "outputs/v161-fresh-massive-transfer-population/population/result.json"
    population_path = PROJECT_ROOT / "outputs/v161-fresh-massive-transfer-population/population/selected-population.json"
    results_doc_path = PROJECT_ROOT / "docs/v161-fresh-massive-transfer-population-results.md"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v161_fresh_massive_transfer_population_outcome.py"
    audit_path = PROJECT_ROOT / "outputs/v161-fresh-massive-transfer-population/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v161-fresh-massive-transfer-population-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V161 outcome already frozen")
    if not results_doc_path.is_file():
        raise RuntimeError("write V161 results document before outcome freeze")
    lock = json.loads(lock_path.read_text())
    result = json.loads(result_path.read_text())
    artifact = json.loads(population_path.read_text())
    config = lock["config_payload"]
    inventory = json.loads((PROJECT_ROOT / lock["source_inventory"]).read_text())
    exclusion = json.loads((PROJECT_ROOT / lock["excluded_population"]).read_text())
    reconstructed = select_transfer_population(inventory, exclusion, config)
    reconstructed_gates = evaluate_population_gates(reconstructed, config)
    reconstructed_gates["zero_archive_language_manual_model_API_training_or_execution_access"] = True
    rows = artifact["selected_population"]
    role_class_counts: dict[str, Counter[str]] = defaultdict(Counter)
    role_class_scenarios: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    role_class_intents: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        role_class_counts[row["role"]][row["class_label"]] += 1
        role_class_scenarios[row["role"]][row["class_label"]].add(row["scenario"])
        role_class_intents[row["role"]][row["class_label"]].add(row["intent"])
    dependencies = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    exclusion_ids = {row["candidate_id"] for row in exclusion["selected_population"]}
    selected_ids = {row["candidate_id"] for row in rows}
    checks = {
        "population_lock_and_dependencies_are_exact": bool(
            valid_lock(lock)
            and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependencies)
        ),
        "source_and_exclusion_identities_are_exact": bool(
            file_sha256(PROJECT_ROOT / lock["source_inventory"]) == config["sourceInventorySha256"]
            and inventory["candidate_index_sha256"] == config["sourceCandidateIndexSha256"]
            and file_sha256(PROJECT_ROOT / lock["excluded_population"])
            == config["excludedPopulationFileSha256"]
            and exclusion["selected_population_sha256"] == config["excludedPopulationPayloadSha256"]
        ),
        "population_reconstructs_exactly": reconstructed
        == {key: value for key, value in artifact.items() if key != "provenance"},
        "counts_and_coverage_reconstruct": bool(
            {
                role: dict(sorted(counts.items()))
                for role, counts in sorted(role_class_counts.items())
            }
            == artifact["role_class_counts"]
            and {
                role: {label: len(values) for label, values in sorted(classes.items())}
                for role, classes in sorted(role_class_scenarios.items())
            }
            == artifact["role_class_scenario_counts"]
            and {
                role: {label: len(values) for label, values in sorted(classes.items())}
                for role, classes in sorted(role_class_intents.items())
            }
            == artifact["role_class_intent_counts"]
        ),
        "identifiers_unique_roles_disjoint_and_V101_overlap_zero": bool(
            len({row["population_id"] for row in rows}) == len(rows)
            and len(selected_ids) == len(rows)
            and artifact["role_identifiers_are_disjoint"]
            and not (selected_ids & exclusion_ids)
            and artifact["excluded_population_overlap_count"] == 0
        ),
        "result_gates_and_decision_are_consistent": bool(
            reconstructed_gates == result["gates"]
            and result["passed"] == all(result["gates"].values())
            and result["decision"]
            == (
                config["decisionRule"]["ifEveryPopulationAndDisjointnessGatePasses"]
                if result["passed"]
                else config["decisionRule"]["otherwise"]
            )
        ),
        "text_free_and_zero_access_boundary_holds": bool(
            not artifact["contains_language_tokens_slot_values_or_prompts"]
            and not artifact["provenance"]["contains_source_or_derived_language"]
            and all(
                result["access"][key] == 0
                for key in (
                    "source_archive_reopen_count",
                    "selected_language_record_extraction_count",
                    "emitted_language_record_count",
                    "manual_utterance_inspection_count",
                    "interface_policy_score_count",
                    "model_load_count",
                    "model_generation_count",
                    "LLM_API_call_count",
                    "training_run_count",
                    "real_service_call_count",
                    "external_side_effect_count",
                    "actual_execution_count",
                )
            )
        ),
        "authorization_remains_narrow": bool(
            lock["authorization"]["select_and_emit_one_text_free_fresh_transfer_population"]
            and not lock["authorization"]["modify_salt_exclusions_quotas_roles_or_gates"]
            and not lock["authorization"]["reopen_source_archive_or_extract_selected_language"]
            and not lock["authorization"]["read_protected_transfer_language"]
            and not lock["authorization"]["run_interface_policy_model_hybrid_API_training_induction_authority_action_or_execution"]
        ),
    }
    passed = all(checks.values())
    decision = (
        "freeze_positive_V161_fresh_MASSIVE_transfer_population"
        if result["passed"]
        else "freeze_negative_V161_fresh_MASSIVE_transfer_population"
    ) if passed else "reject_V161_transfer_population_outcome"
    audit = {
        "schema_version": "161-fresh-massive-transfer-population-outcome-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "scientific_population_feasibility_passed": result["passed"],
        "decision": decision,
        "checks": checks,
        "independent_summary": {
            "selected_candidate_count": len(rows),
            "excluded_population_overlap_count": len(selected_ids & exclusion_ids),
            "role_class_counts": {
                role: dict(sorted(counts.items()))
                for role, counts in sorted(role_class_counts.items())
            },
            "role_class_scenario_counts": artifact["role_class_scenario_counts"],
            "role_class_intent_counts": artifact["role_class_intent_counts"],
            "failed_gates": sorted(key for key, value in result["gates"].items() if not value),
        },
        "access": result["access"],
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    paths = {
        "population_lock": lock_path,
        "result": result_path,
        "population": population_path,
        "source_inventory": PROJECT_ROOT / lock["source_inventory"],
        "excluded_population": PROJECT_ROOT / lock["excluded_population"],
        "verifier": verifier_path,
        "audit": audit_path,
        "results_document": results_doc_path,
    }
    outcome: dict[str, Any] = {
        "schema_version": "161-fresh-massive-transfer-population-outcome-lock",
        "experiment": config["experiment"],
        "outcome": {
            "passed": True,
            "scientific_population_feasibility_passed": result["passed"],
            "decision": decision,
            "population_summary": result["population_summary"],
        },
        "authorization": {
            "preregister_automatic_selected_language_extraction": result["passed"],
            "modify_or_rerun_V161_population": False,
            "reopen_archive_or_extract_language_before_extraction_lock": False,
            "read_protected_transfer_language_during_development": False,
            "manually_inspect_selected_language": False,
            "run_interface_policy_or_model_before_separate_locks": False,
            "run_API_training_induction_authority_action_or_execution": False,
        },
    }
    for key, path in paths.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["selected_population_payload_sha256"] = artifact["selected_population_sha256"]
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
