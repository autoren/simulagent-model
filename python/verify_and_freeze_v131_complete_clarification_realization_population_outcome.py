#!/usr/bin/env python3
import json
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v131_complete_clarification_realization_population import (
    build_catalog, evaluate_gates, excluded_identifiers, select_population,
)


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v131-complete-clarification-realization-population-lock.json"
    doc_path = PROJECT_ROOT / "docs/v131-complete-clarification-realization-population-results.md"
    audit_path = PROJECT_ROOT / "outputs/v131-complete-clarification-realization-population/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v131-complete-clarification-realization-population-outcome-lock.json"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v131_complete_clarification_realization_population_outcome.py"
    if audit_path.exists() or outcome_path.exists(): raise RuntimeError("V131 outcome already frozen")
    if not doc_path.is_file(): raise RuntimeError("write V131 results first")
    lock = json.loads(lock_path.read_text()); config = lock["config_payload"]
    inventory = json.loads((PROJECT_ROOT / lock["source_inventory"]).read_text())
    catalog = json.loads((PROJECT_ROOT / lock["choice_catalog"]).read_text())
    population = json.loads((PROJECT_ROOT / lock["fixture_population"]).read_text())
    excluded = excluded_identifiers(config["excludedPopulations"], PROJECT_ROOT)
    expected_catalog = build_catalog(inventory, config)
    expected_population = select_population(inventory, expected_catalog, excluded, config)
    gates = evaluate_gates(catalog, population, config)
    dependencies = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    checks = {
        "lock_and_dependencies_exact": bool(payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) == lock["lock_payload_sha256"] and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependencies)),
        "catalog_exact": catalog == expected_catalog,
        "population_exact": population == expected_population,
        "all_population_gates_pass": all(gates.values()),
        "zero_language_model_and_execution": not catalog["contains_language"] and not population["contains_language"] and config["populationGates"]["maximumActualExecutionCount"] == 0,
    }
    passed = all(checks.values())
    audit = {"schema_version": "131-complete-clarification-realization-population-outcome-audit", "experiment": config["experiment"], "passed": passed, "checks": checks, "population_gates": gates}
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed: raise SystemExit(1)
    paths = {"analysis_lock": lock_path, "choice_catalog": PROJECT_ROOT / lock["choice_catalog"], "fixture_population": PROJECT_ROOT / lock["fixture_population"], "verifier": verifier_path, "audit": audit_path, "results_document": doc_path}
    outcome: dict[str, Any] = {
        "schema_version": "131-complete-clarification-realization-population-outcome-lock",
        "experiment": "v131_complete_clarification_realization_population_outcome_lock",
        "outcome": {"passed": True, "audit_pass": True, "decision": config["decisionRule"]["ifEveryCatalogPopulationAndAccessGatePasses"], "summary": {"choice_count": catalog["choice_count"], "fixture_count": population["fixture_count"], "source_record_count": population["source_record_count"], "missing_control_count": population["missing_control_count"], "cell_count": population["cell_count"], "known_pair_coverage": population["known_pair_coverage"], "novel_domain_coverage": population["novel_domain_coverage"], "unsupported_domain_coverage": population["unsupported_domain_coverage"]}},
        "authorization": {
            "modify_rerun_or_reselect_V131": False,
            "preregister_local_model_complete_answer_realization_protocol": True,
            "extract_language_or_run_model_before_next_lock": False,
            "open_protected_or_begin_induction_or_richer_planning": False,
            "run_API_training_action_or_execution": False,
        },
    }
    for key, path in paths.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT)); outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome); outcome_path.write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
