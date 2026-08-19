#!/usr/bin/env python3
import json
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v125_sgd_catalog_population import build_catalog, evaluate_gates, select_populations


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v125-sgd-catalog-population-lock.json"
    doc_path = PROJECT_ROOT / "docs/v125-sgd-catalog-population-results.md"
    audit_path = PROJECT_ROOT / "outputs/v125-sgd-catalog-population/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v125-sgd-catalog-population-outcome-lock.json"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v125_sgd_catalog_population_outcome.py"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V125 outcome already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write V125 result document first")
    lock = json.loads(lock_path.read_text())
    inventory = json.loads((PROJECT_ROOT / lock["source_inventory"]).read_text())
    catalog = json.loads((PROJECT_ROOT / lock["choice_catalog"]).read_text())
    populations = json.loads((PROJECT_ROOT / lock["selected_populations"]).read_text())
    expected_catalog = build_catalog(inventory, lock["config_payload"])
    expected_populations = select_populations(inventory, lock["config_payload"])
    gates = evaluate_gates(catalog, populations, lock["config_payload"])
    dependencies = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    checks = {
        "lock_and_dependencies_exact": bool(payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) == lock["lock_payload_sha256"] and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependencies)),
        "catalog_selection_exact": catalog == expected_catalog,
        "population_selection_exact": populations == expected_populations,
        "all_population_gates_pass": all(gates.values()),
        "zero_language_model_and_execution": not catalog["contains_language"] and not populations["contains_language"] and lock["config_payload"]["populationGates"]["maximumActualExecutionCount"] == 0,
    }
    passed = all(checks.values())
    audit = {"schema_version": "125-sgd-catalog-population-outcome-audit", "experiment": lock["config_payload"]["experiment"], "passed": passed, "checks": checks, "population_gates": gates}
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise SystemExit(1)
    paths = {"analysis_lock": lock_path, "choice_catalog": PROJECT_ROOT / lock["choice_catalog"], "selected_populations": PROJECT_ROOT / lock["selected_populations"], "verifier": verifier_path, "audit": audit_path, "results_document": doc_path}
    outcome: dict[str, Any] = {
        "schema_version": "125-sgd-catalog-population-outcome-lock",
        "experiment": "v125_sgd_catalog_population_outcome_lock",
        "outcome": {"passed": True, "audit_pass": True, "decision": lock["config_payload"]["decisionRule"]["ifEveryCatalogPopulationAndAccessGatePasses"], "summary": {"choice_count": catalog["choice_count"], "training_record_count": populations["training_record_count"], "evaluation_record_count": populations["evaluation_record_count"], "evaluation_class_counts": populations["evaluation_class_counts"], "known_pair_coverage": populations["known_pair_coverage"], "novel_domain_coverage": populations["novel_domain_coverage"], "unsupported_domain_coverage": populations["unsupported_domain_coverage"]}},
        "authorization": {
            "modify_rerun_or_reselect_V125": False,
            "preregister_cross_dataset_retrieval_geometry_selectivity_design": True,
            "extract_selected_language_or_evaluate_before_next_lock": False,
            "run_language_model_or_open_protected": False,
            "begin_induction_or_richer_planning": False,
            "run_API_training_action_or_execution": False,
        },
    }
    for key, path in paths.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    outcome_path.write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
