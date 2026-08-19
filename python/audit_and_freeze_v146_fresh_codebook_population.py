#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, valid_lock, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v146_fresh_codebook_population import audit_population, build_population


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v146-fresh-codebook-population.json"
    plan_path = PROJECT_ROOT / "docs/v146-fresh-codebook-population-plan.md"
    results_path = PROJECT_ROOT / "docs/v146-fresh-codebook-population-results.md"
    protocol_path = PROJECT_ROOT / "python/v146_fresh_codebook_population.py"
    tests_path = PROJECT_ROOT / "python/test_v146_fresh_codebook_population.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v146_fresh_codebook_population.py"
    output_dir = PROJECT_ROOT / "outputs/v146-fresh-codebook-population/design"
    audit_path = PROJECT_ROOT / "outputs/v146-fresh-codebook-population/design-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v146-fresh-codebook-population-outcome-lock.json"
    if output_dir.exists() or audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V146 already frozen")
    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV145OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    prior_paths = [PROJECT_ROOT / "outputs/v135-controlled-open-world-minimal-pairs/design/public-fixtures.json", PROJECT_ROOT / "outputs/v142-certificate-interface-population/design/public-fixtures.json"]
    prior = sum((json.loads(path.read_text()) for path in prior_paths), [])
    population = build_population(config)
    result = audit_population(population, config, prior)
    checks = {
        "V145_valid_and_authorizes_fresh_population_protocol_design_only": bool(valid_lock(parent) and parent["outcome"]["passed"] and parent["outcome"]["finite_codebook_structurally_feasible"] and parent["authorization"]["design_fresh_codebook_scoring_population_and_protocol"] and not parent["authorization"]["run_language_or_model"]),
        "all_population_codebook_leakage_gates_pass": result["passed"],
        "no_exact_V135_or_V142_conversation_overlap": result["exact_prior_conversation_overlap_count"] == 0,
        "full_oracle_code_coverage": result["oracle_code_coverage"] == 1.0,
        "zero_model_API_training_execution": all(config["gates"][key] == 0 for key in ("maximumModelLoadCount", "maximumModelGenerationCount", "maximumAPICallCount", "maximumTrainingRunCount", "maximumActualExecutionCount")),
        "required_files_exist": all(path.is_file() for path in (config_path, plan_path, results_path, protocol_path, tests_path, auditor_path)),
    }
    passed = all(checks.values())
    decision = config["decisionRule"]["ifEveryPopulationCodebookLeakageAndAccessGatePasses"] if passed else config["decisionRule"]["otherwise"]
    output_dir.mkdir(parents=True, exist_ok=False)
    assets = {"choice_catalog": output_dir / "choice-catalog.json", "certificate_codebook": output_dir / "certificate-codebook.json", "public_fixtures": output_dir / "public-fixtures.json", "hidden_fixtures": output_dir / "hidden-fixtures.json", "population_summary": output_dir / "population-summary.json"}
    for key, path in assets.items():
        write_json(path, population[key])
    audit = {"schema_version": "146-fresh-codebook-population-design-audit", "experiment": config["experiment"], "passed": passed, "checks": checks, "result": result, "decision": decision, "access": {"model_load_count": 0, "model_generation_count": 0, "API_call_count": 0, "training_run_count": 0, "actual_execution_count": 0}}
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {"config": config_path, "parent_outcome": parent_path, "plan": plan_path, "results_document": results_path, "protocol": protocol_path, "tests": tests_path, "auditor": auditor_path, "design_audit": audit_path, **assets}
    outcome: dict[str, Any] = {"schema_version": "146-fresh-codebook-population-outcome-lock", "experiment": config["experiment"], "outcome": {"passed": True, "fresh_population_and_codebook_pass": True, "decision": decision, "summary": result["summary"]}, "authorization": {"preregister_closed_alternative_scoring_protocol": True, "run_language_or_model_before_separate_preregistration": False, "open_or_generate_on_V142_test": False, "run_API_training_induction_authority_action_or_execution": False}}
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps({"passed": passed, "decision": decision, "summary": result["summary"], "checks": checks}, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
