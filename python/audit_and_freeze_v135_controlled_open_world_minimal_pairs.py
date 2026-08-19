#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, valid_lock, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v135_controlled_open_world_minimal_pairs import build_catalog, build_population, evaluate_gates


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v135-controlled-open-world-minimal-pairs.json"
    plan_path = PROJECT_ROOT / "docs/v135-controlled-open-world-minimal-pairs-plan.md"
    protocol_path = PROJECT_ROOT / "python/v135_controlled_open_world_minimal_pairs.py"
    tests_path = PROJECT_ROOT / "python/test_v135_controlled_open_world_minimal_pairs.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v135_controlled_open_world_minimal_pairs.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v135_controlled_open_world_minimal_pairs_outcome.py"
    parent_path = PROJECT_ROOT / "configs/v134-semantic-novelty-source-design-outcome-lock.json"
    audit_path = PROJECT_ROOT / "outputs/v135-controlled-open-world-minimal-pairs/design-audit.json"
    catalog_path = PROJECT_ROOT / "outputs/v135-controlled-open-world-minimal-pairs/design/choice-catalog.json"
    public_path = PROJECT_ROOT / "outputs/v135-controlled-open-world-minimal-pairs/design/public-fixtures.json"
    hidden_path = PROJECT_ROOT / "outputs/v135-controlled-open-world-minimal-pairs/design/hidden-fixtures.json"
    summary_path = PROJECT_ROOT / "outputs/v135-controlled-open-world-minimal-pairs/design/population-summary.json"
    lock_path = PROJECT_ROOT / "configs/v135-controlled-open-world-minimal-pairs-lock.json"
    if any(path.exists() for path in (audit_path, catalog_path, public_path, hidden_path, summary_path, lock_path)):
        raise RuntimeError("V135 already frozen")

    config = json.loads(config_path.read_text())
    parent = json.loads(parent_path.read_text())
    catalog = build_catalog(config)
    population = build_population(config)
    gates = evaluate_gates(catalog, population, config)
    authorization = config["authorization"]
    checks = {
        "V134_remains_frozen_and_untouched": bool(valid_lock(parent) and parent["outcome"]["passed"]),
        "new_authority_is_synthetic_only": bool(
            authorization["separateSyntheticDevelopmentBranchOnly"]
            and not authorization["touchOrExtractV134Language"]
            and not authorization["runModelInV135"]
            and not authorization["runAPITrainingActionOrExecution"]
        ),
        "all_structural_observability_and_access_gates_pass": all(gates.values()),
        "pass_authorizes_only_model_free_successor": bool(
            config["decisionRule"]["passAuthorizesModelFreeSequentialValueAuditOnly"]
            and not config["decisionRule"]["passAuthorizesLocalOrAPIModelRun"]
            and not config["decisionRule"]["passAuthorizesV134LanguageAccess"]
            and not config["decisionRule"]["passAuthorizesInductionTrainingAuthorityActionOrExecution"]
        ),
        "code_complete_before_freeze": all(path.is_file() for path in (config_path, plan_path, protocol_path, tests_path, auditor_path, verifier_path)),
    }
    passed = all(checks.values())
    decision = (
        config["decisionRule"]["ifEveryStructuralObservabilityAndAccessGatePasses"]
        if passed
        else config["decisionRule"]["otherwise"]
    )
    audit = {
        "schema_version": "135-controlled-open-world-minimal-pairs-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "checks": checks,
        "gates": gates,
        "decision": decision,
        "summary": {
            "choice_count": catalog["choice_count"],
            "family_count": population["family_count"],
            "group_count": population["group_count"],
            "fixture_count": population["fixture_count"],
            "split_counts": population["split_counts"],
            "cue_validation_rate": population["cue_validation_rate"],
            "clarification_resolution_rate": population["clarification_resolution_rate"],
        },
        "access": {
            "V134_language_read_count": 0,
            "external_language_read_count": 0,
            "model_load_count": 0,
            "model_generation_count": 0,
            "API_call_count": 0,
            "training_run_count": 0,
            "actual_execution_count": 0,
        },
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    public_rows = population.pop("public_fixtures")
    hidden_rows = population.pop("hidden_fixtures")
    write_json(catalog_path, catalog)
    write_json(public_path, public_rows)
    write_json(hidden_path, hidden_rows)
    write_json(summary_path, population)
    dependencies = {
        "config": config_path,
        "parent_outcome": parent_path,
        "plan": plan_path,
        "protocol": protocol_path,
        "tests": tests_path,
        "auditor": auditor_path,
        "verifier": verifier_path,
        "design_audit": audit_path,
        "choice_catalog": catalog_path,
        "public_fixtures": public_path,
        "hidden_fixtures": hidden_path,
        "population_summary": summary_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "135-controlled-open-world-minimal-pairs-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "freeze_controlled_synthetic_asset": True,
            "run_model_free_sequential_value_audit": True,
            "modify_regenerate_or_relabel_V135": False,
            "run_local_or_API_model": False,
            "touch_V134_language": False,
            "run_induction_training_authority_action_or_execution": False,
        },
    }
    for key, path in dependencies.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT))
        lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock)
    write_json(lock_path, lock)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
