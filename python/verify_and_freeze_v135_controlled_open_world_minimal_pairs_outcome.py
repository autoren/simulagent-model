#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v135_controlled_open_world_minimal_pairs import build_catalog, build_population, evaluate_gates


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v135-controlled-open-world-minimal-pairs-lock.json"
    doc_path = PROJECT_ROOT / "docs/v135-controlled-open-world-minimal-pairs-results.md"
    audit_path = PROJECT_ROOT / "outputs/v135-controlled-open-world-minimal-pairs/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v135-controlled-open-world-minimal-pairs-outcome-lock.json"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v135_controlled_open_world_minimal_pairs_outcome.py"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V135 outcome already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write V135 results first")

    lock = json.loads(lock_path.read_text())
    config = lock["config_payload"]
    expected_catalog = build_catalog(config)
    expected_population = build_population(config)
    expected_public = expected_population.pop("public_fixtures")
    expected_hidden = expected_population.pop("hidden_fixtures")
    expected_gates = evaluate_gates(expected_catalog, build_population(config), config)
    catalog = json.loads((PROJECT_ROOT / lock["choice_catalog"]).read_text())
    public_rows = json.loads((PROJECT_ROOT / lock["public_fixtures"]).read_text())
    hidden_rows = json.loads((PROJECT_ROOT / lock["hidden_fixtures"]).read_text())
    summary = json.loads((PROJECT_ROOT / lock["population_summary"]).read_text())
    dependencies = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    checks = {
        "lock_and_dependencies_exact": bool(
            payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"})
            == lock["lock_payload_sha256"]
            and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependencies)
        ),
        "catalog_exact": catalog == expected_catalog,
        "public_fixtures_exact": public_rows == expected_public,
        "hidden_fixtures_exact": hidden_rows == expected_hidden,
        "population_summary_exact": summary == expected_population,
        "all_gates_pass": all(expected_gates.values()),
        "zero_external_language_model_or_execution": all(
            config["gates"][key] == 0
            for key in (
                "maximumV134LanguageReadCount",
                "maximumModelLoadCount",
                "maximumModelGenerationCount",
                "maximumAPICallCount",
                "maximumTrainingRunCount",
                "maximumActualExecutionCount",
            )
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "135-controlled-open-world-minimal-pairs-outcome-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "checks": checks,
        "gates": expected_gates,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise SystemExit(1)

    paths = {
        "analysis_lock": lock_path,
        "choice_catalog": PROJECT_ROOT / lock["choice_catalog"],
        "public_fixtures": PROJECT_ROOT / lock["public_fixtures"],
        "hidden_fixtures": PROJECT_ROOT / lock["hidden_fixtures"],
        "population_summary": PROJECT_ROOT / lock["population_summary"],
        "verifier": verifier_path,
        "audit": audit_path,
        "results_document": doc_path,
    }
    outcome: dict[str, Any] = {
        "schema_version": "135-controlled-open-world-minimal-pairs-outcome-lock",
        "experiment": "v135_controlled_open_world_minimal_pairs_outcome_lock",
        "outcome": {
            "passed": True,
            "audit_pass": True,
            "controlled_benchmark_pass": True,
            "decision": config["decisionRule"]["ifEveryStructuralObservabilityAndAccessGatePasses"],
            "summary": {
                "choice_count": catalog["choice_count"],
                "family_count": summary["family_count"],
                "group_count": summary["group_count"],
                "fixture_count": summary["fixture_count"],
                "split_counts": summary["split_counts"],
                "cue_validation_rate": summary["cue_validation_rate"],
                "clarification_resolution_rate": summary["clarification_resolution_rate"],
            },
        },
        "authorization": {
            "modify_regenerate_or_relabel_V135": False,
            "run_model_free_sequential_value_audit": True,
            "run_local_or_API_model": False,
            "touch_V134_language": False,
            "run_induction_training_authority_action_or_execution": False,
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
