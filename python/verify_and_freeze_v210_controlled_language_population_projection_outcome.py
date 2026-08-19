#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v10_protocol import file_sha256
from v210_controlled_language_population_projection import (
    audit_evaluation,
    canonical_jsonl,
    evaluate_population,
    generate_population,
    project_development_surfaces,
)
from v22r2_grounding import PROJECT_ROOT


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v210-controlled-language-population-projection-lock.json"
    lock = json.loads(lock_path.read_text())
    audit_path = PROJECT_ROOT / "outputs/v210-controlled-language-population-projection/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v210-controlled-language-population-projection-outcome-lock.json"
    results_path = PROJECT_ROOT / "docs/v210-controlled-language-population-projection-results.md"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V210 outcome already frozen")

    dependency_keys = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    dependencies_exact = valid_lock(lock) and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependency_keys)
    config = lock["config_payload"]
    parent_outcome = json.loads((PROJECT_ROOT / lock["parent_V209r1_outcome"]).read_text())
    repair_lock = json.loads((PROJECT_ROOT / parent_outcome["repair_lock"]).read_text())
    v209_lock = json.loads((PROJECT_ROOT / repair_lock["parent_V209_design_lock"]).read_text())
    parent_config = v209_lock["config_payload"]
    artifacts = {key: PROJECT_ROOT / value for key, value in config["artifacts"].items()}

    rebuilt = generate_population(config, parent_config)
    stored = {
        "DEVELOPMENT": {"surfaces": read_jsonl(artifacts["developmentSurface"]), "truth": read_jsonl(artifacts["developmentTruth"])},
        "PROTECTED": {"surfaces": read_jsonl(artifacts["protectedSurface"]), "truth": read_jsonl(artifacts["protectedTruth"])},
    }
    artifact_exact = all(
        canonical_jsonl(stored[role][kind]) == canonical_jsonl(rebuilt[role][kind])
        for role in ("DEVELOPMENT", "PROTECTED")
        for kind in ("surfaces", "truth")
    )
    rebuilt_predictions = project_development_surfaces(rebuilt["DEVELOPMENT"]["surfaces"], config)
    prediction_exact = canonical_jsonl(read_jsonl(artifacts["developmentProjection"])) == canonical_jsonl(rebuilt_predictions)
    rebuilt_summary = evaluate_population(rebuilt, rebuilt_predictions, config, parent_config)
    rebuilt_audit = audit_evaluation(rebuilt_summary, config)
    summary_exact = json.loads(artifacts["summary"].read_text()) == rebuilt_summary
    result = json.loads(artifacts["result"].read_text())
    scientific_pass = rebuilt_audit["population_projection_gates_passed"]
    expected_decision = config["decisionRule"]["ifEveryIntegrityPopulationProjectionAndAccessGatePasses" if scientific_pass else "otherwise"]
    result_exact = bool(
        result["passed"] == rebuilt_audit["access_gates_passed"]
        and result["population_projection_gates_passed"] == scientific_pass
        and result["checks"] == rebuilt_audit["checks"]
        and result["role_checks"] == rebuilt_audit["role_checks"]
        and result["access_checks"] == rebuilt_audit["access_checks"]
        and result["summary"] == rebuilt_summary
        and result["decision"] == expected_decision
    )
    checks = {
        "design_lock_and_dependencies_are_exact": dependencies_exact,
        "all_role_separated_artifacts_regenerate_byte_exactly": artifact_exact,
        "development_projection_regenerates_byte_exactly": prediction_exact,
        "summary_reconstructs_exactly": summary_exact,
        "result_reconstructs_exactly": result_exact,
        "access_audit_passes": rebuilt_audit["access_gates_passed"],
        "results_document_exists": results_path.is_file(),
        "protected_surface_remained_unread_by_baseline_and_manual_inspection": bool(
            rebuilt_summary["access"]["protected_surface_baseline_read_count"] == 0
            and rebuilt_summary["access"]["protected_surface_manual_read_count"] == 0
        ),
    }
    passed = all(checks.values())
    outcome_audit = {
        "schema_version": "210-controlled-language-population-projection-outcome-audit",
        "experiment": lock["experiment"],
        "passed": passed,
        "population_projection_gates_passed": scientific_pass,
        "decision": "freeze_verified_V210_population_projection" if passed else "freeze_failed_V210_verification",
        "checks": checks,
        "summary": rebuilt_summary,
    }
    write_json(audit_path, outcome_audit)
    if not passed:
        print(json.dumps(outcome_audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "evaluation_lock": lock_path,
        "audit": audit_path,
        "development_surface": artifacts["developmentSurface"],
        "development_truth": artifacts["developmentTruth"],
        "protected_surface": artifacts["protectedSurface"],
        "protected_truth": artifacts["protectedTruth"],
        "development_projection": artifacts["developmentProjection"],
        "summary": artifacts["summary"],
        "result": artifacts["result"],
        "results_document": results_path,
        "verifier": PROJECT_ROOT / lock["verifier"],
    }
    outcome: dict[str, Any] = {
        "schema_version": "210-controlled-language-population-projection-outcome-lock",
        "experiment": lock["experiment"],
        "outcome": {"passed": True, "population_projection_gates_passed": scientific_pass, "decision": expected_decision, "summary": rebuilt_summary},
        "authorization": {
            "preregister_separate_deterministic_development_baseline_design_only": scientific_pass,
            "open_protected_or_run_model": False,
            "API_training_registration_authority_action_or_execution": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps(outcome_audit, indent=2, sort_keys=True))
    print(json.dumps({"outcome_lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
