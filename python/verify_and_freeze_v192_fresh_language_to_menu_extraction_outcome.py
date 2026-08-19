#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v10_protocol import file_sha256
from v192_fresh_language_to_menu_extraction import audit_extraction, build_extraction
from v22r2_grounding import PROJECT_ROOT


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v192-fresh-language-to-menu-extraction-lock.json"
    lock = json.loads(lock_path.read_text())
    output_root = PROJECT_ROOT / "outputs/v192-fresh-language-to-menu-extraction/extraction"
    audit_path = PROJECT_ROOT / "outputs/v192-fresh-language-to-menu-extraction/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v192-fresh-language-to-menu-extraction-outcome-lock.json"
    results_path = PROJECT_ROOT / "docs/v192-fresh-language-to-menu-extraction-results.md"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V192 outcome already verified or frozen")
    dependencies_exact = valid_lock(lock) and all(
        file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"]
        for key in (
            "config",
            "parent_V191_outcome",
            "parent_V191_population_lock",
            "source_archive",
            "development_identities",
            "hidden_targets",
            "protocol",
            "tests",
            "runner",
            "verifier",
            "auditor",
            "design_audit",
        )
    )
    rebuilt = build_extraction(
        (PROJECT_ROOT / lock["source_archive"]).read_bytes(),
        json.loads((PROJECT_ROOT / lock["development_identities"]).read_text()),
        json.loads((PROJECT_ROOT / lock["hidden_targets"]).read_text()),
        lock["config_payload"],
    )
    expected = {
        "development-language.json": rebuilt["development_language"],
        "extraction-summary.json": rebuilt["summary"],
    }
    artifacts_exact = all(
        (output_root / name).is_file() and json.loads((output_root / name).read_text()) == value
        for name, value in expected.items()
    )
    result_path = output_root / "result.json"
    result = json.loads(result_path.read_text())
    audit = audit_extraction(rebuilt, lock["config_payload"])
    result_exact = bool(
        result["passed"] == audit["passed"]
        and result["checks"] == audit["checks"]
        and result["summary"] == audit["summary"]
        and result["decision"]
        == lock["config_payload"]["decisionRule"]["ifEveryExtractionProjectionIsolationAndAccessGatePasses"]
    )
    checks = {
        "design_lock_and_dependencies_are_exact": dependencies_exact,
        "extraction_artifacts_reconstruct_exactly": artifacts_exact,
        "result_reconstructs_exactly": result_exact,
        "all_extraction_gates_pass": bool(audit["passed"] and result["passed"]),
        "results_document_exists": results_path.is_file(),
        "protected_model_authority_and_execution_access_remain_zero": bool(
            result["summary"]["protected_language_read_count"] == 0
            and result["summary"]["model_load_count"] == 0
            and result["summary"]["API_call_count"] == 0
            and result["summary"]["ontology_registration_count"] == 0
            and result["summary"]["actual_execution_count"] == 0
        ),
    }
    passed = all(checks.values())
    outcome_audit = {
        "schema_version": "192-fresh-language-to-menu-extraction-outcome-audit",
        "experiment": lock["experiment"],
        "passed": passed,
        "extraction_gates_passed": bool(audit["passed"]),
        "decision": "freeze_verified_V192_extraction" if passed else "freeze_failed_V192_verification",
        "checks": checks,
        "summary": result["summary"],
    }
    write_json(audit_path, outcome_audit)
    if not passed:
        print(json.dumps(outcome_audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "extraction_lock": lock_path,
        "audit": audit_path,
        "development_language": output_root / "development-language.json",
        "extraction_summary": output_root / "extraction-summary.json",
        "result": result_path,
        "results_document": results_path,
        "verifier": PROJECT_ROOT / lock["verifier"],
    }
    outcome: dict[str, Any] = {
        "schema_version": "192-fresh-language-to-menu-extraction-outcome-lock",
        "experiment": lock["experiment"],
        "outcome": {
            "passed": True,
            "extraction_gates_passed": True,
            "decision": "freeze_V192_language_and_allow_shadow_interface_frontier_preregistration_only",
            "summary": result["summary"],
        },
        "authorization": {
            "preregister_shadow_menu_interface_and_oracle_frontier_only": True,
            "immediate_interface_scoring_or_model_run": False,
            "read_protected_language_or_run_model_API_training": False,
            "register_prune_mutate_call_service_act_or_execute": False,
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
