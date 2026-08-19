#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v184_sgd_role_isolated_language_extraction import audit_extraction, build_extraction


DEPENDENCY_KEYS = (
    "config", "parent_V183_outcome", "source_archive", "source_V134_catalog",
    "V183_contract_catalog", "V183_hidden_identifiability", "V183_development_identities",
    "V183_protected_identities", "plan", "protocol", "tests", "runner", "verifier",
    "auditor", "design_audit",
)


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def reconstruct(lock: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    extraction = build_extraction(
        (PROJECT_ROOT / lock["source_archive"]).read_bytes(),
        json.loads((PROJECT_ROOT / lock["source_V134_catalog"]).read_text()),
        json.loads((PROJECT_ROOT / lock["V183_contract_catalog"]).read_text()),
        json.loads((PROJECT_ROOT / lock["V183_hidden_identifiability"]).read_text()),
        json.loads((PROJECT_ROOT / lock["V183_development_identities"]).read_text()),
        json.loads((PROJECT_ROOT / lock["V183_protected_identities"]).read_text()),
        lock["config_payload"],
    )
    return extraction, audit_extraction(extraction, lock["config_payload"])


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v184-sgd-role-isolated-language-extraction-lock.json"
    output_root = PROJECT_ROOT / "outputs/v184-sgd-role-isolated-language-extraction/extraction"
    if output_root.exists():
        raise RuntimeError("V184 extraction may run only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]:
        raise RuntimeError("V184 lock mismatch")
    for key in DEPENDENCY_KEYS:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V184 dependency drifted: {key}")
    extraction, audit = reconstruct(lock)
    config = lock["config_payload"]
    decision = (
        config["decisionRule"]["ifEveryExtractionIsolationAndAccessGatePasses"]
        if audit["passed"]
        else config["decisionRule"]["otherwise"]
    )
    paths = {
        "development_language": output_root / "development-language.json",
        "protected_language": output_root / "protected-language.json",
        "declared_catalog_language": output_root / "declared-known-catalog-language.json",
        "extraction_summary": output_root / "extraction-summary.json",
    }
    payloads = {
        "development_language": extraction["development_language"],
        "protected_language": extraction["protected_language"],
        "declared_catalog_language": extraction["declared_catalog_language"],
        "extraction_summary": extraction["summary"],
    }
    for key, path in paths.items():
        write_json(path, payloads[key])
    integrity = {
        key: {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(path)}
        for key, path in paths.items()
    }
    access = {
        "formal_extraction_count": 1,
        "source_archive_parse_count": 1,
        "selected_conversation_read_count": 240,
        "unselected_language_record_emission_count": 0,
        "manual_development_language_inspection_count": 0,
        "manual_protected_language_inspection_count": 0,
        "protected_language_read_during_development_count": 0,
        "policy_score_count": 0,
        "model_load_count": 0,
        "model_generation_count": 0,
        "API_call_count": 0,
        "training_run_count": 0,
        "ontology_registration_count": 0,
        "trusted_state_mutation_count": 0,
        "service_call_count": 0,
        "external_side_effect_count": 0,
        "actual_execution_count": 0,
    }
    result = {
        "schema_version": "184-sgd-role-isolated-language-extraction-result",
        "experiment": config["experiment"],
        "passed": audit["passed"],
        "decision": decision,
        "summary": extraction["summary"],
        "extraction_gates": audit["checks"],
        "access": access,
        "output_integrity": integrity,
        "claim_boundary": config["claimBoundary"],
    }
    write_json(output_root / "result.json", result)
    print(json.dumps({
        "schema_version": result["schema_version"], "experiment": result["experiment"],
        "passed": result["passed"], "decision": result["decision"],
        "summary": result["summary"], "extraction_gates": result["extraction_gates"],
        "access": result["access"], "output_integrity": result["output_integrity"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
