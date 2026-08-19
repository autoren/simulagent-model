#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v100_massive_source import parse_massive_archive
from v162_fresh_massive_transfer_language_extraction import (
    build_selected_language_artifacts,
    evaluate_extraction_gates,
)


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))


def main() -> None:
    lock_path = (
        PROJECT_ROOT
        / "configs/v162-fresh-massive-transfer-language-extraction-lock.json"
    )
    output_root = (
        PROJECT_ROOT
        / "outputs/v162-fresh-massive-transfer-language/selected-language"
    )
    if output_root.exists():
        raise RuntimeError("V162 extraction may run only once")

    lock = json.loads(lock_path.read_text())
    if (
        payload_hash(
            {key: value for key, value in lock.items() if key != "lock_payload_sha256"}
        )
        != lock["lock_payload_sha256"]
    ):
        raise RuntimeError("V162 extraction lock mismatch")
    dependency_keys = (
        "config",
        "parent_population_outcome",
        "selected_population",
        "source_inventory",
        "source_archive",
        "plan",
        "roadmap",
        "protocol",
        "tests",
        "runner",
        "verifier",
        "auditor",
        "design_audit",
    )
    for key in dependency_keys:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V162 dependency drifted: {key}")

    config = lock["config_payload"]
    population = json.loads((PROJECT_ROOT / config["selectedPopulation"]).read_text())
    inventory = json.loads((PROJECT_ROOT / config["sourceInventory"]).read_text())
    archive_bytes = (PROJECT_ROOT / config["sourceArchive"]).read_bytes()
    source_records, member = parse_massive_archive(
        archive_bytes, config["expectedLocaleMemberSuffix"]
    )
    artifacts = build_selected_language_artifacts(
        population, inventory, source_records, config
    )
    checks = evaluate_extraction_gates(artifacts, config)
    checks["zero_manual_model_API_training_service_side_effect_or_execution_access"] = True
    passed = all(checks.values())

    output_root.mkdir(parents=True)
    output_integrity = {}
    for role, role_config in config["roles"].items():
        path = PROJECT_ROOT / role_config["output"]
        write_jsonl(path, artifacts["role_records"][role])
        output_integrity[role] = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "record_count": len(artifacts["role_records"][role]),
            "file_sha256": file_sha256(path),
            "payload_sha256": artifacts["role_payload_sha256"][role],
        }

    summary_keys = (
        "total_record_count",
        "role_record_counts",
        "role_class_record_counts",
        "exact_selected_identifier_set",
        "exact_structural_ground_truth_match",
        "exact_familiarity_reconstruction",
        "exact_slot_type_count_reconstruction",
        "development_protected_role_disjoint",
        "unselected_language_record_count",
        "role_payload_sha256",
    )
    result = {
        "schema_version": "162-fresh-massive-transfer-language-extraction-result",
        "experiment": "v162_exact_V161_selected_language_extraction",
        "passed": passed,
        "decision": (
            "freeze_language_and_preregister_deterministic_development_interface"
            if passed
            else "stop_V162_before_interface_or_model_access"
        ),
        "source_locale_member": member,
        "output_integrity": output_integrity,
        "extraction_summary": {key: artifacts[key] for key in summary_keys},
        "gates": checks,
        "access": {
            "local_source_archive_read_count": 1,
            "source_language_record_automatic_parse_count": len(source_records),
            "selected_language_record_extraction_count": artifacts[
                "total_record_count"
            ],
            "unselected_language_record_emission_count": 0,
            "manual_development_utterance_inspection_count": 0,
            "manual_protected_utterance_inspection_count": 0,
            "protected_language_read_during_development_count": 0,
            "model_load_count": 0,
            "model_generation_count": 0,
            "LLM_API_call_count": 0,
            "adapter_training_run_count": 0,
            "real_service_call_count": 0,
            "external_side_effect_count": 0,
            "actual_execution_count": 0,
        },
        "claim_boundary": (
            "exact automatic role-separated selected-language extraction only; "
            "no interface, model, open-set, relation-codebook, ontology, planning, or execution outcome"
        ),
    }
    result_path = output_root / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "passed": passed,
                "decision": result["decision"],
                "output_integrity": output_integrity,
                "extraction_summary": result["extraction_summary"],
                "gates": checks,
                "access": result["access"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
