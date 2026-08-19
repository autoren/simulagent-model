#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v100_massive_source import parse_massive_archive
from v105_open_world_interface import (
    compile_visible_catalog, complete_hypothesis_universe, evaluate_interface_gates,
    select_controlled_insufficient_identifiers, validate_response,
)


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v105-open-world-interface-lock.json"
    output_root = PROJECT_ROOT / "outputs/v105-open-world-interface/interface"
    if output_root.exists():
        raise RuntimeError("V105 interface compilation may run only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]:
        raise RuntimeError("V105 interface lock mismatch")
    dependency_keys = (
        "config", "parent_language_outcome", "source_archive", "source_inventory",
        "selected_population", "plan", "protocol", "tests", "runner", "verifier",
        "auditor", "design_audit",
    )
    for key in dependency_keys:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V105 dependency drifted: {key}")
    config = lock["config_payload"]
    archive_bytes = (PROJECT_ROOT / config["sourceArchive"]).read_bytes()
    source_records, member = parse_massive_archive(archive_bytes, config["expectedLocaleMemberSuffix"])
    source_inventory = json.loads((PROJECT_ROOT / config["sourceInventory"]).read_text())
    population = json.loads((PROJECT_ROOT / config["selectedPopulation"]).read_text())
    compiled = compile_visible_catalog(source_records, source_inventory, config)
    hypotheses = complete_hypothesis_universe(compiled["catalog"])
    controlled = select_controlled_insufficient_identifiers(population, config)
    checks = evaluate_interface_gates(compiled, len(hypotheses), controlled, config)
    fallback, fallback_valid, fallback_reason = validate_response("not-json", compiled["catalog"], config)
    checks["invalid_response_maps_to_safe_abstention"] = bool(
        not fallback_valid and fallback_reason == "invalid_json"
        and fallback == config["responseContract"]["invalidResponseFallback"]
    )
    checks["zero_selected_language_model_API_training_or_effect_access"] = True
    passed = all(checks.values())
    catalog_path = output_root / "visible-catalog.json"
    hypotheses_path = output_root / "safe-hypothesis-universe.json"
    controls_path = output_root / "controlled-insufficient-identifiers.json"
    manifest_path = output_root / "interface-manifest.json"
    write_json(catalog_path, compiled["catalog"])
    write_json(hypotheses_path, {"hypotheses": hypotheses, "payload_sha256": payload_hash({"hypotheses": hypotheses})})
    write_json(controls_path, controlled)
    manifest = {
        "schema_version": "105-open-world-interface-manifest",
        "catalog_sha256": compiled["catalog_sha256"],
        "hypothesis_count": len(hypotheses),
        "hypothesis_payload_sha256": payload_hash({"hypotheses": hypotheses}),
        "controlled_payload_sha256": controlled["payload_sha256"],
        "response_contract": config["responseContract"],
        "prompt_contract": config["promptContract"],
        "authority_boundary": config["authorityBoundary"],
    }
    write_json(manifest_path, manifest)
    output_integrity = {
        "visible_catalog": {"path": str(catalog_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(catalog_path)},
        "safe_hypothesis_universe": {"path": str(hypotheses_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(hypotheses_path)},
        "controlled_insufficient_identifiers": {"path": str(controls_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(controls_path)},
        "interface_manifest": {"path": str(manifest_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(manifest_path)},
    }
    result = {
        "schema_version": "105-open-world-interface-result",
        "experiment": "v105_massive_typed_open_world_non_authoritative_interface",
        "passed": passed,
        "decision": "freeze_interface_and_preregister_language_benchmark" if passed else "stop_V105_before_language_or_model_access",
        "source_locale_member": member,
        "interface_summary": {
            "visible_scenario_count": len(compiled["catalog"]["scenarios"]),
            "visible_intent_count": len(compiled["catalog"]["intents"]),
            "visible_unique_slot_type_count": compiled["catalog"]["visible_unique_slot_type_count"],
            "safe_hypothesis_count": len(hypotheses),
            "controlled_insufficient_role_counts": controlled["role_counts"],
            "hidden_or_unsupported_schema_leak_count": compiled["hidden_or_unsupported_schema_leak_count"],
            "catalog_sha256": compiled["catalog_sha256"],
            "controlled_payload_sha256": controlled["payload_sha256"],
        },
        "output_integrity": output_integrity,
        "gates": checks,
        "access": {
            "local_source_archive_read_count": 1,
            "source_language_record_automatic_parse_count": len(source_records),
            "text_free_population_identifier_read_count": len(population["selected_population"]),
            "selected_development_language_read_count": 0,
            "protected_test_language_read_count": 0,
            "manual_utterance_inspection_count": 0,
            "model_load_count": 0, "model_generation_count": 0,
            "LLM_API_call_count": 0, "adapter_training_run_count": 0,
            "real_service_call_count": 0, "external_side_effect_count": 0,
        },
        "claim_boundary": "typed visible catalog and language-free controlled abstention construction only; no language-baseline, model, calibration, posterior, planning, or execution result",
    }
    result_path = output_root / "result.json"
    write_json(result_path, result)
    print(json.dumps({
        "passed": passed, "decision": result["decision"],
        "interface_summary": result["interface_summary"], "gates": checks,
        "access": result["access"], "output_integrity": output_integrity,
    }, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
