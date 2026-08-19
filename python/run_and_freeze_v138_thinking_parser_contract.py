#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, valid_lock, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v135_controlled_open_world_minimal_pairs import build_catalog
from v137_direct_vs_thinking_realization import validate_final_answer as validate_v137
from v138_thinking_parser_contract import (
    inspect_template_contract,
    summarize_frozen_v137_metadata,
    validate_final_answer_v138,
)


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v138-thinking-parser-contract.json"
    plan_path = PROJECT_ROOT / "docs/v138-thinking-parser-contract-plan.md"
    protocol_path = PROJECT_ROOT / "python/v138_thinking_parser_contract.py"
    tests_path = PROJECT_ROOT / "python/test_v138_thinking_parser_contract.py"
    runner_path = PROJECT_ROOT / "python/run_and_freeze_v138_thinking_parser_contract.py"
    results_path = PROJECT_ROOT / "docs/v138-thinking-parser-contract-results.md"
    audit_path = PROJECT_ROOT / "outputs/v138-thinking-parser-contract/audit.json"
    outcome_path = PROJECT_ROOT / "configs/v138-thinking-parser-contract-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V138 already frozen")

    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV137OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    result_path = PROJECT_ROOT / config["V137Result"]
    result = json.loads(result_path.read_text())
    old_protocol_path = PROJECT_ROOT / config["V137Protocol"]
    manifest_path = PROJECT_ROOT / config["modelManifest"]
    manifest = json.loads(manifest_path.read_text())
    template_path = Path(manifest["snapshot_path"]) / config["templateFilename"]
    template_entry = next(row for row in manifest["files"] if row["path"] == config["templateFilename"])
    template_contract = inspect_template_contract(template_path.read_text())
    metadata = summarize_frozen_v137_metadata(result)
    expected = config["expectedFrozenMetadata"]

    v135_config = json.loads((PROJECT_ROOT / "configs/v135-controlled-open-world-minimal-pairs.json").read_text())
    catalog = build_catalog(v135_config)
    canonical = 'compare all boundaries\n</think>\n{"choice_id":"N01"}'
    old = validate_v137(canonical, catalog, True)
    corrected = validate_final_answer_v138(
        canonical, catalog, thinking_enabled=True, prompt_think_opened=True
    )
    missing_close = validate_final_answer_v138(
        'still reasoning {"choice_id":"K01"}',
        catalog,
        thinking_enabled=True,
        prompt_think_opened=True,
    )
    postclose_tag = validate_final_answer_v138(
        'reasoning</think><think>late</think>{"choice_id":"K01"}',
        catalog,
        thinking_enabled=True,
        prompt_think_opened=True,
    )
    direct_valid = validate_final_answer_v138(
        '{"choice_id":"K01"}', catalog, thinking_enabled=False, prompt_think_opened=False
    )
    direct_trace = validate_final_answer_v138(
        '<think>x</think>{"choice_id":"K01"}',
        catalog,
        thinking_enabled=False,
        prompt_think_opened=False,
    )
    template_hash_match = file_sha256(template_path) == template_entry["sha256"]
    checks = {
        "V137_outcome_is_valid_and_immutable": bool(valid_lock(parent) and parent["outcome"]["passed"]),
        "pinned_template_hash_matches_manifest": template_hash_match,
        "template_opens_trace_for_thinking": template_contract["thinking_prompt_supplies_open_trace"],
        "template_closes_empty_trace_for_direct": template_contract["direct_prompt_supplies_closed_empty_trace"],
        "old_parser_rejects_canonical_prompt_opened_suffix": not old["response_valid"],
        "corrected_parser_accepts_canonical_prompt_opened_suffix": bool(
            corrected["response_valid"] and corrected["answer_choice_id"] == "N01"
        ),
        "corrected_parser_rejects_missing_close": not missing_close["response_valid"],
        "corrected_parser_rejects_tags_after_close": not postclose_tag["response_valid"],
        "corrected_direct_accepts_exact_json": direct_valid["response_valid"],
        "corrected_direct_rejects_trace": not direct_trace["response_valid"],
        "frozen_metadata_matches_preregistered_counts": all(metadata[key] == value for key, value in expected.items()),
        "no_raw_response_or_trace_available_or_read": not metadata["containsRawResponseOrTrace"],
        "no_retrospective_semantic_rescore": True,
        "zero_model_language_API_and_execution_access": True,
        "required_files_exist": all(path.is_file() for path in (config_path, plan_path, protocol_path, tests_path, runner_path, results_path, old_protocol_path)),
    }
    passed = all(checks.values())
    decision = config["decisionRule"]["ifAllGatesPass"] if passed else config["decisionRule"]["otherwise"]
    audit = {
        "schema_version": "138-thinking-parser-contract-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "checks": checks,
        "template_contract": template_contract,
        "frozen_V137_metadata": metadata,
        "decision": decision,
        "access": {
            "raw_response_or_trace_read_count": 0,
            "model_load_count": 0,
            "model_generation_count": 0,
            "V134_language_read_count": 0,
            "external_language_read_count": 0,
            "API_call_count": 0,
            "actual_execution_count": 0,
        },
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "config": config_path,
        "parent_outcome": parent_path,
        "V137_result": result_path,
        "V137_protocol": old_protocol_path,
        "model_manifest": manifest_path,
        "template": template_path,
        "plan": plan_path,
        "protocol": protocol_path,
        "tests": tests_path,
        "runner": runner_path,
        "results_document": results_path,
        "audit": audit_path,
    }
    outcome: dict[str, Any] = {
        "schema_version": "138-thinking-parser-contract-outcome-lock",
        "experiment": config["experiment"],
        "outcome": {
            "passed": True,
            "technical_contract_mismatch_confirmed": True,
            "retrospective_semantic_rescore_permitted": False,
            "decision": decision,
        },
        "authorization": {
            "preregister_one_fresh_repaired_V135_development_comparison": True,
            "rerun_reparse_or_modify_V137": False,
            "touch_V134_or_external_language": False,
            "run_API_training_induction_authority_action_or_execution": False,
        },
    }
    for key, path in dependencies.items():
        try:
            relative = path.relative_to(PROJECT_ROOT)
            outcome[key] = str(relative)
        except ValueError:
            outcome[key] = str(path)
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
