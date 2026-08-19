#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v102r1_presto_context_source import parse_presto_archive
from v103_presto_target_syntax_census import build_target_syntax_census, evaluate_target_syntax_gates


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v103-presto-target-syntax-census-lock.json"
    census_root = PROJECT_ROOT / "outputs/v103-presto-target-syntax/census"
    if census_root.exists():
        raise RuntimeError("V103 census may run only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]:
        raise RuntimeError("V103 census lock mismatch")
    dependency_keys = (
        "config", "parent_source_outcome", "source_archive", "scientific_config", "plan",
        "protocol", "tests", "runner", "verifier", "auditor", "design_audit",
    )
    for key in dependency_keys:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V103 dependency drifted: {key}")
    config = lock["config_payload"]
    scientific = json.loads((PROJECT_ROOT / config["unchangedScientificConfig"]).read_text())
    source_path = PROJECT_ROOT / config["sourceArchive"]
    source_bytes = source_path.read_bytes()
    records, members = parse_presto_archive(source_bytes, scientific["archive"]["requiredMemberBasenames"])
    census = build_target_syntax_census(records, scientific, config)
    checks = evaluate_target_syntax_gates(census, config)
    checks["zero_language_identifier_manual_model_API_training_service_or_side_effect_access"] = True
    passed = all(checks.values())
    census_root.mkdir(parents=True)
    census_path = census_root / "target-syntax-census.json"
    artifact = {
        "provenance": {
            "source_archive": config["sourceArchive"],
            "source_archive_sha256": config["sourceArchiveSha256"],
            "parsed_members": members,
            "contains_language_literals_identifiers_or_root_names": False,
        },
        **census,
    }
    census_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    result = {
        "schema_version": "103-presto-target-syntax-census-result",
        "experiment": "v103_presto_target_syntax_census",
        "passed": passed,
        "decision": (
            "preregister_new_PRESTO_literal_family_dependency_construction"
            if passed else "close_PRESTO_paired_insufficiency_branch"
        ),
        "census": str(census_path.relative_to(PROJECT_ROOT)),
        "census_sha256": file_sha256(census_path),
        "census_summary": census,
        "gates": checks,
        "access": {
            "persisted_source_archive_read_count": 1,
            "language_record_automatic_parse_count": len(records),
            "emitted_language_record_count": 0,
            "emitted_candidate_identifier_count": 0,
            "manual_utterance_inspection_count": 0,
            "model_load_count": 0, "model_generation_count": 0,
            "LLM_API_call_count": 0, "adapter_training_run_count": 0,
            "real_service_call_count": 0, "external_side_effect_count": 0,
        },
        "claim_boundary": "aggregate syntax/dependency diagnostic only; no language, model, or abstention outcome",
    }
    result_path = census_root / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "passed": passed, "decision": result["decision"],
        "census_summary": census, "gates": checks, "access": result["access"],
    }, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
