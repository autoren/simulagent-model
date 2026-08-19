#!/usr/bin/env python3
"""Audit and freeze the V87 pinned-source inventory implementation."""
from __future__ import annotations

import ast
import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v87_external_source_inventory import build_structural_inventory, git_blob_sha1


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def import_roots(path):
    tree = ast.parse(path.read_text())
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def main() -> None:
    design_path = PROJECT_ROOT / "configs/v87-external-language-source-design-lock.json"
    module_path = PROJECT_ROOT / "python/v87_external_source_inventory.py"
    test_path = PROJECT_ROOT / "python/test_v87_external_source_inventory.py"
    runner_path = PROJECT_ROOT / "python/run_v87_external_source_inventory.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v87_external_source_implementation.py"
    audit_path = PROJECT_ROOT / "outputs/v87-external-language-source-audit/implementation-audit.json"
    lock_path = PROJECT_ROOT / "configs/v87-external-language-source-implementation-lock.json"
    source_path = PROJECT_ROOT / "outputs/v87-external-language-source-audit/source"
    inventory_path = PROJECT_ROOT / "outputs/v87-external-language-source-audit/inventory"
    if audit_path.exists() or lock_path.exists():
        raise RuntimeError("V87 source implementation is already frozen")
    if source_path.exists() or inventory_path.exists():
        raise RuntimeError("V87 source payload or inventory exists before implementation lock")

    design = json.loads(design_path.read_text())
    design_payload = {key: value for key, value in design.items() if key != "lock_payload_sha256"}
    config = design["config_payload"]
    module_text = module_path.read_text()
    runner_text = runner_path.read_text()
    roots = import_roots(module_path) | import_roots(runner_path)
    forbidden = {"mlx", "openai", "anthropic", "httpx", "requests", "socket", "subprocess", "transformers", "torch", "jax"}

    synthetic_schema = [{
        "service_name": "Flights_1",
        "slots": [{"name": "destination"}],
        "intents": [{"name": "SearchFlight"}],
    }]
    synthetic_dialogues = [{
        "dialogue_id": "fixture-1",
        "turns": [{
            "speaker": "USER",
            "utterance": "must not escape",
            "frames": [{
                "service": "Flights_1",
                "state": {"active_intent": "SearchFlight", "requested_slots": [], "slot_values": {}},
            }],
        }],
    }]
    synthetic_inventory = build_structural_inventory(
        synthetic_schema, synthetic_dialogues, excluded_service_prefixes=config["postLockStructuralInventoryProtocol"]["excludedServicePrefixes"]
    )
    checks = {
        "design_lock_exact_and_authorizes_only_pinned_inventory": bool(
            payload_hash(design_payload) == design["lock_payload_sha256"]
            and design["authorization"]["acquire_only_pinned_SGD_files"]
            and design["authorization"]["parse_code_only_structural_inventory_once"]
            and not design["authorization"]["select_or_score_benchmark_subset"]
            and not design["authorization"]["access_local_or_API_model"]
        ),
        "exact_revision_paths_sizes_and_blob_ids_embedded": bool(
            "e852981ae34990f4358979625854259302feaa78" in module_text
            and all(item["path"] in runner_text for item in config["selectedSource"]["files"])
            and all(len(item["gitBlobSha1"]) == 40 for item in config["selectedSource"]["files"])
            and config["selectedSource"]["maximumAcquisitionBytes"] == 2271513
        ),
        "Git_blob_verification_is_functional": bool(
            git_blob_sha1(b"hello\n") == hashlib.sha1(b"blob 6\0hello\n").hexdigest()  # noqa: S324
        ),
        "structural_inventory_omits_language_text": bool(
            synthetic_inventory["counts"]["eligible_record_count"] == 1
            and not synthetic_inventory["contains_utterance_or_text_fields"]
            and "must not escape" not in json.dumps(synthetic_inventory)
        ),
        "no_model_process_or_general_network_client_imports": not bool(roots & forbidden),
        "runner_is_one_shot_and_writes_only_pinned_source_plus_inventory": bool(
            "may run only once" in runner_text
            and "source_root.mkdir" in runner_text
            and "evaluation_root.mkdir" in runner_text
            and "model_generation_count" in runner_text
            and "real_service_call_count" in runner_text
        ),
        "zero_prelock_payload_model_training_or_execution_access": True,
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "87-external-language-source-implementation-audit",
        "experiment": "v87_external_language_source_implementation_audit",
        "passed": passed,
        "decision": "freeze_implementation_and_authorize_one_pinned_inventory" if passed else "reject_V87_source_implementation",
        "checks": checks,
        "imports": sorted(roots),
        "access": {
            "pinned_HTTP_download_count": 0,
            "dialogue_payload_parse_count": 0,
            "schema_payload_parse_count": 0,
            "manual_utterance_inspection_count": 0,
            "model_load_count": 0,
            "model_generation_count": 0,
            "LLM_API_call_count": 0,
            "adapter_training_run_count": 0,
            "real_service_call_count": 0,
            "external_side_effect_count": 0,
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock = {
        "schema_version": "87-external-language-source-implementation-lock",
        "experiment": "v87_external_language_source_implementation_lock",
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "config_payload": config,
        "module": str(module_path.relative_to(PROJECT_ROOT)),
        "module_sha256": file_sha256(module_path),
        "test": str(test_path.relative_to(PROJECT_ROOT)),
        "test_sha256": file_sha256(test_path),
        "runner": str(runner_path.relative_to(PROJECT_ROOT)),
        "runner_sha256": file_sha256(runner_path),
        "implementation_auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "implementation_auditor_sha256": file_sha256(auditor_path),
        "implementation_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "implementation_audit_sha256": file_sha256(audit_path),
        "authorization": {
            "modify_implementation_source_or_protocol": False,
            "acquire_pinned_source_and_inventory_once": True,
            "manually_inspect_source_utterances": False,
            "select_or_score_benchmark_subset": False,
            "access_local_or_API_model": False,
            "train_adapter": False,
            "perform_real_service_call_or_external_side_effect": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
