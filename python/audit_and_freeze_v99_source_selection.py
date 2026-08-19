#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v99_source_selection import audit_source_selection


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def valid_lock(payload: dict[str, Any]) -> bool:
    return payload_hash({key: value for key, value in payload.items() if key != "lock_payload_sha256"}) == payload.get("lock_payload_sha256")


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v99-open-world-source-selection.json"
    parent_path = PROJECT_ROOT / "configs/v98-test-schema-feasibility-outcome-lock.json"
    document_path = PROJECT_ROOT / "docs/v99-open-world-source-selection.md"
    protocol_path = PROJECT_ROOT / "python/v99_source_selection.py"
    tests_path = PROJECT_ROOT / "python/test_v99_source_selection.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v99_source_selection.py"
    audit_path = PROJECT_ROOT / "outputs/v99-open-world-source-selection/source-selection-audit.json"
    lock_path = PROJECT_ROOT / "configs/v99-open-world-source-selection-lock.json"
    if audit_path.exists() or lock_path.exists():
        raise RuntimeError("V99 source selection is already frozen")
    config = json.loads(config_path.read_text())
    parent = json.loads(parent_path.read_text())
    selection = audit_source_selection(config)
    checks = {
        "V98_negative_schema_outcome_is_exact": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and not parent["outcome"]["scientific_schema_feasibility_passed"]
            and parent["outcome"]["inventory_summary"]["novel_service_family_count"] == 2
            and not parent["authorization"]["preregister_test_dialogue_source_pool_metadata"]
        ),
        "source_selection_checks_pass": selection["passed"],
        "decision_document_records_primary_evidence_and_rejections": bool(
            document_path.is_file()
            and "MASSIVE" in document_path.read_text()
            and "PRESTO" in document_path.read_text()
            and "CLINC150" in document_path.read_text()
            and "paired" in document_path.read_text().lower()
        ),
        "config_protocol_tests_and_auditor_exist": all(
            path.is_file() for path in (config_path, protocol_path, tests_path, auditor_path)
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "99-open-world-source-selection-audit",
        "experiment": "v99_open_world_source_selection_audit",
        "passed": passed,
        "decision": "freeze_MASSIVE_core_PRESTO_paired_insufficiency_selection" if passed else "reject_V99_source_selection",
        "checks": checks,
        "selection_checks": selection["checks"],
        "access": config["access"],
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "config": config_path,
        "parent_schema_outcome": parent_path,
        "decision_document": document_path,
        "protocol": protocol_path,
        "tests": tests_path,
        "auditor": auditor_path,
        "audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "99-open-world-source-selection-lock",
        "experiment": "v99_open_world_source_selection_lock",
        "outcome": {
            "passed": True,
            "decision": audit["decision"],
            "core_source": config["decision"]["coreSource"],
            "insufficient_evidence_source": config["decision"]["insufficientEvidenceSource"],
        },
        "authorization": config["authorization"],
    }
    for key, path in dependencies.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT))
        lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
