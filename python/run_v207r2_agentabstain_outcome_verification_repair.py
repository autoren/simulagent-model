#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from v10_protocol import file_sha256
from v207r2_agentabstain_outcome_verification_repair import evaluate_repair
from v22r2_grounding import PROJECT_ROOT


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v207r2-agentabstain-outcome-verification-repair-lock.json"
    lock = json.loads(lock_path.read_text())
    if not valid_lock(lock):
        raise RuntimeError("invalid V207r2 lock")
    dependency_keys = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    if not all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependency_keys):
        raise RuntimeError("V207r2 locked dependency changed")
    output_root = PROJECT_ROOT / "outputs/v207r2-agentabstain-outcome-verification-repair/repair"
    if output_root.exists():
        raise RuntimeError("V207r2 output exists")

    source_lock = json.loads((PROJECT_ROOT / lock["source_V207r1_lock"]).read_text())
    repair = evaluate_repair(
        source_lock,
        json.loads((PROJECT_ROOT / lock["source_failed_outcome_audit"]).read_text()),
        json.loads((PROJECT_ROOT / lock["source_summary"]).read_text()),
        json.loads((PROJECT_ROOT / lock["source_result"]).read_text()),
        lock["config_payload"],
    )
    decision = lock["config_payload"]["decisionRule"][
        "ifExactBookkeepingFailureAndEverySubstantiveV207r1CheckPasses" if repair["passed"] else "otherwise"
    ]
    result = {
        "schema_version": "207r2-agentabstain-outcome-verification-repair-result",
        "experiment": lock["experiment"],
        "passed": repair["passed"],
        "decision": decision,
        "repair": repair,
        "source_artifact_mutation_count": 0,
        "network_metadata_read_count": 0,
        "scientific_evaluation_or_model_rerun_count": 0,
        "task_language_read_count": 0,
        "API_call_count": 0,
        "tool_call_count": 0,
        "actual_execution_count": 0,
    }
    write_json(output_root / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not repair["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
