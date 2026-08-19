#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v135_controlled_open_world_minimal_pairs import build_catalog
from v136_controlled_clarification_value import evaluate, evaluate_gates


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v136-controlled-clarification-value-lock.json"
    doc_path = PROJECT_ROOT / "docs/v136-controlled-clarification-value-results.md"
    audit_path = PROJECT_ROOT / "outputs/v136-controlled-clarification-value/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v136-controlled-clarification-value-outcome-lock.json"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v136_controlled_clarification_value_outcome.py"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V136 outcome already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write V136 results first")
    lock = json.loads(lock_path.read_text())
    config = lock["config_payload"]
    parent = json.loads((PROJECT_ROOT / lock["parent_outcome"]).read_text())
    v135_config = json.loads((PROJECT_ROOT / parent["analysis_lock"]).read_text())["config_payload"]
    catalog = build_catalog(v135_config)
    expected = evaluate(config, v135_config, catalog)
    expected_gates = evaluate_gates(expected, config)
    result = json.loads((PROJECT_ROOT / lock["result"]).read_text())
    dependencies = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    checks = {
        "lock_and_dependencies_exact": bool(
            payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) == lock["lock_payload_sha256"]
            and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependencies)
        ),
        "result_exact": result == expected,
        "all_gates_pass": all(expected_gates.values()),
        "zero_language_model_or_execution": all(
            config["gates"][key] == 0
            for key in (
                "maximumV134LanguageReadCount",
                "maximumExternalLanguageReadCount",
                "maximumModelLoadCount",
                "maximumModelGenerationCount",
                "maximumAPICallCount",
                "maximumTrainingRunCount",
                "maximumActualExecutionCount",
            )
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "136-controlled-clarification-value-outcome-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "checks": checks,
        "gates": expected_gates,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise SystemExit(1)
    paths = {"analysis_lock": lock_path, "result": PROJECT_ROOT / lock["result"], "verifier": verifier_path, "audit": audit_path, "results_document": doc_path}
    outcome: dict[str, Any] = {
        "schema_version": "136-controlled-clarification-value-outcome-lock",
        "experiment": "v136_controlled_clarification_value_outcome_lock",
        "outcome": {
            "passed": True,
            "audit_pass": True,
            "model_free_value_pass": True,
            "decision": config["decisionRule"]["ifEveryValueSafetyAndAccessGatePasses"],
            "summary": {key: value for key, value in result.items() if key not in {"rows", "clear_rows"}},
        },
        "authorization": {
            "modify_rerun_or_relax_V136": False,
            "preregister_direct_vs_thinking_successor": True,
            "run_local_or_API_model": False,
            "touch_V134_language": False,
            "run_induction_training_authority_action_or_execution": False,
        },
    }
    for key, path in paths.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    outcome_path.write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
