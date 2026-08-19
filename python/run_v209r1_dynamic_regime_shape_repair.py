#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from v10_protocol import file_sha256
from v209r1_dynamic_regime_shape_repair import audit_oracle, evaluate_oracle, repair_diagnostics
from v22r2_grounding import PROJECT_ROOT


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v209r1-controlled-language-observation-pomdp-shape-repair-lock.json"
    lock = json.loads(lock_path.read_text())
    if not valid_lock(lock):
        raise RuntimeError("invalid V209r1 repair lock")
    dependency_keys = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    for key in dependency_keys:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V209r1 locked dependency changed: {key}")
    output_root = PROJECT_ROOT / "outputs/v209r1-controlled-language-observation-pomdp-shape-repair/evaluation"
    if output_root.exists():
        raise RuntimeError("V209r1 output already exists")

    parent_lock = json.loads((PROJECT_ROOT / lock["parent_V209_design_lock"]).read_text())
    config = parent_lock["config_payload"]
    repair = repair_diagnostics(config)
    summary = evaluate_oracle(config)
    audit = audit_oracle(summary, config)
    scientific_pass = audit["scientific_gates_passed"]
    decision = config["decisionRule"][
        "ifEveryOracleIntegrityScientificAndAccessGatePasses" if scientific_pass else "otherwise"
    ]
    result = {
        "schema_version": "209r1-controlled-language-observation-POMDP-shape-repair-result",
        "experiment": lock["experiment"],
        "passed": audit["access_gates_passed"],
        "scientific_oracle_passed": scientific_pass,
        "decision": decision,
        "claim_boundary": lock["repair_config_payload"]["claimBoundary"],
        "repair_diagnostics": repair,
        "checks": audit["checks"],
        "access_checks": audit["access_checks"],
        "summary": summary,
        "authorization": {
            "preregister_fresh_controlled_language_population_design_only": bool(audit["access_gates_passed"] and scientific_pass),
            "open_language_population_or_run_model": False,
            "API_training_registration_authority_action_or_execution": False,
        },
    }
    write_json(output_root / "repair-diagnostics.json", repair)
    write_json(output_root / "summary.json", summary)
    write_json(output_root / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not audit["access_gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
