#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v168_fixed_ontology_reversible_sandbox import build_fixtures, evaluate_census, evaluate_gates


DEPENDENCY_KEYS = (
    "config", "parent_V167r1_outcome", "roadmap", "plan", "protocol", "tests",
    "runner", "verifier", "auditor", "design_audit",
)


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def reconstruct(lock: dict[str, Any]) -> dict[str, Any]:
    fixtures = build_fixtures(lock["config_payload"])
    return {"fixtures": fixtures, "evaluation": evaluate_census(fixtures, lock["config_payload"])}


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v168-fixed-ontology-reversible-sandbox-lock.json"
    output_root = PROJECT_ROOT / "outputs/v168-fixed-ontology-reversible-sandbox/census"
    if output_root.exists():
        raise RuntimeError("V168 formal simulated census may run only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]:
        raise RuntimeError("V168 lock mismatch")
    for key in DEPENDENCY_KEYS:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V168 dependency drifted: {key}")
    artifacts = reconstruct(lock)
    config = lock["config_payload"]
    access = {
        "formal_census_run_count": 1,
        "simulated_fixture_count": len(artifacts["fixtures"]),
        "evaluation_record_count": 0,
        "manual_judgment_count": 0,
        "model_load_count": 0,
        "model_generation_count": 0,
        "API_call_count": 0,
        "training_run_count": 0,
        "provisional_ontology_use_count": 0,
        "real_service_call_count": 0,
        "external_side_effect_count": 0,
        "real_execution_count": 0,
    }
    gates = evaluate_gates(artifacts["evaluation"], access, config)
    passed = all(gates.values())
    decision = config["decisionRule"]["ifEverySandboxGatePasses"] if passed else config["decisionRule"]["otherwise"]
    fixtures_path = output_root / "fixture-manifest.json"
    results_path = output_root / "transaction-results.json"
    summary_path = output_root / "sandbox-summary.json"
    write_json(fixtures_path, {"fixtures": artifacts["fixtures"], "project_authored_development": True})
    write_json(results_path, {"results": artifacts["evaluation"]["results"], "contains_language": False})
    write_json(summary_path, artifacts["evaluation"]["summary"])
    output_integrity = {
        key: {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(path)}
        for key, path in {"fixture_manifest": fixtures_path, "transaction_results": results_path, "sandbox_summary": summary_path}.items()
    }
    result = {
        "schema_version": "168-fixed-ontology-reversible-sandbox-result",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": decision,
        "summary": artifacts["evaluation"]["summary"],
        "gates": gates,
        "access": access,
        "output_integrity": output_integrity,
        "claim_boundary": config["claimBoundary"],
    }
    write_json(output_root / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
