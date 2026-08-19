#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v169_fresh_constraint_state_population import audit_population, build_population


DEPENDENCY_KEYS = (
    "config", "parent_V168_outcome", "source_V166_outcome", "V165_hidden_records",
    "roadmap", "plan", "protocol", "tests", "runner", "verifier", "auditor", "design_audit",
)


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def reconstruct(lock: dict[str, Any]) -> dict[str, Any]:
    hidden = json.loads((PROJECT_ROOT / lock["V165_hidden_records"]).read_text())
    population = build_population(hidden, lock["config_payload"])
    audit = audit_population(population, hidden, lock["config_payload"])
    return {"population": population, "audit": audit}


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v169-fresh-constraint-state-population-lock.json"
    output_root = PROJECT_ROOT / "outputs/v169-fresh-constraint-state-population/population"
    if output_root.exists():
        raise RuntimeError("V169 population may be built only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]:
        raise RuntimeError("V169 lock mismatch")
    for key in DEPENDENCY_KEYS:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V169 dependency drifted: {key}")
    artifacts = reconstruct(lock)
    config = lock["config_payload"]
    passed = artifacts["audit"]["passed"]
    decision = config["decisionRule"]["ifEveryPopulationGatePasses"] if passed else config["decisionRule"]["otherwise"]
    states_path = output_root / "constraint-states.json"
    eligible_path = output_root / "planner-eligible-state-ids.json"
    excluded_path = output_root / "excluded-v165-signatures.json"
    summary_path = output_root / "population-summary.json"
    write_json(states_path, {"states": artifacts["population"]["states"], "contains_policy_scores": False})
    write_json(eligible_path, {"state_ids": artifacts["population"]["eligible_state_ids"], "selection_uses_policy_scores": False})
    write_json(excluded_path, {"constraint_signatures": artifacts["population"]["excluded_V165_signatures"]})
    write_json(summary_path, artifacts["population"]["summary"])
    output_integrity = {
        key: {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(path)}
        for key, path in {"constraint_states": states_path, "eligible_state_ids": eligible_path, "excluded_signatures": excluded_path, "population_summary": summary_path}.items()
    }
    access = {
        "formal_population_build_count": 1, "hidden_development_source_read_count": 1,
        "planner_policy_score_count": 0, "evaluation_record_count": 0, "manual_judgment_count": 0,
        "model_load_count": 0, "model_generation_count": 0, "API_call_count": 0,
        "training_run_count": 0, "ontology_registration_count": 0, "real_service_call_count": 0,
        "external_side_effect_count": 0, "actual_execution_count": 0,
    }
    result = {
        "schema_version": "169-fresh-constraint-state-population-result",
        "experiment": config["experiment"], "passed": passed, "decision": decision,
        "summary": artifacts["population"]["summary"], "gates": artifacts["audit"]["checks"],
        "access": access, "output_integrity": output_integrity, "claim_boundary": config["claimBoundary"],
    }
    write_json(output_root / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
