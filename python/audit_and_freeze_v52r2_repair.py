#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repair", default="configs/v52r2-joint-normalization-repair.json"
    )
    parser.add_argument(
        "--plan", default="docs/v52r2-joint-normalization-repair-plan.md"
    )
    parser.add_argument("--output", default="configs/v52r2-repair-lock.json")
    args = parser.parse_args()
    repair_path, plan_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.repair, args.plan, args.output)
    )
    if output.exists():
        raise RuntimeError("V52r2 repair already frozen")
    repair = json.loads(repair_path.read_text())
    outcome_path = PROJECT_ROOT / repair["sourceOutcomeLock"]
    implementation_path = PROJECT_ROOT / repair["sourceImplementationLock"]
    seal_path = PROJECT_ROOT / repair["sourcePopulationSeal"]
    outcome = json.loads(outcome_path.read_text())
    result_path = PROJECT_ROOT / outcome["result"]
    result = json.loads(result_path.read_text())
    errors = []
    failed = sorted(
        key for key, passed in outcome["gate_checks"].items() if not passed
    )
    if failed != ["normalization", "scale_normalization"]:
        errors.append(f"V52 failed gates are not the two diagnosed gates: {failed}")
    if outcome["qualification_passed"] or result["evaluation_run_number"] != 1:
        errors.append("V52 source failure or one-run state is invalid")
    if repair["diagnosis"]["failedGates"] != failed:
        errors.append("V52r2 repair diagnosis does not match frozen outcome")
    implementation = json.loads(implementation_path.read_text())
    seal = json.loads(seal_path.read_text())
    if (
        seal["implementation_lock_sha256"] != file_sha256(implementation_path)
        or result["population_seal_sha256"] != file_sha256(seal_path)
    ):
        errors.append("V52 source implementation, seal, and result are not bound")
    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v52r2-implementation-lock.json",
            "configs/v52r2-outcome-lock.json",
            "outputs/v52r2-joint-normalization-repair/evaluation-attempt.json",
            "outputs/v52r2-joint-normalization-repair/evaluation",
        )
    )
    if not downstream_absent:
        errors.append("V52r2 downstream artifact exists before repair lock")
    if errors:
        raise RuntimeError("; ".join(errors))
    lock = {
        "schema_version": 52,
        "revision": "r2",
        "experiment": "v52r2_repair_lock",
        "repair": str(repair_path.relative_to(PROJECT_ROOT)),
        "repair_sha256": file_sha256(repair_path),
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "source_outcome_lock": str(outcome_path.relative_to(PROJECT_ROOT)),
        "source_outcome_lock_sha256": file_sha256(outcome_path),
        "source_implementation_lock": str(implementation_path.relative_to(PROJECT_ROOT)),
        "source_implementation_lock_sha256": file_sha256(implementation_path),
        "source_population_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "source_population_seal_sha256": file_sha256(seal_path),
        "single_authorized_change": repair["singleAuthorizedChange"],
        "invariants": repair["invariants"],
        "maximum_permitted_base_vs_repair_tv": repair["maximumPermittedBaseVsRepairTv"],
        "checks": {
            "exactly_two_diagnosed_gates_failed": True,
            "all_other_gates_passed": True,
            "source_one_run_frozen": True,
            "source_chain_bound": True,
            "downstream_absent": downstream_absent,
        },
        "authorization": {
            "write_and_audit_repair_implementation": True,
            "run_repair_evaluation": False,
            "model_access": False,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
