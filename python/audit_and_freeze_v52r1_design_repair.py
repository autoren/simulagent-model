#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair", default="configs/v52r1-particle-budget-repair.json")
    parser.add_argument("--plan", default="docs/v52r1-particle-budget-repair-plan.md")
    parser.add_argument("--output", default="configs/v52r1-design-lock.json")
    args = parser.parse_args()
    repair_path, plan_path, output = tuple(
        (PROJECT_ROOT / value).resolve() for value in (args.repair, args.plan, args.output)
    )
    if output.exists():
        raise RuntimeError("V52r1 design already frozen")
    repair = json.loads(repair_path.read_text())
    source_path = PROJECT_ROOT / repair["sourceDesignLock"]
    source = json.loads(source_path.read_text())
    config = copy.deepcopy(source["config_payload"])
    errors = []
    changes = repair["singleAuthorizedChange"]
    if config["particleBudgets"]["budgets"] != changes["particleBudgets.budgets"]["old"]:
        errors.append("V52 source budgets do not match repair diagnosis")
    if config["particleBudgets"]["primaryBudget"] != changes["particleBudgets.primaryBudget"]["old"]:
        errors.append("V52 source primary budget does not match repair diagnosis")
    if config["scaleStress"]["particleBudget"] != changes["scaleStress.particleBudget"]["old"]:
        errors.append("V52 source scale budget does not match repair diagnosis")
    new_budgets = changes["particleBudgets.budgets"]["new"]
    if not (
        new_budgets == [31, 127, 509]
        and all(math.gcd(value, 4) == 1 for value in new_budgets)
        and changes["particleBudgets.primaryBudget"]["new"] == max(new_budgets)
        and changes["scaleStress.particleBudget"]["new"] == max(new_budgets)
    ):
        errors.append("V52r1 replacement budgets are not the preregistered coprime values")
    if any((PROJECT_ROOT / path).exists() for path in (
        "configs/v52-implementation-lock.json",
        "configs/v52r1-implementation-lock.json",
        "data/v52-rao-blackwellized-particle-filtering",
        "outputs/v52-rao-blackwellized-particle-filtering/evaluation-attempt.json",
        "outputs/v52-rao-blackwellized-particle-filtering/evaluation",
    )):
        errors.append("V52 downstream artifact exists before budget repair lock")
    if errors:
        raise RuntimeError("; ".join(errors))
    config["particleBudgets"]["budgets"] = new_budgets
    config["particleBudgets"]["primaryBudget"] = changes["particleBudgets.primaryBudget"]["new"]
    config["scaleStress"]["particleBudget"] = changes["scaleStress.particleBudget"]["new"]
    lock = {
        "schema_version": 52,
        "revision": "r1",
        "experiment": "v52r1_design_lock",
        "source_design_lock": str(source_path.relative_to(PROJECT_ROOT)),
        "source_design_lock_sha256": file_sha256(source_path),
        "repair": str(repair_path.relative_to(PROJECT_ROOT)),
        "repair_sha256": file_sha256(repair_path),
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "config_payload": config,
        "single_authorized_change": changes,
        "checks": {
            "source_design_passed": json.loads(
                (PROJECT_ROOT / source["design_audit"]).read_text()
            )["passed"],
            "new_budgets_coprime_to_four": all(
                math.gcd(value, 4) == 1 for value in new_budgets
            ),
            "same_cost_class": all(
                abs(new - old) <= 3
                for new, old in zip(new_budgets, changes["particleBudgets.budgets"]["old"], strict=True)
            ),
            "downstream_absent": True,
        },
        "authorization": {
            "write_particle_implementation": True,
            "construct_particle_populations": False,
            "run_particle_evaluation": False,
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
