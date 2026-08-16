#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair", default="configs/v53r1-design-repair.json")
    parser.add_argument("--plan", default="docs/v53r1-design-repair-plan.md")
    parser.add_argument(
        "--audit", default="outputs/v53r1-design-repair/design-audit.json"
    )
    parser.add_argument("--output", default="configs/v53r1-design-lock.json")
    args = parser.parse_args()
    repair_path, plan_path, audit_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.repair, args.plan, args.audit, args.output)
    )
    if output.exists():
        raise RuntimeError("V53r1 design already frozen")
    repair = json.loads(repair_path.read_text())
    audit = json.loads(audit_path.read_text())
    source_path = (PROJECT_ROOT / repair["sourceDesignLock"]).resolve()
    source = json.loads(source_path.read_text())
    if (
        not audit["passed"]
        or audit["repair_sha256"] != file_sha256(repair_path)
        or audit["preregistration_sha256"] != file_sha256(plan_path)
        or audit["source_design_lock_sha256"] != file_sha256(source_path)
    ):
        raise RuntimeError("V53r1 audit is not bound to the repair chain")

    config = copy.deepcopy(source["config_payload"])
    config["revision"] = "r1"
    config["experiment"] = "v53r1_continuous_parameter_smc_squared"
    config["pmcmcReference"]["proposalStandardDeviation"] = 1.3
    config["exactBenchmark"]["thetaBins"] = 10
    config["exactBenchmark"]["repeatAggregation"] = (
        "equal_weight_posterior_mixture_across_three_independent_repeats"
    )
    lock = {
        "schema_version": 53,
        "revision": "r1",
        "experiment": "v53r1_design_lock",
        "source_design_lock": str(source_path.relative_to(PROJECT_ROOT)),
        "source_design_lock_sha256": file_sha256(source_path),
        "repair": str(repair_path.relative_to(PROJECT_ROOT)),
        "repair_sha256": file_sha256(repair_path),
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "design_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "design_audit_sha256": file_sha256(audit_path),
        "config_payload": config,
        "authorization": {
            "write_and_audit_smc_squared_implementation": True,
            "construct_v53r1_populations": False,
            "run_v53r1_evaluation": False,
            "active_intervention_selection": False,
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
