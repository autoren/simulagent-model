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
    parser.add_argument("--repair", default="configs/v53r2-ambiguity-probe-repair.json")
    parser.add_argument("--plan", default="docs/v53r2-ambiguity-probe-repair-plan.md")
    parser.add_argument("--output", default="configs/v53r2-design-lock.json")
    parser.add_argument(
        "--audit", default="outputs/v53r2-ambiguity-probe-repair/design-audit.json"
    )
    args = parser.parse_args()
    repair_path, plan_path, output, audit_path = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.repair, args.plan, args.output, args.audit)
    )
    if output.exists():
        raise RuntimeError("V53r2 design already frozen")
    repair = json.loads(repair_path.read_text())
    source_path = (PROJECT_ROOT / repair["sourceDesignLock"]).resolve()
    source = json.loads(source_path.read_text())
    errors = []
    source_ok = (
        source["authorization"]["write_and_audit_smc_squared_implementation"]
        and not source["authorization"]["construct_v53r1_populations"]
        and file_sha256(PROJECT_ROOT / source["repair"]) == source["repair_sha256"]
        and file_sha256(PROJECT_ROOT / source["preregistration"])
        == source["preregistration_sha256"]
    )
    if not source_ok:
        errors.append("V53r1 source design is not intact and pre-population")
    changes_ok = repair["authorizedDesignChanges"] == {
        "ambiguityProbeReplicateOrdinalsPerTemplate": [0],
        "ambiguityProbeActionSchedule": "all_wait_in_support_and_query",
        "controlReplicateOrdinalsPerTemplate": [0, 1],
    }
    if not changes_ok:
        errors.append("V53r2 ambiguity repair exceeds the diagnosed fixed coverage change")
    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v53-implementation-lock.json",
            "configs/v53r1-implementation-lock.json",
            "configs/v53r2-design-lock.json",
            "configs/v53r2-implementation-lock.json",
            "data/v53-continuous-parameter-smc2",
            "data/v53r1-continuous-parameter-smc2",
            "data/v53r2-continuous-parameter-smc2",
        )
    )
    if not downstream_absent:
        errors.append("V53 implementation lock or candidate population already exists")
    audit = {
        "schema_version": 53,
        "revision": "r2",
        "experiment": "v53r2_design_audit",
        "passed": not errors,
        "decision": "authorize_v53r2_design_lock" if not errors else "repair_v53r2_design",
        "errors": errors,
        "source_design_lock": str(source_path.relative_to(PROJECT_ROOT)),
        "source_design_lock_sha256": file_sha256(source_path),
        "repair": str(repair_path.relative_to(PROJECT_ROOT)),
        "repair_sha256": file_sha256(repair_path),
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "checks": {
            "source_design_bound_and_prepopulation": source_ok,
            "fixed_narrow_ambiguity_coverage": changes_ok,
            "downstream_absent": downstream_absent,
        },
        "data_access": {
            "candidate_population_records_accessed": 0,
            "smc_squared_evaluation_runs": 0,
            "pmcmc_reference_evaluation_runs": 0,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if errors:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    config = copy.deepcopy(source["config_payload"])
    config["revision"] = "r2"
    config["experiment"] = "v53r2_continuous_parameter_smc_squared"
    config["exactBenchmark"]["ambiguityProbeReplicateOrdinalsPerTemplate"] = [0]
    config["exactBenchmark"]["ambiguityProbeActionSchedule"] = "all_wait_in_support_and_query"
    config["controls"]["controlReplicateOrdinalsPerTemplate"] = [0, 1]
    lock = {
        "schema_version": 53,
        "revision": "r2",
        "experiment": "v53r2_design_lock",
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
            "construct_v53r2_populations": False,
            "run_v53r2_evaluation": False,
            "active_intervention_selection": False,
            "model_access": False,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"audit": audit, "lock": lock}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
