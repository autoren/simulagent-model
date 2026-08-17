#!/usr/bin/env python3
"""Audit and freeze the V68 development-only sensitivity-screen design."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v68-development-screening.json"
    plan_path = PROJECT_ROOT / "docs/v68-development-screening-plan.md"
    audit_path = PROJECT_ROOT / "outputs/v68-development-screening/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v68-development-design-lock.json"
    if lock_path.exists():
        raise RuntimeError("V68 development design is already frozen")
    config = json.loads(config_path.read_text())
    source_path = PROJECT_ROOT / config["sourceFeasibilityLock"]
    source = json.loads(source_path.read_text())
    errors: list[str] = []

    source_payload = {key: value for key, value in source.items() if key != "lock_payload_sha256"}
    source_ok = bool(
        payload_hash(source_payload) == source["lock_payload_sha256"]
        and source["authorization"]["write_and_audit_development_only_exact_infrastructure"]
        and not source["authorization"]["run_development_only_outcome_screening"]
        and not source["authorization"]["run_confirmatory_policy_EIG_regret_or_SMC2_outcomes"]
    )
    if not source_ok:
        errors.append("V68 source-feasibility binding or development-infrastructure authorization failed")

    development = config["developmentModels"]
    registered = config["unknownDynamicsFamily"]
    models_ok = bool(
        len(development) == 4
        and {row["file"] for row in development}
        == {
            "4x3_nonterminating.POMDP",
            "tiger-alt-start.POMDP",
            "tmaze2.POMDP",
            "tmaze5.POMDP",
        }
        and all(len(row["canonicalActionCycle"]) in {3, 4} for row in development)
        and registered["inheritParametersFrom"] == "V64"
        and registered["thetaSupport"] == [0.6, 0.95]
        and registered["identityPrior"] == [0.5, 0.5]
        and registered["commandChannelLayerProjectAuthored"]
    )
    if not models_ok:
        errors.append("development model census, action cycles, or inherited family differs")

    census = config["publicPrefixCensus"]
    planning = config["exactPlanning"]
    protocol_ok = bool(
        census["depths"] == [0, 1]
        and census["retainEveryReachableActionObservationHistory"]
        and not census["deduplication"]
        and not census["selectionOrRejection"]
        and census["minimumProbabilityThreshold"] == 0.0
        and planning["horizonActions"] == 3
        and planning["primaryQuadratureNodes"] == 65
        and planning["convergenceQuadratureNodes"] == 129
        and planning["tieTolerance"] == 1e-12
    )
    if not protocol_ok:
        errors.append("complete census or exact planning protocol is incomplete")

    controls = config["controls"]
    gates = config["gates"]
    gates_ok = bool(
        set(controls) == {"map", "posteriorSampling", "openLoop", "myopicReward", "informationOnly"}
        and gates["minimumDevelopmentModels"] == 4
        and gates["minimumRetainedRecords"] == 20
        and gates["minimumCompletedRecordFraction"] == 1.0
        and gates["maximumPrimaryVsConvergenceNormalizedValueError"] == 1e-8
        and gates["minimumPrimaryActionInConvergenceOptimalSetRate"] == 1.0
        and gates["minimumExactBAMinusMAPRootActionDisagreementRecords"] == 3
        and gates["minimumExactBAMinusMAPMaterialRegretRecords"] == 2
        and gates["materialNormalizedRegret"] == 0.005
        and gates["minimumMaximumNormalizedMAPRegret"] == 0.01
        and gates["minimumExactBAMinusOpenLoopMaterialRegretRecords"] == 2
        and gates["minimumExactBAMinusPosteriorSamplingMaterialRegretRecords"] == 1
        and all(
            gates[key] == 0
            for key in (
                "maximumConfirmatoryModelsScored",
                "maximumRecordSelectionOrRejectionCount",
                "maximumHumanRecordAccessCount",
                "maximumModelForwardPassCount",
                "maximumAdapterTrainingRunCount",
            )
        )
    )
    if not gates_ok:
        errors.append("controls or prospective noncompensatory gates are incomplete")

    boundary = config["claimBoundary"]
    stage = config["stageAuthorization"]
    boundary_ok = bool(
        boundary["developmentOnly"]
        and boundary["exactOracleSensitivityScreen"]
        and not any(
            boundary[key]
            for key in (
                "confirmatoryReplication",
                "approximateInference",
                "SMC2",
                "externalUncertaintyFamily",
                "humanData",
                "modelAccess",
                "adapterTraining",
            )
        )
        and set(config["firewall"].values()) == {"forbidden"}
        and stage["auditAndFreezeDevelopmentDesign"]
        and not any(value for key, value in stage.items() if key != "auditAndFreezeDevelopmentDesign")
    )
    if not boundary_ok:
        errors.append("development-only boundary, firewall, or design-only authorization failed")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "python/v68_multi_environment_exact.py",
            "python/test_v68_multi_environment_exact.py",
            "configs/v68-development-implementation-lock.json",
            "configs/v68-development-census-seal.json",
            "configs/v68-development-evaluator-lock.json",
            "configs/v68-development-outcome-lock.json",
            "outputs/v68-development-screening/census.jsonl",
            "outputs/v68-development-screening/evaluation",
        )
    )
    if not downstream_absent:
        errors.append("V68 development implementation or outcomes exist before design lock")

    checks = {
        "source_feasibility_binding_and_authorization": source_ok,
        "four_previously_exposed_development_models_and_frozen_family": models_ok,
        "complete_depth_zero_one_census_and_exact_horizon_three_protocol": protocol_ok,
        "five_controls_and_prospective_noncompensatory_gates": gates_ok,
        "development_only_boundary_firewall_and_authorization": boundary_ok,
        "downstream_development_artifacts_absent": downstream_absent,
    }
    audit = {
        "schema_version": "68-development-screening",
        "experiment": "v68_development_design_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_development_design_and_authorize_exact_infrastructure_only"
            if not errors
            else "reject_v68_development_design"
        ),
        "errors": errors,
        "checks": checks,
        "access": {
            "development_policy_or_planning_values_computed": 0,
            "confirmatory_models_scored": 0,
            "SMC2_runs": 0,
            "human_records": 0,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["passed"]:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    lock = {
        "schema_version": "68-development-screening",
        "experiment": "v68_development_design_lock",
        "source_feasibility_lock": str(source_path.relative_to(PROJECT_ROOT)),
        "source_feasibility_lock_sha256": file_sha256(source_path),
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "config_payload": config,
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "design_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "design_audit_sha256": file_sha256(audit_path),
        "authorization": {
            "modify_source_inventory_or_development_design": False,
            "write_and_audit_exact_infrastructure": True,
            "construct_development_census": False,
            "write_and_audit_development_evaluator": False,
            "run_development_screen": False,
            "score_confirmatory_models": False,
            "run_SMC2": False,
            "human_data": False,
            "model_access": False,
            "adapter_training": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
