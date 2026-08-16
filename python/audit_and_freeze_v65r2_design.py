#!/usr/bin/env python3
"""Audit and freeze the targeted V65r2 extinct-identity repair design."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v64_external_eig import action_index, load_family, observation_index


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def boolean_identity_support(family, record: dict[str, Any], identity: int) -> tuple[bool, int | None]:
    observation = observation_index(family, record["initial_observation"])
    support = (family.model.initial > 0.0) & (
        family.model.observation[0, :, observation] > 0.0
    )
    if not np.any(support):
        return False, -1
    for tick, (action, observed) in enumerate(
        zip(record["actions"], record["observations"], strict=True)
    ):
        action_id = action_index(family, action)
        observation_id = observation_index(family, observed)
        theta_supports = family.transitions[identity, :, action_id] > 0.0
        if not np.all(theta_supports == theta_supports[0][None, :, :]):
            raise RuntimeError("V65r2 transition support unexpectedly varies over theta")
        successor = np.any(support[:, None] & theta_supports[0], axis=0)
        support = successor & (
            family.model.observation[action_id, :, observation_id] > 0.0
        )
        if not np.any(support):
            return False, tick
    return True, None


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair", default="configs/v65r2-extinct-identity-repair.json")
    parser.add_argument(
        "--plan", default="docs/v65r2-extinct-identity-repair-plan.md"
    )
    parser.add_argument(
        "--audit", default="outputs/v65r2-extinct-identity-repair/design-audit.json"
    )
    parser.add_argument("--output", default="configs/v65r2-design-lock.json")
    args = parser.parse_args()

    repair_path = (PROJECT_ROOT / args.repair).resolve()
    plan_path = (PROJECT_ROOT / args.plan).resolve()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    if output_path.exists():
        raise RuntimeError("V65r2 design already frozen")
    repair = json.loads(repair_path.read_text())
    outcome_path = (PROJECT_ROOT / repair["sourceOutcomeLock"]).resolve()
    outcome = json.loads(outcome_path.read_text())
    evaluator_path = (PROJECT_ROOT / outcome["evaluation_implementation_lock"]).resolve()
    evaluator = json.loads(evaluator_path.read_text())
    implementation_path = (PROJECT_ROOT / evaluator["implementation_lock"]).resolve()
    implementation = json.loads(implementation_path.read_text())
    source_design_path = (PROJECT_ROOT / implementation["design_lock"]).resolve()
    source_design = json.loads(source_design_path.read_text())
    base = source_design["config_payload"]
    errors: list[str] = []

    outcome_payload = {
        key: value for key, value in outcome.items() if key != "lock_payload_sha256"
    }
    outcome_ok = bool(
        payload_hash(outcome_payload) == outcome["lock_payload_sha256"]
        and outcome["authorization"]["preregister_v65r2_extinct_identity_repair"]
        and not outcome["authorization"]["modify_or_rerun_v65r1"]
        and not outcome["authorization"]["reward_planning"]
        and file_sha256(PROJECT_ROOT / outcome["failure"]) == outcome["failure_sha256"]
        and file_sha256(PROJECT_ROOT / outcome["outcome_audit"])
        == outcome["outcome_audit_sha256"]
        and file_sha256(PROJECT_ROOT / outcome["outcome_auditor"])
        == outcome["outcome_auditor_sha256"]
        and file_sha256(evaluator_path)
        == outcome["evaluation_implementation_lock_sha256"]
    )
    if not outcome_ok:
        errors.append("V65r1 failed outcome or narrow repair authorization is invalid")

    scope = repair["repair"]
    durable = repair["durableAttemptProtocol"]
    scope_ok = bool(
        repair["defectDiscoveryStage"].startswith("sole_immutable_v65r1")
        and scope["scope"]
        == "identity_branch_extinction_classification_and_joint_measure_assembly_only"
        and scope["supportOracle"].startswith("boolean_forward_reachability")
        and scope["exactZeroIdentityHandling"].startswith(
            "return_log_evidence_negative_infinity"
        )
        and scope["positiveSupportParticleExtinctionHandling"].startswith(
            "raise_a_distinct"
        )
        and scope["bothIdentitiesImpossibleHandling"].startswith("raise_impossible")
        and scope["jointNormalization"].startswith("normalize_only_finite")
        and scope["subset"].startswith("reuse_the_immutable")
        and durable["attemptMarker"].startswith("atomically_write_attempt.json")
        and durable["repeatInvocation"].startswith("reject_if_attempt.json")
        and durable["exception"].startswith("atomically_write_failure.json")
        and len(repair["unchanged"]) == 11
    )
    if not scope_ok:
        errors.append("V65r2 is not confined to exact-zero branch handling and durability")

    boundary = repair["claimBoundary"]
    boundary_ok = bool(
        boundary["targetedRepairOfFailedV65r1"]
        and not any(
            boundary[key]
            for key in (
                "independentExactBenchmarkReplication",
                "sequentialApproximateAdaptiveRollout",
                "rewardPlanning",
                "formalVerification",
                "humanData",
                "modelAccess",
                "adapterTraining",
            )
        )
    )
    stage = repair["stageAuthorization"]
    firewall_ok = bool(
        set(repair["firewall"].values())
        == {
            "forbidden",
            "forbidden_on_the_sealed_subset",
        }
        and stage["writeAndAuditRepairImplementation"]
        and not any(
            value
            for key, value in stage.items()
            if key != "writeAndAuditRepairImplementation"
        )
    )
    if not boundary_ok or not firewall_ok:
        errors.append("V65r2 claim boundary, firewall, or stage authorization is invalid")

    invariant_hashes = {
        "subset": payload_hash(base["subset"]),
        "smcSquared": payload_hash(base["smcSquared"]),
        "gates": payload_hash(base["gates"]),
        "seeds": payload_hash(base["seeds"]),
        "controls": payload_hash(base["controls"]),
        "exactReference": payload_hash(base["exactReference"]),
        "comparisonTargets": payload_hash(base["comparisonTargets"]),
    }
    invariant_ok = bool(
        base["subset"]["records"] == 48
        and base["smcSquared"]["outerThetaParticleBudgets"] == [31, 127, 509]
        and base["smcSquared"]["innerStateParticleBudget"] == 127
        and base["smcSquared"]["independentRepeatsPerBudget"] == 3
        and base["approximateAcquisition"]["raoBlackwellizeKnownConditionalState"]
        and base["approximateAcquisition"]["poolBeforeScore"]
    )
    if not invariant_ok:
        errors.append("frozen V65r1 evaluation invariants are unavailable or changed")

    subset_seal_path = (PROJECT_ROOT / evaluator["subset_seal"]).resolve()
    subset_seal = json.loads(subset_seal_path.read_text())
    subset_path = (PROJECT_ROOT / subset_seal["files"]["subset_public"]["path"]).resolve()
    records = read_jsonl(subset_path)
    family = load_family()
    support_rows = []
    for record in records:
        per_identity = [
            boolean_identity_support(family, record, identity) for identity in range(2)
        ]
        if not all(row[0] for row in per_identity):
            support_rows.append(
                {
                    "record_id": record["record_id"],
                    "prefix_length": record["prefix_length"],
                    "identity_supported": [row[0] for row in per_identity],
                    "extinction_ticks_zero_based": [row[1] for row in per_identity],
                }
            )
    impossible_fixture = {
        "record_id": "v65r2-both-identities-impossible",
        "prefix_length": 0,
        "initial_observation": "good",
        "actions": [],
        "observations": [],
    }
    impossible_support = [
        boolean_identity_support(family, impossible_fixture, identity)[0]
        for identity in range(2)
    ]
    fixture_ok = bool(
        support_rows
        == [
            {
                "record_id": repair["mandatoryImplementationFixtures"][
                    "sealedOneIdentityZeroSupport"
                ],
                "prefix_length": 5,
                "identity_supported": [True, False],
                "extinction_ticks_zero_based": [None, 4],
            }
        ]
        and impossible_support == [False, False]
        and all(
            np.all(
                (family.transitions[identity] > 0.0)
                == (family.transitions[identity, 0][None, :, :, :] > 0.0)
            )
            for identity in range(2)
        )
    )
    if not fixture_ok:
        errors.append("independent Boolean support fixtures did not reproduce repair domain")

    downstream = (
        "configs/v65r2-implementation-lock.json",
        "configs/v65r2-evaluation-implementation-lock.json",
        "configs/v65r2-outcome-lock.json",
        "python/v65r2_smc2_eig.py",
        "python/evaluate_v65r2_eig.py",
        "outputs/v65r2-extinct-identity-repair/evaluation",
    )
    downstream_absent = not any((PROJECT_ROOT / path).exists() for path in downstream)
    if not downstream_absent:
        errors.append("V65r2 implementation or evaluation exists before design lock")

    checks = {
        "V65r1_failed_outcome_and_repair_authorization": outcome_ok,
        "repair_scope_and_durable_attempt_protocol": scope_ok,
        "claim_boundary_firewall_and_implementation_only_authorization": (
            boundary_ok and firewall_ok
        ),
        "all_original_evaluation_invariants_bound": invariant_ok,
        "independent_boolean_support_fixtures": fixture_ok,
        "V65r2_downstream_absent": downstream_absent,
    }
    audit = {
        "schema_version": "65r2",
        "experiment": "v65r2_design_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_v65r2_design_and_authorize_repair_implementation_only"
            if not errors and all(checks.values())
            else "reject_v65r2_design"
        ),
        "errors": errors,
        "checks": checks,
        "support_fixtures": {
            "sealed_records_checked": len(records),
            "sealed_one_identity_zero_rows": support_rows,
            "both_identity_impossible_fixture": impossible_fixture,
            "both_identity_impossible_support": impossible_support,
            "candidate_EIG_scores_computed": 0,
        },
        "unchanged_payload_sha256": invariant_hashes,
        "data_access": {
            "sealed_public_records_loaded": len(records),
            "truth_fields_accessed": 0,
            "candidate_EIG_scores_computed": 0,
            "V65r1_evaluation_reruns": 0,
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

    effective = copy.deepcopy(base)
    effective["schemaVersion"] = "65r2"
    effective["extinctIdentityRepair"] = copy.deepcopy(scope)
    effective["durableAttemptProtocol"] = copy.deepcopy(durable)
    effective["claimBoundary"].update(boundary)
    lock = {
        "schema_version": "65r2",
        "experiment": "v65r2_design_lock",
        "source_v65r1_outcome_lock": str(outcome_path.relative_to(PROJECT_ROOT)),
        "source_v65r1_outcome_lock_sha256": file_sha256(outcome_path),
        "source_v65r1_design_lock": str(source_design_path.relative_to(PROJECT_ROOT)),
        "source_v65r1_design_lock_sha256": file_sha256(source_design_path),
        "subset_seal": str(subset_seal_path.relative_to(PROJECT_ROOT)),
        "subset_seal_sha256": file_sha256(subset_seal_path),
        "repair": str(repair_path.relative_to(PROJECT_ROOT)),
        "repair_sha256": file_sha256(repair_path),
        "repair_payload": repair,
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "design_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "design_audit_sha256": file_sha256(audit_path),
        "config_payload": effective,
        "unchanged_payload_sha256": invariant_hashes,
        "authorization": {
            "modify_or_rerun_v65r1": False,
            "modify_v65r2_design": False,
            "write_and_audit_repair_implementation": True,
            "write_and_audit_durable_evaluator": False,
            "run_evaluation": False,
            "reward_planning": False,
            "formal_verification": False,
            "human_data": False,
            "model_access": False,
            "adapter_training": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    output_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "audit_passed": audit["passed"],
                "checks": checks,
                "sealed_zero_identity_rows": len(support_rows),
                "lock": str(output_path.relative_to(PROJECT_ROOT)),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
