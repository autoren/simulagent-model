#!/usr/bin/env python3
"""Freeze the failed V65r1 one-shot outcome and independently audit its cause."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v64_external_eig import action_index, load_family, observation_index


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def identity_log_evidence(family, record: dict[str, Any], identity: int) -> tuple[float, int | None]:
    observation = observation_index(family, record["initial_observation"])
    belief = (
        family.theta_weights[:, None]
        * family.model.initial[None, :]
        * family.model.observation[0, :, observation][None, :]
    )
    probability = float(belief.sum())
    if probability <= 0.0:
        return -math.inf, -1
    belief /= probability
    log_evidence = math.log(probability)
    for tick, (action, observed) in enumerate(
        zip(record["actions"], record["observations"], strict=True)
    ):
        action_id = action_index(family, action)
        observation_id = observation_index(family, observed)
        predicted = np.einsum(
            "qs,qst->qt", belief, family.transitions[identity, :, action_id]
        )
        weighted = (
            predicted
            * family.model.observation[action_id, :, observation_id][None, :]
        )
        probability = float(weighted.sum())
        if probability <= 0.0:
            return -math.inf, tick
        belief = weighted / probability
        log_evidence += math.log(probability)
    return log_evidence, None


def main() -> None:
    evaluator_lock_path = PROJECT_ROOT / "configs/v65r1-evaluation-implementation-lock.json"
    failure_path = (
        PROJECT_ROOT / "outputs/v65r1-nested-predictive-repair/evaluation/failure.json"
    )
    result_path = (
        PROJECT_ROOT / "outputs/v65r1-nested-predictive-repair/evaluation/result.json"
    )
    raw_path = (
        PROJECT_ROOT
        / "outputs/v65r1-nested-predictive-repair/evaluation/record-budget-cells.jsonl"
    )
    audit_path = PROJECT_ROOT / "outputs/v65r1-nested-predictive-repair/outcome-audit.json"
    output_path = PROJECT_ROOT / "configs/v65r1-outcome-lock.json"
    if output_path.exists():
        raise RuntimeError("V65r1 outcome already frozen")

    evaluator_lock = json.loads(evaluator_lock_path.read_text())
    subset_seal_path = PROJECT_ROOT / evaluator_lock["subset_seal"]
    subset_seal = json.loads(subset_seal_path.read_text())
    failure = json.loads(failure_path.read_text())
    errors: list[str] = []

    frozen_evaluator_ok = bool(
        evaluator_lock["authorization"]["run_one_immutable_evaluation"]
        and not evaluator_lock["authorization"]["run_additional_evaluation"]
        and file_sha256(subset_seal_path) == evaluator_lock["subset_seal_sha256"]
        and all(
            file_sha256(PROJECT_ROOT / relative) == digest
            for relative, digest in evaluator_lock["source_sha256"].items()
        )
        and file_sha256(
            PROJECT_ROOT / evaluator_lock["evaluation_implementation_audit"]
        )
        == evaluator_lock["evaluation_implementation_audit_sha256"]
    )
    if not frozen_evaluator_ok:
        errors.append("frozen evaluator or sealed subset binding failed")

    failure_integrity_ok = bool(
        failure["bindings"]["evaluation_implementation_lock_sha256"]
        == file_sha256(evaluator_lock_path)
        and failure["bindings"]["subset_seal_sha256"] == file_sha256(subset_seal_path)
        and failure["access"]["logical_evaluation_attempts"] == 1
        and failure["one_shot_authorization_consumed"]
        and not failure["passed"]
        and failure["decision"] == "do_not_authorize_reward_planning"
        and failure["exception"]
        == {
            "message": "all V65 outer particles became extinct",
            "source": "python/v65_smc2_eig.py:526",
            "type": "RuntimeError",
        }
        and not failure["result_json_written"]
        and not failure["raw_record_budget_cells_written"]
        and not result_path.exists()
        and not raw_path.exists()
    )
    if not failure_integrity_ok:
        errors.append("V65r1 failure record or absent-result accounting is invalid")

    subset_path = PROJECT_ROOT / subset_seal["files"]["subset_public"]["path"]
    records = read_jsonl(subset_path)
    family = load_family()
    extinction_rows = []
    for record in records:
        evidences = []
        extinction_ticks = []
        for identity in range(2):
            evidence, tick = identity_log_evidence(family, record, identity)
            evidences.append(evidence)
            extinction_ticks.append(tick)
        if any(tick is not None for tick in extinction_ticks):
            extinction_rows.append(
                {
                    "record_id": record["record_id"],
                    "prefix_length": record["prefix_length"],
                    "identity_log_evidence": [
                        value if math.isfinite(value) else "-Infinity" for value in evidences
                    ],
                    "extinction_ticks_zero_based": extinction_ticks,
                }
            )
    fatal = failure["diagnosis"]["first_proven_fatal_sealed_case"]
    cause_reproduced = bool(
        len(records) == 48
        and extinction_rows
        == [
            {
                "record_id": fatal["record_id"],
                "prefix_length": fatal["prefix_length"],
                "identity_log_evidence": fatal["identity_log_evidence"],
                "extinction_ticks_zero_based": [None, fatal["extinction_tick_zero_based"]],
            }
        ]
        and fatal["extinct_identity_index"] == 1
        and family.identity_names[1] == fatal["extinct_identity"]
        and math.isfinite(extinction_rows[0]["identity_log_evidence"][0])
    )
    if not cause_reproduced:
        errors.append("independent exact support audit did not reproduce the structural failure")

    zero_access_ok = bool(
        failure["access"]["truth_field_access_count"] == 0
        and failure["access"]["human_record_access_count"] == 0
        and failure["access"]["model_forward_pass_count"] == 0
        and failure["access"]["adapter_training_run_count"] == 0
        and failure["diagnosis"]["V65r1_evaluation_reruns_after_failure"] == 0
        and failure["diagnosis"]["candidate_EIG_scoring_runs_after_failure"] == 0
        and failure["diagnosis"]["exact_identity_support_audits_after_failure"] == 1
    )
    if not zero_access_ok:
        errors.append("post-failure access or no-rerun boundary failed")

    checks = {
        "frozen_evaluator_and_subset_bindings": frozen_evaluator_ok,
        "one_shot_failure_and_absent_result_integrity": failure_integrity_ok,
        "independent_exact_identity_support_cause_reproduced": cause_reproduced,
        "no_rerun_EIG_human_model_or_training_access": zero_access_ok,
    }
    audit = {
        "schema_version": "65r1",
        "experiment": "v65r1_outcome_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_failed_outcome_and_authorize_preregistration_of_v65r2_extinct_identity_repair_only"
            if not errors and all(checks.values())
            else "reject_v65r1_outcome_record"
        ),
        "errors": errors,
        "checks": checks,
        "exact_support_diagnosis": {
            "sealed_records_checked": len(records),
            "identity_conditioned_filters_checked": 2 * len(records),
            "records_with_exact_identity_extinction": len(extinction_rows),
            "rows": extinction_rows,
            "candidate_EIG_scores_computed": 0,
        },
        "data_access": {
            "sealed_public_records_loaded": len(records),
            "truth_fields_accessed": 0,
            "V64_selection_audit_records_loaded": 0,
            "V64_evaluation_records_loaded": 0,
            "human_records": 0,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
            "V65r1_evaluation_reruns": 0,
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["passed"]:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock_payload = {
        "schema_version": "65r1",
        "experiment": "v65r1_failed_outcome_lock",
        "evaluation_implementation_lock": str(
            evaluator_lock_path.relative_to(PROJECT_ROOT)
        ),
        "evaluation_implementation_lock_sha256": file_sha256(evaluator_lock_path),
        "failure": str(failure_path.relative_to(PROJECT_ROOT)),
        "failure_sha256": file_sha256(failure_path),
        "outcome_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "outcome_audit_sha256": file_sha256(audit_path),
        "outcome_auditor": "python/audit_and_freeze_v65r1_outcome.py",
        "outcome_auditor_sha256": file_sha256(Path(__file__).resolve()),
        "decision": "do_not_authorize_reward_planning",
        "authorization": {
            "modify_or_rerun_v65r1": False,
            "preregister_v65r2_extinct_identity_repair": True,
            "run_v65r2_before_preregistration_and_freeze": False,
            "reward_planning": False,
            "formal_verification": False,
            "human_data": False,
            "model_access": False,
            "adapter_training": False,
        },
    }
    lock_payload["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_path.write_text(json.dumps(lock_payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "audit_passed": audit["passed"],
                "decision": lock_payload["decision"],
                "records_with_exact_identity_extinction": len(extinction_rows),
                "fatal_record": extinction_rows[0]["record_id"],
                "lock": str(output_path.relative_to(PROJECT_ROOT)),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
