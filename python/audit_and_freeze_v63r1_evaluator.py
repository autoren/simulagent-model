#!/usr/bin/env python3
"""Freeze the V63r1 pooled-repeat evaluator before its one repair run."""
from __future__ import annotations

import argparse
import hashlib
import json
import math

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from evaluate_v63r1_repair import pool_smc2_repeats


def fixture_result(identity, joint, evidence):
    return {
        "identity": identity,
        "theta_values": [0.7, 0.9],
        "theta_weights": [0.5, 0.5],
        "joint_bins": joint,
        "current_side": identity + [0.0],
        "next_observation": [0.0, identity[0], identity[1], 0.0],
        "log_evidence": evidence,
        "atoms": [
            {"identity": 0, "theta": 0.7, "state": 2, "weight": identity[0]},
            {"identity": 1, "theta": 0.9, "state": 3, "weight": identity[1]},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-lock", default="configs/v63r1-design-lock.json")
    parser.add_argument(
        "--audit", default="outputs/v63r1-repeat-pooling-repair/evaluation-implementation-audit.json"
    )
    parser.add_argument("--output", default="configs/v63r1-evaluation-implementation-lock.json")
    args = parser.parse_args()
    design_path = (PROJECT_ROOT / args.design_lock).resolve()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    if output_path.exists():
        raise RuntimeError("V63r1 evaluator already frozen")
    design = json.loads(design_path.read_text())
    source_outcome = json.loads((PROJECT_ROOT / design["source_v63_outcome_lock"]).read_text())
    original_result_path = (PROJECT_ROOT / source_outcome["result"]).resolve()
    errors = []
    design_ok = bool(
        design["authorization"]["write_and_audit_repair_evaluator"]
        and not design["authorization"]["run_one_repair_evaluation"]
        and file_sha256(PROJECT_ROOT / design["repair"]) == design["repair_sha256"]
        and file_sha256(PROJECT_ROOT / design["preregistration"])
        == design["preregistration_sha256"]
        and file_sha256(PROJECT_ROOT / design["design_audit"])
        == design["design_audit_sha256"]
        and file_sha256(original_result_path) == design["original_v63_result_sha256"]
    )
    if not design_ok:
        errors.append("V63r1 repair design is not intact or pre-evaluation")
    left = fixture_result([0.8, 0.2], {"0:0": 0.8, "1:0": 0.2}, math.log(0.4))
    right = fixture_result([0.2, 0.8], {"0:0": 0.2, "1:0": 0.8}, math.log(0.6))
    pooled = pool_smc2_repeats([left, right])
    pooling_checks = {
        "identity_equal_weight_mixture": all(
            abs(value - 0.5) <= 1e-15 for value in pooled["identity"]
        ),
        "joint_equal_weight_mixture": all(
            abs(pooled["joint_bins"][key] - 0.5) <= 1e-15 for key in ("0:0", "1:0")
        ),
        "theta_repeat_weights_sum": abs(sum(pooled["theta_weights"]) - 1.0) <= 1e-15,
        "atom_repeat_weights_sum": abs(sum(row["weight"] for row in pooled["atoms"]) - 1.0) <= 1e-15,
        "log_mean_evidence": abs(pooled["log_evidence"] - math.log(0.5)) <= 1e-15,
    }
    if not all(pooling_checks.values()):
        errors.append("V63r1 equal-weight pooling fixtures failed")
    evaluator_path = PROJECT_ROOT / "python/evaluate_v63r1_repair.py"
    source = evaluator_path.read_text()
    scope_checks = {
        "three_repeats_pooled_before_metrics": (
            "pooled = pool_smc2_repeats(independent)" in source
            and source.index("pooled = pool_smc2_repeats(independent)")
            < source.index('"metrics": inference_metrics(exact, pooled')
        ),
        "one_metric_row_per_budget": "for budget in config[\"smcSquared\"][\"outerThetaParticleBudgets\"]" in source,
        "same_exact_population": '"exact-public.jsonl"' in source and '"exact-truth.jsonl"' in source,
        "no_SBC_scale_or_runtime_worker": all(
            token not in source for token in ("sbc_worker", "scale_worker", "runtime_crosscheck(config")
        ),
        "original_subsections_reused": all(
            token in source for token in (
                'original["simulation_based_calibration"]',
                'original["scale_stress"]',
                'original["runtime_crosscheck"]',
            )
        ),
        "original_result_immutable": "original_result_path.write_text" not in source,
        "active_selection_absent": "expected_information_gain" not in source,
        "model_access_absent": all(token not in source for token in ("transformers", "mlx_lm", "torch")),
    }
    if not all(scope_checks.values()):
        errors.append("V63r1 evaluator exceeds the pooling-only repair scope")
    repair_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v63r1-evaluation-implementation-lock.json",
            "configs/v63r1-outcome-lock.json",
            "outputs/v63r1-repeat-pooling-repair/evaluation/attempt.json",
            "outputs/v63r1-repeat-pooling-repair/evaluation/result.json",
        )
    )
    if not repair_absent:
        errors.append("V63r1 repair evaluation or outcome already exists")
    audit = {
        "schema_version": "63r1",
        "experiment": "v63r1_evaluation_implementation_audit",
        "passed": not errors,
        "decision": "authorize_one_v63r1_repair_evaluation" if not errors else "repair_v63r1_evaluator",
        "errors": errors,
        "checks": {
            "repair_design_intact_and_pre_evaluation": design_ok,
            "pooling_fixtures": all(pooling_checks.values()),
            "pooling_only_scope": all(scope_checks.values()),
            "repair_evaluation_absent": repair_absent,
        },
        "pooling_checks": pooling_checks,
        "scope_checks": scope_checks,
        "source_sha256": {
            "python/evaluate_v63r1_repair.py": file_sha256(evaluator_path),
        },
        "data_access": {
            "sealed_population_records_read": 0,
            "repair_candidate_runs": 0,
            "human_record_access_count": 0,
            "model_forward_pass_count": 0,
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if errors:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    lock = {
        "schema_version": "63r1",
        "experiment": "v63r1_evaluation_implementation_lock",
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "evaluation_implementation_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "evaluation_implementation_audit_sha256": file_sha256(audit_path),
        "source_sha256": audit["source_sha256"],
        "authorization": {
            "modify_v63_or_v63r1_artifacts": False,
            "run_one_repair_evaluation": True,
            "active_intervention_selection": False,
            "model_access": False,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"audit": audit, "lock": lock}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
