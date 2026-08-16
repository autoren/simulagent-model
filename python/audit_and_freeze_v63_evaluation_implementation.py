#!/usr/bin/env python3
"""Freeze the V63 scorer/runtime bridge before the one candidate evaluation."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

from evaluate_v63_inference import (
    map_tv,
    randomized_rank,
    sequence_tv,
    weighted_quantile,
    weighted_wasserstein,
)
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population-seal", default="configs/v63-population-seal.json")
    parser.add_argument(
        "--audit", default="outputs/v63-external-unknown-dynamics/evaluation-implementation-audit.json"
    )
    parser.add_argument("--output", default="configs/v63-evaluation-implementation-lock.json")
    args = parser.parse_args()
    seal_path = (PROJECT_ROOT / args.population_seal).resolve()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    if output_path.exists():
        raise RuntimeError("V63 evaluation implementation already frozen")
    seal = json.loads(seal_path.read_text())
    implementation = json.loads((PROJECT_ROOT / seal["implementation_lock"]).read_text())
    design = json.loads((PROJECT_ROOT / implementation["design_lock"]).read_text())
    config = design["config_payload"]
    errors: list[str] = []
    seal_ok = bool(
        seal["authorization"]["write_and_audit_evaluation_implementation"]
        and not seal["authorization"]["run_one_candidate_evaluation"]
        and file_sha256(PROJECT_ROOT / seal["implementation_lock"])
        == seal["implementation_lock_sha256"]
        and file_sha256(PROJECT_ROOT / seal["manifest"]) == seal["manifest_sha256"]
        and file_sha256(PROJECT_ROOT / seal["population_audit"])
        == seal["population_audit_sha256"]
    )
    if not seal_ok:
        errors.append("V63 population seal is not intact or pre-evaluation")
    source_paths = [
        PROJECT_ROOT / "python/evaluate_v63_inference.py",
        PROJECT_ROOT / "python/official_v63_runtime_crosscheck.py",
    ]
    for source in source_paths:
        ast.parse(source.read_text())
    evaluation_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "outputs/v63-external-unknown-dynamics/evaluation/attempt.json",
            "outputs/v63-external-unknown-dynamics/evaluation/result.json",
            "configs/v63-outcome-lock.json",
        )
    )
    if not evaluation_absent:
        errors.append("V63 candidate evaluation or outcome already exists")
    metric_fixtures = {
        "sequence_tv": abs(sequence_tv([0.75, 0.25], [0.5, 0.5]) - 0.25) <= 1e-15,
        "map_tv": abs(map_tv({"a": 0.75, "b": 0.25}, {"a": 0.5, "c": 0.5}) - 0.5) <= 1e-15,
        "wasserstein": abs(weighted_wasserstein([0.0, 1.0], [0.5, 0.5], [0.5], [1.0]) - 0.5) <= 1e-15,
        "weighted_quantile": weighted_quantile([0.0, 1.0], [0.25, 0.75], 0.5) == 1.0,
        "randomized_rank_bounds": all(
            0 <= randomized_rank(1.0, [0.0, 1.0, 1.0, 2.0], seed) <= 4
            for seed in range(32)
        ),
    }
    metric_fixture_rate = sum(metric_fixtures.values()) / len(metric_fixtures)
    if metric_fixture_rate != 1.0:
        errors.append("V63 evaluation metric fixtures failed")
    evaluator_source = source_paths[0].read_text()
    runtime_source = source_paths[1].read_text()
    firewall_checks = {
        "candidate_calls_have_public_record_argument": (
            "exact_inference(anchor, record, config)" in evaluator_source
            and "smc2_inference(\n                anchor, record, config" in evaluator_source
        ),
        "candidate_api_has_no_truth_parameter": all(
            token not in evaluator_source
            for token in ("smc2_inference(anchor, truth", "exact_inference(anchor, truth")
        ),
        "population_hashes_checked_before_workers": (
            evaluator_source.index("sealed V63 population hash mismatch")
            < evaluator_source.index("run_parallel(\n        exact_worker")
        ),
        "single_result_refusal": "V63 sealed evaluation result already exists" in evaluator_source,
        "logical_attempt_bound": "logical_evaluation_attempt" in evaluator_source,
        "pinned_runtime_loaded_unchanged": "spec.loader.exec_module(module)" in runtime_source,
        "runtime_source_mutation_absent": all(
            token not in runtime_source for token in ("write_text(args.runtime", "open(args.runtime, 'w'")
        ),
        "active_selection_absent": "expected_information_gain" not in evaluator_source,
        "model_access_absent": all(token not in evaluator_source for token in ("transformers", "mlx_lm", "torch")),
    }
    if not all(firewall_checks.values()):
        errors.append("V63 evaluation access/runtime firewall audit failed")
    gate_names = {
        "minimumCompletedExactBenchmarkFraction", "minimumNormalizationRate",
        "maximumExactReferenceIdentityTv", "maximumExactReferenceThetaWasserstein",
        "maximumPrimaryMeanIdentityTv", "maximumPrimaryQ95IdentityTv",
        "maximumPrimaryMeanThetaWasserstein", "maximumPrimaryQ95ThetaWasserstein",
        "maximumPrimaryMeanBinnedIdentityThetaTv", "maximumPrimaryQ95BinnedIdentityThetaTv",
        "maximumPrimaryMeanCurrentSideTv", "maximumPrimaryQ95CurrentSideTv",
        "maximumPrimaryMeanNextObservationTv", "maximumPrimaryQ95NextObservationTv",
        "maximumMeanAbsoluteLogEvidenceError", "maximumPrimaryMinusMediumMeanError",
        "maximumMediumMinusLowMeanError", "minimumRankChiSquarePValue",
        "maximumAbsoluteRankBinZ", "maximumAbsoluteCoverageZ",
        "maximumTargetIdentityExtinctionRate", "maximumFalseIdentityCollapseRate",
        "maximumFalseThetaCollapseRate", "minimumFinalOuterEssFraction",
        "minimumDistinctThetaAncestorFraction", "maximumUnintendedStreamCollisions",
        "maximumOuterFingerprintCollisionRate", "maximumInnerFingerprintCollisionRate",
        "minimumControlsDetectedOrDominated", "minimumRuntimeCrosscheckCompletionFraction",
        "maximumRuntimeTransitionArrayError", "maximumRuntimeObservationArrayError",
        "maximumRuntimeEmpiricalProbabilityExcess", "minimumImplementationMutantKillRate",
        "minimumAnalyticFixturePassRate", "minimumScaleStressCompletionFraction",
        "minimumScaleStressNormalizationRate", "maximumUnexpectedEvaluationAttemptCount",
        "maximumHumanRecordAccessCount", "maximumModelForwardPassCount",
    }
    all_gates_bound = set(config["gates"]) == gate_names and all(
        f'gates["{name}"]' in evaluator_source
        or name in {
            "maximumUnexpectedEvaluationAttemptCount", "maximumHumanRecordAccessCount",
            "maximumModelForwardPassCount",
        }
        for name in gate_names
    )
    if not all_gates_bound:
        errors.append("V63 evaluator does not bind the complete frozen gate set")
    audit = {
        "schema_version": 63,
        "experiment": "v63_evaluation_implementation_audit",
        "passed": not errors,
        "decision": "authorize_one_v63_candidate_evaluation" if not errors else "repair_v63_evaluator",
        "errors": errors,
        "checks": {
            "population_seal_intact_and_pre_evaluation": seal_ok,
            "candidate_evaluation_absent": evaluation_absent,
            "metric_fixture_pass_rate": metric_fixture_rate == 1.0,
            "access_and_runtime_firewalls": all(firewall_checks.values()),
            "all_frozen_gates_bound": all_gates_bound,
        },
        "metric_fixtures": metric_fixtures,
        "firewall_checks": firewall_checks,
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): file_sha256(path) for path in source_paths
        },
        "data_access": {
            "sealed_population_records_read": 0,
            "candidate_inference_runs": 0,
            "human_record_access_count": 0,
            "simulated_human_record_count": 0,
            "model_forward_pass_count": 0,
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if errors:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    lock = {
        "schema_version": 63,
        "experiment": "v63_evaluation_implementation_lock",
        "population_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "population_seal_sha256": file_sha256(seal_path),
        "evaluation_implementation_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "evaluation_implementation_audit_sha256": file_sha256(audit_path),
        "source_sha256": audit["source_sha256"],
        "authorization": {
            "modify_design_implementation_population_or_evaluator": False,
            "run_one_candidate_evaluation": True,
            "active_intervention_selection": False,
            "reward_or_planning_evaluation": False,
            "access_human_v58_records": False,
            "simulate_human_records": False,
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
