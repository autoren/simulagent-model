#!/usr/bin/env python3
"""Audit the V61 candidate runner without reading sealed candidate models."""
from __future__ import annotations

import argparse
import inspect
import json
import tempfile
from pathlib import Path

from evaluate_v61_verification import aggregate, evaluate_policy_model
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v56_verification import tool_versions, write_explicit_model


EVALUATION_FILES = (
    "python/evaluate_v61_verification.py",
    "python/audit_and_summarize_v61.py",
    "python/freeze_v61_outcome.py",
    "python/audit_v61_evaluation_implementation.py",
)

FROZEN_DEPENDENCIES = (
    "python/v61_verification.py",
    "configs/v61-design-lock.json",
    "configs/v61-implementation-lock.json",
    "configs/v61-verification-bundle-seal.json",
)


def tiny_model():
    return {
        "states": [
            {"id": 0, "kind": "root"},
            {"id": 1, "kind": "terminal", "success": True},
            {"id": 2, "kind": "terminal", "success": False},
            {"id": 3, "kind": "done"},
        ],
        "transitions": [
            {"source": 0, "target": 1, "probability": 0.25, "reward": -0.01, "annotations": []},
            {"source": 0, "target": 2, "probability": 0.75, "reward": -0.01, "annotations": []},
            {"source": 1, "target": 3, "probability": 1.0, "reward": 1.0, "annotations": []},
            {"source": 2, "target": 3, "probability": 1.0, "reward": 0.0, "annotations": []},
            {"source": 3, "target": 3, "probability": 1.0, "reward": 0.0, "annotations": []},
        ],
        "root_state": 0, "done_state": 3,
    }


def synthetic_fixture():
    model = tiny_model()
    symbolic = {
        "invariant_checks": 4, "invariant_passes": 4,
        "support_checks": 1, "support_passes": 1,
        "probability_passes": 1, "maximum_probability_error": 0.0,
        "totality_checks": 2, "totality_passes": 2,
        "deployment_checks": 1, "deployment_passes": 1,
        "z3_unknown_count": 0, "nonterminal_deadlock_count": 0,
        "counterexamples": [],
    }
    meta = {
        "independent_executor": {
            "success_probability": 0.25, "expected_return": 0.24,
            "recursive_calls": 3, "maximum_transition_normalization_error": 0.0,
        },
        "symbolic": symbolic,
        "source_binding": {
            "tree_hash_match": True, "root_action_match": True,
            "search_metadata_match": True,
        },
        "exact_belief_weight_sum": 1.0,
        "v60_stored_monte_carlo_return": 0.24,
        "v60_monte_carlo_simultaneous_radius": 0.05,
    }
    manifest_row = {
        "id": "synthetic__r0", "task_id": "synthetic", "record": 0,
        "replicate": 0, "horizon": 3, "states": 4, "transitions": 5,
    }
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)
        write_explicit_model(model, directory)
        (directory / "model.meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n"
        )
        return evaluate_policy_model(directory, manifest_row)


def passing_records(fixture):
    rows = []
    for horizon in (3, 5, 7):
        for index in range(24):
            row = json.loads(json.dumps(fixture))
            row.update({
                "id": f"synthetic_h{horizon}_{index:02d}",
                "task_id": f"synthetic_{index // 3:02d}",
                "record": index // 3, "replicate": index % 3,
                "horizon": horizon,
            })
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle-seal", default="configs/v61-verification-bundle-seal.json"
    )
    parser.add_argument(
        "--output", default="outputs/v61-long-horizon-policy-verification/evaluation-implementation-audit.json"
    )
    args = parser.parse_args()
    seal_path = (PROJECT_ROOT / args.bundle_seal).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    seal = json.loads(seal_path.read_text())
    implementation_path = PROJECT_ROOT / seal["implementation_lock"]
    implementation = json.loads(implementation_path.read_text())
    design = json.loads((PROJECT_ROOT / implementation["design_lock"]).read_text())
    config = design["config_payload"]
    implementation_audit = json.loads(
        (PROJECT_ROOT / implementation["implementation_audit"]).read_text()
    )
    errors = []
    manifest_path = PROJECT_ROOT / seal["manifest"]
    manifest = json.loads(manifest_path.read_text())
    seal_bound = (
        seal["authorization"]["write_and_audit_v61_candidate_runner"]
        and not seal["authorization"]["run_v61_candidate_verification"]
        and not seal["authorization"]["modify_v61_bundle"]
        and not seal["authorization"]["access_v59_audit_truth"]
        and file_sha256(manifest_path) == seal["manifest_sha256"]
        and file_sha256(PROJECT_ROOT / seal["bundle_audit"])
        == seal["bundle_audit_sha256"]
        and file_sha256(implementation_path) == seal["implementation_lock_sha256"]
        and manifest["policy_count"] == 72
        and manifest["horizon_counts"] == {"3": 24, "5": 24, "7": 24}
    )
    if not seal_bound:
        errors.append("V61 verification bundle seal is not intact")
    source = inspect.getsource(evaluate_policy_model)
    evaluator_firewall = (
        set(inspect.signature(evaluate_policy_model).parameters)
        == {"directory", "manifest_row"}
        and all(name not in source for name in (
            "exact_inference(", "smc2_inference(", "plan_domain_fast(",
            "continuous_unit_transition(", '[["truth"]]', '["truth"]',
        ))
    )
    if not evaluator_firewall:
        errors.append("V61 evaluator crosses the frozen-policy firewall")
    fixture = synthetic_fixture()
    fixture_ok = (
        fixture["completed"] and fixture["tree_hash_match"]
        and fixture["root_action_match"] and fixture["search_metadata_match"]
        and fixture["exact_belief_normalized"]
        and fixture["termination_probability_error"] <= 1e-12
        and fixture["success_probability_error"] <= 1e-12
        and fixture["expected_return_error"] <= 1e-12
        and fixture["v60_monte_carlo_within_simultaneous_bound"]
        and fixture["transition_distribution_normalized"] and fixture["finite"]
    )
    if not fixture_ok:
        errors.append("V61 altered synthetic end-to-end runner fixture failed")
    controls = {
        "mutation_kill_rate": implementation_audit["mutation_kill_rate"],
        "analytic_fixture_pass_rate": implementation_audit["analytic_fixture_pass_rate"],
    }
    integrity = {
        "truth_field_access_count": 0, "source_result_mutation_count": 0,
        "verification_bundle_hash_mismatch_count": 0,
        "tool_version_mismatch_count": 0,
        "unexpected_verification_attempt_count": 0,
    }
    aggregated = aggregate(passing_records(fixture), config, integrity, controls)
    aggregate_ok = (
        len(aggregated["checks"]) == len(config["gates"]) == 27
        and aggregated["passed"]
    )
    if not aggregate_ok:
        errors.append("V61 27-gate noncompensatory aggregation is invalid")
    observed_versions = tool_versions()
    versions_ok = observed_versions == implementation["tool_versions"]
    if not versions_ok:
        errors.append("V61 pinned tool versions changed")
    single_attempt_ok = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v61-evaluation-implementation-lock.json",
            "configs/v61-outcome-lock.json",
            "outputs/v61-long-horizon-policy-verification/verification-attempt.json",
            "outputs/v61-long-horizon-policy-verification/verification",
            "outputs/v61-long-horizon-policy-verification/post-result-audit.json",
            "docs/v61-results.md",
        )
    )
    if not single_attempt_ok:
        errors.append("V61 candidate verification or downstream artifact already exists")
    audit = {
        "schema_version": 61,
        "experiment": "v61_evaluation_implementation_audit",
        "passed": not errors,
        "decision": "authorize_v61_evaluation_implementation_lock" if not errors else "repair_v61_evaluation_implementation",
        "errors": errors,
        "verification_bundle_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "verification_bundle_seal_sha256": file_sha256(seal_path),
        "implementation_lock": str(implementation_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(implementation_path),
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "manifest_sha256": file_sha256(manifest_path),
        "evaluation_files_sha256": {
            path: file_sha256(PROJECT_ROOT / path) for path in EVALUATION_FILES
        },
        "frozen_dependencies_sha256": {
            path: file_sha256(PROJECT_ROOT / path) for path in FROZEN_DEPENDENCIES
        },
        "tool_versions": observed_versions,
        "checks": {
            "sealed_bundle_and_implementation": seal_bound,
            "frozen_policy_and_truth_firewall": evaluator_firewall,
            "altered_synthetic_end_to_end_runner_fixture": fixture_ok,
            "twenty_seven_noncompensatory_qualification_checks": aggregate_ok,
            "pinned_tool_versions": versions_ok,
            "single_attempt_and_downstream_absence": single_attempt_ok,
        },
        "fixture_metrics": {
            "states": fixture["states"], "transitions": fixture["transitions"],
            "maximum_storm_reference_error": max(
                fixture["termination_probability_error"],
                fixture["success_probability_error"], fixture["expected_return_error"],
            ),
            "qualification_check_count": len(aggregated["checks"]),
        },
        "data_access": {
            "v61_candidate_policy_models_accessed": 0,
            "v61_candidate_verification_runs": 0,
            "altered_synthetic_runner_fixture_models": 1,
            "truth_field_access_count": 0,
            "additional_v60_evaluation_runs": 0,
        },
    }
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
