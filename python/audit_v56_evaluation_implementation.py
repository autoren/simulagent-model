#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import tempfile
from pathlib import Path

from evaluate_v56_verification import aggregate, evaluate_policy_model
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v42_stateful import atom_universe, entities, world_signature
from v53_smc2 import mechanic_registry
from v56_verification import compile_policy_dtmc, tool_versions, write_policy_bundle


EVALUATION_FILES = (
    "python/evaluate_v56_verification.py",
    "python/audit_and_summarize_v56.py",
    "python/freeze_v56_outcome.py",
    "python/audit_v56_evaluation_implementation.py",
)

FROZEN_DEPENDENCIES = (
    "python/v56_verification.py",
    "configs/v56-design-lock.json",
    "configs/v56-implementation-lock.json",
    "configs/v56-verification-bundle-seal.json",
)


def _wait_policy(observation: str, depth: int = 3) -> dict:
    node = {"terminal": True, "value": 1.0}
    for _ in range(depth):
        node = {
            "terminal": False,
            "value": 1.0,
            "selected_action": {"id": "wait", "binding": {}},
            "selected_action_key": "wait",
            "branches": {observation: node},
            "observation_probabilities": {observation: 1.0},
            "action_values": {"wait": 1.0},
            "optimal_action_keys": ["wait"],
        }
    return node


def _synthetic_runner_fixture(config: dict) -> dict:
    entity_rows = entities(2)
    world = {atom: False for atom in atom_universe(entity_rows)}
    observation = world_signature(world)
    registry = mechanic_registry(5303)[:1]
    policy = _wait_policy(observation)
    atoms = [{
        "program_index": 0,
        "node_index": 0,
        "theta": 0.5,
        "configuration_key": "synthetic",
        "world": world,
        "queue": [],
        "weight": 1.0,
    }]
    goal = {"atom": sorted(world)[0], "value": False}
    model = compile_policy_dtmc(
        atoms, policy, registry, entity_rows, goal, 3, 0, config
    )
    metadata = {
        "frozen_root_action_key": "wait",
        "reconstructed_root_action_key": "wait",
        "reconstructed_root_value_error": 0.0,
        "direct_executor": {
            "success_probability": 1.0,
            "expected_return": 1.0,
        },
        "frozen_root_value": 1.0,
        "independent_policy_value": 1.0,
    }
    manifest_row = {
        "cohort": "synthetic",
        "id": "synthetic_wait",
        "record": 0,
        "states": len(model["states"]),
        "transitions": len(model["transitions"]),
        "frozen_root_action_key": "wait",
        "reconstructed_root_value_error": 0.0,
    }
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)
        write_policy_bundle(model, policy, directory, metadata)
        return evaluate_policy_model(directory, manifest_row)


def _passing_records(fixture: dict) -> list[dict]:
    rows = []
    for index in range(48):
        row = json.loads(json.dumps(fixture))
        row["cohort"] = "v55" if index < 32 else "v55r1"
        row["id"] = f"stub_{index:05d}"
        row["record"] = index if index < 32 else index - 32
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle-seal", default="configs/v56-verification-bundle-seal.json"
    )
    parser.add_argument(
        "--output",
        default="outputs/v56-symbolic-probabilistic-policy-verification/evaluation-implementation-audit.json",
    )
    args = parser.parse_args()
    seal_path = (PROJECT_ROOT / args.bundle_seal).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    seal = json.loads(seal_path.read_text())
    implementation_path = PROJECT_ROOT / seal["implementation_lock"]
    implementation = json.loads(implementation_path.read_text())
    design = json.loads(
        (PROJECT_ROOT / implementation["design_lock"]).read_text()
    )
    config = design["config_payload"]
    implementation_audit = json.loads(
        (PROJECT_ROOT / implementation["implementation_audit"]).read_text()
    )
    errors: list[str] = []

    manifest_path = PROJECT_ROOT / seal["manifest"]
    manifest = json.loads(manifest_path.read_text())
    seal_bound = (
        seal["authorization"]["write_and_audit_v56_candidate_runner"]
        and not seal["authorization"]["run_v56_candidate_formal_verification"]
        and not seal["authorization"]["modify_v56_bundle"]
        and file_sha256(manifest_path) == seal["manifest_sha256"]
        and file_sha256(PROJECT_ROOT / seal["bundle_audit"])
        == seal["bundle_audit_sha256"]
        and file_sha256(implementation_path) == seal["implementation_lock_sha256"]
        and manifest["policy_count"] == 48
        and manifest["cohort_counts"] == {"v55": 32, "v55r1": 16}
    )
    if not seal_bound:
        errors.append("V56 verification bundle seal is not intact")

    source = inspect.getsource(evaluate_policy_model)
    evaluator_firewall = (
        set(inspect.signature(evaluate_policy_model).parameters)
        == {"directory", "manifest_row"}
        and all(name not in source for name in (
            "exact_inference(", "plan_exact(", "evaluate_policy(",
            '["truth"]', "['truth']"
        ))
    )
    if not evaluator_firewall:
        errors.append("V56 candidate evaluator crosses the frozen-policy firewall")

    fixture = _synthetic_runner_fixture(config)
    fixture_ok = (
        fixture["completed"]
        and fixture["reconstructed_root_action_match"]
        and fixture["reconstructed_root_value_error"] == 0.0
        and fixture["symbolic"]["invariant_checks"]
        == fixture["symbolic"]["invariant_passes"]
        and fixture["symbolic"]["support_checks"]
        == fixture["symbolic"]["support_passes"]
        and fixture["symbolic"]["totality_checks"]
        == fixture["symbolic"]["totality_passes"]
        and fixture["symbolic"]["z3_unknown_count"] == 0
        and fixture["symbolic"]["nonterminal_deadlock_count"] == 0
        and fixture["termination_probability_error"] <= 1e-12
        and fixture["success_probability_error"] <= 1e-12
        and fixture["expected_return_error_against_frozen_value"] <= 1e-12
        and fixture[
            "expected_return_error_against_independent_policy_evaluator"
        ] <= 1e-12
        and fixture["transition_distribution_normalized"]
        and fixture["finite"]
    )
    if not fixture_ok:
        errors.append("V56 altered synthetic end-to-end runner fixture failed")

    controls = {
        "mutation_kill_rate": implementation_audit["mutation_controls"][
            "kill_rate"
        ],
        "analytic_fixture_pass_rate": sum(
            row["passed"] for row in implementation_audit["analytic_storm_fixtures"]
        ) / len(implementation_audit["analytic_storm_fixtures"]),
    }
    integrity = {
        "truth_field_access_count": 0,
        "source_result_mutation_count": 0,
        "verification_bundle_hash_mismatch_count": 0,
        "tool_version_mismatch_count": 0,
        "unexpected_verification_attempt_count": 0,
    }
    aggregated = aggregate(_passing_records(fixture), config, integrity, controls)
    aggregate_ok = (
        len(aggregated["checks"]) == len(config["gates"]) == 25
        and set(aggregated["checks"]) == {
            "completed_policy_fraction", "policy_count", "v55_policy_count",
            "v55r1_policy_count", "reconstructed_root_action_match_rate",
            "reconstructed_root_value_error",
            "reachable_state_invariant_proof_rate",
            "reachable_transition_support_equivalence_proof_rate",
            "policy_observation_totality_rate", "nonterminal_deadlock_count",
            "z3_unknown_count", "storm_completed_model_fraction",
            "termination_probability_error", "success_probability_error",
            "expected_return_error_against_frozen_value",
            "expected_return_error_against_independent_policy_evaluator",
            "transition_distribution_normalization_rate", "finite_result_rate",
            "implementation_mutant_kill_rate", "analytic_fixture_pass_rate",
            "truth_field_access_count", "source_result_mutation_count",
            "verification_bundle_hash_mismatch_count",
            "tool_version_mismatch_count",
            "unexpected_verification_attempt_count",
        }
        and aggregated["passed"]
    )
    if not aggregate_ok:
        errors.append("V56 25-gate noncompensatory aggregation is invalid")

    observed_versions = tool_versions()
    versions_ok = observed_versions == implementation["tool_versions"]
    if not versions_ok:
        errors.append("V56 pinned tool versions changed")

    single_attempt_ok = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v56-evaluation-implementation-lock.json",
            "configs/v56-outcome-lock.json",
            "outputs/v56-symbolic-probabilistic-policy-verification/evaluation-attempt.json",
            "outputs/v56-symbolic-probabilistic-policy-verification/evaluation",
            "outputs/v56-symbolic-probabilistic-policy-verification/post-result-audit.json",
            "docs/v56-results.md",
        )
    )
    if not single_attempt_ok:
        errors.append("V56 candidate verification or downstream artifact already exists")

    audit = {
        "schema_version": 56,
        "experiment": "v56_evaluation_implementation_audit",
        "passed": not errors,
        "decision": (
            "authorize_v56_evaluation_implementation_lock" if not errors
            else "repair_v56_evaluation_implementation"
        ),
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
            "twenty_five_noncompensatory_qualification_checks": aggregate_ok,
            "pinned_tool_versions": versions_ok,
            "single_attempt_and_downstream_absence": single_attempt_ok,
        },
        "fixture_metrics": {
            "states": fixture["states"],
            "transitions": fixture["transitions"],
            "symbolic_invariant_checks": fixture["symbolic"]["invariant_checks"],
            "symbolic_support_checks": fixture["symbolic"]["support_checks"],
            "maximum_storm_reference_error": max(
                fixture["termination_probability_error"],
                fixture["success_probability_error"],
                fixture["expected_return_error_against_frozen_value"],
            ),
            "qualification_check_count": len(aggregated["checks"]),
        },
        "data_access": {
            "v56_candidate_policy_models_accessed": 0,
            "v56_candidate_formal_verification_runs": 0,
            "altered_synthetic_runner_fixture_models": 1,
            "truth_field_access_count": 0,
            "additional_v55_or_v55r1_evaluation_runs": 0,
        },
    }
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
