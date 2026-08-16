#!/usr/bin/env python3
"""Audit the V62 candidate runner before its single external evaluation."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from evaluate_v62_external import aggregate
from test_v62_external_pomdp import make_tmaze
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def model_payload(model) -> dict[str, object]:
    return {
        "name": model.name,
        "states": list(model.states),
        "actions": list(model.actions),
        "observations": list(model.observations),
        "discount": model.discount,
        "initial": model.initial.tolist(),
        "transition": model.transition.tolist(),
        "observation": model.observation.tolist(),
        "reward": model.reward.tolist(),
    }


def synthetic_aggregate_inputs(config: dict[str, object]):
    exact_records = []
    for entry in config["benchmark"]["models"]:
        for horizon in entry["horizons"]:
            is_tmaze = entry["id"].startswith("tmaze")
            values = {
                "exact_history": 3.0 if is_tmaze else 5.0,
                "observation_only": 1.0 if is_tmaze else 0.0,
                "map_collapse": 3.0 if is_tmaze else (5.0 if horizon == 1 else 0.0),
                "fully_observed_oracle": 3.0 if is_tmaze else 10.0,
                "uniform_random": 0.0,
            }
            exact_records.append({
                "model_id": entry["id"],
                "horizon": horizon,
                "candidate_value": values["exact_history"],
                "reference_value": values["exact_history"],
                "candidate_reference_value_error": 0.0,
                "root_actions": {"0": 2 if is_tmaze else 0},
                "root_optimal_set_membership_rate": 1.0,
                "reachable_belief_count": 1,
                "reachable_belief_normalization_rate": 1.0,
                "maximum_bellman_residual": 0.0,
                "terminal_detection_agreement": True,
                "validation": {
                    "transition_normalized": True,
                    "observation_normalized": True,
                    "initial_normalized": True,
                    "finite_reward_and_discount": True,
                },
                "exact_policy_values": values,
                "return_range": [-100.0, 10.0],
            })
    rollout_records = []
    for record in exact_records:
        for policy in config["externalRollout"]["policies"]:
            value = record["exact_policy_values"][policy]
            rollout_records.append({
                "model_id": record["model_id"],
                "horizon": record["horizon"],
                "policy": policy,
                "episodes": config["externalRollout"]["episodesPerTaskPolicy"],
                "seed": 1,
                "mean_return": value,
                "minimum_return": value,
                "maximum_return": value,
                "finite_return_rate": 1.0,
                "terminated_fraction": 1.0,
            })
    integrity = {
        "source_bundle_hash_mismatch_count": 0,
        "upstream_source_mutation_count": 0,
        "tool_version_mismatch_count": 0,
        "unexpected_evaluation_attempt_count": 0,
        "human_record_access_count": 0,
        "model_forward_pass_count": 0,
    }
    controls = {
        "external_source_binding_rate": 1.0,
        "license_binding_rate": 1.0,
        "independent_parser_agreement_rate": 1.0,
        "maximum_transition_array_error": 0.0,
        "maximum_observation_array_error": 0.0,
        "maximum_reward_array_error": 0.0,
        "maximum_initial_belief_error": 0.0,
        "maximum_discount_error": 0.0,
        "implementation_mutant_kill_rate": 1.0,
        "analytic_fixture_pass_rate": 1.0,
    }
    return exact_records, rollout_records, integrity, controls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-seal", default="configs/v62-external-bundle-seal.json")
    parser.add_argument("--runtime-python", default="data/v62-external-pomdp-transfer/runtime/bin/python")
    parser.add_argument(
        "--output", default="outputs/v62-external-pomdp-transfer/evaluation-implementation-audit.json"
    )
    args = parser.parse_args()
    seal_path = (PROJECT_ROOT / args.bundle_seal).resolve()
    runtime_python = PROJECT_ROOT / args.runtime_python
    output = (PROJECT_ROOT / args.output).resolve()
    seal = json.loads(seal_path.read_text())
    implementation_path = PROJECT_ROOT / seal["implementation_lock"]
    implementation = json.loads(implementation_path.read_text())
    design = json.loads((PROJECT_ROOT / implementation["design_lock"]).read_text())
    config = design["config_payload"]
    bundle = PROJECT_ROOT / seal["bundle"]
    manifest_path = PROJECT_ROOT / seal["manifest"]
    manifest = json.loads(manifest_path.read_text())
    errors: list[str] = []

    seal_ok = (
        seal["authorization"]["write_and_audit_evaluation_implementation"]
        and not seal["authorization"]["run_one_candidate_evaluation"]
        and file_sha256(manifest_path) == seal["manifest_sha256"]
        and file_sha256(implementation_path) == seal["implementation_lock_sha256"]
        and all(
            file_sha256(bundle / relative) == binding["sha256"]
            for relative, binding in manifest["files"].items()
        )
    )
    if not seal_ok:
        errors.append("V62 external bundle seal is not intact")

    actual_evaluation_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "outputs/v62-external-pomdp-transfer/evaluation-attempt.json",
            "outputs/v62-external-pomdp-transfer/evaluation",
            "outputs/v62-external-pomdp-transfer/post-result-audit.json",
            "configs/v62-evaluation-implementation-lock.json",
            "configs/v62-outcome-lock.json",
            "docs/v62-results.md",
        )
    )
    if not actual_evaluation_absent:
        errors.append("V62 candidate evaluation or downstream artifacts already exist")

    exact, rollout, integrity, controls = synthetic_aggregate_inputs(config)
    aggregate_result = aggregate(exact, rollout, config, integrity, controls)
    aggregate_ok = (
        aggregate_result["passed"]
        and len(aggregate_result["checks"]) == 32
        and all(aggregate_result["checks"].values())
    )
    mutated_integrity = dict(integrity)
    mutated_integrity["human_record_access_count"] = 1
    mutated = aggregate(*synthetic_aggregate_inputs(config)[:2], config, mutated_integrity, controls)
    noncompensatory_ok = not mutated["passed"] and not mutated["checks"]["human_record_access_count"]
    if not aggregate_ok or not noncompensatory_ok:
        errors.append("V62 aggregation does not implement all noncompensatory gates")

    with tempfile.TemporaryDirectory(prefix="v62-official-smoke-") as temp_dir:
        synthetic_bundle = Path(temp_dir) / "bundle"
        runtime_destination = synthetic_bundle / "source/pobax/envs/classic/pomdp.py"
        runtime_destination.parent.mkdir(parents=True)
        shutil.copy2(bundle / "source/pobax/envs/classic/pomdp.py", runtime_destination)
        model = make_tmaze(2)
        model_path = synthetic_bundle / "models/synthetic/model.json"
        model_path.parent.mkdir(parents=True)
        model_path.write_text(json.dumps(model_payload(model), indent=2, sort_keys=True) + "\n")
        request_path = Path(temp_dir) / "request.json"
        result_path = Path(temp_dir) / "result.json"
        request_path.write_text(json.dumps({"cells": [{
            "model_id": "synthetic", "horizon": 4, "policy": "exact_history",
            "episodes": 64, "seed": 6203,
        }]}) + "\n")
        completed = subprocess.run(
            [
                str(runtime_python), str(PROJECT_ROOT / "python/official_v62_rollout.py"),
                "--bundle", str(synthetic_bundle), "--request", str(request_path),
                "--output", str(result_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        smoke = json.loads(result_path.read_text()) if result_path.exists() else {}
        expected = 4.0 * 0.9**3
        official_smoke_ok = (
            completed.returncode == 0
            and len(smoke.get("records", [])) == 1
            and abs(smoke["records"][0]["mean_return"] - expected) <= 1e-12
            and smoke["records"][0]["finite_return_rate"] == 1.0
        )
    if not official_smoke_ok:
        errors.append("pinned POBAX runtime synthetic smoke failed")

    evaluation_files = [
        "python/official_v62_rollout.py",
        "python/evaluate_v62_external.py",
        "python/audit_and_summarize_v62.py",
        "python/freeze_v62_outcome.py",
    ]
    frozen_dependencies = [
        "python/v62_external_pomdp.py",
        "python/audit_v62_implementation.py",
        "python/v10_protocol.py",
        "python/v22r2_grounding.py",
    ]
    forbidden_network_tokens = ("git clone", "git fetch", "http://", "https://", "curl ", "wget ")
    evaluation_text = "\n".join((PROJECT_ROOT / path).read_text() for path in evaluation_files[:2])
    offline_ok = all(token not in evaluation_text for token in forbidden_network_tokens)
    if not offline_ok:
        errors.append("candidate evaluation implementation contains network retrieval")

    result = {
        "schema_version": 62,
        "experiment": "v62_evaluation_implementation_audit",
        "passed": not errors,
        "decision": "freeze_v62_evaluation_implementation" if not errors else "repair_v62_evaluation_implementation",
        "errors": errors,
        "external_bundle_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "external_bundle_seal_sha256": file_sha256(seal_path),
        "implementation_lock": str(implementation_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(implementation_path),
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "manifest_sha256": file_sha256(manifest_path),
        "evaluation_files_sha256": {
            path: file_sha256(PROJECT_ROOT / path) for path in evaluation_files
        },
        "frozen_dependencies_sha256": {
            path: file_sha256(PROJECT_ROOT / path) for path in frozen_dependencies
        },
        "runtime_versions": implementation["runtime_versions"],
        "synthetic_aggregate_gate_count": len(aggregate_result["checks"]),
        "official_runtime_synthetic_mean": smoke.get("records", [{}])[0].get("mean_return"),
        "official_runtime_smoke_stderr": completed.stderr[-4000:],
        "official_runtime_smoke_stdout": completed.stdout[-4000:],
        "checks": {
            "sealed_external_bundle_binding": seal_ok,
            "actual_candidate_evaluation_absent": actual_evaluation_absent,
            "all_thirty_two_gates_aggregate": aggregate_ok,
            "noncompensatory_integrity_rejection": noncompensatory_ok,
            "pinned_official_runtime_synthetic_smoke": official_smoke_ok,
            "offline_candidate_evaluation": offline_ok,
        },
        "data_access": {
            "external_runtime_source_files_read": 1,
            "external_model_definition_files_read": 0,
            "external_candidate_evaluations": 0,
            "human_authored_v58_records": 0,
            "model_forward_passes": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
