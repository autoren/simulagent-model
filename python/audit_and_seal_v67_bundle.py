#!/usr/bin/env python3
"""Independently audit and seal the exhaustive V67 verification bundle."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v67_verification import canonical_json, policy_tree_hash


REQUIRED_FILES = (
    "model.tra", "model.lab", "model.rew", "model.meta.json", "policy-tree.json"
)


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def json_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def parse_explicit(directory: Path) -> dict[str, Any]:
    transition_lines = (directory / "model.tra").read_text().splitlines()
    if not transition_lines or transition_lines[0] != "dtmc":
        raise ValueError("explicit model is not a DTMC")
    transitions = []
    for line in transition_lines[1:]:
        source, target, probability = line.split()
        transitions.append((int(source), int(target), float(probability)))
    rewards = []
    for line in (directory / "model.rew").read_text().splitlines():
        source, target, reward = line.split()
        rewards.append((int(source), int(target), float(reward)))
    if [(a, b) for a, b, _ in transitions] != [(a, b) for a, b, _ in rewards]:
        raise ValueError("transition and reward edge orders disagree")
    grouped: dict[int, float] = {}
    for source, _, probability in transitions:
        grouped[source] = grouped.get(source, 0.0) + probability
    labels = (directory / "model.lab").read_text()
    return {
        "transition_count": len(transitions),
        "finite": all(math.isfinite(value) for _, _, value in transitions + rewards),
        "normalized_rows": all(abs(value - 1.0) <= 1e-10 for value in grouped.values()),
        "init_label": " init" in labels,
        "done_label": " done" in labels,
    }


def main() -> None:
    implementation_path = PROJECT_ROOT / "configs/v67-implementation-lock.json"
    implementation = json.loads(implementation_path.read_text())
    implementation_payload = {
        key: value for key, value in implementation.items()
        if key != "lock_payload_sha256"
    }
    bundle_root = PROJECT_ROOT / "outputs/v67-independent-bounded-policy-verification/bundle"
    manifest_path = bundle_root / "bundle-manifest.json"
    audit_path = PROJECT_ROOT / "outputs/v67-independent-bounded-policy-verification/bundle-audit.json"
    seal_path = PROJECT_ROOT / "configs/v67-verification-bundle-seal.json"
    if seal_path.exists():
        raise RuntimeError("V67 bundle already sealed")
    manifest = json.loads(manifest_path.read_text())
    design_path = PROJECT_ROOT / implementation["design_lock"]
    design = json.loads(design_path.read_text())
    source_cells_path = PROJECT_ROOT / design["source_v66_record_cells"]
    source_rows = [
        json.loads(line) for line in source_cells_path.read_text().splitlines() if line
    ]
    source_by_id = {str(row["record_id"]): row for row in source_rows}
    errors: list[str] = []

    authorization_ok = bool(
        payload_hash(implementation_payload) == implementation["lock_payload_sha256"]
        and implementation["authorization"]["build_verification_bundle"]
        and implementation["authorization"]["load_and_execute_all_source_policies"]
        and not implementation["authorization"]["run_verification"]
        and manifest["implementation_lock_sha256"] == file_sha256(implementation_path)
    )
    if not authorization_ok:
        errors.append("V67 implementation lock or bundle authorization failed")

    source_hashes_ok = bool(
        manifest["source_v66_record_cells_sha256"] == file_sha256(source_cells_path)
        == design["source_v66_record_cells_sha256"]
        and manifest["source_v66_result_sha256"]
        == file_sha256(PROJECT_ROOT / design["source_v66_result"])
        == design["source_v66_result_sha256"]
        and manifest["source_v66_outcome_lock_sha256"]
        == file_sha256(PROJECT_ROOT / design["source_v66_outcome_lock"])
        == design["source_v66_outcome_lock_sha256"]
        and manifest["public_subset_sha256"]
        == file_sha256(PROJECT_ROOT / manifest["public_subset"])
        and manifest["pinned_source_model_sha256"]
        == file_sha256(PROJECT_ROOT / manifest["pinned_source_model"])
    )
    if not source_hashes_ok:
        errors.append("V67 bundle source hash binding failed")

    policies = manifest["policies"]
    census_ok = bool(
        manifest["policy_count"] == len(policies) == 96
        and manifest["record_count"] == len(manifest["records"]) == 48
        and sum(row["policy_kind"] == "exact_policy" for row in policies) == 48
        and sum(row["policy_kind"] == "pooled_SMC2_policy" for row in policies) == 48
        and len({row["policy_id"] for row in policies}) == 96
    )
    if not census_ok:
        errors.append("V67 bundle policy census is incomplete")

    file_mismatches = 0
    policy_binding_passes = 0
    record_binding_passes = 0
    explicit_passes = 0
    belief_passes = 0
    invariant_passes = 0
    totality_passes = 0
    normalization_passes = 0
    finite_passes = 0
    nonterminal_deadlocks = 0
    scalar_errors = []
    compiler_errors = []
    for row in policies:
        directory = bundle_root / row["directory"]
        for name in REQUIRED_FILES:
            path = directory / name
            registered = row["files"].get(name, {})
            if (
                not path.exists()
                or registered.get("sha256") != file_sha256(path)
                or registered.get("size") != path.stat().st_size
            ):
                file_mismatches += 1
        policy = json.loads((directory / "policy-tree.json").read_text())
        meta = json.loads((directory / "model.meta.json").read_text())
        source = source_by_id.get(str(row["record_id"]))
        if source is not None and (
            row["source_row_hash"] == json_hash(source)
            and meta["source_row_hash"] == json_hash(source)
        ):
            record_binding_passes += 1
        if source is not None and (
            policy == source[row["policy_kind"]]
            and row["policy_tree_hash"] == policy_tree_hash(policy)
            and meta["policy_tree_hash"] == policy_tree_hash(policy)
        ):
            policy_binding_passes += 1
        explicit = parse_explicit(directory)
        explicit_passes += int(
            explicit["init_label"] and explicit["done_label"]
            and explicit["transition_count"] == meta["model_summary"]["transitions"]
        )
        belief_passes += int(abs(float(meta["exact_root_belief_mass"]) - 1.0) <= 1e-10)
        checks = meta["compiler_checks"]
        scalar_checks = meta["scalar_checks"]
        invariant_passes += int(
            checks["node_invariants"] == checks["node_invariant_passes"]
            and scalar_checks["node_invariants"] == scalar_checks["node_invariant_passes"]
        )
        totality_passes += int(
            checks["branch_totality_checks"] == checks["branch_totality_passes"]
            and scalar_checks["total_branches"] == scalar_checks["total_branch_passes"]
        )
        normalization_passes += int(
            explicit["normalized_rows"]
            and checks["transition_normalization_checks"]
            == checks["transition_normalization_passes"]
        )
        nonterminal_deadlocks += int(checks["nonterminal_deadlocks"])
        finite_passes += int(explicit["finite"] and meta["model_summary"]["finite"])
        scalar_errors.append(float(meta["absolute_scalar_error_against_frozen_V66"]))
        compiler_errors.append(float(meta["absolute_compiler_error_against_scalar"]))

    pair_errors = [
        abs(
            float(row["independent_exact_minus_SMC2_value"])
            - float(row["frozen_V66_exact_minus_SMC2_value"])
        )
        for row in manifest["records"]
    ]
    digest_input = [
        {
            "policy_id": row["policy_id"],
            "files": row["files"],
            "source_row_hash": row["source_row_hash"],
            "policy_tree_hash": row["policy_tree_hash"],
        }
        for row in policies
    ]
    digest_ok = manifest["bundle_file_digest"] == payload_hash({"files": digest_input})
    gates = design["config_payload"]["gates"]
    gate_results = {
        "policy_census": census_ok,
        "source_policy_hash_match_rate": policy_binding_passes / 96 == 1.0,
        "source_record_binding_rate": record_binding_passes / 96 == 1.0,
        "exact_root_belief_normalization_rate": belief_passes / 96 == 1.0,
        "reachable_policy_node_invariant_rate": invariant_passes / 96 == 1.0,
        "positive_observation_branch_totality_rate": totality_passes / 96 == 1.0,
        "transition_distribution_normalization_rate": normalization_passes / 96 == 1.0,
        "nonterminal_deadlock_count": nonterminal_deadlocks == 0,
        "finite_result_rate": finite_passes / 96 == 1.0,
        "independent_executor_against_frozen_V66": max(scalar_errors) <= gates[
            "maximumIndependentExecutorErrorAgainstFrozenV66Value"
        ],
        "compiler_against_scalar_executor": max(compiler_errors) <= 1e-10,
        "exact_minus_SMC2_pairs_reproduced": max(pair_errors) <= gates[
            "maximumReproducedExactMinusSMC2PairErrorAgainstV66"
        ],
        "bundle_file_hashes": file_mismatches == 0 and digest_ok,
        "source_hashes": source_hashes_ok,
        "implementation_authorization": authorization_ok,
        "no_early_Storm_runs": manifest["access"]["Storm_source_policy_runs"] == 0,
        "truth_human_model_and_adapter_access_zero": all(
            manifest["access"][key] == 0 for key in (
                "truth_fields", "human_records", "model_forward_passes",
                "adapter_training_runs",
            )
        ),
    }
    if not all(gate_results.values()):
        errors.append("one or more V67 bundle gates failed")
    if (PROJECT_ROOT / "outputs/v67-independent-bounded-policy-verification/verification").exists():
        errors.append("V67 verification exists before bundle seal")

    audit = {
        "schema_version": "67",
        "experiment": "v67_bundle_audit",
        "passed": not errors,
        "decision": "seal_bundle_and_authorize_durable_evaluator_only" if not errors else "reject_bundle",
        "errors": errors,
        "gate_results": gate_results,
        "metrics": {
            "policy_count": len(policies),
            "policy_count_by_kind": {
                kind: sum(row["policy_kind"] == kind for row in policies)
                for kind in ("exact_policy", "pooled_SMC2_policy")
            },
            "source_policy_hash_match_rate": policy_binding_passes / 96,
            "source_record_binding_rate": record_binding_passes / 96,
            "exact_root_belief_normalization_rate": belief_passes / 96,
            "reachable_policy_node_invariant_rate": invariant_passes / 96,
            "positive_observation_branch_totality_rate": totality_passes / 96,
            "transition_distribution_normalization_rate": normalization_passes / 96,
            "nonterminal_deadlock_count": nonterminal_deadlocks,
            "finite_result_rate": finite_passes / 96,
            "maximum_independent_executor_error_against_frozen_V66_value": max(scalar_errors),
            "maximum_compiler_error_against_scalar_executor": max(compiler_errors),
            "maximum_reproduced_exact_minus_SMC2_pair_error_against_V66": max(pair_errors),
            "verification_bundle_hash_mismatch_count": file_mismatches + int(not digest_ok),
        },
        "access": manifest["access"],
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if errors:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    seal = {
        "schema_version": "67",
        "experiment": "v67_verification_bundle_seal",
        "implementation_lock": str(implementation_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(implementation_path),
        "bundle_manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "bundle_manifest_sha256": file_sha256(manifest_path),
        "bundle_file_digest": manifest["bundle_file_digest"],
        "bundle_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "bundle_audit_sha256": file_sha256(audit_path),
        "source_v66_record_cells_sha256": manifest["source_v66_record_cells_sha256"],
        "policy_count": 96,
        "authorization": {
            "modify_or_rerun_v66": False,
            "modify_v67_design_implementation_or_bundle": False,
            "write_and_audit_durable_evaluator": True,
            "run_verification": False,
            "truth_field_access": False,
            "human_data": False,
            "model_access": False,
            "adapter_training": False,
        },
    }
    seal["lock_payload_sha256"] = payload_hash(seal)
    seal_path.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"bundle_seal": str(seal_path), "sha256": file_sha256(seal_path)}, indent=2))


if __name__ == "__main__":
    main()
