#!/usr/bin/env python3
"""Build 96 independently compiled V67 verification bundles."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v67_verification import (
    canonical_json,
    compile_policy_dtmc,
    condition_public_history,
    dtmc_statistics,
    execute_policy_scalar,
    file_sha256 as independent_file_sha256,
    finite_dtmc,
    load_pinned_family,
    policy_tree_hash,
    write_explicit_dtmc,
)


POLICY_SPECS = (
    ("exact_policy", "exact_Bayes_adaptive"),
    ("pooled_SMC2_policy", "pooled_SMC2_Bayes_adaptive_exact_environment"),
)
REQUIRED_FILES = (
    "model.tra", "model.lab", "model.rew", "model.meta.json", "policy-tree.json"
)


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def json_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    implementation_lock_path = PROJECT_ROOT / "configs/v67-implementation-lock.json"
    implementation_lock = json.loads(implementation_lock_path.read_text())
    implementation_payload = {
        key: value for key, value in implementation_lock.items()
        if key != "lock_payload_sha256"
    }
    if payload_hash(implementation_payload) != implementation_lock["lock_payload_sha256"]:
        raise RuntimeError("V67 implementation lock payload mismatch")
    if not (
        implementation_lock["authorization"]["load_and_execute_all_source_policies"]
        and implementation_lock["authorization"]["build_verification_bundle"]
        and not implementation_lock["authorization"]["run_verification"]
    ):
        raise RuntimeError("V67 implementation lock does not authorize bundle construction")
    design_lock_path = PROJECT_ROOT / implementation_lock["design_lock"]
    design_lock = json.loads(design_lock_path.read_text())
    if file_sha256(design_lock_path) != implementation_lock["design_lock_sha256"]:
        raise RuntimeError("V67 design lock hash mismatch")
    source_lock_path = PROJECT_ROOT / design_lock["source_v66_outcome_lock"]
    source_lock = json.loads(source_lock_path.read_text())
    source_cells_path = PROJECT_ROOT / design_lock["source_v66_record_cells"]
    source_result_path = PROJECT_ROOT / design_lock["source_v66_result"]
    public_path = PROJECT_ROOT / design_lock["config_payload"]["sourcePolicies"]["publicSubset"]
    if file_sha256(source_cells_path) != design_lock["source_v66_record_cells_sha256"]:
        raise RuntimeError("frozen V66 record-cell hash mismatch")
    if file_sha256(source_result_path) != design_lock["source_v66_result_sha256"]:
        raise RuntimeError("frozen V66 result hash mismatch")
    if file_sha256(source_lock_path) != design_lock["source_v66_outcome_lock_sha256"]:
        raise RuntimeError("frozen V66 outcome-lock hash mismatch")

    rows = [json.loads(line) for line in source_cells_path.read_text().splitlines() if line]
    public = [json.loads(line) for line in public_path.read_text().splitlines() if line]
    public_by_id = {str(row["record_id"]): row for row in public}
    if len(rows) != 48 or len(public) != 48 or len(public_by_id) != 48:
        raise RuntimeError("V67 requires exactly 48 unique source records")
    row_ids = [str(row["record_id"]) for row in rows]
    if len(set(row_ids)) != 48 or set(row_ids) != set(public_by_id):
        raise RuntimeError("V66 cells and public subset record IDs disagree")

    family = load_pinned_family(PROJECT_ROOT, nodes=257)
    output_root = PROJECT_ROOT / "outputs/v67-independent-bounded-policy-verification"
    final_bundle = output_root / "bundle"
    if final_bundle.exists():
        raise RuntimeError("V67 bundle already exists")
    output_root.mkdir(parents=True, exist_ok=True)
    policy_manifest: list[dict[str, Any]] = []
    record_manifest: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="bundle-build-", dir=output_root) as temporary:
        temporary_root = Path(temporary)
        for record_index, source_row in enumerate(rows):
            record_id = str(source_row["record_id"])
            public_record = public_by_id[record_id]
            if int(public_record["prefix_length"]) != int(source_row["prefix_length"]):
                raise RuntimeError(f"prefix mismatch for {record_id}")
            belief, evidence = condition_public_history(family, public_record)
            if abs(float(belief.sum()) - 1.0) > 1e-10 or evidence <= 0.0:
                raise RuntimeError(f"invalid exact root belief for {record_id}")
            source_row_hash = json_hash(source_row)
            public_record_hash = json_hash(public_record)
            per_kind_values: dict[str, float] = {}
            for policy_kind, value_key in POLICY_SPECS:
                policy = source_row[policy_kind]
                policy_id = f"{record_index:03d}-{record_id}-{policy_kind}"
                directory = temporary_root / policy_id
                scalar = execute_policy_scalar(family, belief, policy, horizon=3)
                model, compiler_checks = compile_policy_dtmc(
                    family, belief, policy, horizon=3
                )
                compiled = dtmc_statistics(model)
                frozen_value = float(source_row["strategy_values"][value_key])
                scalar_value = float(scalar["value"])
                per_kind_values[policy_kind] = scalar_value
                write_explicit_dtmc(model, directory)
                policy_path = directory / "policy-tree.json"
                write_json(policy_path, policy)
                meta = {
                    "schema_version": "67",
                    "experiment": "v67_policy_bundle",
                    "policy_id": policy_id,
                    "record_index": record_index,
                    "record_id": record_id,
                    "prefix_length": int(source_row["prefix_length"]),
                    "policy_kind": policy_kind,
                    "source_value_key": value_key,
                    "source_row_hash": source_row_hash,
                    "public_record_hash": public_record_hash,
                    "policy_tree_hash": policy_tree_hash(policy),
                    "exact_history_evidence": float(evidence),
                    "exact_root_belief_mass": float(belief.sum()),
                    "frozen_V66_exact_environment_value": frozen_value,
                    "independent_scalar_value": scalar_value,
                    "compiled_direct_value": float(compiled["expected_return"]),
                    "compiled_direct_termination_probability": float(
                        compiled["termination_probability"]
                    ),
                    "absolute_scalar_error_against_frozen_V66": abs(
                        scalar_value - frozen_value
                    ),
                    "absolute_compiler_error_against_scalar": abs(
                        float(compiled["expected_return"]) - scalar_value
                    ),
                    "scalar_checks": scalar,
                    "compiler_checks": compiler_checks,
                    "model_summary": {
                        "states": len(model["states"]),
                        "transitions": len(model["transitions"]),
                        "root_state": int(model["root_state"]),
                        "done_state": int(model["done_state"]),
                        "finite": finite_dtmc(model),
                    },
                    "access": {
                        "public_records": 1,
                        "source_policy_trees": 1,
                        "truth_fields": 0,
                        "human_records": 0,
                        "model_forward_passes": 0,
                        "adapter_training_runs": 0,
                    },
                }
                meta_path = directory / "model.meta.json"
                write_json(meta_path, meta)
                files = {
                    name: {
                        "sha256": independent_file_sha256(directory / name),
                        "size": (directory / name).stat().st_size,
                    }
                    for name in REQUIRED_FILES
                }
                policy_manifest.append({
                    "policy_id": policy_id,
                    "directory": policy_id,
                    "record_index": record_index,
                    "record_id": record_id,
                    "prefix_length": int(source_row["prefix_length"]),
                    "policy_kind": policy_kind,
                    "source_row_hash": source_row_hash,
                    "public_record_hash": public_record_hash,
                    "policy_tree_hash": policy_tree_hash(policy),
                    "frozen_V66_value": frozen_value,
                    "independent_scalar_value": scalar_value,
                    "compiled_direct_value": float(compiled["expected_return"]),
                    "files": files,
                })
            record_manifest.append({
                "record_index": record_index,
                "record_id": record_id,
                "prefix_length": int(source_row["prefix_length"]),
                "source_row_hash": source_row_hash,
                "public_record_hash": public_record_hash,
                "independent_exact_minus_SMC2_value": (
                    per_kind_values["exact_policy"]
                    - per_kind_values["pooled_SMC2_policy"]
                ),
                "frozen_V66_exact_minus_SMC2_value": float(
                    source_row["primary"]["exact_value_regret"]
                ),
            })
            print(f"built {record_index + 1}/48 records ({len(policy_manifest)}/96 policies)", flush=True)
        digest_input = [
            {
                "policy_id": row["policy_id"],
                "files": row["files"],
                "source_row_hash": row["source_row_hash"],
                "policy_tree_hash": row["policy_tree_hash"],
            }
            for row in policy_manifest
        ]
        manifest = {
            "schema_version": "67",
            "experiment": "v67_verification_bundle_manifest",
            "policy_count": len(policy_manifest),
            "record_count": len(record_manifest),
            "policy_kinds": [value[0] for value in POLICY_SPECS],
            "implementation_lock": str(implementation_lock_path.relative_to(PROJECT_ROOT)),
            "implementation_lock_sha256": file_sha256(implementation_lock_path),
            "design_lock": str(design_lock_path.relative_to(PROJECT_ROOT)),
            "design_lock_sha256": file_sha256(design_lock_path),
            "source_v66_outcome_lock": str(source_lock_path.relative_to(PROJECT_ROOT)),
            "source_v66_outcome_lock_sha256": file_sha256(source_lock_path),
            "source_v66_record_cells": str(source_cells_path.relative_to(PROJECT_ROOT)),
            "source_v66_record_cells_sha256": file_sha256(source_cells_path),
            "source_v66_result": str(source_result_path.relative_to(PROJECT_ROOT)),
            "source_v66_result_sha256": file_sha256(source_result_path),
            "public_subset": str(public_path.relative_to(PROJECT_ROOT)),
            "public_subset_sha256": file_sha256(public_path),
            "pinned_source_model": str(
                (PROJECT_ROOT / "data/v63-external-unknown-dynamics/source-checkout/pobax/"
                 "envs/classic/POMDP/4x3_nonterminating.POMDP").relative_to(PROJECT_ROOT)
            ),
            "pinned_source_model_sha256": independent_file_sha256(
                PROJECT_ROOT / "data/v63-external-unknown-dynamics/source-checkout/pobax/"
                "envs/classic/POMDP/4x3_nonterminating.POMDP"
            ),
            "bundle_file_digest": payload_hash({"files": digest_input}),
            "policies": policy_manifest,
            "records": record_manifest,
            "access": {
                "public_records": 48,
                "source_V66_policy_trees": 96,
                "truth_fields": 0,
                "V66_evaluation_reruns": 0,
                "Storm_source_policy_runs": 0,
                "human_records": 0,
                "model_forward_passes": 0,
                "adapter_training_runs": 0,
            },
        }
        write_json(temporary_root / "bundle-manifest.json", manifest)
        temporary_root.rename(final_bundle)
    print(json.dumps({
        "bundle": str(final_bundle),
        "manifest_sha256": file_sha256(final_bundle / "bundle-manifest.json"),
        "policies": len(policy_manifest),
    }, indent=2))


if __name__ == "__main__":
    main()
