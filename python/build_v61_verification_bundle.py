#!/usr/bin/env python3
"""Reconstruct, compile, and pre-verify all 72 frozen V60 primary policies."""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v53_smc2 import exact_inference, pool_smc2_repeats, smc2_inference
from v54_eig import belief_atoms_from_exact
from v55r1_planning import planning_registry
from v56_verification import (
    finite_model, model_statistics, transition_rows_normalize,
    write_explicit_model,
)
from v59_planning import assert_search_payload_is_public, tree_payload
from v60_decision_calibration import plan_domain_fast, smc2_atoms_for_planning
from evaluate_v60_decision_calibration import frozen_seed, search_summary
from v61_verification import (
    compile_search_policy_dtmc, hoeffding_radius,
    independent_policy_statistics, verify_compiled_model_symbolically,
)


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def binding(search, frozen: dict) -> dict:
    reconstructed = search_summary(search)
    metadata_fields = (
        "simulations_run", "tree_nodes", "branching_action_nodes",
        "visited_action_nodes", "root_action_rows",
    )
    return {
        "tree_hash_match": reconstructed["tree_sha256"] == frozen["tree_sha256"],
        "root_action_match": (
            reconstructed["selected_action_key"] == frozen["selected_action_key"]
        ),
        "search_metadata_match": all(
            reconstructed[field] == frozen[field] for field in metadata_fields
        ),
        "reconstructed": reconstructed,
        "frozen": frozen,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--implementation-lock", default="configs/v61-implementation-lock.json"
    )
    parser.add_argument(
        "--output", default="data/v61-long-horizon-policy-verification"
    )
    args = parser.parse_args()
    implementation_path = (PROJECT_ROOT / args.implementation_lock).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    if output.exists():
        raise RuntimeError("V61 verification bundle target already exists")
    implementation = json.loads(implementation_path.read_text())
    if (
        not implementation["authorization"]["construct_and_audit_v61_verification_bundle"]
        or not implementation["authorization"]["reconstruct_v60_source_policies"]
        or implementation["authorization"]["run_v61_candidate_verification"]
        or implementation["authorization"]["access_v59_audit_truth"]
        or file_sha256(PROJECT_ROOT / implementation["implementation"])
        != implementation["implementation_sha256"]
        or file_sha256(PROJECT_ROOT / implementation["implementation_audit"])
        != implementation["implementation_audit_sha256"]
    ):
        raise RuntimeError("V61 implementation lock is not intact or authorized")
    design = json.loads((PROJECT_ROOT / implementation["design_lock"]).read_text())
    config = design["config_payload"]
    source_outcome_path = PROJECT_ROOT / config["sourceV60OutcomeLock"]
    source_outcome = json.loads(source_outcome_path.read_text())
    source_result_path = PROJECT_ROOT / source_outcome["result"]
    if file_sha256(source_result_path) != source_outcome["result_sha256"]:
        raise RuntimeError("V61 source V60 result changed")
    source_result = json.loads(source_result_path.read_text())
    source_by_id = {row["id"]: row for row in source_result["records"]}

    seal_path = PROJECT_ROOT / config["sourcePolicies"]["populationSeal"]
    if file_sha256(seal_path) != design["population_seal_sha256"]:
        raise RuntimeError("V61 source population seal changed")
    seal = json.loads(seal_path.read_text())
    public_artifact = seal["artifacts"]["public_file"]
    public_path = PROJECT_ROOT / public_artifact["path"]
    if file_sha256(public_path) != public_artifact["sha256"]:
        raise RuntimeError("V61 public source population changed")
    public_rows = read_jsonl(public_path)
    if len(public_rows) != config["sourcePolicies"]["tasks"]:
        raise RuntimeError("V61 source task census changed")
    for row in public_rows:
        assert_search_payload_is_public(row)

    v53 = json.loads(
        (PROJECT_ROOT / "configs/v53r2-design-lock.json").read_text()
    )["config_payload"]
    v59 = json.loads(
        (PROJECT_ROOT / "configs/v59-design-lock.json").read_text()
    )["config_payload"]
    v60 = json.loads(
        (PROJECT_ROOT / "configs/v60-design-lock.json").read_text()
    )["config_payload"]
    v55r1 = json.loads(
        (PROJECT_ROOT / "configs/v55r1-design-lock.json").read_text()
    )["config_payload"]
    registry = planning_registry(v55r1)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".v61-bundle-build-", dir=output.parent
    ) as temp:
        temp_root = Path(temp) / output.name
        temp_root.mkdir()
        manifest_rows = []
        started = time.time()
        completed = 0
        for public_record in sorted(public_rows, key=lambda row: row["record"]):
            public = public_record["public"]
            query, goal = public["query"], public["goal"]
            entity_rows = query["entities"]
            horizon = public["planning_horizon"]
            tick = query["prefix_length"]
            inference_record = {
                "id": public_record["id"],
                "supports": public["supports"],
                "query": query,
            }
            exact = exact_inference(registry, inference_record, v53)
            exact_atoms = belief_atoms_from_exact(exact)
            repeated = [
                smc2_inference(
                    registry, inference_record, v53,
                    config["sourcePolicies"]["inferenceBudget"], repeat,
                    "v60-decision-calibration",
                )
                for repeat in range(config["sourcePolicies"]["inferenceRepeats"])
            ]
            approximate_atoms = smc2_atoms_for_planning(
                pool_smc2_repeats(repeated)
            )
            frozen_record = source_by_id[public_record["id"]]
            frozen_cells = {
                cell["replicate"]: cell for cell in frozen_record["cells"]
                if cell["inference_budget"]
                == config["sourcePolicies"]["inferenceBudget"]
            }
            if set(frozen_cells) != {0, 1, 2}:
                raise RuntimeError("V61 source V60 primary replicate census changed")
            for replicate in range(config["sourcePolicies"]["replicatesPerTask"]):
                seed = frozen_seed(v60, "search", public_record["id"], replicate)
                policy = plan_domain_fast(
                    approximate_atoms, registry, entity_rows, goal,
                    horizon, tick, config["sourcePolicies"]["searchBudget"],
                    seed, v59,
                )
                frozen_cell = frozen_cells[replicate]
                source_search = frozen_cell["approximate_belief_search"]
                source_binding = binding(policy, source_search)
                if not all(
                    source_binding[field] for field in (
                        "tree_hash_match", "root_action_match",
                        "search_metadata_match",
                    )
                ):
                    raise RuntimeError(
                        f"V61 source binding failed for {public_record['id']}/r{replicate}"
                    )
                direct = independent_policy_statistics(
                    exact_atoms, policy, registry, entity_rows, goal,
                    horizon, tick, v59,
                )
                model = compile_search_policy_dtmc(
                    exact_atoms, policy, registry, entity_rows, goal,
                    horizon, tick, v59,
                )
                graph = model_statistics(model)
                symbolic = verify_compiled_model_symbolically(model)
                maximum_preseal_error = max(
                    abs(graph["expected_return"] - direct["expected_return"]),
                    abs(graph["success_probability"] - direct["success_probability"]),
                    abs(graph["termination_probability"] - 1.0),
                    direct["maximum_transition_normalization_error"],
                    symbolic["maximum_probability_error"],
                )
                symbolic_passed = (
                    symbolic["invariant_checks"] == symbolic["invariant_passes"]
                    and symbolic["support_checks"] == symbolic["support_passes"]
                    and symbolic["support_checks"] == symbolic["probability_passes"]
                    and symbolic["totality_checks"] == symbolic["totality_passes"]
                    and symbolic["deployment_checks"] == symbolic["deployment_passes"]
                    and symbolic["z3_unknown_count"] == 0
                    and symbolic["nonterminal_deadlock_count"] == 0
                )
                if (
                    maximum_preseal_error > 1e-10
                    or not symbolic_passed
                    or not transition_rows_normalize(model)
                    or not finite_model(model)
                ):
                    raise RuntimeError(
                        f"V61 preseal verification failed for "
                        f"{public_record['id']}/r{replicate}: {maximum_preseal_error}"
                    )
                stored_mc = frozen_cell["exact_minus_approximate_policy"][
                    "control_mean_return"
                ]
                duplicate_mc = frozen_cell[
                    "approximate_minus_observation_blind_policy"
                ]["candidate_mean_return"]
                mc_internal_error = abs(stored_mc - duplicate_mc)
                radius = hoeffding_radius(config, horizon)
                exact_mc_error = abs(direct["expected_return"] - stored_mc)
                policy_id = f"{public_record['id']}__r{replicate}"
                relative = Path(f"h{horizon}") / policy_id
                directory = temp_root / relative
                directory.mkdir(parents=True)
                write_explicit_model(model, directory)
                policy_payload = {
                    "schema_version": 61,
                    "id": policy_id,
                    "task_id": public_record["id"],
                    "replicate": replicate,
                    "search_seed": seed,
                    "search": search_summary(policy),
                    "tree": tree_payload(policy.root),
                }
                (directory / "policy-tree.json").write_text(
                    json.dumps(policy_payload, indent=2, sort_keys=True) + "\n"
                )
                metadata = {
                    "schema_version": 61,
                    "experiment": "v61_policy_model_bundle",
                    "id": policy_id,
                    "task_id": public_record["id"],
                    "record": public_record["record"],
                    "history_class": public_record["history_class"],
                    "horizon": horizon,
                    "replicate": replicate,
                    "goal": goal,
                    "source_population_seal": str(seal_path.relative_to(PROJECT_ROOT)),
                    "source_population_seal_sha256": file_sha256(seal_path),
                    "source_public_artifact": public_artifact,
                    "source_outcome_lock": str(source_outcome_path.relative_to(PROJECT_ROOT)),
                    "source_outcome_lock_sha256": file_sha256(source_outcome_path),
                    "source_result": str(source_result_path.relative_to(PROJECT_ROOT)),
                    "source_result_sha256": file_sha256(source_result_path),
                    "source_binding": source_binding,
                    "exact_belief_atoms": len(exact_atoms),
                    "exact_belief_weight_sum": sum(float(atom["weight"]) for atom in exact_atoms),
                    "independent_executor": direct,
                    "compiled_graph": graph,
                    "symbolic": symbolic,
                    "states": len(model["states"]),
                    "transitions": len(model["transitions"]),
                    "maximum_preseal_reference_error": maximum_preseal_error,
                    "v60_stored_monte_carlo_return": stored_mc,
                    "v60_duplicate_monte_carlo_return": duplicate_mc,
                    "v60_monte_carlo_internal_error": mc_internal_error,
                    "v60_monte_carlo_exact_error": exact_mc_error,
                    "v60_monte_carlo_simultaneous_radius": radius,
                    "v60_monte_carlo_within_simultaneous_bound": exact_mc_error <= radius,
                    "truth_field_access_count": 0,
                    "candidate_verification_runs": 0,
                }
                (directory / "model.meta.json").write_text(
                    json.dumps(metadata, indent=2, sort_keys=True) + "\n"
                )
                files = []
                for name in config["verificationBundle"]["requiredFilesPerPolicy"]:
                    path = directory / name
                    files.append({
                        "path": str((relative / name).as_posix()),
                        "sha256": file_sha256(path),
                        "bytes": path.stat().st_size,
                    })
                manifest_rows.append({
                    "id": policy_id, "task_id": public_record["id"],
                    "record": public_record["record"], "replicate": replicate,
                    "horizon": horizon, "directory": str(relative.as_posix()),
                    "states": len(model["states"]),
                    "transitions": len(model["transitions"]),
                    "exact_belief_atoms": len(exact_atoms),
                    "tree_sha256": policy.tree_sha256,
                    "selected_action_key": policy.selected_action_key,
                    "maximum_preseal_reference_error": maximum_preseal_error,
                    "v60_monte_carlo_exact_error": exact_mc_error,
                    "v60_monte_carlo_simultaneous_radius": radius,
                    "files": files,
                })
                completed += 1
                print(json.dumps({
                    "completed": completed, "total": 72, "id": policy_id,
                    "horizon": horizon, "states": len(model["states"]),
                    "transitions": len(model["transitions"]),
                    "exact_return": direct["expected_return"],
                    "mc_error": exact_mc_error, "mc_radius": radius,
                    "seconds": time.time() - started,
                }, sort_keys=True), flush=True)
        manifest_rows.sort(key=lambda row: (row["record"], row["replicate"]))
        manifest = {
            "schema_version": 61,
            "experiment": "v61_verification_bundle_manifest",
            "implementation_lock": str(implementation_path.relative_to(PROJECT_ROOT)),
            "implementation_lock_sha256": file_sha256(implementation_path),
            "source_result": str(source_result_path.relative_to(PROJECT_ROOT)),
            "source_result_sha256": file_sha256(source_result_path),
            "policy_count": len(manifest_rows),
            "horizon_counts": {
                str(horizon): sum(row["horizon"] == horizon for row in manifest_rows)
                for horizon in (3, 5, 7)
            },
            "policies": manifest_rows,
            "public_task_records_accessed": len(public_rows),
            "truth_field_access_count": 0,
            "candidate_verification_runs": 0,
            "runtime_seconds": time.time() - started,
        }
        (temp_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        temp_root.replace(output)
    print(json.dumps({
        "bundle": str(output.relative_to(PROJECT_ROOT)),
        "manifest_sha256": file_sha256(output / "manifest.json"),
        "policies": 72,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
