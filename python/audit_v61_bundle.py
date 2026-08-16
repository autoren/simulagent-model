#!/usr/bin/env python3
"""Audit the constructed V61 policy/model bundle before sealing it."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def explicit_model_audit(directory, expected_states: int, expected_transitions: int):
    transition_lines = (directory / "model.tra").read_text().splitlines()
    reward_lines = (directory / "model.rew").read_text().splitlines()
    if not transition_lines or transition_lines[0] != "dtmc":
        return False
    rows = transition_lines[1:]
    grouped = defaultdict(float)
    finite = True
    sources = set()
    targets = set()
    for line in rows:
        source, target, probability = line.split()
        source, target, probability = int(source), int(target), float(probability)
        grouped[source] += probability
        sources.add(source); targets.add(target)
        finite &= math.isfinite(probability) and probability >= 0.0
    rewards_finite = all(
        math.isfinite(float(line.split()[2])) for line in reward_lines
    )
    state_ids = sources | targets
    return (
        len(rows) == len(reward_lines) == expected_transitions
        and len(state_ids) == expected_states
        and state_ids == set(range(expected_states))
        and all(abs(total - 1.0) <= 1e-12 for total in grouped.values())
        and set(grouped) == state_ids
        and finite and rewards_finite
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle", default="data/v61-long-horizon-policy-verification"
    )
    parser.add_argument(
        "--output", default="outputs/v61-long-horizon-policy-verification/bundle-audit.json"
    )
    args = parser.parse_args()
    bundle = (PROJECT_ROOT / args.bundle).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    implementation_path = PROJECT_ROOT / manifest["implementation_lock"]
    implementation = json.loads(implementation_path.read_text())
    errors = []
    source_ok = (
        implementation["authorization"]["construct_and_audit_v61_verification_bundle"]
        and not implementation["authorization"]["run_v61_candidate_verification"]
        and not implementation["authorization"]["access_v59_audit_truth"]
        and manifest["implementation_lock_sha256"] == file_sha256(implementation_path)
        and file_sha256(PROJECT_ROOT / implementation["implementation"])
        == implementation["implementation_sha256"]
        and file_sha256(PROJECT_ROOT / manifest["source_result"])
        == manifest["source_result_sha256"]
    )
    if not source_ok:
        errors.append("V61 implementation or source binding failed")
    census_ok = (
        manifest["policy_count"] == 72
        and manifest["horizon_counts"] == {"3": 24, "5": 24, "7": 24}
        and len(manifest["policies"]) == 72
        and len({row["id"] for row in manifest["policies"]}) == 72
        and len({(row["task_id"], row["replicate"]) for row in manifest["policies"]}) == 72
    )
    if not census_ok:
        errors.append("V61 verification bundle census is incomplete or duplicated")

    files_checked = hash_matches = model_checks = model_passes = 0
    tree_matches = action_matches = metadata_matches = belief_normalized = 0
    symbolic_passes = 0
    maximum_reference_error = maximum_mc_excess = maximum_mc_internal_error = 0.0
    truth_accesses = manifest["truth_field_access_count"]
    source_bindings = set()
    expected_names = {
        "model.tra", "model.lab", "model.rew", "model.meta.json", "policy-tree.json"
    }
    for row in manifest["policies"]:
        actual_names = {file_row["path"].split("/")[-1] for file_row in row["files"]}
        if actual_names != expected_names:
            errors.append(f"{row['id']} required-file set mismatch")
        for file_row in row["files"]:
            path = bundle / file_row["path"]
            files_checked += 1
            hash_matches += int(
                path.exists() and path.stat().st_size == file_row["bytes"]
                and file_sha256(path) == file_row["sha256"]
            )
        directory = bundle / row["directory"]
        meta = json.loads((directory / "model.meta.json").read_text())
        policy = json.loads((directory / "policy-tree.json").read_text())
        model_checks += 1
        model_passes += int(explicit_model_audit(
            directory, row["states"], row["transitions"]
        ))
        binding = meta["source_binding"]
        tree_matches += int(
            binding["tree_hash_match"]
            and policy["search"]["tree_sha256"] == row["tree_sha256"]
            == binding["frozen"]["tree_sha256"]
        )
        action_matches += int(
            binding["root_action_match"]
            and policy["search"]["selected_action_key"]
            == row["selected_action_key"]
            == binding["frozen"]["selected_action_key"]
        )
        metadata_matches += int(binding["search_metadata_match"])
        belief_normalized += int(abs(meta["exact_belief_weight_sum"] - 1.0) <= 1e-10)
        symbolic = meta["symbolic"]
        symbolic_passes += int(
            symbolic["invariant_checks"] == symbolic["invariant_passes"]
            and symbolic["support_checks"] == symbolic["support_passes"]
            and symbolic["support_checks"] == symbolic["probability_passes"]
            and symbolic["totality_checks"] == symbolic["totality_passes"]
            and symbolic["deployment_checks"] == symbolic["deployment_passes"]
            and symbolic["z3_unknown_count"] == 0
            and symbolic["nonterminal_deadlock_count"] == 0
        )
        maximum_reference_error = max(
            maximum_reference_error, meta["maximum_preseal_reference_error"],
            row["maximum_preseal_reference_error"],
        )
        maximum_mc_internal_error = max(
            maximum_mc_internal_error, meta["v60_monte_carlo_internal_error"]
        )
        maximum_mc_excess = max(
            maximum_mc_excess,
            max(0.0, meta["v60_monte_carlo_exact_error"]
                - meta["v60_monte_carlo_simultaneous_radius"]),
        )
        truth_accesses += meta["truth_field_access_count"]
        source_bindings.add((
            meta["source_population_seal"], meta["source_population_seal_sha256"],
            meta["source_outcome_lock"], meta["source_outcome_lock_sha256"],
            meta["source_result"], meta["source_result_sha256"],
        ))
        for path_key, sha_key in (
            ("source_population_seal", "source_population_seal_sha256"),
            ("source_outcome_lock", "source_outcome_lock_sha256"),
            ("source_result", "source_result_sha256"),
        ):
            if file_sha256(PROJECT_ROOT / meta[path_key]) != meta[sha_key]:
                errors.append(f"{row['id']} source binding changed")
    files_ok = files_checked == hash_matches == 72 * 5
    models_ok = model_checks == model_passes == 72
    binding_ok = (
        tree_matches == action_matches == metadata_matches == belief_normalized == 72
        and len(source_bindings) == 1
        and maximum_reference_error <= 1e-10
        and maximum_mc_internal_error <= 1e-15
    )
    symbolic_ok = symbolic_passes == 72
    integrity_ok = (
        truth_accesses == 0
        and manifest["public_task_records_accessed"] == 24
        and manifest["candidate_verification_runs"] == 0
    )
    if not files_ok:
        errors.append("V61 required file hash/size audit failed")
    if not models_ok:
        errors.append("V61 explicit DTMC structural audit failed")
    if not binding_ok:
        errors.append("V61 reconstructed policy or exact-belief source binding failed")
    if not symbolic_ok:
        errors.append("V61 preseal independent symbolic proof failed")
    if not integrity_ok:
        errors.append("V61 truth or verification firewall failed")
    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v61-verification-bundle-seal.json",
            "configs/v61-evaluation-implementation-lock.json",
            "configs/v61-outcome-lock.json",
            "outputs/v61-long-horizon-policy-verification/verification-attempt.json",
            "outputs/v61-long-horizon-policy-verification/verification",
        )
    )
    if not downstream_absent:
        errors.append("V61 downstream artifact exists before bundle seal")
    audit = {
        "schema_version": 61,
        "experiment": "v61_verification_bundle_audit",
        "passed": not errors,
        "decision": "authorize_v61_bundle_seal" if not errors else "repair_v61_bundle",
        "errors": errors,
        "bundle": str(bundle.relative_to(PROJECT_ROOT)),
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "manifest_sha256": file_sha256(manifest_path),
        "implementation_lock": str(implementation_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(implementation_path),
        "checks": {
            "implementation_authorization_and_source_binding": source_ok,
            "exhaustive_seventy_two_policy_census": census_ok,
            "required_file_hashes_and_sizes": files_ok,
            "explicit_dtmc_structure_normalization_and_finiteness": models_ok,
            "frozen_tree_action_metadata_and_exact_belief_binding": binding_ok,
            "independent_symbolic_and_probability_proofs": symbolic_ok,
            "truth_and_verification_firewalls": integrity_ok,
            "downstream_absent": downstream_absent,
        },
        "metrics": {
            "policy_count": manifest["policy_count"],
            "horizon_counts": manifest["horizon_counts"],
            "files_checked": files_checked,
            "file_hash_matches": hash_matches,
            "model_checks": model_checks, "model_passes": model_passes,
            "tree_hash_matches": tree_matches,
            "root_action_matches": action_matches,
            "search_metadata_matches": metadata_matches,
            "exact_belief_normalized": belief_normalized,
            "symbolic_policy_passes": symbolic_passes,
            "maximum_preseal_reference_error": maximum_reference_error,
            "maximum_v60_monte_carlo_internal_error": maximum_mc_internal_error,
            "maximum_v60_monte_carlo_excess_over_simultaneous_bound": maximum_mc_excess,
            "truth_field_access_count": truth_accesses,
            "candidate_verification_runs": manifest["candidate_verification_runs"],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
