#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from decimal import Decimal
from fractions import Fraction

from generate_v50_history import (
    corpus_hash,
    prior_structural_keys,
    support_mask_schedule,
)
from v10_protocol import file_sha256
from v22_relational import canonical_json
from v22r2_grounding import PROJECT_ROOT
from v46_stochastic import mechanic_registry as v46_registry
from v47_sampling import mechanic_registry as v47_registry
from v48_composition import mechanic_registry as v48_registry
from v49_belief import conditional_suffix_from_map, masked_trace, mechanic_registry as v49_registry, trajectory_map
from v50_belief import (
    decimal_map,
    kl_divergence,
    latest_only_evidence,
    mechanic_registry,
    time_shuffled_evidence,
    total_variation,
)


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fraction_map(rows):
    return {
        canonical_json(row["suffix"]): Fraction(row["mass"]["numerator"], row["mass"]["denominator"])
        for row in rows
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-lock", default="configs/v50-implementation-lock.json")
    parser.add_argument("--output", default="outputs/v50-history-dependent-belief-filtering/corpus-audit.json")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.implementation_lock).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    lock = json.loads(lock_path.read_text())
    errors = []
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            errors.append(f"implementation changed: {path}")
    data = PROJECT_ROOT / "data/v50-history-dependent-belief-filtering"
    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    rows = []
    for split in ("development_fit", "development_evaluation"):
        path = data / f"{split}.jsonl"
        if not path.is_file():
            errors.append(f"missing corpus split: {split}")
            continue
        rows.extend(read(path))
    if rows and corpus_hash(rows) != lock["expected_corpus_sha256"]:
        errors.append("V50 corpus hash differs from implementation lock")

    registry = mechanic_registry()
    registry_keys = {row["key"] for row in registry}
    previous_keys = {
        row["key"]
        for source in (v46_registry, v47_registry, v48_registry, v49_registry)
        for row in source()
    }
    target_keys = {row["target"]["program_key"] for row in rows}
    if len(rows) != 48 or target_keys != registry_keys or target_keys & previous_keys:
        errors.append("V50 corpus program population is invalid")

    previous_cases = prior_structural_keys()
    support_keys, query_keys = set(), set()
    trial_alignment, support_mask_reproduction = [], []
    history_dependent, time_dependent, all_outcome_schedule_valid = [], [], []
    oracle_reproduction, target_retention = [], []
    for record in rows:
        target = next(row for row in registry if row["key"] == record["target"]["program_key"])
        references = {row["id"]: row for row in record["reference"]["matched_fully_observed_support_interventions"]}
        for support in record["agent_input"]["support_interventions"]:
            support_keys.add(support["structural_key"])
            support_mask_reproduction.append(support["masks"] == support_mask_schedule(
                support["entities"], support["sequence_length"], support["id"],
                lock["config_payload"]["population"]["maskSeed"],
            ))
            reference = references[support["id"]]
            for masked_id, full_id in zip(
                support["realized_masked_trace_ids"], reference["realized_full_trajectory_ids"], strict=True
            ):
                expected = masked_trace(reference["full_trajectory_catalog"][full_id], support["masks"])
                trial_alignment.append(expected == support["masked_trace_catalog"][masked_id])
        oracle_by_id = {row["id"]: row for row in record["oracle_queries"]}
        for query in record["agent_input"]["queries"]:
            query_keys.add(query["structural_key"])
            oracle = oracle_by_id[query["id"]]
            prefix = query["query_prefix_length"]
            info_step = query["informative_step"]
            schedule_shape = (
                info_step < prefix - 1
                and bool(query["masks"][info_step])
                and not set(query["masks"][info_step]) & set(query["masks"][prefix - 1])
                and query["latest_only_prefix_observations"] == latest_only_evidence(query["masked_prefix_observations"])
                and query["time_shuffled_prefix_observations"]
                == time_shuffled_evidence(query["masked_prefix_observations"], info_step)
            )
            full = trajectory_map(
                target["program"], query["entities"],
                {row["atom"]: row["allowed_values"][0] for row in query["initial_state"]},
                query["actions"],
            )
            schedule_histories = {}
            schedule_ok = schedule_shape
            for trajectory_key in full:
                evidence = masked_trace(json.loads(trajectory_key), query["masks"])[:prefix]
                key = canonical_json(evidence)
                if key in schedule_histories:
                    continue
                _, truth_fraction = conditional_suffix_from_map(full, evidence, prefix)
                latest = latest_only_evidence(evidence)
                _, latest_fraction = conditional_suffix_from_map(full, latest, prefix)
                shuffled = time_shuffled_evidence(evidence, info_step)
                shuffled_mass, shuffled_fraction = conditional_suffix_from_map(full, shuffled, prefix)
                truth = decimal_map(truth_fraction)
                latest_truth = decimal_map(latest_fraction)
                shuffled_truth = decimal_map(shuffled_fraction) if shuffled_mass else {}
                history_tv = total_variation(truth, latest_truth)
                history_kl = kl_divergence(truth, latest_truth)
                shuffled_kl = kl_divergence(truth, shuffled_truth)
                schedule_ok = schedule_ok and history_tv >= 0.10 and history_kl >= 0.05 and shuffled_kl >= 0.05
                schedule_histories[key] = (truth_fraction, latest_fraction, shuffled_fraction, history_tv)
            all_outcome_schedule_valid.append(schedule_ok)
            actual = schedule_histories[canonical_json(query["masked_prefix_observations"])]
            expected_truth, expected_latest, expected_shuffled, expected_tv = actual
            reproduced = (
                fraction_map(oracle["true_complete_history_conditional_suffix_distribution"]) == expected_truth
                and fraction_map(oracle["true_latest_only_conditional_suffix_distribution"]) == expected_latest
                and fraction_map(oracle["true_time_shuffled_conditional_suffix_distribution"]) == expected_shuffled
                and abs(oracle["oracle_full_history_vs_latest_only_tv"] - expected_tv) < 1e-12
            )
            oracle_reproduction.append(reproduced)
            history_dependent.append(expected_tv >= 0.10)
            time_dependent.append(oracle["oracle_time_shuffled_kl_nats"] >= 0.05)
            truth_keys = set(expected_truth)
            target_retention.extend(
                canonical_json(oracle["compatible_full_trajectory_catalog"][identifier][prefix:]) in truth_keys
                for identifier in oracle["heldout_full_trajectory_ids"]
            )

    contract = lock["config_payload"]["historyDependenceContract"]
    history_rate = sum(history_dependent) / len(history_dependent) if history_dependent else 0
    if history_rate < contract["minimumOracleHistoryDependentQueryFraction"]:
        errors.append("V50 oracle history-dependence quota failed")
    if support_keys & query_keys or (support_keys | query_keys) & previous_cases:
        errors.append("V50 structural case firewall failed")
    if not all(trial_alignment) or not all(support_mask_reproduction):
        errors.append("V50 support mask or trial alignment failed")
    if not all(all_outcome_schedule_valid) or not all(oracle_reproduction) or not all(target_retention):
        errors.append("V50 query schedule, oracle, or target-retention audit failed")

    counts = manifest.get("counts", {})
    expected_counts = {
        "mechanics": 48,
        "support_interventions": 576,
        "support_trials": 18432,
        "queries": 1152,
        "heldout_conditional_continuations": 73728,
    }
    if any(counts.get(key) != value for key, value in expected_counts.items()):
        errors.append("V50 manifest counts are invalid")
    public_leak = any(
        any(token in canonical_json(record["agent_input"]) for token in (
            "program_key", "true_complete_history", "heldout_full", "full_trajectory_catalog"
        ))
        for record in rows
    )
    if public_leak:
        errors.append("V50 scorer-only fields leaked into agent input")
    audit = {
        "schema_version": 50,
        "experiment": "v50_corpus_audit",
        "passed": not errors,
        "decision": "authorize_v50_corpus_seal" if not errors else "repair_v50_corpus",
        "errors": errors,
        "implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(lock_path),
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "manifest_sha256": file_sha256(manifest_path) if manifest_path.is_file() else None,
        "corpus_sha256": corpus_hash(rows) if rows else None,
        "checks": {
            "fresh_programs": not bool(target_keys & previous_keys),
            "fresh_cases": not bool((support_keys | query_keys) & previous_cases),
            "support_query_disjoint": not bool(support_keys & query_keys),
            "support_mask_reproduction": all(support_mask_reproduction),
            "exact_trial_alignment": all(trial_alignment),
            "oracle_history_dependent_query_fraction": history_rate,
            "time_dependent_query_fraction": sum(time_dependent) / len(time_dependent) if time_dependent else 0,
            "all_possible_observations_qualify": all(all_outcome_schedule_valid),
            "oracle_reproduction": all(oracle_reproduction),
            "target_continuation_retention": sum(target_retention) / len(target_retention) if target_retention else 0,
            "public_scorer_leak": public_leak,
        },
        "counts": expected_counts,
        "data_access": {
            "development_runs": 0,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
