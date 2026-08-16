#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from decimal import Decimal

from generate_v49_partial import corpus_hash, mask_schedule, prior_structural_keys
from v10_protocol import file_sha256
from v22_relational import canonical_json
from v22r2_grounding import PROJECT_ROOT
from v46_stochastic import mechanic_registry as v46_registry
from v47_sampling import mechanic_registry as v47_registry
from v48_composition import mechanic_registry as v48_registry
from v49_belief import decimal_map, map_latent_predictive, masked_trace, mechanic_registry


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-lock", default="configs/v49-implementation-lock.json")
    parser.add_argument("--output", default="outputs/v49-passive-partial-observation/corpus-audit.json")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.implementation_lock).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    lock = json.loads(lock_path.read_text())
    errors = []
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            errors.append(f"implementation changed: {path}")
    data = PROJECT_ROOT / "data/v49-passive-partial-observation"
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
        errors.append("V49 corpus hash differs from implementation lock")

    registry = mechanic_registry()
    registry_keys = {row["key"] for row in registry}
    previous_keys = {row["key"] for source in (v46_registry, v47_registry, v48_registry) for row in source()}
    target_keys = {row["target"]["program_key"] for row in rows}
    if len(rows) != 48 or target_keys != registry_keys or target_keys & previous_keys:
        errors.append("V49 corpus program population is invalid")

    previous_cases = prior_structural_keys()
    support_keys = set()
    query_keys = set()
    trial_alignment = []
    mask_reproduction = []
    query_ambiguity = []
    hidden_influence = []
    target_retention = []
    for record in rows:
        target = next(row for row in registry if row["key"] == record["target"]["program_key"])
        references = {row["id"]: row for row in record["reference"]["matched_fully_observed_support_interventions"]}
        for support in record["agent_input"]["support_interventions"]:
            support_keys.add(support["structural_key"])
            mask_reproduction.append(support["masks"] == mask_schedule(
                support["entities"], support["sequence_length"], support["visible_fraction"], support["id"],
                lock["config_payload"]["population"]["maskSeed"],
            ))
            reference = references[support["id"]]
            for masked_id, full_id in zip(
                support["realized_masked_trace_ids"], reference["realized_full_trajectory_ids"], strict=True
            ):
                expected = masked_trace(reference["full_trajectory_catalog"][full_id], support["masks"])
                trial_alignment.append(expected == support["masked_trace_catalog"][masked_id])
        oracle = {row["id"]: row for row in record["oracle_queries"]}
        for query in record["agent_input"]["queries"]:
            query_keys.add(query["structural_key"])
            mask_reproduction.append(query["masks"] == mask_schedule(
                query["entities"], query["sequence_length"], query["visible_fraction"], query["id"],
                lock["config_payload"]["population"]["maskSeed"],
            ))
            oracle_row = oracle[query["id"]]
            truth = {
                canonical_json(row["suffix"]): Decimal(row["mass"]["numerator"]) / Decimal(row["mass"]["denominator"])
                for row in oracle_row["true_partial_conditional_suffix_distribution"]
            }
            query_ambiguity.append(len(truth) > 1)
            prediction = map_latent_predictive(
                [target], [Decimal(1)], query["entities"],
                {row["atom"]: row["allowed_values"][0] for row in query["initial_state"]},
                query["actions"], query["masked_prefix_observations"], query["query_prefix_length"],
            )
            distance = sum(
                abs(prediction.get(key, Decimal(0)) - truth.get(key, Decimal(0)))
                for key in set(prediction) | set(truth)
            ) / 2
            hidden_influence.append(distance > Decimal("1e-6"))
            target_retention.extend(
                canonical_json(oracle_row["compatible_full_trajectory_catalog"][identifier][query["query_prefix_length"]:]) in truth
                for identifier in oracle_row["heldout_full_trajectory_ids"]
            )

    requirements = lock["config_payload"]["constructionRequirements"]
    ambiguity_rate = sum(query_ambiguity) / len(query_ambiguity) if query_ambiguity else 0
    hidden_rate = sum(hidden_influence) / len(hidden_influence) if hidden_influence else 0
    if ambiguity_rate < requirements["minimumFractionQueriesWithNondegenerateOracleLatentBelief"]:
        errors.append("V49 query ambiguity quota failed")
    if hidden_rate < requirements["minimumFractionQueriesWhereHiddenStateCanAffectScoredSuffix"]:
        errors.append("V49 hidden-state influence quota failed")
    if support_keys & query_keys or (support_keys | query_keys) & previous_cases:
        errors.append("V49 structural case firewall failed")
    if not all(trial_alignment) or not all(mask_reproduction) or not all(target_retention):
        errors.append("V49 mask, alignment, or target-retention audit failed")

    counts = manifest.get("counts", {})
    expected_counts = {
        "mechanics": 48,
        "support_interventions": 576,
        "support_trials": 18432,
        "queries": 1152,
        "heldout_conditional_continuations": 73728,
    }
    if any(counts.get(key) != value for key, value in expected_counts.items()):
        errors.append("V49 manifest counts are invalid")
    public_leak = any(
        any(token in canonical_json(record["agent_input"]) for token in ("program_key", "true_partial", "heldout_full", "full_trajectory_catalog"))
        for record in rows
    )
    if public_leak:
        errors.append("V49 scorer-only fields leaked into agent input")

    audit = {
        "schema_version": 49,
        "experiment": "v49_corpus_audit",
        "passed": not errors,
        "decision": "authorize_v49_corpus_seal" if not errors else "repair_v49_corpus",
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
            "mask_reproduction": all(mask_reproduction),
            "exact_trial_alignment": all(trial_alignment),
            "query_ambiguity_rate": ambiguity_rate,
            "hidden_state_influence_rate": hidden_rate,
            "target_latent_continuation_retention": sum(target_retention) / len(target_retention) if target_retention else 0,
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
