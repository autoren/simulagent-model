#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter

from generate_v51_sbc import corpus_hash, make_case, public_case
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v46_stochastic import _configuration_key
from v51_sbc import mechanic_registry, sequential_filter


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def previous_structural_keys():
    result = set()
    root = PROJECT_ROOT / "data/v50-history-dependent-belief-filtering"
    for split in ("development_fit", "development_evaluation"):
        path = root / f"{split}.jsonl"
        if not path.is_file():
            continue
        for record in read(path):
            result.update(
                row["structural_key"]
                for row in record["agent_input"]["support_interventions"]
            )
            result.update(
                row["structural_key"] for row in record["agent_input"]["queries"]
            )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--implementation-lock", default="configs/v51-implementation-lock.json"
    )
    parser.add_argument(
        "--output", default="outputs/v51-simulation-based-calibration/corpus-audit.json"
    )
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.implementation_lock).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    lock = json.loads(lock_path.read_text())
    errors = []
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            errors.append(f"implementation changed: {path}")

    data = PROJECT_ROOT / "data/v51-simulation-based-calibration"
    corpus_path = data / "replications.jsonl"
    manifest_path = data / "manifest.json"
    rows = read(corpus_path) if corpus_path.is_file() else []
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    if not rows:
        errors.append("V51 replication corpus is missing")
    if rows and corpus_hash(rows) != lock["expected_corpus_sha256"]:
        errors.append("V51 corpus differs from implementation precommit")
    if manifest.get("artifact", {}).get("sha256") != (
        file_sha256(corpus_path) if corpus_path.is_file() else None
    ):
        errors.append("V51 manifest is not bound to corpus file")

    config = lock["config_payload"]
    registry = mechanic_registry()
    registry_keys = {row["key"] for row in registry}
    target_counts = Counter(row["target_program_key"] for row in rows)
    expected_replications = config["simulation"]["replications"]
    if len(rows) != expected_replications or set(target_counts) != registry_keys:
        errors.append("V51 prior-predictive target population is incomplete")

    previous = previous_structural_keys()
    support_keys, query_keys = set(), set()
    design_reproduced, target_support_possible, true_configuration_retained = [], [], []
    observation_shapes = []
    lengths = config["simulation"]["sequenceLengths"]
    prefixes = config["simulation"]["queryPrefixLengths"]
    supports_per_replication = config["simulation"]["supportInterventionsPerReplication"]
    for record in rows:
        replication = record["replication"]
        target = registry[record["target_program_index"]]
        if target["key"] != record["target_program_key"]:
            errors.append(f"target index/key mismatch in {record['id']}")
            continue
        for ordinal, support in enumerate(record["supports"]):
            support_keys.add(support["structural_key"])
            length = lengths[(replication + ordinal) % len(lengths)]
            expected = public_case(make_case(
                replication, ordinal, length, config, "support"
            ))
            design_reproduced.append(
                all(support[key] == value for key, value in expected.items())
            )
            world = {
                row["atom"]: row["allowed_values"][0]
                for row in support["initial_state"]
            }
            likelihood, _ = sequential_filter(
                target["program"], support["entities"], world,
                support["actions"], support["observations"],
            )
            target_support_possible.append(bool(likelihood))
            observation_shapes.append(
                len(support["observations"]) == support["sequence_length"]
                and all(
                    set(row) == {"atom", "value"}
                    for step in support["observations"] for row in step
                )
            )
        query = record["query"]
        query_keys.add(query["structural_key"])
        pair = replication % len(lengths)
        expected_query = public_case(make_case(
            replication, 0, lengths[pair], config, "query"
        ))
        design_reproduced.append(
            all(query[key] == value for key, value in expected_query.items())
            and query["prefix_length"] == prefixes[pair]
        )
        world = {
            row["atom"]: row["allowed_values"][0]
            for row in query["initial_state"]
        }
        likelihood, configurations = sequential_filter(
            target["program"], query["entities"], world,
            query["actions"][: query["prefix_length"]], query["observations"],
        )
        retained = {
            _configuration_key(row["world"], row["queue"])
            for row in configurations.values()
        }
        true_configuration_retained.append(
            bool(likelihood) and query["true_configuration_key"] in retained
        )
        observation_shapes.append(
            len(query["observations"]) == query["prefix_length"]
            and all(
                set(row) == {"atom", "value"}
                for step in query["observations"] for row in step
            )
        )

    fresh_cases = not bool((support_keys | query_keys) & previous)
    support_query_disjoint = not bool(support_keys & query_keys)
    unique_cases = (
        len(support_keys) == len(rows) * supports_per_replication
        and len(query_keys) == len(rows)
    )
    if not fresh_cases or not support_query_disjoint or not unique_cases:
        errors.append("V51 structural case firewall failed")
    if not all(design_reproduced):
        errors.append("V51 value-independent designs do not reproduce")
    if not all(target_support_possible) or not all(true_configuration_retained):
        errors.append("V51 simulated truth is not retained by exact target likelihood")
    if not all(observation_shapes):
        errors.append("V51 observation schema is invalid")

    expected_counts = {
        "replications": expected_replications,
        "support_observations": expected_replications * supports_per_replication,
        "query_observations": expected_replications,
    }
    if manifest.get("counts") != expected_counts:
        errors.append("V51 manifest counts are invalid")
    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v51-corpus-seal.json",
            "outputs/v51-simulation-based-calibration/calibration-attempt.json",
            "outputs/v51-simulation-based-calibration/calibration",
        )
    )
    if not downstream_absent:
        errors.append("V51 calibration output exists before corpus seal")

    audit = {
        "schema_version": 51,
        "experiment": "v51_corpus_audit",
        "passed": not errors,
        "decision": "authorize_v51_corpus_seal" if not errors else "repair_v51_corpus",
        "errors": errors,
        "implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(lock_path),
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "manifest_sha256": file_sha256(manifest_path) if manifest_path.is_file() else None,
        "corpus_sha256": corpus_hash(rows) if rows else None,
        "checks": {
            "expected_replication_count": len(rows) == expected_replications,
            "all_prior_programs_represented": set(target_counts) == registry_keys,
            "value_independent_design_reproduction": all(design_reproduced),
            "fresh_cases": fresh_cases,
            "support_query_disjoint": support_query_disjoint,
            "unique_replication_cases": unique_cases,
            "target_support_likelihood_positive": all(target_support_possible),
            "true_query_configuration_retained": all(true_configuration_retained),
            "observation_schema": all(observation_shapes),
            "downstream_absent": downstream_absent,
        },
        "target_prior_counts": dict(sorted(target_counts.items())),
        "counts": expected_counts,
        "data_access": {
            "calibration_runs": 0,
            "calibration_outcomes_accessed": 0,
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
