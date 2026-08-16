#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter

from generate_v52_particle import (
    build_populations,
    observation_design_key,
    population_hash,
    prior_observation_design_keys,
)
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v46_stochastic import _configuration_key
from v51_sbc import sequential_filter
from v52_particle import mechanic_registry


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def cases(record):
    if record["population"] in {"exact", "sbc"}:
        yield from record["supports"]
        yield record["query"]
    else:
        yield from record["episodes"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--implementation-lock", default="configs/v52-implementation-lock.json"
    )
    parser.add_argument(
        "--output",
        default="outputs/v52-rao-blackwellized-particle-filtering/population-audit.json",
    )
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.implementation_lock).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    lock = json.loads(lock_path.read_text())
    errors = []
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            errors.append(f"implementation changed: {path}")

    root = PROJECT_ROOT / "data/v52-rao-blackwellized-particle-filtering"
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    populations = {
        name: read(root / f"{name}.jsonl")
        if (root / f"{name}.jsonl").is_file() else []
        for name in ("exact", "sbc", "scale")
    }
    counts_ok = {
        name: len(rows) for name, rows in populations.items()
    } == lock["expected_population_counts"]
    precommit_ok = population_hash(populations) == lock["expected_population_sha256"]
    reproduced = build_populations(lock["config_payload"])
    reproduction_ok = population_hash(reproduced) == population_hash(populations)
    manifest_ok = (
        manifest.get("implementation_lock_sha256") == file_sha256(lock_path)
        and manifest.get("population_sha256") == population_hash(populations)
        and all(
            manifest.get("artifacts", {}).get(name, {}).get("sha256")
            == file_sha256(root / f"{name}.jsonl")
            for name in populations
        )
    )
    if not counts_ok or not precommit_ok or not reproduction_ok or not manifest_ok:
        errors.append("V52 populations fail count, precommit, reproduction, or manifest binding")

    registry = mechanic_registry()
    keys = {row["key"] for row in registry}
    exact_targets = Counter(row["target_program_key"] for row in populations["exact"])
    scale_targets = Counter(row["target_program_key"] for row in populations["scale"])
    sbc_targets = Counter(row["target_program_key"] for row in populations["sbc"])
    target_balance_ok = (
        set(exact_targets) == keys
        and set(scale_targets) == keys
        and set(sbc_targets) == keys
        and set(exact_targets.values()) == {2}
        and set(scale_targets.values()) == {2}
    )
    if not target_balance_ok:
        errors.append("V52 target-program coverage or balance failed")

    prior = prior_observation_design_keys()
    split_keys = {}
    schema_ok = True
    identity_ok = True
    for name, rows in populations.items():
        values = []
        for record in rows:
            for case in cases(record):
                values.append(case["observation_design_key"])
                identity_ok &= case["observation_design_key"] == observation_design_key(case)
                expected_steps = (
                    case.get("prefix_length", case["sequence_length"])
                    if record["population"] in {"exact", "sbc"}
                    and case is record.get("query")
                    else case["sequence_length"]
                )
                schema_ok &= (
                    len(case["actions"]) == case["sequence_length"]
                    and len(case["masks"]) == case["sequence_length"]
                    and len(case["observations"]) == expected_steps
                    and all(
                        set(row) == {"atom", "value"}
                        for step in case["observations"] for row in step
                    )
                )
        split_keys[name] = values
    all_keys = [value for values in split_keys.values() for value in values]
    uniqueness_ok = len(all_keys) == len(set(all_keys))
    freshness_ok = not bool(set(all_keys) & prior)
    if not all((schema_ok, identity_ok, uniqueness_ok, freshness_ok)):
        errors.append("V52 observation-design schema, identity, uniqueness, or freshness failed")

    positive = []
    retained = []
    for name in ("exact", "sbc"):
        for record in populations[name]:
            target = registry[record["target_program_index"]]
            for support in record["supports"]:
                world = {
                    row["atom"]: row["allowed_values"][0]
                    for row in support["initial_state"]
                }
                likelihood, _ = sequential_filter(
                    target["program"], support["entities"], world,
                    support["actions"], support["observations"],
                )
                positive.append(bool(likelihood))
            query = record["query"]
            world = {
                row["atom"]: row["allowed_values"][0]
                for row in query["initial_state"]
            }
            likelihood, configurations = sequential_filter(
                target["program"], query["entities"], world,
                query["actions"][: query["prefix_length"]], query["observations"],
            )
            possible = {
                _configuration_key(row["world"], row["queue"])
                for row in configurations.values()
            }
            retained.append(
                bool(likelihood) and query["true_configuration_key"] in possible
            )
    truth_ok = all(positive) and all(retained)
    if not truth_ok:
        errors.append("V52 target likelihood or simulated true configuration was lost")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v52-population-seal.json",
            "outputs/v52-rao-blackwellized-particle-filtering/evaluation-attempt.json",
            "outputs/v52-rao-blackwellized-particle-filtering/evaluation",
        )
    )
    if not downstream_absent:
        errors.append("V52 evaluation exists before population seal")

    audit = {
        "schema_version": 52,
        "experiment": "v52_population_audit",
        "passed": not errors,
        "decision": (
            "authorize_v52_population_seal" if not errors else "repair_v52_populations"
        ),
        "errors": errors,
        "implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(lock_path),
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "manifest_sha256": file_sha256(manifest_path) if manifest_path.is_file() else None,
        "population_sha256": population_hash(populations),
        "checks": {
            "expected_counts": counts_ok,
            "implementation_precommit": precommit_ok,
            "deterministic_reproduction": reproduction_ok,
            "manifest_binding": manifest_ok,
            "target_program_coverage_and_balance": target_balance_ok,
            "observation_schema": schema_ok,
            "observation_identity": identity_ok,
            "globally_unique_designs": uniqueness_ok,
            "fresh_against_v46_through_v51": freshness_ok,
            "target_likelihood_and_truth_retention": truth_ok,
            "downstream_absent": downstream_absent,
        },
        "counts": {name: len(rows) for name, rows in populations.items()},
        "observation_designs": len(all_keys),
        "data_access": {
            "particle_evaluation_runs": 0,
            "sealed_evaluation_outcomes_accessed": 0,
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
