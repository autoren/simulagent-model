#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter

from generate_v53_smc2 import (
    build_populations,
    observation_design_key,
    population_hash,
    prior_observation_design_keys,
)
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v46_stochastic import _configuration_key
from v53_smc2 import (
    continuous_sequential_filter,
    instantiate_program,
    mechanic_registry,
)


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def cases(record):
    yield from record["supports"]
    yield record["query"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--implementation-lock", default="configs/v53r2-implementation-lock.json"
    )
    parser.add_argument(
        "--output", default="outputs/v53r2-continuous-parameter-smc2/population-audit.json"
    )
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.implementation_lock).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    lock = json.loads(lock_path.read_text())
    config = lock["config_payload"]
    errors = []
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            errors.append(f"implementation changed: {path}")

    root = PROJECT_ROOT / "data/v53r2-continuous-parameter-smc2"
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    populations = {
        name: read(root / f"{name}.jsonl") if (root / f"{name}.jsonl").is_file() else []
        for name in ("exact", "sbc", "scale")
    }
    counts = {name: len(rows) for name, rows in populations.items()}
    counts_ok = counts == lock["expected_population_counts"]
    precommit_ok = population_hash(populations) == lock["expected_population_sha256"]
    reproduced = build_populations(config)
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
    if not all((counts_ok, precommit_ok, reproduction_ok, manifest_ok)):
        errors.append("population count, precommit, reproduction, or manifest binding failed")

    registry = mechanic_registry(config["population"]["templateSeed"])
    keys = {row["key"] for row in registry}
    exact_targets = Counter(row["target_program_key"] for row in populations["exact"])
    scale_targets = Counter(row["target_program_key"] for row in populations["scale"])
    sbc_targets = Counter(row["target_program_key"] for row in populations["sbc"])
    target_ok = (
        set(exact_targets) == set(scale_targets) == set(sbc_targets) == keys
        and set(exact_targets.values()) == {config["exactBenchmark"]["recordsPerTemplate"]}
        and set(scale_targets.values()) == {config["scaleStress"]["recordsPerTemplate"]}
    )
    if not target_ok:
        errors.append("target-template coverage or exact/scale balance failed")

    prior = prior_observation_design_keys()
    designs, schema_ok, identity_ok = [], True, True
    for rows in populations.values():
        for record in rows:
            for episode in cases(record):
                designs.append(episode["observation_design_key"])
                identity_ok &= episode["observation_design_key"] == observation_design_key(episode)
                expected = (
                    episode["prefix_length"]
                    if episode is record["query"] else episode["sequence_length"]
                )
                schema_ok &= (
                    len(episode["actions"]) == episode["sequence_length"]
                    and len(episode["masks"]) == episode["sequence_length"]
                    and len(episode["observations"]) == expected
                    and all(
                        set(value) == {"atom", "value"}
                        for step in episode["observations"] for value in step
                    )
                )
    uniqueness_ok = len(designs) == len(set(designs))
    freshness_ok = not bool(set(designs) & prior)
    if not all((schema_ok, identity_ok, uniqueness_ok, freshness_ok)):
        errors.append("observation schema, identity, uniqueness, or prior freshness failed")

    positive, retained = [], []
    for name in ("exact", "sbc"):
        for record in populations[name]:
            target = registry[record["target_program_index"]]
            program = instantiate_program(target["template"], record["target_theta"])
            for episode in record["supports"]:
                world = {
                    row["atom"]: row["allowed_values"][0]
                    for row in episode["initial_state"]
                }
                likelihood, _ = continuous_sequential_filter(
                    program, episode["entities"], world,
                    episode["actions"], episode["observations"],
                )
                positive.append(bool(likelihood))
            query = record["query"]
            world = {
                row["atom"]: row["allowed_values"][0]
                for row in query["initial_state"]
            }
            likelihood, configurations = continuous_sequential_filter(
                program, query["entities"], world,
                query["actions"][:query["prefix_length"]], query["observations"],
            )
            possible = {
                _configuration_key(value["world"], value["queue"])
                for value in configurations.values()
            }
            retained.append(
                bool(likelihood) and query["true_configuration_key"] in possible
            )
    truth_ok = all(positive) and all(retained)
    if not truth_ok:
        errors.append("target likelihood or true hidden configuration was lost")

    probes = [row for row in populations["exact"] if row["ambiguity_probe"]]
    probe_ok = (
        len(probes) == len(registry)
        and Counter(row["target_program_key"] for row in probes) == Counter(keys)
        and all(
            action["id"] == "wait"
            for record in probes for episode in cases(record)
            for action in episode["actions"]
        )
    )
    pmcmc_ok = (
        sum(record["pmcmc_reference"] for record in populations["exact"])
        == config["pmcmcReference"]["records"]
    )
    if not probe_ok or not pmcmc_ok:
        errors.append("fixed ambiguity-probe or PMCMC-reference quota failed")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v53r2-population-seal.json",
            "outputs/v53r2-continuous-parameter-smc2/evaluation-attempt.json",
            "outputs/v53r2-continuous-parameter-smc2/evaluation",
        )
    )
    if not downstream_absent:
        errors.append("evaluation exists before population seal")
    audit = {
        "schema_version": 53,
        "revision": "r2",
        "experiment": "v53r2_population_audit",
        "passed": not errors,
        "decision": "authorize_v53r2_population_seal" if not errors else "repair_v53r2_populations",
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
            "target_template_coverage": target_ok,
            "observation_schema_identity_uniqueness_freshness": all((schema_ok, identity_ok, uniqueness_ok, freshness_ok)),
            "target_likelihood_and_truth_retention": truth_ok,
            "fixed_ambiguity_probes": probe_ok,
            "fixed_pmcmc_quota": pmcmc_ok,
            "downstream_absent": downstream_absent,
        },
        "counts": counts,
        "observation_designs": len(designs),
        "data_access": {
            "smc_squared_evaluation_runs": 0,
            "pmcmc_reference_runs": 0,
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
