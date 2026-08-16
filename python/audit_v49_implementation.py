#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from decimal import Decimal
from fractions import Fraction

from generate_v49_partial import intervention, mask_schedule
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v42_stateful import deterministic_world, entities
from v46_stochastic import mechanic_registry as v46_registry
from v47_sampling import mechanic_registry as v47_registry
from v48_composition import mechanic_registry as v48_registry
from v49_belief import (
    conditional_suffix_distribution,
    masked_trace,
    mechanic_registry,
    prefix_configurations,
    query_predictive,
    trajectory_map,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-lock", default="configs/v49-design-lock.json")
    parser.add_argument("--output", default="outputs/v49-passive-partial-observation/implementation-audit.json")
    args = parser.parse_args()
    design_path = (PROJECT_ROOT / args.design_lock).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    design = json.loads(design_path.read_text())
    errors = []
    if not design["authorization"]["write_belief_inference_implementation"]:
        errors.append("V49 design does not authorize implementation")
    if design["config_sha256"] != file_sha256(PROJECT_ROOT / design["config"]):
        errors.append("V49 design config changed after lock")
    if design["preregistration_sha256"] != file_sha256(PROJECT_ROOT / design["preregistration"]):
        errors.append("V49 preregistration changed after lock")

    required = (
        "python/v49_belief.py",
        "python/generate_v49_partial.py",
        "python/evaluate_v49_partial.py",
        "python/audit_v49_corpus.py",
        "python/seal_v49_corpus.py",
        "python/audit_and_summarize_v49.py",
        "python/freeze_v49_outcome.py",
        "scripts/run-v49-passive-partial-observation.sh",
    )
    missing = [path for path in required if not (PROJECT_ROOT / path).is_file()]
    if missing:
        errors.append(f"V49 implementation files missing: {missing}")

    registry = mechanic_registry()
    previous = {row["key"] for source in (v46_registry, v47_registry, v48_registry) for row in source()}
    keys = {row["key"] for row in registry}
    counts = Counter((row["family"], row["probability"]) for row in registry)
    registry_ok = len(registry) == 48 and len(keys) == 48 and not keys & previous and set(counts.values()) == {4}
    if not registry_ok:
        errors.append("V49 registry is not fresh, unique, and balanced")

    config = design["config_payload"]
    case = intervention(7, config, "implementation-audit")
    target = next(row for row in registry if row["timing"] == "delayed")
    full = trajectory_map(target["program"], case["entities"], case["initial_world"], case["actions"])
    masked = {}
    for key, mass in full.items():
        observation = json.dumps(masked_trace(json.loads(key), case["masks"]), sort_keys=True, separators=(",", ":"))
        masked[observation] = masked.get(observation, Fraction(0)) + mass
    likelihood_normalized = sum(masked.values(), Fraction(0)) == 1
    if not likelihood_normalized:
        errors.append("V49 masked likelihood does not normalize")

    trajectory = json.loads(next(iter(full)))
    prefix_length = 1
    evidence = masked_trace(trajectory, case["masks"])[:prefix_length]
    evidence_mass, truth = conditional_suffix_distribution(
        target["program"], case["entities"], case["initial_world"], case["actions"], evidence, prefix_length
    )
    prediction, weights, _ = query_predictive(
        [target], [Decimal(1)], case["entities"], case["initial_world"], case["actions"], evidence, prefix_length
    )
    oracle_exact = evidence_mass > 0 and all(
        prediction.get(key, Decimal(0)) == Decimal(value.numerator) / Decimal(value.denominator)
        for key, value in truth.items()
    ) and set(prediction) == set(truth)
    if not oracle_exact:
        errors.append("V49 oracle-program filter is not exact")

    queue_latent = False
    audit_entities = entities(2)
    for delayed_mechanic in [row for row in registry if row["timing"] == "delayed"]:
        audit_world = deterministic_world(audit_entities, delayed_mechanic["id"])
        action = {"id": "pulse", "binding": {"actor": "unit_0", "target": "unit_1"}}
        configurations_mass, configurations = prefix_configurations(
            delayed_mechanic["program"], audit_entities, audit_world, [action], [[]]
        )
        del configurations_mass
        queue_signatures = {json.dumps(row["queue"], sort_keys=True) for row in configurations.values()}
        if len(queue_signatures) > 1:
            queue_latent = True
            break
    if not queue_latent:
        errors.append("V49 delayed-event queue is not represented in latent configurations")

    masks_reproducible = case["masks"] == mask_schedule(
        case["entities"], case["sequence_length"], case["visible_fraction"], case["id"],
        config["population"]["maskSeed"],
    )
    if not masks_reproducible:
        errors.append("V49 masks are not deterministic from public structure and mask seed")

    downstream_absent = not any((PROJECT_ROOT / path).exists() for path in (
        "configs/v49-implementation-lock.json",
        "data/v49-passive-partial-observation",
        "outputs/v49-passive-partial-observation/development",
    ))
    if not downstream_absent:
        errors.append("V49 downstream artifact exists before implementation lock")

    audit = {
        "schema_version": 49,
        "experiment": "v49_implementation_audit",
        "passed": not errors,
        "decision": "authorize_v49_implementation_lock" if not errors else "repair_v49_implementation",
        "errors": errors,
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "checks": {
            "design_bound": design["config_sha256"] == file_sha256(PROJECT_ROOT / design["config"]),
            "implementation_complete": not missing,
            "fresh_balanced_registry": registry_ok,
            "masked_likelihood_normalized": likelihood_normalized,
            "oracle_program_filter_exact": oracle_exact,
            "belief_normalized": abs(sum(weights, Decimal(0)) - 1) < Decimal("1e-80"),
            "latent_queue_represented": queue_latent,
            "value_independent_masks_reproducible": masks_reproducible,
            "downstream_absent": downstream_absent,
        },
        "registry": {
            "mechanics": len(registry),
            "overlap_v46_v47_v48": len(keys & previous),
            "family_probability_cells": {f"{family}|{probability}": count for (family, probability), count in sorted(counts.items())},
        },
        "data_access": {
            "synthetic_audit_cases": 2,
            "development_mechanics_constructed": 0,
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
