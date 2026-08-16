#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from decimal import Decimal, localcontext

from evaluate_v52_particle import aggregate_exact, evaluate_exact_population
from generate_v52_particle import build_exact
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v42_stateful import deterministic_world, effect, entities, unary
from v46_stochastic import _rule, canonical_program, mechanic_registry as v46_registry, stochastic
from v47_sampling import mechanic_registry as v47_registry
from v48_composition import mechanic_registry as v48_registry
from v49_belief import mechanic_registry as v49_registry
from v50_belief import mechanic_registry as v50_registry
from v51_sbc import mechanic_registry as v51_registry
from v52_particle import mechanic_registry, particle_filter_episode, stream_id


REQUIRED = (
    "python/v52_particle.py",
    "python/generate_v52_particle.py",
    "python/evaluate_v52_particle.py",
    "python/test_v52_particle.py",
    "python/audit_v52_populations.py",
    "python/seal_v52_populations.py",
    "python/audit_and_summarize_v52.py",
    "python/freeze_v52_outcome.py",
    "scripts/run-v52-rao-blackwellized-particle-filtering.sh",
)


def lock_hash(payload):
    value = dict(payload)
    value.pop("lock_payload_sha256", None)
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-lock", default="configs/v52r1-design-lock.json")
    parser.add_argument(
        "--output",
        default="outputs/v52-rao-blackwellized-particle-filtering/implementation-audit.json",
    )
    args = parser.parse_args()
    design_path = (PROJECT_ROOT / args.design_lock).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    design = json.loads(design_path.read_text())
    config = design["config_payload"]
    errors = []

    design_integrity = (
        design.get("lock_payload_sha256") == lock_hash(design)
        and file_sha256(PROJECT_ROOT / design["source_design_lock"])
        == design["source_design_lock_sha256"]
        and file_sha256(PROJECT_ROOT / design["repair"]) == design["repair_sha256"]
    )
    if not design_integrity or not design["authorization"]["write_particle_implementation"]:
        errors.append("V52r1 design lock is invalid or does not authorize implementation")
    missing = [path for path in REQUIRED if not (PROJECT_ROOT / path).is_file()]
    if missing:
        errors.append(f"V52 implementation files missing: {missing}")

    registry = mechanic_registry()
    previous = {
        row["key"]
        for source in (
            v46_registry, v47_registry, v48_registry,
            v49_registry, v50_registry, v51_registry,
        )
        for row in source()
    }
    counts = Counter((row["family"], row["probability"]) for row in registry)
    registry_ok = (
        len(registry) == 48
        and len({row["key"] for row in registry}) == 48
        and not {row["key"] for row in registry} & previous
        and set(counts.values()) == {4}
    )
    if not registry_ok:
        errors.append("V52 registry is not fresh, unique, and balanced")

    entity_rows = entities(2)
    world = deterministic_world(entity_rows, "v52-implementation-audit")
    world["u:active:unit_1"] = False
    program = canonical_program({"rules": [
        _rule("pulse", stochastic_immediate=[
            stochastic("1/4", effect("set_true", unary("active", "target")))
        ]),
        _rule("route"),
    ]})
    action = {
        "id": "pulse", "binding": {"actor": "unit_0", "target": "unit_1"}
    }
    evidence = [[{"atom": "u:active:unit_1", "value": True}]]
    observed_log_likelihood, _, _ = particle_filter_episode(
        program, entity_rows, world, [action], evidence, 31, 9,
        ("audit", "exact-likelihood"),
    )
    with localcontext() as context:
        context.prec = 100
        exact_log_error = abs(observed_log_likelihood - Decimal("0.25").ln())
    exact_local_ok = exact_log_error <= Decimal("1e-90")
    if not exact_local_ok:
        errors.append("V52 exact local branch/evidence calculation failed")

    smoke_config = copy.deepcopy(config)
    for key, value in tuple(smoke_config["population"].items()):
        if key.endswith("Seed") and isinstance(value, int):
            smoke_config["population"][key] = value + 1_000_003
    smoke_config["exactBenchmark"]["recordsPerMechanic"] = 1
    smoke_records = build_exact(registry[:8], smoke_config, set(), set())
    (
        rows, controls, stream_collisions, fingerprint_collisions, fingerprint_pairs
    ) = evaluate_exact_population(smoke_records, registry, smoke_config)
    smoke = aggregate_exact(
        rows, controls, stream_collisions, fingerprint_collisions,
        fingerprint_pairs, smoke_config,
    )
    budgets = smoke_config["particleBudgets"]["budgets"]
    low, medium, primary = (str(value) for value in budgets)
    convergence_ok = (
        smoke["by_budget"][primary]["mean_core_tv"]
        <= smoke["by_budget"][medium]["mean_core_tv"]
        <= smoke["by_budget"][low]["mean_core_tv"]
    )
    stream_ok = (
        smoke["unintended_stream_collision_count"] == 0
        and smoke["stochastic_fingerprint_collision_rate"] == 0
        and fingerprint_pairs > 0
        and stream_id(1, "collision") == stream_id(1, "collision")
        and stream_id(1, "collision", 0) != stream_id(1, "collision", 1)
    )
    if not convergence_ok:
        errors.append("V52 altered-seed smoke does not improve monotonically with budget")
    if not stream_ok:
        errors.append("V52 altered-seed random-stream integrity failed")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v52-implementation-lock.json",
            "configs/v52-population-seal.json",
            "data/v52-rao-blackwellized-particle-filtering",
            "outputs/v52-rao-blackwellized-particle-filtering/evaluation-attempt.json",
            "outputs/v52-rao-blackwellized-particle-filtering/evaluation",
        )
    )
    if not downstream_absent:
        errors.append("V52 downstream artifact exists before implementation lock")

    audit = {
        "schema_version": 52,
        "experiment": "v52_implementation_audit",
        "passed": not errors,
        "decision": (
            "authorize_v52_implementation_lock" if not errors
            else "repair_v52_implementation"
        ),
        "errors": errors,
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "checks": {
            "design_integrity": design_integrity,
            "implementation_complete": not missing,
            "fresh_balanced_registry": registry_ok,
            "exact_local_branch_and_evidence": exact_local_ok,
            "altered_seed_budget_convergence": convergence_ok,
            "random_stream_integrity": stream_ok,
            "downstream_absent": downstream_absent,
        },
        "smoke": {
            "records": len(smoke_records),
            "rows": len(rows),
            "exact_log_likelihood_error": str(exact_log_error),
            "mean_core_tv_by_budget": {
                key: value["mean_core_tv"] for key, value in smoke["by_budget"].items()
            },
            "stream_collisions": smoke["unintended_stream_collision_count"],
            "fingerprint_pairs": fingerprint_pairs,
            "fingerprint_collisions": fingerprint_collisions,
        },
        "data_access": {
            "altered_seed_smoke_records": len(smoke_records),
            "sealed_population_records_accessed": 0,
            "particle_evaluation_runs": 0,
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
