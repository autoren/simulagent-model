#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import Counter
from fractions import Fraction

from evaluate_v53_smc2 import evaluate_exact
from generate_v53_smc2 import build_exact
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v42_stateful import deterministic_world, effect, entities, unary
from v46_stochastic import _rule, canonical_program, stochastic
from v53_smc2 import (
    continuous_particle_filter_episode,
    exact_inference,
    instantiate_program,
    mechanic_registry,
    parameterize_program,
    pmmh_conditional_chains,
)


REQUIRED = (
    "python/v53_smc2.py",
    "python/generate_v53_smc2.py",
    "python/evaluate_v53_smc2.py",
    "python/test_v53_smc2.py",
    "python/audit_v53_populations.py",
    "python/seal_v53_populations.py",
    "python/audit_and_summarize_v53.py",
    "python/freeze_v53_outcome.py",
    "scripts/run-v53r2-continuous-parameter-smc2.sh",
)


def lock_hash(payload):
    value = dict(payload)
    value.pop("lock_payload_sha256", None)
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-lock", default="configs/v53r2-design-lock.json")
    parser.add_argument(
        "--output", default="outputs/v53r2-continuous-parameter-smc2/implementation-audit.json"
    )
    args = parser.parse_args()
    design_path = (PROJECT_ROOT / args.design_lock).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    design = json.loads(design_path.read_text())
    config = design["config_payload"]
    errors = []
    design_ok = (
        design["lock_payload_sha256"] == lock_hash(design)
        and file_sha256(PROJECT_ROOT / design["source_design_lock"])
        == design["source_design_lock_sha256"]
        and file_sha256(PROJECT_ROOT / design["repair"]) == design["repair_sha256"]
        and design["authorization"]["write_and_audit_smc_squared_implementation"]
    )
    if not design_ok:
        errors.append("V53r2 design lock is invalid or does not authorize implementation")
    missing = [path for path in REQUIRED if not (PROJECT_ROOT / path).is_file()]
    if missing:
        errors.append(f"V53r2 implementation files missing: {missing}")

    registry = mechanic_registry(config["population"]["templateSeed"])
    family_counts = Counter(row["family"] for row in registry)
    registry_ok = (
        len(registry) == len({row["key"] for row in registry}) == 8
        and set(family_counts.values()) == {2}
        and set(family_counts) == set(config["population"]["families"])
        and all("$theta" in canonical_program_text(row["template"]) for row in registry)
    )
    if not registry_ok:
        errors.append("V53r2 parameterized registry is not unique and balanced")

    entity_rows = entities(2)
    world = deterministic_world(entity_rows, "v53-implementation-audit")
    world["u:active:unit_1"] = False
    finite = canonical_program({"rules": [
        _rule("pulse", stochastic_immediate=[
            stochastic("1/2", effect("set_true", unary("active", "target")))
        ]),
        _rule("route"),
    ]})
    program = instantiate_program(parameterize_program(finite), 0.321)
    action = {"id": "pulse", "binding": {"actor": "unit_0", "target": "unit_1"}}
    evidence = [[{"atom": "u:active:unit_1", "value": True}]]
    observed, _, _ = continuous_particle_filter_episode(
        program, entity_rows, world, [action], evidence, 31, 17,
        ("audit", "continuous-local"),
    )
    exact_local_error = abs(float(observed) - math.log(float(Fraction("0.32100000000000001"))))
    exact_local_ok = exact_local_error <= 1e-14
    if not exact_local_ok:
        errors.append("continuous local branch likelihood is not exact")

    fixture = copy.deepcopy(config)
    for key, value in tuple(fixture["population"].items()):
        if key.endswith("Seed") and isinstance(value, int):
            fixture["population"][key] = value + 1_000_000
    fixture["exactBenchmark"].update({
        "records": 2,
        "recordsPerTemplate": 2,
        "supportEpisodesPerRecord": 2,
        "supportSequenceLengths": [3, 4],
        "querySequenceLengths": [5, 6],
        "queryPrefixLengths": [3, 4],
        "quadratureNodes": 33,
    })
    fixture["smcSquared"].update({
        "outerThetaParticleBudgets": [7, 15, 31],
        "primaryOuterThetaParticleBudget": 31,
        "independentRepeatsOnExactBenchmark": 2,
        "innerStateParticleBudget": 7,
        "rejuvenationStepsPerOuterResampling": 1,
        "outerEssThresholdFraction": 1.01,
    })
    fixture["pmcmcReference"].update({
        "records": 1,
        "chains": 2,
        "warmupIterationsPerChain": 50,
        "retainedIterationsPerChain": 100,
        "innerStateParticleBudget": 15,
    })
    altered_registry = mechanic_registry(fixture["population"]["templateSeed"])
    smoke_records = build_exact(altered_registry[:1], fixture, set(), set())
    rows, pmcmc_rows, controls, streams, fingerprints = evaluate_exact(
        smoke_records, altered_registry, fixture
    )
    normalization_ok = all(row["normalization"] for row in rows)
    probe_exact = exact_inference(altered_registry, smoke_records[0], fixture)
    ambiguity_ok = (
        smoke_records[0]["ambiguity_probe"]
        and all(abs(value - 1 / 8) <= 1e-13 for value in probe_exact["program"])
        and max(probe_exact["program"]) <= 0.60
    )
    stream_ok = all(
        len(values) == len(set(values)) for values in streams.values()
    ) and bool(streams["outer"]) and bool(streams["inner"])
    pmcmc_acceptance = sum(row["acceptance_rate"] for row in pmcmc_rows) / len(pmcmc_rows)
    pmcmc_ok = 0.10 <= pmcmc_acceptance <= 0.75
    controls_ok = len(controls) == 2
    if not normalization_ok:
        errors.append("altered-seed exact/SMC fixture does not normalize")
    if not ambiguity_ok:
        errors.append("fixed altered-seed ambiguity probe is not analytic")
    if not stream_ok:
        errors.append("altered-seed random streams collide")
    if not pmcmc_ok:
        errors.append("altered-seed PMCMC proposal is outside its intended range")
    if not controls_ok:
        errors.append("fixed paired control fixture is incomplete")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v53r2-implementation-lock.json",
            "configs/v53r2-population-seal.json",
            "data/v53r2-continuous-parameter-smc2",
            "outputs/v53r2-continuous-parameter-smc2/evaluation-attempt.json",
            "outputs/v53r2-continuous-parameter-smc2/evaluation",
        )
    )
    if not downstream_absent:
        errors.append("V53r2 downstream artifact exists before implementation lock")
    audit = {
        "schema_version": 53,
        "revision": "r2",
        "experiment": "v53r2_implementation_audit",
        "passed": not errors,
        "decision": "authorize_v53r2_implementation_lock" if not errors else "repair_v53r2_implementation",
        "errors": errors,
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "checks": {
            "design_integrity": design_ok,
            "implementation_complete": not missing,
            "fresh_balanced_parameterized_registry": registry_ok,
            "exact_continuous_local_likelihood": exact_local_ok,
            "altered_seed_normalization": normalization_ok,
            "analytic_ambiguity_probe": ambiguity_ok,
            "random_stream_integrity": stream_ok,
            "pmcmc_proposal_fixture": pmcmc_ok,
            "fixed_control_pair": controls_ok,
            "downstream_absent": downstream_absent,
        },
        "smoke": {
            "records": len(smoke_records),
            "rows": len(rows),
            "exact_local_log_likelihood_error": exact_local_error,
            "pmcmc_acceptance_rate": pmcmc_acceptance,
            "outer_streams": len(streams["outer"]),
            "inner_streams": len(streams["inner"]),
            "outer_resampling_forced_only_in_fixture": True,
        },
        "data_access": {
            "altered_seed_smoke_records": len(smoke_records),
            "sealed_population_records_accessed": 0,
            "smc_squared_evaluation_runs": 0,
            "pmcmc_reference_evaluation_runs": 0,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


def canonical_program_text(template):
    return json.dumps(template, sort_keys=True, separators=(",", ":"))


if __name__ == "__main__":
    main()
