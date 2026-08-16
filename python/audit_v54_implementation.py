#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import inspect
import json
import math

from generate_v53_smc2 import build_exact
from generate_v54_eig import (
    action_schedule,
    attach_adaptive_selection_and_outcome,
    build_history,
    history_class_for_record,
    selection_population_class_counts,
)
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v42_stateful import entities
from v53_smc2 import exact_inference, mechanic_registry
from v54_eig import (
    attempted_outcome_leak_selection,
    belief_atoms_from_exact,
    candidate_interventions,
    expected_information_gain_from_joint,
    score_all_interventions,
    score_control_policies,
    select_score,
)


IMPLEMENTATION_FILES = (
    "python/v54_eig.py",
    "python/generate_v54_eig.py",
    "python/test_v54_eig.py",
    "python/audit_v54_implementation.py",
)

BASE_DEPENDENCIES = (
    "python/v46_stochastic.py",
    "python/v49_belief.py",
    "python/v53_smc2.py",
    "configs/v53r2-design-lock.json",
    "configs/v53r2-outcome-lock.json",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-lock", default="configs/v54-design-lock.json")
    parser.add_argument(
        "--output", default="outputs/v54-exact-one-step-eig/implementation-audit.json"
    )
    args = parser.parse_args()
    design_path = (PROJECT_ROOT / args.design_lock).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    design = json.loads(design_path.read_text())
    config = design["config_payload"]
    v53 = json.loads(
        (PROJECT_ROOT / "configs/v53r2-design-lock.json").read_text()
    )["config_payload"]
    errors = []

    design_bound = (
        design["authorization"]["write_and_audit_exact_eig_implementation"]
        and not design["authorization"]["construct_v54_active_populations"]
        and not design["authorization"]["run_v54_active_evaluation"]
        and file_sha256(PROJECT_ROOT / design["config"]) == design["config_sha256"]
        and file_sha256(PROJECT_ROOT / design["preregistration"])
        == design["preregistration_sha256"]
        and file_sha256(PROJECT_ROOT / design["design_audit"])
        == design["design_audit_sha256"]
    )
    if not design_bound:
        errors.append("V54 design lock is not intact or does not authorize implementation")

    candidates_ok = all(
        len(candidate_interventions(entities(count))) == expected
        for count, expected in ((2, 5), (3, 13))
    )
    if not candidates_ok:
        errors.append("V54 candidate enumeration is incomplete")

    counts = selection_population_class_counts()
    allocation_ok = (
        counts == config["population"]["historyClasses"]
        and all(
            [history_class_for_record(program * 8 + offset) for offset in range(8)]
            == [
                "prior_like_all_wait", "prior_like_all_wait",
                "mixed_informative", "mixed_informative",
                "mixed_informative", "mixed_informative",
                "pending_delayed_event", "pending_delayed_event",
            ]
            for program in range(8)
        )
    )
    if not allocation_ok:
        errors.append("V54 history classes are imbalanced or truth-dependent")

    entity_rows = entities(3)
    schedule_checks = {
        "all_wait": all(
            row["id"] == "wait"
            for row in action_schedule(
                entity_rows, 5, "v54-audit", "prior_like_all_wait", True
            )
        ),
        "pending_context_ends_pulse": action_schedule(
            entity_rows, 5, "v54-audit", "pending_delayed_event", True
        )[-1]["id"] == "pulse",
        "mixed_begins_pulse_route": [
            row["id"] for row in action_schedule(
                entity_rows, 5, "v54-audit", "mixed_informative", True
            )[:2]
        ] == ["pulse", "route"],
    }
    if not all(schedule_checks.values()):
        errors.append("V54 history-class action fixtures are invalid")

    prior = {"low": 0.5, "high": 0.5}
    joint = {
        "failure": {"low": 0.375, "high": 0.125},
        "success": {"low": 0.125, "high": 0.375},
    }
    analytic = expected_information_gain_from_joint(prior, joint)
    expected = math.log(2) - (
        -0.25 * math.log(0.25) - 0.75 * math.log(0.75)
    )
    analytic_ok = (
        abs(analytic["eig"] - expected) <= 1e-14
        and abs(analytic["eig"] - analytic["entropy_eig"]) <= 1e-14
        and analytic["normalizes"] and analytic["finite"]
    )
    if not analytic_ok:
        errors.append("V54 closed-form mutual-information fixture failed")

    fixture = copy.deepcopy(v53)
    for key, value in tuple(fixture["population"].items()):
        if key.endswith("Seed") and isinstance(value, int):
            fixture["population"][key] = value + 3_000_000
    fixture["exactBenchmark"].update({
        "recordsPerTemplate": 1,
        "supportEpisodesPerRecord": 2,
        "supportSequenceLengths": [3, 4],
        "querySequenceLengths": [4, 5],
        "queryPrefixLengths": [3, 4],
        "quadratureNodes": 17,
    })
    registry = mechanic_registry(fixture["population"]["templateSeed"])

    generator_fixture = copy.deepcopy(config)
    for key, value in tuple(generator_fixture["population"].items()):
        if key.endswith("Seed") and isinstance(value, int):
            generator_fixture["population"][key] = value + 4_000_000
    used = set()
    generated_fixtures = [
        build_history(
            "selection", record_index, registry, generator_fixture, used, set()
        )
        for record_index in (0, 2, 6)
    ]
    adaptive_fixture = build_history(
        "adaptive_sbc", 0, registry, generator_fixture, used, set()
    )
    adaptive_fixture = attach_adaptive_selection_and_outcome(
        adaptive_fixture, registry, fixture, generator_fixture
    )
    generator_fixture_ok = (
        [row["history_class"] for row in generated_fixtures]
        == [
            "prior_like_all_wait", "mixed_informative", "pending_delayed_event"
        ]
        and all(
            len(row["public_history"]["query"]["actions"])
            == len(row["public_history"]["query"]["observations"])
            for row in generated_fixtures
        )
        and len(adaptive_fixture["realized_outcome"]["observations"]) == 3
        and adaptive_fixture["selected_intervention"]["key"]
        in {
            row["key"]
            for row in candidate_interventions(
                adaptive_fixture["public_history"]["query"]["entities"]
            )
        }
    )
    if not generator_fixture_ok:
        errors.append("V54 altered-seed history or adaptive-outcome generator fixture failed")

    record = build_exact(registry[:1], fixture, set(), set())[0]
    exact = exact_inference(registry, record, fixture)
    atoms = belief_atoms_from_exact(exact)
    scores = score_all_interventions(
        atoms, registry, record["query"]["entities"],
        record["query"]["prefix_length"],
    )
    maximum_reference_error = max(
        abs(row["eig"] - row["reference_eig"]) for row in scores
    )
    maximum_identity_error = max(
        abs(row["eig"] - row["entropy_eig"]) for row in scores
    )
    exact_fixture_ok = (
        len(scores) == 5
        and all(row["normalizes"] and row["finite"] for row in scores)
        and min(row["eig"] for row in scores) >= -1e-12
        and maximum_reference_error <= 1e-12
        and maximum_identity_error <= 1e-12
        and select_score(scores)["selected"]["intervention_key"]
        in select_score(scores)["optimal_keys"]
    )
    if not exact_fixture_ok:
        errors.append("V54 altered-seed exact EIG fixture failed")

    controls = score_control_policies(
        atoms, registry, record["query"]["entities"],
        record["query"]["prefix_length"],
    )
    controls_ok = set(controls) == {
        "primary", "uniform_random_mean_eig", "predictive_entropy",
        "state_only_information", "map_program", "theta_point_mass",
        "likelihood_squared",
    }
    leakage_rejected = False
    try:
        attempted_outcome_leak_selection({}, {"secret": True})
    except PermissionError:
        leakage_rejected = True
    selector_signature_ok = not ({"truth", "outcome", "realized_outcome"} & set(
        inspect.signature(score_all_interventions).parameters
    ))
    if not controls_ok or not leakage_rejected or not selector_signature_ok:
        errors.append("V54 controls or pre-outcome selection firewall are incomplete")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v54-implementation-lock.json",
            "configs/v54-population-seal.json",
            "configs/v54-outcome-lock.json",
            "data/v54-exact-one-step-eig",
            "outputs/v54-exact-one-step-eig/evaluation-attempt.json",
            "outputs/v54-exact-one-step-eig/evaluation",
        )
    )
    if not downstream_absent:
        errors.append("V54 downstream population or evaluation artifact exists")

    audit = {
        "schema_version": 54,
        "experiment": "v54_implementation_audit",
        "passed": not errors,
        "decision": (
            "authorize_v54_implementation_lock" if not errors
            else "repair_v54_implementation"
        ),
        "errors": errors,
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "implementation_files_sha256": {
            path: file_sha256(PROJECT_ROOT / path) for path in IMPLEMENTATION_FILES
        },
        "base_dependencies_sha256": {
            path: file_sha256(PROJECT_ROOT / path) for path in BASE_DEPENDENCIES
        },
        "checks": {
            "design_bound_and_prepopulation": design_bound,
            "complete_candidate_enumeration": candidates_ok,
            "truth_independent_history_class_allocation": allocation_ok,
            "history_class_action_fixtures": schedule_checks,
            "altered_seed_population_generator_fixture": generator_fixture_ok,
            "closed_form_mutual_information": analytic_ok,
            "altered_seed_exact_fixture": exact_fixture_ok,
            "control_implementations": controls_ok,
            "outcome_leakage_rejected": leakage_rejected,
            "selector_signature_excludes_truth_and_outcome": selector_signature_ok,
            "downstream_absent": downstream_absent,
        },
        "fixture_metrics": {
            "candidate_count": len(scores),
            "maximum_primary_reference_eig_error": maximum_reference_error,
            "maximum_entropy_identity_error": maximum_identity_error,
            "minimum_eig": min(row["eig"] for row in scores),
            "maximum_eig": max(row["eig"] for row in scores),
            "analytic_binary_eig": analytic["eig"],
        },
        "data_access": {
            "v54_candidate_population_records_accessed": 0,
            "v54_population_generator_executions": 0,
            "v54_active_evaluation_runs": 0,
            "v54_adaptive_sbc_runs": 0,
            "altered_seed_implementation_fixture_records": 1,
            "altered_seed_generator_fixture_records": 4,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
