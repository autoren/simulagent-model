#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from evaluate_v55_planning import aggregate, qualification
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        default="outputs/v55-short-horizon-bayes-adaptive-planning/evaluation/result.json",
    )
    parser.add_argument(
        "--output",
        default="outputs/v55-short-horizon-bayes-adaptive-planning/post-result-audit.json",
    )
    parser.add_argument("--summary", default="docs/v55-results.md")
    args = parser.parse_args()
    result_path, output, summary = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.result, args.output, args.summary)
    )
    result = json.loads(result_path.read_text())
    seal_path = PROJECT_ROOT / result["population_seal"]
    seal = json.loads(seal_path.read_text())
    implementation_path = PROJECT_ROOT / seal["implementation_lock"]
    implementation = json.loads(implementation_path.read_text())
    design = json.loads((PROJECT_ROOT / implementation["design_lock"]).read_text())
    config = design["config_payload"]
    errors = []

    bindings_ok = (
        result["evaluation_run"] == 1
        and result["population_seal_sha256"] == file_sha256(seal_path)
        and result["evaluation_implementation_lock_sha256"]
        == file_sha256(PROJECT_ROOT / result["evaluation_implementation_lock"])
        and seal["population"]["sha256"]
        == file_sha256(PROJECT_ROOT / seal["population"]["path"])
    )
    if not bindings_ok:
        errors.append("V55 result is not bound to the sealed population and implementation")

    recomputed_metrics = aggregate(result["records"], config, True)
    aggregation_ok = recomputed_metrics == result["metrics"]
    recomputed_qualification = qualification(recomputed_metrics, config["gates"])
    qualification_ok = recomputed_qualification == result["qualification"]
    check_count_ok = len(recomputed_qualification["checks"]) == 20
    if not aggregation_ok or not qualification_ok or not check_count_ok:
        errors.append("V55 metric aggregation or noncompensatory qualification is not reproducible")

    record_count_ok = (
        len(result["records"]) == config["population"]["planningTasks"]
        and len({row["id"] for row in result["records"]})
        == len(result["records"])
    )
    record_schema_ok = all(
        set(row["baseline_values"]) == {
            "open_loop", "greedy", "map_program", "posterior_mean_theta",
            "eig_only", "belief_update_disabled",
        }
        and set(row["baseline_regrets"]) == set(row["baseline_values"])
        and row["belief_atoms"] > 0
        for row in result["records"]
    )
    if not record_count_ok or not record_schema_ok:
        errors.append("V55 result records are incomplete, duplicated, or malformed")

    metrics = result["metrics"]
    exact = metrics["exact_correctness"]
    integrity = metrics["integrity"]
    independent_ok = (
        exact["maximum_root_value_error_against_scalar_reference"] <= 1e-10
        and exact["maximum_independent_policy_evaluation_error"] <= 1e-10
        and exact["maximum_bellman_residual"] <= 1e-10
        and exact["root_optimal_set_membership_rate"] == 1.0
    )
    integrity_ok = all(value == 0 for value in integrity.values())
    if not independent_ok or not integrity_ok:
        errors.append("V55 exact-reference or integrity audit failed")

    audit = {
        "schema_version": 55,
        "experiment": "v55_post_result_audit",
        "passed": not errors,
        "decision": (
            "accept_v55_result" if not errors else "reject_v55_result_artifact"
        ),
        "errors": errors,
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "population_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "population_seal_sha256": file_sha256(seal_path),
        "checks": {
            "sealed_bindings": bindings_ok,
            "metric_aggregation_reproduced": aggregation_ok,
            "qualification_reproduced": qualification_ok,
            "qualification_check_count": check_count_ok,
            "record_count_ids_and_schema": record_count_ok and record_schema_ok,
            "independent_exact_references": independent_ok,
            "selection_and_stream_integrity": integrity_ok,
        },
        "qualification": recomputed_qualification,
        "data_access": {
            "additional_planning_evaluation_runs": 0,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    decision = (
        "authorize symbolic/probabilistic policy-verification preregistration only"
        if recomputed_qualification["passed"] and not errors
        else "do not advance; localize the failed sealed V55 gates"
    )
    quality = metrics["decision_quality"]
    behavior = metrics["nonmyopic_behavior"]
    controls = metrics["controls"]
    summary.write_text(
        "# V55 results: exact short-horizon Bayes-adaptive planning\n\n"
        f"Decision: `{decision}`.\n\n"
        "V55 evaluates exact three-action belief-space planning over program identity, "
        "continuous theta quadrature, and hidden world/queue configuration. It does not "
        "claim long-horizon, approximate, learned, language-grounded, or formally verified planning.\n\n"
        "## Sealed results\n\n"
        f"- All qualification gates passed: `{recomputed_qualification['passed']}` "
        f"({sum(recomputed_qualification['checks'].values())}/20).\n"
        f"- Maximum primary/scalar root-value error: "
        f"`{exact['maximum_root_value_error_against_scalar_reference']}`.\n"
        f"- Maximum independent policy-evaluation error: "
        f"`{exact['maximum_independent_policy_evaluation_error']}`.\n"
        f"- Mean Bayes-adaptive value: `{quality['mean_bayes_adaptive_value']}`.\n"
        f"- Mean open-loop value: `{quality['mean_open_loop_value']}`.\n"
        f"- Mean adaptive minus open-loop value: "
        f"`{quality['mean_bayes_adaptive_minus_open_loop_value']}`.\n"
        f"- Positive value-of-adaptation fraction: "
        f"`{quality['positive_value_of_adaptation_fraction']}`.\n"
        f"- Non-myopic root-action fraction: "
        f"`{behavior['nonmyopic_root_action_fraction']}`.\n"
        f"- Information-then-control fraction: "
        f"`{behavior['information_then_control_policy_fraction']}`.\n"
        f"- Delayed-consequence sensitivity fraction: "
        f"`{behavior['delayed_consequence_sensitive_policy_fraction']}`.\n"
        f"- Controls detected or dominated: `{controls['detected_or_dominated']}/7`.\n"
        f"- Clairvoyant upper-bound violation rate: "
        f"`{quality['clairvoyant_upper_bound_violation_rate']}`.\n"
        f"- Integrity violations: `{sum(integrity.values())}`.\n\n"
        "## Boundary\n\n"
        "A full pass authorizes only a new preregistration for symbolic and probabilistic "
        "verification of the frozen finite-horizon policy. Formal verification itself, "
        "longer horizons, approximate search, language grounding, model access, and training remain blocked.\n"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
