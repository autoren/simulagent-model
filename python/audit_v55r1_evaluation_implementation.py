#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import inspect
import json

from evaluate_v55r1_planning import aggregate, evaluate_record
from generate_v55r1_planning import build_record, prior_observation_design_keys
from v10_protocol import file_sha256
from v22_relational import unary_atom
from v22r2_grounding import PROJECT_ROOT
from v55r1_planning import delay_suppressed_registry, planning_registry


EVALUATION_FILES = (
    "python/evaluate_v55r1_planning.py",
    "python/audit_and_summarize_v55r1.py",
    "python/freeze_v55r1_outcome.py",
    "python/audit_v55r1_evaluation_implementation.py",
)

FROZEN_DEPENDENCIES = (
    "python/v53_smc2.py",
    "python/v54_eig.py",
    "python/v55_planning.py",
    "python/v55r1_planning.py",
    "python/generate_v55r1_planning.py",
    "configs/v53r2-design-lock.json",
    "configs/v55-outcome-lock.json",
    "configs/v55r1-design-lock.json",
    "configs/v55r1-implementation-lock.json",
    "configs/v55r1-population-seal.json",
)


def passing_stub(identifier: int, history: str) -> dict:
    sensitive = identifier < 2
    return {
        "id": f"stub_{identifier}",
        "record": identifier,
        "history_class": history,
        "goal": {"atom": "u:active:unit_0", "value": False},
        "belief_atoms": 1,
        "root_value": 0.5,
        "reference_value": 0.5,
        "root_value_error": 0.0,
        "root_optimal_set_member": True,
        "independent_policy_value": 0.5,
        "independent_policy_error": 0.0,
        "selected_action_key": "wait",
        "delay_suppressed_value": 0.4 if sensitive else 0.5,
        "delay_suppressed_optimal_action_keys": ["wait"],
        "root_action_changes_under_delay_suppression": False,
        "absolute_root_value_change_under_delay_suppression": 0.1 if sensitive else 0.0,
        "delayed_consequence_sensitive": sensitive,
        "integrity": {
            "normalization_checks": 1,
            "normalization_passes": 1,
            "candidate_omissions": 0,
            "tie_break_violations": 0,
            "finite_checks": 1,
            "finite_passes": 1,
        },
        "finite": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--population-seal", default="configs/v55r1-population-seal.json"
    )
    parser.add_argument(
        "--output",
        default="outputs/v55r1-delayed-consequence-adequacy-confirmation/evaluation-implementation-audit.json",
    )
    args = parser.parse_args()
    seal_path = (PROJECT_ROOT / args.population_seal).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    seal = json.loads(seal_path.read_text())
    implementation = json.loads(
        (PROJECT_ROOT / seal["implementation_lock"]).read_text()
    )
    design = json.loads(
        (PROJECT_ROOT / implementation["design_lock"]).read_text()
    )
    config = design["config_payload"]
    v53 = json.loads(
        (PROJECT_ROOT / "configs/v53r2-design-lock.json").read_text()
    )["config_payload"]
    errors: list[str] = []

    seal_bound = (
        seal["authorization"]["write_and_audit_v55r1_evaluation_implementation"]
        and not seal["authorization"]["run_v55r1_evaluation"]
        and file_sha256(PROJECT_ROOT / seal["population"])
        == seal["population_sha256"]
        and file_sha256(PROJECT_ROOT / seal["population_audit"])
        == seal["population_audit_sha256"]
        and file_sha256(PROJECT_ROOT / seal["implementation_lock"])
        == seal["implementation_lock_sha256"]
    )
    if not seal_bound:
        errors.append("V55r1 population seal is not intact")

    evaluation_source = inspect.getsource(evaluate_record)
    firewall_ok = (
        "truth" not in evaluation_source
        and "future_observation" not in evaluation_source
        and set(inspect.signature(evaluate_record).parameters)
        == {"public_record", "registry", "v53_config", "config", "suppressed"}
    )
    if not firewall_ok:
        errors.append("V55r1 evaluator can access truth or future outcomes")

    fixture_config = copy.deepcopy(config)
    for key, value in tuple(fixture_config["population"].items()):
        if key.endswith("Seed") and isinstance(value, int):
            fixture_config["population"][key] = value + 7_000_000
    registry = planning_registry(fixture_config)
    used, prior = set(), prior_observation_design_keys()
    fixture = build_record(
        1, 1,
        {"atom": unary_atom("active", "unit_1"), "value": True},
        registry, fixture_config, used, prior,
    )
    public_fixture = {
        "id": fixture["id"],
        "record": fixture["record"],
        "history_class": fixture["history_class"],
        "public": fixture["public"],
    }
    exact_config = copy.deepcopy(v53)
    exact_config["exactBenchmark"]["quadratureNodes"] = 5
    fixture_result = evaluate_record(
        public_fixture, registry, exact_config, fixture_config,
        delay_suppressed_registry(registry, 3),
    )
    exact_fixture_ok = (
        fixture_result["root_value_error"] <= 1e-12
        and fixture_result["independent_policy_error"] <= 1e-12
        and fixture_result["root_optimal_set_member"]
        and fixture_result["finite"]
        and fixture_result["integrity"]["candidate_omissions"] == 0
        and fixture_result["integrity"]["tie_break_violations"] == 0
    )
    if not exact_fixture_ok:
        errors.append("V55r1 altered-seed evaluator fixture failed")

    stubs = [
        passing_stub(index, "prior_like_all_wait" if index % 2 == 0 else "mixed_informative")
        for index in range(16)
    ]
    aggregated = aggregate(stubs, config)
    aggregate_ok = (
        len(aggregated["checks"]) == 14
        and aggregated["passed"]
        and aggregated["metrics"]["delayed_consequence"][
            "delayed_consequence_sensitive_task_count"
        ] == 2
        and aggregated["metrics"]["delayed_consequence"][
            "delayed_sensitive_task_count_by_history_class"
        ] == {"prior_like_all_wait": 1, "mixed_informative": 1}
    )
    if not aggregate_ok:
        errors.append("V55r1 metric or qualification aggregation is invalid")

    single_attempt_ok = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v55r1-evaluation-implementation-lock.json",
            "configs/v55r1-outcome-lock.json",
            "outputs/v55r1-delayed-consequence-adequacy-confirmation/evaluation-attempt.json",
            "outputs/v55r1-delayed-consequence-adequacy-confirmation/evaluation",
            "outputs/v55r1-delayed-consequence-adequacy-confirmation/post-result-audit.json",
            "docs/v55r1-results.md",
        )
    )
    if not single_attempt_ok:
        errors.append("V55r1 evaluation or downstream artifact already exists")

    audit = {
        "schema_version": 55,
        "revision": "r1",
        "experiment": "v55r1_evaluation_implementation_audit",
        "passed": not errors,
        "decision": (
            "authorize_v55r1_evaluation_implementation_lock" if not errors
            else "repair_v55r1_evaluation_implementation"
        ),
        "errors": errors,
        "population_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "population_seal_sha256": file_sha256(seal_path),
        "evaluation_files_sha256": {
            path: file_sha256(PROJECT_ROOT / path) for path in EVALUATION_FILES
        },
        "frozen_dependencies_sha256": {
            path: file_sha256(PROJECT_ROOT / path) for path in FROZEN_DEPENDENCIES
        },
        "checks": {
            "sealed_population_and_implementation": seal_bound,
            "truth_and_future_observation_firewall": firewall_ok,
            "altered_seed_exact_evaluator_fixture": exact_fixture_ok,
            "fourteen_noncompensatory_qualification_checks": aggregate_ok,
            "single_attempt_and_downstream_absence": single_attempt_ok,
        },
        "fixture_metrics": {
            "root_value": fixture_result["root_value"],
            "root_value_error": fixture_result["root_value_error"],
            "independent_policy_error": fixture_result["independent_policy_error"],
            "absolute_delay_suppression_value_change": fixture_result[
                "absolute_root_value_change_under_delay_suppression"
            ],
            "belief_atoms": fixture_result["belief_atoms"],
            "qualification_check_count": len(aggregated["checks"]),
        },
        "data_access": {
            "v55r1_candidate_population_records_accessed": 0,
            "v55r1_planning_evaluation_runs": 0,
            "additional_v55_planning_evaluation_runs": 0,
            "altered_seed_evaluator_fixture_records": 1,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
    }
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
