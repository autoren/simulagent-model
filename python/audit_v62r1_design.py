#!/usr/bin/env python3
"""Audit the V62r1 terminal-residual measurement-repair design."""
from __future__ import annotations

import argparse
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/v62r1-terminal-bellman-residual-repair.json"
    )
    parser.add_argument(
        "--plan", default="docs/v62r1-terminal-bellman-residual-repair-plan.md"
    )
    parser.add_argument(
        "--output", default="outputs/v62r1-terminal-residual-repair/design-audit.json"
    )
    args = parser.parse_args()
    config_path, plan_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.config, args.plan, args.output)
    )
    config = json.loads(config_path.read_text())
    errors: list[str] = []

    outcome_path = PROJECT_ROOT / config["sourceV62OutcomeLock"]
    evaluation_lock_path = PROJECT_ROOT / config[
        "sourceV62EvaluationImplementationLock"
    ]
    seal_path = PROJECT_ROOT / config["sourceV62ExternalBundleSeal"]
    diagnostic_path = PROJECT_ROOT / config["sourcePostHocDiagnostic"]
    outcome = json.loads(outcome_path.read_text())
    evaluation_lock = json.loads(evaluation_lock_path.read_text())
    seal = json.loads(seal_path.read_text())
    diagnostic = json.loads(diagnostic_path.read_text())

    source_ok = (
        not outcome["qualification_passed"]
        and outcome["failed_checks"] == ["independent_bellman_residual"]
        and outcome["authorization"]["preregister_terminal_residual_measurement_repair"]
        and outcome["evaluation_implementation_lock_sha256"]
        == file_sha256(evaluation_lock_path)
        and outcome["external_bundle_seal_sha256"] == file_sha256(seal_path)
        and outcome["result_sha256"] == file_sha256(PROJECT_ROOT / outcome["result"])
    )
    if not source_ok:
        errors.append("V62 is not an intact isolated failed source outcome")

    counts = diagnostic["node_counts"]
    diagnostic_ok = (
        diagnostic["passed"]
        and diagnostic["outcome_lock_sha256"] == file_sha256(outcome_path)
        and diagnostic["result_sha256"] == outcome["result_sha256"]
        and counts
        == {
            "reachable": 66,
            "terminal_support": 4,
            "old_residual_failures": 4,
            "nonterminal_old_residual_failures": 0,
            "terminal_decision_violations": 0,
        }
        and all(diagnostic["checks"].values())
    )
    if not diagnostic_ok:
        errors.append("post-hoc evidence does not support a terminal-only repair")

    change = config["registeredChange"]
    change_ok = (
        set(change["allowedImplementationChanges"])
        == {
            "new_terminal_aware_residual_checker",
            "repair_tests_mutation_audit_rescore_reporting_and_lock_plumbing",
        }
        and "positive_belief_support_is_entirely_absorbing" in change["newMetric"]
        and "value_or_q_value" in change["newMetric"]
    )
    if not change_ok:
        errors.append("registered repair is missing or broader than the terminal metric")

    rescore = config["repairRescore"]
    audit = config["implementationAudit"]
    gates_ok = (
        rescore["taskCells"] == 6
        and rescore["reachableBeliefNodes"] == 66
        and rescore["attempts"] == 1
        and rescore["maximumCorrectedResidual"] <= 1e-10
        and rescore["oldResidualReproductionTolerance"] == 0.0
        and rescore["otherMetricReproductionTolerance"] == 0.0
        and len(audit["analyticFixtures"]) == 5
        and len(audit["mutants"]) == 6
        and audit["requiredFixturePassRate"] == 1.0
        and audit["requiredMutantKillRate"] == 1.0
        and len(config["gates"]) == 12
    )
    if not gates_ok:
        errors.append("repair census, audit, or noncompensatory gates changed")

    immutable = config["immutableInputs"]
    boundary = config["claimBoundary"]
    firewall_ok = (
        all(value == "forbidden" for value in config["firewall"].values())
        and not immutable["newExternalRollouts"]
        and not immutable["newCandidateEvaluation"]
        and not immutable["newCorpusOrModel"]
        and not immutable["humanOrModelAccess"]
        and boundary["measurementRepairOnImmutableV62Artifacts"]
        and not boundary["independentExternalReplication"]
        and not boundary["smc2UnknownMechanicPortability"]
        and not boundary["formalSafety"]
    )
    if not firewall_ok:
        errors.append("repair immutability, firewall, or claim boundary is invalid")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v62r1-design-lock.json",
            "configs/v62r1-implementation-lock.json",
            "configs/v62r1-outcome-lock.json",
            "outputs/v62r1-terminal-residual-repair/rescore-attempt.json",
            "outputs/v62r1-terminal-residual-repair/rescore/result.json",
            "docs/v62r1-results.md",
        )
    )
    if not downstream_absent:
        errors.append("V62r1 downstream artifacts already exist")

    result = {
        "schema_version": "62r1",
        "experiment": "v62r1_design_audit",
        "passed": not errors,
        "decision": "freeze_v62r1_design" if not errors else "reject_v62r1_design",
        "errors": errors,
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "source_v62_outcome_lock": str(outcome_path.relative_to(PROJECT_ROOT)),
        "source_v62_outcome_lock_sha256": file_sha256(outcome_path),
        "source_v62_evaluation_implementation_lock": str(
            evaluation_lock_path.relative_to(PROJECT_ROOT)
        ),
        "source_v62_evaluation_implementation_lock_sha256": file_sha256(
            evaluation_lock_path
        ),
        "source_v62_external_bundle_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "source_v62_external_bundle_seal_sha256": file_sha256(seal_path),
        "source_post_hoc_diagnostic": str(diagnostic_path.relative_to(PROJECT_ROOT)),
        "source_post_hoc_diagnostic_sha256": file_sha256(diagnostic_path),
        "checks": {
            "isolated_immutable_v62_failure": source_ok,
            "terminal_only_post_hoc_localization": diagnostic_ok,
            "single_registered_measurement_change": change_ok,
            "frozen_rescore_audit_and_twelve_gates": gates_ok,
            "immutability_firewall_and_claim_boundary": firewall_ok,
            "downstream_absence": downstream_absent,
        },
        "data_access": {
            "repair_rescores": 0,
            "new_candidate_evaluations": 0,
            "new_external_rollouts": 0,
            "human_records": 0,
            "model_forward_passes": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
