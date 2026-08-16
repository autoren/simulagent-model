#!/usr/bin/env python3
"""Recompute, audit, and summarize the single V60 result."""
from __future__ import annotations

import argparse
import json

from evaluate_v60_decision_calibration import aggregate
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result", default="outputs/v60-approximate-belief-decision-calibration/evaluation/result.json"
    )
    parser.add_argument(
        "--attempt", default="outputs/v60-approximate-belief-decision-calibration/evaluation-attempt.json"
    )
    parser.add_argument(
        "--audit", default="outputs/v60-approximate-belief-decision-calibration/post-result-audit.json"
    )
    parser.add_argument("--summary", default="docs/v60-results.md")
    args = parser.parse_args()
    result_path, attempt_path, audit_path, summary_path = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.result, args.attempt, args.audit, args.summary)
    )
    if audit_path.exists() or summary_path.exists():
        raise RuntimeError("V60 result has already been audited or summarized")
    result = json.loads(result_path.read_text())
    attempt = json.loads(attempt_path.read_text())
    evaluation_lock_path = PROJECT_ROOT / result["evaluation_implementation_lock"]
    evaluation_lock = json.loads(evaluation_lock_path.read_text())
    implementation_path = PROJECT_ROOT / evaluation_lock["implementation_lock"]
    implementation = json.loads(implementation_path.read_text())
    design = json.loads((PROJECT_ROOT / implementation["design_lock"]).read_text())
    implementation_audit = json.loads(
        (PROJECT_ROOT / implementation["implementation_audit"]).read_text()
    )
    seal_path = PROJECT_ROOT / result["population_seal"]
    errors: list[str] = []

    binding_ok = (
        result["evaluation_run"] == attempt["attempt"] == 1
        and result["evaluation_implementation_lock_sha256"]
        == file_sha256(evaluation_lock_path)
        and attempt["evaluation_implementation_lock_sha256"]
        == file_sha256(evaluation_lock_path)
        and result["population_seal_sha256"] == file_sha256(seal_path)
        and attempt["population_seal_sha256"] == file_sha256(seal_path)
        and all(
            file_sha256(PROJECT_ROOT / path) == digest
            for section in ("evaluation_files_sha256", "frozen_dependencies_sha256")
            for path, digest in evaluation_lock[section].items()
        )
    )
    if not binding_ok:
        errors.append("V60 result is not bound to the single frozen attempt")

    recomputed = aggregate(
        result["records"], design["config_payload"],
        implementation_audit["fixture_metrics"],
    )
    aggregation_ok = (
        recomputed["metrics"] == result["metrics"]
        and recomputed["checks"] == result["qualification"]["checks"]
        and recomputed["passed"] == result["qualification"]["passed"]
        and len(recomputed["checks"]) == 23
    )
    if not aggregation_ok:
        errors.append("V60 metrics or 23 qualification gates do not reproduce")

    seal = json.loads(seal_path.read_text())
    public_artifact = seal["artifacts"]["public_file"]
    firewall_ok = (
        result["candidate_public_records"] == design["config_payload"]["population"]["publicTasks"]
        and result["candidate_audit_records_accessed"] == 0
        and result["metrics"]["integrity"]["truth_field_access_count"] == 0
        and result["metrics"]["integrity"]["future_observation_access_count"] == 0
        and result["metrics"]["integrity"]["latent_conditioned_rollout_access_count"] == 0
        and file_sha256(PROJECT_ROOT / public_artifact["path"]) == public_artifact["sha256"]
    )
    if not firewall_ok:
        errors.append("V60 public-only candidate or rollout firewall failed")

    audit = {
        "schema_version": 60,
        "experiment": "v60_post_result_audit",
        "passed": not errors,
        "decision": "freeze_v60_outcome" if not errors else "invalidate_v60_outcome",
        "errors": errors,
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "attempt": str(attempt_path.relative_to(PROJECT_ROOT)),
        "attempt_sha256": file_sha256(attempt_path),
        "evaluation_implementation_lock": str(evaluation_lock_path.relative_to(PROJECT_ROOT)),
        "evaluation_implementation_lock_sha256": file_sha256(evaluation_lock_path),
        "population_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "population_seal_sha256": file_sha256(seal_path),
        "checks": {
            "single_attempt_and_frozen_bindings": binding_ok,
            "metric_and_twenty_three_gate_recomputation": aggregation_ok,
            "public_only_truth_future_and_latent_firewalls": firewall_ok,
        },
        "qualification": result["qualification"],
        "metrics": result["metrics"],
        "data_access": {
            "candidate_public_records": result["candidate_public_records"],
            "candidate_audit_records": 0,
            "evaluation_runs": 1,
            "human_authored_v58_records": 0,
            "model_forward_passes": 0,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    metrics = result["metrics"]
    belief, decision, returns, integrity = (
        metrics["belief"], metrics["decision"], metrics["returns"], metrics["integrity"]
    )
    failed = [name for name, passed in result["qualification"]["checks"].items() if not passed]
    status = "PASS" if result["qualification"]["passed"] else "FAIL"
    summary = f"""# V60 approximate-belief decision-calibration results

## Outcome

**Qualification: {status}**

The single sealed run composed the frozen V53r2 SMC² posterior with the frozen V59 search semantics on all 24 V59 public tasks. Candidate evaluation accessed no audit records, and all deployed policies were scored under the exact posterior.

## Posterior agreement at 509 outer particles

- Mean program TV: `{belief['primary_mean_program_tv']:.6f}`.
- Mean theta Wasserstein distance: `{belief['primary_mean_theta_wasserstein']:.6f}`.
- Mean binned program-theta TV: `{belief['primary_mean_binned_program_theta_tv']:.6f}`.
- Mean / q95 current-configuration TV: `{belief['primary_mean_configuration_tv']:.6f}` / `{belief['primary_q95_configuration_tv']:.6f}`.

## Decision and return calibration

- Horizon-3 exact optimal-set membership: `{decision['primary_horizon_three_exact_optimal_set_membership_rate']:.6f}`.
- Horizon-3 mean exact root regret: `{decision['primary_horizon_three_mean_exact_root_regret']:.6f}`.
- Approximate/exact-belief search root-action agreement: `{decision['primary_approximate_exact_search_root_action_agreement_rate']:.6f}`.
- Exact-belief minus approximate-belief policy return: `{decision['primary_exact_belief_minus_approximate_belief_policy_return']:.6f}`.
- Scale approximate minus observation-blind return: `{returns['primary_scale_approximate_minus_observation_blind_return']:.6f}`.
- Scale task-level lower 95% bound: `{returns['primary_scale_observation_contingency_lower_95_bound']:.6f}`.
- Primary-minus-low inference-budget approximate return: `{returns['primary_minus_low_budget_approximate_policy_return']:.6f}`.

Failed preregistered gates: `{', '.join(failed) if failed else 'none'}`.

## Boundary

This outcome concerns the frozen SMC² implementation, eight-template symbolic registry, 24-task population, and bounded 1,024-simulation search. It does not establish exact long-horizon optimality, general-purpose or amortized inference, formal safety, human-authored language robustness, or model/adapter performance. V58 remains deferred.

## Integrity

Normalization, simulation-budget accounting, deterministic replay, finite returns, and implementation-mutant detection were noncompensatory. Truth-field, future-observation, and latent-conditioned-rollout access counts are zero; budget accounting was `{integrity['simulation_budget_accounting_rate']:.6f}`.
"""
    summary_path.write_text(summary)
    print(json.dumps(audit, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
