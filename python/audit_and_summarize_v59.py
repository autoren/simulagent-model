#!/usr/bin/env python3
"""Audit the sealed V59 result and write its immutable scientific summary."""
from __future__ import annotations

import argparse
import json

from evaluate_v59_planning import aggregate
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        default="outputs/v59-budgeted-root-sampled-planning/evaluation/result.json",
    )
    parser.add_argument(
        "--attempt",
        default="outputs/v59-budgeted-root-sampled-planning/evaluation-attempt.json",
    )
    parser.add_argument(
        "--audit",
        default="outputs/v59-budgeted-root-sampled-planning/post-result-audit.json",
    )
    parser.add_argument("--summary", default="docs/v59-results.md")
    args = parser.parse_args()
    result_path, attempt_path, audit_path, summary_path = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.result, args.attempt, args.audit, args.summary)
    )
    if audit_path.exists() or summary_path.exists():
        raise RuntimeError("V59 result has already been audited or summarized")
    result = json.loads(result_path.read_text())
    attempt = json.loads(attempt_path.read_text())
    lock_path = PROJECT_ROOT / result["evaluation_implementation_lock"]
    lock = json.loads(lock_path.read_text())
    seal_path = PROJECT_ROOT / result["population_seal"]
    seal = json.loads(seal_path.read_text())
    implementation = json.loads(
        (PROJECT_ROOT / seal["implementation_lock"]).read_text()
    )
    design = json.loads(
        (PROJECT_ROOT / implementation["design_lock"]).read_text()
    )
    implementation_audit_path = PROJECT_ROOT / implementation["implementation_audit"]
    implementation_audit = json.loads(implementation_audit_path.read_text())
    errors: list[str] = []

    binding_ok = (
        result["evaluation_run"] == 1
        and attempt["attempt"] == 1
        and result["evaluation_implementation_lock_sha256"] == file_sha256(lock_path)
        and attempt["evaluation_implementation_lock_sha256"] == file_sha256(lock_path)
        and result["population_seal_sha256"] == file_sha256(seal_path)
        and attempt["population_seal_sha256"] == file_sha256(seal_path)
        and lock["population_seal_sha256"] == file_sha256(seal_path)
        and all(
            file_sha256(PROJECT_ROOT / path) == digest
            for section in ("evaluation_files_sha256", "frozen_dependencies_sha256")
            for path, digest in lock[section].items()
        )
    )
    if not binding_ok:
        errors.append("V59 result is not bound to the single sealed attempt")

    recomputed = aggregate(
        result["records"], design["config_payload"],
        implementation_audit["fixture_metrics"],
    )
    aggregation_ok = (
        recomputed["metrics"] == result["metrics"]
        and recomputed["checks"] == result["qualification"]["checks"]
        and recomputed["passed"] == result["qualification"]["passed"]
        and len(recomputed["checks"]) == 17
    )
    if not aggregation_ok:
        errors.append("V59 metrics or noncompensatory gates do not reproduce")

    public_artifact = seal["artifacts"]["public_file"]
    firewall_ok = (
        result["candidate_population_records"] == seal_manifest_count(seal)
        and result["candidate_audit_truth_records_accessed"] == 0
        and result["metrics"]["integrity"]["truth_field_access_count"] == 0
        and result["metrics"]["integrity"]["future_observation_access_count"] == 0
        and result["metrics"]["integrity"]["latent_conditioned_rollout_access_count"] == 0
        and file_sha256(PROJECT_ROOT / public_artifact["path"])
        == public_artifact["sha256"]
    )
    if not firewall_ok:
        errors.append("V59 public-only candidate or latent/future firewall failed")

    metrics = result["metrics"]
    audit = {
        "schema_version": 59,
        "experiment": "v59_post_result_audit",
        "passed": not errors,
        "decision": (
            "freeze_v59_outcome" if not errors else "invalidate_v59_outcome"
        ),
        "errors": errors,
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "attempt": str(attempt_path.relative_to(PROJECT_ROOT)),
        "attempt_sha256": file_sha256(attempt_path),
        "evaluation_implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "evaluation_implementation_lock_sha256": file_sha256(lock_path),
        "population_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "population_seal_sha256": file_sha256(seal_path),
        "checks": {
            "single_attempt_and_frozen_bindings": binding_ok,
            "metric_and_seventeen_gate_recomputation": aggregation_ok,
            "public_only_truth_future_and_latent_firewalls": firewall_ok,
        },
        "qualification": result["qualification"],
        "metrics": metrics,
        "data_access": {
            "candidate_public_records": result["candidate_population_records"],
            "candidate_audit_truth_records": 0,
            "evaluation_runs": 1,
            "human_authored_v58_records": 0,
            "model_forward_passes": 0,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    exact = metrics["exact_reference"]
    returns = metrics["returns"]
    integrity = metrics["integrity"]
    status = "PASS" if result["qualification"]["passed"] else "FAIL"
    failed = [name for name, passed in result["qualification"]["checks"].items() if not passed]
    summary = f"""# V59 budgeted root-sampled planning results

## Outcome

**Qualification: {status}**

The sealed one-shot run evaluated 24 public tasks at horizons 3, 5, and 7, with three search budgets and three replicates per task-budget cell. It accessed no audit-truth records.

## Primary findings

- High-budget exact horizon-3 root-optimal-set membership: `{exact['high_budget_root_optimal_set_membership_rate']:.6f}`.
- High-budget exact horizon-3 mean root regret: `{exact['high_budget_mean_root_regret']:.6f}`.
- High-minus-low budget candidate return: `{returns['high_minus_low_budget_candidate_return']:.6f}`.
- Scale high-budget candidate minus observation-blind return: `{returns['scale_high_budget_candidate_minus_observation_blind_return']:.6f}`.
- Scale tasks with positive observation-contingency advantage: `{returns['scale_task_positive_observation_contingency_fraction']:.6f}`.
- Scale task-level paired lower 95% bound: `{returns['scale_task_high_budget_paired_difference_lower_95_bound']:.6f}`.
- Candidate tree observation-branching rate: `{integrity['tree_observation_branching_rate']:.6f}`.
- Deterministic replay and budget accounting rates: `{integrity['deterministic_replay_rate']:.6f}` and `{integrity['simulation_budget_accounting_rate']:.6f}`.

Failed preregistered gates: `{', '.join(failed) if failed else 'none'}`.

## Claim boundary

This result concerns bounded root-sampled observation-contingent search given the frozen exact calibrated belief and symbolic simulator. It does not establish exact long-horizon optimality, approximate-belief correctness, formal or worst-case safety, human-authored language robustness, or model/adapter performance. V58 remains deferred; synthetic records do not count as human evidence.

## Integrity

The population, evaluator, single attempt, result, audit, and outcome lock are hash-bound. Candidate evaluation opened only the sealed public population. Truth-field, future-observation, and latent-conditioned-rollout access counts are all zero.
"""
    summary_path.write_text(summary)
    print(json.dumps(audit, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


def seal_manifest_count(seal: dict) -> int:
    manifest = json.loads((PROJECT_ROOT / seal["manifest"]).read_text())
    return manifest["count"]


if __name__ == "__main__":
    main()
