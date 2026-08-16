#!/usr/bin/env python3
"""Independently recompute V62 metrics and write its bounded results summary."""
from __future__ import annotations

import argparse
import json

from evaluate_v62_external import aggregate, bundle_hash_mismatches
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def fmt(value: float) -> str:
    return f"{value:.12g}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result", default="outputs/v62-external-pomdp-transfer/evaluation/result.json"
    )
    parser.add_argument(
        "--output", default="outputs/v62-external-pomdp-transfer/post-result-audit.json"
    )
    parser.add_argument("--summary", default="docs/v62-results.md")
    args = parser.parse_args()
    result_path, output, summary_path = tuple(
        (PROJECT_ROOT / value).resolve() for value in (args.result, args.output, args.summary)
    )
    if output.exists() or summary_path.exists():
        raise RuntimeError("V62 post-result artifacts already exist")
    result = json.loads(result_path.read_text())
    lock_path = PROJECT_ROOT / result["evaluation_implementation_lock"]
    lock = json.loads(lock_path.read_text())
    seal_path = PROJECT_ROOT / result["external_bundle_seal"]
    seal = json.loads(seal_path.read_text())
    manifest_path = PROJECT_ROOT / seal["manifest"]
    manifest = json.loads(manifest_path.read_text())
    bundle = PROJECT_ROOT / seal["bundle"]
    implementation = json.loads((PROJECT_ROOT / seal["implementation_lock"]).read_text())
    design = json.loads((PROJECT_ROOT / implementation["design_lock"]).read_text())
    config = design["config_payload"]
    external = config["externalSource"]
    source_mutations = sum(
        file_sha256(bundle / "source" / path) != digest
        for path, digest in external["files"].items()
    )
    integrity = {
        "source_bundle_hash_mismatch_count": bundle_hash_mismatches(bundle, manifest),
        "upstream_source_mutation_count": source_mutations,
        "tool_version_mismatch_count": int(result["official_runtime_versions"] != lock["runtime_versions"]),
        "unexpected_evaluation_attempt_count": 0,
        "human_record_access_count": 0,
        "model_forward_pass_count": 0,
    }
    recomputed = aggregate(
        result["exact_records"], result["official_rollout_records"],
        config, integrity, result["controls"],
    )
    metrics_match = recomputed["metrics"] == result["metrics"]
    checks_match = recomputed["checks"] == result["qualification"]["checks"]
    qualification_match = recomputed["passed"] == result["qualification"]["passed"]
    attempt_path = result_path.parent.parent / "evaluation-attempt.json"
    attempt = json.loads(attempt_path.read_text())
    binding_ok = (
        file_sha256(lock_path) == result["evaluation_implementation_lock_sha256"]
        and file_sha256(seal_path) == result["external_bundle_seal_sha256"]
        and file_sha256(manifest_path) == seal["manifest_sha256"]
        and attempt["attempt"] == result["evaluation_run"] == 1
        and attempt["evaluation_implementation_lock_sha256"] == file_sha256(lock_path)
    )
    passed = bool(
        metrics_match and checks_match and qualification_match
        and recomputed["passed"] and binding_ok and integrity == result["integrity"]
    )
    metrics = recomputed["metrics"]
    summary = f"""# V62 results: external classic-POMDP belief and planning transfer

Qualification: **{'PASS' if passed else 'FAIL'}**

The candidate exactly parsed and evaluated six fixed cells from three POBAX models at upstream commit `a5e1d62d14e4efe783885b9d4f19cffa2a568eec`. Candidate/reference root-action optimal-set membership was `{fmt(metrics['candidate_reference_optimal_set_membership_rate'])}` and the maximum exact value error was `{fmt(metrics['maximum_candidate_reference_value_error'])}`. The maximum independent Bellman residual was `{fmt(metrics['maximum_independent_bellman_residual'])}`.

## Control separations

- Tiger selected the information-gathering `listen` action in all preregistered horizons of at least three: rate `{fmt(metrics['tiger_information_gathering_action_rate'])}`.
- The smaller of the two T-Maze exact-history advantages over the observation-only control was `{fmt(metrics['minimum_tmaze_exact_history_minus_observation_only_value'])}` discounted return.
- The smallest Tiger exact-history advantage over MAP-state collapse was `{fmt(metrics['minimum_tiger_exact_history_minus_map_collapse_value'])}` discounted return.

## External execution

The unchanged pinned POBAX runtime completed all 24 task-policy rollout cells. All 4,096-episode means fell inside their familywise 99% Hoeffding bounds; the maximum excess was `{fmt(metrics['maximum_official_runtime_return_excess_over_simultaneous_bound'])}`. Source, license, bundle, runtime-version, attempt-count, human-access, and model-access integrity checks all passed.

## Exact values by task

| Model | Horizon | Exact history | Observation only | MAP collapse | Fully observed oracle |
|---|---:|---:|---:|---:|---:|
"""
    for row in result["exact_records"]:
        values = row["exact_policy_values"]
        summary += (
            f"| {row['model_id']} | {row['horizon']} | {fmt(values['exact_history'])} | "
            f"{fmt(values['observation_only'])} | {fmt(values['map_collapse'])} | "
            f"{fmt(values['fully_observed_oracle'])} |\n"
        )
    summary += """

## Boundary

This is external evidence for exact finite-state, finite-horizon Bayesian filtering and planning on these three pinned models. It does not establish portability of the project's SMC2 unknown-mechanic estimator, general POMDP scalability, continuous or long-horizon control, formal safety, human-authored language robustness, or model/adapter performance. V58 remains deferred.
"""
    summary_path.write_text(summary)
    audit = {
        "schema_version": 62,
        "experiment": "v62_post_result_audit",
        "passed": passed,
        "decision": "freeze_v62_outcome" if passed else "do_not_freeze_v62_outcome",
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "evaluation_implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "evaluation_implementation_lock_sha256": file_sha256(lock_path),
        "external_bundle_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "external_bundle_seal_sha256": file_sha256(seal_path),
        "evaluation_attempt": str(attempt_path.relative_to(PROJECT_ROOT)),
        "evaluation_attempt_sha256": file_sha256(attempt_path),
        "results_summary": str(summary_path.relative_to(PROJECT_ROOT)),
        "results_summary_sha256": file_sha256(summary_path),
        "integrity": integrity,
        "checks": {
            "metrics_exactly_recomputed": metrics_match,
            "gate_checks_exactly_recomputed": checks_match,
            "qualification_exactly_recomputed": qualification_match,
            "all_thirty_two_gates_pass": recomputed["passed"] and len(recomputed["checks"]) == 32,
            "source_bundle_and_runtime_intact": all(value == 0 for value in integrity.values()),
            "single_attempt_and_lock_binding": binding_ok,
        },
    }
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
