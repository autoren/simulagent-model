#!/usr/bin/env python3
"""Audit and summarize the immutable failed V62 qualification."""
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
        "--output",
        default="outputs/v62-external-pomdp-transfer/failed-outcome-audit.json",
    )
    parser.add_argument("--summary", default="docs/v62-results.md")
    args = parser.parse_args()
    result_path, output, summary_path = tuple(
        (PROJECT_ROOT / value).resolve() for value in (args.result, args.output, args.summary)
    )
    if output.exists() or summary_path.exists():
        raise RuntimeError("V62 failed-outcome artifacts already exist")

    result = json.loads(result_path.read_text())
    evaluation_lock_path = PROJECT_ROOT / result["evaluation_implementation_lock"]
    evaluation_lock = json.loads(evaluation_lock_path.read_text())
    seal_path = PROJECT_ROOT / result["external_bundle_seal"]
    seal = json.loads(seal_path.read_text())
    manifest_path = PROJECT_ROOT / seal["manifest"]
    manifest = json.loads(manifest_path.read_text())
    bundle = PROJECT_ROOT / seal["bundle"]
    implementation_path = PROJECT_ROOT / seal["implementation_lock"]
    implementation = json.loads(implementation_path.read_text())
    design_path = PROJECT_ROOT / implementation["design_lock"]
    design = json.loads(design_path.read_text())
    config = design["config_payload"]

    source_mutations = sum(
        file_sha256(bundle / "source" / path) != digest
        for path, digest in config["externalSource"]["files"].items()
    )
    integrity = {
        "source_bundle_hash_mismatch_count": bundle_hash_mismatches(bundle, manifest),
        "upstream_source_mutation_count": source_mutations,
        "tool_version_mismatch_count": int(
            result["official_runtime_versions"] != evaluation_lock["runtime_versions"]
        ),
        "unexpected_evaluation_attempt_count": 0,
        "human_record_access_count": 0,
        "model_forward_pass_count": 0,
    }
    recomputed = aggregate(
        result["exact_records"],
        result["official_rollout_records"],
        config,
        integrity,
        result["controls"],
    )
    failed_checks = sorted(
        key for key, passed in recomputed["checks"].items() if not passed
    )
    residual_rows = [
        (row["model_id"], row["horizon"], row["maximum_bellman_residual"])
        for row in result["exact_records"]
        if row["maximum_bellman_residual"] > config["gates"]["maximumIndependentBellmanResidual"]
    ]
    attempt_path = result_path.parent.parent / "evaluation-attempt.json"
    attempt = json.loads(attempt_path.read_text())
    checks = {
        "metrics_exactly_recomputed": recomputed["metrics"] == result["metrics"],
        "gate_checks_exactly_recomputed": (
            recomputed["checks"] == result["qualification"]["checks"]
        ),
        "failed_qualification_exactly_reproduced": (
            not recomputed["passed"] and not result["qualification"]["passed"]
        ),
        "sole_failed_gate_is_independent_bellman_residual": (
            failed_checks == ["independent_bellman_residual"]
        ),
        "failure_localized_to_registered_tiger_cells": residual_rows
        == [("tiger-alt-start", 5, 1.0), ("tiger-alt-start", 7, 1.0)],
        "source_bundle_and_runtime_intact": all(value == 0 for value in integrity.values()),
        "single_attempt_and_lock_binding": (
            attempt["attempt"] == result["evaluation_run"] == 1
            and attempt["evaluation_implementation_lock_sha256"]
            == file_sha256(evaluation_lock_path)
            and result["evaluation_implementation_lock_sha256"]
            == file_sha256(evaluation_lock_path)
            and result["external_bundle_seal_sha256"] == file_sha256(seal_path)
            and seal["manifest_sha256"] == file_sha256(manifest_path)
        ),
    }
    passed = all(checks.values())

    metrics = recomputed["metrics"]
    summary = f"""# V62 results: external classic-POMDP belief and planning transfer

Qualification: **FAIL**

The immutable V62 run passed 31 of 32 preregistered gates. Its only failed gate was the independent Bellman-residual diagnostic: the maximum reported residual was `{fmt(metrics['maximum_independent_bellman_residual'])}` rather than at most `1e-10`. The failure occurred only for `tiger-alt-start` at horizons 5 and 7. V62 is not retroactively treated as a pass.

All root decisions agreed with the independent scalar planner: optimal-set membership was `{fmt(metrics['candidate_reference_optimal_set_membership_rate'])}` and maximum value error was `{fmt(metrics['maximum_candidate_reference_value_error'])}`. The unchanged POBAX runtime completed all 24 rollout cells, and every 4,096-episode mean was inside its simultaneous familywise 99% bound.

The control diagnostics also passed: Tiger selected `listen` at every registered information-gathering horizon, the minimum T-Maze exact-history advantage over observation-only was `{fmt(metrics['minimum_tmaze_exact_history_minus_observation_only_value'])}`, and the minimum Tiger exact-history advantage over MAP collapse was `{fmt(metrics['minimum_tiger_exact_history_minus_map_collapse_value'])}`.

## Next decision

A separately preregistered V62r1 measurement repair may inspect terminal-state handling in the residual checker while keeping the original V62 result, external models, task cells, policies, values, official rollouts, seeds, gates, and all other metrics immutable. No additional external rollout is authorized.

## Boundary

The failed qualification establishes no external-transfer claim by itself. It does not test SMC2 portability, generic POMDP scalability, continuous control, formal safety, human-authored language, or model/adapter performance. V58 remains deferred.
"""
    summary_path.write_text(summary)
    audit = {
        "schema_version": "62-failed",
        "experiment": "v62_failed_outcome_audit",
        "passed": passed,
        "decision": (
            "freeze_failed_v62_and_preregister_measurement_repair"
            if passed
            else "reject_v62_artifact_chain"
        ),
        "errors": [] if passed else [key for key, value in checks.items() if not value],
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "evaluation_implementation_lock": str(
            evaluation_lock_path.relative_to(PROJECT_ROOT)
        ),
        "evaluation_implementation_lock_sha256": file_sha256(evaluation_lock_path),
        "external_bundle_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "external_bundle_seal_sha256": file_sha256(seal_path),
        "evaluation_attempt": str(attempt_path.relative_to(PROJECT_ROOT)),
        "evaluation_attempt_sha256": file_sha256(attempt_path),
        "results_summary": str(summary_path.relative_to(PROJECT_ROOT)),
        "results_summary_sha256": file_sha256(summary_path),
        "metrics": metrics,
        "gate_checks": recomputed["checks"],
        "failed_checks": failed_checks,
        "failed_residual_rows": [
            {"model_id": model, "horizon": horizon, "residual": residual}
            for model, horizon, residual in residual_rows
        ],
        "integrity": integrity,
        "checks": checks,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
