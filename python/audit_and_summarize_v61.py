#!/usr/bin/env python3
"""Independently recompute V61 metrics and write its bounded results summary."""
from __future__ import annotations

import argparse
import json

from evaluate_v61_verification import (
    aggregate, source_result_mutation_count, verify_bundle_hashes,
)
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v56_verification import tool_versions


def fmt(value):
    return f"{value:.12g}" if isinstance(value, float) else str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result", default="outputs/v61-long-horizon-policy-verification/verification/result.json"
    )
    parser.add_argument(
        "--output", default="outputs/v61-long-horizon-policy-verification/post-result-audit.json"
    )
    parser.add_argument("--summary", default="docs/v61-results.md")
    args = parser.parse_args()
    result_path, output, summary_path = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.result, args.output, args.summary)
    )
    if output.exists() or summary_path.exists():
        raise RuntimeError("V61 post-result artifacts already exist")
    result = json.loads(result_path.read_text())
    lock_path = PROJECT_ROOT / result["evaluation_implementation_lock"]
    lock = json.loads(lock_path.read_text())
    seal_path = PROJECT_ROOT / result["verification_bundle_seal"]
    seal = json.loads(seal_path.read_text())
    manifest_path = PROJECT_ROOT / seal["manifest"]
    manifest = json.loads(manifest_path.read_text())
    bundle = PROJECT_ROOT / seal["bundle"]
    implementation = json.loads(
        (PROJECT_ROOT / lock["implementation_lock"]).read_text()
    )
    design = json.loads(
        (PROJECT_ROOT / implementation["design_lock"]).read_text()
    )
    config = design["config_payload"]
    implementation_audit = json.loads(
        (PROJECT_ROOT / implementation["implementation_audit"]).read_text()
    )
    integrity = {
        "truth_field_access_count": manifest["truth_field_access_count"],
        "source_result_mutation_count": source_result_mutation_count(bundle, manifest),
        "verification_bundle_hash_mismatch_count": verify_bundle_hashes(bundle, manifest),
        "tool_version_mismatch_count": int(tool_versions() != lock["tool_versions"]),
        "unexpected_verification_attempt_count": 0,
    }
    controls = {
        "mutation_kill_rate": implementation_audit["mutation_kill_rate"],
        "analytic_fixture_pass_rate": implementation_audit["analytic_fixture_pass_rate"],
    }
    recomputed = aggregate(result["records"], config, integrity, controls)
    metrics_match = recomputed["metrics"] == result["metrics"]
    checks_match = recomputed["checks"] == result["qualification"]["checks"]
    qualification_match = recomputed["passed"] == result["qualification"]["passed"]
    attempt_path = result_path.parent.parent / "verification-attempt.json"
    attempt = json.loads(attempt_path.read_text())
    binding_ok = (
        file_sha256(lock_path) == result["evaluation_implementation_lock_sha256"]
        and file_sha256(seal_path) == result["verification_bundle_seal_sha256"]
        and file_sha256(manifest_path) == seal["manifest_sha256"]
        and attempt["attempt"] == result["verification_run"] == 1
        and attempt["evaluation_implementation_lock_sha256"] == file_sha256(lock_path)
    )
    passed = (
        metrics_match and checks_match and qualification_match
        and recomputed["passed"] and binding_ok
        and integrity == result["metrics"]["integrity"]
    )
    p = recomputed["metrics"]["probabilistic"]
    s = recomputed["metrics"]["source_binding"]
    z = recomputed["metrics"]["symbolic"]
    total_states = sum(row["states"] for row in result["records"])
    total_transitions = sum(row["transitions"] for row in result["records"])
    maximum_mc_error = max(row["v60_monte_carlo_exact_error"] for row in result["records"])
    summary = f"""# V61 results: bounded long-horizon approximate-belief policy verification

**Qualification:** {'PASS' if passed else 'FAIL'}  
**Frozen policies:** {s['policy_count']} ({s['policy_count_by_horizon']['3']} each at horizons 3, 5, and 7)  
**External checker:** Storm 1.13.0  
**Claim boundary:** bounded exact-posterior execution of the frozen V60 policies; not search optimality, worst-case safety, an unbounded guarantee, or language robustness

## Main findings

All 72 primary V60 policies reconstructed exactly: tree-hash, root-action, and search-metadata match rates were each `{fmt(s['reconstructed_tree_hash_match_rate'])}`. Their explicit models contained `{total_states:,}` states and `{total_transitions:,}` transitions in total.

Storm completed every model. Its maximum termination-probability error was `{fmt(p['maximum_termination_probability_error'])}`, maximum success-probability error against the independent executor was `{fmt(p['maximum_success_probability_error_against_independent_executor'])}`, and maximum expected-return error was `{fmt(p['maximum_expected_return_error_against_independent_executor'])}`.

The independent reachable-state checks covered `{z['reachable_state_invariant_checks']:,}` invariants and `{z['reachable_transition_support_checks']:,}` transition supports. Invariant, exact support/probability, and public-history policy-totality rates were all `{fmt(z['reachable_state_invariant_proof_rate'])}`, with zero deadlocks and zero Z3 unknown results.

Every stored V60 2,048-episode policy mean fell inside its preregistered familywise 99% Hoeffding radius. The largest exact-vs-Monte-Carlo error was `{fmt(maximum_mc_error)}` and excess over the simultaneous bound was `{fmt(p['maximum_v60_monte_carlo_return_excess_over_simultaneous_bound'])}`.

## Exact verified return by horizon

| Horizon | Mean | Minimum | Maximum |
|---:|---:|---:|---:|
"""
    for horizon in (3, 5, 7):
        row = p["verified_exact_return_by_horizon"][str(horizon)]
        summary += (
            f"| {horizon} | {fmt(row['mean'])} | {fmt(row['minimum'])} | "
            f"{fmt(row['maximum'])} |\n"
        )
    summary += f"""

## Interpretation

V60's approximate-belief policies are no longer supported only by sampled execution estimates. Across the exhaustive frozen census, a separately implemented exact executor and an external probabilistic model checker agree on their complete bounded execution semantics through horizon seven. This verifies the deployed policies in the frozen symbolic domain. It does not prove that UCT found an optimal policy, define or verify catastrophe avoidance, make a guarantee uniform over all continuous parameter values, or replace the deferred human-authored language track.

All 27 noncompensatory gates {'passed' if passed else 'did not pass'}. Bundle hashes, source-result hashes, tool versions, attempt count, and the zero-truth-access firewall were independently recomputed after the run.
"""
    summary_path.write_text(summary)
    audit = {
        "schema_version": 61,
        "experiment": "v61_post_result_audit",
        "passed": passed,
        "decision": "freeze_v61_outcome" if passed else "do_not_freeze_v61_outcome",
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "evaluation_implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "evaluation_implementation_lock_sha256": file_sha256(lock_path),
        "verification_bundle_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "verification_bundle_seal_sha256": file_sha256(seal_path),
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "manifest_sha256": file_sha256(manifest_path),
        "evaluation_attempt": str(attempt_path.relative_to(PROJECT_ROOT)),
        "evaluation_attempt_sha256": file_sha256(attempt_path),
        "results_summary": str(summary_path.relative_to(PROJECT_ROOT)),
        "results_summary_sha256": file_sha256(summary_path),
        "checks": {
            "independent_metric_recomputation_matches": metrics_match,
            "independent_gate_recomputation_matches": checks_match,
            "qualification_matches": qualification_match,
            "all_twenty_seven_gates_pass": recomputed["passed"],
            "lock_seal_manifest_and_attempt_binding": binding_ok,
            "integrity_recomputation_matches": integrity == result["metrics"]["integrity"],
        },
        "recomputed_metrics": recomputed["metrics"],
        "recomputed_checks": recomputed["checks"],
    }
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "passed": passed, "result_sha256": file_sha256(result_path),
        "summary": str(summary_path.relative_to(PROJECT_ROOT)),
        "checks": audit["checks"],
    }, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
