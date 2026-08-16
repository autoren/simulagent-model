#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math

from evaluate_v56_verification import aggregate, verify_bundle_hashes
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        default="outputs/v56-symbolic-probabilistic-policy-verification/evaluation/result.json",
    )
    parser.add_argument(
        "--audit",
        default="outputs/v56-symbolic-probabilistic-policy-verification/post-result-audit.json",
    )
    parser.add_argument("--summary", default="docs/v56-results.md")
    args = parser.parse_args()
    result_path, audit_path, summary_path = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.result, args.audit, args.summary)
    )
    if audit_path.exists() or summary_path.exists():
        raise FileExistsError("V56 post-result artifacts already exist")
    result = json.loads(result_path.read_text())
    lock_path = PROJECT_ROOT / result["evaluation_implementation_lock"]
    lock = json.loads(lock_path.read_text())
    seal_path = PROJECT_ROOT / result["verification_bundle_seal"]
    seal = json.loads(seal_path.read_text())
    manifest_path = PROJECT_ROOT / result["manifest"]
    manifest = json.loads(manifest_path.read_text())
    bundle = PROJECT_ROOT / seal["bundle"]
    attempt_path = result_path.parent.parent / "evaluation-attempt.json"
    errors: list[str] = []

    sealed = (
        result["evaluation_run"] == 1
        and result["evaluation_implementation_lock_sha256"]
        == file_sha256(lock_path)
        and result["verification_bundle_seal_sha256"] == file_sha256(seal_path)
        and result["manifest_sha256"] == file_sha256(manifest_path)
        and seal["manifest_sha256"] == file_sha256(manifest_path)
        and lock["verification_bundle_seal_sha256"] == file_sha256(seal_path)
        and verify_bundle_hashes(bundle, manifest) == 0
        and attempt_path.exists()
        and json.loads(attempt_path.read_text())["evaluation_run"] == 1
    )
    if not sealed:
        errors.append("V56 result is not bound to the one-shot sealed inputs")

    records = result["records"]
    expected_order = [
        (row["cohort"], row["id"], row["record"])
        for row in manifest["policies"]
    ]
    observed_order = [
        (row["cohort"], row["id"], row["record"]) for row in records
    ]
    records_ok = (
        len(records) == 48
        and observed_order == expected_order
        and len({(row["cohort"], row["id"]) for row in records}) == 48
        and all(row["completed"] and row["storm_completed"] for row in records)
    )
    if not records_ok:
        errors.append("V56 result records are incomplete, failed, duplicated, or reordered")

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
    controls = {
        "mutation_kill_rate": implementation_audit["mutation_controls"][
            "kill_rate"
        ],
        "analytic_fixture_pass_rate": sum(
            row["passed"] for row in implementation_audit["analytic_storm_fixtures"]
        ) / len(implementation_audit["analytic_storm_fixtures"]),
    }
    reproduced = aggregate(records, config, result["metrics"]["integrity"], controls)
    metrics_ok = reproduced["metrics"] == result["metrics"]
    checks_ok = reproduced["checks"] == result["qualification"]["checks"]
    qualification_ok = reproduced["passed"] == result["qualification"]["passed"]
    if not metrics_ok:
        errors.append("V56 aggregate metrics do not reproduce")
    if not checks_ok or not qualification_ok:
        errors.append("V56 qualification does not reproduce")

    algebra_ok = True
    for row in records:
        storm = row["storm"]
        direct = row["direct_executor"]
        comparisons = (
            (
                row["termination_probability_error"],
                abs(storm["termination_probability"] - 1.0),
            ),
            (
                row["success_probability_error"],
                abs(storm["success_probability"] - direct["success_probability"]),
            ),
            (
                row["expected_return_error_against_frozen_value"],
                abs(storm["expected_return"] - row["frozen_root_value"]),
            ),
            (
                row[
                    "expected_return_error_against_independent_policy_evaluator"
                ],
                abs(storm["expected_return"] - row["independent_policy_value"]),
            ),
        )
        algebra_ok = algebra_ok and all(
            math.isclose(left, right, rel_tol=0.0, abs_tol=1e-15)
            for left, right in comparisons
        )
        symbolic = row["symbolic"]
        algebra_ok = algebra_ok and (
            symbolic["invariant_passes"] <= symbolic["invariant_checks"]
            and symbolic["support_passes"] <= symbolic["support_checks"]
            and symbolic["totality_passes"] <= symbolic["totality_checks"]
        )
    if not algebra_ok:
        errors.append("V56 record-level reference or symbolic algebra is inconsistent")

    integrity = result["metrics"]["integrity"]
    integrity_ok = all(value == 0 for value in integrity.values())
    if not integrity_ok:
        errors.append("V56 sealed input, truth, tool, source, or attempt integrity failed")

    audit = {
        "schema_version": 56,
        "experiment": "v56_post_result_audit",
        "passed": not errors,
        "decision": "accept_v56_result" if not errors else "invalidate_v56_result",
        "errors": errors,
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "evaluation_attempt": str(attempt_path.relative_to(PROJECT_ROOT)),
        "evaluation_attempt_sha256": file_sha256(attempt_path),
        "evaluation_implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "evaluation_implementation_lock_sha256": file_sha256(lock_path),
        "verification_bundle_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "verification_bundle_seal_sha256": file_sha256(seal_path),
        "qualification": result["qualification"],
        "checks": {
            "one_shot_sealed_bindings": sealed,
            "record_count_order_ids_and_completion": records_ok,
            "metric_aggregation_reproduced": metrics_ok,
            "qualification_reproduced": checks_ok and qualification_ok,
            "record_level_reference_and_symbolic_algebra": algebra_ok,
            "truth_source_bundle_tool_and_attempt_integrity": integrity_ok,
        },
        "data_access": {
            "additional_v56_candidate_verification_runs": 0,
            "storm_invocations": 0,
            "z3_solver_invocations": 0,
            "additional_v55_or_v55r1_evaluation_runs": 0,
            "truth_field_access_count": 0,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    source = result["metrics"]["source_binding"]
    symbolic = result["metrics"]["symbolic"]
    probabilistic = result["metrics"]["probabilistic"]
    if result["qualification"]["passed"]:
        decision = (
            "V56 qualifies bounded posterior-expected verification of all 48 "
            "frozen three-action policies. It authorizes only preregistration "
            "of the next definition-transfer and human-authored-language tracks."
        )
    else:
        decision = (
            "V56 does not qualify the bounded formal-verification layer; the "
            "failed gate must be localized in a new preregistered revision."
        )
    summary_path.write_text(
        "# V56 results: symbolic and probabilistic policy verification\n\n"
        f"Decision: {decision}\n\n"
        "## Sealed results\n\n"
        f"- All 25 noncompensatory gates passed: `{result['qualification']['passed']}`.\n"
        f"- Policies completed: `{int(source['completed_policy_fraction'] * 48)}/48` "
        f"(`{json.dumps(source['policy_count_by_cohort'], sort_keys=True)}`).\n"
        f"- Root-action reproduction rate: `{source['reconstructed_root_action_match_rate']}`.\n"
        f"- Maximum root-value reconstruction error: `{source['maximum_reconstructed_root_value_error']}`.\n"
        f"- Reachable state invariant proofs: `{symbolic['reachable_state_invariant_checks']}` "
        f"at rate `{symbolic['reachable_state_invariant_proof_rate']}`.\n"
        f"- Reachable transition-support proofs: `{symbolic['reachable_transition_support_checks']}` "
        f"at rate `{symbolic['reachable_transition_support_equivalence_proof_rate']}`.\n"
        f"- Observation-totality checks: `{symbolic['policy_observation_totality_checks']}` "
        f"at rate `{symbolic['policy_observation_totality_rate']}`.\n"
        f"- Z3 unknowns / nonterminal deadlocks: `{symbolic['z3_unknown_count']}` / "
        f"`{symbolic['nonterminal_deadlock_count']}`.\n"
        f"- Storm completion rate: `{probabilistic['storm_completed_model_fraction']}`.\n"
        f"- Maximum termination-probability error: `{probabilistic['maximum_termination_probability_error']}`.\n"
        f"- Maximum success-probability error: `{probabilistic['maximum_success_probability_error_against_direct_executor']}`.\n"
        f"- Maximum return error vs frozen value: `{probabilistic['maximum_expected_return_error_against_frozen_value']}`.\n"
        f"- Maximum return error vs independent evaluator: `{probabilistic['maximum_expected_return_error_against_independent_policy_evaluator']}`.\n"
        f"- Integrity violations: `{sum(integrity.values())}`.\n\n"
        "## Claim boundary\n\n"
        "This verifies only bounded, three-action execution of the frozen V55/V55r1 "
        "policies under their posterior mixture. It does not establish worst-case "
        "safety, parameter-uniform guarantees, unbounded or long-horizon behavior, "
        "planner optimality outside the sealed tasks, or open-language grounding.\n"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
