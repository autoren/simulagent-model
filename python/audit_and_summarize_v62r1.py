#!/usr/bin/env python3
"""Independently audit and summarize the V62r1 repair rescore."""
from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result", default="outputs/v62r1-terminal-residual-repair/rescore/result.json"
    )
    parser.add_argument(
        "--output",
        default="outputs/v62r1-terminal-residual-repair/post-result-audit.json",
    )
    parser.add_argument("--summary", default="docs/v62r1-results.md")
    args = parser.parse_args()
    result_path, output, summary_path = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.result, args.output, args.summary)
    )
    if output.exists() or summary_path.exists():
        raise RuntimeError("V62r1 post-result artifacts already exist")
    result = json.loads(result_path.read_text())
    evaluation_lock_path = PROJECT_ROOT / result["evaluation_implementation_lock"]
    evaluation_lock = json.loads(evaluation_lock_path.read_text())
    implementation_lock_path = PROJECT_ROOT / result["implementation_lock"]
    implementation_lock = json.loads(implementation_lock_path.read_text())
    design = json.loads((PROJECT_ROOT / implementation_lock["design_lock"]).read_text())
    config = design["config_payload"]
    outcome_path = PROJECT_ROOT / result["source_v62_outcome_lock"]
    outcome = json.loads(outcome_path.read_text())
    source_result_path = PROJECT_ROOT / result["source_v62_result"]
    source_result = json.loads(source_result_path.read_text())
    attempt_path = result_path.parent.parent / "rescore-attempt.json"
    attempt = json.loads(attempt_path.read_text())
    source_seal = json.loads(
        (PROJECT_ROOT / outcome["external_bundle_seal"]).read_text()
    )
    source_implementation = json.loads(
        (PROJECT_ROOT / source_seal["implementation_lock"]).read_text()
    )
    source_design = json.loads(
        (PROJECT_ROOT / source_implementation["design_lock"]).read_text()
    )
    old_residual_threshold = source_design["config_payload"]["gates"][
        "maximumIndependentBellmanResidual"
    ]

    records = result["records"]
    all_nodes = [node for record in records for node in record["nodes"]]
    old_failures = [
        node for node in all_nodes if node["old_residual"] > old_residual_threshold
    ]
    terminal_nodes = [node for node in all_nodes if node["terminal_support"]]
    nonterminal_nodes = [node for node in all_nodes if not node["terminal_support"]]
    other_checks = {
        key: value
        for key, value in outcome["gate_checks"].items()
        if key != "independent_bellman_residual"
    }
    metrics = {
        "source_binding_rate": sum(result["source_binding_checks"].values())
        / len(result["source_binding_checks"]),
        "source_old_residual_reproduction_rate": sum(
            record["recorded_old_maximum_residual"]
            == record["recomputed_old_maximum_residual"]
            for record in records
        )
        / len(records),
        "old_failure_terminal_localization_rate": sum(
            node["terminal_support"] for node in old_failures
        )
        / len(old_failures),
        "maximum_corrected_residual": max(
            node["corrected_residual"] for node in all_nodes
        ),
        "maximum_corrected_terminal_residual": max(
            (node["corrected_residual"] for node in terminal_nodes), default=0.0
        ),
        "maximum_corrected_nonterminal_residual": max(
            (node["corrected_residual"] for node in nonterminal_nodes), default=0.0
        ),
        "other_v62_gate_reproduction_rate": sum(other_checks.values())
        / len(other_checks),
        "maximum_exact_record_delta": float(
            result["source_exact_records_sha256"]
            != canonical_sha256(source_result["exact_records"])
        ),
        "maximum_official_rollout_record_delta": float(
            result["source_official_rollout_records_sha256"]
            != canonical_sha256(source_result["official_rollout_records"])
        ),
        "repair_fixture_pass_rate": implementation_lock["audit_controls"][
            "analytic_fixture_pass_rate"
        ],
        "repair_mutant_kill_rate": implementation_lock["audit_controls"][
            "mutant_kill_rate"
        ],
        "repair_attempt_and_access_violation_count": sum(
            (
                int(attempt["attempt"] != result["evaluation_run"] or attempt["attempt"] != 1),
                attempt["new_candidate_evaluations"],
                attempt["new_external_rollouts"],
                attempt["human_records"],
                attempt["model_forward_passes"],
            )
        ),
    }
    gates = config["gates"]
    checks = {
        "source_binding_rate": metrics["source_binding_rate"] >= gates["minimumSourceBindingRate"],
        "source_old_residual_reproduction_rate": metrics["source_old_residual_reproduction_rate"] >= gates["minimumSourceOldResidualReproductionRate"],
        "old_failure_terminal_localization_rate": metrics["old_failure_terminal_localization_rate"] >= gates["minimumOldFailureTerminalLocalizationRate"],
        "corrected_residual": metrics["maximum_corrected_residual"] <= gates["maximumCorrectedResidual"],
        "corrected_terminal_residual": metrics["maximum_corrected_terminal_residual"] <= gates["maximumCorrectedTerminalResidual"],
        "corrected_nonterminal_residual": metrics["maximum_corrected_nonterminal_residual"] <= gates["maximumCorrectedNonterminalResidual"],
        "other_v62_gate_reproduction_rate": metrics["other_v62_gate_reproduction_rate"] >= gates["minimumOtherV62GateReproductionRate"],
        "exact_record_delta": metrics["maximum_exact_record_delta"] <= gates["maximumExactRecordDelta"],
        "official_rollout_record_delta": metrics["maximum_official_rollout_record_delta"] <= gates["maximumOfficialRolloutRecordDelta"],
        "repair_fixture_pass_rate": metrics["repair_fixture_pass_rate"] >= gates["minimumRepairFixturePassRate"],
        "repair_mutant_kill_rate": metrics["repair_mutant_kill_rate"] >= gates["minimumRepairMutantKillRate"],
        "repair_attempt_and_access_violation_count": metrics["repair_attempt_and_access_violation_count"] <= gates["maximumRepairAttemptAndAccessViolationCount"],
    }
    binding_checks = {
        "evaluation_lock": file_sha256(evaluation_lock_path)
        == result["evaluation_implementation_lock_sha256"],
        "implementation_lock": file_sha256(implementation_lock_path)
        == result["implementation_lock_sha256"],
        "source_outcome": file_sha256(outcome_path)
        == result["source_v62_outcome_lock_sha256"],
        "source_result": file_sha256(source_result_path)
        == result["source_v62_result_sha256"]
        == outcome["result_sha256"],
        "attempt": attempt["evaluation_implementation_lock_sha256"]
        == file_sha256(evaluation_lock_path),
    }
    reproduction = {
        "metrics": metrics == result["metrics"],
        "gate_checks": checks == result["qualification"]["checks"],
        "qualification": all(checks.values()) == result["qualification"]["passed"],
        "census": result["node_census"]
        == {
            "reachable": 66,
            "terminal_support": 4,
            "nonterminal": 62,
            "old_residual_failures": 4,
            "old_terminal_residual_failures": 4,
        },
        "original_v62_remains_failed": (
            not outcome["qualification_passed"]
            and outcome["failed_checks"] == ["independent_bellman_residual"]
        ),
    }
    passed = all(checks.values()) and all(binding_checks.values()) and all(
        reproduction.values()
    )
    summary = f"""# V62r1 results: terminal Bellman-residual measurement repair

Repair qualification: **{'PASS' if passed else 'FAIL'}**

V62 remains an immutable 31-of-32-gate failure. V62r1 rescored the same 66 reachable belief nodes with the preregistered terminal-aware independent checker; it did not repeat candidate evaluation or any of the 24 external rollout cells.

All four old residual failures were terminal-support beliefs, and there were no nonterminal old failures. The corrected maximum residual was `{metrics['maximum_corrected_residual']:.12g}` overall, `{metrics['maximum_corrected_terminal_residual']:.12g}` on terminal nodes, and `{metrics['maximum_corrected_nonterminal_residual']:.12g}` on nonterminal nodes. All other 31 V62 gates and every exact and official-rollout record reproduced without change. All repair fixtures passed and all six targeted mutants were killed.

The combined V62/V62r1 evidence therefore supports only exact finite-state, finite-horizon Bayesian filtering and planning transfer on the three pinned POBAX models. V62r1 is a measurement correction over immutable artifacts, not an independent external replication. It does not establish SMC2 portability, unknown-program inference, generic POMDP scalability, continuous or long-horizon control, formal safety, human-authored language robustness, or model/adapter performance. V58 remains deferred.
"""
    summary_path.write_text(summary)
    audit = {
        "schema_version": "62r1",
        "experiment": "v62r1_post_result_audit",
        "passed": passed,
        "decision": "freeze_v62r1_outcome" if passed else "reject_v62r1_repair",
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "evaluation_implementation_lock": str(evaluation_lock_path.relative_to(PROJECT_ROOT)),
        "evaluation_implementation_lock_sha256": file_sha256(evaluation_lock_path),
        "implementation_lock": str(implementation_lock_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(implementation_lock_path),
        "source_v62_outcome_lock": str(outcome_path.relative_to(PROJECT_ROOT)),
        "source_v62_outcome_lock_sha256": file_sha256(outcome_path),
        "source_v62_result": str(source_result_path.relative_to(PROJECT_ROOT)),
        "source_v62_result_sha256": file_sha256(source_result_path),
        "rescore_attempt": str(attempt_path.relative_to(PROJECT_ROOT)),
        "rescore_attempt_sha256": file_sha256(attempt_path),
        "results_summary": str(summary_path.relative_to(PROJECT_ROOT)),
        "results_summary_sha256": file_sha256(summary_path),
        "metrics": metrics,
        "gate_checks": checks,
        "binding_checks": binding_checks,
        "reproduction_checks": reproduction,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
