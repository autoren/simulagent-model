"""Audit V28 integrity and write its exposed-development report."""

from __future__ import annotations

import argparse
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def query_lines(path):
    return [
        line for line in path.read_text().splitlines(keepends=True)
        if line.strip() and json.loads(line)["role"] == "query"
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v28-marginal-map-lock.json")
    parser.add_argument("--result", default="outputs/v28-marginal-map/evaluation/result.json")
    parser.add_argument("--audit", default="outputs/v28-marginal-map/post-result-audit.json")
    parser.add_argument("--markdown", default="docs/v28-results.md")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.lock).resolve()
    result_path = (PROJECT_ROOT / args.result).resolve()
    lock = json.loads(lock_path.read_text())
    result = json.loads(result_path.read_text())
    attempt_path = PROJECT_ROOT / "outputs/v28-marginal-map/evaluation-attempt.json"
    attempt = json.loads(attempt_path.read_text())
    predictions_path = PROJECT_ROOT / result["grounding_predictions"]
    diagnostics_path = PROJECT_ROOT / result["episode_diagnostics"]
    gates = lock["gates"]
    evaluation = result["grounding"]["by_split"]["grounding_evaluation"]
    integration = result["integration"]
    reproduced = {
        "oracle_oracle_exact": integration["oracle_support_oracle_query"]["transition_set_exact_match"] >= gates["minimumOracleOracleExact"],
        "evaluation_support_exact_graph": evaluation["exact_support_graph"] >= gates["minimumEvaluationSupportExactGraph"],
        "frozen_support_oracle_query_exact": integration["frozen_support_oracle_query"]["transition_set_exact_match"] >= gates["minimumFrozenSupportOracleQueryExact"],
        "oracle_support_frozen_query_exact": integration["oracle_support_frozen_query"]["transition_set_exact_match"] >= gates["minimumOracleSupportFrozenQueryExact"],
        "frozen_frozen_exact": integration["frozen_support_frozen_query"]["transition_set_exact_match"] >= gates["minimumFrozenFrozenExact"],
        "frozen_support_target_retention": integration["frozen_support_oracle_query"]["target_retention_rate"] >= gates["minimumFrozenSupportTargetRetention"],
        "frozen_support_empty_version_space": integration["frozen_support_oracle_query"]["empty_version_space_rate"] <= gates["maximumFrozenSupportEmptyVersionSpace"],
    }
    reference = lock["source_v27_reference"]
    if all(reproduced.values()):
        expected_decision = "authorize_query_exact_graph_repair_no_fresh_benchmark_yet"
    elif integration["frozen_support_oracle_query"]["transition_set_exact_match"] > reference["frozen_support_oracle_query_exact"]:
        expected_decision = "marginal_program_map_improves_support_continue_query_repair_no_lora"
    else:
        expected_decision = "marginal_program_map_insufficient_revisit_support_identifiability_no_lora"
    source_checks = {
        key: file_sha256(PROJECT_ROOT / lock["source"][key]) == lock["source"][f"{key}_sha256"]
        for key in (
            "sourceV27Lock", "sourceV27Result", "sourceV27PostAudit",
            "sourceV27EdgeMetadata", "sourceV27EdgeScores", "sourceV27Predictions",
            "sourceV27Diagnostics", "sourceV27NativeMatchDiagnostic",
        )
    }
    checks = {
        "lock_matches": result["protocol_lock_sha256"] == file_sha256(lock_path),
        "sources_intact": all(source_checks.values()),
        "attempt_completed_once": (
            attempt["attempt_number"] == 1 and attempt["status"] == "completed"
            and attempt["result_sha256"] == file_sha256(result_path)
        ),
        "predictions_intact": result["grounding_predictions_sha256"] == file_sha256(predictions_path),
        "diagnostics_intact": result["episode_diagnostics_sha256"] == file_sha256(diagnostics_path),
        "query_predictions_byte_equivalent": query_lines(
            PROJECT_ROOT / lock["source"]["sourceV27Predictions"]
        ) == query_lines(predictions_path),
        "gates_reproduced": result["checks"] == reproduced,
        "pass_reproduced": result["passed"] == all(reproduced.values()),
        "decision_reproduced": result["decision"] == expected_decision,
        "one_shot_zero_fit": (
            result["data_access"]["new_model_forward_passes"] == 0
            and result["data_access"]["marginal_map_evaluations"] == 1
            and result["data_access"]["head_fits"] == 0
            and result["data_access"]["threshold_fits"] == 0
            and result["data_access"]["hyperparameter_selections"] == 0
            and result["data_access"]["adapter_training_runs"] == 0
        ),
        "no_lora_or_fresh_benchmark": not result["lora_authorized"] and not result["fresh_benchmark_constructed"],
    }
    audit = {
        "schema_version": 28,
        "experiment": "v28_post_result_integrity_audit",
        "passed": all(checks.values()),
        "decision": "accept_v28_exposed_development_result" if all(checks.values()) else "quarantine_v28_result",
        "checks": checks,
        "source_checks": source_checks,
        "reproduced_gates": reproduced,
    }
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    support = integration["frozen_support_oracle_query"]
    lines = [
        "# V28 results: marginal program MAP", "",
        f"Decision: `{result['decision']}`.", "",
        "V28 reused V27's complete frozen support candidate set and changed only program",
        "selection from joint/Viterbi MAP to marginal MAP. It made zero model calls and fit",
        "no parameter or threshold. Query predictions are byte-identical to V27.", "",
        "## V27 to V28", "",
        "| Metric | V27 | V28 |", "|---|---:|---:|",
        f"| Evaluation exact support graph | {reference['evaluation_support_exact_graph']:.3f} | {evaluation['exact_support_graph']:.3f} |",
        f"| Frozen support / oracle query exact | {reference['frozen_support_oracle_query_exact']:.3f} | {support['transition_set_exact_match']:.3f} |",
        f"| Frozen / frozen exact | {reference['frozen_frozen_exact']:.3f} | {integration['frozen_support_frozen_query']['transition_set_exact_match']:.3f} |",
        f"| Target retention | {reference['target_retention']:.3f} | {support['target_retention_rate']:.3f} |",
        f"| Empty version space | {reference['empty_version_space']:.3f} | {support['empty_version_space_rate']:.3f} |", "",
        "## Marginal diagnostics", "",
        f"Evaluation target-program top-1 rate: {result['marginal_search']['evaluation_target_program_selection_rate']:.3f}.",
        f"Median target-program rank: {result['marginal_search']['evaluation_median_target_program_rank']:.1f}.",
        f"Compatibility-state deduplication: {result['marginal_search']['compatibility_deduplication_rate']:.3f}.",
        f"Post-result integrity audit: `{'pass' if audit['passed'] else 'fail'}`.", "",
    ]
    (PROJECT_ROOT / args.markdown).write_text("\n".join(lines))
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
