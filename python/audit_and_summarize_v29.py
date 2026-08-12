"""Audit V29 integrity and write its exposed-development report."""

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
    parser.add_argument("--lock", default="configs/v29-posterior-graph-lock.json")
    parser.add_argument("--result", default="outputs/v29-posterior-graph/evaluation/result.json")
    parser.add_argument("--audit", default="outputs/v29-posterior-graph/post-result-audit.json")
    parser.add_argument("--markdown", default="docs/v29-results.md")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.lock).resolve()
    result_path = (PROJECT_ROOT / args.result).resolve()
    lock = json.loads(lock_path.read_text())
    result = json.loads(result_path.read_text())
    attempt = json.loads((PROJECT_ROOT / "outputs/v29-posterior-graph/evaluation-attempt.json").read_text())
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
    reference = lock["source_v28_reference"]
    if all(reproduced.values()):
        expected_decision = "authorize_query_exact_graph_repair_no_fresh_benchmark_yet"
    elif evaluation["exact_support_graph"] > reference["evaluation_support_exact_graph"]:
        expected_decision = "posterior_graph_decoding_improves_exact_support_continue_no_lora"
    else:
        expected_decision = "posterior_graph_decoding_insufficient_revisit_language_scores_no_lora"
    source_checks = {
        key: file_sha256(PROJECT_ROOT / lock["source"][key]) == lock["source"][f"{key}_sha256"]
        for key in (
            "sourceV28Lock", "sourceV28Result", "sourceV28PostAudit",
            "sourceV28Predictions", "sourceV28Diagnostics",
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
            PROJECT_ROOT / lock["source"]["sourceV28Predictions"]
        ) == query_lines(predictions_path),
        "gates_reproduced": result["checks"] == reproduced,
        "pass_reproduced": result["passed"] == all(reproduced.values()),
        "decision_reproduced": result["decision"] == expected_decision,
        "one_shot_zero_fit": (
            result["data_access"]["new_model_forward_passes"] == 0
            and result["data_access"]["posterior_graph_evaluations"] == 1
            and result["data_access"]["head_fits"] == 0
            and result["data_access"]["threshold_fits"] == 0
            and result["data_access"]["hyperparameter_selections"] == 0
            and result["data_access"]["adapter_training_runs"] == 0
        ),
        "no_lora_or_fresh_benchmark": not result["lora_authorized"] and not result["fresh_benchmark_constructed"],
    }
    audit = {
        "schema_version": 29,
        "experiment": "v29_post_result_integrity_audit",
        "passed": all(checks.values()),
        "decision": "accept_v29_exposed_development_result" if all(checks.values()) else "quarantine_v29_result",
        "checks": checks, "source_checks": source_checks,
        "reproduced_gates": reproduced,
    }
    (PROJECT_ROOT / args.audit).write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    support = integration["frozen_support_oracle_query"]
    lines = [
        "# V29 results: posterior-marginal support graph decoding", "",
        f"Decision: `{result['decision']}`.", "",
        "V29 changed only the Bayes decision rule for support graphs. It integrated over the",
        "shared program and all other support graphs, emitted one graph per scene, made zero",
        "model calls, and reused query predictions byte-for-byte.", "",
        "| Metric | V28 | V29 |", "|---|---:|---:|",
        f"| Evaluation exact support graph | {reference['evaluation_support_exact_graph']:.3f} | {evaluation['exact_support_graph']:.3f} |",
        f"| Frozen support / oracle query exact | {reference['frozen_support_oracle_query_exact']:.3f} | {support['transition_set_exact_match']:.3f} |",
        f"| Frozen / frozen exact | {reference['frozen_frozen_exact']:.3f} | {integration['frozen_support_frozen_query']['transition_set_exact_match']:.3f} |",
        f"| Target retention | {reference['target_retention']:.3f} | {support['target_retention_rate']:.3f} |",
        f"| Empty version space | {reference['empty_version_space']:.3f} | {support['empty_version_space_rate']:.3f} |", "",
        f"Mean selected evaluation graph posterior: {result['posterior_graph']['evaluation_mean_selected_graph_posterior']:.3f}.",
        f"Post-result integrity audit: `{'pass' if audit['passed'] else 'fail'}`.", "",
    ]
    (PROJECT_ROOT / args.markdown).write_text("\n".join(lines))
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
