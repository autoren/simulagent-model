"""Audit V27 integrity and write its exposed-development report."""

from __future__ import annotations

import argparse
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v27-support-map-lock.json")
    parser.add_argument("--result", default="outputs/v27-support-map/evaluation/result.json")
    parser.add_argument("--audit", default="outputs/v27-support-map/post-result-audit.json")
    parser.add_argument("--markdown", default="docs/v27-results.md")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.lock).resolve()
    result_path = (PROJECT_ROOT / args.result).resolve()
    lock = json.loads(lock_path.read_text())
    result = json.loads(result_path.read_text())
    edge_metadata_path = PROJECT_ROOT / "outputs/v27-support-map/edge-scores/metadata.json"
    edge_metadata = json.loads(edge_metadata_path.read_text())
    edge_attempt = json.loads((PROJECT_ROOT / "outputs/v27-support-map/edge-decoder-attempt.json").read_text())
    map_attempt = json.loads((PROJECT_ROOT / "outputs/v27-support-map/map-evaluation-attempt.json").read_text())
    predictions_path = PROJECT_ROOT / result["grounding_predictions"]
    diagnostics_path = PROJECT_ROOT / result["episode_diagnostics"]
    def query_lines(path):
        return [
            line for line in path.read_text().splitlines(keepends=True)
            if line.strip() and json.loads(line)["role"] == "query"
        ]

    query_source = query_lines(PROJECT_ROOT / lock["source"]["v26_predictions"])
    query_result = query_lines(predictions_path)
    gates = lock["gates"]["development"]
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
    if all(reproduced.values()):
        expected_decision = "authorize_query_exact_graph_repair_no_fresh_benchmark_yet"
    elif integration["frozen_support_oracle_query"]["transition_set_exact_match"] > lock["source_v26_reference"]["frozen_support_oracle_query_exact"]:
        expected_decision = "support_map_improves_execution_continue_match_repair_no_lora"
    else:
        expected_decision = "outcome_constrained_support_map_insufficient_no_lora"
    checks = {
        "lock_matches": result["protocol_lock_sha256"] == file_sha256(lock_path),
        "edge_attempt_completed_once": (
            edge_attempt["attempt_number"] == 1
            and edge_attempt["status"] == "completed"
            and edge_attempt["metadata_sha256"] == file_sha256(edge_metadata_path)
        ),
        "edge_metadata_matches_lock": edge_metadata["protocol_lock_sha256"] == file_sha256(lock_path),
        "map_attempt_completed_once": map_attempt["attempt_number"] == 1 and map_attempt["status"] == "completed" and map_attempt["result_sha256"] == file_sha256(result_path),
        "edge_score_artifact_intact": edge_metadata["score_artifact_sha256"] == file_sha256(PROJECT_ROOT / edge_metadata["score_artifact"]),
        "predictions_intact": result["grounding_predictions_sha256"] == file_sha256(predictions_path),
        "diagnostics_intact": result["episode_diagnostics_sha256"] == file_sha256(diagnostics_path),
        "query_predictions_byte_equivalent": query_source == query_result,
        "gates_reproduced": result["checks"] == reproduced,
        "pass_reproduced": result["passed"] == all(reproduced.values()),
        "decision_reproduced": result["decision"] == expected_decision,
        "one_shot_zero_fit": (
            result["data_access"]["new_model_forward_passes"] == lock["limits"]["newModelForwardPasses"]
            and result["data_access"]["joint_map_evaluations"] == 1
            and result["data_access"]["head_fits"] == 0
            and result["data_access"]["threshold_fits"] == 0
            and result["data_access"]["hyperparameter_selections"] == 0
            and result["data_access"]["adapter_training_runs"] == 0
        ),
        "no_lora_or_fresh_benchmark": not result["lora_authorized"] and not result["fresh_benchmark_constructed"],
    }
    audit = {
        "schema_version": 27, "experiment": "v27_post_result_integrity_audit",
        "passed": all(checks.values()),
        "decision": "accept_v27_exposed_development_result" if all(checks.values()) else "quarantine_v27_result",
        "checks": checks, "reproduced_development_gates": reproduced,
    }
    (PROJECT_ROOT / args.audit).write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    ref = lock["source_v26_reference"]
    support = integration["frozen_support_oracle_query"]
    lines = [
        "# V27 results: outcome-constrained support MAP", "",
        f"Decision: `{result['decision']}`.", "",
        "V27 is an exposed-data support-only experiment. It reused all V26 query predictions,",
        "scored 1,103 remaining support proposal edges, and selected support graphs jointly with one",
        "shared episode program under observed-transition consistency.", "",
        "## Support and integration", "",
        f"Evaluation exact support graphs: {evaluation['exact_support_graph']:.3f}.",
        f"Frozen-support/oracle-query execution changed from {ref['frozen_support_oracle_query_exact']:.3f} to {support['transition_set_exact_match']:.3f};",
        f"target retention changed from {ref['target_retention']:.3f} to {support['target_retention_rate']:.3f},",
        f"and empty version spaces changed from {ref['empty_version_space']:.3f} to {support['empty_version_space_rate']:.3f}.", "",
        "| Support graph | Query graph | Transition-set exact | Target retention | Empty |",
        "|---|---|---:|---:|---:|",
    ]
    for name in lock["integration_conditions"]:
        row = integration[name]
        lines.append(
            f"| {row['support_mode']} | {row['query_mode']} | {row['transition_set_exact_match']:.3f} | "
            f"{row['target_retention_rate']:.3f} | {row['empty_version_space_rate']:.3f} |"
        )
    lines.extend([
        "", "## Search diagnostics", "",
        f"Mean retained graphs per support scene: {result['graph_search']['mean_graph_branches']:.1f}; "
        f"target graph branch coverage: {result['graph_search']['target_graph_in_branch_rate']:.3f}; "
        f"episode fallback rate: {result['graph_search']['episode_fallback_rate']:.3f}.", "",
        "No query prediction, model weight, head, threshold, ontology, DSL, or executor changed.",
        "No fresh benchmark was constructed.", "",
        f"Post-result integrity audit: `{'pass' if audit['passed'] else 'fail'}`.", "",
    ])
    (PROJECT_ROOT / args.markdown).write_text("\n".join(lines))
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
