"""Integrity audit for the completed V22r2a amended evaluation."""

from __future__ import annotations

import argparse
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v22r2a-evaluation-amendment-lock.json")
    parser.add_argument("--result", default="outputs/v22r2-relational-grounding/evaluation-v22r2a/result.json")
    parser.add_argument("--output", default="outputs/v22r2-relational-grounding/post-result-audit.json")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.lock).resolve()
    result_path = (PROJECT_ROOT / args.result).resolve()
    lock = json.loads(lock_path.read_text())
    result = json.loads(result_path.read_text())
    original_lock = json.loads((PROJECT_ROOT / lock["source"]["original_lock"]).read_text())
    predictions_path = PROJECT_ROOT / result["grounding_predictions"]
    heads_path = PROJECT_ROOT / result["heads_artifact"]
    attempt_path = PROJECT_ROOT / "outputs/v22r2-relational-grounding/evaluation-v22r2a-attempt.json"
    attempt = json.loads(attempt_path.read_text())
    predictions = [
        json.loads(line) for line in predictions_path.read_text().splitlines() if line.strip()
    ]
    gates = original_lock["gates"]["development"]
    grounding = result["grounding"]["by_split"]
    integration = result["integration"]
    expected_checks = {
        "oracle_oracle_transition_set_exact": integration["oracle_support_oracle_query"]["transition_set_exact_match"] >= gates["minimumOracleOracleTransitionSetExact"],
        "fit_atom_assignment": grounding["grounding_fit"]["atom_assignment_accuracy"] >= gates["minimumFitAtomAssignmentAccuracy"],
        "evaluation_atom_assignment": grounding["grounding_evaluation"]["atom_assignment_accuracy"] >= gates["minimumEvaluationAtomAssignmentAccuracy"],
        "evaluation_truth_status": grounding["grounding_evaluation"]["truth_status_accuracy"] >= gates["minimumEvaluationTruthStatusAccuracy"],
        "evaluation_relation_orientation": grounding["grounding_evaluation"]["relation_argument_order_accuracy"] >= gates["minimumEvaluationRelationOrientationAccuracy"],
        "evaluation_exact_scene_graph": grounding["grounding_evaluation"]["exact_scene_graph"] >= gates["minimumEvaluationExactSceneGraph"],
        "frozen_frozen_transition_set_exact": integration["frozen_support_frozen_query"]["transition_set_exact_match"] >= gates["minimumFrozenFrozenTransitionSetExact"],
        "frozen_support_target_retention": integration["frozen_support_oracle_query"]["target_retention_rate"] >= gates["minimumFrozenSupportTargetRetention"],
        "frozen_support_empty_version_space": integration["frozen_support_oracle_query"]["empty_version_space_rate"] <= gates["maximumFrozenSupportEmptyVersionSpaceRate"],
    }
    checks = {
        "amendment_lock_matches": result["amendment_lock_sha256"] == file_sha256(lock_path),
        "original_lock_matches": result["original_protocol_lock_sha256"] == lock["source"]["original_lock_sha256"],
        "feature_artifact_matches": result["feature_artifact_sha256"] == lock["source"]["feature_artifact_sha256"],
        "prediction_artifact_matches": file_sha256(predictions_path) == result["grounding_predictions_sha256"],
        "head_artifact_matches": file_sha256(heads_path) == result["heads_artifact_sha256"],
        "attempt_completed": attempt["status"] == "completed" and attempt["result_sha256"] == file_sha256(result_path),
        "all_scenes_predicted_once": len(predictions) == 384 and len({row["scene_id"] for row in predictions}) == 384,
        "registered_checks_reproduced": expected_checks == result["checks"],
        "pass_flag_reproduced": result["passed"] == all(expected_checks.values()),
        "decision_reproduced": result["decision"] == "develop_probabilistic_support_interface_no_lora",
        "failed_attempt_disclosed": result["aborted_attempts_before_predictions"] == 1,
        "zero_amended_model_forwards": result["data_access"]["new_model_forward_passes"] == 0,
        "zero_selection_and_adapters": (
            result["data_access"]["hyperparameter_selections"] == 0
            and result["data_access"]["adapter_training_runs"] == 0
        ),
    }
    output = {
        "schema_version": "22r2a",
        "experiment": "v22r2a_post_result_integrity_audit",
        "passed": all(checks.values()),
        "decision": "accept_v22r2a_negative_result" if all(checks.values()) else "quarantine_v22r2a_result",
        "checks": checks,
        "registered_gate_checks": expected_checks,
        "result_passed_development_gates": result["passed"],
        "result_decision": result["decision"],
    }
    output_path = (PROJECT_ROOT / args.output).resolve()
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    if not output["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
