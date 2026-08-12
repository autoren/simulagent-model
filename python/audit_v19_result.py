"""Independent post-result replay and integrity audit for the one-shot V19 evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from audit_v18_benchmark import read_records
from audit_v19_compatibility import read_scenes
from evaluate_v19_frozen_integration import condition_modes, evaluate_condition, grounding_summary
from v10_protocol import file_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def canonical(value: Any) -> str:
    # JSON artifacts stringify integer histogram keys. Normalize through JSON once
    # before sorting so numeric insertion order cannot create a false mismatch.
    return json.dumps(json.loads(json.dumps(value)), sort_keys=True, separators=(",", ":"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v19-frozen-integration-lock.json")
    parser.add_argument("--result", default="outputs/v19-frozen-integration/evaluation/result.json")
    parser.add_argument("--correction", default="outputs/v19-frozen-integration/error-conditioning-replay.json")
    parser.add_argument("--output", default="outputs/v19-frozen-integration/post-result-audit.json")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.lock).resolve()
    result_path = (PROJECT_ROOT / args.result).resolve()
    lock = json.loads(lock_path.read_text())
    result = json.loads(result_path.read_text())
    correction_path = (PROJECT_ROOT / args.correction).resolve()
    correction = json.loads(correction_path.read_text())
    errors = []
    if result["protocol_lock_sha256"] != file_sha256(lock_path):
        errors.append("Result protocol lock hash differs")
    if result["evaluation_number"] != 1:
        errors.append("Result is not the sole locked evaluation")
    metadata_path = PROJECT_ROOT / "outputs/v19-frozen-integration/features/metadata.json"
    metadata = json.loads(metadata_path.read_text())
    feature_path = PROJECT_ROOT / metadata["feature_artifact"]
    if file_sha256(feature_path) != result["feature_artifact_sha256"]:
        errors.append("Result feature artifact hash differs")
    prediction_path = PROJECT_ROOT / result["grounding_predictions"]
    if file_sha256(prediction_path) != result["grounding_predictions_sha256"]:
        errors.append("Grounding prediction artifact hash differs")
    predictions = [json.loads(line) for line in prediction_path.read_text().splitlines() if line]
    scenes = read_scenes(PROJECT_ROOT / lock["source"]["v19_dataset"])
    scenes_by_id = {value["id"]: value for value in scenes}
    if len(predictions) != len(scenes) or {value["scene_id"] for value in predictions} != set(scenes_by_id):
        errors.append("Grounding predictions do not cover every V19 scene")
    episodes = read_records(PROJECT_ROOT / lock["source"]["v18_dataset"])
    primary = [value for value in episodes if value["split"] == lock["primary_split"]]
    replay_grounding = {}
    replay_views = {}
    for view, role in lock["views"].items():
        selected = [value for value in predictions if value["view"] == view]
        development = [value for value in selected if value["split"] == lock["primary_split"]]
        replay_grounding[view] = {
            "role": role,
            "all": grounding_summary(selected, scenes_by_id),
            "development": grounding_summary(development, scenes_by_id),
        }
        lookup = {
            (value["episode_id"], value["source_item_id"]): value for value in selected
        }
        replay_views[view] = {"role": role, "conditions": {}}
        for condition in lock["conditions"]:
            support_mode, query_mode = condition_modes(condition)
            replay_views[view]["conditions"][condition] = evaluate_condition(
                primary, lookup, support_mode, query_mode
            )
    grounding_match = canonical(replay_grounding) == canonical(result["grounding"])
    conditions_match = canonical(replay_views) == canonical(result["views"])
    if not grounding_match:
        errors.append("Saved grounding aggregates do not reproduce from prediction rows")
    if not conditions_match:
        errors.append("Saved integration conditions do not reproduce from prediction rows")
    gates = lock["gates"]["integration"]
    supported = replay_views["supported"]["conditions"]["frozen_support_frozen_query"]
    oracle = replay_views["supported"]["conditions"]["oracle_support_oracle_query"]
    replay_checks = {
        "oracle_ceiling_reproduced": oracle["episode_metrics"]["episode_macro_transition_set_exact_match"] == 1.0,
        "supported_end_to_end_episode_macro": (
            supported["episode_metrics"]["episode_macro_transition_set_exact_match"]
            >= gates["minimumSupportedEndToEndEpisodeMacroTransitionSetExact"]
        ),
        "supported_empty_version_space": (
            supported["schema_recovery"]["empty_version_space_rate"]
            <= gates["maximumSupportedEmptyVersionSpaceRate"]
        ),
        "supported_target_retention": (
            supported["schema_recovery"]["target_retention_rate"]
            >= gates["minimumSupportedTargetRetentionAfterAllSupports"]
        ),
    }
    checks_match = replay_checks == result["checks"]
    if not checks_match:
        errors.append("Locked integration checks do not reproduce")
    if result["empty_version_policy"] != lock["empty_version_policy"]:
        errors.append("Result empty-version policy differs from preregistration")
    expected_access = {
        "adapter_training_runs": 0,
        "new_linear_fits": 0,
        "v17_head_artifacts_read": 1,
        "v17_model_results_read": 0,
        "v17_records_read": 0,
    }
    if result["data_access"] != expected_access:
        errors.append("Result data-access declaration differs")
    correction_verified = (
        correction["passed"]
        and not correction["primary_metrics_affected"]
        and correction["primary_decision_unchanged"] == result["decision"]
        and correction["source"]["protocol_lock_sha256"] == file_sha256(lock_path)
        and correction["source"]["result_sha256"] == file_sha256(result_path)
        and correction["source"]["grounding_predictions_sha256"] == file_sha256(prediction_path)
    )
    if not correction_verified:
        errors.append("Scope-correct grounding-error replay does not share immutable V19 artifacts")
    passed = not errors and all(replay_checks.values()) and result["passed"]
    audit = {
        "passed": passed,
        "errors": errors,
        "decision": result["decision"] if passed else "v19_result_audit_fails",
        "grounding_predictions": len(predictions),
        "grounding_aggregates_reproduced": grounding_match,
        "integration_conditions_reproduced": conditions_match,
        "locked_checks_reproduced": checks_match,
        "error_conditioning_replay_verified": correction_verified,
        "checks": replay_checks,
        "artifacts": {
            "protocol_lock_sha256": file_sha256(lock_path),
            "feature_artifact_sha256": file_sha256(feature_path),
            "grounding_predictions_sha256": file_sha256(prediction_path),
            "result_sha256": file_sha256(result_path),
            "error_conditioning_replay_sha256": file_sha256(correction_path),
        },
        "data_access": expected_access,
    }
    output_path = (PROJECT_ROOT / args.output).resolve()
    output_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
