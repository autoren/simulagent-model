"""Zero-forward-pass independent replay of the sealed V21 final conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from audit_v19_compatibility import read_scenes
from evaluate_v19_frozen_integration import evaluate_condition as evaluate_hard_condition
from evaluate_v20_probabilistic_interface import evaluate_condition as evaluate_probabilistic_condition
from v10_protocol import file_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal", default="configs/v21-final-dataset-seal.json")
    parser.add_argument("--result", default="outputs/v21-final/evaluation/result.json")
    parser.add_argument("--output", default="outputs/v21-final/post-result-audit.json")
    args = parser.parse_args()
    output = PROJECT_ROOT / args.output
    if output.exists():
        raise RuntimeError(f"V21 replay already exists: {output}")
    seal_path = PROJECT_ROOT / args.seal
    seal = json.loads(seal_path.read_text())
    lock_path = PROJECT_ROOT / seal["execution_lock"]
    lock = json.loads(lock_path.read_text())
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V21 locked implementation changed: {path}")
    result_path = PROJECT_ROOT / args.result
    result = json.loads(result_path.read_text())
    hard_path = PROJECT_ROOT / result["hard_predictions"]
    probabilistic_path = PROJECT_ROOT / result["probabilistic_predictions"]
    hard = read_jsonl(hard_path)
    probabilistic = read_jsonl(probabilistic_path)
    manifest_path = PROJECT_ROOT / seal["manifest"]
    episodes = read_jsonl(manifest_path.parent / "episodes.jsonl")
    scenes = read_scenes(manifest_path.parent)
    hard_lookup = {
        view: {
            (value["episode_id"], value["source_item_id"]): value
            for value in hard if value["view"] == view
        }
        for view in lock["config"]["views"]
    }
    probabilistic_lookup = {
        view: {
            (value["episode_id"], value["source_item_id"]): value
            for value in probabilistic if value["view"] == view
        }
        for view in lock["config"]["views"]
    }
    hard_modes = {
        "oracle_support_oracle_query": ("oracle", "oracle"),
        "frozen_support_oracle_query": ("frozen", "oracle"),
        "oracle_support_frozen_query": ("oracle", "frozen"),
        "frozen_support_frozen_query": ("frozen", "frozen"),
    }
    condition_matches = {}
    replayed_views = {}
    for view in lock["config"]["views"]:
        replayed_hard = {
            name: evaluate_hard_condition(episodes, hard_lookup[view], support_mode, query_mode)
            for name, (support_mode, query_mode) in hard_modes.items()
        }
        replayed_probabilistic = evaluate_probabilistic_condition(
            episodes, probabilistic_lookup[view], "probabilistic", "probabilistic",
            lock["probabilistic_interface"]["credibleProgramMass"],
        )
        replayed_views[view] = {
            "hard": replayed_hard,
            "probabilistic": replayed_probabilistic,
        }
        condition_matches[f"{view}:hard"] = (
            canonical(replayed_hard) == canonical(result["views"][view]["hard_conditions"])
        )
        condition_matches[f"{view}:probabilistic"] = (
            canonical(replayed_probabilistic) == canonical(result["views"][view]["probabilistic_full"])
        )
    hard_primary = replayed_views["supported"]["hard"]["frozen_support_frozen_query"]
    oracle = replayed_views["supported"]["hard"]["oracle_support_oracle_query"]
    probabilistic_primary = replayed_views["supported"]["probabilistic"]
    hard_gate = lock["config"]["gates"]["hardSupported"]
    hard_family_complete = {
        family: hard_primary["by_axis"][family]["complete_episodes"]
        for family in lock["config"]["constructionFamilies"]
    }
    hard_checks = {
        "mechanic_macro": hard_primary["episode_metrics"]["episode_macro_transition_set_exact_match"] >= hard_gate["minimumMechanicMacroTransitionSetExact"],
        "complete_mechanics": hard_primary["episode_metrics"]["complete_episodes"] >= hard_gate["minimumCompleteMechanics"],
        "complete_per_family": all(
            hard_family_complete[family] >= minimum
            for family, minimum in hard_gate["minimumCompletePerFamily"].items()
        ),
        "target_retention": hard_primary["schema_recovery"]["target_retention_rate"] >= hard_gate["minimumTargetRetention"],
        "empty_version_rate": hard_primary["schema_recovery"]["empty_version_space_rate"] <= hard_gate["maximumEmptyVersionRate"],
        "oracle_ceiling": oracle["episode_metrics"]["episode_macro_transition_set_exact_match"] >= hard_gate["minimumOracleCeiling"],
    }
    probabilistic_gate = lock["config"]["gates"]["probabilisticSupported"]
    probabilistic_checks = {
        "eligible": lock["challenger_eligible"],
        "mechanic_macro": probabilistic_primary["episode_metrics"]["episode_macro_transition_set_exact_match"] >= probabilistic_gate["minimumMechanicMacroTransitionSetExact"],
        "complete_mechanics": probabilistic_primary["episode_metrics"]["complete_episodes"] >= probabilistic_gate["minimumCompleteMechanics"],
        "target_retention": probabilistic_primary["schema_recovery"]["target_credible_retention_rate"] >= probabilistic_gate["minimumTargetRetention"],
        "empty_posterior_rate": probabilistic_primary["schema_recovery"]["empty_posterior_rate"] <= probabilistic_gate["maximumEmptyPosteriorRate"],
        "excess_outcomes": probabilistic_primary["anti_widening"]["mean_excess_outcomes"] <= probabilistic_gate["maximumMeanExcessOutcomesPerQuery"],
    }
    hard_pass = all(hard_checks.values())
    probabilistic_pass = all(probabilistic_checks.values())
    expected_decision = (
        "hard_population_transfer_passes_probabilistic_also_passes_proceed_relational"
        if hard_pass and probabilistic_pass else
        "hard_population_transfer_passes_probabilistic_fails_proceed_relational"
        if hard_pass else
        "hard_population_claim_rejected_probabilistic_modularity_supported"
        if probabilistic_pass else
        "both_supported_systems_fail_population_robustness"
    )
    expected_scene_keys = {
        (value["view"], value["episode_id"], value["source_item_id"]) for value in scenes
    }
    hard_scene_keys = {
        (value["view"], value["episode_id"], value["source_item_id"]) for value in hard
    }
    probabilistic_scene_keys = {
        (value["view"], value["episode_id"], value["source_item_id"])
        for value in probabilistic
    }
    feature_metadata_path = PROJECT_ROOT / result["feature_metadata"]
    feature_metadata = json.loads(feature_metadata_path.read_text())
    checks = {
        "result_seal_matches": result["dataset_seal_sha256"] == file_sha256(seal_path),
        "hard_prediction_hash_matches": result["hard_predictions_sha256"] == file_sha256(hard_path),
        "probabilistic_prediction_hash_matches": (
            result["probabilistic_predictions_sha256"] == file_sha256(probabilistic_path)
        ),
        "feature_metadata_hash_matches": result["feature_metadata_sha256"] == file_sha256(feature_metadata_path),
        "feature_artifact_hash_matches": result["feature_artifact_sha256"] == feature_metadata["feature_artifact_sha256"],
        "hard_prediction_inventory_exact": len(hard) == len(scenes) and hard_scene_keys == expected_scene_keys,
        "probabilistic_prediction_inventory_exact": len(probabilistic) == len(scenes) and probabilistic_scene_keys == expected_scene_keys,
        "all_conditions_reproduced": all(condition_matches.values()),
        "hard_gate_checks_reproduced": canonical(hard_checks) == canonical(result["hard_supported_checks"]),
        "hard_gate_decision_reproduced": hard_pass == result["hard_supported_passed"],
        "probabilistic_gate_checks_reproduced": canonical(probabilistic_checks) == canonical(result["probabilistic_supported_checks"]),
        "probabilistic_gate_decision_reproduced": probabilistic_pass == result["probabilistic_supported_passed"],
        "decision_reproduced": result["decision"] == expected_decision,
        "single_evaluation": result["evaluation_number"] == 1,
        "retry_forbidden": result["retry_authorized"] is False,
        "lora_not_authorized": result["lora_authorized"] is False,
    }
    report = {
        "schema_version": 21,
        "experiment": "v21_zero_forward_pass_result_replay",
        "result": args.result,
        "result_sha256": file_sha256(result_path),
        "condition_matches": condition_matches,
        "replayed_hard_supported_checks": hard_checks,
        "replayed_probabilistic_supported_checks": probabilistic_checks,
        "checks": checks,
        "passed": all(checks.values()),
        "decision": result["decision"],
        "data_access": {
            "saved_predictions_read": len(hard) + len(probabilistic),
            "model_forward_passes": 0,
            "feature_extractions": 0,
            "new_linear_fits": 0,
            "adapter_training_runs": 0,
            "final_result_replays": 1,
        },
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
