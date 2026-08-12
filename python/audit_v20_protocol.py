"""Pre-evaluation firewall audit for the V20 development-only interface."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from audit_v18_benchmark import read_records
from audit_v19_compatibility import read_scenes
from v10_protocol import file_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v20-probabilistic-interface.json")
    parser.add_argument("--output", default="outputs/v20-probabilistic-interface/pre-evaluation-audit.json")
    args = parser.parse_args()
    config_path = PROJECT_ROOT / args.config
    config = json.loads(config_path.read_text())
    v19_lock_path = PROJECT_ROOT / config["sourceV19Lock"]
    v19_lock = json.loads(v19_lock_path.read_text())
    for path, expected in v19_lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"Locked V19 implementation changed: {path}")
    result_path = PROJECT_ROOT / config["sourceV19Result"]
    result = json.loads(result_path.read_text())
    post_path = PROJECT_ROOT / config["sourceV19PostAudit"]
    post = json.loads(post_path.read_text())
    correction_path = PROJECT_ROOT / config["sourceV19Correction"]
    correction = json.loads(correction_path.read_text())
    metadata_path = PROJECT_ROOT / config["sourceV19Features"]
    metadata = json.loads(metadata_path.read_text())
    feature_path = PROJECT_ROOT / metadata["feature_artifact"]
    head_path = PROJECT_ROOT / config["sourceDeploymentHeads"]
    scenes = read_scenes(PROJECT_ROOT / v19_lock["source"]["v19_dataset"])
    episodes = read_records(PROJECT_ROOT / v19_lock["source"]["v18_dataset"])
    split_view_counts = Counter((value["split"], value["view"]) for value in scenes)
    checks = {
        "v19_result_passed": result["passed"] is True,
        "v19_post_audit_passed": post["passed"] is True,
        "v19_correction_passed": correction["passed"] is True,
        "v19_lock_matches_result": result["protocol_lock_sha256"] == file_sha256(v19_lock_path),
        "feature_lock_matches": metadata["protocol_lock_sha256"] == file_sha256(v19_lock_path),
        "feature_artifact_matches": metadata["feature_artifact_sha256"] == file_sha256(feature_path),
        "deployment_heads_match": file_sha256(head_path) == v19_lock["source"]["deployment_heads_sha256"],
        "calibration_scenes_present": all(
            split_view_counts[(config["calibrationSplit"], view)] > 0 for view in config["views"]
        ),
        "development_scenes_present": all(
            split_view_counts[(config["evaluationSplit"], view)] > 0 for view in config["views"]
        ),
        "no_new_model_forward_passes": config["limits"]["newModelForwardPassesPermitted"] == 0,
        "no_new_fits": config["limits"]["newLinearFitsPermitted"] == 0,
        "no_final_suite_records": config["limits"]["finalSuiteRecordsPermitted"] == 0,
        "single_development_evaluation": config["limits"]["developmentEvaluationsPermitted"] == 1,
    }
    report = {
        "schema_version": 20,
        "experiment": config["experiment"],
        "config": args.config,
        "config_sha256": file_sha256(config_path),
        "checks": checks,
        "passed": all(checks.values()),
        "source": {
            "v19_lock_sha256": file_sha256(v19_lock_path),
            "v19_result_sha256": file_sha256(result_path),
            "v19_post_audit_sha256": file_sha256(post_path),
            "v19_correction_sha256": file_sha256(correction_path),
            "v19_feature_metadata_sha256": file_sha256(metadata_path),
            "v19_feature_artifact_sha256": file_sha256(feature_path),
            "deployment_heads_sha256": file_sha256(head_path),
        },
        "corpus": {
            "episodes": len(episodes),
            "scenes": len(scenes),
            "episode_splits": dict(Counter(value["split"] for value in episodes)),
            "scene_split_view_counts": {
                f"{split}:{view}": count
                for (split, view), count in sorted(split_view_counts.items())
            },
        },
        "data_access": {
            "saved_v19_features_read": 0,
            "saved_v19_results_read": 1,
            "new_model_forward_passes": 0,
            "new_linear_fits": 0,
            "adapter_training_runs": 0,
            "final_suite_records_created_or_read": 0,
        },
    }
    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
