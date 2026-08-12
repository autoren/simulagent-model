#!/usr/bin/env python3
"""Run the one permitted V6 frozen-probe evaluation on the mirror-reject holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx_lm import load

from binary_metrics import evaluate_binary
from evaluate_v5_challenge_mlx import evidence_contrasts, surface_invariance
from extract_v6_development_features_mlx import forward_layer_six, prompt_for
from train_frozen_linear_probe import error_concentration, grouped_bootstrap
from train_v6_frozen_probe import breakdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v6-frozen-probe-lock.json")
    parser.add_argument("--output-dir", default="outputs/v6-mechanic-holdout/frozen-probe")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_lock(lock_path: Path) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text())
    if lock["mechanic_holdout_evaluations_permitted"] != 1:
        raise RuntimeError("V6 lock does not permit exactly one holdout evaluation")
    if lock["mechanic_holdout_records_read_before_lock"] != 0 or lock["v3_test_records_read"] != 0:
        raise RuntimeError("V6 lock reports forbidden pre-lock data access")
    expected = {
        "protocol lock": (Path(lock["protocol_lock"]), lock["protocol_lock_sha256"]),
        "dataset manifest": (
            Path(lock["dataset_manifest"]),
            lock["dataset_manifest_sha256"],
        ),
        "holdout records": (
            Path(lock["holdout_records"]),
            lock["holdout_records_sha256"],
        ),
        "feature metadata": (
            Path(lock["feature_metadata"]),
            lock["feature_metadata_sha256"],
        ),
        "train features": (Path(lock["train_features"]), lock["train_features_sha256"]),
        "calibration features": (
            Path(lock["calibration_features"]),
            lock["calibration_features_sha256"],
        ),
        "training result": (
            Path(lock["training_result"]),
            lock["training_result_sha256"],
        ),
        "probe": (Path(lock["probe_artifact"]), lock["probe_artifact_sha256"]),
    }
    for label, (path, expected_hash) in expected.items():
        if file_sha256(path) != expected_hash:
            raise RuntimeError(f"V6 {label} changed after probe lock")
    implementation = lock["implementation"]["evaluator"]
    if file_sha256(Path(implementation["path"])) != implementation["sha256"]:
        raise RuntimeError("V6 evaluator changed after protocol lock")
    return lock


def gate_report(
    lock: dict[str, Any],
    canonical: dict[str, Any],
    bootstrap: dict[str, Any],
    surfaces: dict[str, dict[str, Any]],
    invariance: dict[str, Any],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    gates = lock["gates"]
    reference = lock["reference"]["v5_challenge_balanced_accuracy"]
    improvement = canonical["balanced_accuracy"] - reference
    checks = [
        {
            "name": "development_calibration_balanced_accuracy",
            "value": lock["calibration_gate"]["value"],
            "minimum": gates["minimumCalibrationCanonicalBalancedAccuracy"],
        },
        {
            "name": "holdout_canonical_balanced_accuracy",
            "value": canonical["balanced_accuracy"],
            "minimum": gates["minimumHoldoutCanonicalBalancedAccuracy"],
        },
        {
            "name": "holdout_bootstrap_lower_bound",
            "value": bootstrap["balanced_accuracy_95_percentile_interval"][0],
            "minimum": gates["minimumHoldoutBootstrapLowerBound"],
        },
        *[
            {
                "name": f"surface_{surface}_balanced_accuracy",
                "value": surfaces[surface]["balanced_accuracy"],
                "minimum": gates["minimumSurfaceBalancedAccuracy"],
            }
            for surface in ("entity_renamed", "paraphrased")
        ],
        *[
            {
                "name": f"surface_{surface}_prediction_agreement",
                "value": invariance["transformations"][surface]["prediction_agreement"],
                "minimum": gates["minimumSurfacePredictionAgreement"],
            }
            for surface in ("entity_renamed", "paraphrased")
        ],
        {
            "name": "complete_surface_triplet_accuracy",
            "value": invariance["complete_triplet_accuracy"],
            "minimum": gates["minimumCompleteTripletAccuracy"],
        },
        {
            "name": "absolute_improvement_over_v5_challenge",
            "value": improvement,
            "minimum": gates["minimumAbsoluteImprovementOverV5"],
        },
        {
            "name": "evidence_directional_accuracy",
            "value": evidence["directional_accuracy"],
            "minimum": gates["minimumEvidenceDirectionalAccuracy"],
        },
    ]
    for check in checks:
        check["passed"] = check["value"] >= check["minimum"]
    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "reference_v5_challenge_balanced_accuracy": reference,
        "absolute_improvement": improvement,
    }


def main() -> None:
    args = parse_args()
    lock_path = Path(args.lock)
    output_dir = Path(args.output_dir)
    result_path = output_dir / "result.json"
    if result_path.exists():
        raise RuntimeError(f"Locked V6 result already exists and cannot be overwritten: {result_path}")
    lock = verify_lock(lock_path)
    records = [
        json.loads(line)
        for line in Path(lock["holdout_records"]).read_text().splitlines()
        if line
    ]
    if any(record["split"] != "mechanic_holdout" or record["mechanic"] != "mirrorreject" for record in records):
        raise RuntimeError("V6 holdout contains a non-held-out mechanic")
    with np.load(lock["probe_artifact"], allow_pickle=False) as values:
        coefficient = values["coefficient"]
        intercept = values["intercept"]
        scaler_mean = values["scaler_mean"]
        scaler_scale = values["scaler_scale"]
    if any(value.dtype != np.float32 for value in (coefficient, intercept, scaler_mean, scaler_scale)):
        raise RuntimeError("V6 locked probe artifact is not entirely float32")

    model, tokenizer = load(lock["method"]["model"])
    model.eval()
    rows: list[dict[str, Any]] = []
    features: list[np.ndarray] = []
    lengths: list[int] = []
    hidden_dtypes: set[str] = set()
    truncated = 0
    for index, record in enumerate(records, start=1):
        tokens = prompt_for(record, tokenizer)
        if len(tokens) > lock["method"]["max_seq_length"]:
            tokens = tokens[-lock["method"]["max_seq_length"] :]
            truncated += 1
        hidden = forward_layer_six(model, mx.array([tokens]))[0]
        mx.eval(hidden)
        hidden_dtypes.add(str(hidden.dtype))
        feature = np.asarray(mx.mean(hidden.astype(mx.float32), axis=0), dtype=np.float32)
        standardized = ((feature - scaler_mean) / scaler_scale).astype(np.float32)
        raw_score = standardized @ coefficient.T + intercept
        if raw_score.dtype != np.float32:
            raise RuntimeError("V6 holdout decision score is not float32")
        score = float(raw_score.reshape(-1)[0])
        features.append(feature)
        lengths.append(len(tokens))
        rows.append(
            {
                "id": record["id"],
                "split_group": record["split_group"],
                "surface_pair_id": record["surface_pair_id"],
                "surface_variant": record["surface_variant"],
                "evidence_pair_id": record["evidence_intervention_id"],
                "evidence_variant": record["evidence_variant"],
                "mechanic": record["mechanic"],
                "gold_ambiguous": record["target"]["ambiguous"],
                "score": score,
            }
        )
        if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(records)):
            print(f"holdout: scored {index}/{len(records)}", file=sys.stderr, flush=True)
        mx.clear_cache()

    threshold = lock["threshold"]
    canonical_rows = [row for row in rows if row["surface_variant"] == "canonical"]
    canonical = evaluate_binary(
        [row["gold_ambiguous"] for row in canonical_rows],
        [row["score"] for row in canonical_rows],
        threshold,
    )
    canonical_gold = np.asarray([row["gold_ambiguous"] for row in canonical_rows], dtype=bool)
    canonical_scores = np.asarray([row["score"] for row in canonical_rows], dtype=float)
    canonical_groups = np.asarray([row["split_group"] for row in canonical_rows])
    bootstrap = grouped_bootstrap(
        canonical_gold,
        canonical_scores,
        canonical_groups,
        threshold,
        lock["method"]["bootstrap_samples"],
        lock["method"]["bootstrap_seed"] + 1,
    )
    surfaces = breakdown(rows, "surface_variant", threshold)
    evidence_variants = breakdown(canonical_rows, "evidence_variant", threshold)
    invariance = surface_invariance(rows, threshold)
    evidence = evidence_contrasts(rows, threshold)
    gates = gate_report(lock, canonical, bootstrap, surfaces, invariance, evidence)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "holdout-features.npz",
        ids=np.asarray([record["id"] for record in records]),
        layer_06_mean=np.stack(features).astype(np.float32),
    )
    (output_dir / "scores.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    result = {
        "experiment": "v6_locked_mirrorreject_transfer",
        "lock_sha256": file_sha256(lock_path),
        "protocol_lock_sha256": lock["protocol_lock_sha256"],
        "dataset_sha256": lock["dataset_sha256"],
        "holdout_records_sha256": lock["holdout_records_sha256"],
        "probe_artifact_sha256": lock["probe_artifact_sha256"],
        "holdout_evaluation_number": 1,
        "model": lock["method"]["model"],
        "feature": lock["method"]["feature"],
        "threshold": threshold,
        "records": len(rows),
        "base_records": len(canonical_rows),
        "context_groups": len(set(row["split_group"] for row in canonical_rows)),
        "mechanic": "mirrorreject",
        "hidden_dtypes": sorted(hidden_dtypes),
        "feature_dtype": str(np.stack(features).dtype),
        "decision_score_dtype": "float32",
        "minimum_prompt_tokens": min(lengths),
        "maximum_prompt_tokens": max(lengths),
        "truncated_prompts": truncated,
        "canonical": canonical,
        "canonical_grouped_bootstrap": bootstrap,
        "canonical_error_concentration": error_concentration(canonical_rows, threshold),
        "by_surface": surfaces,
        "by_evidence_variant_canonical": evidence_variants,
        "surface_invariance": invariance,
        "evidence_contrasts": evidence,
        "gates": gates,
        "mechanic_holdout_records_read": len(records),
        "v3_test_records_read": 0,
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
