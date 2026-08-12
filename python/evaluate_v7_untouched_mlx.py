#!/usr/bin/env python3
"""Run the one permitted V7 evaluation on the untouched tone-drift mechanic."""

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
from evaluate_v5_challenge_mlx import surface_invariance
from extract_v6_development_features_mlx import forward_layer_six, prompt_for
from train_frozen_linear_probe import error_concentration, grouped_bootstrap
from train_v7_frozen_probe import breakdown
from v7_metrics import (
    gate_report,
    grouped_context_metrics,
    paired_evidence_metrics,
    worst_stratum_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v7-frozen-probe-lock.json")
    parser.add_argument("--output-dir", default="outputs/v7-untouched/frozen-probe")
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
    if lock["untouched_mechanic_evaluations_permitted"] != 1:
        raise RuntimeError("V7 lock does not permit exactly one untouched evaluation")
    if any(lock[key] != 0 for key in (
        "untouched_mechanic_records_read_before_lock",
        "prior_holdout_records_read",
        "v3_test_records_read",
    )):
        raise RuntimeError("V7 lock reports forbidden pre-lock data access")
    expected = {
        "protocol lock": (Path(lock["protocol_lock"]), lock["protocol_lock_sha256"]),
        "dataset manifest": (Path(lock["dataset_manifest"]), lock["dataset_manifest_sha256"]),
        "untouched records": (Path(lock["untouched_records"]), lock["untouched_records_sha256"]),
        "shortcut audit": (Path(lock["shortcut_audit"]), lock["shortcut_audit_sha256"]),
        "feature metadata": (Path(lock["feature_metadata"]), lock["feature_metadata_sha256"]),
        "train features": (Path(lock["train_features"]), lock["train_features_sha256"]),
        "calibration features": (
            Path(lock["calibration_features"]),
            lock["calibration_features_sha256"],
        ),
        "training result": (Path(lock["training_result"]), lock["training_result_sha256"]),
        "probe": (Path(lock["probe_artifact"]), lock["probe_artifact_sha256"]),
    }
    for label, (path, expected_hash) in expected.items():
        if file_sha256(path) != expected_hash:
            raise RuntimeError(f"V7 {label} changed after probe lock")
    implementation = lock["implementation"]["evaluator"]
    if file_sha256(Path(implementation["path"])) != implementation["sha256"]:
        raise RuntimeError("V7 evaluator changed after protocol lock")
    return lock


def main() -> None:
    args = parse_args()
    lock_path = Path(args.lock)
    output_dir = Path(args.output_dir)
    result_path = output_dir / "result.json"
    if result_path.exists():
        raise RuntimeError(f"Locked V7 result already exists: {result_path}")
    lock = verify_lock(lock_path)
    records = [
        json.loads(line)
        for line in Path(lock["untouched_records"]).read_text().splitlines()
        if line
    ]
    if any(
        record["split"] != "untouched_mechanic" or record["mechanic"] != "tonedrift"
        for record in records
    ):
        raise RuntimeError("V7 untouched partition contains another mechanic")
    with np.load(lock["probe_artifact"], allow_pickle=False) as values:
        coefficient = values["coefficient"]
        intercept = values["intercept"]
        scaler_mean = values["scaler_mean"]
        scaler_scale = values["scaler_scale"]
    if any(value.dtype != np.float32 for value in (coefficient, intercept, scaler_mean, scaler_scale)):
        raise RuntimeError("V7 locked probe is not entirely float32")

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
            raise RuntimeError("V7 untouched score is not float32")
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
                "evidence_intervention_kind": record["evidence_intervention_kind"],
                "evidence_variant": record["evidence_variant"],
                "mechanic": record["mechanic"],
                "action_template": record["action_template"],
                "gold_ambiguous": record["target"]["ambiguous"],
                "score": score,
            }
        )
        if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(records)):
            print(f"untouched: scored {index}/{len(records)}", file=sys.stderr, flush=True)
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
    action_templates = breakdown(canonical_rows, "action_template", threshold)
    invariance = surface_invariance(rows, threshold)
    grouped = grouped_context_metrics(canonical_rows, threshold)
    paired = paired_evidence_metrics(rows, threshold)
    worst = worst_stratum_metrics(canonical_rows, threshold, minimum_support=4)
    gates = gate_report(lock, canonical, bootstrap, surfaces, invariance, paired, worst)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "untouched-features.npz",
        ids=np.asarray([record["id"] for record in records]),
        layer_06_mean=np.stack(features).astype(np.float32),
    )
    (output_dir / "scores.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    result = {
        "schema_version": 7,
        "experiment": "v7_locked_tone_drift_transfer",
        "lock_sha256": file_sha256(lock_path),
        "protocol_lock_sha256": lock["protocol_lock_sha256"],
        "dataset_sha256": lock["dataset_sha256"],
        "untouched_records_sha256": lock["untouched_records_sha256"],
        "probe_artifact_sha256": lock["probe_artifact_sha256"],
        "untouched_evaluation_number": 1,
        "model": lock["method"]["model"],
        "feature": lock["method"]["feature"],
        "threshold": threshold,
        "records": len(rows),
        "base_records": len(canonical_rows),
        "context_groups": len(set(row["split_group"] for row in canonical_rows)),
        "mechanic": "tonedrift",
        "hidden_dtypes": sorted(hidden_dtypes),
        "feature_dtype": str(np.stack(features).dtype),
        "decision_score_dtype": "float32",
        "minimum_prompt_tokens": min(lengths),
        "maximum_prompt_tokens": max(lengths),
        "truncated_prompts": truncated,
        "canonical": canonical,
        "canonical_grouped_bootstrap": bootstrap,
        "canonical_grouped_context_metrics": grouped,
        "canonical_error_concentration": error_concentration(canonical_rows, threshold),
        "by_surface": surfaces,
        "by_evidence_variant_canonical": evidence_variants,
        "by_action_template_canonical": action_templates,
        "surface_invariance": invariance,
        "paired_evidence": paired,
        "worst_stratum": worst,
        "gates": gates,
        "untouched_mechanic_records_read": len(records),
        "prior_holdout_records_read": 0,
        "v3_test_records_read": 0,
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
