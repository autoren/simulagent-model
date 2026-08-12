#!/usr/bin/env python3
"""Run the one permitted locked V5 probe evaluation on the challenge holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx_lm import load

from binary_metrics import evaluate_binary
from extract_frozen_qwen_features_mlx import forward_captures, input_variant
from train_frozen_linear_probe import error_concentration, grouped_bootstrap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v5-frozen-probe-lock.json")
    parser.add_argument("--manifest", default="data/v5-challenge/manifest.json")
    parser.add_argument("--records", default="data/v5-challenge/records/challenge.jsonl")
    parser.add_argument("--output-dir", default="outputs/v5-challenge/frozen-probe")
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=3510)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def verify_lock(
    lock: dict[str, Any], manifest: dict[str, Any], records_path: Path, lock_path: Path
) -> None:
    checks = {
        "probe artifact": (Path(lock["probe_artifact"]), lock["probe_artifact_sha256"]),
        "source result": (Path(lock["source_result"]), lock["source_result_sha256"]),
        "feature metadata": (
            Path(lock["source_feature_metadata"]),
            lock["source_feature_metadata_sha256"],
        ),
        "dataset manifest": (
            Path(lock["source_dataset_manifest"]),
            lock["source_dataset_manifest_sha256"],
        ),
    }
    for label, (path, expected) in checks.items():
        observed = file_sha256(path)
        if observed != expected:
            raise RuntimeError(f"Locked {label} hash changed: {observed} != {expected}")
    if text_sha256(lock["system_prompt"]) != lock["system_prompt_sha256"]:
        raise RuntimeError("Locked system prompt hash is invalid")
    records_content = records_path.read_text()
    dataset_hash = text_sha256(f"records/challenge.jsonl\n{records_content}")
    if dataset_hash != manifest["dataset_sha256"]:
        raise RuntimeError(f"Challenge dataset hash changed: {dataset_hash}")
    lock_hash = file_sha256(lock_path)
    if lock_hash != manifest["frozen_probe_lock_sha256"]:
        raise RuntimeError(f"Challenge manifest references a different lock: {lock_hash}")
    if lock["challenge_evaluations_permitted"] != 1:
        raise RuntimeError("Frozen lock does not permit exactly one challenge evaluation")
    if lock["test_records_read"] != 0 or manifest["source_test_records_read"] != 0:
        raise RuntimeError("A locked artifact reports test access")


def safe_metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    labels = [row["gold_ambiguous"] for row in rows]
    if len(set(labels)) < 2:
        return {
            "examples": len(rows),
            "ambiguous_examples": sum(labels),
            "identifiable_examples": len(rows) - sum(labels),
            "metrics_available": False,
        }
    result = evaluate_binary(labels, [row["score"] for row in rows], threshold)
    result["metrics_available"] = True
    return result


def breakdown(
    rows: list[dict[str, Any]], key: str, threshold: float
) -> dict[str, dict[str, Any]]:
    values: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        values[str(row[key])].append(row)
    return {name: safe_metrics(selected, threshold) for name, selected in sorted(values.items())}


def surface_invariance(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    pairs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        pairs[row["surface_pair_id"]][row["surface_variant"]] = row
    transformations: dict[str, Any] = {}
    for variant in ("entity_renamed", "paraphrased"):
        comparisons = []
        for values in pairs.values():
            canonical = values["canonical"]
            transformed = values[variant]
            canonical_prediction = canonical["score"] > threshold
            transformed_prediction = transformed["score"] > threshold
            comparisons.append(
                {
                    "agreement": canonical_prediction == transformed_prediction,
                    "both_correct": (
                        canonical_prediction == canonical["gold_ambiguous"]
                        and transformed_prediction == transformed["gold_ambiguous"]
                    ),
                    "absolute_score_shift": abs(transformed["score"] - canonical["score"]),
                }
            )
        shifts = np.asarray([value["absolute_score_shift"] for value in comparisons])
        transformations[variant] = {
            "pairs": len(comparisons),
            "prediction_agreement": float(np.mean([value["agreement"] for value in comparisons])),
            "both_correct_rate": float(np.mean([value["both_correct"] for value in comparisons])),
            "mean_absolute_score_shift": float(np.mean(shifts)),
            "p95_absolute_score_shift": float(np.quantile(shifts, 0.95)),
        }
    triplet_correct = []
    for values in pairs.values():
        triplet_correct.append(
            all(
                (row["score"] > threshold) == row["gold_ambiguous"]
                for row in values.values()
            )
        )
    return {
        "surface_pairs": len(pairs),
        "complete_triplet_accuracy": float(np.mean(triplet_correct)),
        "transformations": transformations,
    }


def evidence_contrasts(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["surface_variant"] == "canonical" and row["evidence_pair_id"] is not None:
            groups[row["evidence_pair_id"]].append(row)
    directional = []
    complete = []
    summaries = []
    for pair_id, values in sorted(groups.items()):
        ambiguous = [value for value in values if value["gold_ambiguous"]]
        identifiable = [value for value in values if not value["gold_ambiguous"]]
        comparisons = [left["score"] > right["score"] for left in ambiguous for right in identifiable]
        directional.extend(comparisons)
        pair_complete = all(
            (value["score"] > threshold) == value["gold_ambiguous"] for value in values
        )
        complete.append(pair_complete)
        summaries.append(
            {
                "evidence_pair_id": pair_id,
                "records": len(values),
                "ambiguous_records": len(ambiguous),
                "identifiable_records": len(identifiable),
                "directional_accuracy": float(np.mean(comparisons)) if comparisons else 0.0,
                "complete_classification": pair_complete,
            }
        )
    return {
        "groups": len(groups),
        "cross_label_comparisons": len(directional),
        "directional_accuracy": float(np.mean(directional)) if directional else 0.0,
        "complete_group_accuracy": float(np.mean(complete)) if complete else 0.0,
        "group_summaries": summaries,
    }


def gate_report(
    manifest: dict[str, Any],
    canonical: dict[str, Any],
    mechanics: dict[str, dict[str, Any]],
    surfaces: dict[str, dict[str, Any]],
    invariance: dict[str, Any],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    gates = manifest["config"]["evaluationGates"]
    checks = [
        {
            "name": "canonical_balanced_accuracy",
            "value": canonical["balanced_accuracy"],
            "minimum": gates["minimumCanonicalBalancedAccuracy"],
        },
        *[
            {
                "name": f"mechanic_{name}_balanced_accuracy",
                "value": value["balanced_accuracy"],
                "minimum": gates["minimumPerMechanicBalancedAccuracy"],
            }
            for name, value in mechanics.items()
        ],
        *[
            {
                "name": f"surface_{name}_balanced_accuracy",
                "value": surfaces[name]["balanced_accuracy"],
                "minimum": gates["minimumSurfaceBalancedAccuracy"],
            }
            for name in ("entity_renamed", "paraphrased")
        ],
        *[
            {
                "name": f"surface_{name}_prediction_agreement",
                "value": invariance["transformations"][name]["prediction_agreement"],
                "minimum": gates["minimumSurfacePredictionAgreement"],
            }
            for name in ("entity_renamed", "paraphrased")
        ],
        {
            "name": "evidence_directional_accuracy",
            "value": evidence["directional_accuracy"],
            "minimum": gates["minimumEvidenceDirectionalAccuracy"],
        },
    ]
    for check in checks:
        check["passed"] = check["value"] >= check["minimum"]
    return {"passed": all(check["passed"] for check in checks), "checks": checks}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    result_path = output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(
            f"Locked challenge result already exists and cannot be overwritten: {result_path}"
        )
    lock_path = Path(args.lock)
    manifest_path = Path(args.manifest)
    records_path = Path(args.records)
    lock = read_json(lock_path)
    manifest = read_json(manifest_path)
    verify_lock(lock, manifest, records_path, lock_path)
    records = read_jsonl(records_path)
    feature_parts = lock["feature"].split("_")
    layer = int(feature_parts[1])
    pooling = feature_parts[2]
    if pooling != "mean":
        raise RuntimeError(f"Challenge evaluator supports locked mean pooling, got {pooling}")
    with np.load(lock["probe_artifact"], allow_pickle=False) as probe:
        coefficient = probe["coefficient"]
        intercept = probe["intercept"]
        scaler_mean = probe["scaler_mean"]
        scaler_scale = probe["scaler_scale"]
    if any(value.dtype != np.float32 for value in (coefficient, intercept, scaler_mean, scaler_scale)):
        raise RuntimeError("Locked probe artifact is not entirely float32")
    model, tokenizer = load(lock["model"])
    model.eval()
    scores = []
    features = []
    prompt_lengths = []
    truncated = 0
    hidden_dtype = None
    for index, record in enumerate(records, start=1):
        user_input = input_variant(record, lock["input_variant"])
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": lock["system_prompt"]},
                {
                    "role": "user",
                    "content": json.dumps(user_input, sort_keys=True, separators=(",", ":")),
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        tokens = tokenizer.encode(prompt)
        if len(tokens) > lock["max_seq_length"]:
            tokens = tokens[-lock["max_seq_length"] :]
            truncated += 1
        captures = forward_captures(model, mx.array([tokens]), {layer})
        hidden = captures[layer][0]
        hidden_dtype = str(hidden.dtype)
        feature = np.asarray(mx.mean(hidden.astype(mx.float32), axis=0), dtype=np.float32)
        scaled = (feature - scaler_mean) / scaler_scale
        score = scaled @ coefficient[0] + intercept[0]
        if feature.dtype != np.float32 or score.dtype != np.float32:
            raise RuntimeError(f"Challenge scoring left float32: {feature.dtype}, {score.dtype}")
        features.append(feature)
        scores.append(float(score))
        prompt_lengths.append(len(tokens))
        if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(records)):
            print(f"challenge: scored {index}/{len(records)}", file=sys.stderr, flush=True)
        mx.clear_cache()
    threshold = lock["threshold"]
    score_rows = [
        {
            "id": record["id"],
            "base_record_id": record["base_record_id"],
            "base_context_group": record["base_context_group"],
            "split_group": record["base_context_group"],
            "surface_pair_id": record["surface_pair_id"],
            "surface_variant": record["surface_variant"],
            "evidence_pair_id": record["evidence_pair_id"],
            "evidence_variant": record["evidence_variant"],
            "mechanic": record["mechanic"],
            "gold_ambiguous": not record["target"]["identifiable"],
            "score": score,
        }
        for record, score in zip(records, scores)
    ]
    canonical_rows = [row for row in score_rows if row["surface_variant"] == "canonical"]
    canonical_metrics = safe_metrics(canonical_rows, threshold)
    surface_metrics = breakdown(score_rows, "surface_variant", threshold)
    mechanic_metrics = breakdown(canonical_rows, "mechanic", threshold)
    evidence_metrics = breakdown(canonical_rows, "evidence_variant", threshold)
    invariance = surface_invariance(score_rows, threshold)
    evidence = evidence_contrasts(score_rows, threshold)
    bootstrap = grouped_bootstrap(
        np.asarray([row["gold_ambiguous"] for row in canonical_rows], dtype=bool),
        np.asarray([row["score"] for row in canonical_rows]),
        np.asarray([row["base_context_group"] for row in canonical_rows]),
        threshold,
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    result = {
        "experiment": "v5_locked_frozen_probe_challenge",
        "lock_sha256": file_sha256(lock_path),
        "challenge_dataset_sha256": manifest["dataset_sha256"],
        "model": lock["model"],
        "feature": lock["feature"],
        "threshold": threshold,
        "evaluation_split": "challenge",
        "challenge_evaluation_number": 1,
        "test_records_read": 0,
        "records": len(records),
        "base_records": len(canonical_rows),
        "context_groups": len(set(row["base_context_group"] for row in canonical_rows)),
        "truncated_prompts": truncated,
        "minimum_prompt_tokens": min(prompt_lengths),
        "maximum_prompt_tokens": max(prompt_lengths),
        "hidden_dtype": hidden_dtype,
        "feature_dtype": "float32",
        "decision_score_dtype": "float32",
        "canonical": canonical_metrics,
        "canonical_grouped_bootstrap": bootstrap,
        "canonical_error_concentration": error_concentration(canonical_rows, threshold),
        "by_surface": surface_metrics,
        "by_mechanic_canonical": mechanic_metrics,
        "by_evidence_variant_canonical": evidence_metrics,
        "surface_invariance": invariance,
        "evidence_contrasts": evidence,
    }
    result["gates"] = gate_report(
        manifest,
        canonical_metrics,
        mechanic_metrics,
        surface_metrics,
        invariance,
        evidence,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "features.npz",
        ids=np.asarray([record["id"] for record in records]),
        feature=np.stack(features).astype(np.float32),
        prompt_lengths=np.asarray(prompt_lengths, dtype=np.int32),
    )
    (output_dir / "scores.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in score_rows)
    )
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
