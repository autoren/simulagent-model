#!/usr/bin/env python3
"""Development-only V8 shortcut audit with leave-one-mechanic-out folds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from binary_metrics import evaluate_binary, fit_threshold


Record = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/v8")
    parser.add_argument("--config", default="configs/dataset.v8.json")
    parser.add_argument("--output", default="outputs/v8-pre-model/shortcut-audit.json")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[Record]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def canonical_records(records: list[Record]) -> list[Record]:
    return [record for record in records if record["surface_variant"] == "canonical"]


def input_text(record: Record) -> str:
    return json.dumps(record["agent_input"], sort_keys=True, separators=(",", ":"))


def formatting_text(record: Record) -> str:
    """Remove the role-to-status binding while retaining counts and formatting cues."""
    value = record["agent_input"]
    schema = value["action_dependency_schema"]
    evidence = sorted(
        (fact["evidence_state"], fact["value"])
        for fact in value["evidence_ledger"]
    )
    transition_codes = [case["transition_code"] for case in schema["transition_cases"]]
    scrubbed = {
        "task": value["task"],
        "determinant_count": len(schema["transition_determinants"]),
        "transition_case_count": len(schema["transition_cases"]),
        "transition_code_multiplicities": sorted(
            transition_codes.count(code) for code in set(transition_codes)
        ),
        "evidence_multiset": evidence,
        "serialized_length": len(input_text(record)),
        "replica_parity": record["replica"] % 2,
    }
    return json.dumps(scrubbed, sort_keys=True, separators=(",", ":"))


def metadata_features(record: Record) -> dict[str, float]:
    schema = record["agent_input"]["action_dependency_schema"]
    return {
        f"mechanic={record['mechanic']}": 1.0,
        f"action={record['action_template']}": 1.0,
        f"surface={record['surface_variant']}": 1.0,
        f"determinant_count={len(schema['transition_determinants'])}": 1.0,
        f"evidence_count={len(record['agent_input']['evidence_ledger'])}": 1.0,
        f"replica_parity={record['replica'] % 2}": 1.0,
    }


def label(record: Record) -> bool:
    return bool(record["target"]["ambiguous"])


def fit_classifier() -> LogisticRegression:
    return LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=0,
        solver="lbfgs",
    )


def score_fold(
    train: list[Record],
    calibration: list[Record],
    test: list[Record],
    model: Any,
    values: Callable[[Record], Any],
) -> dict[str, Any]:
    train_x = [values(record) for record in train]
    calibration_x = [values(record) for record in calibration]
    test_x = [values(record) for record in test]
    train_y = np.asarray([label(record) for record in train], dtype=bool)
    calibration_y = np.asarray([label(record) for record in calibration], dtype=bool)
    test_y = np.asarray([label(record) for record in test], dtype=bool)
    model.fit(train_x, train_y)
    calibration_scores = model.predict_proba(calibration_x)[:, 1]
    threshold = fit_threshold(calibration_y.tolist(), calibration_scores.tolist())["threshold"]
    test_scores = model.predict_proba(test_x)[:, 1]
    metrics = evaluate_binary(test_y.tolist(), test_scores.tolist(), threshold)
    return {
        "balanced_accuracy": metrics["balanced_accuracy"],
        "roc_auc": metrics["roc_auc"],
        "threshold": threshold,
        "examples": len(test),
    }


def score_length_fold(
    calibration: list[Record],
    test: list[Record],
) -> dict[str, Any]:
    calibration_y = np.asarray([label(record) for record in calibration], dtype=bool)
    test_y = np.asarray([label(record) for record in test], dtype=bool)
    calibration_lengths = np.asarray([len(input_text(record)) for record in calibration], dtype=float)
    test_lengths = np.asarray([len(input_text(record)) for record in test], dtype=float)
    candidates: list[tuple[float, int, float]] = []
    for direction in (1, -1):
        scores = calibration_lengths * direction
        threshold = fit_threshold(calibration_y.tolist(), scores.tolist())["threshold"]
        metric = evaluate_binary(calibration_y.tolist(), scores.tolist(), threshold)
        candidates.append((metric["balanced_accuracy"], direction, threshold))
    _, direction, threshold = max(candidates, key=lambda value: (value[0], -value[1]))
    metrics = evaluate_binary(test_y.tolist(), (test_lengths * direction).tolist(), threshold)
    return {
        "balanced_accuracy": metrics["balanced_accuracy"],
        "roc_auc": metrics["roc_auc"],
        "threshold": threshold,
        "direction": direction,
        "examples": len(test),
    }


def audit_model(records: list[Record], mechanics: list[str], kind: str) -> dict[str, Any]:
    folds: dict[str, Any] = {}
    for heldout in mechanics:
        train = [record for record in records if record["mechanic"] != heldout and record["split"] == "train"]
        calibration = [
            record for record in records
            if record["mechanic"] != heldout and record["split"] == "calibration"
        ]
        test = [record for record in records if record["mechanic"] == heldout]
        if kind == "metadata":
            model = make_pipeline(DictVectorizer(sparse=True), fit_classifier())
            result = score_fold(train, calibration, test, model, metadata_features)
        elif kind == "unigram":
            model = make_pipeline(
                TfidfVectorizer(ngram_range=(1, 1), lowercase=True, token_pattern=r"(?u)\b[\w@:-]+\b"),
                fit_classifier(),
            )
            result = score_fold(train, calibration, test, model, input_text)
        elif kind in ("character_ngram", "relational_character_ngram"):
            model = make_pipeline(
                TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, max_features=30000),
                fit_classifier(),
            )
            result = score_fold(
                train,
                calibration,
                test,
                model,
                formatting_text if kind == "character_ngram" else input_text,
            )
        elif kind == "length":
            result = score_length_fold(calibration, test)
        else:
            raise ValueError(f"unknown audit model {kind}")
        folds[heldout] = result
    balanced = [fold["balanced_accuracy"] for fold in folds.values()]
    aucs = [fold["roc_auc"] for fold in folds.values()]
    return {
        "folds": folds,
        "mean_balanced_accuracy": float(np.mean(balanced)),
        "maximum_fold_balanced_accuracy": float(np.max(balanced)),
        "mean_auc": float(np.mean(aucs)),
        "maximum_fold_auc": float(np.max(aucs)),
        "maximum_fold_auc_separation": float(np.max([max(value, 1.0 - value) for value in aucs])),
    }


def shortcut_gate_report(audits: dict[str, Any], ceilings: dict[str, float]) -> dict[str, Any]:
    balanced_mapping = {
        "metadata": ceilings["maximumMetadataWorstFoldBalancedAccuracy"],
        "unigram": ceilings["maximumUnigramWorstFoldBalancedAccuracy"],
        "character_ngram": ceilings["maximumCharacterNgramWorstFoldBalancedAccuracy"],
        "length": ceilings["maximumLengthWorstFoldBalancedAccuracy"],
    }
    checks = []
    for name, maximum in balanced_mapping.items():
        value = audits[name]["maximum_fold_balanced_accuracy"]
        checks.append({
            "name": f"{name}_maximum_fold_balanced_accuracy",
            "value": value,
            "maximum": maximum,
            "passed": value <= maximum,
        })
    auc_mapping = {
        "unigram": ceilings["maximumUnigramWorstFoldAuc"],
        "character_ngram": ceilings["maximumCharacterNgramWorstFoldAuc"],
        "length": ceilings["maximumLengthWorstFoldAuc"],
    }
    for name, maximum in auc_mapping.items():
        value = audits[name]["maximum_fold_auc_separation"]
        checks.append({
            "name": f"{name}_maximum_fold_auc_separation",
            "value": value,
            "maximum": maximum,
            "passed": value <= maximum,
        })
    return {"passed": all(check["passed"] for check in checks), "checks": checks}


def main() -> None:
    args = parse_args()
    dataset = Path(args.dataset)
    config_path = Path(args.config)
    output_path = Path(args.output)
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    config = json.loads(config_path.read_text())
    if manifest["schema_version"] != 8 or manifest["validation"]["errors"]:
        raise RuntimeError("V8 shortcut audit requires a clean schema-8 manifest")
    if manifest["config"] != config:
        raise RuntimeError("V8 shortcut config does not match the generated manifest")
    records = canonical_records(
        read_jsonl(dataset / "records/train.jsonl") +
        read_jsonl(dataset / "records/calibration.jsonl")
    )
    mechanics = config["mechanics"]
    audits = {
        kind: audit_model(records, mechanics, kind)
        for kind in (
            "metadata",
            "unigram",
            "character_ngram",
            "relational_character_ngram",
            "length",
        )
    }
    gates = shortcut_gate_report(audits, config["shortcutGates"])
    result = {
        "schema_version": 8,
        "experiment": "v8_pre_model_shortcut_audit",
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": file_sha256(manifest_path),
        "dataset_sha256": manifest["dataset_sha256"],
        "records": len(records),
        "mechanics": mechanics,
        "audits": audits,
        "audit_scope": {
            "character_ngram": "Role labels and role-to-status bindings are scrubbed; this is the disallowed formatting shortcut gate.",
            "relational_character_ngram": "Full prompt retained as a legitimate relational baseline; reported but not treated as a shortcut leak.",
        },
        "gates": gates,
        "data_access": {
            "v3_test_records_read": 0,
            "prior_holdout_records_read": 0,
            "v7_tone_drift_records_read": 0,
            "v7_model_results_read": 0,
            "model_features_read": 0,
            "untouched_v8_mechanic_records_read": 0,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not gates["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
