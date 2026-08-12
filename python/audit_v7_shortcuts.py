#!/usr/bin/env python3
"""Reject V7 before model access if metadata or evidence wording predicts the label."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from binary_metrics import evaluate_binary, fit_threshold


Record = dict[str, Any]
TOKEN_PATTERN = re.compile(r"[a-z0-9_:+.-]+")
EVIDENCE_PREFIXES = (
    "causal audit:",
    "direct-evidence rule:",
    "upstream-evidence rule:",
    "consequence rule:",
    "procedure rule:",
    "observability rule:",
    "causal comparison:",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/v7")
    parser.add_argument("--config", default="configs/dataset.v7.json")
    parser.add_argument("--output", default="outputs/v7-pre-model/shortcut-audit.json")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[Record]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_key(record: Record) -> tuple[str, ...]:
    return (
        record["mechanic"],
        record["evidence_variant"],
        record["action_template"],
        record["surface_variant"],
    )


def train_metadata_lookup(records: list[Record]) -> dict[tuple[str, ...], float]:
    labels: dict[tuple[str, ...], list[bool]] = defaultdict(list)
    for record in records:
        labels[metadata_key(record)].append(record["target"]["ambiguous"])
    return {
        key: math.log((sum(values) + 1.0) / (len(values) - sum(values) + 1.0))
        for key, values in labels.items()
    }


def score_metadata(model: dict[tuple[str, ...], float], records: list[Record]) -> list[float]:
    return [model.get(metadata_key(record), 0.0) for record in records]


def evidence_text(record: Record) -> str:
    observation = record["agent_input"]["observation"]
    fields = [*observation.get("beliefs", []), *observation.get("memories", [])]
    return " ".join(
        value for value in fields if value.lower().startswith(EVIDENCE_PREFIXES)
    )


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def train_token_nb(records: list[Record]) -> dict[str, Any]:
    counts: dict[bool, Counter[str]] = {False: Counter(), True: Counter()}
    totals = {False: 0, True: 0}
    vocabulary: set[str] = set()
    for record in records:
        label = record["target"]["ambiguous"]
        tokens = tokenize(evidence_text(record))
        counts[label].update(tokens)
        totals[label] += len(tokens)
        vocabulary.update(tokens)
    return {"counts": counts, "totals": totals, "vocabulary": vocabulary}


def score_token_nb(model: dict[str, Any], records: list[Record]) -> list[float]:
    vocabulary = model["vocabulary"]
    smoothing = 1.0
    denominators = {
        label: model["totals"][label] + smoothing * max(1, len(vocabulary))
        for label in (False, True)
    }
    result = []
    for record in records:
        tokens = Counter(tokenize(evidence_text(record)))
        scores = {}
        for label in (False, True):
            scores[label] = sum(
                count
                * math.log(
                    (model["counts"][label][token] + smoothing) / denominators[label]
                )
                for token, count in tokens.items()
            )
        result.append(scores[True] - scores[False])
    return result


def evaluate_baseline(
    train: list[Record], calibration: list[Record], train_scores: list[float], calibration_scores: list[float]
) -> dict[str, Any]:
    train_gold = [record["target"]["ambiguous"] for record in train]
    calibration_gold = [record["target"]["ambiguous"] for record in calibration]
    threshold = fit_threshold(train_gold, train_scores)["threshold"]
    return {
        "threshold_selection_split": "train",
        "threshold": threshold,
        "train": evaluate_binary(train_gold, train_scores, threshold),
        "calibration": evaluate_binary(calibration_gold, calibration_scores, threshold),
        "distinct_calibration_scores": len(set(calibration_scores)),
    }


def gate_report(config: Record, metadata: Record, evidence: Record) -> Record:
    gates = config["shortcutGates"]
    checks = [
        {
            "name": "metadata_balanced_accuracy",
            "value": metadata["calibration"]["balanced_accuracy"],
            "maximum": gates["maximumMetadataBalancedAccuracy"],
        },
        {
            "name": "evidence_text_balanced_accuracy",
            "value": evidence["calibration"]["balanced_accuracy"],
            "maximum": gates["maximumEvidenceTextBalancedAccuracy"],
        },
        {
            "name": "evidence_text_auc",
            "value": evidence["calibration"]["roc_auc"],
            "maximum": gates["maximumEvidenceTextAuc"],
        },
    ]
    for check in checks:
        check["passed"] = check["value"] <= check["maximum"]
    return {"passed": all(check["passed"] for check in checks), "checks": checks}


def main() -> None:
    args = parse_args()
    root = Path(args.dataset)
    config_path = Path(args.config)
    manifest_path = root / "manifest.json"
    train_path = root / "records/train.jsonl"
    calibration_path = root / "records/calibration.jsonl"
    config = json.loads(config_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema_version"] != 7 or manifest["validation"]["errors"]:
        raise RuntimeError("V7 shortcut audit requires a clean schema-7 corpus")
    if manifest["validation"]["maximum_conditional_label_gap"] > config["maximumConditionalLabelGap"]:
        raise RuntimeError("V7 conditional label-gap gate failed before shortcut audit")
    train = read_jsonl(train_path)
    calibration = read_jsonl(calibration_path)
    metadata_model = train_metadata_lookup(train)
    metadata = evaluate_baseline(
        train,
        calibration,
        score_metadata(metadata_model, train),
        score_metadata(metadata_model, calibration),
    )
    text_model = train_token_nb(train)
    evidence = evaluate_baseline(
        train,
        calibration,
        score_token_nb(text_model, train),
        score_token_nb(text_model, calibration),
    )
    gates = gate_report(config, metadata, evidence)
    report = {
        "schema_version": 7,
        "experiment": "v7_pre_model_shortcut_rejection",
        "dataset_sha256": manifest["dataset_sha256"],
        "manifest_sha256": file_sha256(manifest_path),
        "record_sha256": {
            "train": file_sha256(train_path),
            "calibration": file_sha256(calibration_path),
        },
        "examples": {"train": len(train), "calibration": len(calibration)},
        "conditional_label_gap": manifest["validation"]["maximum_conditional_label_gap"],
        "metadata_lookup": metadata,
        "evidence_text_naive_bayes": evidence,
        "gates": gates,
        "data_access": {
            "v3_test_records_read": 0,
            "prior_holdout_records_read": 0,
            "untouched_mechanic_records_read": 0,
            "model_features_read": 0,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not gates["passed"]:
        raise RuntimeError("V7 pre-model shortcut rejection gate failed")


if __name__ == "__main__":
    main()
