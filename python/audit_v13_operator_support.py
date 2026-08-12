#!/usr/bin/env python3
"""Audit whether V10 template holdouts preserve semantic mention-orientation support."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256, load_locked_records


SIGNATURES = ("gold_only", "opposite_only", "both", "neither")


def mention_signature(record: dict[str, Any], target: dict[str, Any]) -> str:
    hypotheses = next(
        value["statements"]
        for value in record["agent_input"]["state_hypotheses"]
        if value["determinant_id"] == target["determinant_id"]
    )
    evidence = next(
        unit["text"]
        for unit in record["evidence_units"]
        if unit["start"] == target["evidence_span"]["start"]
        and unit["end"] == target["evidence_span"]["end"]
    ).lower()
    present = [statement.lower() in evidence for statement in hypotheses]
    gold_index = 0 if target["current_value"] == "active" else 1
    if present == [True, True]:
        return "both"
    if present[gold_index]:
        return "gold_only"
    if present[1 - gold_index]:
        return "opposite_only"
    return "neither"


def audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        for target in record["target"]["determinant_grounding"]:
            if target["temporal_status"] == "CURRENT":
                counts[record["template_family"]][mention_signature(record, target)] += 1
    templates = sorted(counts)
    by_template = {
        template: {signature: counts[template][signature] for signature in SIGNATURES}
        for template in templates
    }
    holdouts = {}
    for heldout in templates:
        training = Counter()
        for template in templates:
            if template != heldout:
                training.update(counts[template])
        evaluation_signatures = [name for name in SIGNATURES if counts[heldout][name] > 0]
        unsupported = [name for name in evaluation_signatures if training[name] == 0]
        holdouts[heldout] = {
            "training_signature_counts": {name: training[name] for name in SIGNATURES},
            "evaluation_signatures": evaluation_signatures,
            "unsupported_evaluation_signatures": unsupported,
            "support_complete": not unsupported,
        }
    return {
        "current_determinants": sum(sum(value.values()) for value in counts.values()),
        "by_template": by_template,
        "template_holdouts": holdouts,
        "support_incomplete_holdouts": [
            name for name, value in holdouts.items() if not value["support_complete"]
        ],
    }


def main() -> None:
    lock_path = Path("configs/v10-frozen-lock.json")
    v13_result_path = Path("outputs/v13-token-local/evaluation/result.json")
    output_path = Path("outputs/v13-token-local/operator-support-audit.json")
    records = load_locked_records(json.loads(lock_path.read_text()))
    result = audit(records)
    result.update({
        "schema_version": 13,
        "experiment": "v13_post_result_operator_support_audit",
        "v10_frozen_lock_sha256": file_sha256(lock_path),
        "v13_result_sha256": file_sha256(v13_result_path),
        "model_outputs_used": False,
        "data_access": {
            "v3_test_records_read": 0,
            "prior_holdout_records_read": 0,
            "v7_tone_drift_records_read": 0,
            "v7_model_results_read": 0,
            "untouched_v8_mechanic_records_read": 0,
            "final_v9_mechanic_records_read": 0,
        },
    })
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
