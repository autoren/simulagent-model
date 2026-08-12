"""Correct V19's non-gating grounding-error buckets by joining determinant ids."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from audit_v18_benchmark import read_records
from evaluate_v19_frozen_integration import condition_modes, oracle_support, predicted_support
from run_v18_schema_baselines import (
    compatible_assignment_indices, outcome_vocabulary, version_space_answer,
)
from v10_protocol import file_sha256
from v18_schema import allowed_trace_consistent_hypotheses, enumerate_program_hypotheses


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def grounding_errors(predicted: Sequence[dict[str, Any]], oracle: Sequence[dict[str, Any]]) -> int:
    predicted_by_id = {value["determinant_id"]: value["allowed_values"] for value in predicted}
    oracle_by_id = {value["determinant_id"]: value["allowed_values"] for value in oracle}
    if set(predicted_by_id) != set(oracle_by_id):
        raise ValueError("Predicted and oracle determinant ids differ")
    return sum(predicted_by_id[key] != oracle_by_id[key] for key in oracle_by_id)


def bucket(errors: int) -> str:
    return "zero" if errors == 0 else "one" if errors == 1 else "multiple"


def replay_condition(
    episodes: Sequence[dict[str, Any]], lookup: dict[tuple[str, str], dict[str, Any]],
    support_mode: str, query_mode: str,
) -> dict[str, Any]:
    rows = []
    for episode in episodes:
        determinant_ids = tuple(value["id"] for value in episode["agent_input"]["determinant_ontology"])
        outcome_bits = episode["agent_input"]["dsl_contract"]["outcome_bits"]
        target_signature = tuple(episode["target"]["behavioral_signature"])
        support_oracle = oracle_support(episode)
        support_predicted = predicted_support(episode, lookup)
        support = support_oracle if support_mode == "oracle" else support_predicted
        support_errors = (
            0 if support_mode == "oracle" else sum(
                grounding_errors(predicted["allowed_values"], oracle["allowed_values"])
                for predicted, oracle in zip(support_predicted, support_oracle, strict=True)
            )
        )
        version_space = list(enumerate_program_hypotheses(determinant_ids, outcome_bits))
        for trace in support:
            version_space = allowed_trace_consistent_hypotheses(
                version_space, [trace], determinant_ids
            )
        target_retained = any(value.signature == target_signature for value in version_space)
        empty = not version_space
        exact = []
        query_errors = 0
        for query in episode["oracle_grounding"]["queries"]:
            if query_mode == "oracle":
                allowed = query["allowed_values"]
            else:
                allowed = lookup[(episode["id"], query["query_id"])]["groundings"]
                query_errors += grounding_errors(allowed, query["allowed_values"])
            if empty:
                possible = outcome_vocabulary(outcome_bits)
            else:
                indices = compatible_assignment_indices(determinant_ids, allowed)
                possible = version_space_answer(version_space, indices)["possible_transition_codes"]
            exact.append(possible == query["possible_transition_codes"])
        rows.append({
            "episode_id": episode["id"],
            "axis": episode["generalization_axis"],
            "support_errors": support_errors,
            "query_errors": query_errors,
            "query_accuracy": float(np.mean(exact)),
            "complete": all(exact),
            "target_retained": target_retained,
            "empty": empty,
        })
    conditioned = {}
    for name in ("zero", "one", "multiple"):
        values = [value for value in rows if bucket(value["support_errors"]) == name]
        conditioned[name] = {
            "episodes": len(values),
            "episode_macro_transition_set_exact_match": (
                float(np.mean([value["query_accuracy"] for value in values])) if values else None
            ),
            "complete_episodes": sum(value["complete"] for value in values),
            "target_retention_rate": (
                float(np.mean([value["target_retained"] for value in values])) if values else None
            ),
            "empty_version_space_rate": (
                float(np.mean([value["empty"] for value in values])) if values else None
            ),
        }
    return {
        "episodes": len(rows),
        "support_grounding_error_histogram": dict(sorted(Counter(value["support_errors"] for value in rows).items())),
        "query_grounding_error_histogram": dict(sorted(Counter(value["query_errors"] for value in rows).items())),
        "conditioned_on_support_grounding_errors": conditioned,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v19-frozen-integration-lock.json")
    parser.add_argument("--result", default="outputs/v19-frozen-integration/evaluation/result.json")
    parser.add_argument("--output", default="outputs/v19-frozen-integration/error-conditioning-replay.json")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.lock).resolve()
    result_path = (PROJECT_ROOT / args.result).resolve()
    lock = json.loads(lock_path.read_text())
    result = json.loads(result_path.read_text())
    prediction_path = PROJECT_ROOT / result["grounding_predictions"]
    if file_sha256(prediction_path) != result["grounding_predictions_sha256"]:
        raise ValueError("V19 grounding predictions changed before error-conditioning replay")
    predictions = [json.loads(line) for line in prediction_path.read_text().splitlines() if line]
    episodes = [
        value for value in read_records(PROJECT_ROOT / lock["source"]["v18_dataset"])
        if value["split"] == lock["primary_split"]
    ]
    views = {}
    for view in lock["views"]:
        lookup = {
            (value["episode_id"], value["source_item_id"]): value
            for value in predictions if value["view"] == view
        }
        views[view] = {}
        for condition in lock["conditions"]:
            support_mode, query_mode = condition_modes(condition)
            views[view][condition] = replay_condition(
                episodes, lookup, support_mode, query_mode
            )
    supported_full = views["supported"]["frozen_support_frozen_query"]
    novel_full = views["novel_ontology"]["frozen_support_frozen_query"]
    if supported_full["support_grounding_error_histogram"] != {0: 40}:
        raise ValueError("Corrected supported-view support errors are not all zero")
    output = {
        "schema_version": 19,
        "experiment": "v19_scope_correct_grounding_error_conditioning_replay",
        "passed": True,
        "primary_decision_unchanged": result["decision"],
        "primary_metrics_affected": False,
        "superseded_locked_result_fields": [
            "views.*.conditions.*.support_grounding_errors",
            "views.*.conditions.*.query_grounding_errors",
            "views.*.conditions.*.conditioned_on_support_grounding_errors",
        ],
        "cause": (
            "The locked diagnostic zipped ontology-ordered predictions with alphabetically serialized oracle assignments. "
            "Schema search keyed both by determinant id, so grounding, version spaces, query answers, gates, and the decision were unaffected."
        ),
        "views": views,
        "sanity": {
            "supported_zero_error_episodes": supported_full["conditioned_on_support_grounding_errors"]["zero"]["episodes"],
            "novel_zero_error_episodes": novel_full["conditioned_on_support_grounding_errors"]["zero"]["episodes"],
        },
        "source": {
            "protocol_lock_sha256": file_sha256(lock_path),
            "result_sha256": file_sha256(result_path),
            "grounding_predictions_sha256": file_sha256(prediction_path),
        },
        "data_access": {
            "new_model_forward_passes": 0,
            "new_linear_fits": 0,
            "adapter_training_runs": 0,
            "v17_records_read": 0,
            "v17_model_results_read": 0,
        },
    }
    output_path = (PROJECT_ROOT / args.output).resolve()
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "passed": output["passed"],
        "primary_decision_unchanged": output["primary_decision_unchanged"],
        "sanity": output["sanity"],
        "novel_full": novel_full,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
