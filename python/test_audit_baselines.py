"""Regression tests for audits and deterministic baselines."""

from __future__ import annotations

import copy
import unittest

from audit_dataset import audit_signature, exact_prompt_signature
from evaluate_epistemic import evaluate_epistemic, valid_epistemic_schema
from evaluate_predictions import target_has_state_change, valid_schema
from run_baselines import (
    action_majority_predictions,
    empty_transition,
    prompt_lookup_predictions,
    weighted_jaccard,
)
from run_outcome_count_baselines import token_naive_bayes


def target(*, success: bool = True, changed: bool = False) -> dict:
    return {
        "blocked_actions_added": [],
        "blocked_actions_removed": [],
        "environment_changed": False,
        "flags_changed": {"doorUnlocked": True} if changed else {},
        "hidden_actions_concealed": [],
        "hidden_actions_revealed": [],
        "inventory_added": [],
        "inventory_removed": [],
        "next_location": "atrium",
        "reachable_room_delta": 0,
        "success": success,
        "visible_actions_added": [],
        "visible_actions_removed": [],
    }


def record(record_id: str, split: str, result: dict) -> dict:
    return {
        "id": record_id,
        "split": split,
        "scenario_id": f"scenario-{split}",
        "action": {"type": "inspect", "target": "room"},
        "agent_input": {
            "candidate_action": {"key": "inspect:room", "label": "inspect room"},
            "observation": {"location": "atrium"},
        },
        "target": result,
    }


class AuditTests(unittest.TestCase):
    def test_detects_identical_prompt_with_conflicting_targets(self) -> None:
        left = record("left", "train", target(success=True))
        right = record("right", "test", target(success=False))
        right["agent_input"] = copy.deepcopy(left["agent_input"])
        report = audit_signature([left, right], exact_prompt_signature)
        self.assertEqual(report["ambiguous_signature_groups"], 1)
        self.assertEqual(report["cross_split_signature_groups"], 1)
        self.assertEqual(report["signature_limited_exact_match_upper_bound"], 0.5)

    def test_distinguishes_state_change_from_success(self) -> None:
        self.assertFalse(target_has_state_change(target(success=True)))
        self.assertTrue(target_has_state_change(target(success=True, changed=True)))


class BaselineTests(unittest.TestCase):
    def test_empty_transition_is_schema_valid(self) -> None:
        prediction = empty_transition(record("one", "test", target()), True)
        self.assertTrue(valid_schema(prediction))

    def test_action_majority_uses_training_mode(self) -> None:
        train = [
            record("one", "train", target(success=False)),
            record("two", "train", target(success=True)),
            record("three", "train", target(success=True)),
        ]
        prediction = action_majority_predictions(
            train, [record("held-out", "test", target(success=False))]
        )[0]["prediction"]
        self.assertTrue(prediction["success"])

    def test_prompt_lookup_uses_exact_training_prompt(self) -> None:
        train = [record("one", "train", target(success=False))]
        held_out = record("held-out", "test", target(success=True))
        prediction = prompt_lookup_predictions(train, [held_out])[0]
        self.assertTrue(prediction["lookup_hit"])
        self.assertFalse(prediction["prediction"]["success"])

    def test_weighted_jaccard(self) -> None:
        self.assertEqual(weighted_jaccard({"a": 2.0}, {"a": 2.0}), 1.0)
        self.assertEqual(weighted_jaccard({"a": 2.0}, {"b": 2.0}), 0.0)

    def test_token_naive_bayes_uses_visible_input_signal(self) -> None:
        identifiable = {
            "id": "identifiable-train",
            "agent_input": {"observation": {"description": "calm calm stable"}},
            "target": {"identifiable": True, "possible_outcomes": [target()]},
        }
        ambiguous = {
            "id": "ambiguous-train",
            "agent_input": {"observation": {"description": "storm storm shifting"}},
            "target": {
                "identifiable": False,
                "possible_outcomes": [target(), target(changed=True)],
            },
        }
        held_out = {
            **ambiguous,
            "id": "held-out",
            "agent_input": {"observation": {"description": "storm shifting"}},
        }
        prediction = token_naive_bayes([identifiable, ambiguous], [held_out])[0]
        self.assertEqual(prediction["prediction"]["outcome_count"], 2)

    def test_epistemic_schema_requires_identifiability_to_match_outcomes(self) -> None:
        self.assertTrue(
            valid_epistemic_schema(
                {"identifiable": True, "possible_outcomes": [target()]}
            )
        )
        self.assertFalse(
            valid_epistemic_schema(
                {"identifiable": False, "possible_outcomes": [target()]}
            )
        )

    def test_epistemic_schema_rejects_duplicate_outcomes(self) -> None:
        self.assertFalse(
            valid_epistemic_schema(
                {"identifiable": False, "possible_outcomes": [target(), target()]}
            )
        )

    def test_missing_epistemic_prediction_counts_expected_outcomes_as_false_negatives(self) -> None:
        report = evaluate_epistemic(
            [
                {
                    "id": "missing",
                    "target": {
                        "identifiable": True,
                        "possible_outcomes": [target()],
                    },
                }
            ],
            [],
        )
        self.assertEqual(report["coverage"], 0.0)
        self.assertEqual(report["outcome_set_micro"]["fn"], 1)
        self.assertEqual(report["by_identifiability"]["identifiable"]["coverage"], 0.0)


if __name__ == "__main__":
    unittest.main()
