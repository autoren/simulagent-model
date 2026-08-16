"""Tests for V34 operation-focused interface logic."""

from __future__ import annotations

import unittest

import numpy as np

from v34_operation import (
    cross_validate_ridge, operation_prompt, select_alpha, select_prompt_method,
)


class V34OperationTests(unittest.TestCase):
    def test_prompt_contains_definitions_but_not_target_fields(self):
        config = {
            "operationInterface": {
                "classes": ["assert", "deny"], "labelTokens": ["A", "B"],
                "definitions": {"assert": "direct", "deny": "reject"},
            }
        }
        row = {
            "agent_input": {"evidence_text": "The record rejects that mavo is stable."},
            "target": {"predicate": "stable", "arguments": ["mavo"],
                       "truth_status": "false", "factorization": {"outer_operation": "deny"}},
        }
        prompt = operation_prompt(row, config)
        self.assertIn("The record rejects", prompt)
        self.assertIn("A: direct", prompt)
        self.assertNotIn("stable\n", prompt)
        self.assertNotIn("false", prompt)

    def test_group_cv_and_selection_are_deterministic(self):
        features = np.asarray([[-2], [-1], [1], [2], [-3], [3], [-4], [4]], dtype=float)
        targets = np.asarray([0, 0, 1, 1, 0, 1, 0, 1])
        groups = np.asarray(["a", "a", "a", "a", "b", "b", "c", "c"])
        reports = cross_validate_ridge(features, targets, groups, [0.1, 10.0])
        selected = select_alpha(reports)
        self.assertEqual(len(reports), 2)
        self.assertIn(selected["alpha"], (0.1, 10.0))
        self.assertEqual(len(selected["folds"]), 3)

    def test_prompt_method_selection_uses_fit_cv_only(self):
        methods = {
            "semanticHiddenRidge": {"selected_cv": {"mean_group_cv_operation_accuracy": .9, "minimum_group_cv_operation_accuracy": .8, "alpha": 10}},
            "nativeLogitRidge": {"selected_cv": {"mean_group_cv_operation_accuracy": .8, "minimum_group_cv_operation_accuracy": .8, "alpha": 100}},
        }
        self.assertEqual(select_prompt_method(methods), "semanticHiddenRidge")


if __name__ == "__main__":
    unittest.main()
