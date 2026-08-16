"""Tests for V35 binding and assembly helpers."""

from __future__ import annotations

import unittest

import numpy as np

from v35_binding import atom_prompt_layout, fixed_gaussian_projection, select_report


class V35BindingTests(unittest.TestCase):
    def test_prompt_is_target_independent_and_spans_only_evidence_mentions(self):
        config = {
            "atomInterface": {"predicateClasses": ["stable"], "predicateLabelTokens": ["A"]},
            "v32_config": {"ontology": {"unaryPredicates": [{"id": "stable", "entityType": "unit", "trueForm": "{entity} stable", "falseForm": "{entity} unstable"}], "relations": []}},
        }
        row = {"agent_input": {"entities": [{"id": "mavo", "entity_type": "unit"}, {"id": "noru", "entity_type": "unit"}], "evidence_text": "The record describes mavo."}, "target": {"sentinel": 1}}
        content, spans = atom_prompt_layout(row, config)
        self.assertIn("mavo", content)
        self.assertEqual(len(spans["mavo"]), 1)
        self.assertEqual(spans["noru"], [])

    def test_projection_is_deterministic(self):
        values = np.arange(20, dtype=np.float32).reshape(4, 5)
        first = fixed_gaussian_projection(values, 3, 7)
        second = fixed_gaussian_projection(values, 3, 7)
        np.testing.assert_array_equal(first, second)

    def test_report_selection_prefers_regularization_on_tie(self):
        reports = [
            {"alpha": 1.0, "mean_group_cv_primary_accuracy": .9, "minimum_group_cv_primary_accuracy": .8},
            {"alpha": 100.0, "mean_group_cv_primary_accuracy": .9, "minimum_group_cv_primary_accuracy": .8},
        ]
        self.assertEqual(select_report(reports)["alpha"], 100.0)


if __name__ == "__main__":
    unittest.main()
