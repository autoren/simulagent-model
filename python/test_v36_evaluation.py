"""Tests for V36 metric and decision logic."""

from __future__ import annotations

import unittest

from v36_evaluation import decision_from_checks


class V36EvaluationTests(unittest.TestCase):
    def test_pass_magnitude_distinguishes_replication(self):
        checks = {name: True for name in ("predicate", "atom", "relation_order", "lexical_sign", "outer_operation", "compiled_truth", "compiled_exact_fact", "exact_scene", "worst_surface_family", "negative_composition", "pair_x")}
        self.assertEqual(decision_from_checks({"compiled_exact_fact_accuracy": .98}, checks), ("confirmation_pass_preregister_end_to_end_relational_suite", "near_v35_replication"))
        self.assertEqual(decision_from_checks({"compiled_exact_fact_accuracy": .92}, checks)[1], "gate_level_confirmation")

    def test_failure_localization(self):
        base = {name: True for name in ("predicate", "atom", "relation_order", "lexical_sign", "outer_operation", "compiled_truth", "compiled_exact_fact", "exact_scene", "worst_surface_family", "negative_composition")}
        semantic = dict(base); semantic["outer_operation"] = False
        self.assertEqual(decision_from_checks({"compiled_exact_fact_accuracy": .5}, semantic)[1], "semantic_representation_failure")
        atom = dict(base); atom["relation_order"] = False
        self.assertEqual(decision_from_checks({"compiled_exact_fact_accuracy": .5}, atom)[1], "atom_binding_failure")


if __name__ == "__main__":
    unittest.main()
