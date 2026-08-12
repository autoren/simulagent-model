import unittest

import numpy as np

from v18_schema import enumerate_program_hypotheses
from v20_probabilistic_grounding import (
    assignment_distribution,
    conformal_label_set,
    conformal_quantile,
    credible_hypothesis_indices,
    polarity_probabilities,
    posterior_answer,
    program_posterior,
)


class V20ProbabilisticTests(unittest.TestCase):
    def test_corrected_conformal_quantile_and_label_set(self):
        threshold = conformal_quantile([0.01, 0.10, 0.20, 0.40], 0.25)
        self.assertEqual(threshold, 0.40)
        self.assertEqual(conformal_label_set({"inactive": 0.55, "active": 0.45}, 0.50), ["inactive"])
        self.assertEqual(
            conformal_label_set({"inactive": 0.55, "active": 0.45}, 0.60),
            ["inactive", "active"],
        )

    def test_sigmoid_probabilities_sum_to_one(self):
        values = polarity_probabilities(1.75)
        self.assertAlmostEqual(sum(values.values()), 1.0)
        self.assertGreater(values["active"], values["inactive"])

    def test_assignment_distribution_respects_allowed_sets(self):
        values = assignment_distribution(("a", "b"), [
            {
                "determinant_id": "a", "allowed_values": ["inactive", "active"],
                "value_probabilities": {"inactive": 0.25, "active": 0.75},
            },
            {
                "determinant_id": "b", "allowed_values": ["active"],
                "value_probabilities": {"inactive": 0.9, "active": 0.1},
            },
        ])
        self.assertEqual(len(values), 2)
        self.assertAlmostEqual(sum(value["probability"] for value in values), 1.0)
        self.assertTrue(all(value["assignment"]["b"] for value in values))

    def test_program_posterior_and_answer_are_deterministic(self):
        determinant_ids = ("a", "b")
        hypotheses = enumerate_program_hypotheses(determinant_ids, 1)
        target_index = next(
            index for index, value in enumerate(hypotheses)
            if value.component_families == ("var",) and value.relevant_determinants == ("a",)
            and value.signature == ("transition_0", "transition_0", "transition_1", "transition_1")
        )
        traces = []
        for a in (False, True):
            for b in (False, True):
                traces.append({
                    "transition_code": "transition_1" if a else "transition_0",
                    "assignments": [{"assignment": {"a": a, "b": b}, "probability": 1.0}],
                })
        posterior = program_posterior(hypotheses, determinant_ids, traces)
        selected = credible_hypothesis_indices(hypotheses, posterior, 0.95)
        self.assertGreater(posterior[target_index], 0.0)
        self.assertIn(target_index, selected)
        answer = posterior_answer(
            hypotheses, selected,
            assignment_distribution(determinant_ids, [
                {"determinant_id": "a", "allowed_values": ["active"]},
                {"determinant_id": "b", "allowed_values": ["inactive", "active"]},
            ]),
            1,
        )
        self.assertEqual(answer["possible_transition_codes"], ["transition_1"])
        self.assertTrue(answer["identifiable"])

    def test_empty_posterior_returns_complete_vocabulary(self):
        hypotheses = enumerate_program_hypotheses(("a",), 1)
        posterior = np.zeros(len(hypotheses))
        selected = credible_hypothesis_indices(hypotheses, posterior, 0.95)
        answer = posterior_answer(hypotheses, selected, [], 1)
        self.assertEqual(answer["possible_transition_codes"], ["transition_0", "transition_1"])
        self.assertFalse(answer["identifiable"])


if __name__ == "__main__":
    unittest.main()
