from __future__ import annotations

from fractions import Fraction
import json
import unittest

from v22r2_grounding import PROJECT_ROOT
from v179_triple_repetition_robust_feasibility import evaluate_feasibility


class V179TripleRepetitionRobustFeasibilityTest(unittest.TestCase):
    def test_one_state_repetition_code_is_exact_safe_and_structurally_bounded(self) -> None:
        states_all = json.loads(
            (
                PROJECT_ROOT
                / "outputs/v176-four-constraint-confirmation-population/population/constraint-states.json"
            ).read_text()
        )["states"]
        eligible_ids = json.loads(
            (
                PROJECT_ROOT
                / "outputs/v176-four-constraint-confirmation-population/population/confirmation-eligible-state-ids.json"
            ).read_text()
        )["state_ids"]
        targets_all = json.loads(
            (
                PROJECT_ROOT
                / "outputs/v176-four-constraint-confirmation-population/population/target-cases.json"
            ).read_text()
        )["target_cases"]
        planner_config = json.loads(
            (PROJECT_ROOT / "configs/v167-exact-evidence-gathering-planner-lock.json").read_text()
        )["config_payload"]
        state_id = eligible_ids[0]
        state = next(row for row in states_all if row["state_id"] == state_id)
        targets = [row for row in targets_all if row["state_id"] == state_id]
        evaluation = evaluate_feasibility(
            {"states": [state]},
            {"state_ids": [state_id]},
            {"target_cases": targets},
            planner_config,
            [0, 1, 2, 3, 4],
        )
        summary = evaluation["summary"]
        self.assertEqual(summary["state_count"], 1)
        self.assertEqual(summary["target_count"], 16)
        self.assertEqual(summary["full_measurement_adversarial_scenario_count"], 208)
        self.assertEqual(summary["majority_decode_exactness_rate"], 1.0)
        self.assertEqual(summary["robust_clean_version_space_equivalence_rate"], 1.0)
        self.assertEqual(summary["certificate_validity_rate"], 1.0)
        self.assertEqual(summary["certificate_minimality_rate"], 1.0)
        self.assertEqual(summary["horizon_monotonicity_rate"], 1.0)
        self.assertEqual(summary["adaptive_no_greater_than_target_informed_rate"], 1.0)
        self.assertEqual(summary["false_trusted_route_probability"], 0.0)
        adaptive = summary[
            "adaptive_worst_case_trusted_completion_by_block_horizon"
        ]
        upper = summary[
            "target_informed_trusted_upper_bound_by_block_horizon"
        ]
        for horizon in range(5):
            left = Fraction(
                adaptive[str(horizon)]["numerator"],
                adaptive[str(horizon)]["denominator"],
            )
            right = Fraction(
                upper[str(horizon)]["numerator"],
                upper[str(horizon)]["denominator"],
            )
            self.assertLessEqual(left, right)


if __name__ == "__main__":
    unittest.main()
