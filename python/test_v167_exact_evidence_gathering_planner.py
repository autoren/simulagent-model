from __future__ import annotations

import json
import unittest

from v22r2_grounding import PROJECT_ROOT
from v167_exact_evidence_gathering_planner import build_planner_evaluation, evaluate_gates


class V167ExactEvidenceGatheringPlannerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.parent = json.loads((PROJECT_ROOT / "configs/v166-model-free-factored-ontology-baselines-outcome-lock.json").read_text())
        cls.config = json.loads((PROJECT_ROOT / "configs/v167-exact-evidence-gathering-planner.json").read_text())
        predictions = json.loads((PROJECT_ROOT / cls.parent["baseline_predictions"]).read_text())
        hidden = json.loads((PROJECT_ROOT / cls.parent["hidden_records"]).read_text())
        cls.evaluation = build_planner_evaluation(predictions, hidden, cls.config)

    def test_frozen_ambiguous_case_contract(self) -> None:
        summary = self.evaluation["summary"]
        self.assertEqual(summary["case_count"], 48)
        self.assertEqual(summary["candidate_count_values"], [64])
        self.assertEqual(summary["class_coverage_values"], [3])
        self.assertEqual(summary["target_candidate_retention"], 1.0)

    def test_information_has_positive_value(self) -> None:
        summary = self.evaluation["summary"]
        self.assertEqual(summary["positive_value_of_information_case_count"], 48)
        self.assertGreater(summary["history_dependent_second_action_case_count"], 0)
        self.assertGreater(len(summary["unique_exact_bayes_root_queries"]), 1)

    def test_adaptation_strictly_improves_on_open_loop(self) -> None:
        summary = self.evaluation["summary"]
        self.assertGreater(summary["strict_improvement_over_optimal_open_loop_case_count"], 0)
        self.assertEqual(summary["bayes_no_worse_than_every_nonoracle_baseline_case_rate"], 1.0)

    def test_all_preregistered_gates_pass(self) -> None:
        access = {
            "evaluation_record_count": 0, "manual_judgment_count": 0,
            "model_load_count": 0, "model_generation_count": 0, "API_call_count": 0,
            "training_run_count": 0, "ontology_registration_count": 0,
            "trusted_state_mutation_count": 0, "real_service_call_count": 0,
            "external_side_effect_count": 0, "actual_execution_count": 0,
        }
        self.assertTrue(all(evaluate_gates(self.evaluation, access, self.config).values()))


if __name__ == "__main__":
    unittest.main()
