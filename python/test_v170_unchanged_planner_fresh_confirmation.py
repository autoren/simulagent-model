from __future__ import annotations

import json
import unittest

from v22r2_grounding import PROJECT_ROOT
from v170_unchanged_planner_fresh_confirmation import evaluate_fresh_population


class V170UnchangedPlannerFreshConfirmationTest(unittest.TestCase):
    def test_one_synthetic_policy_state_uses_exact_rational_risks(self) -> None:
        population_outcome = json.loads((PROJECT_ROOT / "configs/v169r1-json-key-normalization-repair-outcome-lock.json").read_text())
        states = json.loads((PROJECT_ROOT / population_outcome["constraint_states"]).read_text())
        eligible = json.loads((PROJECT_ROOT / population_outcome["eligible_state_ids"]).read_text())
        planner_lock = json.loads((PROJECT_ROOT / "configs/v167-exact-evidence-gathering-planner-lock.json").read_text())
        one_id = eligible["state_ids"][0]
        one_states = {"states": [next(row for row in states["states"] if row["state_id"] == one_id)]}
        result = evaluate_fresh_population(one_states, {"state_ids": [one_id]}, planner_lock["config_payload"])
        self.assertEqual(result["summary"]["case_count"], 1)
        self.assertEqual(result["summary"]["candidate_count_values"], [64])
        self.assertEqual(result["summary"]["class_coverage_values"], [3])
        self.assertEqual(result["summary"]["bayes_no_worse_than_every_nonoracle_baseline_case_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
