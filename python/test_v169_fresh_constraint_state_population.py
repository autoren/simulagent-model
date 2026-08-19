from __future__ import annotations

import json
import unittest

from v22r2_grounding import PROJECT_ROOT
from v169_fresh_constraint_state_population import all_source_states, audit_population, build_population


class V169FreshConstraintStatePopulationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((PROJECT_ROOT / "configs/v169-fresh-constraint-state-population.json").read_text())
        source = json.loads((PROJECT_ROOT / cls.config["sourceV166OutcomeLock"]).read_text())
        cls.hidden = json.loads((PROJECT_ROOT / source["hidden_records"]).read_text())
        cls.population = build_population(cls.hidden, cls.config)

    def test_complete_source_enumeration(self) -> None:
        self.assertEqual(len(all_source_states()), 112)

    def test_all_selected_states_are_nonoverlapping_and_exact(self) -> None:
        audit = audit_population(self.population, self.hidden, self.config)
        self.assertTrue(audit["passed"])
        self.assertEqual(self.population["summary"]["candidate_count_values"], [64])

    def test_no_policy_scores_exist(self) -> None:
        self.assertEqual(self.population["summary"]["planner_policy_score_count"], 0)
        self.assertGreaterEqual(len(self.population["eligible_state_ids"]), 48)


if __name__ == "__main__":
    unittest.main()
