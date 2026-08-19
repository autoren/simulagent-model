import json
import unittest
from pathlib import Path

from v150_oracle_closed_interaction_policy import build_episodes, evaluate, exact_query_plan


class V150OracleClosedInteractionPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(Path("configs/v150-oracle-closed-interaction-policy.json").read_text())
        cls.witness_config = json.loads(Path("configs/v149-fresh-closed-interaction-population.json").read_text())
        cls.catalog = json.loads(Path("outputs/v149-fresh-closed-interaction-population/design/interaction-catalog.json").read_text())
        hidden = json.loads(Path("outputs/v149-fresh-closed-interaction-population/design/hidden-fixtures.json").read_text())
        allowed = {
            "fixture_id", "split", "group_id", "family_id", "stage", "truth_state_id",
            "compatible_state_ids", "oracle_query_id", "closed_answer_event",
            "presented_candidate_choice_id",
        }
        cls.development = [
            {key: row[key] for key in allowed}
            for row in hidden
            if row["split"] == "development"
        ]

    def test_builds_two_episodes_per_development_group(self):
        episodes = build_episodes(self.development)
        self.assertEqual(len(episodes), 48)
        self.assertEqual(len({row["group_id"] for row in episodes}), 24)

    def test_exact_planner_selects_registered_discriminating_query(self):
        combined = self.witness_config | self.config
        for episode in build_episodes(self.development):
            plan = exact_query_plan(episode, self.catalog, combined)
            self.assertEqual(plan["selected_action"], episode["oracle_query_id"])
            self.assertEqual(plan["selected_expected_cost"], 0.3)
            self.assertEqual(plan["no_query_cost"], 1.0)

    def test_full_oracle_policy_passes_all_gates(self):
        result = evaluate(self.development, self.catalog, self.witness_config, self.config)
        self.assertTrue(result["passed"], result)
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(result["metrics"]["policy_evaluation_count"], 2352)
        self.assertEqual(result["metrics"]["final_exact_accuracy"], 1.0)
        self.assertEqual(result["metrics"]["candidate_and_query_proposal_invariance"], 1.0)


if __name__ == "__main__":
    unittest.main()
