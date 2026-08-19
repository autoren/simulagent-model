from __future__ import annotations

import json
import unittest

from v22r2_grounding import PROJECT_ROOT
from v165_factored_ontology_identifiability_population import candidate_universe
from v173_trusted_only_shadow_integration import build_state_plan, evaluate_target_policy


class V173TrustedOnlyShadowIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        states = json.loads(
            (PROJECT_ROOT / "outputs/v172-trusted-shadow-integration-population/population/constraint-states.json").read_text()
        )["states"]
        targets = json.loads(
            (PROJECT_ROOT / "outputs/v172-trusted-shadow-integration-population/population/target-cases.json").read_text()
        )["target_cases"]
        eligible = next(row for row in states if row["integration_eligible"])
        cls.state = eligible
        cls.target = next(row for row in targets if row["state_id"] == eligible["state_id"])
        cls.planner_config = json.loads(
            (PROJECT_ROOT / "configs/v167-exact-evidence-gathering-planner-lock.json").read_text()
        )["config_payload"]
        cls.sandbox_config = json.loads(
            (PROJECT_ROOT / "configs/v171-stateful-sandbox-sequence-confirmation-lock.json").read_text()
        )["composed_config_payload"]
        cls.universe = {row["candidate_id"]: row for row in candidate_universe()}
        cls.plan = build_state_plan(eligible["candidate_ids"], cls.universe, cls.planner_config)

    def test_one_target_two_policies_remain_behind_consensus_gate(self) -> None:
        for policy in ("no_query_bayes_terminal", "exact_bayes_adaptive"):
            result = evaluate_target_policy(
                self.state,
                self.target,
                policy,
                self.plan,
                self.universe,
                self.planner_config,
                self.sandbox_config,
            )
            self.assertEqual(result["metrics"]["false_trusted_route"], 0)
            self.assertEqual(result["metrics"]["provisional_sandbox_entry"], 0)
            self.assertEqual(result["planner_commit_authorization_count"], 0)
            self.assertTrue(result["gate_reconstructs"])
            self.assertTrue(result["sandbox_exact_final_state"])
            self.assertTrue(result["sandbox_provenance_valid"])


if __name__ == "__main__":
    unittest.main()
