from __future__ import annotations

from fractions import Fraction
import json
import unittest

from v22r2_grounding import PROJECT_ROOT
from v165_factored_ontology_identifiability_population import candidate_universe
from v180_triple_repetition_robust_planner_development import (
    build_plan,
    evaluate_target_policy,
)


class V180TripleRepetitionRobustPlannerDevelopmentTest(unittest.TestCase):
    def test_one_target_two_policies_preserve_cost_and_safety_boundaries(self) -> None:
        states = json.loads(
            (
                PROJECT_ROOT
                / "outputs/v176-four-constraint-confirmation-population/population/constraint-states.json"
            ).read_text()
        )["states"]
        eligible = json.loads(
            (
                PROJECT_ROOT
                / "outputs/v176-four-constraint-confirmation-population/population/confirmation-eligible-state-ids.json"
            ).read_text()
        )["state_ids"]
        targets = json.loads(
            (
                PROJECT_ROOT
                / "outputs/v176-four-constraint-confirmation-population/population/target-cases.json"
            ).read_text()
        )["target_cases"]
        certificates = json.loads(
            (
                PROJECT_ROOT
                / "outputs/v179-triple-repetition-robust-feasibility/census/target-certificate-results.json"
            ).read_text()
        )["target_results"]
        planner_config = json.loads(
            (PROJECT_ROOT / "configs/v167-exact-evidence-gathering-planner-lock.json").read_text()
        )["config_payload"]
        sandbox_config = json.loads(
            (PROJECT_ROOT / "configs/v171-stateful-sandbox-sequence-confirmation-lock.json").read_text()
        )["V168_config_payload"]
        state_id = eligible[0]
        state = next(row for row in states if row["state_id"] == state_id)
        target = next(row for row in targets if row["state_id"] == state_id)
        certificate = next(
            row for row in certificates if row["target_case_id"] == target["target_case_id"]
        )
        universe = {row["candidate_id"]: row for row in candidate_universe()}
        plan = build_plan(
            state["candidate_ids"],
            universe,
            planner_config,
            4,
            Fraction(3, 10),
            Fraction(1, 10),
        )
        results = {
            policy: evaluate_target_policy(
                target,
                policy,
                plan,
                certificate,
                universe,
                sandbox_config,
                Fraction(3, 10),
            )
            for policy in ("immediate_defer", "exact_robust_certification_adaptive")
        }
        for result in results.values():
            self.assertEqual(result["expected_raw_inspections"], 3 * result["expected_blocks"])
            self.assertEqual(result["false_trusted_route"], 0)
            self.assertEqual(result["provisional_sandbox_entry"], 0)
            self.assertEqual(result["planner_commit_authorization_count"], 0)
            self.assertTrue(result["corruption_route_invariant"])
            self.assertTrue(result["sandbox_exact"])
            self.assertTrue(result["invariants_preserved"])
            self.assertTrue(result["provenance_valid"])
            self.assertTrue(result["restart_verified"])


if __name__ == "__main__":
    unittest.main()
