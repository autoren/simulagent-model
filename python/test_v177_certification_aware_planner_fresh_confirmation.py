from __future__ import annotations

from fractions import Fraction
import json
import unittest

from v22r2_grounding import PROJECT_ROOT
from v165_factored_ontology_identifiability_population import candidate_universe
from v175_certification_aware_planner_development import (
    build_plan,
    evaluate_target_policy,
)
from v177_certification_aware_planner_fresh_confirmation import (
    build_certificate_artifact,
)


class V177CertificationAwarePlannerFreshConfirmationTest(unittest.TestCase):
    def test_one_frozen_target_two_policies_preserve_unchanged_safety_boundary(self) -> None:
        states = json.loads(
            (
                PROJECT_ROOT
                / "outputs/v176-four-constraint-confirmation-population/population/constraint-states.json"
            ).read_text()
        )
        eligible = json.loads(
            (
                PROJECT_ROOT
                / "outputs/v176-four-constraint-confirmation-population/population/confirmation-eligible-state-ids.json"
            ).read_text()
        )
        targets = json.loads(
            (
                PROJECT_ROOT
                / "outputs/v176-four-constraint-confirmation-population/population/target-cases.json"
            ).read_text()
        )
        planner_config = json.loads(
            (PROJECT_ROOT / "configs/v167-exact-evidence-gathering-planner-lock.json").read_text()
        )["config_payload"]
        sandbox_config = json.loads(
            (PROJECT_ROOT / "configs/v171-stateful-sandbox-sequence-confirmation-lock.json").read_text()
        )["V168_config_payload"]
        state_id = eligible["state_ids"][0]
        state = next(row for row in states["states"] if row["state_id"] == state_id)
        target = next(
            row for row in targets["target_cases"] if row["state_id"] == state_id
        )
        certificate = build_certificate_artifact(
            {"states": [state]}, {"target_cases": [target]}
        )["target_results"][0]
        universe = {row["candidate_id"]: row for row in candidate_universe()}
        plan = build_plan(
            state["candidate_ids"], universe, planner_config, 5, Fraction(1, 10)
        )
        results = {
            policy: evaluate_target_policy(
                target,
                policy,
                plan,
                certificate,
                universe,
                planner_config,
                sandbox_config,
                Fraction(1, 10),
            )
            for policy in ("immediate_defer", "exact_certification_adaptive")
        }
        for result in results.values():
            self.assertEqual(result["false_trusted_route"], 0)
            self.assertEqual(result["provisional_sandbox_entry"], 0)
            self.assertEqual(result["planner_commit_authorization_count"], 0)
            self.assertTrue(result["sandbox_exact"])
            self.assertTrue(result["invariants_preserved"])
            self.assertTrue(result["provenance_valid"])
            self.assertTrue(result["restart_verified"])
            self.assertTrue(result["authorized_mutations"])


if __name__ == "__main__":
    unittest.main()
