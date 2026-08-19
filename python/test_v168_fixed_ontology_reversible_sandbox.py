from __future__ import annotations

import json
import unittest

from v22r2_grounding import PROJECT_ROOT
from v168_fixed_ontology_reversible_sandbox import (
    SandboxStore, build_fixtures, evaluate_census, evaluate_gates, initial_state,
    proposal_for, run_fixture,
)


class V168FixedOntologyReversibleSandboxTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((PROJECT_ROOT / "configs/v168-fixed-ontology-reversible-sandbox.json").read_text())

    def test_preview_does_not_mutate_and_commit_matches(self) -> None:
        state = initial_state(0)
        proposal = proposal_for("valid_retain", 0, state)
        store = SandboxStore(state, self.config)
        preview = store.preview(proposal)
        self.assertEqual(store.state, state)
        commit = store.commit(proposal, preview)
        self.assertTrue(commit["committed"])
        self.assertEqual(store.state, preview["expected_post_state"])

    def test_invalid_and_fault_conditions_fail_closed(self) -> None:
        fixtures = {row["scenario"]: row for row in build_fixtures(self.config) if row["record_id"]}
        for scenario in ("unauthorized_field", "stale_revision", "malformed_type", "contradictory_duplicate_patch", "unknown_entity", "preview_token_tamper"):
            result = run_fixture(fixtures[scenario], self.config)
            self.assertEqual(result["disposition"], "rejected")
            self.assertTrue(result["exact_final_target_state"])

    def test_corruption_is_detected_and_rolled_back(self) -> None:
        fixture = next(row for row in build_fixtures(self.config) if row["scenario"] == "post_commit_corruption")
        result = run_fixture(fixture, self.config)
        self.assertEqual(result["disposition"], "rolled_back_after_verification_failure")
        self.assertTrue(result["fault_detected"])
        self.assertTrue(result["rollback_recovered"])

    def test_full_census_meets_gates(self) -> None:
        evaluation = evaluate_census(build_fixtures(self.config), self.config)
        access = {
            "evaluation_record_count": 0, "manual_judgment_count": 0,
            "model_load_count": 0, "model_generation_count": 0, "API_call_count": 0,
            "training_run_count": 0, "provisional_ontology_use_count": 0,
            "real_service_call_count": 0, "external_side_effect_count": 0, "real_execution_count": 0,
        }
        self.assertTrue(all(evaluate_gates(evaluation, access, self.config).values()))


if __name__ == "__main__":
    unittest.main()
