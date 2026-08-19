from __future__ import annotations

import json
import unittest

from v22r2_grounding import PROJECT_ROOT
from v176_four_constraint_confirmation_population import (
    all_source_states,
    audit_population,
    build_population,
)


class V176FourConstraintConfirmationPopulationTest(unittest.TestCase):
    def test_complete_population_matches_frozen_structural_census(self) -> None:
        states = all_source_states()
        self.assertEqual(len(states), 1120)
        self.assertEqual(
            len({row["constraint_signature"] for row in states}), 1120
        )
        self.assertEqual({len(row["constraints"]) for row in states}, {4})

        V172_states = json.loads(
            (
                PROJECT_ROOT
                / "outputs/v172-trusted-shadow-integration-population/population/constraint-states.json"
            ).read_text()
        )
        V172_targets = json.loads(
            (
                PROJECT_ROOT
                / "outputs/v172-trusted-shadow-integration-population/population/target-cases.json"
            ).read_text()
        )
        config = json.loads(
            (
                PROJECT_ROOT
                / "configs/v176-four-constraint-confirmation-population.json"
            ).read_text()
        )
        population = build_population(V172_states, V172_targets)
        audit = audit_population(
            population, V172_states, V172_targets, config
        )
        self.assertTrue(audit["passed"], audit["checks"])
        self.assertEqual(audit["summary"]["confirmation_eligible_state_count"], 135)
        self.assertEqual(audit["summary"]["target_case_count"], 2160)
        self.assertEqual(
            audit["summary"]["exact_target_context_signature_overlap_with_V172"],
            0,
        )
        self.assertEqual(
            audit["summary"]["candidate_identity_overlap_with_V172_count"],
            audit["summary"]["unique_target_candidate_count"],
        )


if __name__ == "__main__":
    unittest.main()
