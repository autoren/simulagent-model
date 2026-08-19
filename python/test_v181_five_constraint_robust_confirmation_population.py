from __future__ import annotations

import json
import unittest

from v22r2_grounding import PROJECT_ROOT
from v181_five_constraint_robust_confirmation_population import (
    all_source_states,
    audit_population,
    build_population,
)


def load_source() -> tuple[dict, dict, dict, dict, dict]:
    V172_root = (
        PROJECT_ROOT
        / "outputs/v172-trusted-shadow-integration-population/population"
    )
    V176_root = (
        PROJECT_ROOT
        / "outputs/v176-four-constraint-confirmation-population/population"
    )
    return (
        json.loads((V172_root / "constraint-states.json").read_text()),
        json.loads((V172_root / "target-cases.json").read_text()),
        json.loads((V176_root / "constraint-states.json").read_text()),
        json.loads((V176_root / "target-cases.json").read_text()),
        json.loads(
            (
                PROJECT_ROOT
                / "configs/v181-five-constraint-robust-confirmation-population.json"
            ).read_text()
        ),
    )


class V181FiveConstraintRobustConfirmationPopulationTest(unittest.TestCase):
    def test_complete_population_matches_frozen_structural_census(self) -> None:
        states = all_source_states()
        self.assertEqual(len(states), 1792)
        self.assertEqual(
            len({row["constraint_signature"] for row in states}), 1792
        )
        self.assertEqual({len(row["constraints"]) for row in states}, {5})

        V172_states, V172_targets, V176_states, V176_targets, config = load_source()
        population = build_population(
            V172_states, V172_targets, V176_states, V176_targets
        )
        audit = audit_population(
            population,
            V172_states,
            V172_targets,
            V176_states,
            V176_targets,
            config,
        )
        self.assertTrue(audit["passed"], audit["checks"])
        self.assertEqual(
            audit["summary"]["confirmation_eligible_state_count"], 66
        )
        self.assertEqual(audit["summary"]["target_case_count"], 528)
        self.assertEqual(
            audit["summary"][
                "exact_target_context_signature_overlap_with_V172"
            ],
            0,
        )
        self.assertEqual(
            audit["summary"][
                "exact_target_context_signature_overlap_with_V176"
            ],
            0,
        )
        self.assertEqual(
            audit["summary"]["candidate_identity_overlap_with_V172_count"],
            audit["summary"]["unique_target_candidate_count"],
        )
        self.assertEqual(
            audit["summary"]["candidate_identity_overlap_with_V176_count"],
            audit["summary"]["unique_target_candidate_count"],
        )


if __name__ == "__main__":
    unittest.main()
