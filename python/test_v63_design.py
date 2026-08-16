#!/usr/bin/env python3
from __future__ import annotations

import json
import unittest

import numpy as np

from audit_and_freeze_v63_design import (
    ALTERNATING,
    PERSISTENT,
    feasibility,
    scaled_beta_quadrature,
    transition_for,
)
from v22r2_grounding import PROJECT_ROOT


class V63DesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(
            (PROJECT_ROOT / "configs/v63-external-unknown-dynamics.json").read_text()
        )
        cls.model = json.loads(
            (PROJECT_ROOT / cls.config["externalSource"]["sealedModel"]).read_text()
        )

    def test_scaled_beta_quadrature_normalizes_and_has_expected_mean(self) -> None:
        theta, weights = scaled_beta_quadrature(33, 0.65, 0.95)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=14)
        self.assertAlmostEqual(float(theta @ weights), 0.8, places=14)

    def test_identity_transition_semantics_are_opposite(self) -> None:
        persistent = transition_for(self.model, PERSISTENT, 0.8)
        alternating = transition_for(self.model, ALTERNATING, 0.8)
        self.assertAlmostEqual(float(persistent[0, 2, 2]), 0.8)
        self.assertAlmostEqual(float(persistent[0, 2, 3]), 0.2)
        self.assertAlmostEqual(float(alternating[0, 2, 2]), 0.2)
        self.assertAlmostEqual(float(alternating[0, 2, 3]), 0.8)
        self.assertTrue(np.array_equal(persistent[1:], alternating[1:]))

    def test_family_passes_all_preregistered_feasibility_checks(self) -> None:
        result = feasibility(self.model, self.config)
        self.assertTrue(result["passed"], result)
        self.assertGreater(result["collapse_action_disagreement_count"], 0)
        self.assertIn("listen", result["exact_action_names_seen"])
        self.assertTrue(
            {"open-left", "open-right"}.intersection(result["exact_action_names_seen"])
        )


if __name__ == "__main__":
    unittest.main()
