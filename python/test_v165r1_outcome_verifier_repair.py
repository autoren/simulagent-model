from __future__ import annotations

import unittest

from v165r1_outcome_verifier_repair import (
    sole_population_build_count_alias_mismatch,
    without_population_build_count,
)


class V165r1OutcomeVerifierRepairTests(unittest.TestCase):
    def test_exact_alias_mismatch_is_accepted(self) -> None:
        reconstructed = {"passed": True, "access": {"model_load_count": 0}}
        persisted = {
            "passed": True,
            "access": {"model_load_count": 0, "population_build_count": 1},
        }
        self.assertTrue(
            sole_population_build_count_alias_mismatch(reconstructed, persisted)
        )
        self.assertEqual(without_population_build_count(persisted), reconstructed)

    def test_any_other_difference_is_rejected(self) -> None:
        reconstructed = {"passed": True, "access": {"model_load_count": 0}}
        persisted = {
            "passed": False,
            "access": {"model_load_count": 0, "population_build_count": 1},
        }
        self.assertFalse(
            sole_population_build_count_alias_mismatch(reconstructed, persisted)
        )


if __name__ == "__main__":
    unittest.main()
