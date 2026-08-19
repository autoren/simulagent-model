from __future__ import annotations

import unittest

from v169_fresh_constraint_state_population import exact_version_space
from v172_trusted_shadow_integration_population import all_source_states


class V172TrustedShadowIntegrationPopulationTest(unittest.TestCase):
    def test_complete_three_constraint_source_has_448_unique_states(self) -> None:
        states = all_source_states()
        self.assertEqual(len(states), 448)
        self.assertEqual(len({row["constraint_signature"] for row in states}), 448)
        self.assertEqual({len(row["constraints"]) for row in states}, {3})

    def test_each_three_constraint_truth_table_slice_has_32_candidates(self) -> None:
        representative = all_source_states()[173]
        self.assertEqual(len(exact_version_space(representative["constraints"])), 32)


if __name__ == "__main__":
    unittest.main()
