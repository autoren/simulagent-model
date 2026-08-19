from __future__ import annotations

import unittest

from v117_causal_clarification_simulator import joint_distribution, true_observations


BY_ID = {
    "K00": {"kind": "KNOWN"}, "K01": {"kind": "KNOWN"},
    "N00": {"kind": "NOVEL"}, "U00": {"kind": "UNSUPPORTED"},
    "A00": {"kind": "ABSTAIN"},
}
CONFIG = {"channel": {"nonCorrectMassToUnsure": 0.5}}


class V117CausalSimulatorTests(unittest.TestCase):
    def test_truth_mechanisms_are_distinct(self) -> None:
        self.assertEqual(true_observations("K00", "K00", BY_ID), ("CONFIRM", "DECLARED"))
        self.assertEqual(true_observations("K01", "K00", BY_ID), ("REJECT", "DECLARED"))
        self.assertEqual(true_observations("N00", "K00", BY_ID), ("REJECT", "UNDECLARED_VISIBLE"))
        self.assertEqual(true_observations("U00", "K00", BY_ID), ("REJECT", "OUTSIDE_VISIBLE"))

    def test_joint_channel_is_normalized_and_preserves_perfect_truth(self) -> None:
        for rho in (0.0, 0.5, 1.0):
            distribution = joint_distribution("N00", "K00", 0.95, rho, BY_ID, CONFIG)
            self.assertAlmostEqual(sum(distribution.values()), 1.0)
            perfect = joint_distribution("N00", "K00", 1.0, rho, BY_ID, CONFIG)
            self.assertEqual(perfect[("REJECT", "UNDECLARED_VISIBLE")], 1.0)


if __name__ == "__main__":
    unittest.main()
