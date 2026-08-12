import math
import unittest

from rescore_v4_fp32_mlx import pairwise_log_loss, softplus


class V4Fp32RescoreTests(unittest.TestCase):
    def test_softplus_is_stable_for_large_values(self) -> None:
        self.assertAlmostEqual(softplus(1000.0), 1000.0)
        self.assertAlmostEqual(softplus(-1000.0), 0.0)

    def test_pairwise_log_loss_rewards_correctly_ordered_margins(self) -> None:
        gold = [False, True]
        useful = pairwise_log_loss(gold, [-2.0, 2.0])
        reversed_scores = pairwise_log_loss(gold, [2.0, -2.0])
        self.assertLess(useful, math.log(2.0))
        self.assertGreater(reversed_scores, math.log(2.0))

    def test_pairwise_log_loss_validates_lengths(self) -> None:
        with self.assertRaises(ValueError):
            pairwise_log_loss([False], [])


if __name__ == "__main__":
    unittest.main()
