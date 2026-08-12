import math
import unittest

import numpy as np

from v28_marginal_map import logsumexp, select_marginal_episode_map


class Hypothesis:
    def __init__(self, key):
        self.key = key


class V28MarginalMapTests(unittest.TestCase):
    def test_logsumexp(self):
        self.assertAlmostEqual(logsumexp([0.0, 0.0]), math.log(2.0))
        self.assertEqual(logsumexp([]), -math.inf)

    def test_marginal_evidence_can_beat_one_larger_graph(self):
        hypotheses = [Hypothesis("single_peak"), Hypothesis("distributed_mass")]
        graphs = [[
            {"log_score": math.log(0.45)},
            {"log_score": math.log(0.30)},
            {"log_score": math.log(0.25)},
        ]]
        compatibility = [np.asarray([
            [True, False],
            [False, True],
            [False, True],
        ])]
        selected = select_marginal_episode_map(hypotheses, graphs, compatibility)
        self.assertEqual(selected["program_key"], "distributed_mass")
        self.assertEqual(selected["graph_indices"], (1,))
        self.assertAlmostEqual(selected["posterior"].sum(), 1.0)

    def test_ties_use_program_key(self):
        hypotheses = [Hypothesis("b"), Hypothesis("a")]
        graphs = [[{"log_score": 0.0}]]
        compatibility = [np.asarray([[True, True]])]
        selected = select_marginal_episode_map(hypotheses, graphs, compatibility)
        self.assertEqual(selected["program_key"], "a")


if __name__ == "__main__":
    unittest.main()
