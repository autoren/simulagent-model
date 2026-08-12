import math
import unittest

import numpy as np

from v29_posterior_graph import posterior_marginal_decode


class Hypothesis:
    def __init__(self, key):
        self.key = key


class V29PosteriorGraphTests(unittest.TestCase):
    def test_other_trace_evidence_changes_graph_marginal(self):
        hypotheses = [Hypothesis("a"), Hypothesis("b")]
        graphs = [
            [
                {"log_score": math.log(0.55), "graph_key": "g0"},
                {"log_score": math.log(0.45), "graph_key": "g1"},
            ],
            [
                {"log_score": math.log(0.9), "graph_key": "h0"},
                {"log_score": math.log(0.1), "graph_key": "h1"},
            ],
        ]
        compatibility = [
            np.asarray([[True, False], [False, True]]),
            np.asarray([[False, True], [True, False]]),
        ]
        result = posterior_marginal_decode(hypotheses, graphs, compatibility)
        self.assertEqual(result["program_key"], "b")
        self.assertEqual(result["graph_indices"], (1, 0))
        self.assertTrue(all(0 < value <= 1 for value in result["graph_posteriors"]))

    def test_no_finite_episode_returns_none(self):
        result = posterior_marginal_decode(
            [Hypothesis("a")], [[{"log_score": 0.0, "graph_key": "g"}]],
            [np.asarray([[False]])],
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
