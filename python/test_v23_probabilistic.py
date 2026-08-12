import itertools
import unittest

import numpy as np

from v23_probabilistic_relational import (
    credible_indices,
    k_best_assignments,
    k_best_independent,
    normalized_top_graphs,
)


class V23ProbabilisticTests(unittest.TestCase):
    def test_murty_matches_brute_force_order(self):
        scores = np.asarray([
            [0.2, 0.8, 0.1],
            [0.7, 0.3, 0.4],
            [0.5, 0.2, 0.9],
        ])
        expected = sorted(
            ((sum(scores[row, col] for row, col in enumerate(values)), values)
             for values in itertools.permutations(range(3))),
            key=lambda row: (-row[0], row[1]),
        )
        actual = k_best_assignments(scores, 6)
        self.assertEqual([row[1] for row in actual], [row[1] for row in expected])
        self.assertTrue(np.allclose([row[0] for row in actual], [row[0] for row in expected]))

    def test_independent_top_k_matches_brute_force(self):
        labels = [
            [("a", 0.7), ("b", 0.2), ("c", 0.1)],
            [("x", 0.6), ("y", 0.4)],
            [("m", 0.8), ("n", 0.2)],
        ]
        expected = sorted(
            ((sum(np.log(value[1]) for value in row), tuple(value[0] for value in row))
             for row in itertools.product(*labels)),
            key=lambda row: (-row[0], row[1]),
        )[:7]
        actual = k_best_independent(labels, 7)
        self.assertEqual([row[1] for row in actual], [row[1] for row in expected])
        self.assertTrue(np.allclose([row[0] for row in actual], [row[0] for row in expected]))

    def test_graph_normalization_and_unknown_cap(self):
        assignments = [(0.0, (0, 1)), (-1.0, (1, 0))]
        truth = [(0.0, ("true", "false")), (-0.2, ("unknown", "unknown"))]
        rows = normalized_top_graphs(assignments, truth, 3, maximum_unknown=1)
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(sum(row["probability"] for row in rows), 1.0)

    def test_credible_indices_are_deterministic(self):
        posterior = np.asarray([0.45, 0.45, 0.10])
        self.assertEqual(credible_indices(posterior, ["b", "a", "c"], 0.5), [1, 0])


if __name__ == "__main__":
    unittest.main()
