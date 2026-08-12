import unittest

import numpy as np

from evaluate_v15_full_pipeline import nli_pairs_by_base, unique_current_targets


class V15FullPipelineTests(unittest.TestCase):
    def test_unique_current_targets_reject_conflicts(self):
        base = np.asarray([0, 0, 1])
        self.assertEqual(unique_current_targets(base, np.asarray([1, 1, -1]), 2).tolist(), [1, -1])
        with self.assertRaises(RuntimeError):
            unique_current_targets(base, np.asarray([1, 0, -1]), 2)

    def test_nli_pair_mapping_rejects_conflicts(self):
        base = np.asarray([0, 0, 1])
        pairs = np.asarray([[2, 3], [2, 3], [4, 5]])
        np.testing.assert_array_equal(nli_pairs_by_base(base, pairs, 2), [[2, 3], [4, 5]])
        with self.assertRaises(RuntimeError):
            nli_pairs_by_base(base, np.asarray([[2, 3], [3, 2], [4, 5]]), 2)


if __name__ == "__main__":
    unittest.main()
