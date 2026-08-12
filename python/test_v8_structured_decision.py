import unittest

import numpy as np

from evaluate_v8_structured_decision_mlx import ledger_scores


class V8StructuredDecisionTests(unittest.TestCase):
    def test_score_is_maximum_sensitive_class_margin(self):
        logits = np.zeros((2, 3, 5), dtype=np.float32)
        logits[0, 1, 2] = 3.0
        logits[0, 1, 4] = 1.0
        logits[1, :, 0] = 2.0
        logits[1, :, 2] = 1.5
        np.testing.assert_allclose(ledger_scores(logits), [2.0, -0.5])

    def test_invalid_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            ledger_scores(np.zeros((4, 5), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
