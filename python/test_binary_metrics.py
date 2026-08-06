import unittest

from binary_metrics import evaluate_binary, fit_threshold, nonconstant, roc_auc


class BinaryMetricsTests(unittest.TestCase):
    def test_fits_a_separating_threshold(self) -> None:
        gold = [False, False, True, True]
        scores = [-2.0, -1.0, 1.0, 2.0]
        report = fit_threshold(gold, scores)
        self.assertEqual(report["balanced_accuracy"], 1.0)
        self.assertTrue(nonconstant(report))

    def test_constant_predictions_are_non_discriminating(self) -> None:
        report = evaluate_binary([False, False, True, True], [0.0] * 4, 0.0)
        self.assertEqual(report["balanced_accuracy"], 0.5)
        self.assertFalse(nonconstant(report))

    def test_auc_handles_ties(self) -> None:
        self.assertEqual(roc_auc([False, True], [1.0, 1.0]), 0.5)


if __name__ == "__main__":
    unittest.main()
