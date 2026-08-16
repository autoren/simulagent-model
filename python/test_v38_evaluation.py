import unittest
import numpy as np

from v38_evaluation import predict_hidden, select_method


class V38EvaluationTests(unittest.TestCase):
    def test_masked_candidate_prediction(self):
        hidden = np.asarray([[[2.0], [-2.0]], [[-2.0], [2.0]], [[3.0], [0.0]]], dtype=np.float32)
        mask = np.asarray([[True, True], [True, True], [True, False]])
        targets = np.asarray([0, 1, 0])
        predicted, _ = predict_hidden(hidden, mask, targets, hidden, mask, 10.0)
        np.testing.assert_array_equal(predicted, targets)

    def test_selection_prefers_worst_fold_then_method_order(self):
        reports = [
            {"method": "candidate_span_native_margin", "alpha": None, "mean_group_cv_focus_accuracy": 1.0, "worst_group_cv_focus_accuracy": 0.9},
            {"method": "deterministic_discourse_parser", "alpha": None, "mean_group_cv_focus_accuracy": 1.0, "worst_group_cv_focus_accuracy": 1.0},
        ]
        selected = select_method(reports, ["candidate_span_native_margin", "deterministic_discourse_parser"])
        self.assertEqual(selected["method"], "deterministic_discourse_parser")


if __name__ == "__main__":
    unittest.main()
