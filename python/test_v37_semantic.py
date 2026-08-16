import unittest

import numpy as np

from v37_semantic import cross_validate_component, fit_predict_method, select_method


class V37SemanticTests(unittest.TestCase):
    def test_candidate_hidden_shared_decodes_candidates(self):
        hidden = np.asarray([
            [[2.0, 0.0], [-2.0, 0.0]],
            [[-2.0, 0.0], [2.0, 0.0]],
            [[3.0, 0.0], [-3.0, 0.0]],
            [[-3.0, 0.0], [3.0, 0.0]],
        ], dtype=np.float32)
        bundle = {
            "candidate_hidden": hidden,
            "candidate_margin": hidden[:, :, 0],
            "direct_hidden": hidden[:, 0],
        }
        targets = np.asarray([0, 1, 0, 1])
        predictions, _ = fit_predict_method(
            "candidate_hidden_shared_ridge", 1.0, bundle, targets, bundle
        )
        np.testing.assert_array_equal(predictions, targets)

    def test_selection_uses_cv_ranking(self):
        reports = [
            {"method": "direct_hidden_ridge", "alpha": 1.0, "mean_group_cv_accuracy": 0.8, "worst_group_cv_accuracy": 0.7},
            {"method": "candidate_native_margin", "alpha": None, "mean_group_cv_accuracy": 0.9, "worst_group_cv_accuracy": 0.6},
            {"method": "candidate_hidden_shared_ridge", "alpha": 10.0, "mean_group_cv_accuracy": 0.9, "worst_group_cv_accuracy": 0.8},
        ]
        selected = select_method(reports, [
            "candidate_hidden_shared_ridge", "candidate_native_margin", "direct_hidden_ridge"
        ])
        self.assertEqual(selected["method"], "candidate_hidden_shared_ridge")


if __name__ == "__main__":
    unittest.main()
