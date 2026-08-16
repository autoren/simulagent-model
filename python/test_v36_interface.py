"""Tests for exact V36 readout serialization and inference."""

from __future__ import annotations

import unittest

import numpy as np

from v36_interface import fit_component, predict_component, projection_matrix


class V36InterfaceTests(unittest.TestCase):
    def test_serialized_binary_predictions_match_sklearn(self):
        features = np.asarray([[-2.0, 1.0], [-1.0, .5], [1.0, -.5], [2.0, -1.0]])
        targets = np.asarray([0, 0, 1, 1])
        model, parameters = fit_component(features, targets, 10.0)
        np.testing.assert_array_equal(predict_component(features, parameters), model.predict(features))

    def test_serialized_multiclass_predictions_match_sklearn(self):
        features = np.asarray([[-2.0, 0], [-1.0, 0], [0, 2.0], [0, 1.0], [2.0, -1.0], [1.0, -2.0]])
        targets = np.asarray([0, 0, 1, 1, 2, 2])
        model, parameters = fit_component(features, targets, 1.0)
        np.testing.assert_array_equal(predict_component(features, parameters), model.predict(features))

    def test_projection_is_fixed(self):
        first = projection_matrix(7, 3, 3501)
        second = projection_matrix(7, 3, 3501)
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
