#!/usr/bin/env python3
"""Tests for exact V71 sensor-codebook belief construction."""
from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from v71_cassandra_pomdp import parse_cassandra_pomdp_file
from v71_sensor_codebook import (
    enumerate_public_prefixes,
    initial_joint_belief,
    sensor_observation_models,
    update_joint_belief,
)


SOURCE = Path(
    "data/v71-sensor-codebook/source-checkout/examples/pomdp-files"
)
DEVELOPMENT = (
    "concert.POMDP",
    "ejs1.POMDP",
    "manuel-hartman.2013-09-19.POMDP",
)


class V71SensorCodebookTests(unittest.TestCase):
    def parse(self, filename: str):
        return parse_cassandra_pomdp_file(SOURCE / filename)

    def test_point_models_are_normalized_with_identical_support(self) -> None:
        for filename in DEVELOPMENT:
            with self.subTest(filename=filename):
                sensor = sensor_observation_models(self.parse(filename))
                np.testing.assert_allclose(sensor.sum(axis=-1), 1.0, atol=1e-12)
                np.testing.assert_array_equal(sensor[0] > 0.0, sensor[1] > 0.0)

    def test_initial_joint_belief_has_frozen_latent_prior(self) -> None:
        parsed = self.parse("concert.POMDP")
        belief = initial_joint_belief(parsed)
        np.testing.assert_allclose(belief.sum(axis=1), [0.5, 0.5])
        np.testing.assert_allclose(belief.sum(axis=0), parsed.model.initial)

    def test_update_probability_matches_direct_enumeration(self) -> None:
        parsed = self.parse("ejs1.POMDP")
        sensor = sensor_observation_models(parsed)
        belief = initial_joint_belief(parsed)
        probability, posterior = update_joint_belief(parsed, sensor, belief, 1, 0)
        predicted = np.einsum("zs,sq->zq", belief, parsed.model.transition[1])
        expected = predicted * sensor[:, 1, :, 0]
        self.assertAlmostEqual(probability, float(expected.sum()))
        np.testing.assert_allclose(posterior, expected / expected.sum())

    def test_complete_development_census_has_frozen_upper_bound_size(self) -> None:
        expected = {"concert.POMDP": 7, "ejs1.POMDP": 9, "manuel-hartman.2013-09-19.POMDP": 5}
        for filename, count in expected.items():
            with self.subTest(filename=filename):
                prefixes = enumerate_public_prefixes(self.parse(filename))
                self.assertEqual(len(prefixes), count)
                self.assertEqual(sum(prefix.depth == 0 for prefix in prefixes), 1)
                self.assertTrue(
                    all(np.isclose(prefix.joint_belief.sum(), 1.0) for prefix in prefixes)
                )


if __name__ == "__main__":
    unittest.main()
