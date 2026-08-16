#!/usr/bin/env python3
from __future__ import annotations

import json
import unittest

import numpy as np

from v22r2_grounding import PROJECT_ROOT
from v63_external_inference import (
    ALTERNATING,
    PERSISTENT,
    exact_inference,
    family_transition,
    load_anchor,
    particle_filter_episode,
    posterior_draws,
    simulate_episode,
    smc2_inference,
)


class V63ExternalInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        lock = json.loads((PROJECT_ROOT / "configs/v63-design-lock.json").read_text())
        cls.config = lock["config_payload"]
        cls.anchor = load_anchor(PROJECT_ROOT / cls.config["externalSource"]["sealedModel"])

    def test_family_changes_only_listen_and_normalizes(self) -> None:
        persistent = family_transition(self.anchor, PERSISTENT, 0.8)
        alternating = family_transition(self.anchor, ALTERNATING, 0.8)
        self.assertTrue(np.allclose(persistent.sum(axis=2), 1.0))
        self.assertTrue(np.allclose(alternating.sum(axis=2), 1.0))
        self.assertTrue(np.array_equal(persistent[1:], alternating[1:]))
        self.assertAlmostEqual(float(persistent[0, 2, 2]), 0.8)
        self.assertAlmostEqual(float(alternating[0, 2, 2]), 0.2)

    def test_exact_prior_is_symmetric_and_normalized(self) -> None:
        record = {"id": "fixture-prior", "episodes": [{"observations": [1]}]}
        result = exact_inference(self.anchor, record, self.config)
        self.assertAlmostEqual(sum(result["identity"]), 1.0, places=12)
        self.assertAlmostEqual(sum(result["theta_weights"]), 1.0, places=12)
        self.assertAlmostEqual(sum(result["current_side"]), 1.0, places=12)
        self.assertAlmostEqual(sum(result["next_observation"]), 1.0, places=12)
        self.assertAlmostEqual(result["identity"][0], 0.5, places=12)
        mean_theta = float(np.asarray(result["theta_values"]) @ result["theta_weights"])
        self.assertAlmostEqual(mean_theta, 0.8, places=12)

    def test_exact_temporal_reports_identify_modes(self) -> None:
        same = {"id": "fixture-same", "episodes": [{"observations": [1, 1, 1, 1, 1, 1]}]}
        alternating = {
            "id": "fixture-alternating",
            "episodes": [{"observations": [1, 2, 1, 2, 1, 2]}],
        }
        self.assertGreater(exact_inference(self.anchor, same, self.config)["identity"][0], 0.9)
        self.assertGreater(
            exact_inference(self.anchor, alternating, self.config)["identity"][1], 0.9
        )

    def test_particle_filter_tracks_exact_side_on_development_fixture(self) -> None:
        reports, _ = simulate_episode(self.anchor, PERSISTENT, 0.84, 10, 106301)
        record = {"id": "fixture-pf", "episodes": [{"observations": reports}]}
        exact = exact_inference(self.anchor, record, self.config)
        log_likelihood, states, weights, diagnostic = particle_filter_episode(
            self.anchor,
            PERSISTENT,
            0.84,
            reports,
            4096,
            106329,
            ("fixture",),
            0.5,
        )
        self.assertTrue(np.isfinite(log_likelihood))
        left = float(weights[(states == 2)].sum())
        exact_log, exact_state = __import__(
            "v63_external_inference", fromlist=["exact_filter_episode"]
        ).exact_filter_episode(self.anchor, PERSISTENT, 0.84, reports)
        self.assertLess(abs(log_likelihood - exact_log), 0.08)
        self.assertLess(abs(left - exact_state[2]), 0.05)
        self.assertFalse(diagnostic["extinct"])
        self.assertAlmostEqual(sum(exact["identity"]), 1.0, places=12)

    def test_smc2_matches_exact_on_unsealed_development_fixture(self) -> None:
        episodes = []
        for episode in range(3):
            reports, _ = simulate_episode(
                self.anchor, ALTERNATING, 0.82, 8, 106317 + episode
            )
            episodes.append({"observations": reports})
        record = {"id": "fixture-smc2", "episodes": episodes}
        exact = exact_inference(self.anchor, record, self.config)
        approximate = smc2_inference(
            self.anchor, record, self.config, 127, 0, "implementation_fixture"
        )
        self.assertLess(
            0.5 * float(np.abs(np.asarray(exact["identity"]) - approximate["identity"]).sum()),
            0.15,
        )
        self.assertAlmostEqual(sum(approximate["identity"]), 1.0, places=12)
        self.assertAlmostEqual(sum(approximate["theta_weights"]), 1.0, places=12)
        self.assertAlmostEqual(sum(approximate["current_side"]), 1.0, places=12)
        self.assertAlmostEqual(sum(approximate["next_observation"]), 1.0, places=12)
        draws = posterior_draws(approximate, 16, 106343)
        self.assertEqual(len(draws), 16)
        self.assertTrue(all(draw["identity"] in (0, 1) for draw in draws))


if __name__ == "__main__":
    unittest.main()
