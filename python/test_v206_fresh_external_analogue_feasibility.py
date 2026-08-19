from __future__ import annotations

import json
import unittest
from pathlib import Path

from v206_fresh_external_analogue_feasibility import (
    detect_license,
    evaluate_documentation,
    repository_slug,
)


CONFIG = json.loads(Path("configs/v206-fresh-external-analogue-feasibility.json").read_text())
CANDIDATE = CONFIG["candidates"][0]


class V206MetadataRulesTest(unittest.TestCase):
    def test_repository_slug_is_canonical(self) -> None:
        self.assertEqual(repository_slug("https://github.com/example/project"), "example/project")
        with self.assertRaises(ValueError):
            repository_slug("https://example.com/example/project")

    def test_MIT_license_detection(self) -> None:
        value = b"Permission is hereby granted, free of charge, to any person obtaining a copy"
        self.assertEqual(detect_license(value), "MIT")
        self.assertIsNone(detect_license(None))

    def test_complete_synthetic_official_metadata_passes_every_rule(self) -> None:
        readme = b"""
        Open-world POMDP with an in-episode calibration action using a known reference.
        The agent performs a sensing action and may abstain or hold the critical action.
        An observation model, transition model, and reward model define a delayed reward
        and state-dependent consequence in an exact simulator with deterministic replay.
        """
        license_value = b"Permission is hereby granted, free of charge, to any person obtaining a copy"
        result = evaluate_documentation(CANDIDATE, readme, license_value, CONFIG)
        self.assertTrue(result["qualified"])
        self.assertTrue(all(result["gate_results"].values()))

    def test_passive_OOS_classification_cannot_compensate_for_missing_mechanism(self) -> None:
        readme = b"Open-world out-of-scope intent classification dataset with calibrated confidence."
        license_value = b"Permission is hereby granted, free of charge, to any person obtaining a copy"
        result = evaluate_documentation(CANDIDATE, readme, license_value, CONFIG)
        self.assertFalse(result["qualified"])
        self.assertFalse(result["gate_results"]["action_dependent_information_gathering"])
        self.assertFalse(result["gate_results"]["in_episode_reference_calibration_or_cross_sensor_pathway"])


if __name__ == "__main__":
    unittest.main()
