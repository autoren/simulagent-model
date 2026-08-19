from __future__ import annotations

import json
import unittest
from pathlib import Path

from v210_controlled_language_population_projection import (
    ROLE_NAMES,
    canonical_jsonl,
    generate_population,
    identifier_hash,
    project_development_surfaces,
)


CONFIG = json.loads(Path("configs/v210-controlled-language-population-projection.json").read_text())
PARENT = json.loads(Path(CONFIG["parentV209r1OutcomeLock"]).read_text())
PARENT_LOCK = json.loads(Path(PARENT["repair_lock"]).read_text())
V209_LOCK = json.loads(Path(PARENT_LOCK["parent_V209_design_lock"]).read_text())
PARENT_CONFIG = V209_LOCK["config_payload"]


class V210PopulationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.population = generate_population(CONFIG, PARENT_CONFIG)

    def test_exact_role_counts_and_identifier_hashes(self) -> None:
        all_ids = []
        for role in ROLE_NAMES:
            surfaces = self.population[role]["surfaces"]
            truths = self.population[role]["truth"]
            self.assertEqual(len(surfaces), 270)
            self.assertEqual(len(truths), 270)
            ids = [row["record_id"] for row in surfaces]
            self.assertEqual(identifier_hash(ids), CONFIG["population"]["recordIdHashes"][role])
            all_ids.extend(ids)
        self.assertEqual(identifier_hash(all_ids), CONFIG["population"]["recordIdHashes"]["ALL"])

    def test_generation_is_byte_deterministic(self) -> None:
        regenerated = generate_population(CONFIG, PARENT_CONFIG)
        for role in ROLE_NAMES:
            self.assertEqual(
                canonical_jsonl(self.population[role]["surfaces"]),
                canonical_jsonl(regenerated[role]["surfaces"]),
            )
            self.assertEqual(
                canonical_jsonl(self.population[role]["truth"]),
                canonical_jsonl(regenerated[role]["truth"]),
            )

    def test_surface_and_truth_schemas_are_separate(self) -> None:
        forbidden = {"semantic_regime", "task_state", "semantic_observation_id", "source_probability", "history"}
        for role in ROLE_NAMES:
            self.assertTrue(all(not (set(row) & forbidden) for row in self.population[role]["surfaces"]))
            self.assertTrue(all(forbidden <= set(row) for row in self.population[role]["truth"]))

    def test_projector_accepts_only_one_third_without_truth(self) -> None:
        predictions = project_development_surfaces(self.population["DEVELOPMENT"]["surfaces"], CONFIG)
        accepted = [row for row in predictions if row["accepted"]]
        self.assertEqual(len(predictions), 270)
        self.assertEqual(len(accepted), 90)
        self.assertEqual(len(predictions) - len(accepted), 180)


if __name__ == "__main__":
    unittest.main()
