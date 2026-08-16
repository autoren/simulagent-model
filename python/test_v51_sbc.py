from __future__ import annotations

import copy
import json
import unittest
from decimal import Decimal

from evaluate_v51_sbc import evaluate_replication, map_inference
from generate_v51_sbc import build_replications
from v22r2_grounding import PROJECT_ROOT
from v51_sbc import (
    batch_inference,
    independent_inference,
    mechanic_registry,
    randomized_rank,
)


class V51SimulationBasedCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(
            (PROJECT_ROOT / "configs/v51-simulation-based-calibration.json").read_text()
        )
        cls.small_config = copy.deepcopy(cls.config)
        cls.small_config["simulation"]["replications"] = 6
        for name in (
            "generatorSeed", "priorSeed", "trajectorySeed", "posteriorDrawSeed", "tieBreakSeed"
        ):
            cls.small_config["simulation"][name] += 1_000_000
        cls.registry = mechanic_registry()
        cls.records = build_replications(cls.small_config)

    def test_registry_is_fresh_unique_and_balanced(self):
        self.assertEqual(len(self.registry), 48)
        self.assertEqual(len({row["key"] for row in self.registry}), 48)
        cells = {}
        for row in self.registry:
            key = (row["family"], row["probability"])
            cells[key] = cells.get(key, 0) + 1
        self.assertEqual(set(cells.values()), {4})

    def test_batch_and_stepwise_paths_agree(self):
        for record in self.records:
            result = evaluate_replication(record, self.registry, self.small_config)
            self.assertTrue(result["normalization"])
            self.assertLessEqual(max(result["exact_agreement"].values()), 1e-90)

    def test_map_control_can_exclude_true_configuration(self):
        for record in self.records:
            exact = independent_inference(
                self.registry, record["supports"], record["query"]
            )
            point = map_inference(exact)
            self.assertEqual(sum(point["joint"].values(), Decimal(0)), Decimal(1))
            self.assertEqual(sum(point["configuration"].values(), Decimal(0)), Decimal(1))

    def test_batch_and_stepwise_inputs_are_independently_computed(self):
        record = self.records[0]
        batch = batch_inference(self.registry, record["supports"], record["query"])
        stepwise = independent_inference(self.registry, record["supports"], record["query"])
        self.assertEqual(len(batch["support_program"]), len(stepwise["support_program"]))
        self.assertIsNot(batch["joint"], stepwise["joint"])

    def test_randomized_ties_cover_inclusive_rank_support(self):
        ranks = {randomized_rank(1, [1] * 63, seed) for seed in range(500)}
        self.assertIn(0, ranks)
        self.assertIn(63, ranks)
        self.assertTrue(ranks <= set(range(64)))


if __name__ == "__main__":
    unittest.main()
