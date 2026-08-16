from decimal import Decimal
import json
import unittest

from generate_v50_history import query_case
from v22r2_grounding import PROJECT_ROOT
from v46_stochastic import mechanic_registry as v46_registry
from v47_sampling import mechanic_registry as v47_registry
from v48_composition import mechanic_registry as v48_registry
from v49_belief import mechanic_registry as v49_registry
from v50_belief import (
    latest_only_evidence,
    mechanic_registry,
    query_predictive,
    time_shuffled_evidence,
)


class V50HistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((PROJECT_ROOT / "configs/v50-history-dependent-belief-filtering.json").read_text())

    def test_registry_is_fresh_balanced_and_unique(self):
        registry = mechanic_registry()
        previous = {
            row["key"]
            for source in (v46_registry, v47_registry, v48_registry, v49_registry)
            for row in source()
        }
        self.assertEqual(len(registry), 48)
        self.assertEqual(len({row["key"] for row in registry}), 48)
        self.assertFalse({row["key"] for row in registry} & previous)
        self.assertEqual({row["probability"] for row in registry}, {"1/4", "1/2", "3/4"})
        self.assertTrue(all(
            sum(row["family"] == family for row in registry) == 12
            for family in {row["family"] for row in registry}
        ))

    def test_history_dependent_queries_exist_in_every_family(self):
        registry = mechanic_registry()
        for family in sorted({row["family"] for row in registry}):
            mechanic = next(row for row in registry if row["family"] == family)
            built = query_case(mechanic, 0, self.config, set())
            self.assertGreaterEqual(built["oracle_tv"], 0.10)
            self.assertGreaterEqual(built["history_kl"], 0.05)
            self.assertGreaterEqual(built["shuffled_kl"], 0.05)
            prediction, weights, _ = query_predictive(
                [mechanic], [Decimal(1)], built["case"]["entities"], built["case"]["initial_world"],
                built["case"]["actions"], built["evidence"], built["prefix_length"],
            )
            self.assertAlmostEqual(float(sum(prediction.values(), Decimal(0))), 1.0)
            self.assertEqual(weights, [Decimal(1)])

    def test_ablation_evidence_changes_time_assignment(self):
        evidence = [
            [{"atom": "a", "value": True}],
            [],
            [{"atom": "b", "value": False}],
        ]
        self.assertEqual(latest_only_evidence(evidence), [[], [], evidence[-1]])
        shuffled = time_shuffled_evidence(evidence, 0)
        self.assertEqual(shuffled[0], [])
        self.assertEqual(shuffled[1], evidence[0])
        self.assertEqual(shuffled[2], evidence[2])


if __name__ == "__main__":
    unittest.main()
