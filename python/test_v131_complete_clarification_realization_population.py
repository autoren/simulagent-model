import json
import unittest
from pathlib import Path

from v131_complete_clarification_realization_population import (
    build_catalog, evaluate_gates, excluded_identifiers, select_population,
)


class V131CompleteClarificationPopulationTests(unittest.TestCase):
    def test_balanced_text_free_truth_candidate_census(self):
        root = Path(".")
        config = json.loads(Path("configs/v131-complete-clarification-realization-population.json").read_text())
        inventory = json.loads(Path(config["sourceInventory"]).read_text())
        excluded = excluded_identifiers(config["excludedPopulations"], root)
        catalog = build_catalog(inventory, config)
        population = select_population(inventory, catalog, excluded, config)
        checks = evaluate_gates(catalog, population, config)
        self.assertTrue(all(checks.values()), checks)
        self.assertEqual(population["fixture_count"], 264)
        self.assertEqual(population["cell_count"], 66)
        self.assertEqual(set(population["cell_counts"].values()), {4})
        self.assertEqual(set(population["truth_choice_counts"].values()), {24})
        self.assertEqual(set(population["presented_candidate_counts"].values()), {44})
        self.assertFalse(catalog["contains_language"])
        self.assertFalse(population["contains_language"])


if __name__ == "__main__":
    unittest.main()
