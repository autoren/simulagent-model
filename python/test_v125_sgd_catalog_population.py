import json
import unittest
from pathlib import Path

from v125_sgd_catalog_population import build_catalog, evaluate_gates, select_populations


class V125CatalogPopulationTests(unittest.TestCase):
    def test_frozen_text_free_catalog_and_population(self):
        config = json.loads(Path("configs/v125-sgd-catalog-population.json").read_text())
        inventory = json.loads(Path(config["sourceInventory"]).read_text())
        catalog = build_catalog(inventory, config)
        populations = select_populations(inventory, config)
        checks = evaluate_gates(catalog, populations, config)
        self.assertTrue(all(checks.values()), checks)
        self.assertEqual(catalog["choice_count"], 11)
        self.assertEqual(populations["evaluation_class_counts"], {"known": 192, "novel_valid": 192, "unsupported": 192})
        self.assertFalse(catalog["contains_language"])
        self.assertFalse(populations["contains_language"])


if __name__ == "__main__":
    unittest.main()
