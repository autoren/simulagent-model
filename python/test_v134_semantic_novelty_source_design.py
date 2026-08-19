import json
import unittest
from pathlib import Path

from v134_semantic_novelty_source_design import build_catalog, derive_classes, select_population


class V134SourceDesignTests(unittest.TestCase):
    def test_text_free_balanced_population(self):
        config = json.loads(Path("configs/v134-semantic-novelty-source-design.json").read_text()); inventory = json.loads(Path(config["sourceInventory"]).read_text())
        rows = derive_classes(inventory, "dev"); catalog = build_catalog(rows, config); population = select_population(rows, catalog, config)
        self.assertEqual(catalog["choice_count"], 11); self.assertEqual(population["fixture_count"], 264); self.assertEqual(population["cell_count"], 66)
        self.assertEqual(set(population["cell_counts"].values()), {4}); self.assertEqual(set(population["truth_choice_counts"].values()), {24}); self.assertEqual(set(population["presented_candidate_counts"].values()), {44})
        self.assertFalse(catalog["contains_language"]); self.assertFalse(population["contains_language"])


if __name__ == "__main__": unittest.main()
