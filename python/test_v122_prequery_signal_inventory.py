import json
import unittest
from pathlib import Path
from v122_prequery_signal_inventory import build_inventory


class V122InventoryTests(unittest.TestCase):
    def test_frozen_inventory_boundaries(self):
        config = json.loads(Path("configs/v122-prequery-signal-inventory.json").read_text())
        result = build_inventory(config)
        self.assertTrue(result["outcome_pass"])
        self.assertEqual(result["llm_independent_semantic_families"], ["retrieval_geometry"])
        self.assertEqual(result["signal_evaluated_count"], 0)
        self.assertEqual(result["trigger_fitted_count"], 0)
        self.assertTrue(result["outcome_gates"]["mutability_recorded"])
        excluded = {row["id"] for row in result["excluded_signals"]}
        self.assertIn("realized_query_value_or_regret", excluded)


if __name__ == "__main__":
    unittest.main()
