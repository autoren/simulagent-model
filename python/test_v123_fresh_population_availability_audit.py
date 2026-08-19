import json
import unittest
from pathlib import Path

from v123_fresh_population_availability_audit import run_audit


class V123AvailabilityTests(unittest.TestCase):
    def test_frozen_aggregate_availability(self):
        config = json.loads(Path("configs/v123-fresh-population-availability-audit.json").read_text())
        inventory = json.loads(Path(config["sourceInventory"]).read_text())
        exclusions = [json.loads(Path(path).read_text()) for path in config["excludedPopulations"]]
        result = run_audit(inventory, exclusions, config)
        self.assertTrue(result["outcome_pass"])
        self.assertFalse(result["candidate_requirement_pass"])
        self.assertEqual(result["maximum_balanced_record_count_per_class"], 9)
        self.assertEqual(result["remaining_scenario_counts"]["novel_valid"], 1)
        self.assertEqual(result["language_read_count"], 0)


if __name__ == "__main__":
    unittest.main()
