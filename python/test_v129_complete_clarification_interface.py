import json
import unittest
from pathlib import Path

from v129_complete_clarification_interface import answer_distribution, run_audit


class V129CompleteClarificationTests(unittest.TestCase):
    def test_channels_normalize_and_bias_only_error(self):
        identifiers = ["K1", "K2", "U", "A"]
        symmetric = answer_distribution("K1", "K2", 0.95, "symmetric", identifiers, "A", 0.75)
        biased = answer_distribution("K1", "K2", 0.95, "candidate_attraction", identifiers, "A", 0.75)
        self.assertAlmostEqual(sum(symmetric.values()), 1.0)
        self.assertAlmostEqual(sum(biased.values()), 1.0)
        self.assertEqual(biased["K1"], 0.95)
        self.assertGreater(biased["K2"], symmetric["K2"])

    def test_full_audit_is_aggregate_and_complete(self):
        config = json.loads(Path("configs/v129-complete-clarification-interface.json").read_text())
        catalog = json.loads(Path(config["choiceCatalog"]).read_text())
        baseline = json.loads(Path(config["baselineConfig"]).read_text())
        v119 = json.loads(Path(config["V119Config"]).read_text())
        result = run_audit(catalog, baseline, v119, config)
        self.assertEqual(result["census_pair_count"], 66)
        self.assertEqual(result["individual_pair_emission_count"], 0)
        self.assertNotIn("pairs", result)
        self.assertEqual(len(result["conditions"]), 3)


if __name__ == "__main__": unittest.main()
