import json
import unittest
from pathlib import Path

from v146_fresh_codebook_population import STAGES, audit_population, build_population


class V146FreshCodebookPopulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(Path("configs/v146-fresh-codebook-population.json").read_text())
        cls.population = build_population(cls.config)
        cls.prior = json.loads(Path("outputs/v135-controlled-open-world-minimal-pairs/design/public-fixtures.json").read_text()) + json.loads(Path("outputs/v142-certificate-interface-population/design/public-fixtures.json").read_text())

    def test_counts_and_splits(self):
        summary = self.population["population_summary"]
        self.assertEqual(summary["fixture_count"], 288)
        self.assertEqual(summary["group_count"], 48)
        self.assertEqual(summary["split_counts"], {"development": 144, "test": 144})
        self.assertEqual(set(summary["stage_counts"]), set(STAGES))

    def test_public_rows_hide_ground_truth(self):
        forbidden = {"group_id", "family_id", "stage", "language_class", "truth_choice_id", "compatible_choice_ids", "variant_index"}
        self.assertTrue(all(not (forbidden & set(row)) for row in self.population["public_fixtures"]))

    def test_codebook_covers_all_oracle_certificates(self):
        self.assertEqual(len(self.population["certificate_codebook"]["entries"]), 14)
        result = audit_population(self.population, self.config, self.prior)
        self.assertEqual(result["oracle_code_coverage"], 1.0)

    def test_no_exact_prior_conversation_overlap(self):
        result = audit_population(self.population, self.config, self.prior)
        self.assertEqual(result["exact_prior_conversation_overlap_count"], 0)

    def test_all_population_gates_pass(self):
        result = audit_population(self.population, self.config, self.prior)
        self.assertTrue(result["passed"], result)


if __name__ == "__main__":
    unittest.main()
