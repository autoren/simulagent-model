import json
import unittest
from pathlib import Path

from v145_finite_certificate_codebook import build_abstract_population, build_codebook, evaluate, finalize_code, oracle_code


class V145FiniteCertificateCodebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(Path("configs/v145-finite-certificate-codebook.json").read_text())
        cls.codebook = build_codebook(cls.config)
        cls.population = build_abstract_population(cls.config)

    def test_codebook_has_registered_singletons_and_pairs(self):
        self.assertEqual(len(self.codebook), 14)
        self.assertEqual(sum(row["certificate"]["evidence_status"] == "SUFFICIENT" for row in self.codebook), 8)
        self.assertEqual(sum(row["certificate"]["evidence_status"] == "INSUFFICIENT" for row in self.codebook), 6)

    def test_abstract_population_matches_frozen_topology(self):
        self.assertEqual(len(self.population), 288)
        self.assertEqual(len({row["group_id"] for row in self.population}), 48)
        self.assertEqual(sum(row["stage"] == "ambiguous" for row in self.population), 48)

    def test_oracle_codes_recover_every_structural_truth(self):
        for row in self.population:
            finalized = finalize_code(oracle_code(row), self.codebook)
            self.assertTrue(finalized["code_valid"])
            self.assertEqual(finalized["final_choice_id"], row["truth_choice_id"])
            self.assertEqual(finalized["normalized_certificate"]["compatible_choice_ids"], row["compatible_choice_ids"])

    def test_unknown_codes_fail_closed(self):
        for value in (None, "", "UNKNOWN", {}, [], 17, "S__A00"):
            finalized = finalize_code(value, self.codebook)
            self.assertFalse(finalized["code_valid"])
            self.assertEqual(finalized["final_choice_id"], "A00")
            self.assertTrue(finalized["final_output_structurally_valid"])

    def test_all_model_free_gates_pass(self):
        result = evaluate(self.config)
        self.assertTrue(result["passed"], result)
        self.assertTrue(all(result["checks"].values()))
        self.assertTrue(result["semantic_limitation"]["structural_interface_cannot_detect_semantic_error"])


if __name__ == "__main__":
    unittest.main()
