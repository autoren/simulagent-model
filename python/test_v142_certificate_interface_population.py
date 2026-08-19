import json
import unittest
from pathlib import Path

from v142_certificate_interface_population import (
    STAGES,
    audit_population,
    build_population,
    deterministic_finalize,
    validate_certificate,
)


class V142CertificateInterfacePopulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(Path("configs/v142-certificate-interface-population.json").read_text())
        cls.population = build_population(cls.config)
        cls.catalog = cls.population["choice_catalog"]
        cls.v135_public = json.loads(Path("outputs/v135-controlled-open-world-minimal-pairs/design/public-fixtures.json").read_text())

    def test_population_counts_and_six_stage_groups(self):
        summary = self.population["population_summary"]
        self.assertEqual(summary["fixture_count"], 288)
        self.assertEqual(summary["group_count"], 48)
        self.assertEqual(summary["split_counts"], {"development": 144, "test": 144})
        groups = {}
        for row in self.population["hidden_fixtures"]:
            groups.setdefault(row["group_id"], set()).add(row["stage"])
        self.assertTrue(all(stages == set(STAGES) for stages in groups.values()))

    def test_familiar_and_unfamiliar_known_forms_share_truth(self):
        groups = {}
        for row in self.population["hidden_fixtures"]:
            groups.setdefault(row["group_id"], {})[row["stage"]] = row
        for stages in groups.values():
            self.assertEqual(
                stages["clear_known_familiar"]["truth_choice_id"],
                stages["clear_known_unfamiliar"]["truth_choice_id"],
            )
            self.assertNotEqual(
                stages["clear_known_familiar"]["conversation"],
                stages["clear_known_unfamiliar"]["conversation"],
            )

    def test_certificate_contract_and_deterministic_fallback(self):
        valid = {
            "evidence_status": "SUFFICIENT",
            "compatible_choice_ids": ["K11"],
            "proposed_choice_id": "K11",
        }
        ambiguous = {
            "evidence_status": "INSUFFICIENT",
            "compatible_choice_ids": ["K11", "N11"],
            "proposed_choice_id": "A00",
        }
        inconsistent = {
            "evidence_status": "SUFFICIENT",
            "compatible_choice_ids": ["K11", "N11"],
            "proposed_choice_id": "K11",
        }
        self.assertTrue(validate_certificate(valid, self.catalog, self.config)["certificate_valid"])
        self.assertTrue(validate_certificate(ambiguous, self.catalog, self.config)["certificate_valid"])
        self.assertFalse(validate_certificate(inconsistent, self.catalog, self.config)["certificate_valid"])
        self.assertEqual(json.loads(deterministic_finalize(valid, self.catalog, self.config)["final_json"]), {"choice_id": "K11"})
        self.assertEqual(json.loads(deterministic_finalize(inconsistent, self.catalog, self.config)["final_json"]), {"choice_id": "A00"})

    def test_public_records_do_not_expose_hidden_labels(self):
        forbidden = {"group_id", "family_id", "stage", "language_class", "truth_choice_id", "compatible_choice_ids"}
        self.assertTrue(all(not (forbidden & set(row)) for row in self.population["public_fixtures"]))

    def test_full_audit_passes_and_language_is_fresh(self):
        audit = audit_population(self.population, self.config, self.v135_public)
        self.assertTrue(audit["passed"], audit)
        self.assertEqual(audit["deterministic_finalizer_validity"], 1.0)
        self.assertEqual(audit["exact_conversation_overlap_with_V135_count"], 0)


if __name__ == "__main__":
    unittest.main()
