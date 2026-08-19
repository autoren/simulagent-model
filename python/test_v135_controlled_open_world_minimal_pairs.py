import json
import unittest
from pathlib import Path

from v135_controlled_open_world_minimal_pairs import FORBIDDEN_PUBLIC_KEYS, build_catalog, build_population, evaluate_gates


class V135ControlledOpenWorldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(Path("configs/v135-controlled-open-world-minimal-pairs.json").read_text())
        cls.catalog = build_catalog(cls.config)
        cls.population = build_population(cls.config)

    def test_every_preregistered_gate_passes(self):
        gates = evaluate_gates(self.catalog, self.population, self.config)
        self.assertTrue(all(gates.values()), gates)

    def test_each_group_has_five_exact_counterfactual_stages(self):
        by_group = {}
        for row in self.population["hidden_fixtures"]:
            by_group.setdefault(row["group_id"], []).append(row)
        expected = set(self.config["generation"]["stagesPerGroup"])
        self.assertEqual(len(by_group), 40)
        for rows in by_group.values():
            self.assertEqual({row["phase"] for row in rows}, expected)
            ambiguous = next(row for row in rows if row["phase"] == "ambiguous")
            self.assertEqual(ambiguous["truth_choice_id"], "A00")
            self.assertEqual(len(ambiguous["possible_choice_ids"]), 2)

    def test_public_artifact_contains_no_gold_or_family_metadata(self):
        def keys(value):
            if isinstance(value, dict):
                return set(value).union(*(keys(child) for child in value.values()))
            if isinstance(value, list):
                return set().union(*(keys(child) for child in value)) if value else set()
            return set()

        self.assertFalse(keys(self.population["public_fixtures"]) & FORBIDDEN_PUBLIC_KEYS)

    def test_test_surfaces_use_disjoint_slot_variants(self):
        hidden = self.population["hidden_fixtures"]
        dev = {(row["family_id"], row["variant_index"]) for row in hidden if row["split"] == "development"}
        test = {(row["family_id"], row["variant_index"]) for row in hidden if row["split"] == "test"}
        self.assertFalse(dev & test)


if __name__ == "__main__":
    unittest.main()
