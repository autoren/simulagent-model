import json
import unittest
from pathlib import Path

from v127_sgd_typed_constraint_feasibility import (
    population_gates, run_evaluation, select_fresh_population, signature_decision,
)


class V127TypedConstraintTests(unittest.TestCase):
    def test_population_is_fresh_balanced_and_text_free(self):
        config = json.loads(Path("configs/v127-sgd-typed-constraint-feasibility.json").read_text())
        inventory = json.loads(Path(config["sourceInventory"]).read_text())
        excluded = json.loads(Path(config["excludedV125Population"]).read_text())
        catalog = json.loads(Path(config["choiceCatalog"]).read_text())
        population = select_fresh_population(inventory, excluded, catalog, config)
        self.assertTrue(all(population_gates(population, config).values()))
        self.assertNotIn("utterance", set().union(*(row.keys() for row in population["records"])))

    def test_signature_rule_is_unique_and_threshold_free(self):
        signatures = {
            "svc::a": {"required": {"x"}, "allowed": {"x", "y"}},
            "svc::b": {"required": {"z"}, "allowed": {"z"}},
        }
        unique = signature_decision({"x"}, signatures)
        self.assertEqual(unique["candidate_intent"], "svc::a")
        self.assertFalse(unique["query"])
        empty = signature_decision(set(), signatures)
        self.assertTrue(empty["query"])

    def test_evaluation_emits_aggregates_only(self):
        config = json.loads(Path("configs/v127-sgd-typed-constraint-feasibility.json").read_text())
        catalog = json.loads(Path(config["choiceCatalog"]).read_text())
        baseline = json.loads(Path(config["baselineConfig"]).read_text())
        v119 = json.loads(Path(config["V119Config"]).read_text())
        known = [row for row in catalog["choices"] if row["kind"] == "KNOWN"]
        novel = next(row for row in catalog["choices"] if row["kind"] == "NOVEL_COMPOSITE")
        unsupported = next(row for row in catalog["choices"] if row["kind"] == "UNSUPPORTED_COMPOSITE")
        records = [
            {"record_id": "k", "class_label": "known", "domain": known[0]["domain"], "service": known[0]["service"], "intent": known[0]["intent"]},
            {"record_id": "n", "class_label": "novel_valid", "domain": novel["domain"], "service": "new", "intent": "new"},
            {"record_id": "u", "class_label": "unsupported", "domain": unsupported["domains"][0], "service": "out", "intent": "out"},
        ]
        signatures = {
            row["intent_id"]: {"required": {f"s{i}"}, "allowed": {f"s{i}"}}
            for i, row in enumerate(known)
        }
        evidence = {"k": {"s0"}, "n": set(), "u": set()}
        result = run_evaluation(records, evidence, signatures, catalog, baseline, v119, config)
        self.assertEqual(len(result["conditions"]), 9)
        self.assertNotIn("records", result)
        self.assertEqual(result["individual_record_emission_count"], 0)


if __name__ == "__main__":
    unittest.main()
