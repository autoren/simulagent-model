from __future__ import annotations

import unittest

from v101_massive_population import evaluate_population_gates, select_massive_population
from v93_open_set_source import canonical_sha256


def fixture() -> tuple[dict, dict]:
    rows = []
    cursor = 0
    scenarios = {
        "known_familiar": ("a", "b", "c"),
        "known_unfamiliar": ("a", "b", "c"),
        "novel_valid": ("a", "b"),
        "unsupported": ("z",),
    }
    for partition in ("validation", "test"):
        for label, scenario_names in scenarios.items():
            for scenario in scenario_names:
                for index in range(12):
                    intent = f"{label}_intent_{index % (4 if label != 'novel_valid' else 2)}"
                    rows.append({
                        "candidate_id": f"massive::{cursor}", "source_id": str(cursor),
                        "partition": partition, "scenario": scenario, "intent": intent,
                        "class_label": label,
                        "current_utterance_intent_overlap_count": int(label == "known_familiar"),
                        "slot_type_count": 1,
                    })
                    cursor += 1
    inventory = {"candidate_index": rows, "candidate_index_sha256": canonical_sha256(rows)}
    config = {
        "sourceCandidateIndexSha256": inventory["candidate_index_sha256"],
        "requiredClasses": list(scenarios),
        "selection": {
            "baseSalt": "test", "selectedCandidateCountPerClassPerSplit": 12,
            "scenarioMinimumPerClass": {
                "known_familiar": 2, "known_unfamiliar": 2,
                "novel_valid": 2, "unsupported": 0,
            },
            "roles": {
                "development": {"sourcePartition": "validation"},
                "protected_test": {"sourcePartition": "test"},
            },
        },
        "populationGates": {
            "requiredCandidateCountPerClassPerSplit": 12,
            "requiredCandidateCountPerSplit": 48,
            "requiredTotalCandidateCount": 96,
            "requiredKnownScenarioCoverage": 3,
            "requiredNovelScenarioCoverage": 2,
            "requiredUnsupportedScenarioCoverage": 1,
            "minimumIntentCoveragePerClass": {
                "known_familiar": 3, "known_unfamiliar": 3,
                "novel_valid": 2, "unsupported": 3,
            },
            "maximumTrainPartitionCandidateCount": 0,
        },
    }
    return inventory, config


class V101PopulationTests(unittest.TestCase):
    def test_selection_is_deterministic_balanced_and_text_free(self) -> None:
        inventory, config = fixture()
        first = select_massive_population(inventory, config)
        second = select_massive_population(inventory, config)
        self.assertEqual(first, second)
        self.assertEqual(first["selected_candidate_count"], 96)
        self.assertTrue(first["development_test_identifiers_are_disjoint"])
        self.assertFalse(first["contains_language_tokens_slot_values_or_prompts"])
        self.assertNotIn("utt", set(first["selected_population"][0]))
        self.assertNotIn("annot_utt", set(first["selected_population"][0]))
        self.assertTrue(all(evaluate_population_gates(first, config).values()))

    def test_candidate_index_identity_is_required(self) -> None:
        inventory, config = fixture()
        config["sourceCandidateIndexSha256"] = "bad"
        with self.assertRaises(ValueError):
            select_massive_population(inventory, config)

    def test_split_gate_is_noncompensatory(self) -> None:
        inventory, config = fixture()
        population = select_massive_population(inventory, config)
        population["role_class_counts"]["protected_test"]["unsupported"] = 11
        checks = evaluate_population_gates(population, config)
        self.assertFalse(checks["protected_test_unsupported_candidate_count"])


if __name__ == "__main__":
    unittest.main()
