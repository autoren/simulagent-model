from __future__ import annotations

import unittest

from v161_fresh_massive_transfer_population import (
    evaluate_population_gates,
    select_transfer_population,
)
from v93_open_set_source import canonical_sha256


def fixture() -> tuple[dict, dict, dict]:
    rows = []
    cursor = 0
    scenarios = {
        "known_familiar": ("a", "b", "c"),
        "known_unfamiliar": ("a", "b", "c"),
        "novel_valid": ("a", "b"),
        "unsupported": ("z",),
    }
    excluded_rows = []
    for partition in ("validation", "test"):
        for label, scenario_names in scenarios.items():
            for scenario in scenario_names:
                for index in range(8):
                    intent_count = 2 if label == "novel_valid" else 4
                    row = {
                        "candidate_id": f"massive::{cursor}",
                        "source_id": str(cursor),
                        "partition": partition,
                        "scenario": scenario,
                        "intent": f"{label}_intent_{index % intent_count}",
                        "class_label": label,
                        "current_utterance_intent_overlap_count": int(label == "known_familiar"),
                        "slot_type_count": 1,
                    }
                    rows.append(row)
                    if index == 0:
                        excluded_rows.append(
                            {
                                "candidate_id": row["candidate_id"],
                                "population_id": f"old::{row['candidate_id']}",
                            }
                        )
                    cursor += 1
    inventory = {"candidate_index": rows, "candidate_index_sha256": canonical_sha256(rows)}
    excluded = {
        "selected_population": excluded_rows,
        "selected_population_sha256": canonical_sha256(excluded_rows),
    }
    config = {
        "sourceCandidateIndexSha256": inventory["candidate_index_sha256"],
        "excludedPopulationPayloadSha256": excluded["selected_population_sha256"],
        "requiredClasses": list(scenarios),
        "selection": {
            "baseSalt": "v161-test-salt",
            "selectedCandidateCountPerClassPerRole": 6,
            "scenarioMinimumPerClass": {
                "known_familiar": 1,
                "known_unfamiliar": 1,
                "novel_valid": 1,
                "unsupported": 0,
            },
            "roles": {
                "development_transfer": {"sourcePartition": "validation"},
                "protected_transfer": {"sourcePartition": "test"},
            },
        },
        "populationGates": {
            "requiredExcludedCandidateCount": len(excluded_rows),
            "requiredCandidateCountPerClassPerRole": 6,
            "requiredCandidateCountPerRole": 24,
            "requiredTotalCandidateCount": 48,
            "minimumRemainingCandidateCountPerClassPerRole": 6,
            "requiredKnownScenarioCoverage": 3,
            "requiredNovelScenarioCoverage": 2,
            "requiredUnsupportedScenarioCoverage": 1,
            "minimumIntentCoveragePerClass": {
                "known_familiar": 2,
                "known_unfamiliar": 2,
                "novel_valid": 2,
                "unsupported": 2,
            },
            "requiredOverlapWithExcludedPopulation": 0,
            "maximumTrainPartitionCandidateCount": 0,
        },
    }
    return inventory, excluded, config


class V161FreshMassiveTransferPopulationTests(unittest.TestCase):
    def test_selection_is_deterministic_balanced_disjoint_and_text_free(self):
        inventory, excluded, config = fixture()
        first = select_transfer_population(inventory, excluded, config)
        second = select_transfer_population(inventory, excluded, config)
        self.assertEqual(first, second)
        self.assertEqual(first["selected_candidate_count"], 48)
        self.assertEqual(first["excluded_population_overlap_count"], 0)
        self.assertTrue(first["role_identifiers_are_disjoint"])
        self.assertFalse(first["contains_language_tokens_slot_values_or_prompts"])
        self.assertTrue(all(evaluate_population_gates(first, config).values()))

    def test_every_excluded_identifier_is_removed_before_selection(self):
        inventory, excluded, config = fixture()
        population = select_transfer_population(inventory, excluded, config)
        old_ids = {row["candidate_id"] for row in excluded["selected_population"]}
        new_ids = {row["candidate_id"] for row in population["selected_population"]}
        self.assertFalse(old_ids & new_ids)

    def test_source_and_exclusion_identities_are_required(self):
        inventory, excluded, config = fixture()
        bad_source = dict(config, sourceCandidateIndexSha256="bad")
        with self.assertRaises(ValueError):
            select_transfer_population(inventory, excluded, bad_source)
        bad_exclusion = dict(config, excludedPopulationPayloadSha256="bad")
        with self.assertRaises(ValueError):
            select_transfer_population(inventory, excluded, bad_exclusion)

    def test_overlap_gate_is_noncompensatory(self):
        inventory, excluded, config = fixture()
        population = select_transfer_population(inventory, excluded, config)
        population["excluded_population_overlap_count"] = 1
        checks = evaluate_population_gates(population, config)
        self.assertFalse(checks["zero_overlap_with_excluded_population"])


if __name__ == "__main__":
    unittest.main()
