from __future__ import annotations

import unittest

from v191_fresh_language_to_menu_population import build_population


class V191PopulationTest(unittest.TestCase):
    def test_selection_is_dialogue_disjoint_and_deterministic(self) -> None:
        contracts = {
            "contracts": [
                {
                    "capability_contract_id": "C1",
                    "source_definition_ids": ["S_1::I"],
                    "truth_kinds": ["KNOWN"],
                }
            ]
        }
        candidates = [
            {
                "candidate_id": f"sgd::dev::d{i}::000::S_1::I",
                "partition": "dev",
                "service": "S_1",
                "intent": "I",
            }
            for i in range(8)
        ]
        config = {
            "freshnessContract": {"sourcePartition": "dev", "baseSalt": "x"},
            "population": {
                "role": "development",
                "sourceRecordsPerContract": 2,
                "requiredMissingControlCount": 1,
            },
        }
        previous = {
            "records": [
                {
                    "observation_available": True,
                    "source_candidate_id": "sgd::dev::d0::000::S_1::I",
                }
            ]
        }
        first = build_population({"candidate_index": candidates}, contracts, previous, config)
        second = build_population({"candidate_index": candidates}, contracts, previous, config)
        self.assertEqual(first, second)
        observed = [row for row in first["hidden_targets"]["records"] if row["observation_available"]]
        self.assertEqual(len({row["source_dialogue_id"] for row in observed}), 2)
        self.assertNotIn("d0", {row["source_dialogue_id"] for row in observed})


if __name__ == "__main__":
    unittest.main()
