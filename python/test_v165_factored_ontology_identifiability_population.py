from __future__ import annotations

import json
from pathlib import Path
import unittest

from v165_factored_ontology_identifiability_population import (
    audit_population,
    build_population,
    candidate_universe,
    enumerate_version_space,
    parse_definition,
    registered_tables,
    valuations,
)


class V165FactoredOntologyPopulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(
            Path(
                "configs/v165-factored-ontology-identifiability-population.json"
            ).read_text()
        )
        cls.population = build_population(cls.config)

    def test_candidate_universe_and_registered_partition_are_exact(self) -> None:
        universe = candidate_universe()
        self.assertEqual(len(valuations()), 8)
        self.assertEqual(len(universe), 256)
        self.assertEqual(len(registered_tables()), 9)
        counts = {
            name: sum(row["expressibility_class"] == name for row in universe)
            for name in ("alias", "composition", "provisional_primitive")
        }
        self.assertEqual(
            counts,
            {"alias": 3, "composition": 6, "provisional_primitive": 247},
        )

    def test_population_is_balanced_and_development_only(self) -> None:
        summary = self.population["population_summary"]
        self.assertEqual(summary["record_count"], 144)
        self.assertEqual(summary["cell_count"], 9)
        self.assertEqual(set(summary["cell_counts"].values()), {16})
        self.assertEqual(summary["logical_target_group_count"], 36)
        self.assertEqual(summary["evaluation_record_count"], 0)

    def test_sufficient_ambiguous_and_contradictory_contracts_are_exact(self) -> None:
        rows = self.population["hidden_records"]
        sufficient = [row for row in rows if row["evidence_status"] == "sufficient"]
        ambiguous = [row for row in rows if row["evidence_status"] == "ambiguous"]
        contradictory = [
            row for row in rows if row["evidence_status"] == "contradictory"
        ]
        self.assertTrue(all(row["version_space_size"] == 1 for row in sufficient))
        self.assertTrue(
            all(
                row["version_space_size"] >= 2
                and set(row["version_space_classes"])
                == {"alias", "composition", "provisional_primitive"}
                for row in ambiguous
            )
        )
        self.assertTrue(all(row["version_space_size"] == 0 for row in contradictory))
        self.assertTrue(
            all(
                row["target_candidate_id"] in row["version_space_candidate_ids"]
                for row in sufficient + ambiguous
            )
        )

    def test_definitions_reparse_and_version_spaces_reconstruct(self) -> None:
        namespaces = {
            row["namespace_id"]: row for row in self.config["primitiveNamespaces"]
        }
        for row in self.population["hidden_records"]:
            namespace = namespaces[row["namespace_id"]]
            parsed = parse_definition(
                row["definition"], namespace, row["concept_name"]
            )
            version = enumerate_version_space(
                parsed, row["observations"], namespace
            )
            self.assertEqual(parsed, row["definition_parse"])
            self.assertEqual(
                [candidate["candidate_id"] for candidate in version],
                row["version_space_candidate_ids"],
            )

    def test_public_records_contain_no_hidden_contract_fields(self) -> None:
        forbidden = set(self.config["hiddenFields"])
        self.assertTrue(
            all(not (set(row) & forbidden) for row in self.population["public_records"])
        )
        self.assertTrue(
            all(
                set(row) == set(self.config["publicFields"])
                for row in self.population["public_records"]
            )
        )

    def test_full_population_audit_passes(self) -> None:
        result = audit_population(self.population, self.config)
        self.assertTrue(result["passed"], result["checks"])
        self.assertEqual(result["target_retention_when_noncontradictory"], 1.0)
        self.assertEqual(result["evidence_status_classification_accuracy"], 1.0)
        self.assertEqual(result["renaming_version_space_invariance"], 1.0)


if __name__ == "__main__":
    unittest.main()
