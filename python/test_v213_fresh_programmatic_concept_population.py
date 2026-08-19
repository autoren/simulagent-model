from __future__ import annotations

import json
import unittest
from pathlib import Path

from v213_fresh_programmatic_concept_population import (
    _flip_identifier,
    _opaque,
    _rotate,
    project_public_blueprints,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/v213-fresh-programmatic-concept-population.json").read_text())


class V213PopulationTests(unittest.TestCase):
    def test_opaque_identifiers_are_deterministic_and_role_neutral(self) -> None:
        first = _opaque("group", "salt", "family", 1)
        second = _opaque("group", "salt", "family", 1)
        self.assertEqual(first, second)
        self.assertRegex(first, r"^group-[0-9a-f]{20}$")
        self.assertNotIn("family", first)

    def test_bit_flip_and_rotation_are_exact(self) -> None:
        self.assertEqual("TT-10001111", _flip_identifier("TT-00001111", 0))
        self.assertEqual([2, 3, 1], _rotate([1, 2, 3], 1))
        self.assertEqual([], _rotate([], 7))

    def test_public_projection_reads_only_public_record_payload(self) -> None:
        blueprints = [
            {"blueprint_index": 1, "public_record": {"case_id": "case-b", "value": 2}},
            {"blueprint_index": 0, "public_record": {"case_id": "case-a", "value": 1}},
        ]
        self.assertEqual(
            [{"case_id": "case-a", "value": 1}, {"case_id": "case-b", "value": 2}],
            project_public_blueprints(blueprints),
        )

    def test_frozen_counts_are_internally_consistent(self) -> None:
        design = CONFIG["populationDesign"]
        split = CONFIG["splitDesign"]
        self.assertEqual(
            design["groupCount"] * design["variantsPerGroup"], design["recordCount"]
        )
        self.assertEqual(
            split["developmentGroupCount"] + split["protectedGroupCount"],
            design["groupCount"],
        )


if __name__ == "__main__":
    unittest.main()
