import json
import unittest

from v22r2_grounding import PROJECT_ROOT
from generate_v40_confirmation import PACK_NAMES, build_populations, ontology_pack


class V40ConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((PROJECT_ROOT / "configs/v40-independent-compiler-confirmation.json").read_text())
        cls.populations = build_populations(cls.config)

    def test_fixed_population_sizes(self):
        self.assertEqual(len(self.populations["independent_confirmation"]), 1440)
        self.assertEqual(len(self.populations["independent_safety"]), 120)

    def test_each_pack_has_full_registered_factor_cross(self):
        rows = self.populations["independent_confirmation"]
        for pack in PACK_NAMES:
            selected = [row for row in rows if row["ontology_pack"] == pack]
            self.assertEqual(len(selected), 120)
            cells = {
                (row["oracle_metadata"]["operation"], row["oracle_metadata"]["focus_order"], row["oracle_metadata"]["decoy_kind"], row["oracle_metadata"]["orientation"], row["oracle_metadata"]["sign"])
                for row in selected
            }
            self.assertEqual(len(cells), 120)

    def test_pack_lexicons_are_fresh_and_disjoint(self):
        identifiers = []
        forms = []
        for index in range(len(PACK_NAMES)):
            ontology, cues = ontology_pack(index)
            identifiers.extend(row["id"] for row in ontology["relations"])
            forms.extend(row["direct_positive_form"] for row in ontology["relations"])
            forms.extend(cues.values())
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertEqual(len(forms), len(set(forms)))

    def test_no_targets_in_agent_input(self):
        self.assertFalse(any("target" in row["agent_input"] or "expected" in row["agent_input"] for rows in self.populations.values() for row in rows))


if __name__ == "__main__":
    unittest.main()
