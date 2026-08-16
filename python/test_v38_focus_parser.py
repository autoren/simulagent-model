import copy
import json
import unittest

from v22r2_grounding import PROJECT_ROOT
from generate_v38_focus_parser import build_population
from v38_focus_parser import candidate_prompt, deterministic_focus_index, extract_literal_candidates


class V38FocusParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((PROJECT_ROOT / "configs/v38-ontology-anchored-focus-parser.json").read_text())
        cls.v32 = json.loads((PROJECT_ROOT / "configs/v32-factorized-semantics.json").read_text())

    def test_exact_population_and_controls(self):
        for split in ("ontology_focus_fit", "ontology_focus_validation"):
            rows = build_population(self.config, self.v32, split)
            self.assertEqual(len(rows), 240)
            self.assertEqual({row["oracle_metadata"]["grounded_literal_candidates"] for row in rows}, {1, 2})
            self.assertEqual({row["oracle_metadata"]["focus_order"] for row in rows}, {"focus_first", "focus_second"})

    def test_deterministic_parser_selects_registered_focus(self):
        for row in build_population(self.config, self.v32, "ontology_focus_validation"):
            candidates = extract_literal_candidates(row)
            self.assertEqual(deterministic_focus_index(row, candidates), row["target"]["focus_candidate_index"])

    def test_candidate_prompt_does_not_read_target(self):
        row = build_population(self.config, self.v32, "ontology_focus_fit")[0]
        candidate = extract_literal_candidates(row)[0]
        changed = copy.deepcopy(row); changed["target"] = {"sentinel": True}
        self.assertEqual(candidate_prompt(row, candidate), candidate_prompt(changed, candidate))


if __name__ == "__main__":
    unittest.main()
