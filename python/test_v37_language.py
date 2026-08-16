import copy
import json
import unittest

from v22r2_grounding import PROJECT_ROOT
from generate_v37_semantic_invariance import build_fit_sample, build_validation, corpus_hash
from v37_language import candidate_prompt, normalized_template, validate_registry


class V37LanguageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((PROJECT_ROOT / "configs/v37-semantic-invariance.json").read_text())
        cls.v32 = json.loads((PROJECT_ROOT / "configs/v32-factorized-semantics.json").read_text())

    def test_registry_and_population(self):
        validate_registry(self.config)
        fit = build_fit_sample(self.config, self.v32)
        validation = build_validation(self.config, self.v32)
        self.assertEqual(len(fit), 400)
        self.assertEqual(len(validation), 360)
        self.assertEqual(len({row["scene_id"] for row in validation}), 100)
        self.assertEqual(len({row["oracle_metadata"]["surface_family"] for row in validation}), 10)
        self.assertEqual(len(corpus_hash(validation)), 64)

    def test_candidate_prompts_are_target_independent(self):
        row = build_validation(self.config, self.v32)[0]
        changed = copy.deepcopy(row)
        changed["target"] = {"sentinel": "must not be read"}
        for component, candidates in (
            ("lexical_sign", self.config["interfaces"]["lexicalSignClasses"]),
            ("outer_operation", self.config["interfaces"]["outerOperationClasses"]),
        ):
            for candidate in candidates:
                self.assertEqual(
                    candidate_prompt(row, component, candidate),
                    candidate_prompt(changed, component, candidate),
                )

    def test_templates_are_unique(self):
        values = {
            normalized_template(operation, surface)
            for operation in self.config["interfaces"]["outerOperationClasses"]
            for surface in self.config["freshValidation"]["surfaceNamesPerOperation"]
        }
        self.assertEqual(len(values), 10)


if __name__ == "__main__":
    unittest.main()
