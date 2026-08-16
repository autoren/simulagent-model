import json
import unittest
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from audit_v32_factorized_semantics import audit
from generate_v32_factorized_semantics import build_records, corpus_hash
from v10_protocol import file_sha256
from v32_language import compile_truth, representation_prompt_layout
from v32_structured_model import select_predictions, target_arrays


class V32FactorizedSemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_path = Path("configs/v32-factorized-semantics.json")
        cls.config = json.loads(cls.config_path.read_text())
        cls.rows = build_records(cls.config)

    def test_registered_population(self):
        self.assertEqual(len(self.rows), 3536)
        self.assertEqual(len({row["scene_id"] for row in self.rows}), 1088)
        self.assertEqual(Counter(row["split"] for row in self.rows), {
            "factor_fit": 1456, "factor_calibration": 364,
            "factor_evaluation_paraphrase": 1092,
            "factor_evaluation_composition": 624,
        })

    def test_truth_compiler_is_exhaustive(self):
        cells = {
            (operation, sign): compile_truth(sign, operation, self.config)
            for operation in self.config["factorization"]["outerOperations"]
            for sign in self.config["factorization"]["lexicalSigns"]
        }
        self.assertEqual(len(cells), 10)
        self.assertEqual(cells[("deny", "negative")], "true")
        self.assertEqual(cells[("double_deny", "negative")], "false")
        self.assertEqual(cells[("unresolved", "negative")], "unknown")

    def test_composition_cells_are_held_out_but_primitives_are_supported(self):
        fit = {
            (row["target"]["factorization"]["outer_operation"], row["target"]["factorization"]["lexical_sign"])
            for row in self.rows if row["split"] == "factor_fit"
        }
        composition = {
            (row["target"]["factorization"]["outer_operation"], row["target"]["factorization"]["lexical_sign"])
            for row in self.rows if row["split"] == "factor_evaluation_composition"
        }
        self.assertEqual(composition - fit, {
            ("deny", "negative"), ("double_deny", "negative"),
            ("contrast_select", "negative"),
        })
        self.assertEqual({value for cell in fit for value in cell[1:]}, {"positive", "negative"})
        self.assertEqual({operation for operation, _ in fit}, set(self.config["factorization"]["outerOperations"]))

    def test_controlled_pairs_are_complete(self):
        groups = defaultdict(list)
        for row in self.rows:
            for pair in row["oracle_metadata"]["pairs"]:
                groups[(pair["kind"], pair["id"])].append(row)
        self.assertEqual({len(values) for values in groups.values()}, {2})
        self.assertEqual({kind for kind, _ in groups}, {
            "distractor", "inverse", "argument_reversal", "lexical_sign_assert",
            "unresolved_sign_invariance", "scope_assert_deny",
            "scope_assert_double_deny", "scope_assert_contrast",
        })

    def test_prompt_has_entities_but_no_factor_labels(self):
        for row in self.rows[:100]:
            prompt, spans = representation_prompt_layout(row, self.config)
            self.assertTrue(all(spans.values()))
            self.assertNotIn("lexical_sign", prompt)
            self.assertNotIn("outer_operation", prompt)

    def test_in_memory_audit_passes(self):
        manifest = {
            "config_sha256": file_sha256(self.config_path),
            "corpus_sha256": corpus_hash(self.rows), "artifact_sha256": {},
        }
        result = audit(
            self.rows, self.config, manifest, self.config_path.resolve(),
            enforce_firewall=False,
        )
        self.assertTrue(result["passed"], result["errors"])

    def test_compiled_decoder_differs_only_at_registered_truth_boundary(self):
        row = next(
            row for row in self.rows
            if row["split"] == "factor_evaluation_composition"
            and row["target"]["factorization"] == {
                "lexical_sign": "negative", "outer_operation": "deny",
            }
        )
        target = target_arrays(row, self.config)
        maximum = max(self.config["construction"]["entityCounts"])
        shapes = (5, maximum, maximum + 1, 3, 2, 5)
        outputs = [np.zeros((1, size), dtype=np.float32) for size in shapes]
        for output, key in zip(
            outputs, ("predicate", "argument1", "argument2"), strict=False
        ):
            output[0, target[key]] = 10.0
        outputs[3][0, self.config["sharedHead"]["truthClasses"].index("false")] = 10.0
        outputs[4][0, target["lexicalSign"]] = 10.0
        outputs[5][0, target["outerOperation"]] = 10.0
        direct = select_predictions(
            [row], tuple(outputs), self.config, "direct_truth_head"
        )[0]
        compiled = select_predictions(
            [row], tuple(outputs), self.config, "fixed_registered_truth_compiler"
        )[0]
        self.assertEqual(direct["selected_fields"]["truth_status"], "false")
        self.assertEqual(compiled["selected_fields"]["truth_status"], "true")
        self.assertEqual(
            {key: value for key, value in direct["selected_fields"].items() if key != "truth_status"},
            {key: value for key, value in compiled["selected_fields"].items() if key != "truth_status"},
        )


if __name__ == "__main__": unittest.main()
