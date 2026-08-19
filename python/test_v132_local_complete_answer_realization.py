import json
import unittest
from pathlib import Path

from v132_local_complete_answer_realization import render_prompt, validate_answer, wilson_lower


class V132CompleteAnswerTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path("configs/v132-local-complete-answer-realization.json").read_text())
        self.catalog = json.loads(Path(self.config["choiceCatalog"]).read_text())
        self.prompt_choices = [
            {"choice_id": row["choice_id"], "kind": row["kind"], "domain": row.get("domain")}
            for row in self.catalog["choices"]
        ]

    def test_prompt_contains_complete_menu_without_truth(self):
        payload = json.loads(render_prompt(self.prompt_choices, "find a movie", True, "K01", self.config))
        self.assertEqual(len(payload["choices"]), 11)
        self.assertEqual(payload["preliminary_candidate_under_review"], "K01")
        self.assertNotIn("truth_choice_id", payload)

    def test_strict_answer_validation(self):
        self.assertEqual(validate_answer('{"choice_id":"K01"}', self.catalog), ("K01", True, "valid"))
        self.assertEqual(validate_answer('{"choice_id":"K01","note":"x"}', self.catalog)[0:2], ("A00", False))
        self.assertEqual(validate_answer('not json', self.catalog)[0:2], ("A00", False))

    def test_wilson_lower_is_bounded(self):
        self.assertGreater(wilson_lower(264, 264), 0.98)
        self.assertLess(wilson_lower(0, 264), 0.01)


if __name__ == "__main__":
    unittest.main()
