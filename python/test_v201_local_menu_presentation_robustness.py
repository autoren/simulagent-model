from __future__ import annotations

import json
import unittest

from v201_local_menu_presentation_robustness import _jaccard
from v22r2_grounding import PROJECT_ROOT


class V201InheritanceTest(unittest.TestCase):
    def test_model_prompt_and_gates_are_inherited_exactly(self) -> None:
        current = json.loads((PROJECT_ROOT / "configs/v201-local-menu-presentation-robustness.json").read_text())
        v195_outcome = json.loads((PROJECT_ROOT / current["sourceV195OutcomeLock"]).read_text())
        v195 = json.loads((PROJECT_ROOT / v195_outcome["experiment_lock"]).read_text())["config_payload"]
        v199_outcome = json.loads((PROJECT_ROOT / current["sourceV199OutcomeLock"]).read_text())
        v199 = json.loads((PROJECT_ROOT / v199_outcome["experiment_lock"]).read_text())["config_payload"]
        self.assertEqual(current["model"], v195["model"])
        self.assertEqual(current["prompt"], v195["prompt"])
        self.assertEqual(current["qualificationGates"], v199["futurePairedDevelopmentGates"])

    def test_empty_set_jaccard_is_exact(self) -> None:
        self.assertEqual(_jaccard(set(), set()), 1.0)
        self.assertEqual(_jaccard({"A"}, set()), 0.0)


if __name__ == "__main__":
    unittest.main()
