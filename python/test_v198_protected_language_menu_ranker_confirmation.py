from __future__ import annotations

import json
import unittest

from v22r2_grounding import PROJECT_ROOT


class V198InheritanceTest(unittest.TestCase):
    def test_model_prompt_and_qualification_gates_equal_V195(self) -> None:
        current = json.loads((PROJECT_ROOT / "configs/v198-protected-language-menu-ranker-confirmation.json").read_text())
        old_lock = json.loads((PROJECT_ROOT / current["sourceV195OutcomeLock"]).read_text())
        old = json.loads((PROJECT_ROOT / old_lock["experiment_lock"]).read_text())["config_payload"]
        self.assertEqual(current["model"], old["model"])
        self.assertEqual(current["prompt"], old["prompt"])
        self.assertEqual(current["qualificationGates"], old["qualificationGates"])


if __name__ == "__main__":
    unittest.main()
