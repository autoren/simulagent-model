from __future__ import annotations

import json
import unittest

from v22r2_grounding import PROJECT_ROOT
from v167r1_history_action_metric_repair import corrected_history_count, corrected_summary


class V167r1HistoryActionMetricRepairTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads((PROJECT_ROOT / "outputs/v167-exact-evidence-gathering/planner/result.json").read_text())
        cls.trees = json.loads((PROJECT_ROOT / "outputs/v167-exact-evidence-gathering/planner/case-policy-trees.json").read_text())

    def test_original_overcount_is_preserved(self) -> None:
        self.assertEqual(self.result["summary"]["history_dependent_second_action_case_count"], 48)

    def test_action_only_projection_is_28(self) -> None:
        self.assertEqual(corrected_history_count(self.trees["cases"]), 28)

    def test_only_summary_field_changes(self) -> None:
        evaluation = {"cases": self.trees["cases"], "summary": self.result["summary"]}
        repaired = corrected_summary(evaluation)
        differing = {key for key in repaired if repaired[key] != self.result["summary"][key]}
        self.assertEqual(differing, {"history_dependent_second_action_case_count"})


if __name__ == "__main__":
    unittest.main()
