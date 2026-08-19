from __future__ import annotations
import json, unittest
from v22r2_grounding import PROJECT_ROOT


class V201r2ScopeTest(unittest.TestCase):
    def test_scope_is_serialization_only(self) -> None:
        c = json.loads((PROJECT_ROOT / "configs/v201r2-repair-decision-label-verification.json").read_text())
        self.assertEqual(c["repairContract"]["requiredFalseChecks"], ["repair_reconstructs_exactly"])
        self.assertFalse(c["repairContract"]["sourceArtifactsMayBeModified"])
        self.assertFalse(c["repairContract"]["modelPolicyOrScoringMayBeRerun"])


if __name__ == "__main__": unittest.main()
