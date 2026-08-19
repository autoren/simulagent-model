from __future__ import annotations

import unittest


class V201r1ScopeTest(unittest.TestCase):
    def test_repair_scope_is_single_volatile_field(self) -> None:
        import json
        from v22r2_grounding import PROJECT_ROOT
        config = json.loads((PROJECT_ROOT / "configs/v201r1-elapsed-time-verification-repair.json").read_text())
        self.assertEqual(config["repairContract"]["allowedTopLevelSummaryDifferenceKeys"], ["elapsed_seconds"])
        self.assertFalse(config["repairContract"]["sourceArtifactsMayBeModified"])
        self.assertFalse(config["repairContract"]["modelOrPolicyMayBeRerun"])


if __name__ == "__main__":
    unittest.main()
