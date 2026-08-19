from __future__ import annotations

import json
import unittest

from v22r2_grounding import PROJECT_ROOT
from v196_protected_confirmation_role_binding import audit_binding, build_binding


class V196BindingTest(unittest.TestCase):
    def test_frozen_metadata_builds_expected_dialogue_isolated_role(self) -> None:
        config = json.loads((PROJECT_ROOT / "configs/v196-protected-confirmation-role-binding.json").read_text())
        built = build_binding(
            json.loads((PROJECT_ROOT / config["sourceInventory"]).read_text()),
            json.loads((PROJECT_ROOT / config["contractCatalog"]).read_text()),
            json.loads((PROJECT_ROOT / config["V183HiddenIdentifiability"]).read_text()),
            json.loads((PROJECT_ROOT / config["V191HiddenTargets"]).read_text()),
            config,
        )
        audit = audit_binding(built, config)
        self.assertTrue(audit["passed"])
        self.assertFalse(audit["summary"]["all_dev_confirmation_feasible"])
        self.assertEqual(audit["summary"]["source_record_count"], 113)
        self.assertEqual(audit["summary"]["within_confirmation_dialogue_overlap"], 0)


if __name__ == "__main__":
    unittest.main()
