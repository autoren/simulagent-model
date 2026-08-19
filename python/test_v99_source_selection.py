from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from v99_source_selection import audit_source_selection


ROOT = Path(__file__).resolve().parents[1]


class V99SourceSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(
            (ROOT / "configs/v99-open-world-source-selection.json").read_text()
        )

    def test_locked_selection_passes(self) -> None:
        result = audit_source_selection(self.config)
        self.assertTrue(result["passed"])
        self.assertTrue(all(result["checks"].values()))

    def test_source_identity_cannot_be_used_as_class(self) -> None:
        altered = copy.deepcopy(self.config)
        altered["decision"]["mixSourceIdentityAsClassFeature"] = True
        self.assertFalse(audit_source_selection(altered)["passed"])

    def test_synthetic_context_cannot_enter_external_evaluation(self) -> None:
        altered = copy.deepcopy(self.config)
        altered["presto"]["pairRules"]["syntheticContextAllowed"] = True
        self.assertFalse(audit_source_selection(altered)["passed"])

    def test_any_premature_payload_or_model_access_fails(self) -> None:
        for key in ("payloadDownloadCount", "languageRecordInspectionCount", "modelLoadCount"):
            altered = copy.deepcopy(self.config)
            altered["access"][key] = 1
            self.assertFalse(audit_source_selection(altered)["passed"])


if __name__ == "__main__":
    unittest.main()
