from __future__ import annotations

import json
import unittest

from v22r2_grounding import PROJECT_ROOT
from v184_sgd_role_isolated_language_extraction import _forbidden_key_count


class V184RoleIsolatedLanguageExtractionTest(unittest.TestCase):
    def test_recursive_forbidden_key_detection(self) -> None:
        value = {"record_id": "x", "conversation": [{"utterance": "hello"}], "nested": {"truth_kind": "KNOWN"}}
        self.assertEqual(_forbidden_key_count(value, {"truth_kind", "service"}), 1)

    def test_observable_and_catalog_contracts_are_closed(self) -> None:
        config = json.loads(
            (PROJECT_ROOT / "configs/v184-sgd-role-isolated-language-extraction.json").read_text()
        )
        self.assertIn("truth_contract_id", config["observableRecordContract"]["forbiddenRecordFields"])
        self.assertFalse(config["declaredCatalogContract"]["includeProvisionalOrUnsupportedSchemaLanguage"])
        self.assertFalse(config["decisionRule"]["passAuthorizesImmediateDevelopmentLanguageScoring"])
        self.assertFalse(config["decisionRule"]["passAuthorizesProtectedLanguageReading"])


if __name__ == "__main__":
    unittest.main()
