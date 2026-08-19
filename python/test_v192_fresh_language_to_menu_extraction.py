from __future__ import annotations

import unittest

from v192_fresh_language_to_menu_extraction import _forbidden_key_count


class V192ExtractionTest(unittest.TestCase):
    def test_forbidden_gold_keys_are_recursive(self) -> None:
        value = {"records": [{"conversation": [{"speaker": "USER", "utterance": "x"}], "target_contract_id": "C"}]}
        self.assertEqual(_forbidden_key_count(value, {"target_contract_id", "service"}), 1)


if __name__ == "__main__":
    unittest.main()
