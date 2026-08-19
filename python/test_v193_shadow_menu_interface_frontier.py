from __future__ import annotations

import unittest

from v193_shadow_menu_interface_frontier import normalize_proposal


class V193InterfaceTest(unittest.TestCase):
    def test_valid_ranking_is_preserved(self) -> None:
        self.assertEqual(
            normalize_proposal('{"status":"RANKED","ranked_option_ids":["M02","M01"]}', {"M01", "M02"}),
            {"status": "RANKED", "ranked_option_ids": ["M02", "M01"]},
        )

    def test_truncation_duplicate_and_extra_key_fail_closed(self) -> None:
        valid = {"M01", "M02"}
        cases = [
            '{"status":"RANKED"',
            {"status": "RANKED", "ranked_option_ids": ["M01", "M01"]},
            {"status": "RANKED", "ranked_option_ids": ["M01"], "confidence": 0.9},
        ]
        self.assertTrue(all(normalize_proposal(value, valid)["status"] == "INSUFFICIENT" for value in cases))


if __name__ == "__main__":
    unittest.main()
