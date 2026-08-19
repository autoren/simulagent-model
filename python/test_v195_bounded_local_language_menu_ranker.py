from __future__ import annotations

import unittest

from v195_bounded_local_language_menu_ranker import parse_response


class V195ModelInterfaceTest(unittest.TestCase):
    def test_exact_three_option_response_is_valid(self) -> None:
        parsed = parse_response(
            '{"status":"RANKED","ranked_option_ids":["M03","M02","M01"]}',
            {"M01", "M02", "M03"},
        )
        self.assertTrue(parsed["structural_valid"])
        self.assertEqual(parsed["normalized_proposal"]["ranked_option_ids"], ["M03", "M02", "M01"])

    def test_truncated_or_short_ranking_fails_closed(self) -> None:
        valid = {"M01", "M02", "M03"}
        cases = [
            '{"status":"RANKED"',
            '{"status":"RANKED","ranked_option_ids":["M01"]}',
        ]
        for raw in cases:
            parsed = parse_response(raw, valid)
            self.assertFalse(parsed["structural_valid"])
            self.assertEqual(parsed["normalized_proposal"]["status"], "INSUFFICIENT")


if __name__ == "__main__":
    unittest.main()
