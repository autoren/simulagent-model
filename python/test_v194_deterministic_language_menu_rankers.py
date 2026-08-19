from __future__ import annotations

import unittest

from v194_deterministic_language_menu_rankers import rank_options


class V194RankerTest(unittest.TestCase):
    def test_character_ranker_matches_compact_intent_label(self) -> None:
        record = {"conversation": [{"speaker": "USER", "utterance": "Please find events near me"}]}
        menu = {
            "options": [
                {"option_id": "M01", "domain": "weather", "intent_concept": "getweather"},
                {"option_id": "M02", "domain": "events", "intent_concept": "findevents"},
            ]
        }
        ranking, count = rank_options(record, menu, {"query": "lastUser", "view": "character"})
        self.assertEqual(ranking[0], "M02")
        self.assertEqual(count, 2)


if __name__ == "__main__":
    unittest.main()
