from __future__ import annotations

import unittest
from email.message import Message

from v207r1_agentabstain_shadow_feasibility import _next_link


class V207r1TransportRulesTest(unittest.TestCase):
    def test_extracts_single_explicit_next_link(self) -> None:
        headers = Message()
        headers["Link"] = '<https://huggingface.co/next?cursor=abc>; rel="next"'
        self.assertEqual(_next_link(headers), "https://huggingface.co/next?cursor=abc")

    def test_missing_link_is_terminal(self) -> None:
        self.assertIsNone(_next_link(Message()))

    def test_multiple_next_links_are_rejected(self) -> None:
        headers = Message()
        headers["Link"] = '<https://huggingface.co/a>; rel="next", <https://huggingface.co/b>; rel="next"'
        with self.assertRaises(ValueError):
            _next_link(headers)


if __name__ == "__main__":
    unittest.main()
