import unittest

from audit_v13_operator_support import mention_signature


def record(evidence):
    return {
        "agent_input": {
            "state_hypotheses": [{
                "determinant_id": "relay",
                "statements": ["relay is active", "relay is inactive"],
            }]
        },
        "evidence_units": [{"start": 0, "end": 1, "text": evidence}],
    }


def target(value):
    return {
        "determinant_id": "relay",
        "current_value": value,
        "evidence_span": {"start": 0, "end": 1},
    }


class V13OperatorSupportTests(unittest.TestCase):
    def test_signatures_track_literal_gold_orientation(self):
        self.assertEqual(mention_signature(record("relay is active"), target("active")), "gold_only")
        self.assertEqual(mention_signature(record("not relay is inactive"), target("active")), "opposite_only")
        self.assertEqual(
            mention_signature(record("not relay is inactive; relay is active"), target("active")), "both"
        )


if __name__ == "__main__":
    unittest.main()
