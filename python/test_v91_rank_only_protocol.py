from __future__ import annotations

import json
import unittest

from v91_rank_only_protocol import (
    aggregate_model_rows,
    canonical_complete_priority,
    identifier_exact_match_order,
    lexical_overlap_order,
    parse_and_complete,
    score_response,
)


def fixture(active: str = "BookAppointment") -> dict:
    return {
        "id": "v91-test",
        "source_record_id": "d::turn-000::Services_4",
        "service": "Services_4",
        "dialogue_history": [
            {"speaker": "USER", "utterance": "Please book an appointment."}
        ],
        "schema_context": {
            "service_name": "Services_4",
            "service_description": "Find providers and book appointments.",
            "intents": [
                {"id": "BookAppointment", "description": "Book an appointment."},
                {"id": "FindProvider", "description": "Find a provider."},
            ],
        },
        "allowed_intent_ids": ["BookAppointment", "FindProvider", "NONE"],
        "gold_intent": active,
        "authoritative_state_fingerprint": "frozen-state",
    }


class V91RankOnlyProtocolTests(unittest.TestCase):
    def test_malformed_output_falls_back_to_complete_schema_order(self):
        parsed = parse_and_complete(
            "not json", ["BookAppointment", "FindProvider", "NONE"]
        )
        self.assertFalse(parsed["exact_json"])
        self.assertEqual(
            parsed["completed_priority"],
            ["BookAppointment", "FindProvider", "NONE"],
        )
        self.assertTrue(parsed["canonical_complete_set"])
        self.assertTrue(parsed["canonical_NONE_retained"])

    def test_unknown_duplicates_and_omissions_cannot_prune(self):
        completed = canonical_complete_priority(
            ["BookAppointment", "FindProvider", "NONE"],
            ["NONE", "unknown", "NONE"],
        )
        self.assertEqual(completed, ["NONE", "BookAppointment", "FindProvider"])

    def test_valid_full_permutation_scores_rank_without_state_authority(self):
        record = fixture()
        row = score_response(
            record,
            json.dumps(
                {"intent_priority": ["BookAppointment", "NONE", "FindProvider"]}
            ),
        )
        self.assertTrue(row["raw_full_permutation"])
        self.assertTrue(row["top1"])
        self.assertTrue(row["authoritative_state_preserved"])
        self.assertFalse(row["belief_authority"])
        self.assertFalse(row["action_authority"])
        self.assertFalse(row["pruning_authority"])
        self.assertFalse(row["executable"])

    def test_controls_are_complete_deterministic_orders(self):
        record = fixture()
        for order in (
            lexical_overlap_order(record), identifier_exact_match_order(record)
        ):
            self.assertEqual(set(order), set(record["allowed_intent_ids"]))
            self.assertEqual(len(order), len(record["allowed_intent_ids"]))

    def test_aggregate_compares_against_nonoracle_controls(self):
        record = fixture()
        row = score_response(
            record,
            json.dumps(
                {"intent_priority": ["BookAppointment", "FindProvider", "NONE"]}
            ),
        )
        metrics = aggregate_model_rows([row], [record])
        self.assertEqual(metrics["record_count"], 1)
        self.assertEqual(metrics["canonical_complete_set_rate"], 1.0)
        self.assertIn(metrics["best_nonoracle_MRR_control"], metrics["controls"])


if __name__ == "__main__":
    unittest.main()
