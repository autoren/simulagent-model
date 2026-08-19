from __future__ import annotations

import unittest

from v211r1_compositional_baseline_name_repair import predict_evaluation, repair_diagnostics


class V211r1RepairTest(unittest.TestCase):
    def test_only_compositional_key_changes(self) -> None:
        learned = {name: {"token_to_label": {"dax": "UTTERANCE_ALPHA"}, "blocked_tokens_by_context": {}} for name in ("RAW_LEXICAL", "COMPOSITIONAL_RESPONSE_SPAN")}
        surface = [{"record_id": "R", "context_id": "C", "utterance": "dax"}]
        repaired = predict_evaluation(surface, learned)
        parent = [{**row, "baseline": "CONTEXT_CONTRAST" if row["baseline"] == "COMPOSITIONAL_RESPONSE_SPAN" else row["baseline"]} for row in repaired]
        diagnostics = repair_diagnostics(parent, repaired)
        self.assertTrue(diagnostics["normalized_parent_matches_repaired_exactly"])
        self.assertTrue(diagnostics["prediction_values_match_as_multiset"])
        self.assertEqual(diagnostics["changed_prediction_value_count"], 0)


if __name__ == "__main__": unittest.main()
