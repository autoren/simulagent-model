import json
import unittest
from pathlib import Path

from v154_adaptive_local_question_order import evaluate_condition
from v154r1_outcome_verifier_repair import (
    canonical_json,
    rank_count_key_type_contract,
    sole_json_key_type_mismatch,
    without_rank_counts,
)


class V154r1OutcomeVerifierRepairTests(unittest.TestCase):
    def test_canonical_json_converts_mapping_keys_only_as_json_requires(self):
        value = {"metrics": {"rank_counts": {1: 80, 2: 16}, "score": 0.5}}
        canonical = canonical_json(value)
        self.assertEqual(canonical["metrics"]["rank_counts"], {"1": 80, "2": 16})
        self.assertEqual(canonical["metrics"]["score"], 0.5)

    def test_contract_rejects_a_numeric_or_value_change(self):
        recomputed = {"metrics": {"rank_counts": {1: 80, 2: 16}, "score": 0.5}}
        persisted = {"metrics": {"rank_counts": {"1": 80, "2": 15}, "score": 0.5}}
        self.assertFalse(rank_count_key_type_contract(recomputed, persisted))
        self.assertFalse(sole_json_key_type_mismatch(recomputed, persisted))

    def test_projection_removes_only_rank_counts(self):
        summary = {"metrics": {"rank_counts": {1: 1}, "score": 0.5}, "qualified": False}
        self.assertEqual(
            without_rank_counts(summary),
            {"metrics": {"score": 0.5}, "qualified": False},
        )

    def test_actual_v154_summaries_have_only_the_preregistered_key_type_mismatch(self):
        root = Path(".")
        lock = json.loads((root / "configs/v154-adaptive-local-question-order-lock.json").read_text())
        config = lock["config_payload"]
        hidden = json.loads((root / lock["development_hidden_fixtures"]).read_text())
        answers = json.loads((root / lock["development_answer_metadata"]).read_text())
        catalog = json.loads((root / lock["interaction_catalog"]).read_text())
        witness = json.loads((root / lock["witness_config"]).read_text())
        comparator = json.loads((root / lock["comparator_config"]).read_text())
        result = json.loads((root / "outputs/v154-adaptive-local-question-order/model-realization/result.json").read_text())
        conditions = {
            "direct_summary": root / "outputs/v154-adaptive-local-question-order/model-realization/direct/result.json",
            "bounded_low_reasoning_summary": root / "outputs/v154-adaptive-local-question-order/model-realization/bounded-low-reasoning/result.json",
        }
        for key, path in conditions.items():
            with self.subTest(condition=key):
                condition = json.loads(path.read_text())
                expected = evaluate_condition(
                    condition["fixtures"], hidden, answers, catalog, witness, comparator, config
                )
                self.assertTrue(sole_json_key_type_mismatch(expected, result[key]))


if __name__ == "__main__":
    unittest.main()
