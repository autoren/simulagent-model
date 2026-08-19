import json
import unittest
from pathlib import Path

from v151_local_proposal_query_ranking import evaluate
from v151r1_local_proposal_query_ranking_recovery import (
    derive_partition,
    interrupted_fail_closed,
    recovery_evaluation_config,
)


class V151r1RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = json.loads(Path("configs/v151-local-proposal-query-ranking.json").read_text())
        cls.recovery = json.loads(Path("configs/v151r1-local-proposal-query-ranking-recovery.json").read_text())
        lock = json.loads(Path("configs/v151-local-proposal-query-ranking-lock.json").read_text())
        cls.public = json.loads(Path(lock["development_public_fixtures"]).read_text())
        cls.hidden = json.loads(Path(lock["development_hidden_fixtures"]).read_text())
        cls.answers = json.loads(Path(lock["development_answer_metadata"]).read_text())
        cls.catalog = json.loads(Path(lock["interaction_catalog"]).read_text())
        cls.witness = json.loads(Path(lock["witness_config"]).read_text())
        cls.oracle = json.loads(Path(lock["oracle_config"]).read_text())

    def test_exact_interruption_partition(self):
        paths = sorted(Path("outputs/v151-local-proposal-query-ranking/model-realization/raw-fixtures").glob("*.json"))
        access = json.loads(Path("outputs/v151-local-proposal-query-ranking/model-realization/access-progress.json").read_text())
        partition = derive_partition(self.public, paths, access, self.recovery)
        self.assertEqual(len(partition["persisted_fixture_ids"]), 58)
        self.assertEqual(len(partition["never_started_fixture_ids"]), 37)
        self.assertNotIn(partition["interrupted_fixture_id"], partition["persisted_fixture_ids"])
        self.assertNotIn(partition["interrupted_fixture_id"], partition["never_started_fixture_ids"])
        self.assertEqual(len(set(partition["all_fixture_ids"])), 96)

    def test_recovery_changes_only_access_and_decision_strings(self):
        value = recovery_evaluation_config(self.base, self.recovery)
        self.assertEqual(value["model"], self.base["model"])
        self.assertEqual(value["prompt"], self.base["prompt"])
        self.assertEqual(value["qualificationGates"], self.base["qualificationGates"])
        self.assertEqual(value["population"], self.base["population"])
        self.assertEqual(value["accessGates"]["maximumModelLoadCount"], 2)
        self.assertEqual(value["accessGates"]["maximumModelGenerationCount"], 96)

    def test_interrupted_fixture_is_invalid_non_authoritative_and_fail_closed(self):
        fixture_id = self.public[58]["fixture_id"]
        row = interrupted_fail_closed(fixture_id, self.catalog, self.base)
        self.assertFalse(row["proposal_valid"])
        self.assertEqual(row["evidence_status"], "NEEDS_CLARIFICATION")
        self.assertEqual(row["candidate_state_ids"], [])
        self.assertFalse(row["authoritative_hypothesis_universe_pruned"])
        self.assertFalse(row["executable"])
        self.assertEqual(row["actual_execution_count"], 0)

    def test_oracle_plus_interrupted_fallback_remains_evaluable(self):
        query_ids = [row["query_id"] for row in self.catalog["queries"]]
        completed = {}
        for row in self.hidden:
            if row["fixture_id"] == self.public[58]["fixture_id"]:
                completed[row["fixture_id"]] = interrupted_fail_closed(row["fixture_id"], self.catalog, self.base)
                continue
            status = "NEEDS_CLARIFICATION" if row["stage"] == "request_ambiguous" else "DECIDABLE"
            ranking = [row["oracle_query_id"]] + [query for query in query_ids if query != row["oracle_query_id"]]
            normalized = {
                "evidence_status": status,
                "candidate_state_ids": list(row["compatible_state_ids"]),
                "query_ranking": ranking,
                "confidence": 1.0,
            }
            completed[row["fixture_id"]] = {
                "proposal_valid": True,
                "validation_reason": "valid_registered_proposal",
                "normalized_proposal": normalized,
                **normalized,
                "generation_seconds": 0.0,
                "generated_token_count": 1,
            }
        access = {
            "tokenizer_load_count": 2,
            "model_load_count": 2,
            "model_generation_count": 96,
            "maximum_generation_count_per_fixture": 1,
            "closed_answer_model_generation_count": 0,
            "evaluation_fixture_model_generation_count": 0,
            "retry_count": 0,
            "manual_raw_response_inspection_count": 0,
            "persisted_raw_response_count": 0,
            "API_call_count": 0,
            "training_run_count": 0,
            "real_service_call_count": 0,
            "external_side_effect_count": 0,
            "actual_execution_count": 0,
        }
        result = evaluate(
            completed,
            self.hidden,
            self.answers,
            self.catalog,
            self.witness,
            self.oracle,
            access,
            recovery_evaluation_config(self.base, self.recovery),
        )
        self.assertTrue(result["qualified"], result)
        self.assertEqual(result["metrics"]["final_exact_accuracy_after_trusted_answer"], 1.0)


if __name__ == "__main__":
    unittest.main()
