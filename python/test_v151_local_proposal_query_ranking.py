import json
import unittest
from copy import deepcopy
from pathlib import Path

from v151_local_proposal_query_ranking import evaluate, parse_proposal, render_prompt


class V151LocalProposalQueryRankingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(Path("configs/v151-local-proposal-query-ranking.json").read_text())
        cls.witness_config = json.loads(Path("configs/v149-fresh-closed-interaction-population.json").read_text())
        cls.oracle_config = json.loads(Path("configs/v150-oracle-closed-interaction-policy.json").read_text())
        cls.catalog = json.loads(
            Path("outputs/v149-fresh-closed-interaction-population/design/interaction-catalog.json").read_text()
        )
        full_public = json.loads(
            Path("outputs/v149-fresh-closed-interaction-population/design/public-fixtures.json").read_text()
        )
        full_hidden = json.loads(
            Path("outputs/v149-fresh-closed-interaction-population/design/hidden-fixtures.json").read_text()
        )
        request_stages = {
            "request_known_familiar",
            "request_known_unfamiliar",
            "request_right",
            "request_ambiguous",
        }
        answer_stages = {"closed_answer_known", "closed_answer_right"}
        cls.public = [
            row for row in full_public
            if row["split"] == "development"
            and next(hidden for hidden in full_hidden if hidden["fixture_id"] == row["fixture_id"])["stage"] in request_stages
        ]
        cls.hidden = [
            row for row in full_hidden
            if row["split"] == "development" and row["stage"] in request_stages
        ]
        cls.answers = [
            row for row in full_hidden
            if row["split"] == "development" and row["stage"] in answer_stages
        ]

    @classmethod
    def access(cls):
        return {
            "tokenizer_load_count": 1,
            "model_load_count": 1,
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

    @classmethod
    def oracle_completed(cls):
        query_ids = [row["query_id"] for row in cls.catalog["queries"]]
        completed = {}
        for row in cls.hidden:
            ranking = [row["oracle_query_id"]] + [query for query in query_ids if query != row["oracle_query_id"]]
            completed[row["fixture_id"]] = {
                "proposal_valid": True,
                "validation_reason": "valid_registered_proposal",
                "normalized_proposal": {
                    "evidence_status": "NEEDS_CLARIFICATION" if row["stage"] == "request_ambiguous" else "DECIDABLE",
                    "candidate_state_ids": list(row["compatible_state_ids"]),
                    "query_ranking": ranking,
                    "confidence": 1.0,
                },
                "evidence_status": "NEEDS_CLARIFICATION" if row["stage"] == "request_ambiguous" else "DECIDABLE",
                "candidate_state_ids": list(row["compatible_state_ids"]),
                "query_ranking": ranking,
                "confidence": 1.0,
                "generation_seconds": 0.0,
                "generated_token_count": 10,
                "permanently_non_authoritative": True,
                "authoritative_hypothesis_universe_pruned": False,
                "capability_defined_or_registered": False,
                "executable": False,
                "actual_execution_count": 0,
            }
        return completed

    def test_population_projection_counts_are_exact(self):
        self.assertEqual(len(self.public), 96)
        self.assertEqual(len(self.hidden), 96)
        self.assertEqual(len(self.answers), 48)
        self.assertEqual(len({row["group_id"] for row in self.hidden}), 24)

    def test_prompt_exposes_public_boundaries_and_query_text_but_no_typed_mapping(self):
        fixture = self.public[0]
        payload = json.loads(render_prompt(self.catalog, fixture, self.config))
        self.assertEqual(payload["conversation"], fixture["conversation"])
        self.assertEqual(len(payload["states"]), 7)
        self.assertEqual(len(payload["registered_clarification_questions"]), 6)
        self.assertTrue(all(set(option) == {"option_id", "text"} for query in payload["registered_clarification_questions"] for option in query["options"]))
        rendered = json.dumps(payload)
        for forbidden in (
            "truth_state_id",
            "compatible_state_ids",
            "oracle_query_id",
            "oracle_witness",
            "trusted_witness_available",
            "group_id",
            "family_id",
            "language_class",
            '"witness"',
        ):
            self.assertNotIn(forbidden, rendered)

    def test_parser_accepts_only_complete_registered_proposal(self):
        raw = json.dumps(
            {
                "evidence_status": "NEEDS_CLARIFICATION",
                "candidate_state_ids": ["K41", "N00"],
                "query_ranking": ["Q41", "Q42", "Q43", "Q44", "Q45", "Q46"],
                "confidence": 0.75,
            }
        )
        parsed = parse_proposal(raw, self.catalog, self.config)
        self.assertTrue(parsed["proposal_valid"])
        self.assertEqual(parsed["candidate_state_ids"], ["K41", "N00"])
        self.assertTrue(parsed["permanently_non_authoritative"])
        self.assertFalse(parsed["executable"])

    def test_every_malformed_variant_fails_closed_without_exception(self):
        valid = {
            "evidence_status": "DECIDABLE",
            "candidate_state_ids": ["K41"],
            "query_ranking": ["Q41", "Q42", "Q43", "Q44", "Q45", "Q46"],
            "confidence": 0.8,
        }
        cases = [
            "not json",
            "```json\n" + json.dumps(valid) + "\n```",
            '{"evidence_status":"DECIDABLE","evidence_status":"NEEDS_CLARIFICATION","candidate_state_ids":["K41"],"query_ranking":["Q41","Q42","Q43","Q44","Q45","Q46"],"confidence":0.8}',
            json.dumps({**valid, "extra": True}),
            json.dumps({**valid, "candidate_state_ids": []}),
            json.dumps({**valid, "candidate_state_ids": ["A00"]}),
            json.dumps({**valid, "candidate_state_ids": [["K41"]]}),
            json.dumps({**valid, "query_ranking": ["Q41"] * 6}),
            json.dumps({**valid, "query_ranking": [{"query": "Q41"}] * 6}),
            json.dumps({**valid, "confidence": True}),
            json.dumps({**valid, "confidence": float("nan")}),
        ]
        for raw in cases:
            with self.subTest(raw=raw):
                parsed = parse_proposal(raw, self.catalog, self.config)
                self.assertFalse(parsed["proposal_valid"])
                self.assertEqual(parsed["evidence_status"], "NEEDS_CLARIFICATION")
                self.assertEqual(parsed["candidate_state_ids"], [])
                self.assertEqual(parsed["query_ranking"], self.config["fallbackQueryRanking"])
                self.assertTrue(parsed["permanently_non_authoritative"])
                self.assertFalse(parsed["authoritative_hypothesis_universe_pruned"])
                self.assertEqual(parsed["actual_execution_count"], 0)

    def test_oracle_proposals_pass_every_locked_gate(self):
        result = evaluate(
            self.oracle_completed(),
            self.hidden,
            self.answers,
            self.catalog,
            self.witness_config,
            self.oracle_config,
            self.access(),
            self.config,
        )
        self.assertTrue(result["qualified"], result)
        self.assertTrue(all(result["qualification_gates"].values()))
        self.assertTrue(all(result["access_gates"].values()))
        self.assertEqual(result["metrics"]["sequential_episode_count"], 120)
        self.assertEqual(result["metrics"]["final_exact_accuracy_after_trusted_answer"], 1.0)
        self.assertEqual(result["metrics"]["sequential_mean_decision_cost"], 0.3)

    def test_misleading_model_candidates_cannot_change_trusted_final_state(self):
        completed = deepcopy(self.oracle_completed())
        for row in completed.values():
            row["candidate_state_ids"] = ["K41"]
            row["normalized_proposal"]["candidate_state_ids"] = ["K41"]
        result = evaluate(
            completed,
            self.hidden,
            self.answers,
            self.catalog,
            self.witness_config,
            self.oracle_config,
            self.access(),
            self.config,
        )
        self.assertFalse(result["qualified"])
        self.assertEqual(result["metrics"]["final_exact_accuracy_after_trusted_answer"], 1.0)
        self.assertEqual(result["metrics"]["false_known_after_trusted_answer"], 0.0)
        self.assertEqual(result["metrics"]["authoritative_true_hypothesis_retention"], 1.0)

    def test_irrelevant_ranked_questions_explicitly_fail_closed_before_resolution(self):
        completed = self.oracle_completed()
        for row in self.hidden:
            ranking = completed[row["fixture_id"]]["query_ranking"]
            completed[row["fixture_id"]]["query_ranking"] = ranking[1:] + ranking[:1]
            completed[row["fixture_id"]]["normalized_proposal"]["query_ranking"] = ranking[1:] + ranking[:1]
        result = evaluate(
            completed,
            self.hidden,
            self.answers,
            self.catalog,
            self.witness_config,
            self.oracle_config,
            self.access(),
            self.config,
        )
        self.assertGreater(result["metrics"]["irrelevant_query_intermediate_count"], 0)
        self.assertEqual(result["metrics"]["irrelevant_query_intermediate_fail_closed_rate"], 1.0)
        self.assertEqual(result["metrics"]["final_exact_accuracy_after_trusted_answer"], 1.0)
        self.assertGreater(result["metrics"]["sequential_mean_decision_cost"], 0.3)


if __name__ == "__main__":
    unittest.main()
