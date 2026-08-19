import json
import unittest
from copy import deepcopy
from pathlib import Path

from v154_adaptive_local_question_order import (
    evaluate_condition,
    parse_ranking,
    prepare_bounded_final_prompt_tokens,
    render_prompt,
)


class _CharTokenizer:
    eos_token_ids = [0]

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens if token != 0)


class V154AdaptiveLocalQuestionOrderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(Path("configs/v154-adaptive-local-question-order.json").read_text())
        cls.witness_config = json.loads(Path("configs/v152-fresh-question-order-population.json").read_text())
        cls.comparator_config = json.loads(Path("configs/v153-model-free-question-order-comparators.json").read_text())
        design = Path("outputs/v152-fresh-question-order-population/design")
        cls.catalog = json.loads((design / "interaction-catalog.json").read_text())
        full_public = json.loads((design / "public-fixtures.json").read_text())
        full_hidden = json.loads((design / "hidden-fixtures.json").read_text())
        request_stages = {
            "request_known_familiar", "request_known_unfamiliar", "request_right", "request_ambiguous"
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
    def oracle_completed(cls):
        query_ids = [row["query_id"] for row in cls.catalog["queries"]]
        completed = {}
        for row in cls.hidden:
            ranking = [row["oracle_query_id"]] + [query for query in query_ids if query != row["oracle_query_id"]]
            completed[row["fixture_id"]] = {
                "ranking_valid": True,
                "validation_reason": "valid_registered_query_ranking",
                "normalized_ranking": {"query_ranking": ranking},
                "query_ranking": ranking,
                "permanently_non_authoritative": True,
                "authoritative_hypothesis_universe_pruned": False,
                "capability_defined_or_registered": False,
                "executable": False,
                "actual_execution_count": 0,
                "generated_token_count": 16,
                "generation_seconds": 0.0,
            }
        return completed

    def test_projection_counts(self):
        self.assertEqual(len(self.public), 96)
        self.assertEqual(len(self.hidden), 96)
        self.assertEqual(len(self.answers), 48)
        self.assertEqual(len({row["group_id"] for row in self.hidden}), 24)

    def test_prompt_exposes_questions_without_state_or_witness_fields(self):
        payload = json.loads(render_prompt(self.catalog, self.public[0], self.config))
        self.assertEqual(payload["conversation"], self.public[0]["conversation"])
        self.assertEqual(len(payload["registered_clarification_questions"]), 6)
        self.assertEqual(
            set(payload),
            {"instruction", "registered_clarification_questions", "conversation", "response_contract"},
        )
        self.assertEqual(set(payload["response_contract"]), {"query_ranking"})
        rendered = json.dumps(payload)
        for forbidden in (
            "states", "choice_id", "state_id", "truth_state_id", "compatible_state_ids",
            "oracle_query_id", "oracle_witness", "witness", "candidate_state_ids",
        ):
            self.assertNotIn(forbidden, rendered)

    def test_parser_accepts_only_complete_query_permutation(self):
        ranking = ["Q51", "Q52", "Q53", "Q54", "Q55", "Q56"]
        parsed = parse_ranking(json.dumps({"query_ranking": ranking}), self.catalog, self.config)
        self.assertTrue(parsed["ranking_valid"])
        self.assertEqual(parsed["query_ranking"], ranking)
        self.assertTrue(parsed["permanently_non_authoritative"])
        self.assertFalse(parsed["executable"])

    def test_malformed_rankings_use_frozen_source_order_fallback(self):
        valid = {"query_ranking": ["Q51", "Q52", "Q53", "Q54", "Q55", "Q56"]}
        cases = [
            "not json",
            "```json\n" + json.dumps(valid) + "\n```",
            '{"query_ranking":["Q51","Q52","Q53","Q54","Q55","Q56"],"query_ranking":[]}',
            json.dumps({**valid, "extra": True}),
            json.dumps({"query_ranking": ["Q51"] * 6}),
            json.dumps({"query_ranking": ["Q51", "Q52", "Q53", "Q54", "Q55", "Q99"]}),
            json.dumps({"query_ranking": [{"query": "Q51"}] * 6}),
        ]
        for raw in cases:
            with self.subTest(raw=raw):
                parsed = parse_ranking(raw, self.catalog, self.config)
                self.assertFalse(parsed["ranking_valid"])
                self.assertEqual(parsed["query_ranking"], self.config["fallbackQueryRanking"])
                self.assertFalse(parsed["authoritative_hypothesis_universe_pruned"])
                self.assertEqual(parsed["actual_execution_count"], 0)

    def test_bounded_reasoning_natural_close_discards_nascent_final(self):
        tokenizer = _CharTokenizer()
        prompt = tokenizer.encode("PROMPT<think>\n")
        reasoning = tokenizer.encode("brief reasoning</think>{junk}")
        final_prompt, natural, retained = prepare_bounded_final_prompt_tokens(prompt, reasoning, tokenizer)
        decoded = tokenizer.decode(final_prompt)
        self.assertTrue(natural)
        self.assertEqual(retained, len(reasoning))
        self.assertTrue(decoded.endswith("brief reasoning</think>\n\n"))
        self.assertNotIn("junk", decoded)

    def test_bounded_reasoning_forces_close_and_strips_terminal_eos(self):
        tokenizer = _CharTokenizer()
        prompt = tokenizer.encode("PROMPT<think>\n")
        reasoning = tokenizer.encode("brief reasoning") + [0]
        final_prompt, natural, retained = prepare_bounded_final_prompt_tokens(prompt, reasoning, tokenizer)
        decoded = tokenizer.decode(final_prompt)
        self.assertFalse(natural)
        self.assertEqual(retained, len(reasoning) - 1)
        self.assertTrue(decoded.endswith("brief reasoning\n</think>\n\n"))

    def test_oracle_ranking_passes_all_question_only_gates(self):
        result = evaluate_condition(
            self.oracle_completed(), self.hidden, self.answers, self.catalog,
            self.witness_config, self.comparator_config, self.config,
        )
        self.assertTrue(result["qualified"], result)
        self.assertTrue(all(result["qualification_gates"].values()))
        self.assertEqual(result["metrics"]["sequential_episode_count"], 120)
        self.assertEqual(result["metrics"]["sequential_mean_decision_cost"], 0.3)

    def test_bad_order_fails_utility_but_not_final_safety(self):
        completed = deepcopy(self.oracle_completed())
        source = list(self.config["fallbackQueryRanking"])
        for row in completed.values():
            row["query_ranking"] = source
            row["normalized_ranking"] = {"query_ranking": source}
        result = evaluate_condition(
            completed, self.hidden, self.answers, self.catalog,
            self.witness_config, self.comparator_config, self.config,
        )
        self.assertFalse(result["qualified"])
        self.assertEqual(result["metrics"]["mean_correct_query_rank"], 3.5)
        self.assertEqual(result["metrics"]["final_exact_accuracy_after_trusted_answer"], 1.0)
        self.assertEqual(result["metrics"]["irrelevant_query_intermediate_fail_closed_rate"], 1.0)
        self.assertEqual(result["metrics"]["authoritative_hypothesis_retention"], 1.0)


if __name__ == "__main__":
    unittest.main()
