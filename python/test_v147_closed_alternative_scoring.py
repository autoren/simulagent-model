import json
import unittest
from pathlib import Path

from v145_finite_certificate_codebook import oracle_code
from v147_closed_alternative_scoring import alias_mapping, evaluate, render_prompt, select_scored_code


class V147ClosedAlternativeScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(Path("configs/v147-closed-alternative-scoring.json").read_text())
        cls.catalog = json.loads(Path("outputs/v146-fresh-codebook-population/design/choice-catalog.json").read_text())
        cls.codebook = json.loads(Path("outputs/v146-fresh-codebook-population/design/certificate-codebook.json").read_text())["entries"]
        cls.public = json.loads(Path("outputs/v146-fresh-codebook-population/design/public-fixtures.json").read_text())
        cls.hidden = json.loads(Path("outputs/v146-fresh-codebook-population/design/hidden-fixtures.json").read_text())

    def test_alias_mapping_is_deterministic_complete_and_fixture_specific(self):
        first = alias_mapping(self.public[0]["fixture_id"], self.codebook, self.config["scoring"]["aliases"])
        repeat = alias_mapping(self.public[0]["fixture_id"], self.codebook, self.config["scoring"]["aliases"])
        second = alias_mapping(self.public[1]["fixture_id"], self.codebook, self.config["scoring"]["aliases"])
        self.assertEqual(first, repeat)
        self.assertEqual(set(first), set(self.config["scoring"]["aliases"]))
        self.assertEqual(set(first.values()), {row["certificate_code"] for row in self.codebook})
        self.assertNotEqual(first, second)

    def test_prompt_contains_public_inputs_and_all_alternatives_without_hidden_fields(self):
        fixture = self.public[0]
        payload = json.loads(render_prompt(self.catalog, self.codebook, fixture, self.config))
        self.assertEqual(len(payload["registered_certificate_alternatives"]), 14)
        self.assertEqual(payload["conversation"], fixture["conversation"])
        rendered = json.dumps(payload)
        for forbidden in ("truth_choice_id", "language_class", "group_id", "family_id"):
            self.assertNotIn(forbidden, rendered)

    def test_unique_maximum_selects_registered_certificate(self):
        fixture = self.hidden[0]
        mapping = alias_mapping(fixture["fixture_id"], self.codebook, self.config["scoring"]["aliases"])
        truth_code = oracle_code(fixture)
        truth_alias = next(alias for alias, code in mapping.items() if code == truth_code)
        scores = {alias: -10.0 for alias in mapping}
        scores[truth_alias] = -1.0
        selected = select_scored_code(fixture["fixture_id"], scores, self.codebook, self.config)
        self.assertTrue(selected["selection_valid"])
        self.assertEqual(selected["selected_certificate_code"], truth_code)
        self.assertEqual(selected["final_choice_id"], fixture["truth_choice_id"])

    def test_invalid_nonfinite_and_tied_scores_fail_closed(self):
        fixture_id = self.public[0]["fixture_id"]
        aliases = self.config["scoring"]["aliases"]
        cases = [{}, {alias: 0.0 for alias in aliases}, {**{alias: -1.0 for alias in aliases}, aliases[0]: float("nan")}]
        for scores in cases:
            selected = select_scored_code(fixture_id, scores, self.codebook, self.config)
            self.assertFalse(selected["selection_valid"])
            self.assertEqual(selected["final_choice_id"], "A00")

    def test_all_aliases_have_frozen_equal_character_length(self):
        aliases = self.config["scoring"]["aliases"]
        self.assertEqual({len(alias) for alias in aliases}, {3})

    def test_oracle_score_vector_passes_full_locked_evaluator(self):
        public = [row for row in self.public if row["split"] == "development"]
        hidden = [row for row in self.hidden if row["split"] == "development"]
        completed = {}
        for public_row, hidden_row in zip(
            sorted(public, key=lambda row: row["fixture_id"]),
            sorted(hidden, key=lambda row: row["fixture_id"]),
        ):
            self.assertEqual(public_row["fixture_id"], hidden_row["fixture_id"])
            mapping = alias_mapping(public_row["fixture_id"], self.codebook, self.config["scoring"]["aliases"])
            truth_code = oracle_code(hidden_row)
            truth_alias = next(alias for alias, code in mapping.items() if code == truth_code)
            scores = {alias: -10.0 for alias in mapping}
            scores[truth_alias] = -1.0
            completed[public_row["fixture_id"]] = {
                **select_scored_code(public_row["fixture_id"], scores, self.codebook, self.config),
                "scoring_seconds": 0.0,
            }
        access = {
            "V134_language_read_count": 0,
            "external_language_read_count": 0,
            "tokenizer_load_count": 1,
            "model_load_count": 1,
            "model_generation_count": 0,
            "model_scoring_fixture_count": 144,
            "candidate_sequence_score_count": 2016,
            "test_fixture_score_count": 0,
            "retry_count": 0,
            "manual_raw_response_or_trace_inspection_count": 0,
            "persisted_raw_response_or_trace_count": 0,
            "API_call_count": 0,
            "training_run_count": 0,
            "real_service_call_count": 0,
            "external_side_effect_count": 0,
            "actual_execution_count": 0,
        }
        v136 = json.loads(Path("configs/v136-controlled-clarification-value.json").read_text())
        summary = evaluate(completed, hidden, self.catalog, v136, access, self.config)
        self.assertTrue(summary["qualified"])
        self.assertTrue(all(summary["qualification_gates"].values()))
        self.assertTrue(all(summary["access_gates"].values()))


if __name__ == "__main__":
    unittest.main()
