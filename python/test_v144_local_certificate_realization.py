import json
import unittest
from pathlib import Path

from v142_certificate_interface_population import deterministic_finalize
from v143_oracle_certificate_policy import oracle_certificate
from v144_local_certificate_realization import evaluate, parse_certificate_response, render_prompt


class V144LocalCertificateRealizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(Path("configs/v144-local-certificate-realization.json").read_text())
        cls.v142 = json.loads(Path("configs/v142-certificate-interface-population.json").read_text())
        cls.catalog = json.loads(Path("outputs/v142-certificate-interface-population/design/choice-catalog.json").read_text())
        cls.public = [
            row for row in json.loads(Path("outputs/v142-certificate-interface-population/design/public-fixtures.json").read_text())
            if row["split"] == "development"
        ]
        cls.hidden = [
            row for row in json.loads(Path("outputs/v142-certificate-interface-population/design/hidden-fixtures.json").read_text())
            if row["split"] == "development"
        ]
        cls.v136 = json.loads(Path("configs/v136-controlled-clarification-value.json").read_text())

    def test_prompt_contains_only_public_fixture_fields_and_full_catalog(self):
        fixture = self.public[0]
        payload = json.loads(render_prompt(self.catalog, fixture, self.config))
        self.assertEqual(len(payload["choices"]), 9)
        self.assertEqual(payload["conversation"], fixture["conversation"])
        rendered = json.dumps(payload)
        for forbidden in ("truth_choice_id", "language_class", "group_id", "family_id"):
            self.assertNotIn(forbidden, rendered)
        self.assertNotIn("compatible_choice_ids", fixture)

    def test_parser_accepts_valid_sufficient_and_insufficient_certificates(self):
        sufficient = parse_certificate_response(
            'reasoning</think>\n{"evidence_status":"SUFFICIENT","compatible_choice_ids":["K11"],"proposed_choice_id":"K11"}',
            self.catalog,
            self.v142,
        )
        self.assertTrue(sufficient["certificate_valid"])
        self.assertEqual(sufficient["final_choice_id"], "K11")
        insufficient = parse_certificate_response(
            'reasoning</think>\n{"evidence_status":"INSUFFICIENT","compatible_choice_ids":["K11","N11"],"proposed_choice_id":"A00"}',
            self.catalog,
            self.v142,
        )
        self.assertTrue(insufficient["certificate_valid"])
        self.assertEqual(insufficient["final_choice_id"], "A00")

    def test_parser_fails_closed_without_persistable_certificate(self):
        cases = [
            "unfinished reasoning",
            "reasoning</think> not json",
            'reasoning</think> {"evidence_status":"SUFFICIENT","compatible_choice_ids":["BAD"],"proposed_choice_id":"BAD"}',
        ]
        for raw in cases:
            parsed = parse_certificate_response(raw, self.catalog, self.v142)
            self.assertFalse(parsed["certificate_valid"])
            self.assertIsNone(parsed["normalized_certificate"])
            self.assertEqual(parsed["final_choice_id"], "A00")
            self.assertTrue(parsed["final_output_structurally_valid"])

    def test_oracle_outputs_pass_all_preregistered_gates(self):
        completed = {}
        for row in self.hidden:
            certificate = oracle_certificate(row)
            finalized = deterministic_finalize(certificate, self.catalog, self.v142)
            completed[row["fixture_id"]] = {
                **finalized,
                "normalized_certificate": certificate,
                "thinking_trace_present": True,
                "maximum_new_tokens_hit": False,
                "generated_token_count": 10,
                "generation_seconds": 1.0,
            }
        access = {
            "V134_language_read_count": 0,
            "external_language_read_count": 0,
            "model_load_count": 1,
            "model_generation_count": 144,
            "maximum_generation_count_per_fixture": 1,
            "test_fixture_model_generation_count": 0,
            "retry_count": 0,
            "manual_raw_response_or_trace_inspection_count": 0,
            "persisted_raw_response_or_trace_count": 0,
            "API_call_count": 0,
            "training_run_count": 0,
            "real_service_call_count": 0,
            "external_side_effect_count": 0,
            "actual_execution_count": 0,
        }
        result = evaluate(completed, self.hidden, self.catalog, self.v136, access, self.config)
        self.assertTrue(result["qualified"], result)
        self.assertTrue(all(result["qualification_gates"].values()))
        self.assertTrue(all(result["access_gates"].values()))

    def test_well_formed_wrong_singleton_is_structurally_valid_but_semantically_wrong(self):
        target = next(row for row in self.hidden if row["truth_choice_id"] != "K11" and row["stage"] != "ambiguous")
        certificate = {"evidence_status": "SUFFICIENT", "compatible_choice_ids": ["K11"], "proposed_choice_id": "K11"}
        finalized = deterministic_finalize(certificate, self.catalog, self.v142)
        self.assertTrue(finalized["certificate_valid"])
        self.assertNotEqual(finalized["final_choice_id"], target["truth_choice_id"])


if __name__ == "__main__":
    unittest.main()
