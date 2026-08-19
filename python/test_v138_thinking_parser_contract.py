import json
import unittest
from pathlib import Path

from v135_controlled_open_world_minimal_pairs import build_catalog
from v137_direct_vs_thinking_realization import validate_final_answer as validate_v137
from v138_thinking_parser_contract import (
    inspect_template_contract,
    split_prompt_opened_thinking_suffix,
    summarize_frozen_v137_metadata,
    validate_final_answer_v138,
)


class V138ThinkingParserContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        v135 = json.loads(Path("configs/v135-controlled-open-world-minimal-pairs.json").read_text())
        cls.catalog = build_catalog(v135)

    def test_canonical_prompt_opened_suffix_exposes_v137_mismatch(self):
        suffix = 'compare all boundaries\n</think>\n{"choice_id":"N01"}'
        self.assertFalse(validate_v137(suffix, self.catalog, True)["response_valid"])
        repaired = validate_final_answer_v138(
            suffix, self.catalog, thinking_enabled=True, prompt_think_opened=True
        )
        self.assertTrue(repaired["response_valid"])
        self.assertEqual(repaired["answer_choice_id"], "N01")

    def test_repaired_parser_rejects_unclosed_and_postclosure_tags(self):
        unclosed = validate_final_answer_v138(
            'reasoning {"choice_id":"K01"}',
            self.catalog,
            thinking_enabled=True,
            prompt_think_opened=True,
        )
        postclose = validate_final_answer_v138(
            'reasoning</think><think>x</think>{"choice_id":"K01"}',
            self.catalog,
            thinking_enabled=True,
            prompt_think_opened=True,
        )
        self.assertFalse(unclosed["response_valid"])
        self.assertFalse(postclose["response_valid"])

    def test_direct_contract_remains_exact_json_without_tags(self):
        valid = validate_final_answer_v138(
            '{"choice_id":"K01"}', self.catalog, thinking_enabled=False, prompt_think_opened=False
        )
        invalid = validate_final_answer_v138(
            '<think>x</think>{"choice_id":"K01"}',
            self.catalog,
            thinking_enabled=False,
            prompt_think_opened=False,
        )
        self.assertTrue(valid["response_valid"])
        self.assertFalse(invalid["response_valid"])

    def test_nested_reasoning_before_single_final_closure_is_well_formed(self):
        parsed = split_prompt_opened_thinking_suffix(
            'outer <think>inner</think> outer</think> {"choice_id":"A00"}'
        )
        self.assertTrue(parsed["valid_trace_contract"])
        self.assertEqual(parsed["final_text"], '{"choice_id":"A00"}')

    def test_pinned_template_and_frozen_metadata_are_nonraw(self):
        manifest = json.loads(Path("outputs/v90-capacity-generation/model-acquisition/qwen38_27b_4bit.json").read_text())
        template = (Path(manifest["snapshot_path"]) / "chat_template.jinja").read_text()
        contract = inspect_template_contract(template)
        self.assertTrue(all(contract.values()), contract)
        result = json.loads(Path("outputs/v137-direct-vs-thinking-realization/model-realization/result.json").read_text())
        summary = summarize_frozen_v137_metadata(result)
        self.assertEqual(summary["unclosedThinkingTraceCount"], 93)
        self.assertFalse(summary["containsRawResponseOrTrace"])


if __name__ == "__main__":
    unittest.main()
