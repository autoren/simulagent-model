import json
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from audit_v22_relational import audit
from generate_v22_relational_development import generate
from run_v22_oracle_baselines import evaluate


class V22ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(Path("configs/dataset.v22.json").read_text())
        cls.records = generate(cls.config)

    def test_registered_population_and_splits(self):
        self.assertEqual(len(self.records), 24)
        self.assertEqual(Counter(value["construction_family"] for value in self.records), {
            "unary_selection": 6,
            "relation_conditioned": 6,
            "two_hop_composition": 6,
            "existential_aggregation": 6,
        })
        self.assertEqual(Counter(value["split"] for value in self.records), {
            "development_fit": 12, "development_evaluation": 12,
        })
        self.assertEqual(Counter(
            value["agent_input"]["dsl_contract"]["outcome_bits"] for value in self.records
        ), {1: 12, 2: 12})

    def test_agent_inputs_hide_oracle_state_and_program(self):
        forbidden = {
            "program", "program_key", "epistemic_state", "semantic_signatures",
            "reference_complete_world", "possible_transition_codes",
        }
        for record in self.records:
            serialized = json.dumps(record["agent_input"], sort_keys=True)
            self.assertFalse(any(f'"{value}"' in serialized for value in forbidden))

    def test_relational_entity_count_pairs_change_semantics(self):
        for record in self.records:
            groups = defaultdict(list)
            for query in record["oracle_grounding"]["queries"]:
                if query["query_axis"] == "entity_count_extrapolation":
                    groups[query["metamorphic_group"]].append(query)
            pair = next(iter(groups.values()))
            same = pair[0]["possible_transition_codes"] == pair[1]["possible_transition_codes"]
            if record["construction_family"] in {"two_hop_composition", "existential_aggregation"}:
                self.assertFalse(same)
            else:
                self.assertTrue(same)

    def test_permutation_and_distractor_controls(self):
        for record in self.records:
            groups = defaultdict(list)
            for query in record["oracle_grounding"]["queries"]:
                if query.get("metamorphic_group"):
                    groups[query["metamorphic_group"]].append(query)
            for group, pair in groups.items():
                if ":permutation" in group or ":distractor" in group:
                    self.assertEqual(len(pair), 2)
                    self.assertEqual(
                        pair[0]["possible_transition_codes"],
                        pair[1]["possible_transition_codes"],
                    )

    def test_in_memory_audit_passes(self):
        result = audit(self.records, self.config)
        self.assertTrue(result["passed"], result["errors"])

    def test_oracle_baseline_authorizes_grounding_development(self):
        structural = audit(self.records, self.config)
        result = evaluate(self.records, self.config, structural)
        self.assertTrue(result["passed"])
        self.assertEqual(result["decision"], "authorize_relational_language_grounding_development")
        self.assertEqual(result["metrics"]["schema_recovery_rate"], 1.0)
        self.assertEqual(
            result["metrics"]["exact_lifted_version_space"]["transition_set_exact_match"], 1.0
        )
        self.assertLess(
            result["metrics"]["literal_graph_lookup"]["transition_set_exact_match"], 0.95
        )


if __name__ == "__main__":
    unittest.main()
