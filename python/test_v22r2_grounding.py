import json
import unittest
from collections import Counter
from pathlib import Path

from audit_v22_relational import read_records
from audit_v22r2_grounding import audit
from evaluate_v22r2_relational_grounding import (
    deterministic_negatives,
    integration_condition,
    pair_features,
)
from v22_relational import parse_atom
from v22r2_grounding import (
    build_corpus,
    predicted_epistemic_rows,
    scene_prompt_text,
)


class V22R2GroundingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(Path("configs/dataset.v22r2.json").read_text())
        cls.v22_config = json.loads(Path("configs/dataset.v22.json").read_text())
        source = read_records(Path(cls.config["sourceV22Dataset"]))
        cls.records, cls.scenes = build_corpus(source, cls.config, cls.v22_config)

    def test_registered_population_and_split(self):
        self.assertEqual(len(self.records), 24)
        self.assertEqual(len(self.scenes), 384)
        self.assertEqual(Counter(row["split"] for row in self.records), {
            "grounding_fit": 8, "grounding_calibration": 4,
            "grounding_evaluation": 12,
        })
        self.assertEqual(Counter(row["role"] for row in self.scenes), {
            "support": 72, "query": 312,
        })

    def test_public_interface_hides_axes_and_oracles(self):
        forbidden = {
            "query_axis", "program", "epistemic_state", "allowed_values",
            "truth_label", "possible_transition_codes", "atom_groundings",
        }
        for record in self.records:
            serialized = json.dumps(record["agent_input"], sort_keys=True)
            for key in forbidden:
                self.assertNotIn(f'"{key}"', serialized)

    def test_controlled_orientation_pairs_change_only_ordered_edge(self):
        for record in self.records:
            pair = [
                row for row in record["oracle_grounding"]["queries"]
                if row["query_axis"] == "relation_orientation"
            ]
            self.assertEqual(len(pair), 2)
            worlds = [
                {value["atom"]: value["value"] for value in row["reference_complete_world"]}
                for row in pair
            ]
            changed = {atom for atom in worlds[0] if worlds[0][atom] != worlds[1][atom]}
            binding = pair[0]["action_binding"]
            self.assertEqual(changed, {
                f"r:linked:{binding['actor']}:{binding['target']}",
                f"r:linked:{binding['target']}:{binding['actor']}",
            })

    def test_controlled_topology_pairs_change_only_links(self):
        for record in self.records:
            pair = [
                row for row in record["oracle_grounding"]["queries"]
                if row["query_axis"] == "graph_topology"
            ]
            worlds = [
                {value["atom"]: value["value"] for value in row["reference_complete_world"]}
                for row in pair
            ]
            changed = {atom for atom in worlds[0] if worlds[0][atom] != worlds[1][atom]}
            self.assertTrue(changed)
            self.assertTrue(all(parse_atom(atom)[:2] == ("r", "linked") for atom in changed))

    def test_evaluation_surfaces_are_held_out_not_semantics(self):
        fit_banks = set()
        evaluation_banks = set()
        fit_cells = set()
        evaluation_cells = set()
        for scene in self.scenes:
            for row in scene["target"]["atom_groundings"]:
                cell = (
                    row["predicate_kind"], row["truth_label"], row["semantic_operator"],
                    row["relation_orientation"],
                )
                if scene["split"] == "grounding_evaluation":
                    evaluation_banks.add(row["surface_bank"])
                    evaluation_cells.add(cell)
                elif scene["split"] == "grounding_fit":
                    fit_banks.add(row["surface_bank"])
                    fit_cells.add(cell)
        self.assertEqual(fit_banks, {"fit_a", "fit_b"})
        self.assertEqual(evaluation_banks, {"eval_c", "eval_d"})
        self.assertTrue(evaluation_cells <= fit_cells)

    def test_scene_prompt_excludes_labels_and_mappings(self):
        for scene in self.scenes:
            prompt = scene_prompt_text(scene)
            self.assertNotIn("truth_label", prompt)
            self.assertNotIn("allowed_values", prompt)
            self.assertNotIn("query_axis", prompt)
            self.assertNotIn("atom_groundings", prompt)

    def test_registered_pair_features_and_negative_sampling(self):
        import numpy as np

        evidence = np.asarray([[1.0, -2.0]], dtype=np.float32)
        candidates = np.asarray([[3.0, 4.0], [-1.0, 2.0]], dtype=np.float32)
        self.assertTrue(np.array_equal(pair_features(evidence, candidates), np.asarray([
            [2.0, 6.0, 3.0, -8.0], [2.0, 4.0, -1.0, -4.0],
        ], dtype=np.float32)))
        first = deterministic_negatives(["a", "b", "c", "d", "e"], "c", 3, "fixed")
        second = deterministic_negatives(["e", "d", "c", "b", "a"], "c", 3, "fixed")
        self.assertEqual(first, second)
        self.assertNotIn("c", first)

    def test_gold_prediction_round_trip(self):
        scene = self.scenes[0]
        predictions = [
            {
                "evidence_id": row["evidence_id"],
                "candidate_id": row["candidate_id"],
                "truth_label": row["truth_label"],
            }
            for row in scene["target"]["atom_groundings"]
        ]
        expected = [
            {"atom": row["atom"], "allowed_values": row["allowed_values"]}
            for row in scene["target"]["atom_groundings"]
        ]
        self.assertEqual(predicted_epistemic_rows(scene, predictions), expected)

    def test_gold_graphs_reproduce_all_four_way_conditions(self):
        prediction_lookup = {
            scene["id"]: {
                "epistemic_state": [
                    {"atom": row["atom"], "allowed_values": row["allowed_values"]}
                    for row in scene["target"]["atom_groundings"]
                ]
            }
            for scene in self.scenes
        }
        records = [row for row in self.records if row["split"] == "grounding_evaluation"]
        for support_mode in ("oracle", "frozen"):
            for query_mode in ("oracle", "frozen"):
                result = integration_condition(
                    records, support_mode, query_mode, prediction_lookup,
                    self.v22_config, self.config,
                )
                self.assertEqual(result["transition_set_exact_match"], 1.0)
                self.assertEqual(result["complete_episodes"], 12)
                self.assertEqual(result["target_retention_rate"], 1.0)

    def test_pre_model_audit_passes(self):
        result = audit(self.records, self.scenes, self.config, self.v22_config)
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["decision"], "authorize_v22r2_protocol_lock")
        self.assertEqual(result["surface_and_prompts"]["new_model_forward_passes"], 384)


if __name__ == "__main__":
    unittest.main()
