import json
import unittest
from collections import Counter, defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np

from audit_v31_signed_fact_adaptation import audit
from generate_v31_signed_fact_adaptation import build_records, corpus_hash
from v10_protocol import file_sha256
from v31_language import construction_hash, representation_prompt_layout
from v31_evaluation import summarize_seed
from v31_structured_model import StructuredPointerHead, select_predictions


class V31SignedFactAdaptationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_path = Path("configs/v31-signed-fact-adaptation.json")
        cls.config = json.loads(cls.config_path.read_text())
        cls.rows = build_records(cls.config)

    def test_registered_population(self):
        self.assertEqual(len(self.rows), 2860)
        self.assertEqual(Counter(row["split"] for row in self.rows), {
            "adaptation_fit": 1300, "adaptation_calibration": 260,
            "adaptation_evaluation": 1300,
        })
        families = Counter(row["oracle_metadata"]["surface_family"] for row in self.rows)
        self.assertEqual(len(families), 55)
        self.assertEqual(set(families.values()), {52})
        self.assertEqual(len({row["scene_id"] for row in self.rows}), 880)

    def test_prompt_exposes_each_entity_but_no_target(self):
        for row in self.rows[:100]:
            prompt, spans = representation_prompt_layout(row, self.config)
            self.assertEqual(set(spans), {entity["id"] for entity in row["agent_input"]["entities"]})
            self.assertTrue(all(values for values in spans.values()))
            self.assertNotIn("truth_status", prompt)
            self.assertNotIn("candidate_statement", prompt)

    def test_construction_hashes_are_split_disjoint(self):
        by_split = defaultdict(set)
        for row in self.rows:
            metadata = row["oracle_metadata"]
            by_split[row["split"]].add(metadata["construction_hash"])
            self.assertEqual(
                metadata["construction_hash"],
                construction_hash(metadata["semantic_operator"], metadata["surface_name"]),
            )
        values = list(by_split.values())
        for index, left in enumerate(values):
            for right in values[index + 1:]: self.assertFalse(left & right)

    def test_all_controlled_pairs_have_two_members(self):
        groups = defaultdict(list)
        for row in self.rows:
            for pair in row["oracle_metadata"]["pairs"]:
                groups[(pair["kind"], pair["id"])].append((row, pair))
        self.assertEqual({len(values) for values in groups.values()}, {2})
        self.assertEqual({kind for kind, _ in groups}, {
            "affirmative_double_negation", "affirmative_negated", "argument_reversal",
            "distractor", "false_unknown", "inverse",
        })

    def test_in_memory_audit_passes(self):
        manifest = {
            "config_sha256": file_sha256(self.config_path),
            "corpus_sha256": corpus_hash(self.rows), "artifact_sha256": {},
        }
        result = audit(
            self.rows, self.config, manifest, self.config_path.resolve(), enforce_firewall=False
        )
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["separation"]["construction_hash_overlap_with_v30"], 0)

    def test_structured_head_shapes_and_type_masked_selection(self):
        head = StructuredPointerHead(8, 4, 5, 3)
        outputs = head(
            mx.random.normal((2, 8)), mx.random.normal((2, 5, 8)),
            mx.array([[True, True, True, False, False], [True, True, True, True, False]]),
        )
        mx.eval(*outputs)
        self.assertEqual([value.shape for value in outputs], [(2, 5), (2, 5), (2, 6), (2, 3)])
        row = next(
            value for value in self.rows
            if value["target"]["predicate"] == "feeds" and value["target"]["truth_status"] == "true"
        )
        entities = row["agent_input"]["entities"]
        hub = next(index for index, value in enumerate(entities) if value["entity_type"] == "hub")
        unit = next(index for index, value in enumerate(entities) if value["entity_type"] == "unit")
        logits = (
            np.asarray([[0, 0, 0, 0, 10]], dtype=np.float32),
            np.asarray([[100 if index == unit else 1 for index in range(5)]], dtype=np.float32),
            np.asarray([[100 if index == hub else 1 for index in range(6)]], dtype=np.float32),
            np.asarray([[10, 0, 0]], dtype=np.float32),
        )
        prediction = select_predictions([row], logits, self.config)[0]["selected_fields"]
        self.assertEqual(prediction["argument_1"], entities[hub]["id"])
        self.assertNotEqual(prediction["argument_2"], entities[hub]["id"])

    def test_oracle_predictions_pass_metric_reproduction(self):
        selected = [
            row for row in self.rows if row["split"] == "adaptation_evaluation"
        ]
        predictions = []
        for row in selected:
            target = row["target"]
            predictions.append({
                "id": row["id"], "selected_fields": {
                    "predicate": target["predicate"], "argument_1": target["arguments"][0],
                    "argument_2": target["arguments"][1] if len(target["arguments"]) == 2 else "N/A",
                    "truth_status": target["truth_status"],
                },
            })
        summary = summarize_seed(selected, predictions, self.config, apply_gates=False)
        self.assertEqual(summary["exact_signed_fact_accuracy"], 1.0)
        self.assertEqual(summary["exact_scene_accuracy"], 1.0)


if __name__ == "__main__": unittest.main()
