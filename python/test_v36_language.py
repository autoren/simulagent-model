"""Tests for the immutable V36 registry and deterministic construction."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from generate_v36_confirmation import build_records, corpus_hash
from v36_language import SURFACE_TEMPLATES, construction_hash, normalized_template, validate_registry


class V36LanguageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        design = json.loads(Path("configs/v36-independent-confirmation-design.json").read_text())
        v32_lock = json.loads(Path("configs/v32-factorized-semantics-lock.json").read_text())
        cls.config, cls.v32 = design, v32_lock["config_payload"]

    def test_registry_is_exact_and_unique(self):
        validate_registry(self.config)
        self.assertEqual(sum(len(rows) for rows in SURFACE_TEMPLATES.values()), 15)
        hashes = {construction_hash(operation, surface) for operation, rows in SURFACE_TEMPLATES.items() for surface in rows}
        self.assertEqual(len(hashes), 15)
        self.assertTrue(all("{SLOT}" in normalized_template(operation, surface) for operation, rows in SURFACE_TEMPLATES.items() for surface in rows))

    def test_population_is_exact_and_deterministic(self):
        first = build_records(self.config, self.v32)
        second = build_records(self.config, self.v32)
        self.assertEqual(len(first), 1170)
        self.assertEqual(len({row["scene_id"] for row in first}), 360)
        self.assertEqual(len({row["oracle_metadata"]["surface_family"] for row in first}), 15)
        self.assertEqual(corpus_hash(first), corpus_hash(second))

    def test_all_cells_and_pair_kinds_exist(self):
        rows = build_records(self.config, self.v32)
        cells = {(row["target"]["factorization"]["outer_operation"], row["target"]["factorization"]["lexical_sign"]) for row in rows}
        self.assertEqual(len(cells), 10)
        pair_kinds = {pair["kind"] for row in rows for pair in row["oracle_metadata"]["pairs"]}
        self.assertEqual(pair_kinds, {
            "argument_reversal", "distractor", "inverse", "lexical_sign_assert",
            "scope_assert_contrast", "scope_assert_deny", "scope_assert_double_deny",
            "unresolved_sign_invariance",
        })


if __name__ == "__main__":
    unittest.main()
