#!/usr/bin/env python3
"""Structural and firewall audit of the constructed V36 confirmation corpus."""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter, defaultdict

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v32_language import SURFACE_TEMPLATES as V32_TEMPLATES, compile_truth, representation_prompt_layout
from v34_operation import operation_prompt
from v35_binding import atom_prompt_layout
from v36_language import (
    COLLISION_POLICY, GENERATOR_SEED, NORMALIZATION_VERSION, SURFACE_TEMPLATES,
    normalized_template, validate_registry,
)


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def skeleton_regex(template: str, distractor: str | None = None) -> re.Pattern[str]:
    escaped = re.escape(template)
    escaped = re.sub(r"\\\{[^}]+\\\}", r".+?", escaped)
    prefix = "" if distractor is None else re.escape(distractor) + r"\s+"
    return re.compile(rf"^{prefix}{escaped}$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-lock", default="configs/v36-interface-lock.json")
    parser.add_argument("--manifest", default="data/v36-independent-confirmation/manifest.json")
    parser.add_argument("--output", default="outputs/v36-independent-confirmation/corpus-audit.json")
    args = parser.parse_args()
    interface_path, manifest_path, output_path = map(lambda value: (PROJECT_ROOT / value).resolve(), (args.interface_lock, args.manifest, args.output))
    interface, manifest = json.loads(interface_path.read_text()), json.loads(manifest_path.read_text()); errors = []
    implementation_path = PROJECT_ROOT / interface["implementation_lock"]
    implementation = json.loads(implementation_path.read_text())
    config = {**implementation["config_payload"], "v32_config": implementation["v32_config_payload"]}
    v34_config, v35_config = implementation["v34_config_payload"], {**implementation["v35_config_payload"], "v32_config": implementation["v32_config_payload"]}
    validate_registry(config)
    if manifest["interface_lock_sha256"] != file_sha256(interface_path):
        errors.append("V36 corpus manifest does not bind interface lock")
    artifact_path = PROJECT_ROOT / manifest["artifact"]
    if file_sha256(artifact_path) != manifest["artifact_sha256"]:
        errors.append("V36 corpus artifact changed")
    rows = sorted(jsonl(artifact_path), key=lambda row: row["id"])
    suite = config["confirmationSuite"]
    counts = {
        "records": len(rows), "scenes": len({row["scene_id"] for row in rows}),
        "surface_families": len({row["oracle_metadata"]["surface_family"] for row in rows}),
    }
    if counts != {"records": suite["expectedRecords"], "scenes": suite["expectedScenes"], "surface_families": suite["requiredSurfaceFamilies"]}:
        errors.append("V36 corpus population differs from design")
    expected_cells = {(operation, sign) for operation in suite["outerOperations"] for sign in suite["lexicalSignsPerOperation"]}
    cells = Counter((row["target"]["factorization"]["outer_operation"], row["target"]["factorization"]["lexical_sign"]) for row in rows)
    if set(cells) != expected_cells or set(cells.values()) != {117}:
        errors.append("V36 cell inventory or balance differs from design")
    families = Counter(row["oracle_metadata"]["surface_family"] for row in rows)
    if set(families.values()) != {78}:
        errors.append("V36 surface families are not balanced")
    if len({row["id"] for row in rows}) != len(rows) or len({(row["scene_id"], row["oracle_metadata"]["fact_index"], row["target"]["atom"]) for row in rows}) != len(rows):
        errors.append("V36 record identities are not unique")
    oracle_checks = {
        "compiler": all(compile_truth(row["target"]["factorization"]["lexical_sign"], row["target"]["factorization"]["outer_operation"], config["v32_config"]) == row["target"]["truth_status"] for row in rows),
        "target_not_in_agent_input": all(not (set(row["target"]) & set(row["agent_input"])) for row in rows),
        "schema": all(row["schema_version"] == 36 and row["split"] == suite["split"] for row in rows),
    }
    if not all(oracle_checks.values()):
        errors.append("V36 oracle or agent-input invariant failed")
    pair_members: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        for pair in row["oracle_metadata"]["pairs"]:
            pair_members[(pair["kind"], pair["id"])].append(pair)
    pair_kinds = Counter(kind for kind, _ in pair_members)
    pair_checks = {
        "all_pairs_have_two_members": all(len(members) == 2 for members in pair_members.values()),
        "required_kinds": set(pair_kinds) == {
            "argument_reversal", "distractor", "inverse", "lexical_sign_assert",
            "scope_assert_contrast", "scope_assert_deny", "scope_assert_double_deny",
            "unresolved_sign_invariance",
        },
        "nonzero_each_kind": all(value > 0 for value in pair_kinds.values()),
    }
    if not all(pair_checks.values()):
        errors.append("V36 structural pair inventory failed")
    old_normalized = {re.sub(r"\s+", " ", re.sub(r"\{[^}]+\}", "{SLOT}", template.lower())).strip() for operation in V32_TEMPLATES.values() for template, _ in operation.values()}
    new_normalized = {normalized_template(operation, surface) for operation, values in SURFACE_TEMPLATES.items() for surface in values}
    construction_overlap = sorted(old_normalized & new_normalized)
    if construction_overlap:
        errors.append("V36 normalized construction overlaps V32")
    old_patterns = [skeleton_regex(template) for values in V32_TEMPLATES.values() for template, _ in values.values()]
    from v32_language import DISTRACTOR as V32_DISTRACTOR
    old_patterns.extend(skeleton_regex(template, V32_DISTRACTOR) for values in V32_TEMPLATES.values() for template, _ in values.values())
    possible_exact_overlap = [row["id"] for row in rows if any(pattern.fullmatch(row["agent_input"]["evidence_text"]) for pattern in old_patterns)]
    if possible_exact_overlap:
        errors.append("V36 evidence matches a V32 rendering skeleton")
    prompt_dependency = []
    for row in rows:
        mutated = copy.deepcopy(row); mutated["target"] = {"sentinel": "must not affect prompts"}
        if representation_prompt_layout(row, config["v32_config"]) != representation_prompt_layout(mutated, config["v32_config"]):
            prompt_dependency.append(f"generic:{row['id']}")
        if operation_prompt(row, v34_config) != operation_prompt(mutated, v34_config):
            prompt_dependency.append(f"operation:{row['id']}")
        if atom_prompt_layout(row, v35_config) != atom_prompt_layout(mutated, v35_config):
            prompt_dependency.append(f"atom:{row['id']}")
    if prompt_dependency:
        errors.append("V36 focused prompt depends on target fields")
    if manifest["generator_seed"] != GENERATOR_SEED or manifest["normalization_version"] != NORMALIZATION_VERSION or manifest["collision_policy"] != COLLISION_POLICY:
        errors.append("V36 manifest construction policy differs from implementation")
    result = {
        "schema_version": 36, "experiment": "v36_confirmation_corpus_audit", "passed": not errors,
        "decision": "authorize_v36_confirmation_seal" if not errors else "reject_v36_confirmation_corpus",
        "errors": errors, "population": counts,
        "cell_counts": {f"{operation}|{sign}": value for (operation, sign), value in sorted(cells.items())},
        "surface_family_counts": dict(sorted(families.items())), "oracle_checks": oracle_checks,
        "pair_checks": pair_checks, "pair_counts": dict(sorted(pair_kinds.items())),
        "overlap_checks": {"normalized_construction_overlap": len(construction_overlap), "v32_rendering_skeleton_matches": len(possible_exact_overlap)},
        "prompt_target_dependencies": len(prompt_dependency),
        "source": {"interface_lock_sha256": file_sha256(interface_path), "implementation_lock_sha256": file_sha256(implementation_path), "manifest_sha256": file_sha256(manifest_path), "corpus_artifact_sha256": file_sha256(artifact_path)},
        "data_access": {"confirmation_records_structurally_audited": len(rows), "confirmation_metrics_computed": 0, "model_forward_passes": 0, "v32_evaluation_records_read": 0, "v28_integration_replays": 0},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True); output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
