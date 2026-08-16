#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v56_verification import finite_model, transition_rows_normalize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle", default="data/v56-symbolic-probabilistic-policy-verification"
    )
    parser.add_argument(
        "--output",
        default="outputs/v56-symbolic-probabilistic-policy-verification/bundle-audit.json",
    )
    args = parser.parse_args()
    bundle = (PROJECT_ROOT / args.bundle).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    implementation_path = PROJECT_ROOT / manifest["implementation_lock"]
    implementation = json.loads(implementation_path.read_text())
    errors = []

    source_ok = (
        implementation["authorization"][
            "construct_and_audit_v56_verification_bundle"
        ]
        and not implementation["authorization"][
            "run_v56_candidate_formal_verification"
        ]
        and manifest["implementation_lock_sha256"]
        == file_sha256(implementation_path)
        and file_sha256(PROJECT_ROOT / implementation["implementation"])
        == implementation["implementation_sha256"]
    )
    if not source_ok:
        errors.append("V56 implementation lock or manifest binding failed")

    census_ok = (
        manifest["policy_count"] == 48
        and manifest["cohort_counts"] == {"v55": 32, "v55r1": 16}
        and len(manifest["policies"]) == 48
        and len({(row["cohort"], row["id"]) for row in manifest["policies"]}) == 48
    )
    if not census_ok:
        errors.append("V56 verification bundle census is incomplete or duplicated")

    files_checked = hash_matches = 0
    model_checks = model_passes = 0
    action_matches = 0
    maximum_root_error = maximum_reference_error = 0.0
    truth_accesses = manifest["truth_field_access_count"]
    source_bindings = set()
    for row in manifest["policies"]:
        expected_names = {"model.tra", "model.lab", "model.rew", "model.meta.json", "policy.json"}
        actual_names = {file_row["path"].split("/")[-1] for file_row in row["files"]}
        if actual_names != expected_names:
            errors.append(f"{row['cohort']}/{row['id']} required-file set mismatch")
        for file_row in row["files"]:
            path = bundle / file_row["path"]
            files_checked += 1
            hash_matches += int(
                path.exists()
                and path.stat().st_size == file_row["bytes"]
                and file_sha256(path) == file_row["sha256"]
            )
        meta_path = bundle / row["directory"] / "model.meta.json"
        meta = json.loads(meta_path.read_text())
        model = meta["model"]
        model_checks += 1
        valid_model = (
            transition_rows_normalize(model)
            and finite_model(model)
            and len(model["states"]) == row["states"]
            and len(model["transitions"]) == row["transitions"]
            and model["horizon"] == 3
        )
        model_passes += int(valid_model)
        action_matches += int(
            meta["frozen_root_action_key"]
            == meta["reconstructed_root_action_key"]
            == row["frozen_root_action_key"]
        )
        maximum_root_error = max(
            maximum_root_error,
            meta["reconstructed_root_value_error"],
            row["reconstructed_root_value_error"],
        )
        maximum_reference_error = max(
            maximum_reference_error,
            meta["maximum_preseal_reference_error"],
            row["maximum_preseal_reference_error"],
        )
        truth_accesses += meta["truth_field_access_count"]
        source_bindings.add((
            meta["cohort"], meta["source_population_seal"],
            meta["source_population_seal_sha256"], meta["source_outcome_lock"],
            meta["source_outcome_lock_sha256"], meta["source_result"],
            meta["source_result_sha256"],
        ))
        for path_key, sha_key in (
            ("source_population_seal", "source_population_seal_sha256"),
            ("source_outcome_lock", "source_outcome_lock_sha256"),
            ("source_result", "source_result_sha256"),
        ):
            if file_sha256(PROJECT_ROOT / meta[path_key]) != meta[sha_key]:
                errors.append(f"{row['cohort']}/{row['id']} source binding changed")

    files_ok = files_checked == hash_matches == 48 * 5
    models_ok = model_checks == model_passes == 48
    binding_ok = (
        action_matches == 48
        and maximum_root_error <= 1e-10
        and maximum_reference_error <= 1e-10
        and len(source_bindings) == 2
    )
    integrity_ok = (
        truth_accesses == 0
        and manifest["candidate_formal_verification_runs"] == 0
    )
    if not files_ok:
        errors.append("V56 required file hash/size audit failed")
    if not models_ok:
        errors.append("V56 preseal DTMC structural audit failed")
    if not binding_ok:
        errors.append("V56 reconstructed policy source binding failed")
    if not integrity_ok:
        errors.append("V56 truth or candidate-verification firewall failed")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v56-verification-bundle-seal.json",
            "configs/v56-evaluation-implementation-lock.json",
            "configs/v56-outcome-lock.json",
            "outputs/v56-symbolic-probabilistic-policy-verification/evaluation-attempt.json",
            "outputs/v56-symbolic-probabilistic-policy-verification/evaluation",
        )
    )
    if not downstream_absent:
        errors.append("V56 downstream verification artifact exists before bundle seal")

    audit = {
        "schema_version": 56,
        "experiment": "v56_verification_bundle_audit",
        "passed": not errors,
        "decision": "authorize_v56_bundle_seal" if not errors else "repair_v56_bundle",
        "errors": errors,
        "bundle": str(bundle.relative_to(PROJECT_ROOT)),
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "manifest_sha256": file_sha256(manifest_path),
        "implementation_lock": str(implementation_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(implementation_path),
        "checks": {
            "implementation_authorization_and_binding": source_ok,
            "exhaustive_policy_census": census_ok,
            "required_file_hashes_and_sizes": files_ok,
            "dtmc_structure_normalization_and_finiteness": models_ok,
            "frozen_policy_action_value_and_source_binding": binding_ok,
            "truth_and_verification_firewalls": integrity_ok,
            "downstream_absent": downstream_absent,
        },
        "metrics": {
            "policy_count": manifest["policy_count"],
            "cohort_counts": manifest["cohort_counts"],
            "files_checked": files_checked,
            "file_hash_matches": hash_matches,
            "model_checks": model_checks,
            "model_passes": model_passes,
            "root_action_matches": action_matches,
            "maximum_reconstructed_root_value_error": maximum_root_error,
            "maximum_preseal_reference_error": maximum_reference_error,
            "truth_field_access_count": truth_accesses,
            "candidate_formal_verification_runs": manifest[
                "candidate_formal_verification_runs"
            ],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
