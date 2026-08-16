#!/usr/bin/env python3
"""Pre-human audit of the offline V58 pilot collection toolkit."""
from __future__ import annotations

import argparse
import copy
import inspect
import json

from render_v58_pilot_form import render_offline_form, validate_release_lock
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v58_pilot_intake import audit_pilot_reviews, audit_pilot_submissions


TOOLING_FILES = (
    "python/render_v58_pilot_form.py",
    "python/v58_pilot_intake.py",
    "python/audit_v58_pilot_tooling.py",
    "python/freeze_v58_pilot_tooling.py",
    "docs/v58-offline-collection-toolkit.md",
    "configs/v58-human-submission.schema.json",
    "configs/v58-human-validation.schema.json",
    "configs/v58-human-adjudication.schema.json",
)


def _fixtures(packets: list[dict], protocol: dict) -> tuple[list, list]:
    submissions = []
    validations = []
    index = 0
    for packet in packets:
        for prompt in packet["prompts"]:
            submission_id = f"synthetic_tooling_submission_{index:04d}"
            submissions.append({
                "submission_id": submission_id,
                "packet_id": prompt["packet_id"],
                "prompt_id": prompt["prompt_id"],
                "anonymous_writer_id": prompt["anonymous_writer_id"],
                "collection_round": prompt["collection_round"],
                "submitted_text": "Synthetic offline-tooling fixture only.",
                "timestamp": "2026-08-15T14:00:00Z",
                "consent_and_license_attestation": protocol[
                    "submissionSchema"
                ]["attestation"],
            })
            for validator in ("synthetic_validator_a", "synthetic_validator_b"):
                validations.append({
                    "validation_id": f"{validator}_{index:04d}",
                    "submission_id": submission_id,
                    "anonymous_validator_id": validator,
                    "verdict": "equivalent_unique",
                    "inferred_canonical_ast_or_null": prompt[
                        "intended_semantics"
                    ],
                    "construction_family_realized": True,
                    "source_target_order_preserved_or_null": True,
                    "notes_without_writer_identity": "synthetic fixture",
                    "timestamp": "2026-08-15T15:00:00Z",
                })
            index += 1
    return submissions, validations


def _names(audit: dict) -> set[str]:
    return {name for name, _count in audit["errors"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--handoff-lock",
        default="configs/v58-pilot-collection-handoff-lock.json",
    )
    parser.add_argument(
        "--output",
        default=(
            "outputs/v58-human-authored-known-ontology-language/"
            "pilot-tooling-audit.json"
        ),
    )
    args = parser.parse_args()
    handoff_path = (PROJECT_ROOT / args.handoff_lock).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    handoff = json.loads(handoff_path.read_text())
    seal_path = PROJECT_ROOT / handoff["packet_seal"]
    seal = json.loads(seal_path.read_text())
    protocol = json.loads((PROJECT_ROOT / seal["protocol"]).read_text())
    errors: list[str] = []

    boundary_ok = (
        handoff["decision"]
        == "await_external_human_pilot_coordinator_and_release_authorization"
        and handoff["authorization"]["show_readiness_to_user"]
        and not handoff["authorization"]["release_pilot_packets"]
        and not handoff["authorization"]["collect_pilot_language"]
        and not handoff["authorization"]["write_candidate_parser"]
        and file_sha256(seal_path) == handoff["packet_seal_sha256"]
        and all(
            file_sha256(PROJECT_ROOT / row["path"]) == row["sha256"]
            for row in handoff["pilot_packet_artifacts"]
        )
    )
    if not boundary_ok:
        errors.append("V58 pilot handoff boundary is not intact")

    packets = [
        json.loads((PROJECT_ROOT / row["path"]).read_text())
        for row in handoff["pilot_packet_artifacts"]
    ]
    html_rows = [
        render_offline_form(packet, artifact["sha256"], protocol)
        for packet, artifact in zip(
            packets, handoff["pilot_packet_artifacts"], strict=True
        )
    ]
    html_ok = all(
        html_row.count('class="prompt"') == 0
        and "packet.prompts.forEach" in html_row
        and "application/x-ndjson" in html_row
        and "consent_and_license_attestation" in html_row
        and "submitted_text: text" in html_row
        and not any(token in html_row.casefold() for token in (
            "fetch(", "xmlhttprequest", "websocket", "http://", "https://"
        ))
        for html_row in html_rows
    )
    if not html_ok:
        errors.append("V58 rendered pilot form is not offline or schema-bound")

    source = inspect.getsource(render_offline_form) + inspect.getsource(
        audit_pilot_reviews
    )
    no_model_or_network = not any(token in source.casefold() for token in (
        "requests.", "urllib.", "socket.", "openai", "anthropic", "model.generate"
    ))
    if not no_model_or_network:
        errors.append("V58 collection tooling contains a network or model dependency")

    release_refusal_ok = False
    try:
        validate_release_lock(
            {
                "experiment": "v58_pilot_release_lock",
                "authorization": {
                    "release_pilot_packets": False,
                    "collect_pilot_language": False,
                },
            },
            PROJECT_ROOT / "synthetic-invalid-release-lock.json",
            seal,
        )
    except (KeyError, RuntimeError):
        release_refusal_ok = True
    if not release_refusal_ok:
        errors.append("V58 tooling did not refuse an unauthorized release lock")

    submissions, validations = _fixtures(packets, protocol)
    submission_audit = audit_pilot_submissions(packets, submissions, protocol)
    review_audit = audit_pilot_reviews(
        packets, submissions, validations, [], protocol
    )
    full_fixture_ok = (
        submission_audit["passed"]
        and review_audit["passed"]
        and submission_audit["metrics"]["submissions"] == 120
        and review_audit["metrics"]["validations"] == 240
        and review_audit["metrics"]["accepted_prompts"] == 120
        and review_audit["metrics"]["raw_validator_verdict_agreement"] == 1.0
    )
    if not full_fixture_ok:
        errors.append("V58 synthetic pilot submission/review fixture failed")

    adjudicated_validations = copy.deepcopy(validations)
    adjudicated_validations[1]["verdict"] = "not_equivalent"
    adjudicated_validations[1]["inferred_canonical_ast_or_null"] = None
    first_submission = submissions[0]
    first_prompt = packets[0]["prompts"][0]
    good_adjudication = [{
        "adjudication_id": "synthetic_adjudication_0000",
        "submission_id": first_submission["submission_id"],
        "anonymous_adjudicator_id": "synthetic_adjudicator_c",
        "final_verdict": "equivalent_unique",
        "final_canonical_ast_or_null": first_prompt["intended_semantics"],
        "construction_family_realized": True,
        "source_target_order_preserved_or_null": True,
        "reason": "synthetic disagreement resolution fixture",
        "timestamp": "2026-08-15T16:00:00Z",
    }]
    adjudication_path_ok = audit_pilot_reviews(
        packets,
        submissions,
        adjudicated_validations,
        good_adjudication,
        protocol,
    )["passed"]
    if not adjudication_path_ok:
        errors.append("V58 valid third-person adjudication path failed")

    missing_attestation = copy.deepcopy(submissions)
    missing_attestation[0]["consent_and_license_attestation"][
        "humanAuthoredWithoutGenerativeAssistance"
    ] = False
    target_leak = copy.deepcopy(submissions)
    target_leak[0]["target_ast"] = first_prompt["intended_semantics"]
    wrong_writer = copy.deepcopy(submissions)
    wrong_writer[0]["anonymous_writer_id"] = "wrong_writer"
    duplicate_submission = copy.deepcopy(submissions)
    duplicate_submission[1]["submission_id"] = duplicate_submission[0][
        "submission_id"
    ]
    under_quota = audit_pilot_submissions(
        packets, submissions[:-1], protocol
    )
    submission_attacks = [
        audit_pilot_submissions(packets, rows, protocol)
        for rows in (
            missing_attestation,
            target_leak,
            wrong_writer,
            duplicate_submission,
        )
    ]

    duplicate_validator = copy.deepcopy(validations)
    duplicate_validator[1]["anonymous_validator_id"] = duplicate_validator[0][
        "anonymous_validator_id"
    ]
    writer_validator = copy.deepcopy(validations)
    writer_validator[0]["anonymous_validator_id"] = submissions[0][
        "anonymous_writer_id"
    ]
    low_agreement = copy.deepcopy(validations)
    for index in list(range(1, len(low_agreement), 2))[:13]:
        low_agreement[index]["verdict"] = "not_equivalent"
        low_agreement[index]["inferred_canonical_ast_or_null"] = None
    missing_adjudication = audit_pilot_reviews(
        packets, submissions, adjudicated_validations, [], protocol
    )
    review_attacks = [
        audit_pilot_reviews(packets, submissions, rows, [], protocol)
        for rows in (duplicate_validator, writer_validator, low_agreement)
    ]
    adversarial_ok = (
        all(not row["passed"] for row in submission_attacks)
        and not under_quota["passed"]
        and all(not row["passed"] for row in review_attacks)
        and not missing_adjudication["passed"]
        and "invalid_submission" in _names(submission_attacks[0])
        and "invalid_submission" in _names(submission_attacks[1])
        and "invalid_submission" in _names(submission_attacks[2])
        and "duplicate_submission_id" in _names(submission_attacks[3])
        and "prompt_submission_census" in _names(under_quota)
        and "duplicate_validator" in _names(review_attacks[0])
        and "validator_writer_overlap" in _names(review_attacks[1])
        and "validator_agreement" in _names(review_attacks[2])
        and "missing_or_duplicate_adjudication"
        in _names(missing_adjudication)
    )
    if not adversarial_ok:
        errors.append("V58 malformed/leak/role/quota/agreement attacks survived")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v58-pilot-tooling-lock.json",
            "configs/v58-pilot-release-lock.json",
            "data/v58-human-authored-known-ontology-language/pilot-submissions",
            "data/v58-human-authored-known-ontology-language/pilot-validations",
            "configs/v58-pilot-population-seal.json",
            "configs/v58-candidate-lock.json",
        )
    )
    if not downstream_absent:
        errors.append("V58 release, human text, or candidate artifact already exists")

    audit = {
        "schema_version": 58,
        "experiment": "v58_pilot_offline_tooling_audit",
        "passed": not errors,
        "decision": (
            "authorize_v58_pilot_tooling_lock"
            if not errors else "repair_v58_pilot_tooling"
        ),
        "errors": errors,
        "handoff_lock": str(handoff_path.relative_to(PROJECT_ROOT)),
        "handoff_lock_sha256": file_sha256(handoff_path),
        "packet_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "packet_seal_sha256": file_sha256(seal_path),
        "tooling_files_sha256": {
            path: file_sha256(PROJECT_ROOT / path) for path in TOOLING_FILES
        },
        "checks": {
            "unreleased_handoff_and_packet_boundary": boundary_ok,
            "offline_schema_bound_form_rendering": html_ok,
            "no_network_or_model_dependency": no_model_or_network,
            "unauthorized_release_lock_refused": release_refusal_ok,
            "complete_synthetic_submission_and_review_fixture": full_fixture_ok,
            "third_person_adjudication_path": adjudication_path_ok,
            "malformed_leak_role_quota_and_agreement_attacks_rejected": adversarial_ok,
            "release_human_text_and_candidate_downstream_absent": downstream_absent,
        },
        "fixture_metrics": review_audit["metrics"],
        "data_access": {
            "sealed_text_free_pilot_prompts_accessed": 120,
            "synthetic_submission_fixtures": 120,
            "synthetic_validation_fixtures": 240,
            "human_authored_records_collected": 0,
            "human_authored_text_accessed": 0,
            "pilot_forms_written": 0,
            "pilot_packets_released": 0,
            "evaluation_packets_released": 0,
            "candidate_evaluation_runs": 0,
            "model_forward_passes": 0,
        },
    }
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
