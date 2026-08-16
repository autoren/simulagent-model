"""Structural validators for the text-free V58 collection workflow."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from typing import Any


SUBMISSION_FIELDS = {
    "submission_id",
    "packet_id",
    "prompt_id",
    "anonymous_writer_id",
    "collection_round",
    "submitted_text",
    "timestamp",
    "consent_and_license_attestation",
}
VALIDATION_FIELDS = {
    "validation_id",
    "submission_id",
    "anonymous_validator_id",
    "verdict",
    "inferred_canonical_ast_or_null",
    "construction_family_realized",
    "source_target_order_preserved_or_null",
    "notes_without_writer_identity",
    "timestamp",
}
ADJUDICATION_FIELDS = {
    "adjudication_id",
    "submission_id",
    "anonymous_adjudicator_id",
    "final_verdict",
    "final_canonical_ast_or_null",
    "construction_family_realized",
    "source_target_order_preserved_or_null",
    "reason",
    "timestamp",
}
PROMPT_FIELDS = {
    "packet_id",
    "prompt_id",
    "anonymous_writer_id",
    "writer_cohort",
    "collection_round",
    "construction_family",
    "stratum",
    "abstention_condition",
    "source_record_id",
    "entity_legend",
    "known_ontology_glossary",
    "intended_semantics",
    "writing_instructions",
}


def _timestamp_ok(value: Any) -> bool:
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return False
    return True


def submission_errors(
    submission: dict[str, Any], prompt: dict[str, Any], protocol: dict[str, Any]
) -> list[str]:
    errors = []
    if set(submission) != SUBMISSION_FIELDS:
        errors.append("submission_fields")
    if set(prompt) != PROMPT_FIELDS:
        errors.append("prompt_fields")
        return errors
    expected = {
        "packet_id": prompt["packet_id"],
        "prompt_id": prompt["prompt_id"],
        "anonymous_writer_id": prompt["anonymous_writer_id"],
        "collection_round": prompt["collection_round"],
    }
    if any(submission.get(key) != value for key, value in expected.items()):
        errors.append("prompt_writer_or_round_binding")
    text = submission.get("submitted_text")
    requirements = protocol["submissionSchema"]["textRequirements"]
    if (
        not isinstance(text, str)
        or len(text.strip()) < requirements["minimumUnicodeCharacters"]
        or len(text) > requirements["maximumUnicodeCharacters"]
        or "\n" in text
    ):
        errors.append("text_shape")
    if not _timestamp_ok(submission.get("timestamp")):
        errors.append("timestamp")
    if submission.get("consent_and_license_attestation") != protocol[
        "submissionSchema"
    ]["attestation"]:
        errors.append("attestation")
    return errors


def validation_errors(
    record: dict[str, Any], submission_id: str, protocol: dict[str, Any]
) -> list[str]:
    errors = []
    if set(record) != VALIDATION_FIELDS:
        errors.append("validation_fields")
    if record.get("submission_id") != submission_id:
        errors.append("submission_binding")
    verdict = record.get("verdict")
    if verdict not in protocol["validation"]["verdicts"]:
        errors.append("verdict")
    inferred = record.get("inferred_canonical_ast_or_null")
    if verdict == "equivalent_unique" and not isinstance(inferred, dict):
        errors.append("equivalent_requires_ast")
    if verdict in {"ambiguous", "unsupported"} and inferred is not None:
        errors.append("nonunique_requires_null_ast")
    if not isinstance(record.get("construction_family_realized"), bool):
        errors.append("construction_annotation")
    if record.get("source_target_order_preserved_or_null") not in {
        True,
        False,
        None,
    }:
        errors.append("relation_order_annotation")
    if not isinstance(record.get("notes_without_writer_identity"), str):
        errors.append("notes")
    if not _timestamp_ok(record.get("timestamp")):
        errors.append("timestamp")
    return errors


def adjudication_errors(
    record: dict[str, Any],
    submission_id: str,
    validator_ids: set[str],
    writer_id: str,
    protocol: dict[str, Any],
) -> list[str]:
    errors = []
    if set(record) != ADJUDICATION_FIELDS:
        errors.append("adjudication_fields")
    if record.get("submission_id") != submission_id:
        errors.append("submission_binding")
    if record.get("anonymous_adjudicator_id") in validator_ids:
        errors.append("adjudicator_validator_overlap")
    if record.get("anonymous_adjudicator_id") == writer_id:
        errors.append("adjudicator_writer_overlap")
    verdict = record.get("final_verdict")
    if verdict not in protocol["validation"]["verdicts"]:
        errors.append("verdict")
    inferred = record.get("final_canonical_ast_or_null")
    if verdict == "equivalent_unique" and not isinstance(inferred, dict):
        errors.append("equivalent_requires_ast")
    if verdict in {"ambiguous", "unsupported"} and inferred is not None:
        errors.append("nonunique_requires_null_ast")
    if not isinstance(record.get("construction_family_realized"), bool):
        errors.append("construction_annotation")
    if record.get("source_target_order_preserved_or_null") not in {
        True,
        False,
        None,
    }:
        errors.append("relation_order_annotation")
    if not isinstance(record.get("reason"), str) or not record["reason"].strip():
        errors.append("reason")
    if not _timestamp_ok(record.get("timestamp")):
        errors.append("timestamp")
    return errors


def _validation_disagrees(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = (
        "verdict",
        "inferred_canonical_ast_or_null",
        "construction_family_realized",
        "source_target_order_preserved_or_null",
    )
    return any(left[key] != right[key] for key in keys)


def audit_collection_metadata(
    prompts: list[dict[str, Any]],
    submissions: list[dict[str, Any]],
    validations: list[dict[str, Any]],
    adjudications: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Audit a proposed accepted collection without invoking a candidate parser."""
    errors: list[str] = []
    prompt_by_id = {row.get("prompt_id"): row for row in prompts}
    submission_by_id = {row.get("submission_id"): row for row in submissions}
    if len(prompt_by_id) != len(prompts):
        errors.append("duplicate_prompt_id")
    if len(submission_by_id) != len(submissions):
        errors.append("duplicate_submission_id")
    if any(set(row) != PROMPT_FIELDS for row in prompts):
        errors.append("invalid_prompt_fields")

    submission_by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in submissions:
        prompt = prompt_by_id.get(row.get("prompt_id"))
        if prompt is None:
            errors.append("unknown_prompt")
            continue
        if submission_errors(row, prompt, protocol):
            errors.append("invalid_submission")
        submission_by_prompt[row["prompt_id"]].append(row)
    if any(len(rows) != 1 for rows in submission_by_prompt.values()):
        errors.append("multiple_submissions_per_prompt")
    if set(submission_by_prompt) != set(prompt_by_id):
        errors.append("prompt_submission_census")

    validation_by_submission: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if len({row.get("validation_id") for row in validations}) != len(validations):
        errors.append("duplicate_validation_id")
    for row in validations:
        submission_id = row.get("submission_id")
        if submission_id not in submission_by_id:
            errors.append("validation_unknown_submission")
            continue
        if validation_errors(row, submission_id, protocol):
            errors.append("invalid_validation")
        validation_by_submission[submission_id].append(row)

    adjudication_by_submission: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if len({row.get("adjudication_id") for row in adjudications}) != len(
        adjudications
    ):
        errors.append("duplicate_adjudication_id")
    for row in adjudications:
        adjudication_by_submission[row.get("submission_id")].append(row)

    exact_verdict_agreements = 0
    finalized: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for submission_id, submission in submission_by_id.items():
        records = validation_by_submission.get(submission_id, [])
        if len(records) != protocol["validation"]["validatorsPerSubmission"]:
            errors.append("validator_census")
            continue
        validator_ids = {row["anonymous_validator_id"] for row in records}
        if len(validator_ids) != len(records):
            errors.append("duplicate_validator")
        if submission["anonymous_writer_id"] in validator_ids:
            errors.append("validator_writer_overlap")
        exact_verdict_agreements += int(
            records[0]["verdict"] == records[1]["verdict"]
        )
        disagreement = _validation_disagrees(records[0], records[1])
        adjudication_rows = adjudication_by_submission.get(submission_id, [])
        if disagreement:
            if len(adjudication_rows) != 1:
                errors.append("missing_or_duplicate_adjudication")
                continue
            final = adjudication_rows[0]
            if adjudication_errors(
                final,
                submission_id,
                validator_ids,
                submission["anonymous_writer_id"],
                protocol,
            ):
                errors.append("invalid_adjudication")
                continue
            normalized = {
                "verdict": final["final_verdict"],
                "ast": final["final_canonical_ast_or_null"],
                "construction": final["construction_family_realized"],
                "relation_order": final[
                    "source_target_order_preserved_or_null"
                ],
            }
        else:
            if adjudication_rows:
                errors.append("unneeded_adjudication")
            normalized = {
                "verdict": records[0]["verdict"],
                "ast": records[0]["inferred_canonical_ast_or_null"],
                "construction": records[0]["construction_family_realized"],
                "relation_order": records[0][
                    "source_target_order_preserved_or_null"
                ],
            }
        prompt = prompt_by_id[submission["prompt_id"]]
        finalized.append((prompt, normalized))

    agreement_rate = (
        exact_verdict_agreements / len(submissions) if submissions else 0.0
    )
    if agreement_rate < protocol["validation"][
        "minimumRawAgreementBeforePopulationSeal"
    ]:
        errors.append("validator_agreement")

    accepted_primary = []
    accepted_abstention = []
    for prompt, final in finalized:
        if prompt["stratum"] == "primary":
            if (
                final["verdict"] == "equivalent_unique"
                and final["ast"] == prompt["intended_semantics"]
                and final["construction"]
            ):
                accepted_primary.append(prompt)
        elif (
            prompt["stratum"] == "abstention"
            and final["verdict"] in {"ambiguous", "unsupported"}
            and final["ast"] is None
        ):
            accepted_abstention.append(prompt)
        else:
            errors.append("unknown_stratum")

    quotas = protocol["quotas"]
    pilot_ids = {
        row["anonymous_writer_id"]
        for row in prompts if row["writer_cohort"] == "pilot"
    }
    evaluation_ids = {
        row["anonymous_writer_id"]
        for row in prompts if row["writer_cohort"] == "evaluation"
    }
    if pilot_ids & evaluation_ids:
        errors.append("pilot_evaluation_writer_overlap")
    if len(pilot_ids) != quotas["pilotAuthors"]:
        errors.append("pilot_author_census")
    if len(evaluation_ids) < quotas["minimumEvaluationAuthors"]:
        errors.append("evaluation_author_census")

    primary_counts = Counter(
        (row["writer_cohort"], row["anonymous_writer_id"])
        for row in accepted_primary
    )
    primary_family_counts = Counter(
        (
            row["writer_cohort"],
            row["anonymous_writer_id"],
            row["construction_family"],
        )
        for row in accepted_primary
    )
    abstention_counts = Counter(
        row["anonymous_writer_id"] for row in accepted_abstention
    )
    abstention_family_counts = Counter(
        (row["anonymous_writer_id"], row["construction_family"])
        for row in accepted_abstention
    )
    for writer_id in pilot_ids:
        if primary_counts[("pilot", writer_id)] != quotas[
            "acceptedPrimaryPerPilotAuthor"
        ]:
            errors.append("pilot_primary_quota")
        for family in protocol["constructionSplit"]["pilotExposedFamilies"]:
            if primary_family_counts[("pilot", writer_id, family)] != quotas[
                "acceptedPrimaryPerPilotExposedFamilyPerPilotAuthor"
            ]:
                errors.append("pilot_family_quota")
        for family in protocol["constructionSplit"]["evaluationOnlyFamilies"]:
            if primary_family_counts[("pilot", writer_id, family)] != 0:
                errors.append("evaluation_only_family_leak_into_pilot")
    all_families = (
        protocol["constructionSplit"]["pilotExposedFamilies"]
        + protocol["constructionSplit"]["evaluationOnlyFamilies"]
    )
    for writer_id in evaluation_ids:
        if primary_counts[("evaluation", writer_id)] != quotas[
            "acceptedPrimaryPerEvaluationAuthor"
        ]:
            errors.append("evaluation_primary_quota")
        if abstention_counts[writer_id] != quotas[
            "acceptedAbstentionPerEvaluationAuthor"
        ]:
            errors.append("evaluation_abstention_quota")
        for family in all_families:
            if primary_family_counts[("evaluation", writer_id, family)] != quotas[
                "acceptedPrimaryPerFamilyPerEvaluationAuthor"
            ]:
                errors.append("evaluation_primary_family_quota")
            if abstention_family_counts[(writer_id, family)] != quotas[
                "acceptedAbstentionPerFamilyPerEvaluationAuthor"
            ]:
                errors.append("evaluation_abstention_family_quota")
    if len(accepted_primary) < (
        quotas["pilotAuthors"] * quotas["acceptedPrimaryPerPilotAuthor"]
        + quotas["minimumAcceptedEvaluationPrimary"]
    ):
        errors.append("total_primary_quota")
    if len(accepted_abstention) < quotas["minimumAcceptedEvaluationAbstention"]:
        errors.append("total_abstention_quota")

    return {
        "passed": not errors,
        "errors": sorted(Counter(errors).items()),
        "metrics": {
            "raw_validator_verdict_agreement": agreement_rate,
            "pilot_authors": len(pilot_ids),
            "evaluation_authors": len(evaluation_ids),
            "accepted_primary": len(accepted_primary),
            "accepted_abstention": len(accepted_abstention),
            "adjudications": len(adjudications),
        },
    }
