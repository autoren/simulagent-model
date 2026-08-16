#!/usr/bin/env python3
"""Validate V58 pilot submissions and reviews without printing human text."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from render_v58_pilot_form import validate_release_lock
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v58_collection_protocol import (
    adjudication_errors,
    submission_errors,
    validation_errors,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _prompts(packets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [prompt for packet in packets for prompt in packet["prompts"]]


def audit_pilot_submissions(
    packets: list[dict[str, Any]],
    submissions: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    prompts = _prompts(packets)
    prompt_by_id = {row["prompt_id"]: row for row in prompts}
    if (
        len(packets) != protocol["quotas"]["pilotAuthors"]
        or any(
            packet["writer_cohort"] != "pilot"
            or packet["slot_status"] != "active"
            or len(packet["prompts"])
            != protocol["quotas"]["acceptedPrimaryPerPilotAuthor"]
            for packet in packets
        )
        or len(prompt_by_id) != len(prompts) == 120
    ):
        errors.append("pilot_packet_census")
    submission_ids = [row.get("submission_id") for row in submissions]
    if len(set(submission_ids)) != len(submission_ids):
        errors.append("duplicate_submission_id")
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for submission in submissions:
        prompt = prompt_by_id.get(submission.get("prompt_id"))
        if prompt is None:
            errors.append("unknown_prompt")
            continue
        if submission_errors(submission, prompt, protocol):
            errors.append("invalid_submission")
        by_prompt[prompt["prompt_id"]].append(submission)
    if set(by_prompt) != set(prompt_by_id) or any(
        len(rows) != 1 for rows in by_prompt.values()
    ):
        errors.append("prompt_submission_census")
    counts = Counter(
        (prompt_by_id[row["prompt_id"]]["anonymous_writer_id"],
         prompt_by_id[row["prompt_id"]]["construction_family"])
        for row in submissions if row.get("prompt_id") in prompt_by_id
    )
    for packet in packets:
        writer = packet["anonymous_writer_id"]
        for family in protocol["constructionSplit"]["pilotExposedFamilies"]:
            if counts[(writer, family)] != protocol["quotas"][
                "acceptedPrimaryPerPilotExposedFamilyPerPilotAuthor"
            ]:
                errors.append("writer_family_submission_quota")
    return {
        "passed": not errors,
        "errors": sorted(Counter(errors).items()),
        "metrics": {
            "packets": len(packets),
            "prompts": len(prompts),
            "submissions": len(submissions),
            "writers": len({packet["anonymous_writer_id"] for packet in packets}),
        },
    }


def _disagrees(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return any(
        left[key] != right[key]
        for key in (
            "verdict",
            "inferred_canonical_ast_or_null",
            "construction_family_realized",
            "source_target_order_preserved_or_null",
        )
    )


def audit_pilot_reviews(
    packets: list[dict[str, Any]],
    submissions: list[dict[str, Any]],
    validations: list[dict[str, Any]],
    adjudications: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    submission_audit = audit_pilot_submissions(packets, submissions, protocol)
    errors = [name for name, count in submission_audit["errors"] for _ in range(count)]
    prompts = _prompts(packets)
    prompt_by_id = {row["prompt_id"]: row for row in prompts}
    submission_by_id = {row["submission_id"]: row for row in submissions}
    if len({row.get("validation_id") for row in validations}) != len(validations):
        errors.append("duplicate_validation_id")
    if len({row.get("adjudication_id") for row in adjudications}) != len(adjudications):
        errors.append("duplicate_adjudication_id")
    validation_by_submission: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in validations:
        submission_id = row.get("submission_id")
        if submission_id not in submission_by_id:
            errors.append("validation_unknown_submission")
            continue
        if validation_errors(row, submission_id, protocol):
            errors.append("invalid_validation")
        validation_by_submission[submission_id].append(row)
    adjudication_by_submission: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in adjudications:
        adjudication_by_submission[row.get("submission_id")].append(row)

    verdict_agreements = 0
    accepted_prompts = []
    for submission_id, submission in submission_by_id.items():
        rows = validation_by_submission.get(submission_id, [])
        if len(rows) != protocol["validation"]["validatorsPerSubmission"]:
            errors.append("validator_census")
            continue
        validator_ids = {row["anonymous_validator_id"] for row in rows}
        if len(validator_ids) != 2:
            errors.append("duplicate_validator")
        if submission["anonymous_writer_id"] in validator_ids:
            errors.append("validator_writer_overlap")
        verdict_agreements += int(rows[0]["verdict"] == rows[1]["verdict"])
        disagreement = _disagrees(rows[0], rows[1])
        adjudication_rows = adjudication_by_submission.get(submission_id, [])
        if disagreement:
            if len(adjudication_rows) != 1:
                errors.append("missing_or_duplicate_adjudication")
                continue
            final_row = adjudication_rows[0]
            if adjudication_errors(
                final_row,
                submission_id,
                validator_ids,
                submission["anonymous_writer_id"],
                protocol,
            ):
                errors.append("invalid_adjudication")
                continue
            verdict = final_row["final_verdict"]
            ast = final_row["final_canonical_ast_or_null"]
            construction = final_row["construction_family_realized"]
            relation_order = final_row["source_target_order_preserved_or_null"]
        else:
            if adjudication_rows:
                errors.append("unneeded_adjudication")
            verdict = rows[0]["verdict"]
            ast = rows[0]["inferred_canonical_ast_or_null"]
            construction = rows[0]["construction_family_realized"]
            relation_order = rows[0]["source_target_order_preserved_or_null"]
        prompt = prompt_by_id[submission["prompt_id"]]
        if (
            verdict == "equivalent_unique"
            and ast == prompt["intended_semantics"]
            and construction
            and relation_order is not False
        ):
            accepted_prompts.append(prompt)

    agreement_rate = verdict_agreements / len(submissions) if submissions else 0.0
    if agreement_rate < protocol["validation"][
        "minimumRawAgreementBeforePopulationSeal"
    ]:
        errors.append("validator_agreement")
    accepted_counts = Counter(
        (row["anonymous_writer_id"], row["construction_family"])
        for row in accepted_prompts
    )
    for packet in packets:
        writer = packet["anonymous_writer_id"]
        for family in protocol["constructionSplit"]["pilotExposedFamilies"]:
            if accepted_counts[(writer, family)] != protocol["quotas"][
                "acceptedPrimaryPerPilotExposedFamilyPerPilotAuthor"
            ]:
                errors.append("accepted_writer_family_quota")
    if len(accepted_prompts) != 120:
        errors.append("accepted_pilot_census")
    return {
        "passed": not errors,
        "errors": sorted(Counter(errors).items()),
        "metrics": {
            **submission_audit["metrics"],
            "validations": len(validations),
            "adjudications": len(adjudications),
            "raw_validator_verdict_agreement": agreement_rate,
            "accepted_prompts": len(accepted_prompts),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-lock", required=True)
    parser.add_argument("--submissions", required=True)
    parser.add_argument("--validations")
    parser.add_argument("--adjudications")
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    release_path = (PROJECT_ROOT / args.release_lock).resolve()
    release = json.loads(release_path.read_text())
    seal = json.loads(
        (PROJECT_ROOT / "configs/v58-author-packet-seal.json").read_text()
    )
    validate_release_lock(release, release_path, seal)
    packets = [
        json.loads((PROJECT_ROOT / row["path"]).read_text())
        for row in release["pilot_packet_artifacts"]
    ]
    protocol = json.loads((PROJECT_ROOT / seal["protocol"]).read_text())
    submissions_path = (PROJECT_ROOT / args.submissions).resolve()
    submissions = read_jsonl(submissions_path)
    input_hashes = {str(submissions_path.relative_to(PROJECT_ROOT)): file_sha256(submissions_path)}
    if args.validations:
        validation_path = (PROJECT_ROOT / args.validations).resolve()
        adjudication_path = (PROJECT_ROOT / args.adjudications).resolve()
        validations = read_jsonl(validation_path)
        adjudications = read_jsonl(adjudication_path)
        audit = audit_pilot_reviews(
            packets, submissions, validations, adjudications, protocol
        )
        input_hashes.update({
            str(validation_path.relative_to(PROJECT_ROOT)): file_sha256(validation_path),
            str(adjudication_path.relative_to(PROJECT_ROOT)): file_sha256(adjudication_path),
        })
        stage = "pilot_review_intake"
    else:
        audit = audit_pilot_submissions(packets, submissions, protocol)
        stage = "pilot_submission_intake"
    receipt_path = (PROJECT_ROOT / args.receipt).resolve()
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    receipt = {
        "schema_version": 58,
        "experiment": stage,
        "passed": audit["passed"],
        "errors": audit["errors"],
        "metrics": audit["metrics"],
        "release_lock": str(release_path.relative_to(PROJECT_ROOT)),
        "release_lock_sha256": file_sha256(release_path),
        "packet_seal": release["packet_seal"],
        "packet_seal_sha256": release["packet_seal_sha256"],
        "input_files_sha256": input_hashes,
        "human_text_emitted_to_stdout": 0,
        "candidate_invocations": 0,
        "model_forward_passes": 0,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "receipt": str(receipt_path.relative_to(PROJECT_ROOT)),
        "receipt_sha256": file_sha256(receipt_path),
        "passed": audit["passed"],
        "errors": audit["errors"],
        "metrics": audit["metrics"],
    }, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
