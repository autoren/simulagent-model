from __future__ import annotations

from collections import Counter
from io import BytesIO
import json
import math
import re
import tarfile
from typing import Any

from v126_sgd_retrieval_selectivity import catalog_maps, hypothesis_action_cost, normalized_kind
from v129_complete_clarification_interface import complete_policy


def _member_json(archive: tarfile.TarFile, member: tarfile.TarInfo) -> Any:
    handle = archive.extractfile(member)
    if handle is None: raise ValueError(f"unreadable archive member: {member.name}")
    return json.loads(handle.read())


def extract_selected_language_and_definitions(
    archive_bytes: bytes, population: dict[str, Any], catalog: dict[str, Any], config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    revision = config["extraction"]["archiveRevision"]
    root = f"dstc8-schema-guided-dialogue-{revision}"
    wanted = {
        row["candidate_id"]: row for row in population["fixtures"]
        if row["observation_available"]
    }
    found: dict[str, dict[str, Any]] = {}
    with tarfile.open(fileobj=BytesIO(archive_bytes), mode="r:gz") as archive:
        files = [member for member in archive.getmembers() if member.isfile()]
        if any(member.name.startswith("/") or ".." in member.name.split("/") for member in files):
            raise ValueError("unsafe archive member")
        schema_members = [member for member in files if member.name == f"{root}/train/schema.json"]
        if len(schema_members) != 1: raise ValueError("expected one train schema")
        schemas = _member_json(archive, schema_members[0])
        schema_by_service = {row["service_name"]: row for row in schemas}
        prompt_choices: list[dict[str, Any]] = []
        for choice in catalog["choices"]:
            if choice["kind"] == "KNOWN":
                service = schema_by_service.get(choice["service"])
                if service is None: raise ValueError(f"known service absent from train schema: {choice['service']}")
                matches = [row for row in service["intents"] if row["name"] == choice["intent"]]
                if len(matches) != 1: raise ValueError("known intent definition absent or ambiguous")
                intent = matches[0]
                prompt_choices.append({
                    "choice_id": choice["choice_id"], "kind": "KNOWN", "domain": choice["domain"],
                    "capability": choice["intent"], "description": intent["description"],
                    "required_slots": sorted(intent.get("required_slots", [])),
                    "optional_slots": sorted(intent.get("optional_slots", [])),
                })
            elif choice["kind"] == "NOVEL_COMPOSITE":
                prompt_choices.append({
                    "choice_id": choice["choice_id"], "kind": "VALID_UNDECLARED", "domain": choice["domain"],
                    "meaning": "a coherent request in this visible domain that no KNOWN capability fully covers",
                })
            elif choice["kind"] == "UNSUPPORTED_COMPOSITE":
                prompt_choices.append({
                    "choice_id": choice["choice_id"], "kind": "UNSUPPORTED",
                    "meaning": f"outside all visible domains: {', '.join(catalog['visible_domains'])}",
                })
            else:
                prompt_choices.append({
                    "choice_id": choice["choice_id"], "kind": "INSUFFICIENT",
                    "meaning": "no usable user utterance or insufficient evidence",
                })
        dialogue_members = [
            member for member in files
            if re.fullmatch(rf"{re.escape(root)}/test/dialogues_\d+\.json", member.name)
        ]
        for member in sorted(dialogue_members, key=lambda row: row.name):
            for dialogue in _member_json(archive, member):
                dialogue_id = str(dialogue["dialogue_id"])
                for turn_index, turn in enumerate(dialogue["turns"]):
                    if turn.get("speaker") != "USER": continue
                    matches: list[tuple[str, str]] = []
                    for frame in turn.get("frames", []):
                        service = frame.get("service"); active = frame.get("state", {}).get("active_intent")
                        actions = [action for action in frame.get("actions", []) if action.get("act") == "INFORM_INTENT" and action.get("slot") == "intent"]
                        values = [value for action in actions for value in action.get("values", [])]
                        if len(actions) == 1 and len(values) == 1 and active == values[0] and active != "NONE": matches.append((service, active))
                    if len(matches) != 1: continue
                    service, intent = matches[0]
                    identifier = f"sgd::test::{dialogue_id}::{turn_index:03d}::{service}::{intent}"
                    if identifier in wanted:
                        structural = wanted[identifier]
                        found[identifier] = {
                            "fixture_id": structural["fixture_id"], "candidate_id": identifier,
                            "utterance": turn["utterance"],
                        }
    if set(found) != set(wanted): raise RuntimeError("V132 selected language extraction mismatch")
    language = [found[row["candidate_id"]] for row in population["fixtures"] if row["observation_available"]]
    if len(language) != config["extraction"]["expectedSelectedLanguageRecordCount"]:
        raise RuntimeError("V132 selected language count mismatch")
    summary = {
        "selected_language_record_count": len(language), "unselected_language_record_count": 0,
        "known_definition_count": sum(row["kind"] == "KNOWN" for row in prompt_choices),
        "novel_member_intent_identifier_emission_count": 0,
    }
    return language, prompt_choices, summary


def render_prompt(
    prompt_choices: list[dict[str, Any]], utterance: str | None, observed: bool,
    presented_candidate: str, config: dict[str, Any],
) -> str:
    if observed and not isinstance(utterance, str): raise ValueError("observed fixture requires utterance")
    if not observed and utterance is not None: raise ValueError("missing fixture cannot expose utterance")
    payload = {
        "instruction": (
            "Select exactly one choice_id. The preliminary candidate is only a fallible proposal. "
            "Choose KNOWN only if that exact declared capability fully covers the request. Choose the matching "
            "VALID_UNDECLARED domain choice if the request is coherent in a visible domain but no KNOWN choice "
            "fully covers it. Choose U00 if outside all visible domains. Choose A00 if evidence is insufficient."
        ),
        "preliminary_candidate_under_review": presented_candidate,
        "choices": prompt_choices,
        "observation_available": observed,
        "user_utterance": utterance if observed else config["prompt"]["missingObservationSentinel"],
        "response_contract": {"required_keys": ["choice_id"], "extra_keys_allowed": False},
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def validate_answer(raw: str | dict[str, Any], catalog: dict[str, Any]) -> tuple[str, bool, str]:
    try: value = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError): return "A00", False, "invalid_json"
    if not isinstance(value, dict) or set(value) != {"choice_id"}: return "A00", False, "invalid_keys"
    choice = value.get("choice_id")
    valid = {row["choice_id"] for row in catalog["choices"]}
    if not isinstance(choice, str) or choice not in valid: return "A00", False, "unknown_choice_id"
    return choice, True, "valid"


def wilson_lower(successes: int, total: int, z: float = 1.6448536269514722) -> float:
    if total <= 0: return 0.0
    p = successes / total; denominator = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    return (center - radius) / denominator


def evaluate_realization(
    population: dict[str, Any], completed: dict[str, dict[str, Any]], catalog: dict[str, Any],
    baseline: dict[str, Any], v130: dict[str, Any], access: dict[str, Any], config: dict[str, Any],
) -> dict[str, Any]:
    fixtures = population["fixtures"]
    if set(completed) != {row["fixture_id"] for row in fixtures}: raise ValueError("V132 fixture completion mismatch")
    by_fixture = {row["fixture_id"]: row for row in fixtures}
    by_id = {row["choice_id"]: row for row in catalog["choices"]}
    rows: list[dict[str, Any]] = []
    for fixture_id, result in completed.items():
        fixture = by_fixture[fixture_id]; answer = result["answer_choice_id"]
        rows.append({
            "truth": fixture["truth_choice_id"], "candidate": fixture["presented_candidate_choice_id"],
            "answer": answer, "valid": result["response_valid"],
            "correct": answer == fixture["truth_choice_id"], "kind": normalized_kind(by_id[fixture["truth_choice_id"]]),
        })
    total = len(rows); correct = sum(row["correct"] for row in rows)
    errors = [row for row in rows if not row["correct"]]
    truth_counts = Counter(row["truth"] for row in rows); truth_correct = Counter(row["truth"] for row in rows if row["correct"])
    per_truth = {truth: truth_correct[truth] / count for truth, count in sorted(truth_counts.items())}
    kind_counts = Counter(row["kind"] for row in rows); kind_correct = Counter(row["kind"] for row in rows if row["correct"])
    per_kind = {kind: kind_correct[kind] / count for kind, count in sorted(kind_counts.items())}
    nonknown = [row for row in rows if row["kind"] != "KNOWN"]
    known_ids = {row["choice_id"] for row in catalog["choices"] if row["kind"] == "KNOWN"}
    metrics = {
        "fixture_count": total,
        "structured_response_validity": sum(row["valid"] for row in rows) / total,
        "overall_exact_choice_accuracy": correct / total,
        "one_sided_95_wilson_lower": wilson_lower(correct, total),
        "per_truth_choice_accuracy": per_truth,
        "per_kind_accuracy": per_kind,
        "candidate_attracted_fraction_of_errors": sum(row["answer"] == row["candidate"] for row in errors) / len(errors) if errors else 0.0,
        "abstention_attracted_fraction_of_errors": sum(row["answer"] == "A00" for row in errors) / len(errors) if errors else 0.0,
        "false_known_answer_rate_on_nonknown_truths": sum(row["answer"] in known_ids for row in nonknown) / len(nonknown),
        "error_count": len(errors),
        "confusion_counts": {f"{truth}->{answer}": count for (truth, answer), count in sorted(Counter((row["truth"], row["answer"]) for row in rows).items())},
    }
    eg = config["evidenceGates"]
    evidence_checks = {
        "structured_response_validity": metrics["structured_response_validity"] >= eg["minimumStructuredResponseValidity"],
        "overall_exact_choice_accuracy": metrics["overall_exact_choice_accuracy"] >= eg["minimumOverallExactChoiceAccuracy"],
        "one_sided_95_wilson_lower": metrics["one_sided_95_wilson_lower"] >= eg["minimumOneSided95WilsonLowerBound"],
        "every_truth_choice_accuracy": all(value >= eg["minimumEveryTruthChoiceAccuracy"] for value in per_truth.values()),
        "known_exact_choice_accuracy": per_kind["KNOWN"] >= eg["minimumKnownExactChoiceAccuracy"],
        "novel_composite_accuracy": per_kind["NOVEL"] >= eg["minimumNovelCompositeAccuracy"],
        "unsupported_accuracy": per_kind["UNSUPPORTED"] >= eg["minimumUnsupportedAccuracy"],
        "missing_observation_accuracy": per_kind["ABSTAIN"] >= eg["minimumMissingObservationAccuracy"],
        "candidate_attraction_bounded": metrics["candidate_attracted_fraction_of_errors"] <= eg["maximumCandidateAttractedFractionOfErrors"],
        "abstention_attraction_bounded": metrics["abstention_attracted_fraction_of_errors"] <= eg["maximumAbstentionAttractedFractionOfErrors"],
        "false_known_answer_rate": metrics["false_known_answer_rate_on_nonknown_truths"] <= eg["maximumFalseKnownAnswerRateOnNonKnownTruths"],
    }
    reliability = config["downstreamGates"]["answerReliabilityUsedByFrozenPlanner"]
    cost = v130["completeClarificationChannel"]["totalCost"]
    conditions: dict[str, Any] = {}
    for prior in v130["priorRegimes"]:
        for regime in config["downstreamGates"]["assumedErrorRegimes"]:
            policies = {
                candidate: complete_policy(candidate, reliability, regime, prior["candidateProbability"], catalog, baseline, v130)
                for candidate in sorted({row["candidate"] for row in rows})
            }
            evaluated = []
            for row in rows:
                action = policies[row["candidate"]][row["answer"]]
                exact_action = ("KNOWN", by_id[row["truth"]]["intent_id"]) if row["kind"] == "KNOWN" else ("UNSUPPORTED", None) if row["kind"] == "UNSUPPORTED" else ("ABSTAIN", None)
                evaluated.append({
                    "kind": row["kind"], "cost": hypothesis_action_cost(row["truth"], action, by_id, baseline) + cost,
                    "known_exact": row["kind"] == "KNOWN" and action == exact_action,
                    "unsupported_correct": row["kind"] == "UNSUPPORTED" and action[0] == "UNSUPPORTED",
                    "false_known": row["kind"] != "KNOWN" and action[0] == "KNOWN",
                })
            known_rows = [row for row in evaluated if row["kind"] == "KNOWN"]
            unsupported_rows = [row for row in evaluated if row["kind"] == "UNSUPPORTED"]
            nonknown_rows = [row for row in evaluated if row["kind"] != "KNOWN"]
            key = f"{prior['id']}@{regime}"
            conditions[key] = {
                "mean_regret": sum(row["cost"] for row in evaluated) / len(evaluated),
                "known_exact_probability": sum(row["known_exact"] for row in known_rows) / len(known_rows),
                "unsupported_correct_probability": sum(row["unsupported_correct"] for row in unsupported_rows) / len(unsupported_rows),
                "false_known_probability": sum(row["false_known"] for row in nonknown_rows) / len(nonknown_rows),
            }
    dg = config["downstreamGates"]
    downstream_checks = {
        "mean_regret_every_condition": all(row["mean_regret"] <= dg["maximumMeanRegretEveryPriorAndAssumedRegime"] for row in conditions.values()),
        "known_exact_every_condition": all(row["known_exact_probability"] >= dg["minimumKnownExactEveryPriorAndAssumedRegime"] for row in conditions.values()),
        "unsupported_every_condition": all(row["unsupported_correct_probability"] >= dg["minimumUnsupportedEveryPriorAndAssumedRegime"] for row in conditions.values()),
        "false_known_every_condition": all(row["false_known_probability"] <= dg["maximumFalseKnownEveryPriorAndAssumedRegime"] for row in conditions.values()),
        "complete_hypothesis_retention": dg["requiredTrueHypothesisRetention"] == 1.0,
        "zero_execution": dg["maximumActualExecutionCount"] == 0,
    }
    ag = config["accessGates"]
    access_checks = {
        "source_archive_read_budget": access["source_archive_read_count"] <= ag["maximumSourceArchiveReadCount"],
        "selected_language_parse_budget": access["automatic_selected_language_parse_count"] <= ag["maximumAutomaticSelectedLanguageParseCount"],
        "persisted_language_budget": access["persisted_selected_language_record_count"] <= ag["maximumPersistedSelectedLanguageRecordCount"],
        "zero_manual_language_or_raw_response_inspection": access["manual_language_or_raw_response_inspection_count"] <= ag["maximumManualLanguageOrRawResponseInspectionCount"],
        "zero_original_protected_language": access["original_protected_language_read_count"] <= ag["maximumOriginalProtectedLanguageReadCount"],
        "model_load_budget": access["model_load_count"] <= ag["maximumModelLoadCount"],
        "model_generation_budget": access["model_generation_count"] <= ag["maximumModelGenerationCount"],
        "zero_API_calls": access["LLM_API_call_count"] <= ag["maximumLLMAPICallCount"],
        "zero_training": access["adapter_training_run_count"] <= ag["maximumAdapterTrainingRunCount"],
        "zero_real_service_calls": access["real_service_call_count"] <= ag["maximumRealServiceCallCount"],
        "zero_external_side_effects": access["external_side_effect_count"] <= ag["maximumExternalSideEffectCount"],
    }
    passed = all(evidence_checks.values()) and all(downstream_checks.values()) and all(access_checks.values())
    return {
        "metrics": metrics, "downstream_conditions": conditions,
        "evidence_gates": evidence_checks, "downstream_gates": downstream_checks, "access_gates": access_checks,
        "evidence_pass": all(evidence_checks.values()), "downstream_pass": all(downstream_checks.values()),
        "access_pass": all(access_checks.values()), "outcome_pass": passed,
        "decision": config["decisionRule"]["ifEvidenceDownstreamAndAccessGatesPass"] if passed else config["decisionRule"]["otherwise"],
        "true_hypothesis_retention": 1.0, "actual_execution_count": 0,
    }


__all__ = ["evaluate_realization", "extract_selected_language_and_definitions", "render_prompt", "validate_answer", "wilson_lower"]
