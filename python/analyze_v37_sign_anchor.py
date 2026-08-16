#!/usr/bin/env python3
"""Post-V37 oracle-lexicon diagnostic for ontology-anchored lexical sign.

The parser never reads a target or prediction.  It does use the generator's
positive/negative lexical forms, which are not currently exposed in the agent
input; its score is therefore diagnostic rather than a deployable result.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import re
from typing import Any

from v22r2_grounding import PROJECT_ROOT


V32_CONFIG = PROJECT_ROOT / "configs/v32-factorized-semantics.json"
V32_FIT = PROJECT_ROOT / "data/v32-factorized-semantics/factor_fit.jsonl"
V36 = PROJECT_ROOT / "data/v36-independent-confirmation/independent_confirmation.jsonl"
V37 = PROJECT_ROOT / "data/v37-semantic-invariance/semantic_invariance_validation.jsonl"
V37_PREDICTIONS = PROJECT_ROOT / "outputs/v37-semantic-invariance/evaluation/selected-predictions.jsonl"
OUTPUT = PROJECT_ROOT / "outputs/v37-semantic-invariance/sign-anchor-diagnostic.json"
MARKDOWN = PROJECT_ROOT / "docs/v37-sign-anchor-diagnostic.md"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def render(template: str, arguments: tuple[str, ...]) -> str:
    if len(arguments) == 1:
        return template.format(entity=arguments[0])
    return template.format(source=arguments[0], target=arguments[1])


def lexical_candidates(row: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    entities = row["agent_input"]["entities"]
    by_type: dict[str, list[str]] = defaultdict(list)
    for entity in entities:
        by_type[entity["entity_type"]].append(entity["id"])
    candidates = []
    for predicate in config["ontology"]["unaryPredicates"]:
        for entity in by_type[predicate["entityType"]]:
            for sign, form in (("positive", predicate["trueForm"]), ("negative", predicate["falseForm"])):
                candidates.append({
                    "predicate": predicate["id"], "arguments": (entity,), "sign": sign,
                    "text": render(form, (entity,)),
                })
    for relation in config["ontology"]["relations"]:
        for source in by_type[relation["sourceType"]]:
            for target in by_type[relation["targetType"]]:
                if source == target:
                    continue
                forms = (
                    ("positive", relation["directTrueForm"]),
                    ("negative", relation["directFalseForm"]),
                    ("positive", relation["inverseTrueForm"]),
                    ("negative", relation["inverseFalseForm"]),
                )
                for sign, form in forms:
                    candidates.append({
                        "predicate": relation["id"], "arguments": (source, target), "sign": sign,
                        "text": render(form, (source, target)),
                    })
    return candidates


def anchored_sign(row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    evidence = row["agent_input"]["evidence_text"]
    matches = []
    for candidate in lexical_candidates(row, config):
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(candidate['text'])}(?![A-Za-z0-9_])"
        for match in re.finditer(pattern, evidence, flags=re.IGNORECASE):
            matches.append({**candidate, "start": match.start(), "end": match.end()})
    # Longer match wins if two lexicalizations begin at the same position.
    matches.sort(key=lambda row: (row["start"], -(row["end"] - row["start"]), row["text"]))
    if not matches:
        return {"sign": None, "matched_literals": 0, "distinct_signs": []}
    return {
        "sign": matches[0]["sign"],
        "matched_literals": len(matches),
        "distinct_signs": sorted({row["sign"] for row in matches}),
        "first_match": {key: matches[0][key] for key in ("predicate", "arguments", "sign", "text", "start", "end")},
    }


def score(rows, config):
    details = []
    for row in rows:
        parsed = anchored_sign(row, config)
        details.append({
            "id": row["id"],
            "correct": parsed["sign"] == row["target"]["factorization"]["lexical_sign"],
            "matched_literals": parsed["matched_literals"],
            "both_signs_present": parsed["distinct_signs"] == ["negative", "positive"],
        })
    return {
        "records": len(details),
        "coverage": sum(row["matched_literals"] > 0 for row in details) / len(details),
        "lexical_sign_accuracy": sum(row["correct"] for row in details) / len(details),
        "records_with_both_signs_present": sum(row["both_signs_present"] for row in details),
    }


def main() -> None:
    config = json.loads(V32_CONFIG.read_text())
    corpora = {
        "v32_factor_fit": read_jsonl(V32_FIT),
        "v36_exposed_confirmation": read_jsonl(V36),
        "v37_exposed_validation": read_jsonl(V37),
    }
    scores = {name: score(rows, config) for name, rows in corpora.items()}
    v37_rows = corpora["v37_exposed_validation"]
    predictions = {row["id"]: row["prediction"] for row in read_jsonl(V37_PREDICTIONS)}
    by_placement: dict[str, Counter[str]] = defaultdict(Counter)
    by_operation: dict[str, Counter[str]] = defaultdict(Counter)
    confusion: Counter[tuple[str, str]] = Counter()
    for row in v37_rows:
        target = row["target"]["factorization"]["lexical_sign"]
        predicted = predictions[row["id"]]["lexical_sign"]
        correct = predicted == target
        by_placement[row["oracle_metadata"]["distractor_placement"]]["records"] += 1
        by_placement[row["oracle_metadata"]["distractor_placement"]]["correct"] += correct
        operation = row["target"]["factorization"]["outer_operation"]
        by_operation[operation]["records"] += 1
        by_operation[operation]["correct"] += correct
        confusion[(target, predicted)] += 1
    summarize = lambda groups: {
        name: {"records": values["records"], "lexical_sign_accuracy": values["correct"] / values["records"]}
        for name, values in sorted(groups.items())
    }
    result = {
        "schema_version": 37,
        "experiment": "v37_post_outcome_sign_anchor_diagnostic",
        "status": "descriptive_oracle_lexicon_not_deployable",
        "corpora": scores,
        "selected_sign_readout": {
            "confusion": {f"{target}->{prediction}": count for (target, prediction), count in sorted(confusion.items())},
            "by_distractor_placement": summarize(by_placement),
            "by_operation": summarize(by_operation),
        },
        "limitations": {
            "uses_generator_true_and_false_lexical_forms_not_present_in_agent_input": True,
            "chooses_earliest_grounded_literal_when_both_polarities_are_present": True,
            "does_not_test_reversed_focus_order": True,
            "may_not_be_used_as_a_v37_result_amendment": True,
        },
        "data_access": {
            "model_forward_passes": 0,
            "fits": 0,
            "selection_runs": 0,
            "v32_calibration_records_read": 0,
            "v32_evaluation_records_read": 0,
            "v28_runs": 0,
        },
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    v37_anchor = scores["v37_exposed_validation"]
    lines = [
        "# V37 post-outcome lexical-sign anchoring diagnostic", "",
        "Status: descriptive only. This did not amend V37, fit a model, access V32 calibration/evaluation, or run V28.", "",
        "## Finding", "",
        f"Exact matching against the generator's positive and negative ontology lexicalizations recovers a literal in {v37_anchor['coverage']:.1%} of V37 clauses and obtains {v37_anchor['lexical_sign_accuracy']:.3f} sign accuracy. It also reaches {scores['v36_exposed_confirmation']['lexical_sign_accuracy']:.3f} on exposed V36 and {scores['v32_factor_fit']['lexical_sign_accuracy']:.3f} on V32 fit.", "",
        "This shows that sign is mechanically recoverable once the grounded literal's positive/negative lexical forms are available. It does not show that the current agent can do so: those forms are stored in the generator config, not exposed in `agent_input`, and the diagnostic resolves two-literal cases by choosing the earliest match.", "",
        "## Consequence", "",
        "The justified pivot is an ontology-anchored constrained parser, not another linear hidden-state prompt. A proper next test must expose lexical definitions through the declared ontology, parse grounded literal spans, and include counterexamples where the mentioned opposite precedes the focused literal so that first-match heuristics cannot pass.", "",
        "A stronger frozen grounder remains a later comparator, but V37 does not justify changing the backbone before testing whether the declared symbolic interface can supply the missing lexical anchor.",
    ]
    MARKDOWN.write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
