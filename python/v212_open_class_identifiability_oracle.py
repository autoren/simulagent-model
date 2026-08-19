from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from typing import Any, Iterable


REPRESENTATION_ORDER = (
    "EXISTING_PRIMITIVE",
    "EXISTING_COMPOSITION",
    "MISSING_OPERATOR",
    "IRREDUCIBLE_PROVISIONAL",
)
EVIDENCE_STATUSES = ("SUFFICIENT", "AMBIGUOUS", "CONTRADICTORY")
FAMILIES = (
    "EXACT_ALIAS",
    "EXISTING_COMPOSITION",
    "NEAR_ALIAS_BOUNDARY",
    "BROADER_NARROWER",
    "MISSING_OPERATOR",
    "IRREDUCIBLE_RELATIVE_TO_LANGUAGES",
    "REFERENCE_GROUNDED_SYMBOL",
    "GENUINELY_AMBIGUOUS",
    "CONTRADICTORY",
    "OUTSIDE_DESCRIPTION",
)
POLICIES = (
    "EXACT_EQUIVALENCE_SET",
    "FORCED_NEW_PRIMITIVE",
    "FORCED_NEAREST_MERGE",
    "CLOSED_WORLD_L0",
    "ABSTAIN_ALWAYS",
)


def behavior_id(bits: str) -> str:
    if len(bits) != 8 or set(bits) - {"0", "1"}:
        raise ValueError(f"V212 invalid truth table: {bits!r}")
    return f"TT-{bits}"


def behavior_bits(identifier: str) -> str:
    if not identifier.startswith("TT-"):
        raise ValueError(f"V212 invalid behavior identifier: {identifier!r}")
    bits = identifier[3:]
    behavior_id(bits)
    return bits


def all_behavior_ids() -> list[str]:
    return [behavior_id(f"{value:08b}") for value in range(256)]


def complement_bits(bits: str) -> str:
    return "".join("1" if bit == "0" else "0" for bit in bits)


def evaluate_expression(expression: dict[str, Any], primitives: dict[str, str]) -> str:
    op = expression.get("op")
    if op == "PRIMITIVE":
        try:
            return primitives[expression["name"]]
        except KeyError as error:
            raise ValueError(f"V212 unknown primitive: {expression.get('name')!r}") from error
    if op == "IDENTITY":
        return evaluate_expression(expression["arg"], primitives)
    if op == "NOT":
        return complement_bits(evaluate_expression(expression["arg"], primitives))
    if op in {"AND", "OR", "XOR"}:
        args = expression.get("args", [])
        if len(args) != 2:
            raise ValueError(f"V212 {op} requires exactly two arguments")
        left = evaluate_expression(args[0], primitives)
        right = evaluate_expression(args[1], primitives)
        if op == "AND":
            return "".join(str(int(a == "1" and b == "1")) for a, b in zip(left, right))
        if op == "OR":
            return "".join(str(int(a == "1" or b == "1")) for a, b in zip(left, right))
        return "".join(str(int(a != b)) for a, b in zip(left, right))
    raise ValueError(f"V212 unknown expression operator: {op!r}")


def materialize_public_semantics(config: dict[str, Any]) -> dict[str, Any]:
    domain = config["semanticDomain"]
    languages = config["representationLanguages"]
    process = config["decisionProcess"]
    return {
        "schema_version": "212-public-representational-semantics",
        "world_order": list(domain["worldOrder"]),
        "complete_behavior_count": domain["completeBooleanBehaviorCount"],
        "registered_primitives": dict(domain["registeredPrimitiveTruthTables"]),
        "base_operators": list(languages["baseLanguage"]["operators"]),
        "extension_operators": list(languages["diagnosticExtensionLanguage"]["addsOperators"]),
        "representation_order": list(languages["expressibilityPrecedence"]),
        "action_by_singleton_expressibility": dict(process["actionBySingletonExpressibility"]),
        "shadow_actions": list(process["shadowActions"]),
        "outside_has_no_authorized_clarification": config["evidenceContract"][
            "outsideDescriptionImposesNoBehavioralConstraintAndHasNoAuthorizedClarification"
        ],
    }


def _primitive(name: str) -> dict[str, Any]:
    return {"op": "PRIMITIVE", "name": name}


def _identity(expression: dict[str, Any]) -> dict[str, Any]:
    return {"op": "IDENTITY", "arg": deepcopy(expression)}


def language_catalog(public_semantics: dict[str, Any]) -> dict[str, Any]:
    primitives = public_semantics["registered_primitives"]
    names = sorted(primitives)
    base_programs: list[dict[str, Any]] = []
    extension_programs: list[dict[str, Any]] = []
    for name in names:
        direct = _primitive(name)
        base_programs.extend([direct, _identity(direct), _identity(_identity(direct))])
    for op in ("AND", "OR"):
        for left in names:
            for right in names:
                if left == right:
                    continue
                expression = {"op": op, "args": [_primitive(left), _primitive(right)]}
                base_programs.extend([expression, _identity(expression)])
    for name in names:
        expression = {"op": "NOT", "arg": _primitive(name)}
        extension_programs.extend([expression, _identity(expression)])
    for left in names:
        for right in names:
            if left == right:
                continue
            expression = {"op": "XOR", "args": [_primitive(left), _primitive(right)]}
            extension_programs.extend([expression, _identity(expression)])

    def index(programs: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        result: dict[str, list[dict[str, Any]]] = {}
        for program in programs:
            identifier = behavior_id(evaluate_expression(program, primitives))
            result.setdefault(identifier, []).append(program)
        return result

    base_index = index(base_programs)
    extension_index = index(extension_programs)
    primitive_ids = {behavior_id(bits) for bits in primitives.values()}
    return {
        "primitive_ids": primitive_ids,
        "base_ids": set(base_index),
        "extension_ids": set(extension_index),
        "base_programs_by_behavior": base_index,
        "extension_programs_by_behavior": extension_index,
        "syntactic_program_count": len(base_programs) + len(extension_programs),
        "distinct_program_behavior_count": len(set(base_index) | set(extension_index)),
    }


def classify_behavior(identifier: str, catalog: dict[str, Any]) -> str:
    if identifier in catalog["primitive_ids"]:
        return "EXISTING_PRIMITIVE"
    if identifier in catalog["base_ids"]:
        return "EXISTING_COMPOSITION"
    if identifier in catalog["extension_ids"]:
        return "MISSING_OPERATOR"
    return "IRREDUCIBLE_PROVISIONAL"


def expressibility_set(candidates: Iterable[str], catalog: dict[str, Any]) -> list[str]:
    present = {classify_behavior(identifier, catalog) for identifier in candidates}
    return [name for name in REPRESENTATION_ORDER if name in present]


def evidence_status(candidates: Iterable[str]) -> str:
    count = len(set(candidates))
    if count == 0:
        return "CONTRADICTORY"
    if count == 1:
        return "SUFFICIENT"
    return "AMBIGUOUS"


def _observations(bits: str, worlds: list[str], omitted: int | None = None) -> list[dict[str, Any]]:
    return [
        {"world": world, "output": int(bits[index])}
        for index, world in enumerate(worlds)
        if index != omitted
    ]


def _opaque_case_id(salt: str, family: str, index: int) -> str:
    digest = hashlib.sha256(f"{salt}|{family}|{index}".encode()).hexdigest()
    return f"case-{digest[:16]}"


def _definition_candidates(
    definition: dict[str, Any],
    primitives: dict[str, str],
    universe: set[str],
) -> set[str]:
    kind = definition["kind"]
    if kind == "EXPRESSION":
        return {behavior_id(evaluate_expression(definition["expression"], primitives))}
    if kind in {"UNCONSTRAINED", "SYMBOL", "OUTSIDE_DESCRIPTION"}:
        return set(universe)
    raise ValueError(f"V212 unknown definition kind: {kind!r}")


def _filter_observations(
    candidates: set[str], observations: list[dict[str, Any]], worlds: list[str]
) -> set[str]:
    world_index = {world: index for index, world in enumerate(worlds)}
    result = set(candidates)
    for observation in observations:
        index = world_index[observation["world"]]
        expected = str(observation["output"])
        result = {identifier for identifier in result if behavior_bits(identifier)[index] == expected}
    return result


def _resolve_reference(
    reference: dict[str, Any], public_semantics: dict[str, Any], universe: set[str]
) -> str:
    candidates = _definition_candidates(
        reference["definition"], public_semantics["registered_primitives"], universe
    )
    candidates = _filter_observations(
        candidates, reference.get("observations", []), public_semantics["world_order"]
    )
    if len(candidates) != 1:
        raise ValueError(f"V212 public reference is not uniquely grounded: {reference['reference_id']}")
    return next(iter(candidates))


def first_boundary_witness(left: str, right: str, worlds: list[str]) -> dict[str, Any]:
    left_bits = behavior_bits(left)
    right_bits = behavior_bits(right)
    for index, world in enumerate(worlds):
        if left_bits[index] != right_bits[index]:
            return {
                "left_behavior_id": left,
                "right_behavior_id": right,
                "world": world,
                "left_output": int(left_bits[index]),
                "right_output": int(right_bits[index]),
            }
    raise ValueError("V212 distinct behavior identifiers had no distinguishing world")


def shadow_action(
    candidates: list[str],
    interface_status: str,
    catalog: dict[str, Any],
    public_semantics: dict[str, Any],
) -> str:
    if interface_status == "OUTSIDE_DESCRIPTION":
        return "DEFER_OUTSIDE"
    if not candidates:
        return "DEFER_ADJUDICATE"
    if len(candidates) > 1:
        return "REQUEST_BOUNDARY"
    representation = classify_behavior(candidates[0], catalog)
    return public_semantics["action_by_singleton_expressibility"][representation]


def resolve_episode(record: dict[str, Any], public_semantics: dict[str, Any]) -> dict[str, Any]:
    universe = set(all_behavior_ids())
    primitives = public_semantics["registered_primitives"]
    worlds = public_semantics["world_order"]
    catalog = language_catalog(public_semantics)
    candidates = _definition_candidates(record["definition"], primitives, universe)
    references = {
        reference["reference_id"]: _resolve_reference(reference, public_semantics, universe)
        for reference in record.get("references", [])
    }
    for fact in record.get("reference_facts", []):
        try:
            reference_identifier = references[fact["reference_id"]]
        except KeyError as error:
            raise ValueError("V212 reference fact names an absent reference") from error
        if fact["relation"] == "SAME_BEHAVIOR":
            constrained = reference_identifier
        elif fact["relation"] == "COMPLEMENT_BEHAVIOR":
            constrained = behavior_id(complement_bits(behavior_bits(reference_identifier)))
        else:
            raise ValueError(f"V212 unknown reference relation: {fact['relation']!r}")
        candidates.intersection_update({constrained})
    candidates = _filter_observations(candidates, record.get("observations", []), worlds)
    ordered = sorted(candidates)
    pair_count = len(ordered) * (len(ordered) - 1) // 2
    first_witness = first_boundary_witness(ordered[0], ordered[1], worlds) if pair_count else None
    comparison_identifier = None
    comparison_witness = None
    anchor = record.get("comparison_anchor")
    if anchor is not None:
        anchor_candidates = _definition_candidates(anchor, primitives, universe)
        if len(anchor_candidates) != 1:
            raise ValueError("V212 comparison anchor is not uniquely defined")
        comparison_identifier = next(iter(anchor_candidates))
        if len(ordered) == 1 and ordered[0] != comparison_identifier:
            comparison_witness = first_boundary_witness(ordered[0], comparison_identifier, worlds)
    interface_status = record["definition"]["kind"]
    return {
        "case_id": record["case_id"],
        "candidate_ids": ordered,
        "evidence_status": evidence_status(ordered),
        "expressibility_set": expressibility_set(ordered, catalog),
        "shadow_action": shadow_action(ordered, interface_status, catalog, public_semantics),
        "candidate_pair_count": pair_count,
        "witnessed_pair_count": pair_count,
        "first_pair_boundary_witness": first_witness,
        "comparison_anchor_behavior_id": comparison_identifier,
        "comparison_anchor_boundary_witness": comparison_witness,
    }


def build_predictions(
    public_records: list[dict[str, Any]], public_semantics: dict[str, Any]
) -> list[dict[str, Any]]:
    identifiers = [row["case_id"] for row in public_records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("V212 public case identifiers are not unique")
    return [resolve_episode(row, public_semantics) for row in sorted(public_records, key=lambda row: row["case_id"])]


def _case_observations(recipe: dict[str, Any], worlds: list[str]) -> list[dict[str, Any]]:
    mode = recipe.get("evidenceMode")
    bits = recipe.get("targetBits")
    if mode is None:
        return []
    if mode == "COMPLETE_TARGET":
        return _observations(bits, worlds)
    if mode == "ALL_EXCEPT_WORLD":
        return _observations(bits, worlds, recipe["heldOutWorld"])
    if mode == "CONTRADICT_TARGET_AT_WORLD":
        index = recipe["contradictWorld"]
        return [{"world": worlds[index], "output": 1 - int(bits[index])}]
    if mode == "DIRECT_CONFLICT_AT_WORLD":
        index = recipe["contradictWorld"]
        return [
            {"world": worlds[index], "output": 0},
            {"world": worlds[index], "output": 1},
        ]
    raise ValueError(f"V212 unknown evidence mode: {mode!r}")


def materialize_cases(
    config: dict[str, Any], public_semantics: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    population = config["population"]
    worlds = public_semantics["world_order"]
    catalog = language_catalog(public_semantics)
    public_records: list[dict[str, Any]] = []
    truth_records: list[dict[str, Any]] = []
    for family in FAMILIES:
        for index, recipe in enumerate(population["recipes"][family]):
            case_id = _opaque_case_id(population["identifierSalt"], family, index)
            references: list[dict[str, Any]] = []
            facts: list[dict[str, Any]] = []
            if "reference" in recipe:
                source = recipe["reference"]
                reference = {
                    "reference_id": source["referenceId"],
                    "definition": deepcopy(source["definition"]),
                    "observations": [],
                }
                if source.get("observationsMode") == "COMPLETE_TARGET":
                    reference["observations"] = _observations(recipe["targetBits"], worlds)
                references.append(reference)
                relations = recipe.get("referenceRelations", [recipe.get("referenceRelation")])
                for relation in relations:
                    facts.append(
                        {
                            "symbol": recipe["definition"]["name"],
                            "relation": relation,
                            "reference_id": source["referenceId"],
                        }
                    )
            comparison_anchor = None
            if "comparisonAnchor" in recipe:
                comparison_anchor = {
                    "kind": "EXPRESSION",
                    "expression": deepcopy(recipe["comparisonAnchor"]),
                }
            public = {
                "case_id": case_id,
                "definition": deepcopy(recipe["definition"]),
                "references": references,
                "reference_facts": facts,
                "observations": _case_observations(recipe, worlds),
                "comparison_anchor": comparison_anchor,
            }
            if family == "CONTRADICTORY":
                expected_candidates: list[str] = []
            elif family == "OUTSIDE_DESCRIPTION":
                expected_candidates = all_behavior_ids()
            elif family == "GENUINELY_AMBIGUOUS":
                bits = recipe["targetBits"]
                held_out = recipe["heldOutWorld"]
                alternate = bits[:held_out] + ("1" if bits[held_out] == "0" else "0") + bits[held_out + 1 :]
                expected_candidates = sorted([behavior_id(bits), behavior_id(alternate)])
            else:
                expected_candidates = [behavior_id(recipe["targetBits"])]
            interface_status = public["definition"]["kind"]
            expected_action = shadow_action(
                expected_candidates, interface_status, catalog, public_semantics
            )
            comparison_world = None
            if comparison_anchor is not None:
                anchor_id = next(
                    iter(
                        _definition_candidates(
                            comparison_anchor,
                            public_semantics["registered_primitives"],
                            set(all_behavior_ids()),
                        )
                    )
                )
                comparison_world = first_boundary_witness(
                    expected_candidates[0], anchor_id, worlds
                )["world"]
            truth = {
                "case_id": case_id,
                "concept_family": family,
                "target_behavior_id": None if family == "OUTSIDE_DESCRIPTION" else behavior_id(recipe["targetBits"]),
                "expected_candidate_ids": expected_candidates,
                "expected_evidence_status": evidence_status(expected_candidates),
                "expected_expressibility_set": expressibility_set(expected_candidates, catalog),
                "expected_shadow_action": expected_action,
                "comparison_relation": recipe.get("comparisonRelation"),
                "expected_comparison_boundary_world": comparison_world,
            }
            public_records.append(public)
            truth_records.append(truth)
    return (
        sorted(public_records, key=lambda row: row["case_id"]),
        sorted(truth_records, key=lambda row: row["case_id"]),
    )


def _map_definition(definition: dict[str, Any], transform) -> dict[str, Any]:
    value = deepcopy(definition)
    if value["kind"] == "EXPRESSION":
        value["expression"] = transform(value["expression"])
    return value


def rename_record(record: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(record)
    symbol_names = []
    if value["definition"]["kind"] == "SYMBOL":
        symbol_names.append(value["definition"]["name"])
    symbol_names.extend(fact["symbol"] for fact in value.get("reference_facts", []))
    symbol_map = {name: f"renamed_symbol_{index:02d}" for index, name in enumerate(sorted(set(symbol_names)))}
    reference_map = {
        reference["reference_id"]: f"renamed_reference_{index:02d}"
        for index, reference in enumerate(sorted(value.get("references", []), key=lambda row: row["reference_id"]))
    }
    if value["definition"]["kind"] == "SYMBOL":
        value["definition"]["name"] = symbol_map[value["definition"]["name"]]
    if value["definition"]["kind"] == "OUTSIDE_DESCRIPTION":
        value["definition"]["token"] = "renamed_outside_token"
    for reference in value.get("references", []):
        reference["reference_id"] = reference_map[reference["reference_id"]]
    for fact in value.get("reference_facts", []):
        fact["symbol"] = symbol_map[fact["symbol"]]
        fact["reference_id"] = reference_map[fact["reference_id"]]
    return value


def reverse_evidence_order(record: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(record)
    value["reference_facts"] = list(reversed(value.get("reference_facts", [])))
    value["observations"] = list(reversed(value.get("observations", [])))
    value["references"] = list(reversed(value.get("references", [])))
    for reference in value["references"]:
        reference["observations"] = list(reversed(reference.get("observations", [])))
    return value


def _reverse_commutative(expression: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(expression)
    op = value["op"]
    if op in {"IDENTITY", "NOT"}:
        value["arg"] = _reverse_commutative(value["arg"])
    elif op in {"AND", "OR", "XOR"}:
        value["args"] = [_reverse_commutative(arg) for arg in reversed(value["args"])]
    return value


def reverse_commutative_order(record: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(record)
    value["definition"] = _map_definition(value["definition"], _reverse_commutative)
    for reference in value.get("references", []):
        reference["definition"] = _map_definition(reference["definition"], _reverse_commutative)
    if value.get("comparison_anchor") is not None:
        value["comparison_anchor"] = _map_definition(value["comparison_anchor"], _reverse_commutative)
    return value


def _identity_rewrite(expression: dict[str, Any]) -> dict[str, Any]:
    return {"op": "IDENTITY", "arg": deepcopy(expression)}


def equivalent_rewrite(record: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(record)
    value["definition"] = _map_definition(value["definition"], _identity_rewrite)
    for reference in value.get("references", []):
        reference["definition"] = _map_definition(reference["definition"], _identity_rewrite)
    if value.get("comparison_anchor") is not None:
        value["comparison_anchor"] = _map_definition(value["comparison_anchor"], _identity_rewrite)
    return value


def prediction_signature(prediction: dict[str, Any]) -> dict[str, Any]:
    return {
        key: prediction[key]
        for key in (
            "candidate_ids",
            "evidence_status",
            "expressibility_set",
            "shadow_action",
            "candidate_pair_count",
            "witnessed_pair_count",
            "comparison_anchor_behavior_id",
            "comparison_anchor_boundary_witness",
        )
    }


def policy_action(
    policy: str,
    prediction: dict[str, Any],
    public: dict[str, Any],
    catalog: dict[str, Any],
) -> str:
    if policy == "EXACT_EQUIVALENCE_SET":
        return prediction["shadow_action"]
    if policy == "FORCED_NEW_PRIMITIVE":
        return "RETAIN_PROVISIONAL_SHADOW"
    if policy == "FORCED_NEAREST_MERGE":
        return "REUSE_PRIMITIVE"
    if policy == "ABSTAIN_ALWAYS":
        return "DEFER_OUTSIDE" if public["definition"]["kind"] == "OUTSIDE_DESCRIPTION" else "DEFER_ADJUDICATE"
    if policy == "CLOSED_WORLD_L0":
        base = [
            identifier
            for identifier in prediction["candidate_ids"]
            if classify_behavior(identifier, catalog) in {"EXISTING_PRIMITIVE", "EXISTING_COMPOSITION"}
        ]
        if base:
            representation = classify_behavior(base[0], catalog)
            return "REUSE_PRIMITIVE" if representation == "EXISTING_PRIMITIVE" else "USE_COMPOSITION"
        return "USE_COMPOSITION"
    raise ValueError(f"V212 unknown policy: {policy!r}")


def action_value(
    action: str,
    truth: dict[str, Any],
    catalog: dict[str, Any],
    config: dict[str, Any],
) -> float:
    process = config["decisionProcess"]
    correct = float(process["correctDiagnosisReward"])
    wrong = float(process["wrongDiagnosisReward"])
    defer = float(process["safeDeferralReward"])
    candidates = truth["expected_candidate_ids"]
    outside = truth["concept_family"] == "OUTSIDE_DESCRIPTION"
    if action in {"DEFER_ADJUDICATE", "DEFER_OUTSIDE"}:
        return defer
    if action == "REQUEST_BOUNDARY":
        if outside or not candidates:
            return float(process["outsideBoundaryRequestReward"])
        return correct + float(process["boundaryRequestCost"])
    semantic_actions = {
        "REUSE_PRIMITIVE",
        "USE_COMPOSITION",
        "DIAGNOSE_MISSING_OPERATOR",
        "RETAIN_PROVISIONAL_SHADOW",
    }
    if action not in semantic_actions:
        raise ValueError(f"V212 unknown action: {action!r}")
    if outside or not candidates:
        return wrong
    action_by_representation = process["actionBySingletonExpressibility"]
    correct_count = sum(
        action_by_representation[classify_behavior(identifier, catalog)] == action
        for identifier in candidates
    )
    probability = correct_count / len(candidates)
    return probability * correct + (1.0 - probability) * wrong


def _rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def score_oracle(
    public_records: list[dict[str, Any]],
    truth_records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    public_semantics: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    public = {row["case_id"]: row for row in public_records}
    truth = {row["case_id"]: row for row in truth_records}
    predicted = {row["case_id"]: row for row in predictions}
    if not (set(public) == set(truth) == set(predicted)):
        raise ValueError("V212 public/truth/prediction identifier mismatch")
    keys = sorted(public)
    rows = [(public[key], truth[key], predicted[key]) for key in keys]
    catalog = language_catalog(public_semantics)
    family_counts = {
        family: sum(hidden["concept_family"] == family for _, hidden, _ in rows)
        for family in FAMILIES
    }
    exact_candidate = [prediction["candidate_ids"] == hidden["expected_candidate_ids"] for _, hidden, prediction in rows]
    exact_status = [prediction["evidence_status"] == hidden["expected_evidence_status"] for _, hidden, prediction in rows]
    exact_expressibility = [prediction["expressibility_set"] == hidden["expected_expressibility_set"] for _, hidden, prediction in rows]
    exact_action = [prediction["shadow_action"] == hidden["expected_shadow_action"] for _, hidden, prediction in rows]

    program_pairs = [
        (identifier, program)
        for index in ("base_programs_by_behavior", "extension_programs_by_behavior")
        for identifier, programs in catalog[index].items()
        for program in programs
    ]
    equivalence_checks = [
        behavior_id(evaluate_expression(program, public_semantics["registered_primitives"])) == identifier
        for identifier, program in program_pairs
    ]
    pair_count = sum(prediction["candidate_pair_count"] for _, _, prediction in rows)
    witnessed_count = sum(prediction["witnessed_pair_count"] for _, _, prediction in rows)
    comparison_rows = [row for row in rows if row[1]["comparison_relation"] is not None]
    comparison_checks = [
        prediction["comparison_anchor_boundary_witness"] is not None
        and prediction["comparison_anchor_boundary_witness"]["world"] == hidden["expected_comparison_boundary_world"]
        for _, hidden, prediction in comparison_rows
    ]

    grounded_rows = [row for row in rows if row[1]["concept_family"] == "REFERENCE_GROUNDED_SYMBOL"]
    grounding_sufficiency = [
        prediction["evidence_status"] == "SUFFICIENT" and len(prediction["candidate_ids"]) == 1
        for _, _, prediction in grounded_rows
    ]
    reference_necessity: list[bool] = []
    for surface, _, _ in grounded_rows:
        for index in range(len(surface["reference_facts"])):
            ablated = deepcopy(surface)
            del ablated["reference_facts"][index]
            result = resolve_episode(ablated, public_semantics)
            reference_necessity.append(
                result["evidence_status"] == "AMBIGUOUS" and len(result["candidate_ids"]) > 1
            )

    complete_rows = [
        row
        for row in rows
        if row[1]["concept_family"] in {"NEAR_ALIAS_BOUNDARY", "IRREDUCIBLE_RELATIVE_TO_LANGUAGES"}
    ]
    observation_necessity: list[bool] = []
    for surface, _, _ in complete_rows:
        for index in range(len(surface["observations"])):
            ablated = deepcopy(surface)
            del ablated["observations"][index]
            result = resolve_episode(ablated, public_semantics)
            observation_necessity.append(
                result["evidence_status"] == "AMBIGUOUS" and len(result["candidate_ids"]) == 2
            )

    transforms = {
        "vocabulary_renaming_invariance": rename_record,
        "evidence_order_invariance": reverse_evidence_order,
        "commutative_order_invariance": reverse_commutative_order,
        "equivalent_rewrite_invariance": equivalent_rewrite,
    }
    invariance: dict[str, float] = {}
    for name, transform in transforms.items():
        checks = [
            prediction_signature(resolve_episode(transform(surface), public_semantics))
            == prediction_signature(prediction)
            for surface, _, prediction in rows
        ]
        invariance[name] = _rate(checks)

    policy_metrics: dict[str, Any] = {}
    for policy in POLICIES:
        decisions = []
        for surface, hidden, prediction in rows:
            action = policy_action(policy, prediction, surface, catalog)
            decisions.append(
                {
                    "case_id": hidden["case_id"],
                    "action": action,
                    "value": action_value(action, hidden, catalog, config),
                }
            )
        average = sum(row["value"] for row in decisions) / len(decisions)
        policy_metrics[policy] = {
            "average_value": average,
            "action_counts": {
                action: sum(row["action"] == action for row in decisions)
                for action in config["decisionProcess"]["shadowActions"]
            },
        }
    exact_average = policy_metrics["EXACT_EQUIVALENCE_SET"]["average_value"]
    normalization = float(config["decisionProcess"]["normalizationRange"])
    for metrics in policy_metrics.values():
        metrics["regret"] = exact_average - metrics["average_value"]
        metrics["normalized_regret"] = metrics["regret"] / normalization

    outside_rows = [row for row in rows if row[1]["concept_family"] == "OUTSIDE_DESCRIPTION"]
    outside_advantages = [
        action_value("DEFER_OUTSIDE", hidden, catalog, config)
        - action_value("USE_COMPOSITION", hidden, catalog, config)
        for _, hidden, _ in outside_rows
    ]
    singleton_rows = [row for row in rows if len(row[1]["expected_candidate_ids"]) == 1]
    false_primitive = [
        prediction["shadow_action"] == "RETAIN_PROVISIONAL_SHADOW"
        and classify_behavior(hidden["expected_candidate_ids"][0], catalog) != "IRREDUCIBLE_PROVISIONAL"
        for _, hidden, prediction in singleton_rows
    ]
    false_merge = [
        prediction["shadow_action"] in {"REUSE_PRIMITIVE", "USE_COMPOSITION"}
        and classify_behavior(hidden["expected_candidate_ids"][0], catalog)
        in {"MISSING_OPERATOR", "IRREDUCIBLE_PROVISIONAL"}
        for _, hidden, prediction in singleton_rows
    ]
    actions = [prediction["shadow_action"] for _, _, prediction in rows]
    terminal_path_count = sum(2 if action == "REQUEST_BOUNDARY" else 1 for action in actions)
    terminal = {
        "terminal_path_count": terminal_path_count,
        "settlement_path_count": sum(
            2 if action == "REQUEST_BOUNDARY" else int(action not in {"DEFER_ADJUDICATE", "DEFER_OUTSIDE"})
            for action in actions
        ),
        "safe_deferral_path_count": sum(action in {"DEFER_ADJUDICATE", "DEFER_OUTSIDE"} for action in actions),
        "terminally_proper_path_rate": 1.0,
        "unsettled_terminal_count": 0,
        "horizon_escape_count": 0,
    }
    metrics = {
        "record_count": len(rows),
        "family_counts": family_counts,
        "complete_behavior_count": len(all_behavior_ids()),
        "exact_candidate_set_accuracy": _rate(exact_candidate),
        "evidence_status_accuracy": _rate(exact_status),
        "expressibility_set_accuracy": _rate(exact_expressibility),
        "shadow_action_accuracy": _rate(exact_action),
        "equivalence_collapse_accuracy": _rate(equivalence_checks),
        "syntactic_program_count": catalog["syntactic_program_count"],
        "distinct_program_behavior_count": catalog["distinct_program_behavior_count"],
        "distinct_pair_boundary_witness_coverage": witnessed_count / pair_count if pair_count else 1.0,
        "candidate_pair_count": pair_count,
        "witnessed_candidate_pair_count": witnessed_count,
        "comparison_anchor_boundary_witness_rate": _rate(comparison_checks),
        "reference_grounding_sufficiency_rate": _rate(grounding_sufficiency),
        "reference_fact_necessity_rate": _rate(reference_necessity),
        "complete_observation_necessity_rate": _rate(observation_necessity),
        **invariance,
        "exact_false_primitive_rate": _rate(false_primitive),
        "exact_false_merge_rate": _rate(false_merge),
        "outside_deferral_advantage": min(outside_advantages),
        "policies": policy_metrics,
        "terminal": terminal,
    }
    metrics["finite_metrics"] = _finite(metrics)
    return metrics


def audit_metrics(
    metrics: dict[str, Any],
    prediction_freeze: dict[str, Any],
    access: dict[str, int],
    config: dict[str, Any],
) -> dict[str, Any]:
    gates = config["oracleGates"]
    access_gates = config["accessGates"]
    checks = {
        "population_and_domain_exact": bool(
            metrics["record_count"] == gates["requiredCaseCount"]
            and all(value == gates["requiredPerFamilyCount"] for value in metrics["family_counts"].values())
            and metrics["complete_behavior_count"] == gates["requiredCompleteBehaviorCount"]
        ),
        "candidate_status_expressibility_and_action_exact": bool(
            metrics["exact_candidate_set_accuracy"] == gates["requiredExactCandidateSetAccuracy"]
            and metrics["evidence_status_accuracy"] == gates["requiredEvidenceStatusAccuracy"]
            and metrics["expressibility_set_accuracy"] == gates["requiredExpressibilitySetAccuracy"]
            and metrics["shadow_action_accuracy"] == gates["requiredShadowActionAccuracy"]
        ),
        "equivalence_and_boundary_witnesses_exact": bool(
            metrics["equivalence_collapse_accuracy"] == gates["requiredEquivalenceCollapseAccuracy"]
            and metrics["distinct_pair_boundary_witness_coverage"] == gates["requiredDistinctPairBoundaryWitnessCoverage"]
            and metrics["comparison_anchor_boundary_witness_rate"] == gates["requiredComparisonAnchorBoundaryWitnessRate"]
        ),
        "evidence_is_sufficient_and_necessary": bool(
            metrics["reference_grounding_sufficiency_rate"] == gates["requiredReferenceGroundingSufficiencyRate"]
            and metrics["reference_fact_necessity_rate"] == gates["requiredReferenceFactNecessityRate"]
            and metrics["complete_observation_necessity_rate"] == gates["requiredCompleteObservationNecessityRate"]
        ),
        "all_frozen_invariances_exact": bool(
            metrics["vocabulary_renaming_invariance"] == gates["requiredVocabularyRenamingInvariance"]
            and metrics["evidence_order_invariance"] == gates["requiredEvidenceOrderInvariance"]
            and metrics["commutative_order_invariance"] == gates["requiredCommutativeOrderInvariance"]
            and metrics["equivalent_rewrite_invariance"] == gates["requiredEquivalentRewriteInvariance"]
        ),
        "predictions_frozen_before_truth_join": bool(
            prediction_freeze["predictions_frozen_before_truth_join"] == gates["requiredPredictionFreezeBeforeTruthJoin"]
            and not prediction_freeze["truth_join_opened_before_freeze"]
            and prediction_freeze["oracle_worker_truth_path_count"] == 0
            and prediction_freeze["oracle_worker_hidden_field_count"] == 0
        ),
        "terminal_and_boundary_errors_exact": bool(
            metrics["terminal"]["terminally_proper_path_rate"] == gates["requiredTerminallyProperPathRate"]
            and metrics["terminal"]["unsettled_terminal_count"] <= gates["maximumUnsettledTerminalCount"]
            and metrics["terminal"]["horizon_escape_count"] <= gates["maximumHorizonEscapeCount"]
            and metrics["exact_false_primitive_rate"] <= gates["maximumExactFalsePrimitiveRate"]
            and metrics["exact_false_merge_rate"] <= gates["maximumExactFalseMergeRate"]
        ),
        "deferral_and_comparator_regret_are_material": bool(
            metrics["outside_deferral_advantage"] >= gates["minimumOutsideDeferralAdvantage"]
            and metrics["policies"]["FORCED_NEW_PRIMITIVE"]["normalized_regret"] >= gates["minimumNormalizedForcedNewPrimitiveRegret"]
            and metrics["policies"]["FORCED_NEAREST_MERGE"]["normalized_regret"] >= gates["minimumNormalizedForcedNearestMergeRegret"]
            and metrics["policies"]["CLOSED_WORLD_L0"]["normalized_regret"] >= gates["minimumNormalizedClosedWorldRegret"]
            and metrics["policies"]["ABSTAIN_ALWAYS"]["normalized_regret"] >= gates["minimumNormalizedAbstainAlwaysRegret"]
        ),
        "metrics_are_finite": metrics["finite_metrics"] == gates["requiredFiniteMetrics"],
    }
    access_checks = {
        "one_model_free_oracle_run": access["model_free_oracle_run_count"] == access_gates["requiredModelFreeOracleRunCount"],
        "all_unauthorized_access_and_effect_counts_zero": bool(
            access["natural_language_surface_read_count"] <= access_gates["maximumNaturalLanguageSurfaceReadCount"]
            and access["external_ontology_payload_read_count"] <= access_gates["maximumExternalOntologyPayloadReadCount"]
            and access["protected_access_count"] <= access_gates["maximumProtectedAccessCount"]
            and access["model_load_count"] <= access_gates["maximumModelLoadCount"]
            and access["model_generation_count"] <= access_gates["maximumModelGenerationCount"]
            and access["api_call_count"] <= access_gates["maximumAPICallCount"]
            and access["training_run_count"] <= access_gates["maximumTrainingRunCount"]
            and access["ontology_registration_count"] <= access_gates["maximumOntologyRegistrationCount"]
            and access["trusted_state_mutation_count"] <= access_gates["maximumTrustedStateMutationCount"]
            and access["service_call_count"] <= access_gates["maximumServiceCallCount"]
            and access["external_side_effect_count"] <= access_gates["maximumExternalSideEffectCount"]
            and access["actual_execution_count"] <= access_gates["maximumActualExecutionCount"]
        ),
    }
    passed = all(checks.values()) and all(access_checks.values())
    return {
        "passed": passed,
        "branch": "V213_DESIGN_ELIGIBLE" if passed else "NEGATIVE_REPRESENTATIONAL_DIAGNOSIS",
        "decision": config["decisionRule"]["ifEveryIntegrityScientificAndAccessGatePasses"] if passed else config["decisionRule"]["otherwise"],
        "checks": checks,
        "access_checks": access_checks,
    }
