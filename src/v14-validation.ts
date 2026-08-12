import type { V10Relation } from './v10-contracts';
import type { V14GroundingRecord, V14SurfaceFamily } from './v14-contracts';
import { deriveV10AllowedValues } from './v10-grounding';
import { canonicalJson } from './serialization';
import { evaluateAllowedTransitions } from './v9-symbolic';
import { v14OperatorBySurface, v14SurfaceFamilies, v14SurfacesByOperator } from './v14-grounding';

export interface V14ValidationResult {
  errors: string[];
  records: Record<'train' | 'evaluation', number>;
  contexts: Record<'train' | 'evaluation', number>;
  templates: Record<string, number>;
  semantic_operators: Record<string, number>;
  intervention_groups: number;
  current_hypothesis_pairs: number;
  unresolved_hypothesis_pairs: number;
  malformed_spans: number;
  malformed_hypotheses: number;
  relation_mismatches: number;
  allowed_value_derivation_mismatches: number;
  symbolic_mismatches: number;
  malformed_context_groups: number;
  malformed_intervention_groups: number;
  imbalanced_current_cells: number;
  operator_signature_mismatches: number;
  unsupported_surface_holdouts: string[];
  complement_cross_split_overlaps: number;
  context_cross_split_overlaps: number;
  duplicate_ids: number;
  duplicate_prompts: number;
  cross_split_duplicate_prompts: number;
  conflicting_duplicate_prompts: number;
  determinant_ids_in_observation: number;
  literal_target_labels_in_observation: number;
}

type MentionSignature = 'gold_only' | 'opposite_only' | 'both' | 'neither';

export function currentMentionSignature(
  record: V14GroundingRecord, target: V14GroundingRecord['target']['determinant_grounding'][number],
): MentionSignature {
  const hypotheses = record.agent_input.state_hypotheses.find((value) =>
    value.determinant_id === target.determinant_id,
  )?.statements;
  if (!hypotheses) return 'neither';
  const text = target.evidence_span.text.toLowerCase();
  const present = hypotheses.map((value) => text.includes(value.toLowerCase()));
  const gold = target.current_value === 'active' ? 0 : 1;
  if (present[0] && present[1]) return 'both';
  if (present[gold]) return 'gold_only';
  if (present[1 - gold]) return 'opposite_only';
  return 'neither';
}

export function validateV14Grounding(records: V14GroundingRecord[]): V14ValidationResult {
  const errors: string[] = [];
  const splits = ['train', 'evaluation'] as const;
  const recordCounts = Object.fromEntries(splits.map((split) => [
    split, records.filter((record) => record.split === split).length,
  ])) as Record<typeof splits[number], number>;
  const contextCounts = Object.fromEntries(splits.map((split) => [
    split, new Set(records.filter((record) => record.split === split).map((record) => record.context_group)).size,
  ])) as Record<typeof splits[number], number>;
  if (records.length !== 4860) errors.push(`V14 expected 4,860 records, found ${records.length}.`);
  if (recordCounts.train === 0 || recordCounts.evaluation === 0) errors.push('V14 requires both splits.');
  if (records.some((record) => record.schema_version !== 14)) errors.push('V14 contains a non-schema-14 record.');

  const duplicateIds = records.length - new Set(records.map((record) => record.id)).size;
  if (duplicateIds) errors.push(`V14 has ${duplicateIds} duplicate ids.`);
  const contextsBySplit = Object.fromEntries(splits.map((split) => [split, new Set(
    records.filter((record) => record.split === split).map((record) => record.context_group),
  )])) as Record<typeof splits[number], Set<string>>;
  const complementsBySplit = Object.fromEntries(splits.map((split) => [split, new Set(
    records.filter((record) => record.split === split).map((record) => record.complement_group),
  )])) as Record<typeof splits[number], Set<string>>;
  const contextOverlap = [...contextsBySplit.train].filter((value) => contextsBySplit.evaluation.has(value)).length;
  const complementOverlap = [...complementsBySplit.train].filter((value) => complementsBySplit.evaluation.has(value)).length;
  if (contextOverlap) errors.push(`V14 has ${contextOverlap} contexts crossing splits.`);
  if (complementOverlap) errors.push(`V14 has ${complementOverlap} complements crossing splits.`);

  let malformedSpans = 0;
  let malformedHypotheses = 0;
  let relationMismatches = 0;
  let derivationMismatches = 0;
  let symbolicMismatches = 0;
  let determinantIds = 0;
  let literalLabels = 0;
  let currentPairs = 0;
  let unresolvedPairs = 0;
  let signatureMismatches = 0;
  const signaturesBySurface = new Map<V14SurfaceFamily, Set<MentionSignature>>();
  for (const record of records) {
    if (record.semantic_operator_family !== v14OperatorBySurface[record.template_family]) signatureMismatches += 1;
    if (record.evidence_units.length !== 7) malformedSpans += 1;
    for (const unit of record.evidence_units) {
      if (
        unit.start < 0 || unit.end <= unit.start || unit.end > record.agent_input.observation.length ||
        record.agent_input.observation.slice(unit.start, unit.end) !== unit.text
      ) malformedSpans += 1;
    }
    const hypothesisById = new Map(record.agent_input.state_hypotheses.map((value) => [value.determinant_id, value]));
    if (hypothesisById.size !== record.action_dependency_schema.transition_determinants.length) malformedHypotheses += 1;
    for (const [index, target] of record.target.determinant_grounding.entries()) {
      const determinant = record.action_dependency_schema.transition_determinants[index];
      const hypothesis = hypothesisById.get(target.determinant_id);
      if (!determinant || determinant.id !== target.determinant_id || !hypothesis || hypothesis.statements[0] === hypothesis.statements[1]) {
        malformedHypotheses += 1;
      }
      const span = target.evidence_span;
      if (
        span.start < 0 || span.end <= span.start || span.end > record.agent_input.observation.length ||
        record.agent_input.observation.slice(span.start, span.end) !== span.text ||
        !record.evidence_units.some((unit) => canonicalJson(unit) === canonicalJson(span))
      ) malformedSpans += 1;
      const expected: [V10Relation, V10Relation] = target.current_value === 'active'
        ? ['ENTAILED', 'CONTRADICTED']
        : target.current_value === 'inactive' ? ['CONTRADICTED', 'ENTAILED'] : ['UNKNOWN', 'UNKNOWN'];
      if (canonicalJson(target.hypothesis_relations) !== canonicalJson(expected)) relationMismatches += 1;
      if (target.temporal_status === 'CURRENT') {
        currentPairs += 1;
        const actual = currentMentionSignature(record, target);
        const required = record.semantic_operator_family === 'affirmative_gold'
          ? 'gold_only' : record.semantic_operator_family === 'negated_opposite' ? 'opposite_only' : 'both';
        if (actual !== required) signatureMismatches += 1;
        const values = signaturesBySurface.get(record.template_family) ?? new Set<MentionSignature>();
        values.add(actual);
        signaturesBySurface.set(record.template_family, values);
      } else {
        unresolvedPairs += 1;
        if (target.current_value !== null) relationMismatches += 1;
      }
      if (canonicalJson(target.allowed_values) !== canonicalJson(
        deriveV10AllowedValues(target.temporal_status, target.hypothesis_relations),
      )) derivationMismatches += 1;
    }
    const symbolic = evaluateAllowedTransitions({
      action_dependency_schema: record.action_dependency_schema,
      determinant_values: record.target.determinant_grounding,
    });
    if (
      symbolic.identifiable !== record.target.identifiable ||
      canonicalJson(symbolic.possible_transition_codes) !== canonicalJson(record.target.possible_transition_codes)
    ) symbolicMismatches += 1;
    const observation = record.agent_input.observation.toLowerCase();
    if (record.action_dependency_schema.transition_determinants.some((value) => observation.includes(value.id.toLowerCase()))) {
      determinantIds += 1;
    }
    if (/\b(?:active|inactive|entailed|contradicted|unknown|identifiable|ambiguous)\b/i.test(observation)) literalLabels += 1;
  }

  const unsupported = v14SurfaceFamilies.filter((heldout) => {
    const evaluation = signaturesBySurface.get(heldout) ?? new Set<MentionSignature>();
    const training = new Set(v14SurfaceFamilies.filter((value) => value !== heldout)
      .flatMap((value) => [...(signaturesBySurface.get(value) ?? [])]));
    return [...evaluation].some((value) => !training.has(value));
  });

  const contextGroups = groupBy(records, (record) => record.context_group);
  let malformedContexts = 0;
  for (const values of contextGroups.values()) {
    if (
      values.length !== 54 || new Set(values.map((value) => value.split)).size !== 1 ||
      new Set(values.map((value) => value.template_family)).size !== 9 ||
      new Set(values.map((value) => value.semantic_operator_family)).size !== 3 ||
      new Set(values.map((value) => value.state_lexicon_family)).size !== 3 ||
      new Set(values.map((value) => value.intervention_member)).size !== 2
    ) malformedContexts += 1;
  }
  if (contextGroups.size !== 90) errors.push(`V14 expected 90 contexts, found ${contextGroups.size}.`);

  const interventionGroups = groupBy(records, (record) => record.intervention_group_id);
  let malformedInterventions = 0;
  for (const values of interventionGroups.values()) {
    if (
      values.length !== 6 || new Set(values.map((value) => value.template_family)).size !== 1 ||
      new Set(values.map((value) => value.state_lexicon_family)).size !== 3 ||
      values.filter((value) => value.intervention_member === 'relevant_unresolved').length !== 3 ||
      values.filter((value) => value.intervention_member === 'relevant_resolved').length !== 3
    ) malformedInterventions += 1;
  }
  if (interventionGroups.size !== 810) errors.push(`V14 expected 810 intervention groups, found ${interventionGroups.size}.`);

  const balanceCells = groupBy(records.flatMap((record) => record.target.determinant_grounding
    .filter((target) => target.temporal_status === 'CURRENT')
    .map((target) => ({ record, value: target.current_value }))), ({ record }) => canonicalJson([
      record.split, record.mechanic, record.template_family, record.state_lexicon_family,
    ]));
  let imbalancedCells = 0;
  for (const values of balanceCells.values()) {
    const active = values.filter((value) => value.value === 'active').length;
    const inactive = values.filter((value) => value.value === 'inactive').length;
    if (active !== inactive || active === 0) imbalancedCells += 1;
  }

  const promptGroups = groupBy(records, (record) => canonicalJson(record.agent_input));
  let duplicatePrompts = 0;
  let conflictingPrompts = 0;
  let crossSplitPrompts = 0;
  for (const values of promptGroups.values()) {
    if (values.length <= 1) continue;
    duplicatePrompts += values.length - 1;
    if (new Set(values.map((value) => canonicalJson(value.target))).size > 1) conflictingPrompts += 1;
    if (new Set(values.map((value) => value.split)).size > 1) crossSplitPrompts += 1;
  }

  const failures: Array<[number | string[], string]> = [
    [malformedSpans, 'malformed spans'], [malformedHypotheses, 'malformed hypotheses'],
    [relationMismatches, 'relation mismatches'], [derivationMismatches, 'allowed-value derivation mismatches'],
    [symbolicMismatches, 'symbolic mismatches'], [malformedContexts, 'malformed context groups'],
    [malformedInterventions, 'malformed intervention groups'], [imbalancedCells, 'imbalanced current cells'],
    [signatureMismatches, 'operator-signature mismatches'], [unsupported, 'unsupported surface holdouts'],
    [duplicatePrompts, 'duplicate prompts'], [conflictingPrompts, 'conflicting prompts'],
    [crossSplitPrompts, 'cross-split prompts'], [determinantIds, 'determinant ids in observations'],
    [literalLabels, 'literal target labels in observations'],
  ];
  for (const [value, label] of failures) {
    const count = Array.isArray(value) ? value.length : value;
    if (count) errors.push(`V14 has ${count} ${label}.`);
  }

  const templates = Object.fromEntries(v14SurfaceFamilies.map((surface) => [
    surface, records.filter((record) => record.template_family === surface).length,
  ]));
  const semanticOperators = Object.fromEntries(Object.keys(v14SurfacesByOperator).map((operator) => [
    operator, records.filter((record) => record.semantic_operator_family === operator).length,
  ]));
  return {
    errors,
    records: recordCounts,
    contexts: contextCounts,
    templates,
    semantic_operators: semanticOperators,
    intervention_groups: interventionGroups.size,
    current_hypothesis_pairs: currentPairs,
    unresolved_hypothesis_pairs: unresolvedPairs,
    malformed_spans: malformedSpans,
    malformed_hypotheses: malformedHypotheses,
    relation_mismatches: relationMismatches,
    allowed_value_derivation_mismatches: derivationMismatches,
    symbolic_mismatches: symbolicMismatches,
    malformed_context_groups: malformedContexts,
    malformed_intervention_groups: malformedInterventions,
    imbalanced_current_cells: imbalancedCells,
    operator_signature_mismatches: signatureMismatches,
    unsupported_surface_holdouts: unsupported,
    complement_cross_split_overlaps: complementOverlap,
    context_cross_split_overlaps: contextOverlap,
    duplicate_ids: duplicateIds,
    duplicate_prompts: duplicatePrompts,
    cross_split_duplicate_prompts: crossSplitPrompts,
    conflicting_duplicate_prompts: conflictingPrompts,
    determinant_ids_in_observation: determinantIds,
    literal_target_labels_in_observation: literalLabels,
  };
}

function groupBy<T>(values: T[], key: (value: T) => string): Map<string, T[]> {
  const result = new Map<string, T[]>();
  for (const value of values) result.set(key(value), [...(result.get(key(value)) ?? []), value]);
  return result;
}
