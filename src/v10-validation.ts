import type { V10GroundingRecord, V10Relation, V10TemplateFamily } from './v10-contracts';
import { canonicalJson } from './serialization';
import { deriveV10AllowedValues } from './v10-grounding';
import { evaluateAllowedTransitions } from './v9-symbolic';

export interface V10ValidationResult {
  errors: string[];
  records: Record<'train' | 'evaluation', number>;
  contexts: Record<'train' | 'evaluation', number>;
  mechanics: Record<string, number>;
  templates: Record<V10TemplateFamily, number>;
  lexicons: Record<string, number>;
  operators: Record<string, { records: number; mechanics: string[] }>;
  intervention_groups: number;
  label_flip_groups: number;
  control_groups: number;
  current_hypothesis_pairs: number;
  unresolved_hypothesis_pairs: number;
  malformed_context_groups: number;
  malformed_intervention_groups: number;
  malformed_spans: number;
  malformed_hypotheses: number;
  relation_mismatches: number;
  allowed_value_derivation_mismatches: number;
  symbolic_mismatches: number;
  imbalanced_current_cells: number;
  complement_cross_split_overlaps: number;
  context_cross_split_overlaps: number;
  duplicate_ids: number;
  duplicate_prompts: number;
  cross_split_duplicate_prompts: number;
  conflicting_duplicate_prompts: number;
  determinant_ids_in_observation: number;
  literal_target_labels_in_observation: number;
}

export function validateV10Grounding(records: V10GroundingRecord[]): V10ValidationResult {
  const errors: string[] = [];
  const splits = ['train', 'evaluation'] as const;
  const recordCounts = Object.fromEntries(splits.map((split) => [
    split,
    records.filter((record) => record.split === split).length,
  ])) as V10ValidationResult['records'];
  const contextCounts = Object.fromEntries(splits.map((split) => [
    split,
    new Set(records.filter((record) => record.split === split).map((record) => record.context_group)).size,
  ])) as V10ValidationResult['contexts'];
  if (records.length !== 3240) errors.push(`V10 expected 3,240 records, found ${records.length}.`);
  if (recordCounts.train === 0 || recordCounts.evaluation === 0) errors.push('V10 requires train and evaluation records.');
  if (records.some((record) => record.schema_version !== 10)) errors.push('V10 contains a non-schema-10 record.');

  const duplicateIds = records.length - new Set(records.map((record) => record.id)).size;
  if (duplicateIds > 0) errors.push(`V10 has ${duplicateIds} duplicate ids.`);
  const contextsBySplit = Object.fromEntries(splits.map((split) => [
    split,
    new Set(records.filter((record) => record.split === split).map((record) => record.context_group)),
  ])) as Record<typeof splits[number], Set<string>>;
  const complementsBySplit = Object.fromEntries(splits.map((split) => [
    split,
    new Set(records.filter((record) => record.split === split).map((record) => record.complement_group)),
  ])) as Record<typeof splits[number], Set<string>>;
  const contextOverlap = [...contextsBySplit.train].filter((value) => contextsBySplit.evaluation.has(value)).length;
  const complementOverlap = [...complementsBySplit.train].filter((value) => complementsBySplit.evaluation.has(value)).length;
  if (contextOverlap > 0) errors.push(`V10 has ${contextOverlap} contexts crossing splits.`);
  if (complementOverlap > 0) errors.push(`V10 has ${complementOverlap} complement groups crossing splits.`);

  let malformedSpans = 0;
  let malformedHypotheses = 0;
  let relationMismatches = 0;
  let derivationMismatches = 0;
  let symbolicMismatches = 0;
  let determinantIds = 0;
  let literalLabels = 0;
  let currentPairs = 0;
  let unresolvedPairs = 0;
  for (const record of records) {
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
      if (
        !determinant || determinant.id !== target.determinant_id || !hypothesis ||
        hypothesis.statements.length !== 2 || hypothesis.statements[0] === hypothesis.statements[1]
      ) malformedHypotheses += 1;
      const span = target.evidence_span;
      if (
        span.start < 0 || span.end <= span.start || span.end > record.agent_input.observation.length ||
        record.agent_input.observation.slice(span.start, span.end) !== span.text ||
        !record.evidence_units.some((unit) => canonicalJson(unit) === canonicalJson(span))
      ) malformedSpans += 1;
      const expectedRelations: [V10Relation, V10Relation] = target.current_value === 'active'
        ? ['ENTAILED', 'CONTRADICTED']
        : target.current_value === 'inactive'
          ? ['CONTRADICTED', 'ENTAILED']
          : ['UNKNOWN', 'UNKNOWN'];
      if (canonicalJson(target.hypothesis_relations) !== canonicalJson(expectedRelations)) relationMismatches += 1;
      if (target.temporal_status === 'CURRENT') {
        currentPairs += 1;
        if (target.current_value === null) relationMismatches += 1;
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
    if (record.action_dependency_schema.transition_determinants.some((value) =>
      observation.includes(value.id.toLowerCase()),
    )) determinantIds += 1;
    if (/\b(?:active|inactive|entailed|contradicted|unknown|identifiable|ambiguous)\b/i.test(observation)) literalLabels += 1;
  }
  if (malformedSpans > 0) errors.push(`V10 has ${malformedSpans} malformed spans.`);
  if (malformedHypotheses > 0) errors.push(`V10 has ${malformedHypotheses} malformed hypothesis sets.`);
  if (relationMismatches > 0) errors.push(`V10 has ${relationMismatches} relation target mismatches.`);
  if (derivationMismatches > 0) errors.push(`V10 has ${derivationMismatches} allowed-value derivation mismatches.`);
  if (symbolicMismatches > 0) errors.push(`V10 has ${symbolicMismatches} symbolic target mismatches.`);
  if (determinantIds > 0) errors.push(`V10 exposes determinant ids in ${determinantIds} observations.`);
  if (literalLabels > 0) errors.push(`V10 exposes literal target labels in ${literalLabels} observations.`);

  const contextGroups = groupBy(records, (record) => record.context_group);
  let malformedContexts = 0;
  for (const values of contextGroups.values()) {
    if (
      values.length !== 36 || new Set(values.map((value) => value.split)).size !== 1 ||
      new Set(values.map((value) => value.mechanic)).size !== 1 ||
      new Set(values.map((value) => value.template_family)).size !== 6 ||
      new Set(values.map((value) => value.state_lexicon_family)).size !== 3 ||
      new Set(values.map((value) => value.intervention_member)).size !== 2
    ) malformedContexts += 1;
  }
  if (contextGroups.size !== 90) errors.push(`V10 expected 90 contexts, found ${contextGroups.size}.`);
  if (malformedContexts > 0) errors.push(`V10 has ${malformedContexts} malformed context groups.`);

  const interventionGroups = groupBy(records, (record) => record.intervention_group_id);
  let malformedInterventions = 0;
  let flipGroups = 0;
  let controlGroups = 0;
  for (const values of interventionGroups.values()) {
    const unresolved = values.filter((value) => value.intervention_member === 'relevant_unresolved');
    const resolved = values.filter((value) => value.intervention_member === 'relevant_resolved');
    const consistent = values.length === 6 && unresolved.length === 3 && resolved.length === 3 &&
      new Set(values.map((value) => value.state_lexicon_family)).size === 3 &&
      new Set(values.map((value) => value.template_family)).size === 1 &&
      new Set(values.map((value) => value.context_group)).size === 1;
    const flip = unresolved.every((value) => !value.target.identifiable) &&
      resolved.every((value) => value.target.identifiable);
    const control = unresolved.every((value) => value.target.identifiable) &&
      resolved.every((value) => value.target.identifiable);
    if (!consistent || (!flip && !control)) malformedInterventions += 1;
    if (flip) flipGroups += 1;
    if (control) controlGroups += 1;
  }
  if (interventionGroups.size !== 540) errors.push(`V10 expected 540 intervention groups, found ${interventionGroups.size}.`);
  if (malformedInterventions > 0) errors.push(`V10 has ${malformedInterventions} malformed intervention groups.`);

  const balanceCells = groupBy(
    records.flatMap((record) => record.target.determinant_grounding
      .filter((target) => target.temporal_status === 'CURRENT')
      .map((target) => ({ record, value: target.current_value }))),
    ({ record }) => canonicalJson([record.split, record.mechanic, record.template_family, record.state_lexicon_family]),
  );
  let imbalancedCells = 0;
  for (const values of balanceCells.values()) {
    const active = values.filter((value) => value.value === 'active').length;
    const inactive = values.filter((value) => value.value === 'inactive').length;
    if (active !== inactive || active === 0) imbalancedCells += 1;
  }
  if (imbalancedCells > 0) errors.push(`V10 has ${imbalancedCells} imbalanced current-state cells.`);

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
  if (duplicatePrompts > 0) errors.push(`V10 has ${duplicatePrompts} duplicate prompts.`);
  if (conflictingPrompts > 0) errors.push(`V10 has ${conflictingPrompts} conflicting duplicate prompts.`);
  if (crossSplitPrompts > 0) errors.push(`V10 has ${crossSplitPrompts} cross-split duplicate prompts.`);

  const mechanics = Object.fromEntries([...new Set(records.map((record) => record.mechanic))].sort().map((value) => [
    value,
    records.filter((record) => record.mechanic === value).length,
  ]));
  const templates = Object.fromEntries([...new Set(records.map((record) => record.template_family))].sort().map((value) => [
    value,
    records.filter((record) => record.template_family === value).length,
  ])) as V10ValidationResult['templates'];
  const lexicons = Object.fromEntries([...new Set(records.map((record) => record.state_lexicon_family))].sort().map((value) => [
    value,
    records.filter((record) => record.state_lexicon_family === value).length,
  ]));
  const operators = Object.fromEntries([...new Set(records.map((record) => record.operator_family))].sort().map((value) => [
    value,
    {
      records: records.filter((record) => record.operator_family === value).length,
      mechanics: [...new Set(records.filter((record) => record.operator_family === value).map((record) => record.mechanic))].sort(),
    },
  ]));
  return {
    errors,
    records: recordCounts,
    contexts: contextCounts,
    mechanics,
    templates,
    lexicons,
    operators,
    intervention_groups: interventionGroups.size,
    label_flip_groups: flipGroups,
    control_groups: controlGroups,
    current_hypothesis_pairs: currentPairs,
    unresolved_hypothesis_pairs: unresolvedPairs,
    malformed_context_groups: malformedContexts,
    malformed_intervention_groups: malformedInterventions,
    malformed_spans: malformedSpans,
    malformed_hypotheses: malformedHypotheses,
    relation_mismatches: relationMismatches,
    allowed_value_derivation_mismatches: derivationMismatches,
    symbolic_mismatches: symbolicMismatches,
    imbalanced_current_cells: imbalancedCells,
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
  for (const value of values) {
    const name = key(value);
    result.set(name, [...(result.get(name) ?? []), value]);
  }
  return result;
}
