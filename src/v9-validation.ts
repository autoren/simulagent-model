import type { V9GroundingRecord, V9OperatorFamily, V9TemplateFamily, V9TemporalStatus } from './v9-contracts';
import { canonicalJson } from './serialization';
import { evaluateAllowedTransitions } from './v9-symbolic';

export interface V9ValidationResult {
  errors: string[];
  records: Record<'train' | 'calibration', number>;
  contexts: Record<'train' | 'calibration', number>;
  mechanics: Record<string, { train: number; calibration: number }>;
  templates: Record<V9TemplateFamily, { train: number; calibration: number }>;
  operators: Record<V9OperatorFamily, { train: number; calibration: number; mechanics: string[] }>;
  temporal_statuses: Record<V9TemporalStatus, number>;
  allowed_value_sets: Record<string, number>;
  intervention_groups: number;
  label_flip_groups: number;
  control_groups: number;
  malformed_intervention_groups: number;
  malformed_context_groups: number;
  malformed_spans: number;
  symbolic_mismatches: number;
  context_cross_split_overlaps: number;
  duplicate_ids: number;
  duplicate_prompts: number;
  conflicting_duplicate_prompts: number;
  cross_split_duplicate_prompts: number;
  determinant_ids_in_observation: number;
  literal_value_labels_in_observation: number;
}

export function validateV9Grounding(records: V9GroundingRecord[]): V9ValidationResult {
  const errors: string[] = [];
  const splits = ['train', 'calibration'] as const;
  const recordCounts = Object.fromEntries(splits.map((split) => [
    split,
    records.filter((record) => record.split === split).length,
  ])) as V9ValidationResult['records'];
  const contextCounts = Object.fromEntries(splits.map((split) => [
    split,
    new Set(records.filter((record) => record.split === split).map((record) => record.context_group)).size,
  ])) as V9ValidationResult['contexts'];
  if (records.length !== 2160) errors.push(`V9 expected 2,160 records, found ${records.length}.`);
  if (recordCounts.train === 0 || recordCounts.calibration === 0) errors.push('V9 needs both splits.');

  const duplicateIds = records.length - new Set(records.map((record) => record.id)).size;
  if (duplicateIds > 0) errors.push(`V9 has ${duplicateIds} duplicate ids.`);
  if (records.some((record) => record.schema_version !== 9)) errors.push('V9 has a non-schema-9 record.');
  const contextsBySplit = Object.fromEntries(splits.map((split) => [
    split,
    new Set(records.filter((record) => record.split === split).map((record) => record.context_group)),
  ])) as Record<typeof splits[number], Set<string>>;
  const contextOverlap = [...contextsBySplit.train].filter((value) => contextsBySplit.calibration.has(value)).length;
  if (contextOverlap > 0) errors.push(`V9 has ${contextOverlap} contexts crossing splits.`);

  let malformedSpans = 0;
  let symbolicMismatches = 0;
  let determinantIdsInObservation = 0;
  let literalValueLabels = 0;
  for (const record of records) {
    if (record.evidence_units.length !== 7) malformedSpans += 1;
    for (const unit of record.evidence_units) {
      if (
        unit.start < 0 || unit.end <= unit.start || unit.end > record.agent_input.observation.length ||
        record.agent_input.observation.slice(unit.start, unit.end) !== unit.text
      ) malformedSpans += 1;
    }
    for (const target of record.target.determinant_grounding) {
      const span = target.evidence_span;
      if (
        span.start < 0 || span.end <= span.start || span.end > record.agent_input.observation.length ||
        record.agent_input.observation.slice(span.start, span.end) !== span.text ||
        !record.evidence_units.some((unit) => canonicalJson(unit) === canonicalJson(span))
      ) malformedSpans += 1;
    }
    const result = evaluateAllowedTransitions({
      action_dependency_schema: record.action_dependency_schema,
      determinant_values: record.target.determinant_grounding,
    });
    if (
      result.identifiable !== record.target.identifiable ||
      canonicalJson(result.possible_transition_codes) !== canonicalJson(record.target.possible_transition_codes)
    ) symbolicMismatches += 1;
    const observation = record.agent_input.observation.toLowerCase();
    if (record.action_dependency_schema.transition_determinants.some((value) =>
      observation.includes(value.id.toLowerCase()),
    )) determinantIdsInObservation += 1;
    if (/\b(?:active|inactive|resolved|unresolved|identifiable|ambiguous)\b/i.test(observation)) {
      literalValueLabels += 1;
    }
  }
  if (malformedSpans > 0) errors.push(`V9 has ${malformedSpans} malformed evidence spans.`);
  if (symbolicMismatches > 0) errors.push(`V9 has ${symbolicMismatches} symbolic target mismatches.`);
  if (determinantIdsInObservation > 0) errors.push(`V9 exposes determinant ids in ${determinantIdsInObservation} observations.`);
  if (literalValueLabels > 0) errors.push(`V9 exposes literal target labels in ${literalValueLabels} observations.`);

  const contextGroups = groupBy(records, (record) => record.context_group);
  let malformedContexts = 0;
  for (const values of contextGroups.values()) {
    if (
      values.length !== 24 ||
      new Set(values.map((value) => value.split)).size !== 1 ||
      new Set(values.map((value) => value.mechanic)).size !== 1 ||
      new Set(values.map((value) => value.template_family)).size !== 4 ||
      new Set(values.map((value) => value.surface_variant)).size !== 3 ||
      new Set(values.map((value) => value.intervention_member)).size !== 2
    ) malformedContexts += 1;
  }
  if (malformedContexts > 0) errors.push(`V9 has ${malformedContexts} malformed semantic contexts.`);

  const interventionGroups = groupBy(records, (record) => record.intervention_group_id);
  let malformedInterventions = 0;
  let flipGroups = 0;
  let controlGroups = 0;
  for (const values of interventionGroups.values()) {
    const unresolved = values.filter((value) => value.intervention_member === 'relevant_unresolved');
    const resolved = values.filter((value) => value.intervention_member === 'relevant_resolved');
    const consistent = values.length === 6 && unresolved.length === 3 && resolved.length === 3 &&
      new Set(values.map((value) => value.surface_variant)).size === 3 &&
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
  if (malformedInterventions > 0) errors.push(`V9 has ${malformedInterventions} malformed intervention groups.`);
  if (flipGroups === 0 || controlGroups === 0) errors.push('V9 requires flip and control groups.');

  const promptGroups = groupBy(records, (record) => canonicalJson(record.agent_input));
  let duplicatePrompts = 0;
  let conflictingPrompts = 0;
  let crossSplitDuplicates = 0;
  for (const values of promptGroups.values()) {
    if (values.length <= 1) continue;
    duplicatePrompts += values.length - 1;
    if (new Set(values.map((record) => canonicalJson(record.target))).size > 1) conflictingPrompts += 1;
    if (new Set(values.map((record) => record.split)).size > 1) crossSplitDuplicates += 1;
  }
  if (conflictingPrompts > 0) errors.push(`V9 has ${conflictingPrompts} duplicate prompts with conflicting targets.`);
  if (crossSplitDuplicates > 0) errors.push(`V9 has ${crossSplitDuplicates} prompts crossing splits.`);

  const mechanics = Object.fromEntries([...new Set(records.map((record) => record.mechanic))].sort().map((mechanic) => [
    mechanic,
    Object.fromEntries(splits.map((split) => [split, records.filter((record) => record.mechanic === mechanic && record.split === split).length])),
  ])) as V9ValidationResult['mechanics'];
  const templateNames = [...new Set(records.map((record) => record.template_family))].sort() as V9TemplateFamily[];
  const templates = Object.fromEntries(templateNames.map((template) => [
    template,
    Object.fromEntries(splits.map((split) => [split, records.filter((record) => record.template_family === template && record.split === split).length])),
  ])) as V9ValidationResult['templates'];
  const operatorNames = [...new Set(records.map((record) => record.operator_family))].sort() as V9OperatorFamily[];
  const operators = Object.fromEntries(operatorNames.map((operator) => [
    operator,
    {
      ...Object.fromEntries(splits.map((split) => [split, records.filter((record) => record.operator_family === operator && record.split === split).length])),
      mechanics: [...new Set(records.filter((record) => record.operator_family === operator).map((record) => record.mechanic))].sort(),
    },
  ])) as V9ValidationResult['operators'];
  for (const [name, counts] of Object.entries({ ...mechanics, ...templates, ...operators })) {
    if (counts.train === 0 || counts.calibration === 0) errors.push(`V9 stratum ${name} lacks both splits.`);
  }

  const temporalStatuses = countBy(records.flatMap((record) =>
    record.target.determinant_grounding.map((value) => value.temporal_status),
  )) as V9ValidationResult['temporal_statuses'];
  const allowedValueSets = countBy(records.flatMap((record) =>
    record.target.determinant_grounding.map((value) => [...value.allowed_values].sort().join('|')),
  ));
  return {
    errors,
    records: recordCounts,
    contexts: contextCounts,
    mechanics,
    templates,
    operators,
    temporal_statuses: temporalStatuses,
    allowed_value_sets: allowedValueSets,
    intervention_groups: interventionGroups.size,
    label_flip_groups: flipGroups,
    control_groups: controlGroups,
    malformed_intervention_groups: malformedInterventions,
    malformed_context_groups: malformedContexts,
    malformed_spans: malformedSpans,
    symbolic_mismatches: symbolicMismatches,
    context_cross_split_overlaps: contextOverlap,
    duplicate_ids: duplicateIds,
    duplicate_prompts: duplicatePrompts,
    conflicting_duplicate_prompts: conflictingPrompts,
    cross_split_duplicate_prompts: crossSplitDuplicates,
    determinant_ids_in_observation: determinantIdsInObservation,
    literal_value_labels_in_observation: literalValueLabels,
  };
}

function groupBy<T>(values: T[], key: (value: T) => string): Map<string, T[]> {
  const result = new Map<string, T[]>();
  for (const value of values) result.set(key(value), [...(result.get(key(value)) ?? []), value]);
  return result;
}

function countBy(values: string[]): Record<string, number> {
  const result: Record<string, number> = {};
  for (const value of values) result[value] = (result[value] ?? 0) + 1;
  return Object.fromEntries(Object.entries(result).sort(([left], [right]) => left.localeCompare(right)));
}
