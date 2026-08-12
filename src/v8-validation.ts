import type {
  V8Mechanic,
  V8StructuredRecord,
  V8SurfaceVariant,
} from './contracts';
import { canonicalJson } from './serialization';

export interface V8ValidationResult {
  errors: string[];
  records: Record<'train' | 'calibration', number>;
  base_records: Record<'train' | 'calibration', number>;
  context_groups: Record<'train' | 'calibration', number>;
  ambiguity_rates: Record<'train' | 'calibration', number>;
  mechanics: Record<V8Mechanic, {
    records: number;
    ambiguous: number;
    identifiable: number;
    intervention_groups: number;
    label_flip_groups: number;
    same_label_control_groups: number;
    unresolved_true_flip_groups: number;
    unresolved_false_flip_groups: number;
  }>;
  intervention_groups: number;
  label_flip_groups: number;
  same_label_control_groups: number;
  incomplete_surface_groups: number;
  malformed_intervention_groups: number;
  mismatched_pair_token_bags: number;
  mismatched_pair_prompt_lengths: number;
  context_cross_split_overlaps: number;
  duplicate_ids: number;
  duplicate_prompts: number;
  conflicting_duplicate_prompts: number;
  answer_bearing_inputs: number;
  tone_drift_records: number;
  v3_test_records_read: 0;
  prior_holdout_records_read: 0;
  v7_tone_drift_records_read: 0;
}

export function validateV8(options: {
  records: V8StructuredRecord[];
  mechanics: V8Mechanic[];
  surfaces: V8SurfaceVariant[];
  minimumLabelFlipGroupsPerMechanic: number;
}): V8ValidationResult {
  const { records, mechanics, surfaces } = options;
  const errors: string[] = [];
  const splits = ['train', 'calibration'] as const;
  const counts = Object.fromEntries(splits.map((split) => [
    split,
    records.filter((record) => record.split === split).length,
  ])) as V8ValidationResult['records'];
  const baseCounts = Object.fromEntries(splits.map((split) => [
    split,
    new Set(records.filter((record) => record.split === split).map((record) => record.surface_group_id)).size,
  ])) as V8ValidationResult['base_records'];
  const contexts = Object.fromEntries(splits.map((split) => [
    split,
    new Set(records.filter((record) => record.split === split).map((record) => record.split_group)).size,
  ])) as V8ValidationResult['context_groups'];
  const rates = Object.fromEntries(splits.map((split) => {
    const values = records.filter((record) => record.split === split);
    return [split, values.filter((record) => record.target.ambiguous).length / values.length];
  })) as V8ValidationResult['ambiguity_rates'];

  const ids = countDuplicates(records.map((record) => record.id));
  if (ids > 0) errors.push(`V8 contains ${ids} duplicate record ids.`);
  if (records.some((record) => record.schema_version !== 8)) errors.push('V8 contains a non-schema-8 record.');
  if (new Set(records.map((record) => record.mechanic)).size !== mechanics.length) {
    errors.push('V8 does not contain every configured mechanic.');
  }
  if (records.some((record) => !mechanics.includes(record.mechanic))) {
    errors.push('V8 contains an unconfigured mechanic.');
  }
  if (counts.train === 0 || counts.calibration === 0) errors.push('V8 requires both train and calibration records.');

  const surfaceGroups = groupBy(records, (record) => record.surface_group_id);
  let incompleteSurfaces = 0;
  for (const values of surfaceGroups.values()) {
    if (
      values.length !== surfaces.length ||
      canonicalStrings(values.map((record) => record.surface_variant)) !== canonicalStrings(surfaces) ||
      new Set(values.map((record) => canonicalJson(record.target))).size !== 1 ||
      new Set(values.map((record) => record.split)).size !== 1
    ) incompleteSurfaces += 1;
  }
  if (incompleteSurfaces > 0) errors.push(`V8 has ${incompleteSurfaces} incomplete or inconsistent surface groups.`);

  const interventionGroups = groupBy(records, (record) => record.intervention_group_id);
  let malformedGroups = 0;
  let tokenBagMismatches = 0;
  let promptLengthMismatches = 0;
  let labelFlipGroups = 0;
  let controlGroups = 0;
  for (const values of interventionGroups.values()) {
    const canonical = values.filter((record) => record.surface_variant === 'canonical');
    const unresolved = canonical.find((record) => record.intervention_member === 'relevant_unresolved');
    const resolved = canonical.find((record) => record.intervention_member === 'relevant_resolved');
    const kinds = new Set(values.map((record) => record.intervention_kind));
    const consistent =
      values.length === surfaces.length * 2 &&
      canonical.length === 2 &&
      Boolean(unresolved && resolved) &&
      kinds.size === 1 &&
      new Set(values.map((record) => record.split)).size === 1 &&
      new Set(values.map((record) => record.split_group)).size === 1 &&
      new Set(values.map((record) => record.mechanic)).size === 1 &&
      new Set(values.map((record) => record.primary_determinant_id)).size === 1;
    if (!consistent || !unresolved || !resolved) {
      malformedGroups += 1;
      continue;
    }
    const isFlip = unresolved.target.ambiguous && !resolved.target.ambiguous;
    const isControl = !unresolved.target.ambiguous && !resolved.target.ambiguous;
    const declared = values[0].intervention_kind;
    if ((declared === 'oracle_label_flip' && !isFlip) ||
        (declared === 'same_label_causal_control' && !isControl)) {
      malformedGroups += 1;
    }
    if (isFlip) labelFlipGroups += 1;
    if (isControl) controlGroups += 1;
    for (const surface of surfaces) {
      const pair = values.filter((record) => record.surface_variant === surface);
      if (pair.length !== 2) continue;
      const bags = pair.map((record) => inputTokenBag(record));
      if (bags[0] !== bags[1]) tokenBagMismatches += 1;
      const lengths = pair.map((record) => canonicalJson(record.agent_input).length);
      if (lengths[0] !== lengths[1]) promptLengthMismatches += 1;
    }
  }
  if (malformedGroups > 0) errors.push(`V8 has ${malformedGroups} malformed intervention groups.`);
  if (tokenBagMismatches > 0) errors.push(`V8 has ${tokenBagMismatches} matched pairs with unequal input token bags.`);
  if (promptLengthMismatches > 0) errors.push(`V8 has ${promptLengthMismatches} matched pairs with unequal serialized lengths.`);

  const mechanicMetrics = Object.fromEntries(mechanics.map((mechanic) => {
    const values = records.filter((record) => record.mechanic === mechanic && record.surface_variant === 'canonical');
    const groups = groupBy(values, (record) => record.intervention_group_id);
    const groupValues = [...groups.values()];
    const flips = groupValues.filter((group) => group.some((record) => record.intervention_kind === 'oracle_label_flip'));
    const controls = groupValues.filter((group) => group.some((record) => record.intervention_kind === 'same_label_causal_control'));
    return [mechanic, {
      records: values.length,
      ambiguous: values.filter((record) => record.target.ambiguous).length,
      identifiable: values.filter((record) => !record.target.ambiguous).length,
      intervention_groups: groups.size,
      label_flip_groups: flips.length,
      same_label_control_groups: controls.length,
      unresolved_true_flip_groups: flips.filter((group) => group[0].primary_resolved_value).length,
      unresolved_false_flip_groups: flips.filter((group) => !group[0].primary_resolved_value).length,
    }];
  })) as V8ValidationResult['mechanics'];
  for (const mechanic of mechanics) {
    const metric = mechanicMetrics[mechanic];
    if (metric.ambiguous === 0 || metric.identifiable === 0) {
      errors.push(`V8 mechanic ${mechanic} does not contain both labels.`);
    }
    if (metric.label_flip_groups < options.minimumLabelFlipGroupsPerMechanic) {
      errors.push(`V8 mechanic ${mechanic} has only ${metric.label_flip_groups} label-flipping groups.`);
    }
    if (metric.unresolved_true_flip_groups === 0 || metric.unresolved_false_flip_groups === 0) {
      errors.push(`V8 mechanic ${mechanic} lacks true/false resolved label-flipping support.`);
    }
  }
  if (controlGroups === 0) errors.push('V8 contains no same-label causal controls.');

  const contextsBySplit = Object.fromEntries(splits.map((split) => [
    split,
    new Set(records.filter((record) => record.split === split).map((record) => record.split_group)),
  ])) as Record<typeof splits[number], Set<string>>;
  const contextOverlap = intersectionSize(contextsBySplit.train, contextsBySplit.calibration);
  if (contextOverlap > 0) errors.push(`V8 has ${contextOverlap} context groups crossing train/calibration.`);

  const promptGroups = groupBy(records, (record) => canonicalJson(record.agent_input));
  let duplicatePrompts = 0;
  let conflictingPrompts = 0;
  for (const values of promptGroups.values()) {
    if (values.length <= 1) continue;
    duplicatePrompts += values.length - 1;
    if (new Set(values.map((record) => canonicalJson(record.target))).size > 1) conflictingPrompts += 1;
  }
  if (conflictingPrompts > 0) errors.push(`V8 has ${conflictingPrompts} prompts with conflicting targets.`);

  const forbiddenPattern = /\b(?:ambiguous|identifiable|resolved_true|resolved_false|outcome_sensitive|outcome_invariant)\b/i;
  const answerBearing = records.filter((record) => forbiddenPattern.test(canonicalJson(record.agent_input))).length;
  if (answerBearing > 0) errors.push(`V8 has ${answerBearing} answer-bearing model inputs.`);
  const toneDrift = records.filter((record) =>
    record.source_scenario_id.includes('tonedrift') || canonicalJson(record.agent_input).includes('tone drift'),
  ).length;
  if (toneDrift > 0) errors.push(`V8 reads ${toneDrift} tone-drift records or inputs.`);

  return {
    errors,
    records: counts,
    base_records: baseCounts,
    context_groups: contexts,
    ambiguity_rates: rates,
    mechanics: mechanicMetrics,
    intervention_groups: interventionGroups.size,
    label_flip_groups: labelFlipGroups,
    same_label_control_groups: controlGroups,
    incomplete_surface_groups: incompleteSurfaces,
    malformed_intervention_groups: malformedGroups,
    mismatched_pair_token_bags: tokenBagMismatches,
    mismatched_pair_prompt_lengths: promptLengthMismatches,
    context_cross_split_overlaps: contextOverlap,
    duplicate_ids: ids,
    duplicate_prompts: duplicatePrompts,
    conflicting_duplicate_prompts: conflictingPrompts,
    answer_bearing_inputs: answerBearing,
    tone_drift_records: toneDrift,
    v3_test_records_read: 0,
    prior_holdout_records_read: 0,
    v7_tone_drift_records_read: 0,
  };
}

function inputTokenBag(record: V8StructuredRecord): string {
  return canonicalJson(canonicalJson(record.agent_input).toLowerCase().match(/[a-z0-9_:@.-]+/g)?.sort() ?? []);
}

function countDuplicates(values: string[]): number {
  return values.length - new Set(values).size;
}

function canonicalStrings(values: string[]): string {
  return [...values].sort().join(',');
}

function groupBy<T>(values: T[], key: (value: T) => string): Map<string, T[]> {
  const result = new Map<string, T[]>();
  for (const value of values) {
    const group = key(value);
    result.set(group, [...(result.get(group) ?? []), value]);
  }
  return result;
}

function intersectionSize<T>(left: Set<T>, right: Set<T>): number {
  return [...left].filter((value) => right.has(value)).length;
}
