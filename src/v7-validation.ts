import type {
  V5EvidenceVariant,
  V5SurfaceVariant,
  V7DevelopmentMechanic,
  V7IdentifiabilityRecord,
  V7Mechanic,
  V7Split,
} from './contracts';
import { canonicalJson } from './serialization';

const splits: V7Split[] = ['train', 'calibration', 'untouched_mechanic'];
const forbiddenKeys = new Set([
  'possible_outcomes',
  'outcome_count',
  'identifiable',
  'empirical_support',
  'oracle_trace',
  'privileged_input',
  'privileged_world_state',
  'transition_rules',
  'target_sha256',
  'source_record_count',
]);

export interface ConditionalGapReport {
  maximum: number;
  full_cell_maximum_distance_from_half: number;
  by_dimension: Record<'evidence_variant' | 'mechanic' | 'action_template' | 'surface_variant', number>;
  full_cells: number;
  minimum_full_cell_support: number;
}

export interface V7ValidationReport {
  errors: string[];
  records: Record<V7Split, number>;
  base_records: Record<V7Split, number>;
  context_groups: Record<V7Split, number>;
  ambiguous_rates: Record<V7Split, number>;
  counts_by_surface: Record<V7Split, Record<V5SurfaceVariant, number>>;
  counts_by_mechanic: Record<V7Split, Partial<Record<V7Mechanic, number>>>;
  counts_by_evidence: Record<V7Split, Partial<Record<V5EvidenceVariant, number>>>;
  label_changing_evidence_groups: Record<V7Split, number>;
  conditional_label_gaps: Record<'train' | 'calibration', ConditionalGapReport>;
  maximum_conditional_label_gap: number;
  context_cross_split_overlaps: number;
  prompt_cross_split_overlaps: number;
  evidence_group_cross_split_overlaps: number;
  source_scenario_development_holdout_overlaps: number;
  duplicate_ids: number;
  duplicate_prompts: number;
  incomplete_surface_groups: number;
  forbidden_training_fields: number;
  v3_test_records_read: 0;
  prior_holdout_records_read: 0;
}

export function validateV7(options: {
  records: V7IdentifiabilityRecord[];
  requiredSurfaces: V5SurfaceVariant[];
  requiredEvidenceVariants: Exclude<V5EvidenceVariant, 'mixed'>[];
  developmentMechanics: V7DevelopmentMechanic[];
  holdoutMechanic: 'tonedrift';
  developmentSeeds: number[];
  holdoutSeeds: number[];
  minimumLabelChangingDevelopmentGroups: number;
  maximumConditionalLabelGap: number;
}): V7ValidationReport {
  const { records } = options;
  const errors: string[] = [];
  const duplicateIds = duplicateCount(records.map((record) => record.id));
  const duplicatePrompts = duplicateCount(records.map((record) => canonicalJson(record.agent_input)));
  const contextOverlaps = crossSplitCount(records, (record) => record.base_context_group);
  const promptOverlaps = crossSplitCount(records, (record) => canonicalJson(record.agent_input));
  const evidenceOverlaps = crossSplitCount(records, (record) => record.evidence_intervention_id);
  const forbiddenFields = records.reduce((sum, record) => sum + countForbidden(record), 0);
  const developmentScenarios = new Set(
    records.filter((record) => record.split !== 'untouched_mechanic').flatMap((record) => record.source_scenario_ids),
  );
  const holdoutScenarios = new Set(
    records.filter((record) => record.split === 'untouched_mechanic').flatMap((record) => record.source_scenario_ids),
  );
  const scenarioOverlaps = [...holdoutScenarios].filter((id) => developmentScenarios.has(id)).length;

  if (records.length === 0) errors.push('V7 corpus is empty.');
  if (duplicateIds > 0) errors.push('V7 contains duplicate ids.');
  if (duplicatePrompts > 0) errors.push('V7 contains duplicate model inputs.');
  if (contextOverlaps > 0) errors.push('V7 base contexts cross splits.');
  if (promptOverlaps > 0) errors.push('V7 model inputs cross splits.');
  if (evidenceOverlaps > 0) errors.push('V7 evidence groups cross splits.');
  if (scenarioOverlaps > 0) errors.push('V7 development scenarios leak into the untouched mechanic.');
  if (forbiddenFields > 0) errors.push('V7 records expose non-binary or privileged targets.');

  const requiredTargetKeys = canonicalJson(['ambiguous', 'invariance']);
  for (const record of records) {
    if (record.schema_version !== 7) errors.push(`${record.id} has a non-V7 schema.`);
    if (canonicalJson(Object.keys(record.target).sort()) !== requiredTargetKeys) {
      errors.push(`${record.id} has an invalid binary-only target.`);
    }
    const permittedSeeds = new Set(
      record.split === 'untouched_mechanic' ? options.holdoutSeeds : options.developmentSeeds,
    );
    if (record.scenario_seeds.length === 0 || record.scenario_seeds.some((seed) => !permittedSeeds.has(seed))) {
      errors.push(`${record.id} contains a seed outside its frozen partition.`);
    }
    if (record.split === 'untouched_mechanic' && record.mechanic !== options.holdoutMechanic) {
      errors.push(`${record.id} leaks a development mechanic into the untouched partition.`);
    }
    if (
      record.split !== 'untouched_mechanic' &&
      !options.developmentMechanics.includes(record.mechanic as V7DevelopmentMechanic)
    ) {
      errors.push(`${record.id} leaks the untouched mechanic into development.`);
    }
    if (record.action_template.length === 0) errors.push(`${record.id} has an empty action template.`);
  }

  const surfaceGroups = groupBy(records, (record) => record.surface_pair_id);
  let incompleteSurfaceGroups = 0;
  for (const [pairId, values] of surfaceGroups) {
    const observed = [...new Set(values.map((record) => record.surface_variant))].sort();
    const expected = [...options.requiredSurfaces].sort();
    const invalid =
      canonicalJson(observed) !== canonicalJson(expected) ||
      values.length !== expected.length ||
      new Set(values.map((record) => record.target.ambiguous)).size !== 1 ||
      new Set(values.map((record) => record.split)).size !== 1 ||
      new Set(values.map((record) => record.invariance_group_id)).size !== 1;
    if (invalid) {
      incompleteSurfaceGroups += 1;
      errors.push(`${pairId} is not a complete, same-label surface invariance group.`);
    }
    const canonical = values.find((record) => record.surface_variant === 'canonical');
    if (
      canonical &&
      values.some((record) =>
        record.surface_variant !== 'canonical' &&
        canonicalJson(record.agent_input) === canonicalJson(canonical.agent_input),
      )
    ) {
      errors.push(`${pairId} contains an unchanged transformed surface.`);
    }
  }

  const canonicalRecords = records.filter((record) => record.surface_variant === 'canonical');
  const labelChanging = mapSplits((split) => {
    const groups = groupBy(
      canonicalRecords.filter((record) =>
        record.split === split && record.evidence_intervention_kind === 'oracle_label_change',
      ),
      (record) => record.evidence_intervention_id,
    );
    return [...groups.values()].filter((values) =>
      new Set(values.map((record) => record.target.ambiguous)).size === 2 &&
      new Set(values.map((record) => record.evidence_variant)).size >= 2,
    ).length;
  });
  if (labelChanging.train < options.minimumLabelChangingDevelopmentGroups) {
    errors.push('V7 training has too few oracle label-changing evidence groups.');
  }
  if (labelChanging.calibration < 1) {
    errors.push('V7 calibration has no oracle label-changing evidence group.');
  }
  if (labelChanging.untouched_mechanic < 1) {
    errors.push('V7 untouched mechanic has no oracle label-changing evidence group.');
  }

  for (const split of ['train', 'calibration'] as const) {
    for (const mechanic of options.developmentMechanics) {
      for (const evidence of options.requiredEvidenceVariants) {
        if (!canonicalRecords.some((record) =>
          record.split === split && record.mechanic === mechanic && record.evidence_variant === evidence,
        )) {
          errors.push(`V7 ${split} is missing ${mechanic}/${evidence}.`);
        }
      }
    }
  }

  const conditional = {
    train: conditionalGapReport(records.filter((record) => record.split === 'train')),
    calibration: conditionalGapReport(records.filter((record) => record.split === 'calibration')),
  };
  const maximumConditionalGap = Math.max(conditional.train.maximum, conditional.calibration.maximum);
  if (maximumConditionalGap > options.maximumConditionalLabelGap) {
    errors.push(
      `V7 conditional label gap ${maximumConditionalGap.toFixed(6)} exceeds ` +
      `${options.maximumConditionalLabelGap.toFixed(6)}.`,
    );
  }

  const recordCounts = mapSplits((split) => records.filter((record) => record.split === split).length);
  const baseCounts = mapSplits((split) => new Set(
    records.filter((record) => record.split === split).map((record) => record.surface_pair_id),
  ).size);
  const contextCounts = mapSplits((split) => new Set(
    records.filter((record) => record.split === split).map((record) => record.base_context_group),
  ).size);
  const ambiguityRates = mapSplits((split) => {
    const selected = canonicalRecords.filter((record) => record.split === split);
    return selected.length === 0
      ? 0
      : selected.filter((record) => record.target.ambiguous).length / selected.length;
  });
  for (const split of splits) {
    if (recordCounts[split] === 0) errors.push(`V7 ${split} is empty.`);
    if (!(ambiguityRates[split] >= 0.35 && ambiguityRates[split] <= 0.65)) {
      errors.push(`V7 ${split} ambiguity rate is outside 0.35–0.65.`);
    }
  }

  return {
    errors,
    records: recordCounts,
    base_records: baseCounts,
    context_groups: contextCounts,
    ambiguous_rates: ambiguityRates,
    counts_by_surface: mapSplits((split) => countBy(
      records.filter((record) => record.split === split),
      (record) => record.surface_variant,
      options.requiredSurfaces,
    )),
    counts_by_mechanic: mapSplits((split) => countBy(
      records.filter((record) => record.split === split),
      (record) => record.mechanic,
      ['relockshort', 'powertrip', 'tonedrift'] as const,
    )),
    counts_by_evidence: mapSplits((split) => countBy(
      records.filter((record) => record.split === split),
      (record) => record.evidence_variant,
      [...options.requiredEvidenceVariants, 'mixed'] as const,
    )),
    label_changing_evidence_groups: labelChanging,
    conditional_label_gaps: conditional,
    maximum_conditional_label_gap: maximumConditionalGap,
    context_cross_split_overlaps: contextOverlaps,
    prompt_cross_split_overlaps: promptOverlaps,
    evidence_group_cross_split_overlaps: evidenceOverlaps,
    source_scenario_development_holdout_overlaps: scenarioOverlaps,
    duplicate_ids: duplicateIds,
    duplicate_prompts: duplicatePrompts,
    incomplete_surface_groups: incompleteSurfaceGroups,
    forbidden_training_fields: forbiddenFields,
    v3_test_records_read: 0,
    prior_holdout_records_read: 0,
  };
}

export function conditionalGapReport(records: V7IdentifiabilityRecord[]): ConditionalGapReport {
  const full = groupBy(records, (record) => canonicalJson([
    record.mechanic,
    record.evidence_variant,
    record.action_template,
    record.surface_variant,
  ]));
  const fullRates = [...full.values()].map(ambiguityRate);
  const dimensions = [
    'evidence_variant',
    'mechanic',
    'action_template',
    'surface_variant',
  ] as const;
  const byDimension = Object.fromEntries(dimensions.map((dimension) => {
    const others = dimensions.filter((value) => value !== dimension);
    const conditioned = groupBy(records, (record) => canonicalJson(
      others.map((name) => record[name]),
    ));
    const gaps = [...conditioned.values()].map((values) => {
      const rates = [...groupBy(values, (record) => String(record[dimension])).values()].map(ambiguityRate);
      return rates.length < 2 ? 0 : Math.max(...rates) - Math.min(...rates);
    });
    return [dimension, gaps.length === 0 ? 0 : Math.max(...gaps)];
  })) as ConditionalGapReport['by_dimension'];
  const fullDistance = fullRates.length === 0
    ? 1
    : Math.max(...fullRates.map((rate) => Math.abs(rate - 0.5)));
  const maximum = Math.max(fullDistance, ...Object.values(byDimension));
  return {
    maximum,
    full_cell_maximum_distance_from_half: fullDistance,
    by_dimension: byDimension,
    full_cells: full.size,
    minimum_full_cell_support: full.size === 0
      ? 0
      : Math.min(...[...full.values()].map((values) => values.length)),
  };
}

function ambiguityRate(values: V7IdentifiabilityRecord[]): number {
  return values.filter((record) => record.target.ambiguous).length / values.length;
}

function countForbidden(value: unknown): number {
  if (Array.isArray(value)) return value.reduce((sum, entry) => sum + countForbidden(entry), 0);
  if (!value || typeof value !== 'object') return 0;
  return Object.entries(value).reduce(
    (sum, [key, entry]) => sum + (forbiddenKeys.has(key) ? 1 : 0) + countForbidden(entry),
    0,
  );
}

function duplicateCount(values: string[]): number {
  return values.length - new Set(values).size;
}

function crossSplitCount<T>(records: T[], key: (record: T) => string): number {
  const observed = new Map<string, V7Split>();
  const overlaps = new Set<string>();
  for (const record of records as Array<T & { split: V7Split }>) {
    const value = key(record);
    const prior = observed.get(value);
    if (prior !== undefined && prior !== record.split) overlaps.add(value);
    observed.set(value, record.split);
  }
  return overlaps.size;
}

function countBy<T, K extends string>(values: T[], key: (value: T) => K, keys: readonly K[]): Record<K, number> {
  return Object.fromEntries(keys.map((value) => [
    value,
    values.filter((entry) => key(entry) === value).length,
  ])) as Record<K, number>;
}

function mapSplits<T>(value: (split: V7Split) => T): Record<V7Split, T> {
  return Object.fromEntries(splits.map((split) => [split, value(split)])) as Record<V7Split, T>;
}

function groupBy<T>(values: T[], key: (value: T) => string): Map<string, T[]> {
  const groups = new Map<string, T[]>();
  for (const value of values) groups.set(key(value), [...(groups.get(key(value)) ?? []), value]);
  return groups;
}
