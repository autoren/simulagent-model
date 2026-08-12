import type {
  AgentIdentifiabilityRecordV4,
  V5ChallengeRecord,
  V5EvidenceVariant,
  V5SurfaceVariant,
  V6DevelopmentMechanic,
  V6IdentifiabilityRecord,
  V6Mechanic,
  V6Split,
} from './contracts';
import { canonicalJson } from './serialization';

const splits: V6Split[] = ['train', 'calibration', 'mechanic_holdout'];
const forbiddenKeys = new Set([
  'possible_outcomes',
  'outcome_count',
  'empirical_support',
  'oracle_trace',
  'privileged_input',
  'privileged_world_state',
  'transition_rules',
  'target_sha256',
  'source_record_count',
]);

export interface V6ValidationReport {
  errors: string[];
  records: Record<V6Split, number>;
  base_records: Record<V6Split, number>;
  context_groups: Record<V6Split, number>;
  ambiguous_rates: Record<V6Split, number>;
  development_ambiguity_rate_gap: number;
  counts_by_surface: Record<V6Split, Record<V5SurfaceVariant, number>>;
  counts_by_mechanic: Record<V6Split, Partial<Record<V6Mechanic, number>>>;
  evidence_intervention_groups: Record<V6Split, number>;
  label_changing_evidence_groups: Record<V6Split, number>;
  intervention_groups_by_mechanic: Record<V6Split, Partial<Record<V6Mechanic, number>>>;
  context_cross_split_overlaps: number;
  prompt_cross_split_overlaps: number;
  evidence_group_cross_split_overlaps: number;
  prompt_prior_overlaps: number;
  source_scenario_prior_overlaps: number;
  source_scenario_development_holdout_overlaps: number;
  duplicate_ids: number;
  duplicate_prompts: number;
  forbidden_training_fields: number;
  source_test_records_read: number;
}

export function validateV6(options: {
  records: V6IdentifiabilityRecord[];
  priorDevelopmentRecords: AgentIdentifiabilityRecordV4[];
  priorChallengeRecords: V5ChallengeRecord[];
  requiredSurfaces: V5SurfaceVariant[];
  developmentMechanics: V6DevelopmentMechanic[];
  holdoutMechanic: 'mirrorreject';
  developmentSeeds: number[];
  holdoutSeeds: number[];
  minimumEvidenceInterventionGroups: number;
  maximumAmbiguityRateGap: number;
}): V6ValidationReport {
  const { records } = options;
  const errors: string[] = [];
  const duplicateIds = duplicateCount(records.map((record) => record.id));
  const duplicatePrompts = duplicateCount(records.map((record) => canonicalJson(record.agent_input)));
  const contexts = crossSplitCount(records, (record) => record.base_context_group);
  const prompts = crossSplitCount(records, (record) => canonicalJson(record.agent_input));
  const evidenceGroups = crossSplitCount(
    records.filter((record) => record.evidence_intervention_id !== null),
    (record) => record.evidence_intervention_id as string,
  );
  const priorPrompts = new Set([
    ...options.priorDevelopmentRecords.map((record) => canonicalJson(record.agent_input)),
    ...options.priorChallengeRecords.map((record) => canonicalJson(record.agent_input)),
  ]);
  const priorScenarios = new Set([
    ...options.priorDevelopmentRecords.flatMap((record) => record.source_scenario_ids),
    ...options.priorChallengeRecords.flatMap((record) => record.source_scenario_ids),
  ]);
  const promptPriorOverlaps = records.filter((record) => priorPrompts.has(canonicalJson(record.agent_input))).length;
  const sourceScenarioPriorOverlaps = new Set(
    records.flatMap((record) => record.source_scenario_ids.filter((id) => priorScenarios.has(id))),
  ).size;
  const developmentScenarios = new Set(
    records.filter((record) => record.split !== 'mechanic_holdout').flatMap((record) => record.source_scenario_ids),
  );
  const holdoutScenarios = new Set(
    records.filter((record) => record.split === 'mechanic_holdout').flatMap((record) => record.source_scenario_ids),
  );
  const developmentHoldoutScenarioOverlaps = [...holdoutScenarios].filter((id) => developmentScenarios.has(id)).length;
  const forbiddenTrainingFields = records.reduce((sum, record) => sum + countForbidden(record), 0);

  if (records.length === 0) errors.push('V6 corpus is empty.');
  if (duplicateIds > 0) errors.push('V6 contains duplicate ids.');
  if (duplicatePrompts > 0) errors.push('V6 contains duplicate prompts.');
  if (contexts > 0) errors.push('V6 base contexts cross splits.');
  if (prompts > 0) errors.push('V6 prompts cross splits.');
  if (evidenceGroups > 0) errors.push('V6 evidence interventions cross splits.');
  if (promptPriorOverlaps > 0) errors.push('V6 prompts overlap V4 development or the V5 challenge.');
  if (sourceScenarioPriorOverlaps > 0) errors.push('V6 scenarios overlap V4 development or the V5 challenge.');
  if (developmentHoldoutScenarioOverlaps > 0) errors.push('V6 source scenarios cross into the mechanic holdout.');
  if (forbiddenTrainingFields > 0) errors.push('V6 records expose privileged counts or transition targets.');

  const requiredTargetKeys = canonicalJson(['ambiguous', 'invariance']);
  for (const record of records) {
    if (record.schema_version !== 6) errors.push(`${record.id} has a non-V6 schema.`);
    if (canonicalJson(Object.keys(record.target).sort()) !== requiredTargetKeys) {
      errors.push(`${record.id} has an invalid binary/invariance target.`);
    }
    const permittedSeeds = new Set(
      record.split === 'mechanic_holdout' ? options.holdoutSeeds : options.developmentSeeds,
    );
    if (record.scenario_seeds.length === 0 || record.scenario_seeds.some((seed) => !permittedSeeds.has(seed))) {
      errors.push(`${record.id} contains a seed outside its preregistered partition.`);
    }
    if (record.split === 'mechanic_holdout' && record.mechanic !== options.holdoutMechanic) {
      errors.push(`${record.id} leaks a development mechanic into the holdout.`);
    }
    if (
      record.split !== 'mechanic_holdout' &&
      !options.developmentMechanics.includes(record.mechanic as V6DevelopmentMechanic)
    ) {
      errors.push(`${record.id} leaks the holdout mechanic into development.`);
    }
  }

  const surfaceGroups = groupBy(records, (record) => record.surface_pair_id);
  for (const [pairId, values] of surfaceGroups) {
    const observed = [...new Set(values.map((record) => record.surface_variant))].sort();
    const expected = [...options.requiredSurfaces].sort();
    if (canonicalJson(observed) !== canonicalJson(expected)) {
      errors.push(`${pairId} does not contain every surface.`);
    }
    if (new Set(values.map((record) => record.target.ambiguous)).size !== 1) {
      errors.push(`${pairId} changes its binary target across surfaces.`);
    }
    if (new Set(values.map((record) => record.split)).size !== 1) {
      errors.push(`${pairId} crosses splits.`);
    }
    if (new Set(values.map((record) => record.invariance_group_id)).size !== 1) {
      errors.push(`${pairId} has inconsistent invariance targets.`);
    }
    const canonical = values.find((record) => record.surface_variant === 'canonical');
    if (
      canonical &&
      values.some((record) => record.surface_variant !== 'canonical' && canonicalJson(record.agent_input) === canonicalJson(canonical.agent_input))
    ) {
      errors.push(`${pairId} contains an unchanged transformed surface.`);
    }
  }

  const interventionStats = interventionStatistics(records);
  for (const split of ['train', 'calibration'] as V6Split[]) {
    if (interventionStats.counts[split] < options.minimumEvidenceInterventionGroups) {
      errors.push(`V6 ${split} has too few evidence intervention groups.`);
    }
  }
  for (const split of ['train'] as V6Split[]) {
    for (const mechanic of options.developmentMechanics) {
      if ((interventionStats.byMechanic[split][mechanic] ?? 0) === 0) {
        errors.push(`V6 ${split} lacks ${mechanic} evidence interventions.`);
      }
    }
  }
  for (const [id, values] of interventionStats.groups) {
    const canonical = values.filter((record) => record.surface_variant === 'canonical');
    const variants = new Set<V5EvidenceVariant>(canonical.map((record) => record.evidence_variant));
    variants.delete('mixed');
    if (variants.size < 2) errors.push(`${id} does not span two explicit evidence variants.`);
    if (new Set(canonical.map((record) => record.mechanic)).size !== 1) {
      errors.push(`${id} crosses mechanics.`);
    }
  }

  const recordCounts = mapSplits((split) => records.filter((record) => record.split === split).length);
  const baseCounts = mapSplits((split) => new Set(records.filter((record) => record.split === split).map((record) => record.surface_pair_id)).size);
  const contextCounts = mapSplits((split) => new Set(records.filter((record) => record.split === split).map((record) => record.base_context_group)).size);
  const ambiguityRates = mapSplits((split) => {
    const canonical = records.filter((record) => record.split === split && record.surface_variant === 'canonical');
    return canonical.filter((record) => record.target.ambiguous).length / canonical.length;
  });
  const developmentGap = Math.abs(ambiguityRates.train - ambiguityRates.calibration);
  if (developmentGap > options.maximumAmbiguityRateGap) {
    errors.push(`V6 development ambiguity-rate gap ${developmentGap.toFixed(6)} exceeds threshold.`);
  }
  for (const split of splits) {
    if (recordCounts[split] === 0) errors.push(`V6 ${split} is empty.`);
    if (!(ambiguityRates[split] >= 0.35 && ambiguityRates[split] <= 0.65)) {
      errors.push(`V6 ${split} ambiguity rate is outside 0.35–0.65.`);
    }
  }
  for (const mechanic of options.developmentMechanics) {
    for (const split of ['train', 'calibration'] as V6Split[]) {
      if (!records.some((record) => record.split === split && record.mechanic === mechanic)) {
        errors.push(`V6 ${split} is missing ${mechanic}.`);
      }
    }
  }

  return {
    errors,
    records: recordCounts,
    base_records: baseCounts,
    context_groups: contextCounts,
    ambiguous_rates: ambiguityRates,
    development_ambiguity_rate_gap: developmentGap,
    counts_by_surface: mapSplits((split) => countBy(
      records.filter((record) => record.split === split),
      (record) => record.surface_variant,
      options.requiredSurfaces,
    )),
    counts_by_mechanic: mapSplits((split) => countBy(
      records.filter((record) => record.split === split),
      (record) => record.mechanic,
      ['relockshort', 'powertrip', 'mirrorreject'],
    )),
    evidence_intervention_groups: interventionStats.counts,
    label_changing_evidence_groups: interventionStats.labelChanging,
    intervention_groups_by_mechanic: interventionStats.byMechanic,
    context_cross_split_overlaps: contexts,
    prompt_cross_split_overlaps: prompts,
    evidence_group_cross_split_overlaps: evidenceGroups,
    prompt_prior_overlaps: promptPriorOverlaps,
    source_scenario_prior_overlaps: sourceScenarioPriorOverlaps,
    source_scenario_development_holdout_overlaps: developmentHoldoutScenarioOverlaps,
    duplicate_ids: duplicateIds,
    duplicate_prompts: duplicatePrompts,
    forbidden_training_fields: forbiddenTrainingFields,
    source_test_records_read: 0,
  };
}

function interventionStatistics(records: V6IdentifiabilityRecord[]) {
  const groups = groupBy(
    records.filter((record) => record.evidence_intervention_id !== null),
    (record) => record.evidence_intervention_id as string,
  );
  const counts = mapSplits((split) => new Set(
    records.filter((record) => record.split === split).flatMap((record) => record.evidence_intervention_id ?? []),
  ).size);
  const labelChanging = mapSplits((split) => [...groups.values()].filter((values) => {
    const canonical = values.filter((record) => record.split === split && record.surface_variant === 'canonical');
    return new Set(canonical.map((record) => record.target.ambiguous)).size > 1;
  }).length);
  const byMechanic = mapSplits((split) => countInterventionMechanics(records, split));
  return { groups, counts, labelChanging, byMechanic };
}

function countInterventionMechanics(
  records: V6IdentifiabilityRecord[],
  split: V6Split,
): Partial<Record<V6Mechanic, number>> {
  return Object.fromEntries(
    (['relockshort', 'powertrip', 'mirrorreject'] as V6Mechanic[]).map((mechanic) => [
      mechanic,
      new Set(records.filter((record) => record.split === split && record.mechanic === mechanic).flatMap((record) => record.evidence_intervention_id ?? [])).size,
    ]),
  );
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
  const observed = new Map<string, V6Split>();
  const overlaps = new Set<string>();
  for (const record of records as Array<T & { split: V6Split }>) {
    const value = key(record);
    const prior = observed.get(value);
    if (prior !== undefined && prior !== record.split) overlaps.add(value);
    observed.set(value, record.split);
  }
  return overlaps.size;
}

function countBy<T, K extends string>(values: T[], key: (value: T) => K, keys: readonly K[]): Record<K, number> {
  return Object.fromEntries(keys.map((value) => [value, values.filter((entry) => key(entry) === value).length])) as Record<K, number>;
}

function mapSplits<T>(value: (split: V6Split) => T): Record<V6Split, T> {
  return Object.fromEntries(splits.map((split) => [split, value(split)])) as Record<V6Split, T>;
}

function groupBy<T>(values: T[], key: (value: T) => string): Map<string, T[]> {
  const groups = new Map<string, T[]>();
  for (const value of values) groups.set(key(value), [...(groups.get(key(value)) ?? []), value]);
  return groups;
}
