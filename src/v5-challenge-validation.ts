import type {
  AgentIdentifiabilityRecordV4,
  V5ChallengeMechanic,
  V5ChallengeRecord,
  V5SurfaceVariant,
} from './contracts';
import { canonicalJson } from './serialization';

export interface V5ChallengeValidationReport {
  errors: string[];
  records: number;
  base_records: number;
  context_groups: number;
  evidence_pair_groups: number;
  ambiguous_rate: number;
  counts_by_surface: Record<V5SurfaceVariant, number>;
  counts_by_mechanic: Record<V5ChallengeMechanic, number>;
  prompt_development_overlaps: number;
  source_scenario_development_overlaps: number;
  duplicate_ids: number;
  duplicate_prompts: number;
  source_test_records_read: number;
}

export function validateV5Challenge(options: {
  records: V5ChallengeRecord[];
  developmentRecords: AgentIdentifiabilityRecordV4[];
  requiredSurfaces: V5SurfaceVariant[];
}): V5ChallengeValidationReport {
  const { records, developmentRecords, requiredSurfaces } = options;
  const errors: string[] = [];
  const duplicateIds = duplicateCount(records.map((record) => record.id));
  const duplicatePrompts = duplicateCount(records.map((record) => canonicalJson(record.agent_input)));
  const developmentPrompts = new Set(
    developmentRecords.map((record) => canonicalJson(record.agent_input)),
  );
  const developmentScenarios = new Set(
    developmentRecords.flatMap((record) => record.source_scenario_ids),
  );
  const promptDevelopmentOverlaps = records.filter((record) =>
    developmentPrompts.has(canonicalJson(record.agent_input)),
  ).length;
  const sourceScenarioDevelopmentOverlaps = new Set(
    records.flatMap((record) =>
      record.source_scenario_ids.filter((scenario) => developmentScenarios.has(scenario)),
    ),
  ).size;
  const bySurfacePair = groupBy(records, (record) => record.surface_pair_id);
  for (const [pairId, values] of bySurfacePair) {
    const surfaces = [...new Set(values.map((record) => record.surface_variant))].sort();
    const expected = [...requiredSurfaces].sort();
    if (canonicalJson(surfaces) !== canonicalJson(expected)) {
      errors.push(`${pairId} does not contain every required surface variant.`);
    }
    if (new Set(values.map((record) => record.target.identifiable)).size !== 1) {
      errors.push(`${pairId} changes its label across surface variants.`);
    }
    if (new Set(values.map((record) => record.base_context_group)).size !== 1) {
      errors.push(`${pairId} crosses base context groups.`);
    }
    if (
      values.some(
        (record) =>
          record.surface_variant !== 'canonical' &&
          canonicalJson(record.agent_input) ===
            canonicalJson(values.find((candidate) => candidate.surface_variant === 'canonical')?.agent_input),
      )
    ) {
      errors.push(`${pairId} contains an unchanged surface transformation.`);
    }
  }
  const evidenceRecords = records.filter(
    (record) => record.surface_variant === 'canonical' && record.evidence_pair_id !== null,
  );
  const byEvidencePair = groupBy(evidenceRecords, (record) => record.evidence_pair_id as string);
  for (const [pairId, values] of byEvidencePair) {
    if (new Set(values.map((record) => record.target.identifiable)).size < 2) {
      errors.push(`${pairId} does not change identifiability.`);
    }
    if (new Set(values.map((record) => record.evidence_variant)).size < 2) {
      errors.push(`${pairId} does not span evidence variants.`);
    }
  }
  if (records.length === 0) errors.push('V5 challenge is empty.');
  if (byEvidencePair.size === 0) errors.push('V5 challenge has no evidence-rung pairs.');
  if (duplicateIds > 0) errors.push('V5 challenge contains duplicate ids.');
  if (duplicatePrompts > 0) errors.push('V5 challenge contains duplicate prompts.');
  if (promptDevelopmentOverlaps > 0) errors.push('V5 challenge prompts overlap V4 development.');
  if (sourceScenarioDevelopmentOverlaps > 0) {
    errors.push('V5 challenge source scenarios overlap V4 development.');
  }
  const canonical = records.filter((record) => record.surface_variant === 'canonical');
  const ambiguousRate = canonical.filter((record) => !record.target.identifiable).length / canonical.length;
  if (ambiguousRate < 0.35 || ambiguousRate > 0.65) {
    errors.push(`V5 canonical ambiguity rate ${ambiguousRate.toFixed(6)} is outside 0.35–0.65.`);
  }
  const mechanics = new Set(canonical.map((record) => record.mechanic));
  for (const mechanic of ['relockshort', 'powertrip'] as V5ChallengeMechanic[]) {
    if (!mechanics.has(mechanic)) errors.push(`V5 challenge is missing ${mechanic}.`);
  }
  return {
    errors,
    records: records.length,
    base_records: bySurfacePair.size,
    context_groups: new Set(records.map((record) => record.base_context_group)).size,
    evidence_pair_groups: byEvidencePair.size,
    ambiguous_rate: ambiguousRate,
    counts_by_surface: countBy(records, (record) => record.surface_variant, requiredSurfaces),
    counts_by_mechanic: countBy(
      canonical,
      (record) => record.mechanic,
      ['relockshort', 'powertrip'],
    ),
    prompt_development_overlaps: promptDevelopmentOverlaps,
    source_scenario_development_overlaps: sourceScenarioDevelopmentOverlaps,
    duplicate_ids: duplicateIds,
    duplicate_prompts: duplicatePrompts,
    source_test_records_read: 0,
  };
}

function duplicateCount(values: string[]): number {
  return values.length - new Set(values).size;
}

function countBy<T, K extends string>(
  values: T[],
  key: (value: T) => K,
  keys: readonly K[],
): Record<K, number> {
  return Object.fromEntries(
    keys.map((value) => [value, values.filter((entry) => key(entry) === value).length]),
  ) as Record<K, number>;
}

function groupBy<T>(values: T[], key: (value: T) => string): Map<string, T[]> {
  const groups = new Map<string, T[]>();
  for (const value of values) groups.set(key(value), [...(groups.get(key(value)) ?? []), value]);
  return groups;
}
