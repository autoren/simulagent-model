import type { AgentIdentifiabilityRecordV4, V4DevelopmentSplit } from './contracts';

export interface V4ValidationReport {
  errors: string[];
  counts: Record<V4DevelopmentSplit, number>;
  group_counts: Record<V4DevelopmentSplit, number>;
  ambiguous_rates: Record<V4DevelopmentSplit, number>;
  ambiguity_rate_gap: number;
  mechanic_share_max_gap: number;
  context_cross_split_overlaps: number;
  prompt_cross_split_overlaps: number;
  source_test_records_read: number;
}

const splits: V4DevelopmentSplit[] = ['train', 'calibration', 'validation'];

export function validateV4(
  records: AgentIdentifiabilityRecordV4[],
  thresholds?: { maximumAmbiguityRateGap?: number; maximumMechanicShareGap?: number },
): V4ValidationReport {
  const errors: string[] = [];
  const ids = new Set<string>();
  const contexts = new Map<string, V4DevelopmentSplit>();
  const prompts = new Map<string, V4DevelopmentSplit>();
  let contextCrossSplitOverlaps = 0;
  let promptCrossSplitOverlaps = 0;
  for (const record of records) {
    if (ids.has(record.id)) errors.push(`Duplicate V4 id ${record.id}.`);
    ids.add(record.id);
    if (record.schema_version !== 4) errors.push(`${record.id} has non-V4 schema.`);
    if (record.split === 'validation' && record.source_split !== 'valid') {
      errors.push(`${record.id} validation record did not come from V3 validation.`);
    }
    if (record.split !== 'validation' && record.source_split !== 'train') {
      errors.push(`${record.id} development record did not come from V3 training.`);
    }
    if (register(contexts, record.split_group, record.split)) contextCrossSplitOverlaps += 1;
    if (register(prompts, JSON.stringify(record.agent_input), record.split)) {
      promptCrossSplitOverlaps += 1;
    }
  }
  const counts = bySplit(records, (record) => record.id);
  const groupCounts = bySplit(records, (record) => record.split_group, true);
  const ambiguousRates = Object.fromEntries(
    splits.map((split) => {
      const selected = records.filter((record) => record.split === split);
      return [split, selected.filter((record) => !record.target.identifiable).length / selected.length];
    }),
  ) as Record<V4DevelopmentSplit, number>;
  const ambiguityRateGap = range(Object.values(ambiguousRates));
  const mechanicLabels = [...new Set(records.flatMap((record) => record.mechanic_labels))];
  const mechanicShareMaxGap = Math.max(
    0,
    ...mechanicLabels.map((label) =>
      range(
        splits.map((split) => {
          const selected = records.filter((record) => record.split === split);
          return selected.filter((record) => record.mechanic_labels.includes(label)).length / selected.length;
        }),
      ),
    ),
  );
  for (const split of splits) {
    if (counts[split] === 0) errors.push(`V4 ${split} split is empty.`);
  }
  if (contextCrossSplitOverlaps > 0) errors.push('V4 contexts cross experiment splits.');
  if (promptCrossSplitOverlaps > 0) errors.push('V4 prompts cross experiment splits.');
  if (
    thresholds?.maximumAmbiguityRateGap !== undefined &&
    ambiguityRateGap > thresholds.maximumAmbiguityRateGap
  ) {
    errors.push(`V4 ambiguity-rate gap ${ambiguityRateGap.toFixed(6)} exceeds threshold.`);
  }
  if (
    thresholds?.maximumMechanicShareGap !== undefined &&
    mechanicShareMaxGap > thresholds.maximumMechanicShareGap
  ) {
    errors.push(`V4 mechanic-share gap ${mechanicShareMaxGap.toFixed(6)} exceeds threshold.`);
  }
  return {
    errors,
    counts,
    group_counts: groupCounts,
    ambiguous_rates: ambiguousRates,
    ambiguity_rate_gap: ambiguityRateGap,
    mechanic_share_max_gap: mechanicShareMaxGap,
    context_cross_split_overlaps: contextCrossSplitOverlaps,
    prompt_cross_split_overlaps: promptCrossSplitOverlaps,
    source_test_records_read: 0,
  };
}

function bySplit(
  records: AgentIdentifiabilityRecordV4[],
  key: (record: AgentIdentifiabilityRecordV4) => string,
  unique = false,
): Record<V4DevelopmentSplit, number> {
  return Object.fromEntries(
    splits.map((split) => {
      const values = records.filter((record) => record.split === split).map(key);
      return [split, unique ? new Set(values).size : values.length];
    }),
  ) as Record<V4DevelopmentSplit, number>;
}

function register(
  values: Map<string, V4DevelopmentSplit>,
  key: string,
  split: V4DevelopmentSplit,
): boolean {
  const previous = values.get(key);
  values.set(key, split);
  return previous !== undefined && previous !== split;
}

function range(values: number[]): number {
  return Math.max(...values) - Math.min(...values);
}
