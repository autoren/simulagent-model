import type { AgentEpistemicRecordV3, DatasetSplit } from './contracts';
import { canonicalJson, sha256 } from './serialization';

export interface V3ValidationReport {
  errors: string[];
  counts: Record<DatasetSplit, number>;
  group_counts: Record<DatasetSplit, number>;
  ambiguous_rates: Record<DatasetSplit, number>;
  ambiguity_rate_gap: number;
  mechanic_share_max_gap: number;
  mechanic_shares: Record<string, Record<DatasetSplit, number>>;
  prompt_cross_split_overlaps: number;
  context_cross_split_overlaps: number;
}

const splits: DatasetSplit[] = ['train', 'valid', 'test'];

export function validateV3(
  records: AgentEpistemicRecordV3[],
  thresholds?: { maximumAmbiguityRateGap?: number; maximumMechanicShareGap?: number },
): V3ValidationReport {
  const errors: string[] = [];
  const ids = new Set<string>();
  const promptSplits = new Map<string, DatasetSplit>();
  const contextSplits = new Map<string, DatasetSplit>();
  let promptCrossSplitOverlaps = 0;
  let contextCrossSplitOverlaps = 0;
  for (const record of records) {
    if (record.schema_version !== 3) errors.push(`${record.id} has non-v3 schema.`);
    if (ids.has(record.id)) errors.push(`Duplicate v3 id ${record.id}.`);
    ids.add(record.id);
    if (record.mechanic_labels.length === 0) errors.push(`${record.id} has no mechanic labels.`);
    if (record.target.possible_outcomes.length < 1) errors.push(`${record.id} has no outcomes.`);
    if (record.target.identifiable !== (record.target.possible_outcomes.length === 1)) {
      errors.push(`${record.id} identifiability disagrees with its outcome count.`);
    }
    if (record.empirical_support.length !== record.target.possible_outcomes.length) {
      errors.push(`${record.id} empirical support length mismatch.`);
    }
    record.target.possible_outcomes.forEach((target, index) => {
      if (record.empirical_support[index]?.target_sha256 !== sha256(canonicalJson(target))) {
        errors.push(`${record.id} empirical support hash mismatch at ${index}.`);
      }
    });
    const prompt = canonicalJson(record.agent_input);
    if (registerSplit(promptSplits, prompt, record.split)) promptCrossSplitOverlaps += 1;
    if (registerSplit(contextSplits, record.split_group, record.split)) contextCrossSplitOverlaps += 1;
  }

  const counts = countBySplit(records);
  const groupCounts = Object.fromEntries(
    splits.map((split) => [
      split,
      new Set(records.filter((record) => record.split === split).map((record) => record.split_group))
        .size,
    ]),
  ) as Record<DatasetSplit, number>;
  for (const split of splits) {
    if (counts[split] === 0) errors.push(`V3 ${split} split is empty.`);
  }
  const ambiguousRates = Object.fromEntries(
    splits.map((split) => {
      const splitRecords = records.filter((record) => record.split === split);
      return [
        split,
        splitRecords.filter((record) => !record.target.identifiable).length / splitRecords.length,
      ];
    }),
  ) as Record<DatasetSplit, number>;
  const ambiguityRateGap = range(Object.values(ambiguousRates));
  const labels = [...new Set(records.flatMap((record) => record.mechanic_labels))].sort();
  const mechanicShares = Object.fromEntries(
    labels.map((label) => [
      label,
      Object.fromEntries(
        splits.map((split) => {
          const splitRecords = records.filter((record) => record.split === split);
          return [
            split,
            splitRecords.filter((record) => record.mechanic_labels.includes(label)).length /
              splitRecords.length,
          ];
        }),
      ),
    ]),
  ) as Record<string, Record<DatasetSplit, number>>;
  const mechanicShareMaxGap = Math.max(
    0,
    ...Object.values(mechanicShares).map((shares) => range(Object.values(shares))),
  );
  if (
    thresholds?.maximumAmbiguityRateGap !== undefined &&
    ambiguityRateGap > thresholds.maximumAmbiguityRateGap
  ) {
    errors.push(
      `Ambiguity rate gap ${ambiguityRateGap.toFixed(6)} exceeds ` +
        `${thresholds.maximumAmbiguityRateGap.toFixed(6)}.`,
    );
  }
  if (
    thresholds?.maximumMechanicShareGap !== undefined &&
    mechanicShareMaxGap > thresholds.maximumMechanicShareGap
  ) {
    errors.push(
      `Mechanic share gap ${mechanicShareMaxGap.toFixed(6)} exceeds ` +
        `${thresholds.maximumMechanicShareGap.toFixed(6)}.`,
    );
  }
  if (promptCrossSplitOverlaps > 0) errors.push('V3 prompts cross splits.');
  if (contextCrossSplitOverlaps > 0) errors.push('V3 context groups cross splits.');
  return {
    errors,
    counts,
    group_counts: groupCounts,
    ambiguous_rates: ambiguousRates,
    ambiguity_rate_gap: ambiguityRateGap,
    mechanic_share_max_gap: mechanicShareMaxGap,
    mechanic_shares: mechanicShares,
    prompt_cross_split_overlaps: promptCrossSplitOverlaps,
    context_cross_split_overlaps: contextCrossSplitOverlaps,
  };
}

function registerSplit(
  values: Map<string, DatasetSplit>,
  key: string,
  split: DatasetSplit,
): boolean {
  const previous = values.get(key);
  values.set(key, split);
  return previous !== undefined && previous !== split;
}

function countBySplit(records: AgentEpistemicRecordV3[]): Record<DatasetSplit, number> {
  return Object.fromEntries(
    splits.map((split) => [split, records.filter((record) => record.split === split).length]),
  ) as Record<DatasetSplit, number>;
}

function range(values: number[]): number {
  return Math.max(...values) - Math.min(...values);
}
