import type { CounterfactualRecord, DatasetSplit } from './contracts';

export interface ValidationSummary {
  recordCount: number;
  splitCounts: Record<DatasetSplit, number>;
  groupCounts: Record<DatasetSplit, number>;
}

export function validateRecords(
  records: CounterfactualRecord[],
  options: { requireAllSplits?: boolean } = {},
): ValidationSummary {
  const errors: string[] = [];
  const ids = new Set<string>();
  const groupSplits = new Map<string, DatasetSplit>();

  for (const record of records) {
    if (ids.has(record.id)) {
      errors.push(`Duplicate record id: ${record.id}`);
    }
    ids.add(record.id);
    const previousSplit = groupSplits.get(record.split_group);
    if (previousSplit && previousSplit !== record.split) {
      errors.push(
        `Split leakage: group ${record.split_group} appears in ${previousSplit} and ${record.split}`,
      );
    }
    groupSplits.set(record.split_group, record.split);
    validateTarget(record, errors);
  }

  if (options.requireAllSplits ?? true) {
    for (const split of ['train', 'valid', 'test'] as const) {
      if (!records.some((record) => record.split === split)) {
        errors.push(`Split ${split} is empty.`);
      }
    }
  }

  if (errors.length > 0) {
    throw new Error(errors.slice(0, 20).join('\n'));
  }

  return {
    recordCount: records.length,
    splitCounts: count(records, (record) => record.split),
    groupCounts: Object.fromEntries(
      (['train', 'valid', 'test'] as const).map((split) => [
        split,
        new Set(
          records.filter((record) => record.split === split).map((record) => record.split_group),
        ).size,
      ]),
    ) as Record<DatasetSplit, number>,
  };
}

function validateTarget(record: CounterfactualRecord, errors: string[]): void {
  const target = record.target;
  if (typeof target.success !== 'boolean' || typeof target.environment_changed !== 'boolean') {
    errors.push(`Invalid boolean target fields for ${record.id}`);
  }
  if (typeof target.next_location !== 'string' || typeof target.reachable_room_delta !== 'number') {
    errors.push(`Invalid scalar target fields for ${record.id}`);
  }
  const arrays = [
    target.inventory_added,
    target.inventory_removed,
    target.visible_actions_added,
    target.visible_actions_removed,
    target.blocked_actions_added,
    target.blocked_actions_removed,
    target.hidden_actions_revealed,
    target.hidden_actions_concealed,
  ];
  if (arrays.some((value) => !Array.isArray(value))) {
    errors.push(`Invalid array target fields for ${record.id}`);
  }
}

function count<T extends string>(
  records: CounterfactualRecord[],
  key: (record: CounterfactualRecord) => T,
): Record<T, number> {
  const result = {} as Record<T, number>;
  for (const record of records) {
    const value = key(record);
    result[value] = (result[value] ?? 0) + 1;
  }
  return result;
}
